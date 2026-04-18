"""
Signal state manager — deduplication, aging, archiving.

State file: state/signals_state.json
Schema per coin:
{
  "SOL/USDT": {
    "direction":       "LONG",
    "entry":           142.5,
    "tp":              147.0,
    "sl":              139.2,
    "qty":             0.68,
    "p_win":           0.64,
    "tp_pct":          0.030,
    "sl_pct":          0.015,
    "first_signal_at": "2026-04-14T10:00:00Z",
    "repeat_count":    0,
    "archived":        false,
    "archive_reason":  null   # "TP_HIT" | "SL_HIT" | "EXPIRED" | "MAX_REPEATS"
  }
}
"""

import os
import json
import math
import logging
from datetime import datetime, timezone, timedelta

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.governance_utils import audit_event

logger = logging.getLogger(__name__)

_STATE_FILE = os.path.join(config.STATE_DIR, 'signals_state.json')


# ── I/O ───────────────────────────────────────────────────────────────────────

def _load_state() -> dict:
    os.makedirs(config.STATE_DIR, exist_ok=True)
    if not os.path.exists(_STATE_FILE):
        return {}
    with open(_STATE_FILE) as f:
        return json.load(f)


def _save_state(state: dict) -> None:
    os.makedirs(config.STATE_DIR, exist_ok=True)
    tmp = _STATE_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
    import shutil
    last_error = None
    for _ in range(5):
        try:
            shutil.move(tmp, _STATE_FILE)
            return
        except PermissionError as e:
            last_error = e
            import time
            time.sleep(0.5)
    raise last_error


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _parse_utc(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.strptime(ts, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _expiry_utc(hours: int | float = None) -> str:
    ttl_hours = config.SIGNAL_TTL_HOURS if hours is None else hours
    return (datetime.now(timezone.utc) + timedelta(hours=ttl_hours)).strftime('%Y-%m-%dT%H:%M:%SZ')


def _get_signal_expiry(signal: dict) -> datetime | None:
    """
    Backward-compatible expiry resolution.

    New signals store `expiry_at` explicitly.
    Legacy signals may only have `first_signal_at`, so infer expiry using the
    configured TTL to avoid leaving pre-migration signals open forever.
    """
    explicit_expiry = _parse_utc(signal.get('expiry_at'))
    if explicit_expiry is not None:
        return explicit_expiry

    first_signal_at = _parse_utc(signal.get('first_signal_at'))
    if first_signal_at is None:
        return None

    ttl_hours = float(signal.get('ttl_hours', config.SIGNAL_TTL_HOURS))
    return first_signal_at + timedelta(hours=ttl_hours)


def _rr(signal: dict, current_price: float) -> float:
    """Remaining risk-reward ratio from current_price to TP / SL."""
    try:
        tp = signal['tp']
        sl = signal['sl']
        if signal['direction'] == 'LONG':
            reward = tp - current_price
            risk   = current_price - sl
        else:
            reward = current_price - tp
            risk   = sl - current_price
        return reward / risk if risk > 0 else 0.0
    except Exception:
        return 0.0


def get_effective_p_win(signal: dict, now: datetime | None = None) -> float:
    """
    Apply optional time decay to signal confidence.
    Compatibility note: if older signals do not carry decay fields, fall back to
    the stored raw probability.
    """
    raw_p = float(signal.get('p_win', 0.0))
    first_signal_at = _parse_utc(signal.get('first_signal_at'))
    if first_signal_at is None:
        return raw_p

    decay_lambda = float(signal.get('decay_lambda', config.SIGNAL_DECAY_LAMBDA))
    current_dt = now or datetime.now(timezone.utc)
    age_hours = max((current_dt - first_signal_at).total_seconds() / 3600, 0.0)
    return round(raw_p * math.exp(-decay_lambda * age_hours), 4)


def expire_stale_signals(now: datetime | None = None) -> int:
    """Archive active signals whose TTL has elapsed."""
    current_dt = now or datetime.now(timezone.utc)
    state = _load_state()
    changed = False
    expired = 0

    for coin, sig in state.items():
        if sig.get('archived', False):
            continue
        expiry_at = _get_signal_expiry(sig)
        if expiry_at is None:
            continue
        if current_dt >= expiry_at:
            sig['archived'] = True
            sig['archive_reason'] = 'EXPIRED'
            sig['state'] = 'EXPIRED'
            state[coin] = sig
            changed = True
            expired += 1
            audit_event('signal_expired', {'coin': coin, 'direction': sig.get('direction')})

    if changed:
        _save_state(state)
    return expired


# ── Public API ────────────────────────────────────────────────────────────────

def is_new_signal(coin: str, direction: str) -> bool:
    """
    Returns True only if we should fire a new signal for this coin+direction.

    Logic:
      - No record at all         → True  (first signal ever)
      - Record is archived       → True  (previous signal closed; can open new)
      - Record active, different direction → True  (LONG→SHORT flip is allowed)
      - Record active, same direction      → False (duplicate; suppress)
    """
    state = _load_state()
    if coin not in state:
        return True
    sig = state[coin]
    # Default is False (assume active) — only True if explicitly archived
    if sig.get('archived', False):
        return True
    # Active signal exists — allow only if direction changed
    if sig.get('direction') != direction:
        return True
    return False


def register_signal(
    coin: str,
    direction: str,
    entry: float,
    tp: float,
    sl: float,
    qty: float,
    p_win: float,
    tp_pct: float,
    sl_pct: float,
    k1: float = 0.0,
    k2: float = 0.0,
) -> None:
    """Register a new signal (overwrites any previous archived entry for this coin).

    k1 and k2 are the ATR multipliers used when computing TP/SL prices.
    They are stored so repeat messages can recalculate ATR-adjusted levels and
    display deltas relative to the original signal.
    """
    state = _load_state()
    state[coin] = {
        'direction':       direction,
        'entry':           round(entry, 8),
        'tp':              round(tp,    8),
        'sl':              round(sl,    8),
        'qty':             round(qty,   8),
        'p_win':           round(p_win, 4),
        'tp_pct':          tp_pct,
        'sl_pct':          sl_pct,
        'k1':              k1,
        'k2':              k2,
        'first_signal_at': _now_utc(),
        'expiry_at':       _expiry_utc(),
        'ttl_hours':       config.SIGNAL_TTL_HOURS,
        'decay_lambda':    config.SIGNAL_DECAY_LAMBDA,
        'repeat_count':    0,
        'state':           'OPEN',
        'archived':        False,
        'archive_reason':  None,
    }
    _save_state(state)
    audit_event('signal_registered', {
        'coin': coin,
        'direction': direction,
        'entry': entry,
        'tp': tp,
        'sl': sl,
        'p_win': p_win,
    })
    logger.info(f'[{coin}] Signal registered: {direction} entry={entry:.4f}')


def check_signal_aging(coin: str, current_price: float) -> str:
    """
    Decide what to do with an existing active signal.
    Returns: 'repeat' | 'archive' | 'already_archived'
    Side effect: increments repeat_count or sets archived on 'archive'.
    """
    state = _load_state()
    if coin not in state:
        return 'already_archived'
    sig = state[coin]

    # Default False — only treat as archived if explicitly set
    if sig.get('archived', False):
        return 'already_archived'

    expiry_at = _get_signal_expiry(sig)
    if expiry_at is not None and datetime.now(timezone.utc) >= expiry_at:
        sig['archived'] = True
        sig['archive_reason'] = 'EXPIRED'
        sig['state'] = 'EXPIRED'
        state[coin] = sig
        _save_state(state)
        return 'archive'

    if sig['repeat_count'] >= config.MAX_SIGNAL_REPEATS:
        sig['archived']       = True
        sig['archive_reason'] = 'MAX_REPEATS'
        sig['state']          = 'ARCHIVED'
        state[coin] = sig
        _save_state(state)
        return 'archive'

    # Still valid for a repeat
    sig['repeat_count'] += 1
    state[coin] = sig
    _save_state(state)
    return 'repeat'


def archive_signal(coin: str, reason: str) -> None:
    """Manually archive a signal with a given reason."""
    state = _load_state()
    if coin in state:
        state[coin]['archived']       = True
        state[coin]['archive_reason'] = reason
        state[coin]['state']          = 'ARCHIVED'
        _save_state(state)


def check_open_signals_status(current_prices: dict[str, float]) -> dict[str, str]:
    """
    For each active signal, check if TP or SL has been hit.
    Returns dict: coin → 'TP_HIT' | 'SL_HIT' | 'OPEN'
    Side effect: archives signals that hit TP or SL.
    """
    state   = _load_state()
    result  = {}
    changed = False

    expire_stale_signals()
    state = _load_state()

    for coin, sig in state.items():
        if sig.get('archived', False):
            continue
        price = current_prices.get(coin)
        if price is None:
            result[coin] = 'OPEN'
            continue

        tp        = sig['tp']
        sl        = sig['sl']
        direction = sig['direction']

        hit = None
        if direction == 'LONG':
            if price >= tp:
                hit = 'TP_HIT'
            elif price <= sl:
                hit = 'SL_HIT'
        else:  # SHORT
            if price <= tp:
                hit = 'TP_HIT'
            elif price >= sl:
                hit = 'SL_HIT'

        if hit:
            sig['archived']       = True
            sig['archive_reason'] = hit
            sig['state']          = hit
            state[coin] = sig
            changed = True
            result[coin] = hit
            audit_event('signal_closed', {'coin': coin, 'reason': hit, 'direction': direction})
        else:
            result[coin] = 'OPEN'

    if changed:
        _save_state(state)
    return result


def get_active_signals() -> list[dict]:
    """Return all non-archived signal records, including coin key."""
    expire_stale_signals()
    state = _load_state()
    return [
        {**v, 'coin': k, 'effective_p_win': get_effective_p_win(v)}
        for k, v in state.items()
        if not v.get('archived', False)   # default False = assume active
    ]


def get_archived_signals(since: datetime | None = None) -> list[dict]:
    """Return archived signals, optionally filtered to those since `since`."""
    state = _load_state()
    out = []
    for k, v in state.items():
        if not v.get('archived', False):
            continue
        if since is not None:
            try:
                ts = datetime.strptime(
                    v['first_signal_at'], '%Y-%m-%dT%H:%M:%SZ'
                ).replace(tzinfo=timezone.utc)
                if ts < since:
                    continue
            except (KeyError, ValueError):
                continue  # skip entries with missing/corrupt timestamps
        out.append({**v, 'coin': k})
    return out


def recalc_open_signal_qty(signal: dict, current_price: float) -> dict | None:
    """
    For a still-open signal, recalculate qty from current_price to SL.
    Returns updated signal dict if new RR >= 1:1, else None.
    """
    rr = _rr(signal, current_price)
    if rr < 1.0:
        return None

    sl   = signal['sl']
    risk = abs(current_price - sl)
    if risk == 0:
        return None

    new_qty = config.MAX_LOSS_USDT / risk
    updated = {**signal, 'qty': round(new_qty, 8), 'rr': round(rr, 3)}
    return updated
