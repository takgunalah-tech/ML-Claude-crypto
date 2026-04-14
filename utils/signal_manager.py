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
import logging
from datetime import datetime, timezone, timedelta

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

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
    shutil.move(tmp, _STATE_FILE)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


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
) -> None:
    """Register a new signal (overwrites any previous archived entry for this coin)."""
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
        'first_signal_at': _now_utc(),
        'repeat_count':    0,
        'archived':        False,
        'archive_reason':  None,
    }
    _save_state(state)
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

    if sig['repeat_count'] >= config.MAX_SIGNAL_REPEATS:
        sig['archived']       = True
        sig['archive_reason'] = 'MAX_REPEATS'
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
            state[coin] = sig
            changed = True
            result[coin] = hit
        else:
            result[coin] = 'OPEN'

    if changed:
        _save_state(state)
    return result


def get_active_signals() -> list[dict]:
    """Return all non-archived signal records, including coin key."""
    state = _load_state()
    return [
        {**v, 'coin': k}
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
