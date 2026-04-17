"""
Telegram notification utilities.

Uses the Bot HTTP API directly (no python-telegram-bot dependency needed).
Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env or environment.
"""

import requests
import logging
from datetime import datetime, timezone, timedelta

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

_API_BASE = 'https://api.telegram.org/bot{token}/sendMessage'


# ── Core send ─────────────────────────────────────────────────────────────────

# def _send(text: str) -> bool:
#     """Send a plain-text message to the configured chat. Returns True on success."""
#     if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
#         logger.warning('Telegram not configured (missing BOT_TOKEN or CHAT_ID).')
#         return False
#     url = _API_BASE.format(token=config.TELEGRAM_BOT_TOKEN)
#     try:
#         resp = requests.post(url, json={
#             'chat_id':    config.TELEGRAM_CHAT_ID,
#             'text':       text,
#             'parse_mode': 'HTML',
#         }, timeout=10)
#         if not resp.ok:
#             # Truncate error body to prevent log bloat from large HTML error pages
#             err_preview = resp.text[:200] if resp.text else ''
#             logger.error(f'Telegram send failed: {resp.status_code} {err_preview}')
#             return False
#         return True
#     except Exception as e:
#         logger.error(f'Telegram request error: {e}')
#         return False

import time  # Add this to your imports at the top
from requests.exceptions import RequestException, Timeout

def _send(text: str) -> bool:
    """Send a plain-text message with retries and longer timeout."""
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        logger.warning('Telegram not configured (missing BOT_TOKEN or CHAT_ID).')
        return False
        
    url = _API_BASE.format(token=config.TELEGRAM_BOT_TOKEN)
    payload = {
        'chat_id':    config.TELEGRAM_CHAT_ID,
        'text':       text,
        'parse_mode': 'HTML',
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Increased timeout to 30 seconds
            resp = requests.post(url, json=payload, timeout=30)
            
            if resp.status_code == 429:  # Rate Limited
                retry_after = resp.json().get('parameters', {}).get('retry_after', 5)
                logger.warning(f"Rate limited. Sleeping {retry_after}s...")
                time.sleep(retry_after)
                continue

            if not resp.ok:
                err_preview = resp.text[:200] if resp.text else ''
                logger.error(f'Telegram send failed: {resp.status_code} {err_preview}')
                return False
                
            return True

        except Timeout:
            logger.warning(f"Telegram timeout on attempt {attempt + 1}. Retrying...")
            time.sleep(2)  # Short pause before retrying
        except RequestException as e:
            logger.error(f'Telegram request error: {e}')
            return False

    logger.error("Telegram send failed after max retries due to timeouts.")
    return False


# ── Signal message ────────────────────────────────────────────────────────────

def send_signal(
    coin: str,
    direction: str,              # 'LONG' or 'SHORT'
    entry: float,                # current candle close (fresh entry if entering now)
    tp: float,                   # TP recalculated from current price + ATR
    sl: float,                   # SL recalculated from current price + ATR
    qty: float,
    p_win: float,
    drivers: list[str],
    is_repeat: bool = False,
    repeat_context: dict | None = None,
) -> bool:
    """
    Send one signal message per coin per direction.

    For NEW signals:
        🟢 LONG Signal: SOL/USDT
        Entry:   142.5000
        TP:      147.0000  (+3.15%)
        SL:      139.2000  (-2.31%)
        Qty:     0.3571 SOL  |  RR: 1.36
        P(win):  0.64
        Drivers: BTC_Breakout_Prob, RSI_14, ATR_expansion

    For REPEAT signals (repeat_context provided):
        🔁 LONG Update #1: SOL/USDT
        ──────────────────────────
        Current price:  143.20   [orig 142.50, +0.49%]
        New TP:         147.80   [was 147.00, +0.54%]
        New SL:         139.80   [was 139.20, +0.43%]
        RR from now:    1.31
        Qty (new):      0.34 SOL
        P(win):         0.66  [was 0.64, +0.02]
        Drivers: BTC_Breakout_Prob, RSI_14...
        ──────────────────────────
        Original: LONG @ 142.50 on 14 Apr 10:00 UTC
        TP/SL not yet hit — still open. Manual decision required.

    `repeat_context` keys: orig_entry, orig_tp, orig_sl, orig_p_win,
                           first_signal_at, repeat_num
    """
    icon    = '🟢' if direction == 'LONG' else '🔴'
    base    = coin.split('/')[0] if '/' in coin else coin
    drivers_str = ', '.join(drivers[:5]) if drivers else 'N/A'

    def _pct(new_val: float, old_val: float) -> str:
        """Format percentage delta: +1.23% or -0.45%"""
        if old_val == 0:
            return ''
        delta = (new_val - old_val) / abs(old_val) * 100
        sign  = '+' if delta >= 0 else ''
        return f'{sign}{delta:.2f}%'

    def _rr(entry_price: float, tp_price: float, sl_price: float) -> float:
        risk   = abs(entry_price - sl_price)
        reward = abs(tp_price    - entry_price)
        return reward / risk if risk > 0 else 0.0

    if not is_repeat or repeat_context is None:
        # ── New signal ──────────────────────────────────────────────────────
        tp_pct_str = _pct(tp, entry)
        sl_pct_str = _pct(sl, entry)
        rr         = _rr(entry, tp, sl)
        msg = (
            f'{icon} <b>{direction} Signal: {coin}</b>\n'
            f'Entry:   {entry:.4f}\n'
            f'TP:      {tp:.4f}  ({tp_pct_str})\n'
            f'SL:      {sl:.4f}  ({sl_pct_str})\n'
            f'Qty:     {qty:.4f} {base}  |  RR: {rr:.2f}\n'
            f'P(win):  {p_win:.2f}\n'
            f'Drivers: {drivers_str}'
        )
    else:
        # ── Repeat signal — show current vs original diff ───────────────────
        ctx          = repeat_context
        orig_entry   = ctx.get('orig_entry', entry)
        orig_tp      = ctx.get('orig_tp', tp)
        orig_sl      = ctx.get('orig_sl', sl)
        orig_p_win   = ctx.get('orig_p_win', p_win)
        repeat_num   = ctx.get('repeat_num', 1)
        fired_at     = ctx.get('first_signal_at', '')

        # Format the original timestamp more readably
        try:
            from datetime import datetime, timezone
            orig_dt = datetime.strptime(fired_at, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
            fired_str = orig_dt.strftime('%d %b %H:%M UTC')
        except Exception:
            fired_str = fired_at

        rr_now     = _rr(entry, tp, sl)
        p_win_diff = p_win - orig_p_win
        p_win_sign = '+' if p_win_diff >= 0 else ''

        sep = '─' * 28
        msg = (
            f'🔁 <b>{direction} Update #{repeat_num}: {coin}</b>\n'
            f'{sep}\n'
            f'Current price:  {entry:.4f}   [orig {orig_entry:.4f}, {_pct(entry, orig_entry)}]\n'
            f'New TP:         {tp:.4f}   [was {orig_tp:.4f}, {_pct(tp, orig_tp)}]\n'
            f'New SL:         {sl:.4f}   [was {orig_sl:.4f}, {_pct(sl, orig_sl)}]\n'
            f'RR from now:    {rr_now:.2f}\n'
            f'Qty (new):      {qty:.4f} {base}\n'
            f'P(win):         {p_win:.2f}  [was {orig_p_win:.2f}, {p_win_sign}{p_win_diff:+.2f}]\n'
            f'Drivers: {drivers_str}\n'
            f'{sep}\n'
            f'Original: {direction} @ {orig_entry:.4f} on {fired_str}\n'
            f'TP/SL not hit — still open. Manual decision required.'
        )

    return _send(msg)


# ── Morning report ────────────────────────────────────────────────────────────

def send_morning_report(
    active_signals: list[dict],
    archived_signals: list[dict],
    current_prices: dict[str, float],
) -> bool:
    """
    Daily 7am summary:
    - Active open signals with current RR and optional recalc qty
    - Closed signals (TP / SL / expired) from the last 24h
    """
    from utils.signal_manager import recalc_open_signal_qty, _rr

    now      = datetime.now(timezone.utc)
    date_str = now.strftime('%Y-%m-%d %H:%M UTC')

    lines = [f'<b>📊 SENTINEL Daily Report</b>\n{date_str}\n']

    # ── Active signals ────────────────────────────────────────────────────────
    lines.append(f'<b>Active signals: {len(active_signals)}</b>')
    if active_signals:
        for sig in active_signals:
            coin = sig['coin']
            price = current_prices.get(coin)
            if price is None:
                logger.warning(f'[{coin}] Current price unavailable — using entry price for RR calc')
                price = sig['entry']

            icon   = '🟢' if sig['direction'] == 'LONG' else '🔴'
            rr     = _rr(sig, price)
            rr_str = f'{rr:.2f}' if rr > 0 else '—'

            line = (
                f'{icon} <b>{coin}</b> {sig["direction"]} '
                f'| Entry {sig["entry"]:.4f} → TP {sig["tp"]:.4f} / SL {sig["sl"]:.4f} '
                f'| RR {rr_str}'
            )

            # Recalc qty suggestion if still good RR
            recalc = recalc_open_signal_qty(sig, price)
            if recalc:
                line += f'\n   ↳ Re-entry qty: {recalc["qty"]:.4f} (RR {recalc["rr"]:.2f})'

            lines.append(line)
    else:
        lines.append('  No open signals.')

    # ── Closed signals (last 24h) ─────────────────────────────────────────────
    lines.append(f'\n<b>Closed (last 24h): {len(archived_signals)}</b>')
    tp_count = sum(1 for s in archived_signals if s.get('archive_reason') == 'TP_HIT')
    sl_count = sum(1 for s in archived_signals if s.get('archive_reason') == 'SL_HIT')
    other    = len(archived_signals) - tp_count - sl_count
    lines.append(f'  ✅ TP hit: {tp_count}  |  ❌ SL hit: {sl_count}  |  ⏹ Other: {other}')

    for sig in archived_signals:
        reason = sig.get('archive_reason', '—')
        icon   = '✅' if reason == 'TP_HIT' else ('❌' if reason == 'SL_HIT' else '⏹')
        lines.append(f'  {icon} {sig["coin"]} {sig["direction"]} @ {sig["entry"]:.4f}')

    msg = '\n'.join(lines)
    return _send(msg)


def send_error_alert(msg: str) -> bool:
    """Send a critical error/warning to Telegram."""
    return _send(f'⚠️ <b>SENTINEL ERROR</b>\n{msg}')
