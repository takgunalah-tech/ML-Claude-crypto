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

def _send(text: str) -> bool:
    """Send a plain-text message to the configured chat. Returns True on success."""
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        logger.warning('Telegram not configured (missing BOT_TOKEN or CHAT_ID).')
        return False
    url = _API_BASE.format(token=config.TELEGRAM_BOT_TOKEN)
    try:
        resp = requests.post(url, json={
            'chat_id':    config.TELEGRAM_CHAT_ID,
            'text':       text,
            'parse_mode': 'HTML',
        }, timeout=10)
        if not resp.ok:
            # Truncate error body to prevent log bloat from large HTML error pages
            err_preview = resp.text[:200] if resp.text else ''
            logger.error(f'Telegram send failed: {resp.status_code} {err_preview}')
            return False
        return True
    except Exception as e:
        logger.error(f'Telegram request error: {e}')
        return False


# ── Signal message ────────────────────────────────────────────────────────────

def send_signal(
    coin: str,
    direction: str,        # 'LONG' or 'SHORT'
    entry: float,
    tp: float,
    sl: float,
    qty: float,
    p_win: float,
    drivers: list[str],
    is_repeat: bool = False,
) -> bool:
    """
    Send one signal message per coin per direction.
    Shows absolute prices for TP and SL (not percentages — unambiguous for both
    LONG and SHORT directions).

    Format:
        🟢 LONG Signal: SOL/USDT
        Entry:   142.5000
        TP:      147.0000
        SL:      139.2000
        Qty:     0.3571 SOL
        P(win):  0.64
        Drivers: BTC_Breakout_Prob, RSI_14, ATR_expansion
    """
    icon        = '🟢' if direction == 'LONG' else '🔴'
    base        = coin.split('/')[0] if '/' in coin else coin
    label       = f'{"🔁 Repeat " if is_repeat else ""}{icon} {direction} Signal'
    drivers_str = ', '.join(drivers[:5]) if drivers else 'N/A'

    msg = (
        f'{label}: <b>{coin}</b>\n'
        f'Entry:   {entry:.4f}\n'
        f'TP:      {tp:.4f}\n'
        f'SL:      {sl:.4f}\n'
        f'Qty:     {qty:.4f} {base}\n'
        f'P(win):  {p_win:.2f}\n'
        f'Drivers: {drivers_str}'
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
