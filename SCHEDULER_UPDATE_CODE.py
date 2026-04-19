# ── Replace the scheduler section in your 00 notebook with this code ──────────

import schedule
import time
from datetime import datetime, timezone, timedelta

# ── Job: hourly inference ─────────────────────────────────────────────────────
def run_hourly_pipeline():
    logger.info('=== Hourly pipeline START ===')
    try:
        # Sleep 3 seconds before updating
        time.sleep(3)

        dfs   = update_all_tickers()
        c_dfs = {k: v for k, v in dfs.items() if '/' in k}
        m_dfs = {k: v for k, v in dfs.items() if '/' not in k}
        if c_dfs.get('BTC/USDT') is None:
            logger.error('BTC/USDT missing — skipping inference.')
            return

        ms     = compute_macro_risk_state(m_dfs)
        btc_a  = compute_anchor_features(c_dfs['BTC/USDT'], 'BTC')
        eth_a  = compute_anchor_features(c_dfs['ETH/USDT'], 'ETH')

        current_prices = {t: float(c_dfs[t]['close'].iloc[-1])
                          for t in config.ALTCOIN_TICKERS if t in c_dfs}
        check_open_signals_status(current_prices)

        for ticker in list_valid_models():
            if ticker not in c_dfs:
                continue
            model, meta = load_model(ticker)
            if model is None:
                continue
            feat_cols = meta['feature_columns']
            lp        = meta['label_params']

            feat_all  = compute_altcoin_features(c_dfs[ticker], btc_a, eth_a, ms)
            feat_live = feat_all.dropna(subset=feat_cols).tail(1)
            if feat_live.empty:
                continue

            try:
                X_live = feat_live[feat_cols]
            except KeyError as e:
                logger.error(f'[{ticker}] Feature parity error: {e}')
                continue

            p_win   = float(model.predict_proba(X_live)[0, 1])
            close   = float(c_dfs[ticker]['close'].iloc[-1])
            atr     = float(feat_live['ATR_14'].iloc[0]) if 'ATR_14' in feat_live.columns else 0.0
            atr_n   = atr / close if close > 0 else 0.0
            drivers = get_shap_drivers(model, X_live, n=5)

            def _fire(direction, tp, sl):
                qty = config.MAX_LOSS_USDT / max(abs(close - sl), 1e-8)
                if is_new_signal(ticker, direction):
                    register_signal(ticker, direction, close, tp, sl, qty,
                                    p_win, lp['tp_pct'], lp['sl_pct'],
                                    lp['k1'], lp['k2'])
                    send_signal(ticker, direction, close, tp, sl, qty, p_win, drivers)
                    logger.info(f'[{ticker}] {direction} signal fired P={p_win:.2f}')
                else:
                    aging = check_signal_aging(ticker, close)
                    if aging == 'repeat':
                        active_sigs = [s for s in get_active_signals() if s['coin'] == ticker]
                        if active_sigs:
                            s = active_sigs[0]
                            repeat_ctx = {
                                'orig_entry':      s['entry'],
                                'orig_tp':         s['tp'],
                                'orig_sl':         s['sl'],
                                'orig_p_win':      s['p_win'],
                                'first_signal_at': s['first_signal_at'],
                                'repeat_num':      s['repeat_count'],
                            }
                            send_signal(ticker, direction, close, tp, sl, qty,
                                        p_win, drivers, is_repeat=True,
                                        repeat_context=repeat_ctx)
                            logger.info(f'[{ticker}] {direction} repeat #{s["repeat_count"]} P={p_win:.2f}')

            if p_win >= config.LONG_THRESHOLD:
                tp = close * (1 + lp['tp_pct'] + lp['k1'] * atr_n)
                sl = close * (1 - lp['sl_pct'] - lp['k2'] * atr_n)
                _fire('LONG', tp, sl)
            elif p_win <= config.SHORT_THRESHOLD:
                tp = close * (1 - lp['tp_pct'] - lp['k1'] * atr_n)
                sl = close * (1 + lp['sl_pct'] + lp['k2'] * atr_n)
                _fire('SHORT', tp, sl)

    except Exception as e:
        logger.error(f'Hourly pipeline error: {e}')
        send_error_alert(f'Hourly pipeline error: {e}')
    logger.info('=== Hourly pipeline END ===')


# ── Job: retrain ──────────────────────────────────────────────────────────────
def run_retrain():
    logger.info('=== Retraining START ===')
    try:
        dfs      = update_all_tickers()
        c_dfs    = {k: v for k, v in dfs.items() if '/' in k}
        m_dfs    = {k: v for k, v in dfs.items() if '/' not in k}
        feat_dfs = build_all_features(c_dfs, m_dfs)
        for t, feat in feat_dfs.items():
            _save_feature_parquet(t, feat)

        now_r  = pd.Timestamp.utcnow().tz_localize(None)
        te_raw = now_r - pd.Timedelta(days=config.TRAIN_END_DAYS)
        te     = te_raw - pd.Timedelta(hours=config.LABEL_HORIZON)
        t1s_r  = now_r - pd.Timedelta(days=config.TEST1[0])
        t1e_r  = now_r - pd.Timedelta(days=config.TEST1[1])
        t2s_r  = now_r - pd.Timedelta(days=config.TEST2[0])
        t2e_r  = now_r - pd.Timedelta(days=config.TEST2[1])

        for ticker in config.ALTCOIN_TICKERS:
            if ticker not in feat_dfs:
                continue
            feat = feat_dfs[ticker]
            fc   = get_feature_cols(feat)
            ts   = pd.to_datetime(feat['timestamp'])
            tr   = feat[ts <= te]
            t1   = feat[(ts > t1s_r) & (ts <= t1e_r)]
            t2   = feat[(ts > t2s_r) & (ts <= t2e_r)]
            if len(tr) < 200:
                continue

            p = find_optimal_label_params(tr, fc, verbose=False)
            tp_pct, sl_pct, k1, k2 = p['tp_pct'], p['sl_pct'], p['k1'], p['k2']

            def al(df):
                df = df.copy()
                df['label'] = generate_labels(df, tp_pct, sl_pct, k1, k2).values
                return df.dropna(subset=['label'])

            tr_l = al(tr)
            t1_l = al(t1)
            t2_l = al(t2)
            if len(tr_l) < 30:
                continue

            m = train_xgboost(tr_l[fc], tr_l['label'].astype(int))

            m1 = evaluate_model(
                m, t1_l[fc], t1_l['label'].astype(int),
                tp_pct=tp_pct, sl_pct=sl_pct, k1=k1, k2=k2,
                atr_norm=_atr_norm(t1_l),
            )
            m2 = evaluate_model(
                m, t2_l[fc], t2_l['label'].astype(int),
                tp_pct=tp_pct, sl_pct=sl_pct, k1=k1, k2=k2,
                atr_norm=_atr_norm(t2_l),
            )

            if check_validity(m1, m2):
                save_model(m, ticker, {
                    'ticker':          ticker,
                    'feature_columns': fc,
                    'label_params':    {'tp_pct': tp_pct, 'sl_pct': sl_pct,
                                        'k1': k1, 'k2': k2},
                    'test1':           m1,
                    'test2':           m2,
                    'trained_at':      datetime.now(timezone.utc).isoformat(),
                    'runner_ups':      p.get('runner_ups', []),
                })
                logger.info(f'[{ticker}] Retrained OK — PF_t1={m1["PF"]:.3f}  PF_t2={m2["PF"]:.3f}')
            else:
                logger.info(f'[{ticker}] Retrain invalid (PF_t1={m1["PF"]:.3f}) — old model kept.')

    except Exception as e:
        logger.error(f'Retrain error: {e}')
        send_error_alert(f'Retrain error: {e}')
    logger.info('=== Retraining END ===')


# ── Job: morning report ───────────────────────────────────────────────────────
def send_morning_report_job():
    logger.info('=== Morning Report Job START ===')
    try:
        dfs = update_all_tickers()
        prices = {t: float(dfs[t]['close'].iloc[-1])
                  for t in config.ALTCOIN_TICKERS if t in dfs and not dfs[t].empty}
        since = datetime.now(timezone.utc) - timedelta(hours=24)
        success = send_morning_report(get_active_signals(), get_archived_signals(since=since), prices)
        if success:
            logger.info('Morning report sent successfully.')
        else:
            logger.warning('Morning report failed to send (check Telegram logs).')
    except Exception as e:
        logger.error(f'Morning report error: {e}')
    logger.info('=== Morning Report Job END ===')


# ── Schedule Setup ────────────────────────────────────────────────────────────
# Hourly at :00 (with 3-sec sleep before update)
schedule.every().hour.at(":00").do(run_hourly_pipeline)

# Retrain every Saturday at 06:00 UTC (US market close)
schedule.every().saturday.at("06:00").do(run_retrain)

# Morning report daily at 08:00 UTC
schedule.every().day.at("08:00").do(send_morning_report_job)

print('Scheduler initialized:')
print('  - Hourly pipeline: every hour at :00')
print('  - Retrain: every Saturday at 06:00 UTC')
print('  - Morning report: every day at 08:00 UTC')

# Run one cycle immediately
print('\nRunning initial inference cycle...')
run_hourly_pipeline()

# Main loop
try:
    print('Scheduler running. Press Ctrl+C to stop.\n')
    while True:
        schedule.run_pending()
        time.sleep(1)
except KeyboardInterrupt:
    print('\nScheduler stopped.')
