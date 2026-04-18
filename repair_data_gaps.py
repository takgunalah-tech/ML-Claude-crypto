"""Audit and optionally repair internal raw-data gaps for configured crypto tickers.

Usage:
  python repair_data_gaps.py
  python repair_data_gaps.py --apply
  python repair_data_gaps.py --apply --ticker ZEC/USDT:USDT
"""

import argparse

import config
from utils.data_utils import _find_missing_ranges, load_base_csv, repair_base_gaps, save_base_csv


def main() -> None:
    parser = argparse.ArgumentParser(description='Audit or repair internal raw crypto data gaps.')
    parser.add_argument('--apply', action='store_true', help='Persist repaired base CSVs.')
    parser.add_argument('--ticker', action='append', help='Limit to one or more specific tickers.')
    args = parser.parse_args()

    tickers = args.ticker or config.CRYPTO_TICKERS
    changed = 0
    blocked = 0

    for ticker in tickers:
        df = load_base_csv(ticker, 'crypto')
        gaps = _find_missing_ranges(df, freq='1h', min_missing=1)
        if not gaps:
            print(f'OK|{ticker}|no_gaps')
            continue

        total_missing = sum(missing for _, _, missing in gaps)
        largest_gap = max(gaps, key=lambda row: row[2])
        print(
            f'GAP|{ticker}|count={len(gaps)}|total_missing={total_missing}'
            f'|largest={largest_gap[2]}|start={largest_gap[0]}|end={largest_gap[1]}'
        )

        repaired = repair_base_gaps(ticker, 'crypto', df)
        if repaired.equals(df):
            print(f'NO_CHANGE|{ticker}|backfill_unavailable_or_unneeded')
            continue

        if not args.apply:
            print(f'READY|{ticker}|rows_before={len(df)}|rows_after={len(repaired)}')
            continue

        try:
            save_base_csv(repaired, ticker, 'crypto')
            print(f'SAVED|{ticker}|rows_before={len(df)}|rows_after={len(repaired)}')
            changed += 1
        except PermissionError:
            print(f'LOCKED|{ticker}|close notebook/scheduler handles and rerun --apply')
            blocked += 1

    print(f'SUMMARY|changed={changed}|blocked={blocked}|mode={"apply" if args.apply else "dry_run"}')


if __name__ == '__main__':
    main()
