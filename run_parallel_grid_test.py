"""
Standalone script for parallel grid search with range-based parameters.
Run this if you want to execute the updated config outside the notebook.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import itertools
import sys

# Import from project utilities
from utils.data_utils import load_snapshot_csv
from utils.data_utils import split_data_by_days
from utils.model_utils import eval_config

# Configuration
TEST_COINS = ['XRP_USDT', 'SOL_USDT', 'LTC_USDT', 'ADA_USDT', 'AAVE_USDT',
              'LINK_USDT', 'AVAX_USDT', 'TRX_USDT', 'FIL_USDT', 'BCH_USDT', 'ZEC_USDT']

# Range-based parameter config
PARAM_CONFIGS = [
    (
        'Range search',
        ('0.005', '0.05'),            # TP range: (min, max)
        ('0.005', '0.05'),            # SL range: (min, max)
        ('0.002',),                   # TP/SL step: (step,)
        ('0.5', '3.0'),               # K1 & K2 range: (min, max) — SAME FOR BOTH
        ('0.5',),                     # K1/K2 step: (step,)
        ('1.05', '1.25'),             # MIN_PF_TEST1 range: (min, max)
        ('0.5', '0.92'),              # PF_STABILITY_RATIO range: (min, max)
        ('0.03',),                    # Gate step: (step,)
    ),
]

def iterate_param_config(config_tuple):
    """
    Convert range-based config tuple into expanded lists.

    Input:
        (name, tp_range, sl_range, tp_sl_step, k12_range, k12_step, min_pf_range, pf_ratio_range, gate_step)
        where K1/K2 use SAME range (not separate ranges)

    Output:
        (name, tp_list, sl_list, k1_list, k2_list, min_pf_list, pf_ratio_list)
    """
    name, tp_r, sl_r, tp_sl_step, k12_r, k12_step, min_pf_r, pf_ratio_r, gate_step = config_tuple

    # Extract step values
    tp_step = float(tp_sl_step[0])
    k12_s = float(k12_step[0])
    gate_s = float(gate_step[0])

    # Expand ranges to lists using np.arange
    tp_list = list(np.arange(float(tp_r[0]), float(tp_r[1]) + 1e-9, tp_step))
    sl_list = list(np.arange(float(sl_r[0]), float(sl_r[1]) + 1e-9, tp_step))
    k12_list = list(np.arange(float(k12_r[0]), float(k12_r[1]) + 1e-9, k12_s))
    k1_list = k2_list = k12_list  # Same range for both
    min_pf_list = list(np.arange(float(min_pf_r[0]), float(min_pf_r[1]) + 1e-9, gate_s))
    pf_ratio_list = list(np.arange(float(pf_ratio_r[0]), float(pf_ratio_r[1]) + 1e-9, gate_s))

    return name, tp_list, sl_list, k1_list, k2_list, min_pf_list, pf_ratio_list


def main():
    print("=" * 80)
    print("PARALLEL GRID SEARCH - RANGE-BASED PARAMETERS")
    print("=" * 80)

    for config_tuple in PARAM_CONFIGS:
        config_name, tp_grid, sl_grid, k1_grid, k2_grid, min_pf_list, pf_ratio_list = iterate_param_config(config_tuple)

        total_combos = len(tp_grid) * len(sl_grid) * len(k1_grid) * len(k2_grid) * len(min_pf_list) * len(pf_ratio_list)
        print(f'\nConfig: {config_name}')
        print(f'Grid size: {len(tp_grid)} × {len(sl_grid)} × {len(k1_grid)} × {len(k2_grid)} × {len(min_pf_list)} × {len(pf_ratio_list)} = {total_combos} combos')
        print("=" * 80)

        all_results = []

        for ticker in TEST_COINS:
            print(f"\nProcessing {ticker}...")

            # Load data
            try:
                df = load_snapshot_csv(ticker, asset_type='crypto')
                if df is None:
                    print(f"  Error: No data found for {ticker}")
                    continue
                df_train, df_test1, df_test2 = split_data_by_days(df, verbose=False)
            except Exception as e:
                print(f"  Error loading data: {e}")
                continue

            # Generate all parameter combinations
            all_combos = list(itertools.product(tp_grid, sl_grid, k1_grid, k2_grid, min_pf_list, pf_ratio_list))

            # PARALLEL EVALUATION using joblib
            print(f"  Evaluating {len(all_combos)} combos with parallel processing...")
            results_list = Parallel(n_jobs=-1, verbose=0)(
                delayed(eval_config)(
                    df_train, df_test1, df_test2,
                    ticker=ticker,
                    tp=tp, sl=sl, k1=k1, k2=k2,
                    min_pf_test1=min_pf,
                    pf_stability_ratio=pf_ratio,
                    verbose=False
                )
                for tp, sl, k1, k2, min_pf, pf_ratio in all_combos
            )

            # Find best combo for this ticker
            best_result = None
            best_combo = None
            best_pf = -1

            for (tp, sl, k1, k2, min_pf, pf_ratio), result in zip(all_combos, results_list):
                if result['pf_test1'] > best_pf and result['status']:  # Only consider valid combos
                    best_pf = result['pf_test1']
                    best_result = result
                    best_combo = (tp, sl, k1, k2, min_pf, pf_ratio)

            # Print result
            if best_result:
                tp, sl, k1, k2, min_pf, pf_ratio = best_combo
                print(f"  {ticker:15} | PASS PF={best_result['pf_test1']:.3f} Stab={best_result['stability']:.3f} (TP={tp:.4f} SL={sl:.4f} K1={k1:.1f} K2={k2:.1f})")
                all_results.append({
                    'ticker': ticker,
                    'tp': tp, 'sl': sl, 'k1': k1, 'k2': k2,
                    'min_pf': min_pf, 'pf_ratio': pf_ratio,
                    'pf_test1': best_result['pf_test1'],
                    'pf_test2': best_result['pf_test2'],
                    'stability': best_result['stability']
                })
            else:
                print(f"  {ticker:15} | FAIL - No valid combo found")

        # Summary table
        if all_results:
            results_df = pd.DataFrame(all_results)
            print("\n" + "=" * 80)
            print("RESULTS SUMMARY")
            print("=" * 80)
            print(results_df.to_string(index=False))

if __name__ == '__main__':
    main()
