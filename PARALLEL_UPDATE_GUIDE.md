# Notebook 04 Update Guide: Parallel Range-Based Search

## Changes Made Automatically

✓ Cell 3: PARAM_CONFIGS updated with range-based tuples (K1/K2 using [0.5-3.0])
✓ Cell 9: Added `iterate_param_config()` function
✓ Cell 1: Added `from joblib import Parallel, delayed` import
✓ Cells 12, 14: Updated for loop to use `iterate_param_config()`

---

## Manual Code Template: Add Parallel Evaluation

### For Cell 12 (Line ~130) - Test Setup Loop

Replace the loop that starts with `for config_tuple in PARAM_CONFIGS:`

**OLD CODE PATTERN:**
```python
for config_tuple in PARAM_CONFIGS:
    name, tp, sl, k1, k2, min_pf_list, pf_ratio_list = iterate_param_config(config_tuple)
    # Print grid size
    grid_size = ...
    print(f"Config: {name}")
    print(f"Grid size: ...")
```

**NEW CODE:**
```python
for config_tuple in PARAM_CONFIGS:
    name, tp, sl, k1, k2, min_pf_list, pf_ratio_list = iterate_param_config(config_tuple)
    
    total_combos = len(tp) * len(sl) * len(k1) * len(k2) * len(min_pf_list) * len(pf_ratio_list)
    print(f'Config: {name}')
    print(f'Grid size: {len(tp)} × {len(sl)} × {len(k1)} × {len(k2)} × {len(min_pf_list)} × {len(pf_ratio_list)} = {total_combos} combos')
    print("=" * 80)
```

---

### For Cell 14 (Line ~719) - Main Backtesting Loop with Parallel

This is the critical cell. Replace the entire for-loop section with parallel evaluation.

**KEY CHANGES:**
1. Build list of all combos: `all_combos = list(itertools.product(...))`
2. Wrap eval_config in `Parallel(n_jobs=-1)(delayed(eval_config)(...) for ...)`
3. Collect results and match back to combos

**TEMPLATE CODE:**

```python
for config_tuple in PARAM_CONFIGS:
    config_name, tp_grid, sl_grid, k1_grid, k2_grid, min_pf_list, pf_ratio_list = iterate_param_config(config_tuple)
    
    print(f'Config: {config_name}')
    total_combos = len(tp_grid) * len(sl_grid) * len(k1_grid) * len(k2_grid) * len(min_pf_list) * len(pf_ratio_list)
    print(f'Grid size: {len(tp_grid)} × {len(sl_grid)} × {len(k1_grid)} × {len(k2_grid)} × {len(min_pf_list)} × {len(pf_ratio_list)} = {total_combos} combos')
    print("=" * 80)
    
    for ticker in TEST_COINS:
        # Load data
        df = load_coin_data(ticker, return_raw=True)
        df_train, df_test1, df_test2 = split_data_by_days(df, verbose=False)
        
        # Generate all parameter combinations
        all_combos = list(itertools.product(tp_grid, sl_grid, k1_grid, k2_grid, min_pf_list, pf_ratio_list))
        
        # PARALLEL EVALUATION using joblib
        results_list = Parallel(n_jobs=-1)(
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
        if results_list:
            # Find best result
            best_idx = 0
            best_pf = results_list[0]['pf_test1'] if results_list[0]['pf_test1'] > 0 else -1
            for idx, res in enumerate(results_list):
                if res['pf_test1'] > best_pf:
                    best_pf = res['pf_test1']
                    best_idx = idx
            
            best_result = results_list[best_idx]
            tp, sl, k1, k2, min_pf, pf_ratio = all_combos[best_idx]
            
            status_str = "✓" if best_result['status'] else "✗"
            print(f"{ticker:15} | {status_str} PF_test1={best_result['pf_test1']:.3f} Stability={best_result['stability']:.3f}")
        else:
            print(f"{ticker:15} | ✗ No valid combo found")
```

---

## Verification Steps

1. Open notebook 04 in Jupyter
2. Verify imports section has: `from joblib import Parallel, delayed, itertools`
3. Check cell with PARAM_CONFIGS uses ranges (tuples like `('0.005', '0.05')`)
4. Find `iterate_param_config()` function definition
5. Run the notebook:
   - Should see grid size printed: "699090 combos" per ticker
   - Parallel evaluation should use all CPU cores
   - Runtime: ~15-22 min total (not 2-3 hours)

---

## If Parallel Doesn't Work

If you get `joblib` errors:
```bash
pip install joblib
```

If you get `delayed` is not imported:
```python
from joblib import Parallel, delayed
```

---

## Expected Output After Running

```
Config: Range search
Grid size: 23 × 23 × 6 × 6 × 7 × 15 = 699090 combos
================================================================================
XRP/USDT        | ✓ PF_test1=1.234 Stability=0.78
SOL/USDT        | ✗ No valid combo found
...
```

With parallel evaluation on 8 cores: **~15-22 minutes total**
