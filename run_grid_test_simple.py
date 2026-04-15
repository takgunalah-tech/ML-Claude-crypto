"""
Simple parallel execution script for notebook 04 grid test.
Runs the actual notebook cells that do the grid search.
"""

import subprocess
import sys

print("=" * 80)
print("RANGE-BASED PARAMETER GRID TEST")
print("=" * 80)
print("\nStarting Jupyter notebook execution...")
print("This will run notebook 04 with range-based parameters.")
print(f"System has {__import__('os').cpu_count()} CPU cores available")
print(f"joblib Parallel will use up to 4 cores for parallel evaluation\n")

# Run the notebook with papermill (or just use nbconvert)
try:
    result = subprocess.run([
        sys.executable, '-m', 'jupyter', 'nbconvert',
        '--to', 'notebook',
        '--execute',
        '--ExecutePreprocessor.timeout=3600',
        '--output', 'notebooks/04_param_grid_tester_results.ipynb',
        'notebooks/04_param_grid_tester.ipynb'
    ], cwd='.')

    print("\n" + "=" * 80)
    print("Notebook execution complete!")
    print("Results saved to: notebooks/04_param_grid_tester_results.ipynb")
    print("=" * 80)

except Exception as e:
    print(f"Error: {e}")
    print("\nAlternative: Run the notebook manually in Jupyter")
