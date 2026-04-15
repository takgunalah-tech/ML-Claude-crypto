"""Detect which ticker has the 31-candle gap"""
import pandas as pd
import os
from config import DATA_WORKING

# Check all snapshot CSVs for gaps
snapshots = [f for f in os.listdir(os.path.join(DATA_WORKING, 'crypto')) if f.endswith('_snapshot.csv')]
snapshots += [f for f in os.listdir(os.path.join(DATA_WORKING, 'macro')) if f.endswith('_snapshot.csv')]

print("Checking for data gaps...\n")

for snap_file in snapshots:
    try:
        if 'crypto' in snap_file:
            path = os.path.join(DATA_WORKING, 'crypto', snap_file)
        else:
            path = os.path.join(DATA_WORKING, 'macro', snap_file)

        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # Check for gaps
        time_diffs = df['timestamp'].diff().dt.total_seconds() / 3600  # convert to hours
        gaps = time_diffs[time_diffs > 1.5]  # gap > 1.5 hours

        if len(gaps) > 0:
            ticker = snap_file.replace('_1h_snapshot.csv', '')
            print(f"{ticker:15s}: Found {len(gaps)} gap(s)")
            for idx, gap_hours in gaps.items():
                if gap_hours > 24:  # significant gap
                    timestamp = df.loc[idx, 'timestamp']
                    print(f"  -> {gap_hours:.0f} hours at {timestamp}")
    except Exception as e:
        print(f"{snap_file}: Error - {e}")

print("\nDone.")
