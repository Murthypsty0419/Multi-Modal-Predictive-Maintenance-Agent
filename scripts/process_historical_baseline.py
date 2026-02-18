import polars as pl
import json
from pathlib import Path
import os
import glob

# Output paths
THRESHOLD_PATH = Path('data/transactional/pump_thresholds.json')
HOURS_PATH = Path('data/transactional/pump_hours.json')
FLAGS_PATH = Path('data/transactional/pump_flags.json')

# Sensors to baseline
SENSOR_COLS = ['Vibration', 'Temperature', 'Pressure']

# Find latest uploaded file in data/historical logs/
HISTORICAL_LOGS_DIR = Path('data/historical logs')
csv_files = sorted(glob.glob(str(HISTORICAL_LOGS_DIR / '*.csv')), key=os.path.getmtime, reverse=True)
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {HISTORICAL_LOGS_DIR}. Please upload a historical log file.")
CSV_PATH = Path(csv_files[0])
print(f"Using latest uploaded file: {CSV_PATH}")
df = pl.read_csv(CSV_PATH)

# 1. Baseline Extraction: 95th percentile per pump per sensor
thresholds = {}
for pump_id, group in df.groupby('pump_id'):
    thresholds[pump_id] = {}
    for sensor in SENSOR_COLS:
        sensor_vals = group[sensor].drop_nulls()
        if len(sensor_vals) > 0:
            thresholds[pump_id][sensor] = float(sensor_vals.quantile(0.95))
        else:
            thresholds[pump_id][sensor] = None

# Save thresholds
THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(THRESHOLD_PATH, 'w') as f:
    json.dump(thresholds, f, indent=2)
print(f"Saved thresholds to {THRESHOLD_PATH}")

# 2. Hour & Flag Extraction
hours = {}
flags = {}
for pump_id, group in df.groupby('pump_id'):
    # Max operational hours
    hours[pump_id] = float(group['Operational_Hours'].max())
    # Maintenance flag: count threshold breaches
    flags[pump_id] = {}
    for sensor in SENSOR_COLS:
        threshold = thresholds[pump_id][sensor]
        if threshold is not None:
            breach_count = (group[sensor] > threshold).sum()
            flags[pump_id][sensor] = int(breach_count)
        else:
            flags[pump_id][sensor] = 0

with open(HOURS_PATH, 'w') as f:
    json.dump(hours, f, indent=2)
print(f"Saved operational hours to {HOURS_PATH}")

with open(FLAGS_PATH, 'w') as f:
    json.dump(flags, f, indent=2)
print(f"Saved maintenance flags to {FLAGS_PATH}")

# 3. Drift Logic: Rolling average & drift flag
DRIFT_PATH = Path('data/transactional/pump_drift.json')
drift_flags = {}
window = 100  # Rolling window size
for pump_id, group in df.groupby('pump_id'):
    drift_flags[pump_id] = {}
    for sensor in SENSOR_COLS:
        threshold = thresholds[pump_id][sensor]
        vals = group[sensor].drop_nulls()
        if threshold is not None and len(vals) >= window:
            roll_avg = vals.rolling_mean(window)
            drift_flag = (roll_avg > threshold).sum() > 0
            drift_flags[pump_id][sensor] = bool(drift_flag)
        else:
            drift_flags[pump_id][sensor] = False

with open(DRIFT_PATH, 'w') as f:
    json.dump(drift_flags, f, indent=2)
print(f"Saved drift flags to {DRIFT_PATH}")

print("Processing complete.")
