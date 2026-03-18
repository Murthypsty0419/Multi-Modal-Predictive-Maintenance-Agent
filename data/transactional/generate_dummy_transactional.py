"""
Generate dummy transactional maintenance CSVs for testing.
- service_schedules.csv
- work_done_logs.csv
- maintenance_requests.csv

Usage: Run this script from data/transactional/
"""

import polars as pl
from datetime import datetime
import os

# Get directory of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Service Schedules ---
schedules = [
    {"pump_id": f"PUMP_{i:02d}", "task_name": "Oil Change", "interval_hours": 2000, "priority": 1}
    for i in range(1, 6)
] + [
    {"pump_id": f"PUMP_{i:02d}", "task_name": "Impeller Inspection", "interval_hours": 5000, "priority": 3}
    for i in range(1, 6)
]
df_schedules = pl.DataFrame(schedules)
df_schedules.write_csv(os.path.join(SCRIPT_DIR, "service_schedules.csv"))

# --- Work Done Logs ---
work_logs = [
    {"pump_id": "PUMP_01", "task_name": "Oil Change", "hours_at_service": 1800, "timestamp": datetime(2026, 2, 1, 10, 0)},
    {"pump_id": "PUMP_02", "task_name": "Oil Change", "hours_at_service": 500, "timestamp": datetime(2026, 1, 15, 9, 30)},
    {"pump_id": "PUMP_03", "task_name": "Oil Change", "hours_at_service": 1200, "timestamp": datetime(2025, 12, 20, 8, 0)},
    {"pump_id": "PUMP_03", "task_name": "Impeller Inspection", "hours_at_service": 4800, "timestamp": datetime(2025, 11, 10, 14, 0)},
    {"pump_id": "PUMP_04", "task_name": "Oil Change", "hours_at_service": 2100, "timestamp": datetime(2026, 2, 10, 11, 0)},
    {"pump_id": "PUMP_05", "task_name": "Oil Change", "hours_at_service": 800, "timestamp": datetime(2026, 1, 5, 7, 0)},
    {"pump_id": "PUMP_05", "task_name": "Impeller Inspection", "hours_at_service": 5100, "timestamp": datetime(2026, 1, 25, 16, 0)},
]
df_work_logs = pl.DataFrame(work_logs)
df_work_logs = df_work_logs.with_columns([
    pl.col("hours_at_service").cast(pl.Int64),
    pl.col("timestamp").cast(pl.Datetime("ns")),
])
df_work_logs.write_csv(os.path.join(SCRIPT_DIR, "work_done_logs.csv"))

# --- Maintenance Requests ---
requests = [
    {"pump_id": "PUMP_02", "description": "Excessive heat and vibration noted in the motor housing.", "priority": "CRITICAL", "status": "OPEN", "created_at": datetime(2026, 2, 15, 13, 0)},
    {"pump_id": "PUMP_04", "description": "Small leak detected near the seal.", "priority": "MEDIUM", "status": "OPEN", "created_at": datetime(2026, 2, 12, 15, 30)},
    {"pump_id": "PUMP_01", "description": "Routine check completed.", "priority": "LOW", "status": "CLOSED", "created_at": datetime(2026, 1, 20, 8, 0)},
]
df_requests = pl.DataFrame(requests)
df_requests = df_requests.with_columns([
    pl.col("created_at").cast(pl.Datetime("ns")),
])
df_requests.write_csv(os.path.join(SCRIPT_DIR, "maintenance_requests.csv"))

print(f"Dummy transactional CSVs generated in {SCRIPT_DIR}")
