import logging
import polars as pl
from pathlib import Path
from typing import Dict, Any

from app.utils.toon import encode

logger = logging.getLogger("oxmaint.feature_node")
logger.propagate = True

async def feature_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Adaptive Baseline Analysis for a pump's historical data.
    - Loads the pump's historical CSV (sorted by Operational_Hours ascending).
    - Calculates 95th percentile for Vibration, Temperature, Pressure.
    - Calculates rolling mean (last 100 rows) for each sensor.
    - Compares current sensor readings to adaptive thresholds.
    - Computes drift score for each sensor.
    - Encodes historical baseline metrics into TOON format.
    Updates AgentState with:
      - adaptive_thresholds
      - is_historical_outlier
      - drift_score
      - historical_toon_string (TOON-encoded baseline metrics)
    """
    logger.info("[feature_node] start: pump_id=%s", state.get("pump_id"))
    # --- Input sanitization ---
    import re
    def sanitize_str(val):
        if not isinstance(val, str): return val
        return re.sub(r'[^\w\s\-\.:]', '', val).strip()
    pump_id = sanitize_str(state.get("pump_id"))
    if pump_id:
        pump_id = str(pump_id).lower()
    sinp = state.get("sensor_input") or {}
    if hasattr(sinp, "model_dump"):
        sinp = sinp.model_dump()
    current_sensor = {str(k): sanitize_str(v) for k, v in (sinp if isinstance(sinp, dict) else {}).items()}
    # Map CSV column names to possible state keys (snake_case from SensorReading)
    _sensor_key_map = {"Vibration": "vibration", "Temperature": "temperature", "Pressure": "pressure"}
    def _current_val(sensor_name):
        v = current_sensor.get(sensor_name)
        if v is not None: return v
        return current_sensor.get(_sensor_key_map.get(sensor_name, sensor_name.lower()))


    import os
    from pathlib import Path

    state_hist_path = state.get("historical_logs_path") if isinstance(state, dict) else None
    if state_hist_path:
        hist_path = Path(state_hist_path)
        logger.info("[feature_node] ✅ using historical_logs_path from request state: %s", hist_path)
    else:
        data_root = Path(os.getenv("DATA_DIR", "/app/data"))
        logs_dir = data_root / "historical_logs"
        os.makedirs(logs_dir, exist_ok=True)
        logger.info("[feature_node] ⚠️  no state path provided, falling back to mounted dir: %s", logs_dir)

        canonical_name = f"{pump_id}_history.csv"
        canonical_path = logs_dir / canonical_name
        logger.info("[feature_node] Handshake audit: expecting canonical_path=%s", canonical_path)
        try:
            logger.info("[feature_node] Files in historical_logs: %s", os.listdir(logs_dir))
        except Exception as e:
            logger.warning("[feature_node] Could not list files in %s: %s", logs_dir, e)
        logger.info("[feature_node] Expecting file: %s", canonical_name)

        for fname in os.listdir(logs_dir):
            abs_fpath = logs_dir / fname
            logger.info("[feature_node] Handshake audit: found file %s", abs_fpath)
            if fname == canonical_name and abs_fpath == canonical_path:
                continue
            if fname.endswith("_history.csv") and pump_id in fname:
                src = logs_dir / fname
                try:
                    import shutil

                    shutil.move(str(src), str(canonical_path))
                    logger.info("[feature_node] Renamed/moved %s to %s", src, canonical_path)
                except Exception as e:
                    logger.warning("[feature_node] Could not move %s to %s: %s", src, canonical_path, e)
        hist_path = canonical_path
    # print(f"[feature_node] Final hist_path to use: {hist_path}")
    # print(f"[feature_node] File exists at hist_path: {os.path.exists(hist_path)}")

    # --- File validation ---
    sensors = ["Vibration", "Temperature", "Pressure"]
    _empty_history = {
        "adaptive_thresholds": {s: None for s in sensors},
        "rolling_means": {s: None for s in sensors},
        "drift_scores": {s: None for s in sensors},
        "is_historical_outlier": False,
        "historical_toon_string": None,
        "history_risk_score": None,  # Always None when history absent or not an outlier
    }
    if not hist_path.exists():
        logger.info("[feature_node] 🚫 historical CSV NOT FOUND at %s → history_risk_score=None", hist_path)
        return {**_empty_history, "triggered_sensors": []}
    logger.info("[feature_node] ✅ historical CSV FOUND at %s → processing…", hist_path)
    if hist_path.stat().st_size > 50 * 1024 * 1024:
        return {**_empty_history, "historical_toon_string": "File too large (>50MB)", "triggered_sensors": []}
    if not hist_path.suffix.lower() == ".csv":
        return {**_empty_history, "historical_toon_string": "Invalid file type (not CSV)", "triggered_sensors": []}

    triggered_sensors = []
    try:
        df = pl.read_csv(hist_path)
        required_cols = {"Operational_Hours", "Vibration", "Temperature", "Pressure"}
        if not required_cols.issubset(set(df.columns)):
            return {**_empty_history, "historical_toon_string": "CSV missing required columns", "triggered_sensors": []}
        df = df.sort("Operational_Hours")
        adaptive_thresholds = {}
        rolling_means = {}
        drift_scores = {}
        is_outlier = False
        baseline_metrics = []
        for sensor in sensors:
            vals = df[sensor].drop_nulls()
            if len(vals) < 10:
                adaptive_thresholds[sensor] = None
                rolling_means[sensor] = None
                drift_scores[sensor] = None
                continue
            # 95th percentile
            p95_val = float(vals.quantile(0.95))
            adaptive_thresholds[sensor] = p95_val
            # Rolling mean (last 100 rows)
            mean_val = float(vals[-100:].mean()) if len(vals) >= 100 else float(vals.mean())
            rolling_means[sensor] = mean_val
            # Drift: (Current Rolling Mean - Global Mean) / Global Std
            global_mean = float(vals.mean())
            global_std = float(vals.std()) if float(vals.std()) > 0 else 1.0
            drift_scores[sensor] = (rolling_means[sensor] - global_mean) / global_std
            # Outlier check (current_sensor may use snake_case keys)
            raw_val = _current_val(sensor)
            try:
                current_val = float(raw_val) if raw_val is not None else None
            except (TypeError, ValueError):
                current_val = None
            if current_val is not None and adaptive_thresholds[sensor] is not None:
                if current_val > adaptive_thresholds[sensor]:
                    is_outlier = True
                    triggered_sensors.append(sensor)
            # Map sensor names to units and add to baseline_metrics
            sensor_lower = sensor.lower()
            unit_map = {
                "vibration": "mm/s",
                "temperature": "C",
                "pressure": "bar"
            }
            baseline_metrics.append({
                "sensor": sensor_lower,
                "unit": unit_map.get(sensor_lower, ""),
                "p95": p95_val,
                "mean": mean_val
            })

        # --- TOON encoding for transactional service history ---
        transactional_toon_string = None
        if "work_done_logs" in state:
            work_logs = state["work_done_logs"]
            # Expecting a list of dicts: [{"date": ..., "task": ..., "hours": ...}, ...]
            if isinstance(work_logs, list) and work_logs:
                last_5 = work_logs[-5:]
                service_history = {"service_history": last_5}
                transactional_toon_string = encode(service_history)

        # Create historical_context dictionary
        historical_context = {
            "pump_id": pump_id,
            "baseline_metrics": baseline_metrics
        }

        # Encode to TOON format: baseline_metrics[N]{sensor, unit, p95, mean}
        historical_toon_string = encode(historical_context)

        # history_risk_score is 0.5 if outlier, None if normal
        history_risk_score = 0.5 if is_outlier else None

        # --- anomaly_query logic ---
        anomaly_query = {
            "is_outlier": is_outlier,
            "triggered_sensors": triggered_sensors,
            "pump_id": pump_id,
            "history_risk_score": history_risk_score
        }
        prev_query = state.get("anomaly_query", "")
        appended_query = prev_query + f" feature_node: {anomaly_query};"

        logger.info(
            "[feature_node] historical CSV loaded: history_risk_score=%s, is_historical_outlier=%s, triggered_sensors=%s",
            history_risk_score,
            is_outlier,
            triggered_sensors,
        )
        return {
            "adaptive_thresholds": adaptive_thresholds,
            "rolling_means": rolling_means,
            "drift_scores": drift_scores,
            "is_historical_outlier": is_outlier,
            "historical_toon_string": historical_toon_string,
            "transactional_toon_string": transactional_toon_string,
            "history_risk_score": history_risk_score,
            "triggered_sensors": triggered_sensors,
            "anomaly_query": appended_query
        }
    except Exception as e:
        logger.error(f"[feature_node] Exception during historical CSV processing: {e}")
        return {**_empty_history, "historical_toon_string": f"Error: {e}", "triggered_sensors": []}
    # (Cleanup moved to orchestrator. No file deletion here.)
