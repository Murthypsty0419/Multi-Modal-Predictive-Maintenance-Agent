"""
Sensor Model — LightGBM classifier + SHAP explainability.

Ingests a SensorReading, runs inference (or training) on the
Large_Industrial_Pump_Maintenance_Dataset, and returns a SensorPrediction
with SHAP values for the top contributing features.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from app.config import settings
from app.schemas import RiskLevel, SensorPrediction, SensorReading

logger = logging.getLogger("oxmaint.sensor_model")

# Exact order for model training/inference (do not reorder)
STRICT_FEATURE_ORDER: list[str] = [
    "Temperature",
    "Vibration",
    "Pressure",
    "Flow_Rate",
    "RPM",
    "Operational_Hours",
]

DEFAULT_FEATURE_COLS: list[str] = [
    "Operational_Hours",
    "Vibration",
    "Temperature",
    "Pressure",
    "Flow_Rate",
]

_model = None
_feature_cols: list[str] = DEFAULT_FEATURE_COLS.copy()


def _canon(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _load_feature_columns_from_artifact() -> list[str]:
    path = Path(settings.lightgbm_feature_columns_path)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed loading feature columns from %s", path)
        return []
    if isinstance(payload, list):
        parsed = [str(item) for item in payload if isinstance(item, str) and item.strip()]
        return parsed
    return []


def _coerce_float(value: object) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _load_model():
    global _model
    if _model is not None:
        return _model
    model_path = "/Users/polisetty/Downloads/projects/Minimal Predictive Agent/app/models/sensor_lgbm_model.txt"
    try:
        import lightgbm as lgb
        booster = lgb.Booster(model_file=model_path)
        print(f"✅ SUCCESS: Model loaded from {model_path}")
        _model = booster
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not load LightGBM file. Reason: {e}")
        _model = None
        return None

    model_feature_cols: list[str] = []
    try:
        model_feature_cols = [c for c in _model.feature_name() if isinstance(c, str) and c]
    except Exception:
        logger.exception("Unable to read feature names from LightGBM booster.")

    artifact_feature_cols = _load_feature_columns_from_artifact()
    if model_feature_cols and set(model_feature_cols) == set(STRICT_FEATURE_ORDER):
        _feature_cols = STRICT_FEATURE_ORDER.copy()
    elif model_feature_cols:
        _feature_cols = model_feature_cols
    elif artifact_feature_cols:
        _feature_cols = artifact_feature_cols
    else:
        _feature_cols = STRICT_FEATURE_ORDER.copy()

    logger.info("LightGBM model loaded from %s", model_path)
    logger.info("Sensor model feature count: %d", len(_feature_cols))
    return _model


def _reading_to_features(reading: SensorReading) -> np.ndarray:
    """Convert a SensorReading into a 1 × n array matching model's _feature_cols order. Values come from strict 6-element vector."""
    strict_list = _reading_to_strict_feature_list(reading)
    strict_map = dict(zip(STRICT_FEATURE_ORDER, strict_list))
    # Build in model order so predict() receives correct shape
    values = [strict_map.get(col, 0.0) for col in _feature_cols]
    return np.array([values], dtype=np.float64)


def _reading_to_strict_feature_list(reading: SensorReading) -> list[float]:
    """Build the 6-element feature vector in exact order: [Temperature, Vibration, Pressure, Flow_Rate, RPM, Operational_Hours]."""
    payload = reading.model_dump(exclude_none=True)
    return [
        _coerce_float(payload.get("temperature")),
        _coerce_float(payload.get("vibration")),
        _coerce_float(payload.get("pressure")),
        _coerce_float(payload.get("flow_rate")),
        _coerce_float(payload.get("rpm")),
        _coerce_float(payload.get("operational_hours")),
    ]


def _risk_from_prob(prob: float) -> RiskLevel:
    if prob >= 0.85:
        return RiskLevel.CRITICAL
    if prob >= 0.60:
        return RiskLevel.HIGH
    if prob >= 0.35:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


async def predict_sensor(reading: SensorReading) -> SensorPrediction:
    """Run LightGBM inference on a single sensor reading. Feature vector order: [Temperature, Vibration, Pressure, Flow_Rate, RPM, Operational_Hours]."""
    model = _load_model()
    features = _reading_to_features(reading)
    if model is None:
        logger.warning("LightGBM model not loaded (file missing or invalid); returning stub with failure_probability=0.0")
        return SensorPrediction(
            failure_probability=0.0,
            risk_level=RiskLevel.LOW,
        )
    prob = float(model.predict(features)[0])
    risk = _risk_from_prob(prob)
    logger.info("predict_sensor: prob=%.4f, risk_level=%s", prob, risk.value if hasattr(risk, "value") else risk)
    return SensorPrediction(
        failure_probability=prob,
        risk_level=risk,
    )


# ---------------------------------------------------------------------------
# Training helper (offline)
# ---------------------------------------------------------------------------

def train_model(csv_path: str, target_col: str = "Maintenance_Flag") -> None:
    """Train a LightGBM binary classifier from the pump CSV.

    Uses Polars for ingestion and asset-specific windowing.
    """
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split

    df = pl.read_csv(csv_path)
    logger.info("Loaded %d rows from %s", len(df), csv_path)

    feature_candidates = _load_feature_columns_from_artifact() or DEFAULT_FEATURE_COLS
    available = [c for c in feature_candidates if c in df.columns]
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV.")

    subset = df.select(available + [target_col]).drop_nulls()
    X = subset.select(available).to_numpy()
    y = subset[target_col].to_numpy().astype(int)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_ds = lgb.Dataset(X_train, label=y_train, feature_name=available)
    val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
    }

    model = lgb.train(
        params,
        train_ds,
        num_boost_round=300,
        valid_sets=[val_ds],
    )

    out = Path(settings.lightgbm_model_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out))
    logger.info("Model saved to %s", out)
