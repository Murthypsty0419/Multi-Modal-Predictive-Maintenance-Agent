"""
Sensor Node — Primary Witness.

Runs LightGBM inference on sensor data regardless of historical CSV.
Uses a strictly ordered 6-element feature vector:
[Temperature, Vibration, Pressure, Flow_Rate, RPM, Operational_Hours].
Always writes sensor_risk_score (0.0–1.0) to state when sensor data is present.
"""

from __future__ import annotations

import logging
from typing import Any

from app.schemas import AgentState

logger = logging.getLogger("oxmaint.sensor_node")


async def sensor_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    Run LightGBM inference on sensor input. Operates independently of historical logs.
    Feature vector is built in order: [Temperature, Vibration, Pressure, Flow_Rate, RPM, Operational_Hours];
    first five from sensor_data, Operational_Hours from UI (current_total_hours) or latest history row.
    """
    from app.models.sensor_model import _load_model, _reading_to_strict_feature_list
    try:
        agent = AgentState(**state)
    except Exception:
        logger.warning("sensor_node: invalid state, skipping")
        return {}
    if agent.sensor_input is None:
        logger.debug("sensor_node: sensor_input is None, skipping")
        return {}
    feature_vector = _reading_to_strict_feature_list(agent.sensor_input)
    print(f"\n🤖 [SENSOR NODE] Input Vector: {feature_vector}")
    logger.info(
        "sensor_node final 6-element feature vector (before model.predict): [Temperature, Vibration, Pressure, Flow_Rate, RPM, Operational_Hours] = %s",
        feature_vector,
    )
    model = _load_model()
    if model is None:
        logger.warning("LightGBM model not loaded; returning sensor_risk_score=0.0")
        prev_query = state.get("anomaly_query", "")
        appended_query = prev_query + f" sensor_risk_score: 0.0;"
        return {"sensor_risk_score": 0.0, "anomaly_query": appended_query}
    prediction = float(model.predict([feature_vector])[0])
    print(f"🎯 [SENSOR NODE] Raw Prediction: {prediction}")
    logger.info("[sensor_node] done: sensor_risk_score=%.4f", prediction)
    prev_query = state.get("anomaly_query", "")
    appended_query = prev_query + f" sensor_risk_score: {prediction};"
    return {"sensor_risk_score": prediction, "anomaly_query": appended_query}
