
"""
Industrial Calculator Toolkit for deterministic calculations.
Exposed as PydanticAI tools for agent use in Fusion/XAI.
"""

from typing import Dict, Any, List
from pydantic import BaseModel, Field


# Dummy agent decorator for illustration (replace with actual agent.tool in your framework)
def tool(func):
    func.is_tool = True
    return func

# 4. Risk Threshold Label
class RiskThresholdInput(BaseModel):
    fused_probability: float = Field(..., description="Fused risk probability (0-1)")

@tool
def get_risk_threshold_label(input: RiskThresholdInput) -> Dict[str, Any]:
    prob = input.fused_probability
    if prob >= 0.85:
        return {"risk_label": "critical"}
    if prob >= 0.60:
        return {"risk_label": "high"}
    if prob >= 0.35:
        return {"risk_label": "medium"}
    return {"risk_label": "low"}

# 1. Service Gap Calculator
class ServiceGapInput(BaseModel):
    current_hours: int = Field(...)
    last_service_hours: int = Field(...)
    interval: int = Field(...)

@tool
def calculate_service_gap(input: ServiceGapInput) -> Dict[str, Any]:
    hours_remaining = max(0, input.interval - (input.current_hours - input.last_service_hours))
    is_overdue = (input.current_hours - input.last_service_hours) >= input.interval
    percentage_of_life_used = min(1.0, (input.current_hours - input.last_service_hours) / input.interval) if input.interval > 0 else 0.0
    return {
        "hours_remaining": hours_remaining,
        "is_overdue": is_overdue,
        "percentage_of_life_used": round(percentage_of_life_used, 3)
    }

# 2. Unit Converter
class UnitConvertInput(BaseModel):
    value: float = Field(...)
    from_unit: str = Field(...)
    to_unit: str = Field(...)

@tool
def unit_converter(input: UnitConvertInput) -> Dict[str, Any]:
    v, f, t = input.value, input.from_unit.lower(), input.to_unit.lower()
    # Temperature
    if (f, t) == ("c", "f") or (f, t) == ("celsius", "fahrenheit"):
        return {"result": v * 9/5 + 32}
    if (f, t) == ("f", "c") or (f, t) == ("fahrenheit", "celsius"):
        return {"result": (v - 32) * 5/9}
    # Pressure
    if (f, t) == ("bar", "psi"):
        return {"result": v * 14.5038}
    if (f, t) == ("psi", "bar"):
        return {"result": v / 14.5038}
    # Time
    if (f, t) == ("days", "hours"):
        return {"result": v * 24}
    if (f, t) == ("hours", "days"):
        return {"result": v / 24}
    return {"error": f"Unsupported conversion: {f} to {t}"}

# 3. Risk Scaler
class RiskScalerInput(BaseModel):
    raw_scores: List[float] = Field(..., description="[sensor, vision, history]")

@tool
def risk_scaler(input: RiskScalerInput) -> Dict[str, Any]:
    """
    Weighted fusion of [sensor, vision, history] risk scores.
    Default weights: sensor=0.5, vision=0.3, history=0.2.
    If a modality is missing (None or 0), redistribute weights proportionally.
    Returns: {fused_score: float, weights: [float, float, float]}
    """
    base_weights = [0.5, 0.3, 0.2]
    scores = input.raw_scores
    if len(scores) != 3:
        return {"error": "raw_scores must be [sensor, vision, history]"}
    present = [s is not None and s != 0 for s in scores]
    if not any(present):
        return {"fused_score": 0.0, "weights": [0.0, 0.0, 0.0]}
    # Zero out missing modalities, redistribute weights
    active_weights = [w if p else 0.0 for w, p in zip(base_weights, present)]
    total = sum(active_weights)
    if total == 0:
        return {"fused_score": 0.0, "weights": [0.0, 0.0, 0.0]}
    norm_weights = [w / total for w in active_weights]
    fused = sum(w * (s or 0.0) for w, s in zip(norm_weights, scores))
    return {"fused_score": round(fused, 4), "weights": norm_weights}