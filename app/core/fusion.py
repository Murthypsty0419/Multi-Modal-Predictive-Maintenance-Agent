"""
Late Fusion + XAI narrative generation.

Combines per-modality predictions using configurable weights,
then calls the Phi-4 SLM to produce a Chain-of-Thought explanation.
"""

from __future__ import annotations

import logging
from typing import Any

from app.schemas import (
    AgentState,
    FinalReport,
    FusionWeights,
    RiskLevel,
    SensorPrediction,
    RAGResult,
    VisionResult,
)

# Import deterministic tools for agent/SLM use
from app.core import tools as industrial_tools
from app.utils.toon import encode, wrap_prompt


logger = logging.getLogger("fusion")
DEFAULT_WEIGHTS = FusionWeights()


def _risk_from_prob(prob: float) -> RiskLevel:
    if prob >= 0.85:
        return RiskLevel.CRITICAL
    if prob >= 0.60:
        return RiskLevel.HIGH
    if prob >= 0.35:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


async def fuse_and_explain(state: AgentState) -> dict[str, Any]:
    """Weighted late fusion across available modalities + XAI narrative.

    Returns a partial state dict to merge back into the AgentState.
    """

    # --- Gather context ---
    weights = DEFAULT_WEIGHTS
    sensor_score = float(getattr(state, "sensor_risk_score", 0.0))
    rag: RAGResult | None = (
        RAGResult(**state.rag_result)
        if isinstance(state.rag_result, dict)
        else state.rag_result
    )
    vision: VisionResult | None = (
        VisionResult(**state.vision_result)
        if isinstance(state.vision_result, dict)
        else state.vision_result
    )
    history = getattr(state, "history_result", None)
    history_risk_score = getattr(state, "history_risk_score", None)
    service_age_score = float(getattr(state, "service_age_risk_score", 0.0))
    transactional_score = float(getattr(state, "transactional_risk_score", 0.0))

    # --- Compute weighted fusion (fallback: do not drag down by missing modalities) ---
    logger.info(
        "[fusion] inputs: sensor_risk_score=%s, vision=%s, history_risk_score=%s",
        getattr(state, "sensor_risk_score", None),
        "present" if vision else "None",
        history_risk_score,
    )
    vision_score = getattr(vision, "visual_risk_score", 0.0) if vision else 0.0
    history_score_val = history_risk_score if history_risk_score is not None else None
    history_score = (history_score_val or 0.0)  # for narrative display only

    # --- Fusion weights for all modalities ---
    # You can adjust these weights as needed; they must sum to 1.0
    fusion_weights = {
        "sensor": 0.8,
        "vision": 0.2,
        "history": 0.1,
        "service_age": 0.2,
        "transactional": 0.2,
    }
    # Normalize weights if needed
    total_weight = sum(fusion_weights.values())
    if total_weight != 1.0:
        fusion_weights = {k: v / total_weight for k, v in fusion_weights.items()}

    # Compute fused probability as weighted sum
    fused_prob = (
        fusion_weights["sensor"] * sensor_score +
        fusion_weights["vision"] * vision_score +
        fusion_weights["history"] * history_score +
        fusion_weights["service_age"] * service_age_score +
        fusion_weights["transactional"] * transactional_score
    )
    weights_used = [fusion_weights["sensor"], fusion_weights["vision"], fusion_weights["history"], fusion_weights["service_age"], fusion_weights["transactional"]]
    logger.info(
        "[fusion] mode=full (all modalities) -> fused_prob=%.4f, weights=%s",
        fused_prob, weights_used
    )

    # --- Get risk label ---
    risk_label = industrial_tools.get_risk_threshold_label(
        industrial_tools.RiskThresholdInput(fused_probability=fused_prob)
    )["risk_label"]
    logger.info(
        "[fusion] result: fused_prob=%.4f, risk_label=%s, weights=%s",
        fused_prob,
        risk_label,
        weights_used,
    )

    # # --- Service health (if history present) ---
    # # Removed unused calculate_service_health logic; already implemented elsewhere.

    # # --- TOON encoding for Phi-4 SLM ---
    # # Example: "SENSORS: 0.82 risk (Vibration spike) | VISION: Leak detected (0.7) | HISTORY: 200hrs overdue | RAG: Manual limit 5.0mm/s."
    # sensor_str = f"SENSORS: {sensor_score:.2f} risk"
    # vision_str = "VISION: "
    # if vision:
    #     if not vision.is_casting_present:
    #         vision_str += "No casting detected. "
    #     else:
    #         findings = []
    #         if vision.has_leaks:
    #             findings.append(f"Leak ({vision.confidence_leaks:.2f})")
    #         if vision.has_cracks:
    #             findings.append(f"Crack ({vision.confidence_cracks:.2f})")
    #         if vision.has_corrosion:
    #             findings.append(f"Corrosion ({vision.confidence_corrosion:.2f})")
    #         vision_str += ", ".join(findings) if findings else "No major defects."
    # history_str = f"HISTORY: {int(history_score * 100)} risk impact"
    # if history and history.overdue_tasks:
    #     history_str += f", Overdue: {', '.join(history.overdue_tasks)}"
    
    # # Use TOON-encoded historical baseline if available
    # historical_toon = state.historical_toon_string
    # if historical_toon:
    #     history_str += f" | Baseline TOON: {historical_toon}"

    # # Inject transactional TOON string as Official Service History
    # transactional_toon = state.transactional_toon_string
    # if transactional_toon:
    #     history_str += f" | Official Service History: {transactional_toon}"
    
    # rag_str = "RAG: "
    # if rag and rag.oem_limits_cited:
    #     rag_str += f"Manual limits: {', '.join(rag.oem_limits_cited)}"
    # else:
    #     rag_str += "No OEM limits cited."
    # system_message = f"{sensor_str} | {vision_str} | {history_str} | {rag_str}"


    # # --- XAI Chain-of-Thought with explicit weights ---
    # cot = []
    # weight_strs = [
    #     f"Sensor ({weights_used[0]*100:.0f}%)",
    #     f"Vision ({weights_used[1]*100:.0f}%)",
    #     f"History ({weights_used[2]*100:.0f}%)"
    # ]
    # cot.append(f"Fusion weights: {', '.join(weight_strs)}.")
    # cot.append(f"Sensor model indicates risk {sensor_score:.2f}.")
    # if vision:
    #     if not vision.is_casting_present:
    #         cot.append("Visual data ignored: no casting detected.")
    #     else:
    #         if vision.has_leaks:
    #             cot.append("Leak detected in casting image.")
    #         if vision.has_cracks:
    #             cot.append("Crack detected in casting image.")
    #         if vision.has_corrosion:
    #             cot.append("Corrosion detected in casting image.")
    # if history and history.overdue_tasks:
    #     cot.append(f"Maintenance overdue: {', '.join(history.overdue_tasks)}.")
    # # Include TOON-encoded historical baseline in Chain-of-Thought
    # if state.historical_toon_string:
    #     cot.append(f"Historical baseline (TOON): {state.historical_toon_string}")
    # # Include TOON-encoded service history in Chain-of-Thought
    # if state.transactional_toon_string:
    #     cot.append(f"Official Service History (TOON): {state.transactional_toon_string}")
    # if rag and rag.oem_limits_cited:
    #     cot.append(f"OEM limits cited: {', '.join(rag.oem_limits_cited)}.")
    # if not cot:
    #     cot.append("No significant risk factors detected.")
    
    explanation = (
        f"The risk score is a weighted fusion of all available modalities: "
        f"Sensor ({fusion_weights['sensor']*100:.0f}%, {sensor_score:.2f}), "
        f"Vision ({fusion_weights['vision']*100:.0f}%, {vision_score:.2f}), "
        f"History ({fusion_weights['history']*100:.0f}%, {history_score:.2f}), "
        f"Service Age ({fusion_weights['service_age']*100:.0f}%, {service_age_score:.2f}), "
        f"Transactional ({fusion_weights['transactional']*100:.0f}%, {transactional_score:.2f}). "
    )
    if vision and getattr(vision, "has_leaks", False):
        explanation += f" Visual seal leak detected (contributes {vision_score:.2f})."
    if history and getattr(history, "overdue_tasks", []):
        explanation += f" History risk impact {history_score:.2f}."
    if service_age_score > 0.5:
        explanation += f" Service age anomaly detected (score {service_age_score:.2f})."
    if transactional_score > 0.5:
        explanation += f" Outstanding transactional/maintenance risk (score {transactional_score:.2f})."
    if not (sensor_score or vision_score or history_score or service_age_score or transactional_score):
        explanation += " No significant risk factors detected."
    explanation = explanation.strip()

    # --- Action Items & Top Signals ---
    action_items: list[str] = []
    top_signals: list[str] = []

    # Add human-readable signals for triggered sensors (historical outliers)
    for sensor in getattr(state, "triggered_sensors", []):
        top_signals.append(f"Historical {sensor} Spike")

    if vision and getattr(vision, "has_leaks", False):
        action_items.append("Inspect for leaks immediately.")
        top_signals.append("Vision: Leak detected in casting image")
    if vision and getattr(vision, "has_cracks", False):
        action_items.append("Inspect for cracks immediately.")
        top_signals.append("Vision: Crack detected in casting image")
    if history and getattr(history, "overdue_tasks", []):
        action_items.append("Perform overdue maintenance tasks.")
        for task in history.overdue_tasks[:3]:
            top_signals.append(f"History overdue: {task}")
    if service_age_score > 0.5:
        action_items.append("Check overdue service schedule.")
        top_signals.append("Service age anomaly detected")
    if transactional_score > 0.5:
        action_items.append("Review outstanding maintenance requests.")
        top_signals.append("Outstanding transactional/maintenance risk")
    if not action_items:
        action_items.append("Continue routine monitoring.")

    # --- FinalReport output ---
    # Expose all modality contributions and weights
    modality_contributions = {
        "sensor": {"score": sensor_score, "weight": fusion_weights["sensor"]},
        "vision": {"score": vision_score, "weight": fusion_weights["vision"]},
        "history": {"score": history_score, "weight": fusion_weights["history"]},
        "service_age": {"score": service_age_score, "weight": fusion_weights["service_age"]},
        "transactional": {"score": transactional_score, "weight": fusion_weights["transactional"]},
    }
    # Add manual_context and anomaly_query to the report output
    manual_context = getattr(state, "manual_context", None)
    anomaly_query = getattr(state, "anomaly_query", None)
    report = FinalReport(
        fused_score=fused_prob,
        status_label=risk_label,
        explanation=explanation,
        top_signals=top_signals,
        action_items=action_items,
    )
    final_json = report.dict()
    final_json["modality_contributions"] = modality_contributions
    final_json["manual_context"] = manual_context
    final_json["anomaly_query"] = anomaly_query
    return final_json


# def _build_explanation(
#     fused_prob: float,
#     risk_level: RiskLevel,
#     sensor_score: float,
#     rag: RAGResult | None,
#     vision: VisionResult | None,
#     contributions: dict[str, float],
# ) -> tuple[str, str]:
#     """Construct Chain-of-Thought and final explanation.

#     In production this calls Phi-4 via the SLM endpoint.  For the scaffold
#     we generate a deterministic template that demonstrates the contract.
#     """
#     cot_parts: list[str] = ["Chain-of-Thought Reasoning:"]

#     # Step 1 — Sensor risk
#     cot_parts.append(f"1. Sensor model (LightGBM) predicts risk={sensor_score:.3f}.")

#     # Step 2 — RAG / manual context
#     if rag and rag.oem_limits_cited:
#         cot_parts.append(
#             f"2. RAG retrieval found {len(rag.chunks)} relevant chunks. "
#             f"OEM limits cited: {'; '.join(rag.oem_limits_cited)}."
#         )
#     else:
#         cot_parts.append("2. No manual/text context retrieved.")

#     # Step 3 — Vision findings
#     if vision and vision.defects_detected:
#         cot_parts.append(
#             f"3. Visual inspection (Phi-4) detected: {', '.join(vision.defects_detected)}. "
#             f"Description: {vision.description}"
#         )
#     else:
#         cot_parts.append("3. No casting image provided or no defects found.")

#     # Step 4 — Fusion
#     cot_parts.append(
#         f"4. Late fusion (weighted) yields risk={fused_prob:.3f} → {risk_level.value}. "
#         f"Active modality contributions: {contributions}."
#     )

#     chain_of_thought = "\n".join(cot_parts)

#     # Final human-readable explanation
#     explanation = (
#         f"Pump {risk_level.value.upper()} risk (probability {fused_prob:.1%}). "
#     )
#     explanation += f"Sensor risk score: {sensor_score:.3f}. "
#     if rag and rag.oem_limits_cited:
#         explanation += f"Manual references: {'; '.join(rag.oem_limits_cited[:2])}. "
#     if vision and vision.defects_detected:
#         explanation += f"Visual defects observed: {', '.join(vision.defects_detected)}."

#     return chain_of_thought, explanation
