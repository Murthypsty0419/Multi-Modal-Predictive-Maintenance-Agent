from __future__ import annotations
from app.core.manual_context_node import manual_context_node
# ... any other imports like 'import os' ...

print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print("!!! ORCHESTRATOR IS LOADING !!!")
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

# ... rest of your code ...
import logging
from app.config import settings
from typing import Any

from langgraph.graph import END, StateGraph

from app.schemas import AgentState, Modality

logger = logging.getLogger("orchestrator")
logger.propagate = False


# ---------------------------------------------------------------------------
# Node functions — each takes AgentState dict, returns partial update
# ---------------------------------------------------------------------------

async def sensor_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run LightGBM inference + SHAP on sensor data (primary witness; independent of historical CSV)."""
    from app.core.nodes.sensor_node import sensor_node as _sensor_node
    result = await _sensor_node(state)
    merged = dict(state)
    merged.update(result)
    # Always preserve manual_context and anomaly_query
    if "manual_context" not in merged and "manual_context" in state:
        merged["manual_context"] = state["manual_context"]
    if "anomaly_query" not in merged and "anomaly_query" in state:
        merged["anomaly_query"] = state["anomaly_query"]
    # Append sensor anomaly findings to anomaly_query if detected
    sensor_anomaly = result.get("sensor_anomaly")
    if sensor_anomaly:
        prev_query = merged.get("anomaly_query", "")
        merged["anomaly_query"] = prev_query + f" Sensor anomaly detected: {sensor_anomaly}."
    return merged




async def vision_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run Groq LLM multimodal analysis: send manual context and image, append LLM answer to context."""
    import os
    import json
    import base64
    try:
        from groq import Groq
    except Exception as exc:
        print(f"[VISION NODE] Groq SDK unavailable; skipping vision analysis: {exc}")
        return dict(state)
    agent = AgentState(**state)
    if agent.vision_input is None:
        return dict(state)
    print("\n🖼️[VISION NODE]")
    manual_context = state.get("manual_context")
    import re
    causes_text = None
    # Handle manual_context as dict or string
    if manual_context:
        if isinstance(manual_context, dict):
            # Try to extract 'causes' key directly
            causes_text = manual_context.get("causes")
            # If not found, fallback to string conversion
            if not causes_text:
                manual_context_str = str(manual_context)
                match = re.search(r"(?i)(causes\\s*:?)(.*?)(\\n\\s*\\w+\\s*:?|$)", manual_context_str, re.DOTALL)
                if match:
                    causes_text = match.group(2).strip()
        elif isinstance(manual_context, str):
            match = re.search(r"(?i)(causes\\s*:?)(.*?)(\\n\\s*\\w+\\s*:?|$)", manual_context, re.DOTALL)
            if match:
                causes_text = match.group(2).strip()
    prompt = None
    if causes_text:
        prompt = (
            "Given the following list of possible causes from the technical manual:\n"
            f"{causes_text}\n"
            "and the attached image, do you see any visual signs or evidence of these causes in the image? only focus on signs of causes even  though u see few good parts. And if u see multiple objecs, just talk abt the one wiith signs of the causes.\n"
            "Respond in JSON only with this schema: "
            "{\"summary\": str, \"has_leaks\": bool, \"has_cracks\": bool, \"has_corrosion\": bool, "
            "\"confidence_leaks\": float, \"confidence_cracks\": float, \"confidence_corrosion\": float, "
            "\"llm_visual_risk_score\": float}."
        )
    else:
        prompt = (
            "Given the attached image and the technical manual context, do you see any visual signs of faults or issues?\n"
            "Respond in JSON only with this schema: "
            "{\"summary\": str, \"has_leaks\": bool, \"has_cracks\": bool, \"has_corrosion\": bool, "
            "\"confidence_leaks\": float, \"confidence_cracks\": float, \"confidence_corrosion\": float, "
            "\"llm_visual_risk_score\": float}."
        )
    image_url = None
    image_b64 = None
    if hasattr(agent.vision_input, "image_base64") and agent.vision_input.image_base64:
        print("[VISION NODE] Received image_base64 from frontend/API.")
        image_b64 = agent.vision_input.image_base64
    elif hasattr(agent.vision_input, "image_path") and agent.vision_input.image_path:
        print(f"[VISION NODE] Reading image from path: {agent.vision_input.image_path}")
        with open(agent.vision_input.image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
    if not image_b64:
        print("[VISION NODE] No image found in vision_input. Skipping vision analysis.")
        return dict(state)
    # Use Groq's Vision-compatible model
    api_key = state.get("groq_api_key") or getattr(getattr(state, "settings", None), "groq_api_key", None) or settings.groq_api_key or os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                    ]
                }
            ],
            temperature=0.2,
            max_completion_tokens=256,
            top_p=1,
            stream=False,
            stop=None,
        )
        print(f"[VISION NODE] Groq Vision completion: {completion}")
        vision_summary = completion.choices[0].message.content or ""
    except Exception as e:
        print(f"[VISION NODE] Error during Groq Vision API call or response parsing: {e}")
        vision_summary = ""

    # Try structured JSON first; fallback to rule-based extraction from free text.
    parsed: dict[str, Any] = {}
    json_candidate = (vision_summary or "").strip()
    # Strip common markdown fences like ```json ... ``` before parsing.
    if json_candidate.startswith("```"):
        lines = json_candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        json_candidate = "\n".join(lines).strip()
    if "{" in json_candidate and "}" in json_candidate:
        start = json_candidate.find("{")
        end = json_candidate.rfind("}")
        json_candidate = json_candidate[start:end + 1]
    try:
        parsed = json.loads(json_candidate) if json_candidate else {}
    except Exception:
        parsed = {}

    summary_text = str(parsed.get("summary") or vision_summary or "").strip()
    lower = summary_text.lower()

    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    has_leaks = bool(parsed.get("has_leaks", any(k in lower for k in ["leak", "seep", "drip"])))
    has_cracks = bool(parsed.get("has_cracks", any(k in lower for k in ["crack", "fracture"])))
    has_corrosion = bool(parsed.get("has_corrosion", any(k in lower for k in ["corrosion", "rust", "oxidation", "pitting"])))

    confidence_leaks = max(0.0, min(1.0, _as_float(parsed.get("confidence_leaks"), 0.8 if has_leaks else 0.0)))
    confidence_cracks = max(0.0, min(1.0, _as_float(parsed.get("confidence_cracks"), 0.8 if has_cracks else 0.0)))
    confidence_corrosion = max(0.0, min(1.0, _as_float(parsed.get("confidence_corrosion"), 0.75 if has_corrosion else 0.0)))

    deterministic_score = min(
        1.0,
        0.45 * confidence_cracks + 0.35 * confidence_leaks + 0.20 * confidence_corrosion,
    )
    llm_score = max(0.0, min(1.0, _as_float(parsed.get("llm_visual_risk_score"), deterministic_score)))
    visual_risk_score = max(0.0, min(1.0, 0.7 * deterministic_score + 0.3 * llm_score))

    defects_detected: list[str] = []
    if has_leaks:
        defects_detected.append("leak")
    if has_cracks:
        defects_detected.append("crack")
    if has_corrosion:
        defects_detected.append("corrosion")

    is_casting_present = bool(summary_text) and ("no casting" not in lower)

    vision_result = {
        "is_casting_present": is_casting_present,
        "has_corrosion": has_corrosion,
        "has_leaks": has_leaks,
        "has_cracks": has_cracks,
        "confidence_casting_present": 1.0 if is_casting_present else 0.0,
        "confidence_corrosion": confidence_corrosion,
        "confidence_leaks": confidence_leaks,
        "confidence_cracks": confidence_cracks,
        "visual_risk_score": visual_risk_score,
        "defects_detected": defects_detected,
        "description": summary_text,
        "severity_score": visual_risk_score,
    }

    merged = dict(state)
    merged["vision_summary"] = summary_text
    merged["vision_result"] = vision_result
    # Always preserve manual_context and anomaly_query
    if "manual_context" not in merged and "manual_context" in state:
        merged["manual_context"] = state["manual_context"]
    if "anomaly_query" not in merged and "anomaly_query" in state:
        merged["anomaly_query"] = state["anomaly_query"]
    # Optionally append to anomaly_query or context
    prev_query = merged.get("anomaly_query", "")
    merged["anomaly_query"] = prev_query + f" vision_summary: {vision_summary};"
    return merged




async def feature_node(state: dict[str, Any]) -> dict[str, Any]:
    print("\n📈[FEATURE NODE]")
    from app.core.feature_node import feature_node as _feature_node
    result = await _feature_node(state)
    merged = dict(state)
    merged.update(result)
    
    # Print detailed output
    history_risk = result.get("history_risk_score")
    is_outlier = result.get("is_historical_outlier", False)
    triggered = result.get("triggered_sensors", [])
    if history_risk is not None:
        print(f"  ✅ history_risk_score={history_risk:.4f}")
        print(f"     is_historical_outlier={is_outlier}")
        print(f"     triggered_sensors={triggered}")
    else:
        print(f"  ⚠️  no historical data available (history_risk_score=None)")
    
    # Always preserve manual_context and anomaly_query
    if "manual_context" not in merged and "manual_context" in state:
        merged["manual_context"] = state["manual_context"]
    if "anomaly_query" not in merged and "anomaly_query" in state:
        merged["anomaly_query"] = state["anomaly_query"]
    return merged




async def service_age_node(state: dict[str, Any]) -> dict[str, Any]:
    print("\n⏳[SERVICE AGE NODE]")
    """Check if a maintenance task is overdue for the pump."""
    try:
        from app.core.service_age_node import service_age_node as _service_age_node
        result = await _service_age_node(state)
        merged = dict(state)
        merged.update(result)
        
        # Print detailed output
        service_age_risk = result.get("service_age_risk_score")
        transactional_risk = result.get("transactional_risk_score")
        overdue = result.get("overdue_tasks", [])
        open_reqs = result.get("open_requests", [])
        age_val = service_age_risk if service_age_risk is not None else 0.0
        trans_val = transactional_risk if transactional_risk is not None else 0.0
        if service_age_risk is not None or transactional_risk is not None:
            print(f"  ✅ service_age_risk_score={age_val:.4f}")
            print(f"     transactional_risk_score={trans_val:.4f}")
            if overdue:
                print(f"     overdue_tasks={len(overdue)}")
                print(f"     overdue_tasks_json={overdue}")
            if open_reqs:
                print(f"     open_requests={len(open_reqs)}")
                print(f"     open_requests_json={open_reqs}")
        else:
            print(f"  ⚠️  no transactional data or scores computed")
        
        # Always preserve manual_context and anomaly_query
        if "manual_context" not in merged and "manual_context" in state:
            merged["manual_context"] = state["manual_context"]
        if "anomaly_query" not in merged and "anomaly_query" in state:
            merged["anomaly_query"] = state["anomaly_query"]
        # Append service age anomaly findings to anomaly_query if detected
        service_age_anomaly = result.get("service_age_anomaly")
        if service_age_anomaly:
            prev_query = merged.get("anomaly_query", "")
            merged["anomaly_query"] = prev_query + f" Service age anomaly detected: {service_age_anomaly}."
        return merged
    except Exception as e:
        print(f"⚠️ DATABASE SKIPPED (service_age): {e}")
        return state


async def fusion_node(state: dict[str, Any]) -> dict[str, Any]:
    print("\n🧩[FUSION NODE]")
    """Late fusion + XAI narrative generation."""
    from app.core.fusion import fuse_and_explain

    # Try to reconstruct AgentState, but handle gracefully if state is incomplete
    try:
        agent = AgentState(**state)
    except Exception:
        # If state is incomplete, return a default report with required fields
        fallback = {
            "fused_score": 0.0,
            "status_label": "low",
            "explanation": "Unable to process: incomplete state",
            "top_signals": [],
            "action_items": ["Check system configuration"]
        }
        # Copy required fields from input state if present
        for key in ("pump_id", "current_total_hours", "request_id"):
            if key in state:
                fallback[key] = state[key]
        merged = dict(state)
        merged.update(fallback)
        return merged
    update = await fuse_and_explain(agent)
    merged = dict(state)
    merged.update(update)
    # Always preserve manual_context and anomaly_query
    if "manual_context" not in merged and "manual_context" in state:
        merged["manual_context"] = state["manual_context"]
    if "anomaly_query" not in merged and "anomaly_query" in state:
        merged["anomaly_query"] = state["anomaly_query"]
    return merged


# ---------------------------------------------------------------------------
# Conditional edge logic — pick which modality nodes to run
# ---------------------------------------------------------------------------

def route_modalities(state: dict[str, Any]) -> list[str]:
    """Return a list of node names to execute in parallel."""
    modalities = state.get("available_modalities", [])
    # Normalize all modalities to lower-case strings for robust comparison
    norm_modalities = [m.value.lower() if hasattr(m, 'value') else str(m).lower() for m in modalities]
    nodes: list[str] = []
    if "sensor" in norm_modalities:
        nodes.append("sensor_node")
    if "text" in norm_modalities:
        nodes.append("text_node")
    if "vision" in norm_modalities:
        nodes.append("vision_node")
    # Only route to feature_node if 'historical' or 'history' is present
    if "historical" in norm_modalities or "history" in norm_modalities:
        nodes.append("feature_node")
    out = nodes if nodes else ["fusion_node"]
    logger.info("[orchestrator] route_modalities: modalities=%s (normalized: %s) -> nodes=%s", modalities, norm_modalities, out)
    return out


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------



# --- StateGraph Implementation (reverted) ---
def build_graph() -> StateGraph:
    graph = StateGraph(dict)
    # Add nodes (linear pipeline)
    graph.add_node("sensor_node", sensor_node)
    graph.add_node("service_age_node", service_age_node)
    graph.add_node("feature_node", feature_node)
    graph.add_node("manual_context_node", manual_context_node)
    graph.add_node("vision_node", vision_node)
    graph.add_node("fusion_node", fusion_node)
    # Route immediately after sensor: run service only when request-scoped
    # transactional files are present.
    def route_after_sensor(state):
        print(f"\n⚡ [route_after_sensor] Checking transactional paths in state...")
        print(f"   State keys: {list(state.keys())[:15]}... (truncated)")
        wdl = state.get("work_done_logs_path")
        ss = state.get("service_schedules_path")
        mr = state.get("maintenance_requests_path")
        print(f"   wdl={wdl}")
        print(f"   ss={ss}")
        print(f"   mr={mr}")
        has_transactional_inputs = any(
            bool(state.get(k))
            for k in ("work_done_logs_path", "service_schedules_path", "maintenance_requests_path")
        )
        logger.info("[orchestrator] route_after_sensor: wdl=%s, ss=%s, mr=%s (paths: %s, %s, %s)", bool(wdl), bool(ss), bool(mr), wdl, ss, mr)
        if has_transactional_inputs:
            print(f"   ✅ transactional files detected → routing to service_age_node")
            logger.info("[orchestrator] ✅ transactional files detected → routing to service_age_node")
            return "service_age_node"
        print(f"   ⚠️ no transactional files in request state → skipping service_age_node")
        logger.info("[orchestrator] ⚠️  no transactional files in request state → skipping service_age_node")
        return route_after_service(state)

    # Request-scoped routing after service/feature/manual nodes.
    def route_after_service(state):
        modalities = state.get("available_modalities", [])
        norm = [m.value.lower() if hasattr(m, "value") else str(m).lower() for m in modalities]
        logger.info("[orchestrator] route_after_service: modalities=%s, normalized=%s", modalities, norm)
        if "historical" in norm:
            logger.info("[orchestrator] → routing to feature_node (historical modality detected)")
            return "feature_node"
        if "text" in norm:
            logger.info("[orchestrator] → routing to manual_context_node (text modality detected)")
            return "manual_context_node"
        if "vision" in norm:
            logger.info("[orchestrator] → routing to vision_node (vision modality detected)")
            return "vision_node"
        logger.info("[orchestrator] → routing to fusion_node (no historical/text/vision modalities)")
        return "fusion_node"

    # Request-scoped routing after feature/manual nodes.
    def route_after_feature(state):
        modalities = state.get("available_modalities", [])
        norm = [m.value.lower() if hasattr(m, "value") else str(m).lower() for m in modalities]
        logger.info("[orchestrator] route_after_feature: modalities=%s, normalized=%s", modalities, norm)
        if "text" in norm:
            logger.info("[orchestrator] → routing to manual_context_node (text modality detected)")
            return "manual_context_node"
        if "vision" in norm:
            logger.info("[orchestrator] → routing to vision_node (vision modality detected)")
            return "vision_node"
        logger.info("[orchestrator] → routing to fusion_node (no text/vision modalities after feature_node)")
        return "fusion_node"

    def route_after_manual(state):
        modalities = state.get("available_modalities", [])
        norm = [m.value.lower() if hasattr(m, "value") else str(m).lower() for m in modalities]
        logger.info("[orchestrator] route_after_manual: modalities=%s, normalized=%s", modalities, norm)
        if "vision" in norm:
            logger.info("[orchestrator] → routing to vision_node (vision modality detected)")
            return "vision_node"
        logger.info("[orchestrator] → routing to fusion_node (no vision modality after manual_context_node)")
        return "fusion_node"

    graph.set_entry_point("sensor_node")
    graph.add_conditional_edges("sensor_node", route_after_sensor)
    graph.add_conditional_edges("service_age_node", route_after_service)
    graph.add_conditional_edges("feature_node", route_after_feature)
    graph.add_conditional_edges("manual_context_node", route_after_manual)
    graph.add_edge("vision_node", "fusion_node")
    graph.add_edge("fusion_node", END)
    return graph

# Compile once at module level
_compiled_graph = build_graph().compile()


async def run_agent(state: AgentState) -> AgentState:
    """Execute the orchestrator graph and return the final AgentState."""
    logger.info(
        "[orchestrator] run_agent start: pump_id=%s, modalities=%s",
        state.pump_id,
        [m.value for m in state.available_modalities] if state.available_modalities else [],
    )
    initial = state.model_dump()
    result = await _compiled_graph.ainvoke(initial)
    logger.info("[orchestrator] run_agent done: graph finished")
    # Ensure pump_id and current_total_hours are preserved
    if "pump_id" not in result or result["pump_id"] is None:
        result["pump_id"] = state.pump_id
    if "current_total_hours" not in result or result["current_total_hours"] is None:
        result["current_total_hours"] = state.current_total_hours

    # --- Cleanup historical_logs CSVs if present ---
    try:
        hist_path = None
        if hasattr(state, 'historical_logs_path') and state.historical_logs_path:
            hist_path = state.historical_logs_path
        elif isinstance(state, dict) and state.get('historical_logs_path'):
            hist_path = state['historical_logs_path']
        if hist_path:
            import os
            if os.path.exists(hist_path):
                os.remove(hist_path)
                logger.info(f"[orchestrator] Cleaned up historical_logs file: {hist_path}")
            else:
                logger.info(f"[orchestrator] historical_logs file already gone or inaccessible: {hist_path}")
        else:
            logger.info("[orchestrator] No historical_logs_path in state; cleanup skipped")
    except Exception as cleanup_exc:
        logger.warning(f"[orchestrator] Cleanup failed: {cleanup_exc}")

    return AgentState(**result)