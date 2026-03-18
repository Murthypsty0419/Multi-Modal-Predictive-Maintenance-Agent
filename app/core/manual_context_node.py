import json
from typing import Any
import asyncio
import re
import requests
from app.config import settings
from sentence_transformers import SentenceTransformer
import numpy as np

# Use Supabase REST API (PostgREST) instead of direct psycopg2
# Only HTTPS/port 443 is reachable; PostgreSQL ports are blocked
SUPABASE_URL = settings.supabase_url.rstrip('/')
SUPABASE_ANON_KEY = settings.supabase_anon_key
MANUAL_TOP_K = 6

_embedding_model: SentenceTransformer | None = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    return _embedding_model


def get_embedding(text: str):
    model = _get_embedding_model()
    return model.encode(text).tolist()


def _extract_json_payload(text: str) -> str:
    """Extract raw JSON text, including responses wrapped in ```json fences."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    return raw


def _ensure_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _ensure_list_of_str(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _format_coverage_duration(value: Any) -> str:
    """Normalize coverage duration to readable text (e.g., 5 -> '5-year')."""
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        years = int(value)
        return f"{years}-year"
    text = str(value).strip()
    if not text:
        return ""
    if text.isdigit():
        return f"{text}-year"
    return text


def _clean_summary_text(text: str, max_sentences: int = 3) -> str:
    """Remove boilerplate/repetition and keep concise sentence count."""
    raw = _extract_json_payload(text or "")
    raw = re.sub(r"\s+", " ", raw).strip()
    if not raw:
        return ""

    # Remove common preambles that LLMs sometimes add.
    preambles = [
        "based on",
        "here's",
        "i will",
        "for a technical report",
    ]
    lowered = raw.lower()
    for marker in preambles:
        idx = lowered.find(marker)
        if idx == 0 and ":" in raw[:180]:
            raw = raw.split(":", 1)[1].strip()
            break

    sentences = re.split(r"(?<=[.!?])\s+", raw)
    deduped: list[str] = []
    seen: set[str] = set()
    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        key = re.sub(r"\s+", " ", s_clean.lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s_clean)
        if len(deduped) >= max_sentences:
            break
    return " ".join(deduped).strip()


def _parse_embedding_value(value: Any) -> list[float] | None:
    """Handle pgvector payloads returned as JSON array or string like "[0.1, ...]"."""
    if isinstance(value, list):
        try:
            return [float(v) for v in value]
        except Exception:
            return None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [float(v) for v in parsed]
        except Exception:
            # Fallback parser for non-JSON vector strings.
            try:
                stripped = text.strip("[]")
                if not stripped:
                    return None
                return [float(part.strip()) for part in stripped.split(",") if part.strip()]
            except Exception:
                return None

    return None

def groq_generate(prompt: str) -> str:
    api_key = settings.groq_api_key
    endpoint = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1024,
        "temperature": 0.2
    }
    response = requests.post(endpoint, headers=headers, json=data)
    resp_json = response.json()
    try:
        return resp_json["choices"][0]["message"]["content"]
    except Exception:
        return ""

async def manual_context_node(state: dict[str, Any]) -> dict[str, Any]:
    anomaly_query = state.get("anomaly_query") or (
        "Please provide Capabilities, Operating limits, sensor thresholds, "
        "Troubleshooting steps, symptoms, and mechanical fault causes, "
        "Warranty: Legal terms, coverage periods, and liability."
    )
    print(f"[manual_context_node] anomaly_query used for embedding: {anomaly_query}")
    loop = asyncio.get_event_loop()

    def fetch_top_k_chunks_rest(query_embedding, k=MANUAL_TOP_K):
        """Fetch vec_manuals via Supabase REST API (PostgREST) over HTTPS."""
        try:
            url = f"{SUPABASE_URL}/rest/v1/vec_manuals"
            embedding_arr = [float(x) for x in np.round(query_embedding, 6)]
            
            headers = {
                "apikey": SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
                "Content-Type": "application/json",
            }
            
            # Fetch records with correct column names: chunk_id, content, embedding, page
            params = {"select": "chunk_id,content,embedding,page", "limit": "1000"}
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code != 200:
                error_text = response.text[:300] if response.text else ""
                print(f"[manual_context_node] REST API error {response.status_code}: {error_text}")
                return []
            
            rows = response.json()
            if not rows or not isinstance(rows, list):
                print(f"[manual_context_node] No rows or invalid response format: {type(rows)}")
                return []
            
            # Calculate cosine similarity and sort
            def cosine_sim(a, b):
                a_np = np.array(a)
                b_np = np.array(b)
                norm_product = np.linalg.norm(a_np) * np.linalg.norm(b_np)
                if norm_product < 1e-8:
                    return 0.0
                return float(np.dot(a_np, b_np) / norm_product)
            
            scored = []
            for row in rows:
                try:
                    embedding = _parse_embedding_value(row.get("embedding"))
                    content = row.get("content")
                    if embedding and content:
                        sim = cosine_sim(embedding_arr, embedding)
                        scored.append((sim, content, row.get("chunk_id"), row.get("page")))
                except Exception as calc_err:
                    print(f"[manual_context_node] Error computing similarity for row: {calc_err}")
                    continue
            
            if not scored:
                print(f"[manual_context_node] No valid embeddings found in {len(rows)} rows")
                return []
            
            scored.sort(reverse=True, key=lambda x: x[0])
            selected = scored[:k]
            print(f"[manual_context_node] Found {len(scored)} chunks, returning top {k}")
            debug_selected = [
                {
                    "rank": i + 1,
                    "score": round(item[0], 5),
                    "chunk_id": item[2],
                    "page": item[3],
                }
                for i, item in enumerate(selected)
            ]
            print(f"[manual_context_node] top_k_selected={debug_selected}")
            return [(content,) for _, content, _, _ in selected]
            
        except Exception as exc:
            print(f"[manual_context_node] REST API fetch failed: {exc}")
            import traceback
            traceback.print_exc()
            return []

    try:
        top_chunks = await loop.run_in_executor(
            None,
            lambda: fetch_top_k_chunks_rest(get_embedding(anomaly_query), MANUAL_TOP_K)
        )
    except Exception as exc:
        print(f"[manual_context_node] top-k retrieval skipped due to DB connectivity issue: {exc}")
        top_chunks = []
    raw_text = " ".join([row[0] for row in top_chunks if row[0]])

    # Grounded-mode guardrail: never ask the LLM to infer manual facts from empty evidence.
    if not raw_text.strip():
        print("[manual_context_node] No grounded manual chunks retrieved; skipping LLM extraction to prevent hallucinations.")
        state = dict(state)
        state["manual_context"] = {
            "normal_range": {},
            "causes": {},
            "warranty": {},
        }
        state["manual_evidence_summary"] = (
            "Manual evidence unavailable for this request (retrieval failed or returned no chunks). "
            "No OEM limits/causes were inferred."
        )
        return state

    # # Log the extracted text from the top-k chunks
    # print("[manual_context_node] Extracted top-k chunk text:")
    # print(raw_text)

    prompt = (
    "You are a specialized Reliability Engineer. Analyze the provided pump manual text "
    "to create a structured diagnostic JSON. Follow these strict rules:\n\n"
    "1. normal_range: Extract ONLY specific numeric thresholds for the specific sensors mentioned in the query(e.g., 'Max Pressure: 60 psi').\n"
    "2. causes: Group troubleshooting steps by symptom and give only for the specific symptom mentioned in prompt. Format as 'Symptom: Possible Cause' "
    "(e.g., 'Vibration: Loose magnet').\n"
    "3. warranty: Extract the coverage duration and the main 'voiding' conditions.\n\n"
    f"TEXT:\n{raw_text}\n\n"
    "Respond ONLY with a valid JSON object. Do not include introductory text. "
    "If a value is unknown, use null."
    )

    groq_response = await loop.run_in_executor(
        None,
        lambda: groq_generate(prompt)
    )

    # Log the LLM response
    print("\n[manual_context_node] LLM response:\n")
    print(groq_response)

    try:
        payload = _extract_json_payload(groq_response)
        parsed = json.loads(payload)
        manual_json = parsed if isinstance(parsed, dict) else {}
    except Exception:
        manual_json = {}

    # Normalize expected top-level structure to prevent runtime type errors.
    manual_json = {
        "normal_range": manual_json.get("normal_range", {}),
        "causes": manual_json.get("causes", {}),
        "warranty": manual_json.get("warranty", {}),
    }
    
    # --- Format as Natural Language via Groq (limits + causes + actions) ---
    triggered_sensors = state.get("triggered_sensors", [])
    
    # Extract warranty directly (no LLM) to prevent hallucination
    warranty_data = _ensure_dict(manual_json.get("warranty", {}))
    coverage_duration = _format_coverage_duration(warranty_data.get("coverage_duration", ""))
    voiding_conditions = _ensure_list_of_str(warranty_data.get("voiding_conditions", []))
    warranty_fact = ""
    if coverage_duration:
        voiding_str = ", ".join(voiding_conditions[:3]) if voiding_conditions else "standard terms apply"
        warranty_fact = f"{coverage_duration} warranty covers manufacturing defects but excludes {voiding_str}."
    
    # LLM handles limits + causes + action items reasoning (not warranty)
    format_prompt = (
        "You are a Reliability Engineer writing a maintenance report. "
        "Extract limits, likely causes, and immediate action items based on OEM manual data.\n\n"
        f"Triggered Sensors: {', '.join(triggered_sensors) if triggered_sensors else 'None'}\n"
        f"OEM Operating Limits: {json.dumps(_ensure_dict(manual_json.get('normal_range', {})), indent=2)}\n"
        f"Causes by Symptom: {json.dumps(_ensure_dict(manual_json.get('causes', {})), indent=2)}\n\n"
        "OUTPUT REQUIREMENTS:\n"
        "- 2-3 sentences: key limits + likely causes + one immediate action item\n"
        "- No warranty discussion, no preamble, no meta-commentary\n"
        "- Factual, direct language. Action items should be specific (e.g., 'inspect bearings', 'check alignment')\n"
        "- Start directly with evidence\n\n"
        "Example: 'OEM limits specify max pressure 60 bar and vibration 7.1 mm/s. "
        "Given Vibration and Pressure anomalies, likely causes include bearing wear or misalignment. "
        "Immediately inspect bearings for wear and verify shaft alignment.'\n\n"
        "Now provide limits, causes, and action items:"
    )
    
    if triggered_sensors:
        evidence_summary = await loop.run_in_executor(
            None,
            lambda: groq_generate(format_prompt)
        )
        evidence_summary = _clean_summary_text(evidence_summary, max_sentences=3)
    else:
        evidence_summary = (
            "OEM manual context retrieved. No sensor-specific anomalies were triggered from historical baseline, "
            "so no sensor-targeted manual action is recommended from this node."
        )
    
    # Append warranty fact (deterministic, no hallucination)
    if warranty_fact:
        evidence_summary = f"{evidence_summary.strip()} {warranty_fact}".strip()
    
    print("[manual_context_node] Evidence summary:\n", evidence_summary)
    
    state = dict(state)
    state["manual_context"] = manual_json
    state["manual_evidence_summary"] = evidence_summary.strip() if evidence_summary else None
    return state