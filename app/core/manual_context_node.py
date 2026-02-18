import json
from typing import Any
import asyncio
import requests
from app.config import settings
from sentence_transformers import SentenceTransformer
import psycopg2
import numpy as np

PG_CONN_STR = settings.psycopg2_dsn

embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
def get_embedding(text: str):
    return embedding_model.encode(text).tolist()

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

    def fetch_top_k_chunks_pg(query_embedding, k=4):
        conn = psycopg2.connect(PG_CONN_STR)
        cur = conn.cursor()
        embedding_str = '[' + ','.join(str(float(x)) for x in np.round(query_embedding, 6)) + ']'
        cur.execute(
            """
            SELECT content
            FROM vec_manuals
            ORDER BY embedding <=> %s::vector(1024)
            LIMIT %s;
            """,
            (embedding_str, k)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows

    top_chunks = await loop.run_in_executor(
        None,
        lambda: fetch_top_k_chunks_pg(get_embedding(anomaly_query), 3)
    )
    raw_text = " ".join([row[0] for row in top_chunks if row[0]])

    # Log the extracted text from the top-k chunks
    print("[manual_context_node] Extracted top-k chunk text:")
    print(raw_text)

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
    print("[manual_context_node] LLM response:")
    print(groq_response)

    try:
        manual_json = json.loads(groq_response)
    except Exception:
        manual_json = {"normal_range": "", "causes": "", "warranty": ""}
    state = dict(state)
    state["manual_context"] = manual_json
    return state