"""
Diagnostic script to verify Supabase RAG retrieval with metadata filtering.

- Embeds a test query using BGE-M3 (sentence-transformers)
- Performs similarity search against Supabase `text_chunks` table
- Applies metadata filter: only retrieves rows where metadata->>'pump_id' matches the provided ID
- Prints retrieved text, similarity score, and category

Usage:
    python tests/test_retrieval.py --pump_id PUMP_01
"""
import os
import argparse
import sys
from typing import List, Dict

import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables (for Supabase keys, etc.)
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SCHEMA = os.getenv("SUPABASE_SCHEMA", "public")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("[ERROR] SUPABASE_URL and SUPABASE_KEY must be set in your environment.")
    sys.exit(1)

# Table and embedding config
TABLE_NAME = "text_chunks"
EMBEDDING_DIM = 1024  # BGE-M3 output size
MODEL_NAME = "BAAI/bge-m3"

# --- Helper Functions ---
def get_query_embedding(query: str) -> List[float]:
    """Embed the query using BGE-M3."""
    model = SentenceTransformer(MODEL_NAME)
    embedding = model.encode([query], normalize_embeddings=True)[0]
    return embedding.tolist()

def supabase_similarity_search(
    embedding: List[float],
    pump_id: str,
    top_k: int = 5
) -> List[Dict]:
    """
    Query Supabase for similar text chunks, filtered by pump_id in metadata.
    Returns a list of dicts with text, score, and category.
    """
    # Supabase PostgREST RPC endpoint for vector search (assumes pgvector extension)
    url = f"{SUPABASE_URL}/rest/v1/rpc/match_text_chunks"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "query_embedding": embedding,
        "match_count": top_k,
        "filter_pump_id": pump_id
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"[ERROR] Supabase RPC failed: {response.status_code} {response.text}")
        return []
    return response.json()

# --- Main Diagnostic ---
def main():
    parser = argparse.ArgumentParser(description="Test Supabase RAG retrieval with pump_id filter.")
    parser.add_argument("--pump_id", required=True, help="Pump ID to filter retrieval (e.g., PUMP_01)")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top results to retrieve")
    args = parser.parse_args()

    test_query = "What is the maximum operating temperature?"
    print(f"\n[INFO] Embedding test query: '{test_query}'\n")
    embedding = get_query_embedding(test_query)

    print(f"[INFO] Querying Supabase for pump_id = {args.pump_id} ...\n")
    results = supabase_similarity_search(embedding, args.pump_id, args.top_k)

    if not results:
        print("[WARN] No results found. Check your Supabase function and data.")
        return

    print(f"[RESULTS] Top {len(results)} matches for pump_id = {args.pump_id}:")
    for i, row in enumerate(results, 1):
        text = row.get("text")
        score = row.get("similarity") or row.get("score")
        category = row.get("category") or row.get("metadata", {}).get("category")
        print(f"\n--- Result #{i} ---")
        print(f"Text: {text[:200]}{'...' if len(text) > 200 else ''}")
        print(f"Similarity Score: {score}")
        print(f"Category: {category}")

if __name__ == "__main__":
    main()