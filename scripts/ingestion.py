"""
Phase 1: Knowledge Ingestion (The Librarian)
- Parse PDF manuals using LlamaParse
- Chunk and embed with BGE-M3
- Store embeddings and metadata in Supabase (vec_manuals table)
- All DB/API calls async and resilient
"""


import os
import asyncio
import warnings
from typing import List, Dict, Any
from dotenv import load_dotenv
from llama_parse import LlamaParse
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

# Suppress DeprecationWarnings and UserWarnings from llama-parse
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Load .env automatically
load_dotenv()


SUPABASE_URL = os.environ.get("SUPABASE_URL")
# Prefer service role key for full permissions (deletion)
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_ANON_KEY")

def parse_pdf_manual(pdf_path: str, api_key: str) -> List[str]:
    # Decrease chunk size for finer granularity and add 15% overlap
    chunk_size = 512
    chunk_overlap = int(chunk_size * 0.15)
    parser = LlamaParse(
        api_key=api_key,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        output_options={"extract_printed_page_number": True}
    )
    docs = parser.load_data(pdf_path)
    # Try to extract page from .metadata['page'], fallback to chunk order
    def get_page(doc, idx):
        if hasattr(doc, "metadata") and isinstance(doc.metadata, dict) and "page" in doc.metadata:
            return doc.metadata["page"]
        return idx + 1  # fallback: chunk order as page
    return [{"content": doc.text, "page": get_page(doc, idx)} for idx, doc in enumerate(docs)]




# Load BGE Large embedding model
def load_bge_embedding_model():
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    return model

def embed_chunks_bge(chunks, model):
    records = []
    for chunk in chunks:
        vector = model.encode(chunk["content"]).tolist()
        records.append({
            "chunk_id": os.urandom(16).hex(),
            "content": chunk["content"],
            "embedding": vector,
            "page": chunk["page"]
        })
    return records

async def store_chunks_supabase(records: List[Dict[str, Any]]):
    client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    try:
        # Empty the table first (delete all rows using page column)
        client.table("vec_manuals").delete().neq("page", 0).execute()
        print("[DB] Cleared vec_manuals table.")
    except Exception as e:
        print(f"[DB ERROR] Failed to clear table: {e}")
    for rec in records:
        try:
            client.table("vec_manuals").insert(rec).execute()
            # print(f"[DB] Inserted: {rec['content'][:30]}...")  # Logging removed for speed
        except Exception as e:
            print(f"[DB ERROR] Failed to insert chunk: {e}")


async def ingest_manual(pdf_path: str):
    llama_api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
    if not llama_api_key:
        raise RuntimeError("LLAMA_CLOUD_API_KEY not set in environment.")
    chunks = parse_pdf_manual(pdf_path, llama_api_key)
    model = load_bge_embedding_model()
    records = embed_chunks_bge(chunks, model)
    await store_chunks_supabase(records)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingestion.py <manual.pdf>")
        exit(1)
    pdf_path = sys.argv[1]
    asyncio.run(ingest_manual(pdf_path))
