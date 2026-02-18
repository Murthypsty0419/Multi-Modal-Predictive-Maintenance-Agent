"""
Sync transactional maintenance data (service schedules, work done logs, maintenance requests) into Supabase.
- Upserts for schedules and logs
- Full insert for maintenance requests
- Double-write maintenance request descriptions to text_chunks with BGE-M3 embedding and rich metadata

Usage:
    python scripts/sync_transactional_to_supabase.py \
        --schedules_path data/transactional/service_schedules.csv \
        --logs_path data/transactional/work_done_logs.csv \
        --requests_path data/transactional/maintenance_requests.csv
"""
import argparse
import os
import polars as pl
from sqlalchemy import create_engine, MetaData, Table, insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from datetime import datetime
import uuid
import json

# --- Config ---
load_dotenv()
DB_URL = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")
if not DB_URL:
    raise RuntimeError("DATABASE_URL or SUPABASE_DB_URL must be set in environment.")

MODEL_NAME = "BAAI/bge-m3"
EMBED_DIM = 1024

# --- Embedding Model ---
_embed_model = None
def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(MODEL_NAME)
    return _embed_model

def embed_text(text: str):
    model = get_embed_model()
    return model.encode([text], normalize_embeddings=True)[0].tolist()

# --- Main Sync Logic ---
def main():
    parser = argparse.ArgumentParser(description="Sync transactional maintenance data to Supabase.")
    parser.add_argument("--schedules_path", default="data/transactional/service_schedules.csv")
    parser.add_argument("--logs_path", default="data/transactional/work_done_logs.csv")
    parser.add_argument("--requests_path", default="data/transactional/maintenance_requests.csv")
    args = parser.parse_args()

    # Read CSVs
    schedules = pl.read_csv(args.schedules_path)
    logs = pl.read_csv(args.logs_path)
    requests = pl.read_csv(args.requests_path)

    # DB setup
    engine = create_engine(DB_URL)
    metadata = MetaData()
    metadata.reflect(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # --- Upsert Service Schedules ---
    schedule_table = metadata.tables["service_schedules"]
    for row in schedules.iter_rows(named=True):
        stmt = pg_insert(schedule_table).values(**row)
        stmt = stmt.on_conflict_do_update(
            index_elements=["pump_id", "task_name"],
            set_={k: row[k] for k in row if k not in ("pump_id", "task_name")}
        )
        session.execute(stmt)
    session.commit()
    print(f"Upserted {len(schedules)} service schedules.")

    # --- Upsert Work Done Logs ---
    logs_table = metadata.tables["work_done_logs"]
    for row in logs.iter_rows(named=True):
        # Convert timestamp to datetime if needed
        if isinstance(row["timestamp"], str):
            row["timestamp"] = datetime.fromisoformat(row["timestamp"])
        stmt = pg_insert(logs_table).values(**row)
        stmt = stmt.on_conflict_do_update(
            index_elements=["pump_id", "task_name", "hours_at_service"],
            set_={k: row[k] for k in row if k not in ("pump_id", "task_name", "hours_at_service")}
        )
        session.execute(stmt)
    session.commit()
    print(f"Upserted {len(logs)} work done logs.")

    # --- Insert Maintenance Requests ---
    requests_table = metadata.tables["maintenance_requests"]
    text_chunks_table = metadata.tables["text_chunks"]
    for row in requests.iter_rows(named=True):
        # Convert created_at to datetime if needed
        if isinstance(row["created_at"], str):
            row["created_at"] = datetime.fromisoformat(row["created_at"])
        # Insert into maintenance_requests
        req_id = str(uuid.uuid4())
        db_row = dict(row)
        db_row["id"] = req_id
        session.execute(insert(requests_table).values(**db_row))
        # Double-write to text_chunks
        embedding = embed_text(row["description"])
        chunk_id = str(uuid.uuid4())
        metadata_obj = {
            "category": "maintenance_request",
            "pump_id": row["pump_id"],
            "priority": row["priority"],
            "status": row["status"]
        }
        text_chunk = {
            "chunk_id": chunk_id,
            "source_document": "maintenance_requests.csv",
            "section": None,
            "content": row["description"],
            "embedding": embedding,
            "metadata": json.dumps(metadata_obj),
            "created_at": row["created_at"]
        }
        session.execute(insert(text_chunks_table).values(**text_chunk))
    session.commit()
    print(f"Inserted {len(requests)} maintenance requests and double-wrote to text_chunks.")

    session.close()
    print("Sync complete.")

if __name__ == "__main__":
    main()
