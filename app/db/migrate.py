"""
Alembic-free migration helper.

Run `python -m app.db.migrate` to create / update tables.
Uses the sync DSN so it can run as a simple script during container startup.
"""

from __future__ import annotations

from sqlalchemy import create_engine, text

from app.config import settings
from app.db.models import Base


def run_migrations() -> None:
    engine = create_engine(settings.postgres_dsn_sync, echo=True)

    with engine.begin() as conn:
        # Ensure pgvector extension exists
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    Base.metadata.create_all(engine)
    print("✔ Database tables created / verified.")
    engine.dispose()


if __name__ == "__main__":
    run_migrations()
