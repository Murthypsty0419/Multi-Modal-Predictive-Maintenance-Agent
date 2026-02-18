"""
Centralised settings loaded from environment / .env file.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict



class Settings(BaseSettings):
    # ── Gemini LLM ──────────────────────────────────────────────────
    gemini_api_key: str = ""
    # ── Groq LLM ───────────────────────────────────────────────────
    groq_api_key: str = ""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Supabase / Postgres ───────────────────────────────────────────
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "oxmaint"
    postgres_user: str = "oxmaint"
    postgres_password: str = "changeme"

    # ── Model IDs / Paths ────────────────────────────────────────────
    phi4_model_id: str = "microsoft/Phi-4-multimodal-instruct"
    bge_m3_model_id: str = "BAAI/bge-m3"
    lightgbm_model_path: str = "app/models/sensor_lgbm_model.txt"
    lightgbm_feature_columns_path: str = "app/models/sensor_feature_columns.json"

    # ── API ──────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "info"

    # ── Streamlit ────────────────────────────────────────────────────
    streamlit_port: int = 8501
    api_base_url: str = "http://api:8000"

    # ── Derived helpers ──────────────────────────────────────────────

    @property
    def postgres_dsn(self) -> str:
        from urllib.parse import quote_plus
        pw = quote_plus(self.postgres_password)
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{pw}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


    @property
    def psycopg2_dsn(self) -> str:
        from urllib.parse import quote_plus
        pw = quote_plus(self.postgres_password)
        return (
            f"postgresql://{self.postgres_user}:{pw}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
