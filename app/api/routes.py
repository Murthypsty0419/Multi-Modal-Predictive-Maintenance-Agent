"""
FastAPI application — prediction, analysis & health endpoints.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.fusion import fuse_and_explain
from app.db.models import InferenceLogRow
from app.db.session import get_session
from app.schemas import (
    AgentState,
    HealthResponse,
    Modality,
    PumpInferenceRequest,
    PumpInferenceResponse,
    SensorReading,
    VisionInput,
)

from fastapi import APIRouter
from supabase import create_client, Client
import os
from fastapi import HTTPException
import polars as pl
from app import config



from dotenv import load_dotenv
load_dotenv()
router = APIRouter()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY")

@router.get("/test-supabase")
def test_supabase():
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        # Try to list tables or select from a known table
        result = supabase.table("inference_logs").select("*").limit(1).execute()
        return {"success": True, "data": result.data}
    except Exception as e:
        return {"success": False, "error": str(e)}
    

logger = logging.getLogger("oxmaint.api")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def _ensure_dirs() -> None:
    """Ensure data sub-directories exist."""
    for sub in ("manuals", "historical_logs", "transactional", "images"):
        (DATA_DIR / sub).mkdir(parents=True, exist_ok=True)


async def _save_uploaded_file(upload: UploadFile | None) -> str | None:
    """Persist an uploaded file to the appropriate data sub-directory.

    Returns the absolute path on disk, or None if no file was provided.
    """
    if upload is None:
        return None

    _ensure_dirs()
    filename = upload.filename or ""

    if filename.endswith("_manual.pdf"):
        target_dir = DATA_DIR / "manuals"
    elif filename.endswith("_history.csv"):
        target_dir = DATA_DIR / "historical_logs"
    elif filename in {"work_done_logs.csv", "service_schedules.csv", "maintenance_requests.csv"}:
        target_dir = DATA_DIR / "transactional"
    else:
        # Fallback: treat as image if possible, otherwise drop into /images
        target_dir = DATA_DIR / "images"

    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename

    contents = await upload.read()
    target_path.write_bytes(contents)
    logger.info("Saved upload %s → %s", filename, target_path)
    return str(target_path)


# ---------------------------------------------------------------------------
# Lifespan — run migrations on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Oxmaint Predictive Agent starting …")
    yield
    logger.info("Shutting down.")


app = FastAPI(

    title="Oxmaint Predictive Agent",
    version="0.1.0",
    lifespan=lifespan,
)

# Register the router for /test-supabase and other endpoints
app.include_router(router)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse()


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PumpInferenceResponse)
async def predict(
    body: PumpInferenceRequest,
    session: AsyncSession = Depends(get_session),
):
    from app.core.orchestrator import run_agent  # deferred to avoid circular

    # Determine available modalities
    modalities: list[Modality] = []
    if body.sensor_reading:
        modalities.append(Modality.SENSOR)
    if body.text_query:
        modalities.append(Modality.TEXT)
    if body.image_base64 or body.image_path:
        modalities.append(Modality.VISION)

    # Build initial agent state
    vision_input = None
    if body.image_base64 or body.image_path:
        vision_input = VisionInput(
            image_base64=body.image_base64,
            image_path=body.image_path,
            pump_id=body.pump_id,
        )

    state = AgentState(
        pump_id=body.pump_id,
        current_total_hours=body.current_total_hours,
        sensor_input=body.sensor_reading,
        text_query=body.text_query,
        vision_input=vision_input,
        available_modalities=modalities,
    )

    # Run LangGraph agent
    result: AgentState = await run_agent(state)

    # Persist inference log
    log_row = InferenceLogRow(
        log_id=result.request_id,
        request_id=result.request_id,
        pump_id=result.pump_id,
        risk_level=result.fused_risk_level.value if result.fused_risk_level else "low",
        failure_probability=result.fused_risk or 0.0,
        explanation=result.explanation,
        chain_of_thought=result.chain_of_thought,
        modality_contributions=result.modality_weights,
        sensor_payload=result.sensor_result.model_dump() if result.sensor_result else None,
        rag_payload=result.rag_result.model_dump() if result.rag_result else None,
        vision_payload=result.vision_result.model_dump() if result.vision_result else None,
    )
    session.add(log_row)
    try:
        await session.commit()
    except Exception:
        logger.exception("Failed to persist inference log; returning prediction response anyway.")
        await session.rollback()

    return PumpInferenceResponse(
        request_id=result.request_id,
        pump_id=result.pump_id,
        risk_level=result.fused_risk_level or "low",
        failure_probability=result.fused_risk or 0.0,
        explanation=result.explanation,
        chain_of_thought=result.chain_of_thought,
        modality_contributions=result.modality_weights,
        sensor_details=result.sensor_result,
        rag_details=result.rag_result,
        vision_details=result.vision_result,
    )


# ---------------------------------------------------------------------------
# POST /analyze  (multipart dashboard entrypoint)
# ---------------------------------------------------------------------------
@app.post("/analyze")
async def analyze(
    asset_id: str = Form(..., alias="asset_id"),
    current_total_hours: int = Form(..., alias="current_total_hours"),
    sensors_json: str = Form(..., alias="sensors"),
    instruction_manual: UploadFile | None = File(default=None),
    historical_logs: UploadFile | None = File(default=None),
    work_done_logs: UploadFile | None = File(default=None),
    service_schedules: UploadFile | None = File(default=None),
    maintenance_requests: UploadFile | None = File(default=None),
    pump_image: UploadFile | None = File(default=None),
) -> Any:
    """Dashboard entrypoint: accept files + live sensor JSON and return XAI report.

    - Saves uploads into data/manuals, data/historical_logs, data/transactional, data/images
      based on the client-side file naming convention.
    - Injects current_total_hours and parsed sensor JSON into the AgentState.
    - Runs the orchestrator graph and returns the compact FinalReport JSON.
    """
    from app.core.orchestrator import run_agent  # local import to avoid circular

    logger.info(
        "[analyze] request start: asset_id=%s, current_total_hours=%s, has_image=%s",
        asset_id,
        current_total_hours,
        pump_image is not None,
    )


    # Persist files to disk
    manual_path = await _save_uploaded_file(instruction_manual)
    hist_path = await _save_uploaded_file(historical_logs)
    wdl_path = await _save_uploaded_file(work_done_logs)
    ss_path = await _save_uploaded_file(service_schedules)
    mr_path = await _save_uploaded_file(maintenance_requests)

    # --- Run transactional CSV ingestion if any uploaded ---
    if wdl_path or ss_path or mr_path:
        try:
            supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
            # Clear tables
            supabase.table("work_done_logs").delete().neq("id", 0).execute()
            supabase.table("service_schedules").delete().neq("id", 0).execute()
            supabase.table("maintenance_requests").delete().neq("id", 0).execute()
            import polars as pl
            def check_columns(df, required, fname):
                missing = [col for col in required if col not in df.columns]
                if missing:
                    msg = f"[analyze] ERROR: {fname} missing columns: {missing}"
                    print(msg)
                    logger.error(msg)
                    return False
                return True
            def upload_csv_polars(table, file_path, required_cols, fname):
                if file_path:
                    df = pl.read_csv(file_path)
                    if not check_columns(df, required_cols, fname):
                        return
                    records = df.to_dicts()
                    for row in records:
                        supabase.table(table).insert(row).execute()
            upload_csv_polars("work_done_logs", wdl_path, ["pump_id", "task_name", "hours_at_service", "timestamp"], "work_done_logs.csv")
            upload_csv_polars("service_schedules", ss_path, ["pump_id", "task_name", "interval_hours", "priority"], "service_schedules.csv")
            upload_csv_polars("maintenance_requests", mr_path, ["pump_id", "description", "priority", "status", "created_at"], "maintenance_requests.csv")
            logger.info(f"[analyze] Transactional CSVs ingested and uploaded to Supabase.")
            print("!!! TRANSACTIONAL CSVs INGESTED & UPLOADED !!!")
        except Exception as exc:
            logger.error(f"[analyze] Transactional CSV ingestion failed: {exc}")
    image_path = await _save_uploaded_file(pump_image)

    # --- Run ingestion pipeline for manual if uploaded ---
    if manual_path:
        try:
            from scripts.ingestion import parse_pdf_manual, embed_chunks_bge, load_bge_embedding_model, store_chunks_supabase
            import os
            api_key = os.environ.get("LLAMA_CLOUD_API_KEY") or getattr(config, "LLAMA_CLOUD_API_KEY", None)
            chunks = parse_pdf_manual(manual_path, api_key)
            model = load_bge_embedding_model()
            records = embed_chunks_bge(chunks, model)
            await store_chunks_supabase(records)
            logger.info(f"[analyze] Manual ingested and uploaded to Supabase: {manual_path}")
        except Exception as exc:
            logger.error(f"[analyze] Manual ingestion failed: {exc}")

    # Parse sensor JSON
    try:
        sensor_payload = json.loads(sensors_json)
    except json.JSONDecodeError:
        logger.exception("Invalid sensors JSON payload: %s", sensors_json)
        raise


    # Map live sensor fields into SensorReading
    sensor_input = SensorReading(
        pump_id=asset_id,
        operational_hours=current_total_hours,  # Keep as int to avoid floating-point drift
        temperature=sensor_payload.get("temperature"),
        vibration=sensor_payload.get("vibration"),
        rpm=sensor_payload.get("rpm"),
        pressure=sensor_payload.get("pressure"),
        flow_rate=sensor_payload.get("flow_rate"),
    )

    # Determine available modalities
    modalities: list[Modality] = [Modality.SENSOR]
    if pump_image is not None:
        modalities.append(Modality.VISION)
    # Only add Modality.HISTORICAL, never Modality.HISTORY
    if Modality.HISTORY in modalities:
        modalities.remove(Modality.HISTORY)
    modalities.append(Modality.HISTORICAL)

    vision_input = None
    if image_path:
        vision_input = VisionInput(image_path=image_path, pump_id=asset_id)

    state = AgentState(
        pump_id=asset_id,
        current_total_hours=current_total_hours,
        sensor_input=sensor_input,
        vision_input=vision_input,
        available_modalities=modalities,
    )
    # Add historical_logs_path to state dict for orchestrator nodes
    state_dict = state.model_dump()
    if hist_path:
        state_dict["historical_logs_path"] = hist_path

    logger.info(
        "[analyze] modalities=%s, sensor_input keys=%s",
        [m.value for m in modalities],
        list(sensor_payload.keys()),
    )

    # Run orchestrator graph
    result_state: AgentState = await run_agent(AgentState(**state_dict))

    # If fusion_node already ran, return its output directly
    if (
        result_state.fused_score is not None
        and result_state.explanation
        and hasattr(result_state, "top_signals")
        and hasattr(result_state, "action_items")
    ):
        logger.info(
            "[analyze] done: fused_score=%.4f, status_label=%s (from graph)",
            result_state.fused_score,
            getattr(result_state, "status_label", ""),
        )
        return {
            "fused_score": result_state.fused_score,
            "status_label": getattr(result_state, "status_label", ""),
            "explanation": result_state.explanation,
            "top_signals": getattr(result_state, "top_signals", []),
            "action_items": getattr(result_state, "action_items", []),
        }

    # Otherwise, derive compact XAI report (legacy fallback)
    report = await fuse_and_explain(result_state)
    logger.info(
        "[analyze] done: fused_score=%.4f, status_label=%s (fallback)",
        report.get("fused_score", 0),
        report.get("status_label", ""),
    )
    return report


# ---------------------------------------------------------------------------
# POST /ingest endpoints for manual and transactional data uploads
# ---------------------------------------------------------------------------


@router.post("/ingest-manual")
async def ingest_manual(
    instruction_manual: UploadFile = File(...),
):
    """Ingest a PDF manual, parse, embed, and store in Supabase."""
    from scripts.ingestion import parse_pdf_manual, embed_chunks_bge, load_bge_embedding_model, store_chunks_supabase
    import tempfile
    api_key = os.environ.get("LLAMA_CLOUD_API_KEY") or getattr(config, "LLAMA_CLOUD_API_KEY", None)
    if not api_key:
        raise HTTPException(status_code=500, detail="LLAMA_CLOUD_API_KEY not set")
    # Save uploaded PDF to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await instruction_manual.read())
        tmp_path = tmp.name
    # Parse and embed
    chunks = parse_pdf_manual(tmp_path, api_key)
    model = load_bge_embedding_model()
    records = embed_chunks_bge(chunks, model)
    await store_chunks_supabase(records)
    return {"status": "success", "chunks": len(records)}


@router.post("/upload-transactional")
async def upload_transactional(
    work_done_logs: UploadFile = File(...),
    service_schedules: UploadFile = File(...),
    maintenance_requests: UploadFile = File(...),
):
    """Clear and upload transactional CSVs to Supabase."""
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    # Clear tables
    supabase.table("work_done_logs").delete().neq("id", 0).execute()
    supabase.table("service_schedules").delete().neq("id", 0).execute()
    supabase.table("maintenance_requests").delete().neq("id", 0).execute()
    # Helper to upload CSV using Polars
    def upload_csv_polars(table, file):
        file.file.seek(0)
        df = pl.read_csv(file.file)
        records = df.to_dicts()
        for row in records:
            supabase.table(table).insert(row).execute()
    upload_csv_polars("work_done_logs", work_done_logs)
    upload_csv_polars("service_schedules", service_schedules)
    upload_csv_polars("maintenance_requests", maintenance_requests)
    return {"status": "success"}