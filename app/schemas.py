"""
Pydantic schemas for all modalities in the Oxmaint Predictive Agent.

Covers: Sensor input, Text/RAG input, Vision input, Agent state,
        Fusion output, API request/response contracts, and Inference logging.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Final XAI Fusion Output
# ---------------------------------------------------------------------------
class FinalReport(BaseModel):
    """Compact XAI report returned by the `/analyze` endpoint."""

    fused_score: float
    status_label: str
    explanation: str
    # High-signal drivers the UI can surface directly
    top_signals: list[str] = Field(default_factory=list)
    # Concrete recommended actions
    action_items: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Modality(str, Enum):
    SENSOR = "sensor"
    TEXT = "text"
    VISION = "vision"
    HISTORICAL = "historical"
    HISTORY = "history"


# ---------------------------------------------------------------------------
# Sensor Modality
# ---------------------------------------------------------------------------
class SensorReading(BaseModel):
    """Single time-point reading from the pump sensor CSV."""

    pump_id: str = Field(..., description="Unique asset identifier")
    temperature: float | None = None
    vibration: float | None = None
    pressure: float | None = None
    flow_rate: float | None = None
    rpm: float | None = None
    operational_hours: int = Field(..., ge=0, description="Total operational hours as integer to avoid floating-point drift")


class SensorPrediction(BaseModel):
    """LightGBM output for the sensor modality."""

    failure_probability: float = Field(..., ge=0, le=1)


# ---------------------------------------------------------------------------
# Text / RAG Modality
# ---------------------------------------------------------------------------
class TextChunk(BaseModel):
    """A single chunk stored in pgvector."""

    chunk_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    source_document: str
    section: str | None = None
    content: str
    embedding: list[float] | None = Field(default=None, exclude=True)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGQuery(BaseModel):
    """Query sent to the text/retrieval pipeline."""

    query_text: str
    pump_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class RAGResult(BaseModel):
    """Retrieval output from BGE-M3 + pgvector search."""

    chunks: list[TextChunk] = Field(default_factory=list)
    oem_limits_cited: list[str] = Field(
        default_factory=list,
        description="Specific OEM threshold references found in retrieved text",
    )
    confidence: float = Field(0.0, ge=0, le=1)


# ---------------------------------------------------------------------------
# Vision Modality
# ---------------------------------------------------------------------------
class VisionInput(BaseModel):
    """Image payload for Phi-4 Multimodal defect analysis."""

    image_path: str | None = Field(default=None, description="Local path to casting image")
    image_base64: str | None = Field(default=None, description="Base64-encoded image bytes")
    pump_id: str | None = None


class VisionResult(BaseModel):
    """Phi-4 Multimodal output describing physical defects."""

    # Specific booleans for orchestrator logic
    is_casting_present: bool = False
    has_corrosion: bool = False
    has_leaks: bool = False
    has_cracks: bool = False
    # Confidence scores for each finding (0.0 to 1.0, only meaningful if corresponding boolean is True)
    confidence_casting_present: float = 0.0
    confidence_corrosion: float = 0.0
    confidence_leaks: float = 0.0
    confidence_cracks: float = 0.0
    # Visual risk score (0.0 to 1.0)
    visual_risk_score: float = Field(0.0, ge=0, le=1)
    # Legacy/compat fields
    defects_detected: list[str] = Field(
        default_factory=list,
        description="e.g. ['crack', 'corrosion', 'pitting']",
    )
    description: str = Field("", description="Free-text description of observed damage")
    severity_score: float = Field(0.0, ge=0, le=1)

# ---------------------------------------------------------------------------
# History Modality Result
# ---------------------------------------------------------------------------
class HistoryResult(BaseModel):
    overdue_tasks: list[str] = Field(default_factory=list, description="List of overdue maintenance tasks")
    active_requests: list[dict] = Field(default_factory=list, description="List of active maintenance requests (dicts)")
    history_risk_impact: float = Field(0.0, ge=0, le=1, description="Risk impact from maintenance history")


# ---------------------------------------------------------------------------
# Historical Baseline (derived from CSV)
# ---------------------------------------------------------------------------
class HistoricalBaseline(BaseModel):
    """Statistical baseline for a pump derived from earlier operational hours."""

    pump_id: str
    window_hours: float = Field(..., description="Hours of data used for baseline")
    mean_vibration: float | None = None
    mean_temperature: float | None = None
    mean_pressure: float | None = None
    mean_flow_rate: float | None = None
    failure_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent / Orchestrator State  (LangGraph)
# ---------------------------------------------------------------------------
class AgentState(BaseModel):

    # Sensors that triggered historical outlier detection
    triggered_sensors: list[str] = Field(default_factory=list)
    """Typed state flowing through the LangGraph orchestrator."""

    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    pump_id: str
    current_total_hours: int = Field(..., description="Total lifetime running hours of the pump at request time")

    # Inputs — any may be None (graceful missing-modality handling)
    sensor_input: SensorReading | None = None
    text_query: str | None = None
    vision_input: VisionInput | None = None
    historical_baseline: HistoricalBaseline | None = None

    # Per-modality results
    rag_result: RAGResult | None = None
    vision_result: VisionResult | None = None
    history_result: HistoryResult | None = None

    # Primary witness: sensor risk 0.0–1.0 (set by sensor node regardless of history)
    sensor_risk_score: float = 0.0
    # Adaptive baseline: set by feature_node when historical CSV present; None when absent
    history_risk_score: float | None = None

    # Historical baseline TOON encoding
    historical_toon_string: str | None = Field(
        default=None,
        description="TOON-encoded historical baseline metrics for LLM consumption"
    )

    # Transactional service history TOON encoding
    transactional_toon_string: str | None = Field(
        default=None,
        description="TOON-encoded service history and maintenance requests for LLM consumption"
    )


    # Fusion
    fused_score: float | None = None  # renamed from fused_risk
    status_label: RiskLevel | None = None  # renamed from fused_risk_level
    modality_weights: dict[str, float] = Field(default_factory=dict)

    # XAI narrative
    chain_of_thought: str = ""
    explanation: str = ""

    # Metadata
    available_modalities: list[Modality] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Late-Fusion Configuration
# ---------------------------------------------------------------------------
class FusionWeights(BaseModel):
    """Configurable weights for late fusion across modalities."""

    sensor: float = Field(0.50, ge=0, le=1)
    text: float = Field(0.20, ge=0, le=1)
    vision: float = Field(0.20, ge=0, le=1)
    historical: float = Field(0.10, ge=0, le=1)


# ---------------------------------------------------------------------------
# API Request / Response Contracts
# ---------------------------------------------------------------------------
class PumpInferenceRequest(BaseModel):
    """POST /predict body — multi-modal pump analysis request."""

    pump_id: str
    current_total_hours: int = Field(..., description="Total lifetime running hours of the pump at request time")
    sensor_reading: SensorReading
    text_query: str | None = None
    image_base64: str | None = None
    image_path: str | None = None


class PumpInferenceResponse(BaseModel):
    """POST /predict response — the final JSON contract."""

    request_id: str
    pump_id: str
    risk_level: RiskLevel
    failure_probability: float = Field(..., ge=0, le=1)
    explanation: str = Field(
        ...,
        description=(
            "Machine-generated narrative synthesizing SHAP values, "
            "RAG context (OEM limits), and visual findings."
        ),
    )
    chain_of_thought: str
    modality_contributions: dict[str, float] = Field(default_factory=dict)
    sensor_details: SensorPrediction | None = None
    rag_details: RAGResult | None = None
    vision_details: VisionResult | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """GET /health response."""

    status: str = "ok"
    version: str = "0.1.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Inference Log (persisted to Postgres)
# ---------------------------------------------------------------------------
class InferenceLog(BaseModel):
    """Row written to the `inference_logs` table after every prediction."""

    log_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    request_id: str
    pump_id: str
    risk_level: RiskLevel
    failure_probability: float
    explanation: str
    chain_of_thought: str
    modality_contributions: dict[str, float] = Field(default_factory=dict)
    sensor_payload: dict[str, Any] | None = None
    rag_payload: dict[str, Any] | None = None
    vision_payload: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
