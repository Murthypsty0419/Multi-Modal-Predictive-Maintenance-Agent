# Multi Modal Predictive Maintenance Agent

## Overview

This is an end-to-end asset-health agent designed for industrial centrifugal pumps. The system leverages multi-modal data: sensor readings, maintenance logs, technical manuals, and pump images to deliver actionable, explainable health diagnostics and failure predictions.

---

## Problem Statement

Industrial pumps are critical assets in manufacturing and process industries. Unexpected failures lead to costly downtime and safety risks. This project aims to build a robust, explainable agent that predicts pump health and provides root-cause diagnostics by fusing:
- Real-time sensor data
- Maintenance/service history
- Technical manuals (retrieved via RAG)
- Visual inspection (image analysis)

---

## System Architecture

The core pipeline is orchestrated using a **LangGraph** linear flow, with each node responsible for a specific modality or fusion step:


```mermaid
flowchart LR
	A[API Request] --> SN[Sensor Node] --> SAN[Service Age Node]

	SAN --> C1((historical_logs?))
	C1 -->|yes| FN[Feature Node]
	C1 -->|no| C2((instruction_manual?))

	FN --> C2
	C2 -->|yes| MCN[Manual Context Node]
	C2 -->|no| C3((pump_image?))

	MCN --> C3
	C3 -->|yes| VN[Vision Node]
	C3 -->|no| FUSE[Fusion Node]

	VN --> FUSE
	FUSE --> R[Diagnostic Report]
```

---

## End-to-End Flow (Node Execution & Model Usage)

| Node | Condition to Run | Model(s) Used |
|------|-----------------|---------------|
| **sensor_node** | Always — pipeline entry point | LightGBM (local) + SHAP for explainability |
| **service_age_node** | Always, after sensor_node | None (logic-based overdue check) |
| **feature_node** | Always, after service_age_node; handles historical log analysis when `historical_logs` is provided | None (statistical outlier detection) |
| **manual_context_node** | Only if `instruction_manual` PDF is present | **BAAI/bge-large-en-v1.5** (local, SentenceTransformers) for embedding & retrieval; **llama-3.1-8b-instant** (Groq API) for structured JSON extraction |
| **vision_node** | Only if `pump_image` is present | **meta-llama/llama-4-scout-17b-16e-instruct** (Groq API) for multimodal image + text analysis |
| **fusion_node** | Always — pipeline exit point | None (weighted risk aggregation, logic only) |

### PDF Ingestion (Offline)
Technical manuals are parsed offline using **LlamaParse** (LlamaIndex API) into text chunks. Chunks are embedded with **BAAI/bge-large-en-v1.5** and stored in Supabase with **pgvector** for later retrieval by the manual context node.

---

### Node Descriptions

**sensor_node**: Runs a **LightGBM** model on real-time sensor data (temperature, vibration, pressure, flow, rpm) and uses **SHAP** for feature importance. Outputs a risk score and anomaly notes.

**service_age_node**: Checks if the pump is overdue for scheduled maintenance based on operational hours and service logs. No ML model — logic-based check only. Flags overdue assets and appends to the anomaly query.

**feature_node**: Detects historical outliers by comparing current sensor readings to baseline statistics from historical logs. No ML model — statistical comparison only.

**manual_context_node**: Embeds the anomaly query using **BAAI/bge-large-en-v1.5** (local, SentenceTransformers) and retrieves the top-k relevant chunks from Supabase pgvector. The retrieved text is then passed to **llama-3.1-8b-instant** via the Groq API to extract structured diagnostic JSON (sensor thresholds, fault causes, warranty info).

**vision_node**: Sends the pump image and manual context to **meta-llama/llama-4-scout-17b-16e-instruct** via the Groq API for multimodal analysis. Detects leaks, corrosion, cracks, and other visual faults. Appends findings to the anomaly query.

**fusion_node**: Combines all modality risk scores using configurable weights. No ML model — generates a final risk score, status label (low/medium/high/critical), top signals, action items, and a human-readable explanation.

---

## Installation & Usage

### Prerequisites
- Docker & Docker Compose
- Python 3.12+ (for local development)
- Supabase/Postgres instance (for manual/document storage)

### Quickstart (Docker)
```bash
git clone https://github.com/Murthypsty0419/Multi-Modal-Predictive-Maintenance-Agent
cd pump-health
docker-compose up --build
```
The API will be available at `http://localhost:8000` and the Streamlit frontend at `http://localhost:8501`.

### Manual Setup (Dev)
1. Create a Python virtual environment and install dependencies:
	```bash
	python -m venv .venv
	source .venv/bin/activate
	pip install -r requirements.txt
	```
2. Set up environment variables (see `.env.example`).
3. Start the API:
	```bash
	uvicorn app.api.routes:app --reload
	```
4. Start the frontend:
	```bash
	streamlit run frontend/app.py
	```

---

## Evaluation & Optimization

### Latency Bottleneck Analysis
- **Initial Bottleneck**: Early experiments with local large language models caused high latency (baseline inference ~45,369 ms).
- **Optimization**: Vision and text/manual analysis offloaded to external APIs, reducing local VRAM load and improving speed.

### Performance Metrics
- **p50 Latency**: ~42s
- **p95 Latency**: ~45.4s (cold start)
- **Throughput**: 1.32 pumps/minute

---

## Deliverables

- Full pipeline code (API, orchestrator, nodes, frontend)
- Dockerized deployment (docker-compose)
- Supabase schema for manual storage
- Model weights and config for LightGBM
- Example manuals and sensor logs
- Streamlit dashboard for interactive analysis
- Mermaid.js architecture diagram (see above)

---

## Production Deployment (Hosted Service)

To deploy this project as a hosted service (e.g., on a cloud VM, VPS, or managed container platform):

### 1. Prepare Your Environment
- Provision a Linux VM or container host (Ubuntu 22.04+ recommended).
- Install Docker and Docker Compose (or use Docker Compose V2 plugin).
- Set up DNS and firewall rules to expose ports 80/443 (for frontend) and 8000 (API) as needed.

### 2. Configure Environment Variables
- Copy `.env.example` to `.env` and fill in all required secrets and connection details (database, Supabase, API keys, etc).
- For production, use strong passwords and secure API keys.

### 3. Build and Start the Stack
```bash
# Clone the repository
git clone https://github.com/Murthypsty0419/Multi-Modal-Predictive-Maintenance-Agent
cd pump-health

# Build and start all services (API, frontend, database)
docker compose up --build -d
```
- The API will be available at `http://<your-server-ip>:8000`
- The Streamlit frontend will be available at `http://<your-server-ip>:8501`

### 4. Database Persistence & Backups
- The PostgreSQL/pgvector database uses a Docker volume (`pgdata`) for persistent storage.
- Set up regular backups of this volume for disaster recovery.

### 5. Monitoring & Logs
- Use `docker compose logs -f` to monitor service logs.
- Consider integrating with cloud monitoring or log aggregation tools for production.

---

## Documentation

- See [AI_USAGE.md](./AI_USAGE.md) for details on how Gemini 3 Flash was used for architecture, debugging, and optimization.
- For API usage and schema, see [app/schemas.py](./app/schemas.py).
