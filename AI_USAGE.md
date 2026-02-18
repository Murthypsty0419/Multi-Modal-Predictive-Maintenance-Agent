## How AI Was Used in This Project

This file documents how AI tools and LLMs were used to build, debug, and optimize the Multi-Modal Predictive Maintenance Agent. The focus is on practical engineering value and transparency.

---


### 1. Where AI Helped

- **Gemini 3 Flash** served as a technical assistant in several areas:
  - **Supabase Schema Debugging:** Used Gemini to double-check the schema for the `vec_manuals` table, ensuring compatibility with pgvector and efficient chunk retrieval.
  - **Fusion Weights:** Sought advice on initial weights for the late fusion node, then adjusted based on test results and domain knowledge.
  - **LangGraph State Management:** Used Gemini to help design context propagation (manual_context, anomaly_query) through the pipeline, ensuring nothing was dropped between nodes.
  - **API/DB Alignment:** Cross-checked that API contracts matched the DB schema and frontend requirements.

---


### 2. Model Choices and Performance Tuning

- **Sensor Node:** LightGBM was selected for speed and explainability, with SHAP added for feature importance.
- **Text/Manual Context:** Manuals are stored in Supabase with pgvector. Retrieval and extraction are performed via external API calls.
- **Vision:** Image analysis is performed by an external vision API to avoid local VRAM bottlenecks and achieve faster, more accurate results.

- **Performance:**
  - Gemini suggested offloading vision and text/manual analysis to external APIs to meet latency targets. These changes were validated with real logs and user feedback before adoption.

---


### 3. How AI Outputs Were Checked

- **Manual Cross-Referencing:**
  - AI-generated troubleshooting steps and warranty info were checked against the original PDF manuals.
  - Fusion node outputs (risk scores, explanations) were compared to known failure cases in historical logs.

- **Explainability:**
  - SHAP values and XAI narratives were reviewed to ensure the system’s recommendations were clear and actionable.

---


### 4. Documentation and Audit Trail

- Every AI-generated code suggestion, schema tweak, or optimization was reviewed and edited before merging.
- A clear record of all major design and optimization decisions is kept here and in commit messages.

---


## Bottom Line

AI tools like Gemini 3 Flash were useful for brainstorming, validation, and speeding up research, but all critical engineering and design decisions were made with careful review. AI was a tool, not the architect.

For more details, see the main [README.md](./README.md) and the codebase.
