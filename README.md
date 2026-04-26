# Health Trust

Health Trust is a full-stack agentic healthcare intelligence system built for the Databricks AI Hackathon. It processes around 10,000 Indian healthcare facility records and turns them into searchable, scored, and explainable healthcare intelligence.

The system generates facility-level Trust Scores, detects contradictions in claimed capabilities, extracts evidence snippets from raw records, supports plain-English search, and provides detailed facility pages with confidence intervals and score breakdowns. It also aggregates facility assessments into district-level Desert Scores, allowing NGOs and planners to identify regions missing critical capabilities such as ICU, dialysis, oncology, surgery, and emergency care.

---

## What It Does

**For patients and families:** Search in plain English — "find a trustworthy ICU hospital near Delhi" — and get ranked results with trust scores, capability breakdowns, and explanations of why a facility is reliable, uncertain, or risky.

**For NGOs and planners:** A separate dashboard maps healthcare deserts at the district level, showing where critical capabilities are absent and which regions should be prioritised for intervention.

**For accountability:** Instead of showing unverified facility claims at face value, Health Trust explains the evidence behind every capability, detects internal contradictions, and cross-references top results against live public web sources.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM extraction | OpenAI `gpt-4o-mini` |
| Agent planner / validator | Anthropic `claude-haiku-4-5` |
| Vector search | ChromaDB + `all-MiniLM-L6-v2` embeddings |
| Web verification | Tavily Search API |
| API | FastAPI + Uvicorn |
| Data layer | Pandas + Parquet |
| Schemas | Pydantic v2 |
| Observability | MLflow 3 tracing (SQLite) |
| Frontend | Lovable / React |

---

## Architecture

### 1. LLM Extraction (`backend/extract_llm.py`)
Feeds each raw facility row to `gpt-4o-mini` with a domain-specific system prompt. The model maps messy specialties and descriptions into 16 standardised capabilities (ICU, surgery, dialysis, oncology, obstetrics, cardiology, emergency, anesthesia, pediatrics, mental health, dentistry, primary care, ophthalmology, orthopedics, dermatology, runs 24/7), assigns a status (confirmed / inferred / contradicted / unknown) to each, quotes the evidence text, and detects contradictions with severity scores 1–5.

Extraction is resumable, cost-monitored (pauses before $10), and handles validation failures with automatic retry. Total cost for 10,000 facilities: ~$0.64 using gpt-4o-mini.

### 2. Trust Scoring (`backend/trust_score.py`)
Deterministic Python scoring across four sub-scores:

```
Overall = (Internal Consistency × 0.35)
        + (Capability Plausibility × 0.30)
        + (Activity Signal × 0.15)
        + (Completeness × 0.20)
```

- **Internal Consistency** — starts at 100, deducts based on contradiction severity (severity × 5 points each)
- **Capability Plausibility** — checks whether raw data has prerequisites for each claimed capability (e.g. ICU requires ventilator in equipment, surgery requires anesthesia + operating theatre)
- **Activity Signal** — recency of update, social media presence, custom logo, follower count
- **Completeness** — percentage of 7 critical fields filled (phone, address, specialties, equipment, capability, doctors, capacity)

Confidence interval width narrows as completeness increases. Web verification adjusts it further at query time.

### 3. District Aggregation (`backend/districts.py`)
Groups all facilities by state and district. Produces a Desert Score per district:

```
Desert Score = 0.40 × (100 − avg_trust_score)
             + 0.30 × min(100, 20 × num_unverified_critical_caps)
             + 0.20 × (1 − facilities_per_100k_normalised) × 100
             + 0.10 × contradiction_density × 100
```

Higher score = worse served. Handles 137 messy raw state values normalised to 36 official Indian states/UTs.

### 4. Vector Store (`backend/vector_store.py`)
Builds a ChromaDB collection of 9,997 facility documents using `all-MiniLM-L6-v2` sentence embeddings. Stores trust score and active capabilities as metadata for post-retrieval filtering.

### 5. Agent Pipeline (`backend/agent.py`)
Five-stage reasoning pipeline on every `POST /query`:

1. **Planner** — Claude Haiku converts natural language into a structured query plan (intent, capability filters, location, sort order). Heuristic fallback if LLM unavailable.
2. **Retriever** — ChromaDB semantic search for facilities, or direct dataframe query for desert results.
3. **Validator** — Claude Haiku validates each candidate against the query intent. Intent-aware prompts: different criteria for find_facilities vs find_suspicious.
4. **Web Verifier** — Tavily Search cross-references top 3 results against live public web. Adjusts confidence intervals based on whether claimed capabilities appear online.
5. **Composer** — Returns ranked results with reasoning steps, confidence interval, and query plan.

MLflow 3 records every step as named spans: `trustmap_query → 1_planner → 2_retriever → 3_validator → 4_web_verification → 5_composer`.

### 6. Evidence Extraction (`backend/evidence.py`)
For each capability claim, retrieves the actual quoted text from the source parquet that supports or contradicts it. Powers the Evidence section on facility detail pages — judges and users see real cited text from the raw data, not just field names.

### 7. API (`backend/api.py`)
FastAPI with five endpoints:

| Endpoint | Description |
|---|---|
| `GET /` | Health check |
| `POST /query` | Natural language search |
| `GET /facility/{id}` | Full assessment + evidence snippets |
| `GET /districts` | District desert data (filterable by state, capability) |
| `GET /facility-pins` | Up to 500 map pins for the facility layer |

All data is loaded into memory at startup. District cache is pre-computed so every filter is a pure list comprehension.

---

## Data

- `data/facilities_clean.parquet` — 10,000 cleaned facility rows
- `backend/data/assessments_llm.parquet` — 10,017 LLM-extracted assessments (10,012 full-quality, 5 fallback)
- `backend/data/districts.parquet` — 2,364 district aggregates with desert scores
- `backend/data/chroma_db/` — ChromaDB vector index (9,997 documents)
- `backend/data/web_verifications.json` — pre-verified cache for 20 facilities
- `mlruns/mlflow.db` — MLflow trace store

---

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000

# View MLflow traces
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
```

Required environment variables in `.env`:
```
OPENAI_API_KEY=...
ANTHROPIC_AUTH_TOKEN=...
ANTHROPIC_BASE_URL=...
```

---

## Limitations

Health Trust is a hackathon prototype and should be used as a discovery and planning tool, not as a replacement for direct medical verification. Facility capabilities should always be confirmed with local providers before clinical or emergency decisions are made.

Some parts of the system are intentionally prototype-level: login is role selection rather than real authentication, population estimates default to 500,000 for districts where census data was unavailable, and web verification is applied to top search results rather than every facility. These limitations can be improved with production data sources, census-grade population data, and deeper provider verification.

The architecture is designed to be extensible. The current implementation uses a lightweight stack for rapid development, but the data pipeline maps naturally to Databricks-native components: Parquet files → Delta tables, ChromaDB → Mosaic AI Vector Search, local MLflow → MLflow on Databricks.
