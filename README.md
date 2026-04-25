# TrustMap India

TrustMap India is an agentic search engine for Indian healthcare facilities, built on a dataset of 10,000 facilities with messy, inconsistent records. Each facility is scored on a trust signal that estimates whether it actually offers the services and capabilities it claims. Users can search in plain English and get ranked, trust-aware results.

## Progress Brief

We scaffolded the project with `data/`, `backend/`, `frontend/`, and `demo/`, set up a Python virtual environment, and installed the core data/API/LLM dependencies. The raw Excel dataset was cleaned into `data/facilities_clean.parquet` with lowercase columns, parsed list fields, and 10,000 surviving facility rows. We also created `demo/planted_contradictions.json` with demo-worthy contradiction examples such as specialty sprawl, missing equipment, and type-specialty mismatches.

The backend now has Pydantic v2 contracts in `backend/schemas.py` for capability claims, contradictions, facility assessments, query plans, and query responses. A stub FastAPI app in `backend/api.py` serves realistic validated responses for `/`, `/query`, `/facility/{facility_id}`, `/districts`, and `/facility-pins`, with CORS enabled for frontend testing. The API was self-tested with FastAPI's test client and can run locally on `http://127.0.0.1:8000`.

For extraction, we added OpenAI setup support in `.env` and `backend/test_openai.py`, then built `backend/select_priority_rows.py` to split the dataset into `data/priority_rows.parquet` and `data/non_priority_rows.parquet`, exactly 5,000 rows each. The priority set includes all 5 demo facilities, 11 additional planted contradiction rows, 1,500 high-completeness rows, and 3,484 stratified random rows. One data-quality note for later: `address_stateOrRegion` has 137 unique values, so state names need normalization before building the desert map.

We also created `backend/extract_llm.py` for a 20-row OpenAI extraction test using `gpt-4o-mini`. The first run had 8/20 fallback rows; after tightening the prompt, enum mapping, scoring schema, max tokens, and retry instructions, fallback rows dropped to 4/20. The model is catching the right contradiction patterns when it validates, but the extractor still needs one more reliability pass before scaling to all 5,000 priority rows.
