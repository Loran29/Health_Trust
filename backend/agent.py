from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import openai
from openai import AsyncOpenAI
import chromadb
import pandas as pd
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from dotenv import load_dotenv

try:
    import mlflow
    import warnings as _w
    _w.filterwarnings("ignore", category=FutureWarning, module="mlflow")
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from backend.trust_score import (
        activity_signal_score,
        capability_plausibility_score,
        completeness_score,
        compute_overall,
        confidence_interval,
        internal_consistency_score,
    )
    from backend.districts import normalize_state
except ImportError:
    from trust_score import (  # type: ignore[no-redef]
        activity_signal_score,
        capability_plausibility_score,
        completeness_score,
        compute_overall,
        confidence_interval,
        internal_consistency_score,
    )
    from districts import normalize_state  # type: ignore[no-redef]

# Optional — Tavily web verification. Agent works fine without it.
try:
    from backend.tavily_validator import verify_facility as _tavily_verify
    _TAVILY_AVAILABLE = True
except ImportError:
    try:
        from tavily_validator import verify_facility as _tavily_verify  # type: ignore[no-redef]
        _TAVILY_AVAILABLE = True
    except ImportError:
        _TAVILY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

# ---------------------------------------------------------------------------
# MLflow tracing setup (optional — agent works fine without it)
# ---------------------------------------------------------------------------

if MLFLOW_AVAILABLE:
    try:
        mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
        mlflow.set_experiment("trustmap_india")
    except Exception:
        MLFLOW_AVAILABLE = False


class _MlSpan:
    """Fault-tolerant MLflow span wrapper — never raises, never blocks the pipeline."""
    __slots__ = ("_cm", "_span")

    def __init__(self, name: str) -> None:
        self._cm: Any = None
        self._span: Any = None
        if MLFLOW_AVAILABLE:
            try:
                self._cm = mlflow.start_span(name=name)
                self._span = self._cm.__enter__()
            except Exception:
                self._cm = None

    def log(self, **attrs: Any) -> None:
        if self._span is None:
            return
        try:
            for k, v in attrs.items():
                self._span.set_attribute(k, v)
        except Exception:
            pass

    def close(self) -> None:
        if self._cm is not None:
            try:
                self._cm.__exit__(None, None, None)
            except Exception:
                pass
            self._cm = None


ASSESSMENTS_PATH = Path("backend/data/assessments_llm.parquet")
FACILITIES_PATH = Path("data/facilities_clean.parquet")
DISTRICTS_PATH = Path("backend/data/districts.parquet")
CHROMA_PATH = Path("backend/data/chroma_db")
COLLECTION_NAME = "health_trust_facilities"

# Use Haiku for speed and low cost; same proxy Claude Code uses
LLM_MODEL = "gpt-4o-mini"

ALL_CAPABILITIES = [
    "emergency", "icu", "surgery", "obstetrics", "dialysis", "oncology",
    "cardiology", "anesthesia", "pediatrics", "mental_health", "dentistry",
    "primary_care", "ophthalmology", "orthopedics", "dermatology", "runs_24_7",
]

PLANNER_SYSTEM = (
    "You parse healthcare facility search queries for TrustMap India. "
    "Return ONLY a single valid JSON object — no markdown, no explanation.\n\n"
    'Schema: {"location_filters": {"state": string|null, "district": string|null, '
    '"region_type": "rural"|"urban"|null}, '
    '"capability_filters": [<list from: '
    + ", ".join(ALL_CAPABILITIES)
    + ">], "
    '"trust_filter": {"min_score": int|null, "max_score": int|null}, '
    '"sort_by": "trust_score_asc"|"trust_score_desc"|"relevance", '
    '"intent": "find_facilities"|"find_deserts"|"find_suspicious", '
    '"reasoning": "<one sentence>"}\n\n'
    "Rules:\n"
    "- find_suspicious: suspicious/fake/too good/contradictions -> sort trust_score_asc\n"
    "- find_deserts: desert/worst/lacking/gaps/no access -> districts not facilities\n"
    "- find_facilities: default, reliable/good facilities -> sort trust_score_desc\n"
    "Clinical mappings: C-section->obstetrics+surgery; kidney/renal->dialysis; "
    "cancer->oncology; heart->cardiology; dental/dentist->dentistry; "
    "children/paeds->pediatrics; ICU->icu; maternity->obstetrics."
)

# ---------------------------------------------------------------------------
# Heuristic planner (fallback when LLM is unavailable)
# ---------------------------------------------------------------------------

_STATE_KEYS = {
    "maharashtra": "Maharashtra", "karnataka": "Karnataka", "delhi": "Delhi",
    "gujarat": "Gujarat", "rajasthan": "Rajasthan", "uttar pradesh": "Uttar Pradesh",
    "west bengal": "West Bengal", "tamil nadu": "Tamil Nadu",
    "andhra pradesh": "Andhra Pradesh", "telangana": "Telangana",
    "kerala": "Kerala", "bihar": "Bihar", "madhya pradesh": "Madhya Pradesh",
    "chhattisgarh": "Chhattisgarh", "jharkhand": "Jharkhand", "assam": "Assam",
    "punjab": "Punjab", "haryana": "Haryana", "himachal pradesh": "Himachal Pradesh",
    "uttarakhand": "Uttarakhand", "odisha": "Odisha", "goa": "Goa",
    "manipur": "Manipur", "meghalaya": "Meghalaya", "mizoram": "Mizoram",
    "nagaland": "Nagaland", "sikkim": "Sikkim", "tripura": "Tripura",
    "arunachal pradesh": "Arunachal Pradesh", "jammu": "Jammu & Kashmir",
    "kashmir": "Jammu & Kashmir", "ladakh": "Ladakh", "chandigarh": "Chandigarh",
    "puducherry": "Puducherry", "pondicherry": "Puducherry",
}

_CAP_KEYS: list[tuple[str, str]] = [
    ("dialysis", "dialysis"), ("kidney", "dialysis"), ("renal", "dialysis"),
    ("emergency", "emergency"), ("accident", "emergency"), ("trauma", "emergency"),
    ("c-section", "obstetrics"), ("caesarean", "obstetrics"), ("maternity", "obstetrics"),
    ("obstetric", "obstetrics"), ("gynae", "obstetrics"), ("gynecolog", "obstetrics"),
    ("surgery", "surgery"), ("surgical", "surgery"), ("operation", "surgery"),
    ("icu", "icu"), ("intensive care", "icu"), ("critical care", "icu"),
    ("cancer", "oncology"), ("oncolog", "oncology"), ("chemotherapy", "oncology"),
    ("cardiac", "cardiology"), ("cardiology", "cardiology"), ("heart", "cardiology"),
    ("dental", "dentistry"), ("dentist", "dentistry"), ("dentistry", "dentistry"),
    ("pediatric", "pediatrics"), ("children", "pediatrics"), ("child hospital", "pediatrics"),
    ("mental health", "mental_health"), ("psychiatr", "mental_health"),
    ("eye", "ophthalmology"), ("ophthal", "ophthalmology"), ("vision", "ophthalmology"),
    ("orthop", "orthopedics"), ("bone", "orthopedics"), ("fracture", "orthopedics"),
    ("dermatol", "dermatology"), ("skin", "dermatology"),
    ("primary care", "primary_care"), ("general practitioner", "primary_care"),
    ("anesthesia", "anesthesia"), ("anaesthesia", "anesthesia"),
    ("24/7", "runs_24_7"), ("24 hour", "runs_24_7"), ("round the clock", "runs_24_7"),
]


def _heuristic_plan(query: str) -> dict:
    q = query.lower()

    # Intent
    if any(k in q for k in ("suspicious", "fake", "too good", "contradict", "untrustworthy", "shady", "dubious")):
        intent, sort_by = "find_suspicious", "trust_score_asc"
    elif any(k in q for k in ("desert", "worst", "lacking", "gaps", "no access", "underserved", "scarce", "missing")):
        intent, sort_by = "find_deserts", "relevance"
    else:
        intent, sort_by = "find_facilities", "trust_score_desc"

    # State
    state = None
    for key, canonical in _STATE_KEYS.items():
        if key in q:
            state = canonical
            break

    # Region
    region_type = "rural" if "rural" in q else ("urban" if "urban" in q or " city" in q else None)

    # Capabilities (deduplicated)
    caps: list[str] = []
    for kw, cap in _CAP_KEYS:
        if kw in q and cap not in caps:
            caps.append(cap)

    return {
        "location_filters": {"state": state, "district": None, "region_type": region_type},
        "capability_filters": caps,
        "trust_filter": {"min_score": None, "max_score": None},
        "sort_by": sort_by,
        "intent": intent,
        "reasoning": f"Heuristic: {intent} | state={state} | caps={caps}",
        "_source": "heuristic",
    }


# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------

_llm: AsyncOpenAI | None = None
_chroma_coll: chromadb.Collection | None = None
_merged_df: pd.DataFrame | None = None
_districts_df: pd.DataFrame | None = None

# Web verification cache — loaded once from the pre-verified JSON
_WEB_VERIFICATIONS_PATH = Path("backend/data/web_verifications.json")
_web_cache: dict[tuple[str, str], dict] = {}


def _load_web_cache() -> None:
    global _web_cache
    try:
        if _WEB_VERIFICATIONS_PATH.exists():
            with open(_WEB_VERIFICATIONS_PATH, encoding="utf-8") as f:
                records = json.load(f)
            _web_cache = {
                (r["facility_name"].lower(), r["city"].lower()): r
                for r in records
            }
    except Exception:
        pass


_load_web_cache()


def _get_llm() -> AsyncOpenAI:
    global _llm
    if _llm is None:
        _llm = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    return _llm


def _build_chroma(client: chromadb.ClientAPI) -> chromadb.Collection:
    try:
        from backend.vector_store import build_collection
    except ImportError:
        from vector_store import build_collection  # type: ignore[no-redef]

    print("[agent] ChromaDB collection missing — rebuilding from parquet (~60s on first boot)...")
    assessments = pd.read_parquet(ASSESSMENTS_PATH)
    facilities = pd.read_parquet(FACILITIES_PATH)
    facilities["facility_id"] = facilities.apply(
        lambda r: _make_facility_id(r.get("name"), r.get("address_city")), axis=1
    )
    merged = assessments.merge(
        facilities, on="facility_id", how="inner", suffixes=("_asmt", "_raw")
    )
    print(f"[agent] Indexing {len(merged):,} records into ChromaDB...")
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return build_collection(client, merged)


def _get_chroma() -> chromadb.Collection:
    global _chroma_coll
    if _chroma_coll is None:
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        ef = DefaultEmbeddingFunction()
        try:
            coll = client.get_collection(COLLECTION_NAME, embedding_function=ef)
            if coll.count() == 0:
                coll = _build_chroma(client)
        except Exception:
            coll = _build_chroma(client)
        _chroma_coll = coll
    return _chroma_coll


def _reset_chroma() -> chromadb.Collection:
    global _chroma_coll
    _chroma_coll = None
    return _get_chroma()


def _make_facility_id(name: Any, city: Any) -> str:
    n = "" if not name or str(name).lower() in {"nan", "none"} else str(name)
    c = "" if not city or str(city).lower() in {"nan", "none"} else str(city)
    return re.sub(r"[^a-z0-9]+", "-", f"{n} {c}".lower()).strip("-")


def _get_merged() -> pd.DataFrame:
    global _merged_df
    if _merged_df is None:
        assessments = pd.read_parquet(ASSESSMENTS_PATH)
        facilities = pd.read_parquet(FACILITIES_PATH)
        facilities["facility_id"] = facilities.apply(
            lambda r: _make_facility_id(r.get("name"), r.get("address_city")), axis=1
        )
        _merged_df = (
            assessments.merge(facilities, on="facility_id", how="inner", suffixes=("_asmt", "_raw"))
            .set_index("facility_id", drop=False)
        )
    return _merged_df


def _get_districts() -> pd.DataFrame:
    global _districts_df
    if _districts_df is None:
        _districts_df = pd.read_parquet(DISTRICTS_PATH)
    return _districts_df


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _parse_json_list(raw: Any) -> list:
    try:
        items = json.loads(raw) if isinstance(raw, str) else raw
        return items if isinstance(items, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _active_caps(capability_claims_json: Any) -> list[str]:
    items = _parse_json_list(capability_claims_json)
    return [c["capability"] for c in items if c.get("status") in {"confirmed", "inferred"}]


def _score_and_ci(row: pd.Series) -> tuple[float, list[float]]:
    row_dict = row.to_dict()
    subs = {
        "internal_consistency": internal_consistency_score(row_dict),
        "capability_plausibility": capability_plausibility_score(row_dict, row),
        "activity_signal": activity_signal_score(row),
        "completeness": completeness_score(row),
    }
    overall = round(compute_overall(subs), 1)
    lo, hi = confidence_interval(subs)
    return overall, [round(lo, 1), round(hi, 1)]


def _matches_state(meta: dict, plan_state: str | None) -> bool:
    if not plan_state:
        return True
    norm_plan = normalize_state(plan_state)
    if norm_plan == "Unknown":
        ps = plan_state.lower()
        return ps in meta.get("state", "").lower() or ps in meta.get("district", "").lower()
    return (
        normalize_state(meta.get("state", "")) == norm_plan
        or normalize_state(meta.get("district", "")) == norm_plan
    )


def _matches_caps(meta: dict, doc: str, cap_filters: list[str]) -> bool:
    if not cap_filters:
        return True
    stored = {c.strip().lower() for c in meta.get("capabilities", "").split(",") if c.strip()}
    dl = doc.lower()
    return any(cf.lower() in stored or cf.lower() in dl for cf in cap_filters)


def _clean_plan(raw: dict) -> dict:
    """Ensure no None values slip through for required fields."""
    loc = raw.get("location_filters") or {}
    return {
        "location_filters": {
            "state": loc.get("state") or None,
            "district": loc.get("district") or None,
            "region_type": loc.get("region_type") or None,
        },
        "capability_filters": raw.get("capability_filters") or [],
        "trust_filter": raw.get("trust_filter") or {},
        "sort_by": raw.get("sort_by") or "relevance",
        "intent": raw.get("intent") or "find_facilities",
        "reasoning": raw.get("reasoning") or "",
        "_source": raw.get("_source", "llm"),
    }


# ---------------------------------------------------------------------------
# Step 1 — Planner
# ---------------------------------------------------------------------------

async def _plan(query: str) -> dict:
    try:
        msg = await _get_llm().chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": query},
            ],
            max_tokens=400,
        )
        raw_text = msg.choices[0].message.content.strip()
        # Strip ```json ... ``` if present
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        raw_text = match.group(1) if match else raw_text
        plan = json.loads(raw_text)
        return _clean_plan(plan)
    except (openai.APIError, openai.APIConnectionError) as exc:
        print(f"  [LLM unavailable: {exc.__class__.__name__}] Using heuristic planner.")
        return _heuristic_plan(query)
    except (json.JSONDecodeError, IndexError):
        return _heuristic_plan(query)


# ---------------------------------------------------------------------------
# Step 2 — Retriever
# ---------------------------------------------------------------------------

# Mirrors _CRITICAL_CAPS_IN_PARQUET / _build_districts_cache in api.py
_DESERT_CRITICAL_CAPS = [
    "anesthesia", "cardiology", "dialysis", "emergency",
    "icu", "obstetrics", "oncology", "pediatrics", "surgery",
]


def _recompute_desert_score(r: dict, df_cols: "set[str]") -> int:
    """Same weighted formula as api.py _build_districts_cache."""
    avg_trust = float(r.get("avg_trust_score") or 50.0)
    num_facilities = int(r.get("total_facilities") or 0)
    population = float(r.get("population") or 0.0)

    num_unverified = sum(
        1 for cap in _DESERT_CRITICAL_CAPS
        if f"cap_{cap}" not in df_cols or int(r.get(f"cap_{cap}") or 0) == 0
    ) + 1  # neonatal always unverified

    fac_norm = min(1.0, (num_facilities / (population / 100_000)) / 10.0) if population > 0 else 0.0
    contradiction_density = (100.0 - avg_trust) / 100.0

    score = round(
        0.40 * (100 - avg_trust)
        + 0.30 * min(100, 20 * num_unverified)
        + 0.20 * (1 - fac_norm) * 100
        + 0.10 * contradiction_density * 100
    )
    return max(0, min(100, score))


def _retrieve_deserts(plan: dict) -> tuple[list[dict], str]:
    df = _get_districts().copy()
    state = (plan["location_filters"]).get("state")
    cap_filters = plan["capability_filters"]

    desc = "all India"
    if state:
        norm = normalize_state(state)
        if norm != "Unknown":
            df = df[df["state_clean"] == norm]
            desc = norm
        else:
            df = df[df["state_clean"].str.lower().str.contains(state.lower(), na=False)]
            desc = state

    # Recompute desert scores using the same formula as /districts endpoint
    df_cols: set[str] = set(df.columns)
    records = df.to_dict(orient="records")
    for r in records:
        r["_desert_score"] = _recompute_desert_score(r, df_cols)

    cap_col = f"cap_{cap_filters[0]}" if cap_filters else None
    if cap_col and cap_col in df_cols:
        records.sort(key=lambda r: int(r.get(cap_col) or 0))
        desc += f" | sorted by {cap_filters[0]} scarcity"
    else:
        records.sort(key=lambda r: r["_desert_score"], reverse=True)

    rows: list[dict] = []
    for r in records[:10]:
        top_gaps = r.get("top_gaps", "[]")
        if isinstance(top_gaps, str):
            try:
                top_gaps = json.loads(top_gaps)
            except json.JSONDecodeError:
                top_gaps = []
        entry: dict = {
            "district": r.get("district", ""),
            "state": r.get("state_clean", ""),
            "desert_score": r["_desert_score"],
            "total_facilities": int(r.get("total_facilities", 0)),
            "trustworthy_count": int(r.get("trustworthy_count", 0)),
            "avg_trust_score": round(float(r.get("avg_trust_score", 0)), 1),
            "population": int(r.get("population", 500_000)),
            "top_gaps": top_gaps,
        }
        for cap in cap_filters:
            col = f"cap_{cap}"
            entry[col] = int(r.get(col, 0))
        rows.append(entry)

    return rows, f"Retrieved {len(rows)} worst districts in {desc}"


async def _retrieve_facilities(query: str, plan: dict) -> tuple[list[dict], str]:
    coll = _get_chroma()
    merged = _get_merged()

    state = plan["location_filters"].get("state")
    cap_filters = plan["capability_filters"]
    intent = plan["intent"]
    n_raw = 120 if (state or cap_filters) else 60

    try:
        res = coll.query(query_texts=[query], n_results=min(n_raw, coll.count()))
    except Exception as exc:
        return [], f"ChromaDB error: {exc}"

    ids, metas, docs = res["ids"][0], res["metadatas"][0], res["documents"][0]

    # Filter by state + capabilities
    filtered = [
        (fid, meta, doc)
        for fid, meta, doc in zip(ids, metas, docs)
        if _matches_state(meta, state) and _matches_caps(meta, doc, cap_filters)
    ]
    if not filtered:
        filtered = list(zip(ids, metas, docs))

    candidates: list[dict] = []
    for fid, meta, doc in filtered:
        if fid not in merged.index:
            trust = float(meta.get("trust_score", 50.0))
            candidates.append({
                "facility_id": fid,
                "facility_name": doc.split(" | ")[0] if " | " in doc else fid,
                "city": meta.get("district", ""),
                "state": meta.get("state", ""),
                "trust_score": trust,
                "confidence_interval": [round(max(0.0, trust - 12.5), 1), round(min(100.0, trust + 12.5), 1)],
                "capabilities": [c.strip() for c in meta.get("capabilities", "").split(",") if c.strip()],
                "contradictions": [],
                "contradiction_count": 0,
                "reasoning_summary": "",
            })
            continue

        row = merged.loc[fid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        trust, ci = _score_and_ci(row)
        caps = _active_caps(row.get("capability_claims", ""))
        contras = _parse_json_list(row.get("contradictions", ""))
        candidates.append({
            "facility_id": fid,
            "facility_name": str(row.get("facility_name", "") or ""),
            "city": str(row.get("city", "") or ""),
            "state": str(row.get("state", "") or ""),
            "trust_score": trust,
            "confidence_interval": ci,
            "capabilities": caps,
            "contradictions": [
                {"type": c.get("contradiction_type", ""), "severity": c.get("severity", 0),
                 "reason": c.get("why_contradictory", "")}
                for c in contras
            ],
            "contradiction_count": len(contras),
            "reasoning_summary": str(row.get("reasoning_summary", "") or ""),
        })

    reverse = (intent == "find_facilities")
    candidates.sort(key=lambda x: x["trust_score"], reverse=reverse)
    top10 = candidates[:10]

    loc_desc = state or "all India"
    cap_desc = ", ".join(cap_filters) if cap_filters else "all capabilities"
    return top10, f"Found {len(filtered)} candidates in {loc_desc}, filtered by {cap_desc}"


# ---------------------------------------------------------------------------
# Step 3 — Validator
# ---------------------------------------------------------------------------

async def _validate_one(query: str, candidate: dict, intent: str) -> tuple[bool, str]:
    name = candidate.get("facility_name") or candidate.get("district", "?")
    city = candidate.get("city") or candidate.get("state", "")
    state_ = candidate.get("state", "")
    caps = ", ".join(candidate.get("capabilities") or []) or "(none)"
    trust = candidate.get("trust_score", "N/A")
    n_contra = candidate.get("contradiction_count", 0)
    fac_type = candidate.get("facility_type", "unknown")

    if intent == "find_suspicious":
        prompt = (
            "The user is looking for suspicious or untrustworthy facilities. "
            "A facility MATCHES if it has a low trust score (under 60), contradictions, "
            "or capability claims that seem implausible for its type. "
            f"Query: '{query}'. Facility: {name} in {city}. Type: {fac_type}. "
            f"Trust score: {trust}. Contradictions: {n_contra}. Capabilities claimed: {caps}. "
            "Does this facility match what the user is looking for? Reply YES or NO with one sentence."
        )
    else:  # find_facilities
        prompt = (
            "The user is looking for reliable facilities that can provide specific services. "
            "A facility MATCHES if it has the required capabilities (confirmed or inferred) "
            "and is in the right location. "
            f"Query: '{query}'. Facility: {name} in {city}. "
            f"Trust score: {trust}. Capabilities: {caps}. Location: {state_}. "
            "Does this facility match what the user is looking for? Reply YES or NO with one sentence."
        )
    try:
        msg = await _get_llm().chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
        )
        text = msg.choices[0].message.content.strip()
        return text.upper().startswith("YES"), text
    except Exception as exc:
        return True, f"Validation skipped ({exc.__class__.__name__})"


async def _validate(
    query: str, candidates: list[dict], intent: str
) -> tuple[list[dict], list[str]]:
    outcomes = await asyncio.gather(*[_validate_one(query, c, intent) for c in candidates])
    steps: list[str] = []
    passed: list[dict] = []
    for cand, (ok, reason) in zip(candidates, outcomes):
        label = "PASS" if ok else "FAIL"
        name = cand.get("facility_name") or cand.get("district", "?")
        steps.append(f"  [{label}] {name}: {reason}")
        if ok:
            passed.append(cand)
    return passed[:5], steps


# ---------------------------------------------------------------------------
# Step 4 — Web Verification (optional, fault-tolerant)
# ---------------------------------------------------------------------------

def _adjust_ci(ci: list[float], web_verified: bool, confirmed_count: int) -> list[float]:
    lo, hi = ci[0], ci[1]
    if web_verified and confirmed_count > 0:
        lo, hi = lo + 5, hi - 5      # narrow: extra confidence
    elif web_verified and confirmed_count == 0:
        lo, hi = lo - 10, hi + 10    # widen: found but caps unsubstantiated
    else:
        lo, hi = lo - 15, hi + 15    # widen more: not found on web at all
    return [round(max(0.0, min(100.0, lo)), 1), round(max(0.0, min(100.0, hi)), 1)]


async def _web_verify_results(results: list[dict]) -> tuple[list[dict], str]:
    if not _TAVILY_AVAILABLE:
        return results, "Web Verification: Skipped (service unavailable)"

    try:
        top3 = results[:3]
        verified_count = 0
        total_confirmed = 0
        detail_parts: list[str] = []

        for result in top3:
            name = result.get("facility_name", "")
            city = result.get("city", "")
            caps = result.get("capabilities", [])

            # Check pre-verified cache first
            cache_key = (name.lower(), city.lower())
            if cache_key in _web_cache:
                wv = _web_cache[cache_key]
            else:
                # Live Tavily call with 5-second timeout
                try:
                    wv = await asyncio.wait_for(
                        asyncio.to_thread(_tavily_verify, name, city, caps),
                        timeout=5.0,
                    )
                except Exception:
                    result["web_verified"] = None
                    result["web_sources"] = []
                    result["web_capabilities_confirmed"] = []
                    result["web_capabilities_unconfirmed"] = []
                    continue

            confirmed = wv.get("capabilities_confirmed_by_web") or []
            unconfirmed = wv.get("capabilities_not_found_on_web") or []
            web_verified_flag: bool = bool(wv.get("web_verified", False))

            result["web_verified"] = web_verified_flag
            result["web_sources"] = wv.get("web_sources") or []
            result["web_capabilities_confirmed"] = confirmed
            result["web_capabilities_unconfirmed"] = unconfirmed

            if "confidence_interval" in result:
                result["confidence_interval"] = _adjust_ci(
                    result["confidence_interval"], web_verified_flag, len(confirmed)
                )

            if web_verified_flag:
                verified_count += 1
            total_confirmed += len(confirmed)

            n_src = wv.get("sources_found", 0)
            if not web_verified_flag:
                detail_parts.append(f"{name} not found on web")
            elif len(confirmed) == 0:
                detail_parts.append(
                    f"{name} found on {n_src} sources but "
                    f"{len(unconfirmed)} claimed capabilities have no web evidence"
                )
            else:
                detail_parts.append(
                    f"{name} verified on {n_src} sources ({len(confirmed)} caps confirmed)"
                )

        checked = sum(1 for r in top3 if r.get("web_verified") is not None)
        detail = "; ".join(detail_parts) if detail_parts else "no detail available"
        step = (
            f"Web Verification: Checked {checked} facilities against public web. "
            f"{verified_count} verified, {total_confirmed} capabilities confirmed by web sources. "
            + detail + "."
        )
        return results, step

    except Exception as exc:
        return results, f"Web Verification: Skipped (error: {exc.__class__.__name__}: {str(exc)[:80]})"


# ---------------------------------------------------------------------------
# Step 5 — Composer
# ---------------------------------------------------------------------------

def _avg_ci(results: list[dict]) -> list[float]:
    cis = [r["confidence_interval"] for r in results if "confidence_interval" in r]
    if not cis:
        return [0.0, 100.0]
    return [round(sum(c[0] for c in cis) / len(cis), 1), round(sum(c[1] for c in cis) / len(cis), 1)]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def run_query(query: str) -> dict:
    t0 = time.time()
    steps: list[str] = []

    _root = _MlSpan("trustmap_query")
    _root.log(query_text=query, timestamp=str(datetime.now()))

    # Step 1 — Plan
    _sp1 = _MlSpan("1_planner")
    plan = await _plan(query)
    steps.append(f"Planning: [{plan.get('_source','llm')}] {plan['reasoning']}")
    _sp1.log(
        intent=plan["intent"],
        num_capability_filters=len(plan.get("capability_filters", [])),
        location=str((plan.get("location_filters") or {}).get("state")),
        planner_source=plan.get("_source", "llm"),
    )
    _sp1.close()

    intent = plan["intent"]
    _root.log(intent=intent)

    # Step 2 — Retrieve
    _sp2 = _MlSpan("2_retriever")
    if intent == "find_deserts":
        candidates, retrieve_desc = _retrieve_deserts(plan)
    else:
        candidates, retrieve_desc = await _retrieve_facilities(query, plan)
    steps.append(f"Retrieving: {retrieve_desc}")
    _sp2.log(num_candidates=len(candidates), retrieve_desc=retrieve_desc)
    _sp2.close()

    if not candidates:
        steps.append("No results found.")
        _root.log(num_results=0, total_time_ms=int((time.time() - t0) * 1000))
        _root.close()
        return {"results": [], "reasoning_steps": steps, "confidence_interval": [0.0, 100.0], "plan": plan}

    # Step 3 — Validate
    _sp3 = _MlSpan("3_validator")
    if intent == "find_deserts":
        final = candidates[:5]
        steps.append(f"Validating: skipped for find_deserts, returning top {len(final)}")
        _sp3.log(skipped=True, num_passed=len(final), num_rejected=0)
    else:
        validated, val_steps = await _validate(query, candidates, intent)
        steps.extend(val_steps)
        n_pass, n_total = len(validated), len(candidates)
        steps.append(f"Validating: {n_pass}/{n_total} facilities confirmed as matching query")
        final = validated if validated else candidates[:3]
        _sp3.log(num_passed=n_pass, num_rejected=n_total - n_pass, num_candidates=n_total)
    _sp3.close()

    # Step 4 — Web Verification
    _sp4 = _MlSpan("4_web_verification")
    if intent in {"find_suspicious", "find_facilities"}:
        final, web_step = await _web_verify_results(final)
        steps.append(web_step)
        n_verified = sum(1 for r in final if r.get("web_verified") is True)
        n_sources = sum(len(r.get("web_sources", [])) for r in final)
        _sp4.log(ran=True, num_verified=n_verified, num_sources_found=n_sources)
    else:
        steps.append("Web Verification: Skipped (not applicable for desert queries)")
        _sp4.log(ran=False, reason="find_deserts")
    _sp4.close()

    # Step 5 — Compose
    _sp5 = _MlSpan("5_composer")
    steps.append(f"Composing: returning {len(final)} results sorted by {plan['sort_by']}")
    trust_scores = [r.get("trust_score", 0) for r in final if "trust_score" in r]
    avg_trust = round(sum(trust_scores) / len(trust_scores), 1) if trust_scores else 0.0
    total_ms = int((time.time() - t0) * 1000)
    _sp5.log(num_results=len(final), avg_trust_score=avg_trust, total_time_ms=total_ms)
    _sp5.close()

    _root.log(num_results=len(final), avg_trust_score=avg_trust, total_time_ms=total_ms)
    _root.close()

    if MLFLOW_AVAILABLE:
        print(f"[mlflow] Trace logged  intent={intent}  results={len(final)}  {total_ms}ms")

    return {
        "results": final,
        "reasoning_steps": steps,
        "confidence_interval": _avg_ci(final),
        "plan": plan,
    }


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

def _print_response(query: str, resp: dict) -> None:
    bar = "=" * 72
    print(f"\n{bar}")
    print(f"QUERY: {query}")
    print(bar)

    p = resp.get("plan", {})
    src = p.get("_source", "llm")
    print(f"\n[Planner -- {src}]")
    print(f"  intent     : {p.get('intent')}")
    print(f"  caps       : {p.get('capability_filters')}")
    print(f"  state      : {(p.get('location_filters') or {}).get('state')}")
    print(f"  sort_by    : {p.get('sort_by')}")
    print(f"  reasoning  : {p.get('reasoning')}")

    steps = resp.get("reasoning_steps", [])
    print("\n[Reasoning steps]")
    for s in steps:
        print(f"  {s}")

    results = resp.get("results", [])
    print(f"\nResults ({len(results)}):")
    for i, r in enumerate(results, 1):
        if "desert_score" in r:
            print(
                f"  #{i}  {r['district']}, {r['state']}"
                f"  desert={r['desert_score']}  facilities={r['total_facilities']}"
                f"  top_gaps={r.get('top_gaps', [])[:3]}"
            )
        else:
            print(
                f"  #{i}  {r.get('facility_name')}  ({r.get('city')}, {r.get('state')})"
                f"  trust={r.get('trust_score')}  CI={r.get('confidence_interval')}"
            )
            caps = r.get("capabilities", [])
            if caps:
                print(f"       caps: {caps[:4]}")
            wv = r.get("web_verified")
            if wv is not None:
                conf = r.get("web_capabilities_confirmed", [])
                unconf = r.get("web_capabilities_unconfirmed", [])
                src_count = len(r.get("web_sources", []))
                print(f"       web_verified={wv}  sources={src_count}  "
                      f"confirmed={conf}  unconfirmed={unconf}")

    print(f"\nAvg CI: {resp.get('confidence_interval')}")


async def _main() -> None:
    queries = [
        "Suspicious dental clinics in India",
        "Emergency C-section in rural Maharashtra",
        "Worst dialysis deserts in India",
        "Multi-specialty hospitals that look too good to be true",
    ]
    for q in queries:
        resp = await run_query(q)
        _print_response(q, resp)


if __name__ == "__main__":
    asyncio.run(_main())
