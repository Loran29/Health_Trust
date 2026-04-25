from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import anthropic
import chromadb
import pandas as pd
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from dotenv import load_dotenv

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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

ASSESSMENTS_PATH = Path("backend/data/assessments_llm.parquet")
FACILITIES_PATH = Path("data/facilities_clean.parquet")
DISTRICTS_PATH = Path("backend/data/districts.parquet")
CHROMA_PATH = Path("backend/data/chroma_db")
COLLECTION_NAME = "health_trust_facilities"

# Use Haiku for speed and low cost; same proxy Claude Code uses
LLM_MODEL = "claude-haiku-4-5-20251001"

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

_llm: anthropic.AsyncAnthropic | None = None
_chroma_coll: chromadb.Collection | None = None
_merged_df: pd.DataFrame | None = None
_districts_df: pd.DataFrame | None = None


def _get_llm() -> anthropic.AsyncAnthropic:
    global _llm
    if _llm is None:
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY", "")
        _llm = anthropic.AsyncAnthropic(base_url=base_url, auth_token=auth_token)
    return _llm


def _get_chroma() -> chromadb.Collection:
    global _chroma_coll
    if _chroma_coll is None:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        ef = DefaultEmbeddingFunction()
        _chroma_coll = client.get_collection(COLLECTION_NAME, embedding_function=ef)
    return _chroma_coll


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
        msg = await _get_llm().messages.create(
            model=LLM_MODEL,
            system=PLANNER_SYSTEM,
            messages=[{"role": "user", "content": query}],
            max_tokens=400,
        )
        raw_text = msg.content[0].text.strip()
        # Strip ```json ... ``` if present
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        raw_text = match.group(1) if match else raw_text
        plan = json.loads(raw_text)
        return _clean_plan(plan)
    except (anthropic.APIError, anthropic.APIConnectionError) as exc:
        print(f"  [LLM unavailable: {exc.__class__.__name__}] Using heuristic planner.")
        return _heuristic_plan(query)
    except (json.JSONDecodeError, IndexError):
        return _heuristic_plan(query)


# ---------------------------------------------------------------------------
# Step 2 — Retriever
# ---------------------------------------------------------------------------

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

    cap_col = f"cap_{cap_filters[0]}" if cap_filters else None
    if cap_col and cap_col in df.columns:
        df = df.sort_values(cap_col)
        desc += f" | sorted by {cap_filters[0]} scarcity"
    else:
        df = df.sort_values("desert_score")

    rows: list[dict] = []
    for r in df.head(10).to_dict(orient="records"):
        top_gaps = r.get("top_gaps", "[]")
        if isinstance(top_gaps, str):
            try:
                top_gaps = json.loads(top_gaps)
            except json.JSONDecodeError:
                top_gaps = []
        entry: dict = {
            "district": r.get("district", ""),
            "state": r.get("state_clean", ""),
            "desert_score": round(float(r.get("desert_score", 0)), 1),
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

async def _validate_one(query: str, candidate: dict) -> tuple[bool, str]:
    name = candidate.get("facility_name") or candidate.get("district", "?")
    city = candidate.get("city") or candidate.get("state", "")
    state_ = candidate.get("state", "")
    caps = ", ".join(candidate.get("capabilities") or []) or "(none)"
    trust = candidate.get("trust_score", candidate.get("desert_score", "N/A"))
    n_contra = candidate.get("contradiction_count", 0)
    score_label = "desert score" if "desert_score" in candidate else "trust score"

    prompt = (
        f"Query: '{query}'. "
        f"Result: {name} in {city}, {state_}. "
        f"Capabilities: {caps}. "
        f"{score_label.capitalize()}: {trust}. Contradictions: {n_contra}. "
        "Does this result match the query? Reply ONLY: YES or NO and one sentence reason."
    )
    try:
        msg = await _get_llm().messages.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
        )
        text = msg.content[0].text.strip()
        return text.upper().startswith("YES"), text
    except Exception as exc:
        return True, f"Validation skipped ({exc.__class__.__name__})"


async def _validate(
    query: str, candidates: list[dict]
) -> tuple[list[dict], list[str]]:
    outcomes = await asyncio.gather(*[_validate_one(query, c) for c in candidates])
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
# Step 4 — Composer
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
    steps: list[str] = []

    # Step 1 — Plan
    plan = await _plan(query)
    steps.append(f"Planning: [{plan.get('_source','llm')}] {plan['reasoning']}")

    intent = plan["intent"]

    # Step 2 — Retrieve
    if intent == "find_deserts":
        candidates, retrieve_desc = _retrieve_deserts(plan)
    else:
        candidates, retrieve_desc = await _retrieve_facilities(query, plan)
    steps.append(f"Retrieving: {retrieve_desc}")

    if not candidates:
        steps.append("No results found.")
        return {"results": [], "reasoning_steps": steps, "confidence_interval": [0.0, 100.0], "plan": plan}

    # Step 3 — Validate
    if intent == "find_deserts":
        # Desert results are district-level stats — no facility YES/NO needed
        final = candidates[:5]
        steps.append(f"Validating: skipped for find_deserts, returning top {len(final)}")
    else:
        validated, val_steps = await _validate(query, candidates)
        steps.extend(val_steps)
        n_pass, n_total = len(validated), len(candidates)
        steps.append(f"Validating: {n_pass}/{n_total} facilities confirmed as matching query")
        final = validated if validated else candidates[:3]

    # Step 4 — Compose
    steps.append(f"Composing: returning {len(final)} results sorted by {plan['sort_by']}")

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
    print(f"\n[Planner — {src}]")
    print(f"  intent     : {p.get('intent')}")
    print(f"  caps       : {p.get('capability_filters')}")
    print(f"  state      : {(p.get('location_filters') or {}).get('state')}")
    print(f"  sort_by    : {p.get('sort_by')}")
    print(f"  reasoning  : {p.get('reasoning')}")

    steps = resp.get("reasoning_steps", [])
    for s in steps:
        if s.startswith("Retrieving:") or s.startswith("Validating:") or s.startswith("Composing:"):
            print(f"\n{s}")

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

    print(f"\nAvg CI: {resp.get('confidence_interval')}")


async def _main() -> None:
    queries = [
        "Emergency C-section in rural Maharashtra",
        "Suspicious dental clinics in India",
        "Worst dialysis deserts in India",
    ]
    for q in queries:
        resp = await run_query(q)
        _print_response(q, resp)


if __name__ == "__main__":
    asyncio.run(_main())
