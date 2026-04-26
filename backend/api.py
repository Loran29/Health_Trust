from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query as QueryParam, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from backend.agent import run_query
    from backend.trust_score import (
        activity_signal_score,
        capability_plausibility_score,
        completeness_score,
        compute_overall,
        confidence_interval as compute_ci,
        internal_consistency_score,
    )
    from backend.schemas import (
        Capability,
        CapabilityClaim,
        CapabilityStatus,
        Contradiction,
        ContradictionType,
        FacilityAssessment,
        FacilitySearchResult,
        QueryPlan,
        TrustSubscores,
    )
except ImportError:
    from agent import run_query  # type: ignore[no-redef]
    from trust_score import (  # type: ignore[no-redef]
        activity_signal_score,
        capability_plausibility_score,
        completeness_score,
        compute_overall,
        confidence_interval as compute_ci,
        internal_consistency_score,
    )
    from schemas import (  # type: ignore[no-redef]
        Capability,
        CapabilityClaim,
        CapabilityStatus,
        Contradiction,
        ContradictionType,
        FacilityAssessment,
        FacilitySearchResult,
        QueryPlan,
        TrustSubscores,
    )

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ASSESSMENTS_PATH = Path("backend/data/assessments_llm.parquet")
FACILITIES_PATH = Path("data/facilities_clean.parquet")
DISTRICTS_PATH = Path("backend/data/districts.parquet")

# ---------------------------------------------------------------------------
# Module-level data cache (loaded once at process startup)
# ---------------------------------------------------------------------------

_assessments: pd.DataFrame | None = None   # indexed by facility_id
_facilities: pd.DataFrame | None = None    # indexed by facility_id
_merged: pd.DataFrame | None = None        # inner join, indexed by facility_id
_districts: pd.DataFrame | None = None
_districts_cache: list[dict] | None = None

# 9 critical capabilities that actually exist as cap_* columns in districts parquet
# "neonatal" is in the allowed UI list but NOT in ALL_CAPABILITIES — treated as always 0
_CRITICAL_CAPS_IN_PARQUET = [
    "anesthesia", "cardiology", "dialysis", "emergency",
    "icu", "obstetrics", "oncology", "pediatrics", "surgery",
]
# Full allowed list for top_capability_gaps (10 items, neonatal included)
_ALLOWED_GAPS = {
    "anesthesia", "cardiology", "dialysis", "emergency",
    "icu", "neonatal", "obstetrics", "oncology", "pediatrics", "surgery",
}


def _make_fid(name: Any, city: Any) -> str:
    n = "" if not name or str(name).lower() in {"nan", "none"} else str(name)
    c = "" if not city or str(city).lower() in {"nan", "none"} else str(city)
    return re.sub(r"[^a-z0-9]+", "-", f"{n} {c}".lower()).strip("-")


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        return default if f != f else f  # NaN → default
    except (TypeError, ValueError):
        return default


def _safe_int_clamp(val: Any, lo: int = 0, hi: int = 100) -> int:
    try:
        return max(lo, min(hi, int(round(float(val)))))
    except (TypeError, ValueError):
        return lo


def _build_districts_cache(df: pd.DataFrame) -> list[dict]:
    """Pre-compute the districts response array once at startup."""
    rows: list[dict] = []
    for _, r in df.iterrows():
        avg_trust = _safe_float(r.get("avg_trust_score"), 50.0)
        num_facilities = int(r.get("total_facilities", 0) or 0)
        population = _safe_float(r.get("population"), 0.0)

        # Count critical caps with 0 confirmed/inferred facilities in district
        num_unverified = 0
        for cap in _CRITICAL_CAPS_IN_PARQUET:
            col = f"cap_{cap}"
            if col not in df.columns or _safe_int_clamp(r.get(col), 0, 10**9) == 0:
                num_unverified += 1
        # neonatal is never in parquet → always unverified
        num_unverified += 1  # neonatal

        # facilities_per_100k normalised
        if population > 0:
            per_100k = num_facilities / (population / 100_000)
            fac_norm = min(1.0, per_100k / 10.0)
        else:
            fac_norm = 0.0

        contradiction_density = (100.0 - avg_trust) / 100.0

        desert_score = round(
            0.40 * (100 - avg_trust)
            + 0.30 * min(100, 20 * num_unverified)
            + 0.20 * (1 - fac_norm) * 100
            + 0.10 * contradiction_density * 100
        )
        desert_score = max(0, min(100, desert_score))

        # top_capability_gaps: parse existing top_gaps, filter to allowed set,
        # then pad with unverified critical caps sorted by cap count ascending
        existing_gaps: list[str] = []
        raw_gaps = r.get("top_gaps", "[]")
        try:
            parsed = json.loads(raw_gaps) if isinstance(raw_gaps, str) else (raw_gaps or [])
            existing_gaps = [g for g in (parsed if isinstance(parsed, list) else []) if g in _ALLOWED_GAPS]
        except (json.JSONDecodeError, TypeError):
            pass

        # Add missing critical caps sorted by count asc (lowest coverage first)
        cap_counts: list[tuple[int, str]] = []
        for cap in _CRITICAL_CAPS_IN_PARQUET:
            if cap not in existing_gaps:
                col = f"cap_{cap}"
                cnt = _safe_int_clamp(r.get(col), 0, 10**9) if col in df.columns else 0
                cap_counts.append((cnt, cap))
        cap_counts.sort()
        for _, cap in cap_counts:
            if cap not in existing_gaps:
                existing_gaps.append(cap)
        if "neonatal" not in existing_gaps:
            existing_gaps.append("neonatal")

        top_gaps = existing_gaps[:5]  # max 5

        rows.append({
            "state": str(r.get("state_clean", "") or ""),
            "district": str(r.get("district", "") or ""),
            "num_facilities": num_facilities,
            "avg_trust_score": round(avg_trust, 1),
            "desert_score": desert_score,
            "top_capability_gaps": top_gaps,
            "population": int(population) if population > 0 else None,
        })
    return rows


def _load_data() -> None:
    global _assessments, _facilities, _merged, _districts, _districts_cache
    asmt = pd.read_parquet(ASSESSMENTS_PATH)
    fac = pd.read_parquet(FACILITIES_PATH)
    fac["facility_id"] = fac.apply(
        lambda r: _make_fid(r.get("name"), r.get("address_city")), axis=1
    )
    _assessments = asmt.set_index("facility_id", drop=False)
    _facilities = fac.set_index("facility_id", drop=False)
    _merged = (
        asmt.merge(fac, on="facility_id", how="inner", suffixes=("_asmt", "_raw"))
        .set_index("facility_id", drop=False)
    )
    _districts = pd.read_parquet(DISTRICTS_PATH)
    _districts_cache = _build_districts_cache(_districts)
    print(
        f"[api] Loaded: {len(asmt):,} assessments | {len(fac):,} facilities | "
        f"{len(_merged):,} joined | {len(_districts):,} districts"
    )


_load_data()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="TrustMap India API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    text: str = Field(min_length=1)
    user_lat: float | None = None
    user_lng: float | None = None


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _parse_json_list(raw: Any) -> list:
    if isinstance(raw, list):
        return raw
    try:
        items = json.loads(raw)
        return items if isinstance(items, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _get_latlon(facility_id: str) -> tuple[float, float]:
    if _merged is not None and facility_id in _merged.index:
        row = _merged.loc[facility_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        # After merge with suffixes ("_asmt","_raw"), prefer raw facility coords
        lat = _safe_float(row.get("latitude_raw") or row.get("latitude_asmt") or row.get("latitude"))
        lon = _safe_float(row.get("longitude_raw") or row.get("longitude_asmt") or row.get("longitude"))
        return lat, lon
    return 0.0, 0.0


def _ci_ints(ci_raw: Any) -> tuple[int, int]:
    """Convert [float, float] or (float, float) to a valid (int, int) CI."""
    try:
        lo, hi = int(round(float(ci_raw[0]))), int(round(float(ci_raw[1])))
    except (TypeError, ValueError, IndexError):
        lo, hi = 0, 100
    lo = max(0, min(100, lo))
    hi = max(0, min(100, hi))
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _top_contradiction_text(contras: list[dict]) -> str | None:
    if not contras:
        return None
    worst = max(contras, key=lambda c: c.get("severity", 0))
    return (worst.get("reason") or worst.get("why_contradictory") or None)


def _match_reason(r: dict) -> str:
    caps = r.get("capabilities") or []
    rs = r.get("reasoning_summary", "")
    if rs:
        return str(rs)[:120]
    if caps:
        return f"Matched on {', '.join(caps[:3])} capabilities."
    return f"{r.get('facility_name', 'Facility')} matched your query."


def _agent_result_to_search_result(r: dict) -> FacilitySearchResult:
    lat, lon = _get_latlon(r["facility_id"])
    return FacilitySearchResult(
        facility_id=r["facility_id"],
        facility_name=r.get("facility_name", ""),
        city=r.get("city", ""),
        state=r.get("state", ""),
        latitude=lat,
        longitude=lon,
        overall_trust_score=_safe_int_clamp(r.get("trust_score", 50)),
        top_contradiction=_top_contradiction_text(r.get("contradictions", [])),
        match_reason=_match_reason(r),
    )


def _agent_plan_to_query_plan(plan: dict) -> QueryPlan:
    loc = plan.get("location_filters") or {}
    valid_caps: list[Capability] = []
    for c in (plan.get("capability_filters") or []):
        try:
            valid_caps.append(Capability(c))
        except ValueError:
            pass
    reasoning = plan.get("reasoning", "")
    return QueryPlan(
        location_filters={
            "state": loc.get("state"),
            "city": loc.get("district") or loc.get("city"),
            "region_type": loc.get("region_type"),
        },
        capability_filters=valid_caps,
        constraints=[reasoning] if reasoning else [],
    )


def _safe_serialise(records: list[dict]) -> list[dict]:
    """Convert numpy scalars and NaN to plain Python types for JSON."""
    out = []
    for r in records:
        entry: dict[str, Any] = {}
        for k, v in r.items():
            if hasattr(v, "item"):          # numpy scalar
                v = v.item()
            elif isinstance(v, float) and v != v:  # NaN
                v = None
            elif isinstance(v, str):
                try:                         # top_gaps is stored as JSON string
                    parsed = json.loads(v)
                    if isinstance(parsed, list):
                        v = parsed
                except (json.JSONDecodeError, ValueError):
                    pass
            entry[k] = v
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "TrustMap India"}


@app.post("/query")
async def query(request: QueryRequest) -> dict:
    t0 = time.time()
    try:
        resp = await run_query(request.text)
    except Exception as exc:
        return {
            "results": [],
            "reasoning_steps": [f"Error: {exc}"],
            "confidence_interval": [0, 100],
            "query_time_ms": int((time.time() - t0) * 1000),
            "query_plan": {},
        }

    results: list[dict] = []
    for r in resp.get("results", []):
        # Desert results (find_deserts) don't map to FacilitySearchResult
        if "desert_score" in r:
            results.append({
                "district": r.get("district", ""),
                "state": r.get("state", ""),
                "desert_score": r.get("desert_score", 0),
                "total_facilities": r.get("total_facilities", 0),
                "top_gaps": r.get("top_gaps", []),
            })
        else:
            try:
                results.append(_agent_result_to_search_result(r).model_dump())
            except Exception:
                continue

    ci = list(_ci_ints(resp.get("confidence_interval", [0, 100])))

    return {
        "results": results,
        "reasoning_steps": resp.get("reasoning_steps", []),
        "confidence_interval": ci,
        "query_time_ms": int((time.time() - t0) * 1000),
        "query_plan": _agent_plan_to_query_plan(resp.get("plan", {})).model_dump(),
    }


@app.get("/facility/{facility_id}", response_model=FacilityAssessment)
def get_facility(facility_id: str) -> FacilityAssessment:
    if _assessments is None or facility_id not in _assessments.index:
        raise HTTPException(status_code=404, detail="Facility not found")

    asmt = _assessments.loc[facility_id]
    if isinstance(asmt, pd.DataFrame):
        asmt = asmt.iloc[0]

    # Use merged row for trust score recomputation (has raw facility fields)
    if _merged is not None and facility_id in _merged.index:
        raw = _merged.loc[facility_id]
        if isinstance(raw, pd.DataFrame):
            raw = raw.iloc[0]
    else:
        raw = asmt

    # Recompute subscores (LLM stub values are all 50 — use real functions)
    row_dict = raw.to_dict()
    ic = internal_consistency_score(row_dict)
    cp = capability_plausibility_score(row_dict, raw)
    act = activity_signal_score(raw)
    comp = completeness_score(raw)
    subscores = {
        "internal_consistency": ic,
        "capability_plausibility": cp,
        "activity_signal": act,
        "completeness": comp,
    }
    overall = compute_overall(subscores)
    ci = compute_ci(subscores)

    # Parse capability_claims
    cap_claims: list[CapabilityClaim] = []
    for item in _parse_json_list(asmt.get("capability_claims", "[]")):
        try:
            cap_claims.append(CapabilityClaim.model_validate(item))
        except Exception:
            try:
                item = dict(item)
                item["status"] = item.get("status", "unknown")
                try:
                    CapabilityStatus(item["status"])
                except ValueError:
                    item["status"] = "unknown"
                cap_claims.append(CapabilityClaim.model_validate(item))
            except Exception:
                pass

    # Parse contradictions
    contras: list[Contradiction] = []
    for item in _parse_json_list(asmt.get("contradictions", "[]")):
        try:
            contras.append(Contradiction.model_validate(item))
        except Exception:
            try:
                item = dict(item)
                try:
                    ContradictionType(item.get("contradiction_type", ""))
                except ValueError:
                    item["contradiction_type"] = "other"
                contras.append(Contradiction.model_validate(item))
            except Exception:
                pass

    ci_int = _ci_ints(ci)

    return FacilityAssessment(
        facility_id=facility_id,
        facility_name=str(asmt.get("facility_name") or ""),
        city=str(asmt.get("city") or ""),
        state=str(asmt.get("state") or ""),
        latitude=_safe_float(asmt.get("latitude")),
        longitude=_safe_float(asmt.get("longitude")),
        facility_type=str(asmt.get("facility_type") or "unknown"),
        capability_claims=cap_claims,
        contradictions=contras,
        trust_subscores=TrustSubscores(
            internal_consistency=_safe_int_clamp(ic),
            capability_plausibility=_safe_int_clamp(cp),
            activity_signal=_safe_int_clamp(act),
            completeness=_safe_int_clamp(comp),
        ),
        overall_trust_score=_safe_int_clamp(overall),
        confidence_interval=ci_int,
        reasoning_summary=str(asmt.get("reasoning_summary") or "")[:200],
    )


@app.get("/districts")
def get_districts(
    state: str | None = QueryParam(default=None),
    capability: str | None = QueryParam(default=None),
) -> Response:
    cache = _districts_cache or []
    output = cache

    if state:
        sl = state.lower()
        output = [d for d in output if d["state"].lower() == sl]

    if capability:
        cap_lower = capability.lower()
        # Filter to districts where that capability is listed as a gap
        # (i.e. it's under-served in that district)
        output = [d for d in output if cap_lower in d["top_capability_gaps"]]

    return Response(content=json.dumps(output), media_type="application/json")


@app.get("/facility-pins")
def get_facility_pins() -> list[dict]:
    """Real facility pins from joined data, capped at 500 highest-trust for map performance."""
    if _merged is None:
        return []

    # Deduplicate, drop rows with no coordinates, sort by trust score
    df = _merged.copy()
    df = df[~df.index.duplicated(keep="first")]
    # After merge, lat/lon are suffixed; prefer raw facility coords
    lat_col = "latitude_raw" if "latitude_raw" in df.columns else "latitude"
    lon_col = "longitude_raw" if "longitude_raw" in df.columns else "longitude"
    df = df[(df[lat_col].notna()) & (df[lon_col].notna())]
    df = df[(df[lat_col] != 0.0) | (df[lon_col] != 0.0)]

    trust_col = "overall_trust_score_asmt" if "overall_trust_score_asmt" in df.columns else "overall_trust_score"
    df = df.sort_values(trust_col, ascending=False)

    pins = []
    for _, row in df.iterrows():
        contra_raw = row.get("contradictions", "[]")
        has_contra = bool(
            contra_raw
            and str(contra_raw) not in ("[]", "", "null", "None")
        )
        pins.append({
            "facility_id": str(row.get("facility_id", "")),
            "name": str(row.get("facility_name", "") or ""),
            "latitude": _safe_float(row.get(lat_col)),
            "longitude": _safe_float(row.get(lon_col)),
            "trust_score": _safe_int_clamp(row.get(trust_col, 50)),
            "has_contradictions": has_contra,
        })
    return pins
