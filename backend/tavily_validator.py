"""
Tavily-powered web verification for TrustMap India.

Cross-references our LLM-extracted facility assessments against live web data
to flag facilities whose claimed capabilities have no real-world evidence.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from tavily import TavilyClient

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from backend.trust_score import (
        activity_signal_score,
        capability_plausibility_score,
        completeness_score,
        compute_overall,
        internal_consistency_score,
    )
except ImportError:
    from trust_score import (  # type: ignore[no-redef]
        activity_signal_score,
        capability_plausibility_score,
        completeness_score,
        compute_overall,
        internal_consistency_score,
    )

ASSESSMENTS_PATH = Path("backend/data/assessments_llm.parquet")
VERIFICATIONS_PATH = Path("backend/data/web_verifications.json")

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

# Capability keywords used to scan web result text
_CAP_KEYWORDS: dict[str, list[str]] = {
    "emergency":     ["emergency", "casualty", "trauma", "24/7", "24 hour"],
    "icu":           ["icu", "intensive care", "critical care", "ventilator"],
    "surgery":       ["surgery", "surgical", "operation theatre", "OT"],
    "obstetrics":    ["obstetrics", "maternity", "delivery", "gynecology", "gynaecology", "c-section"],
    "dialysis":      ["dialysis", "kidney", "renal", "nephrology"],
    "oncology":      ["oncology", "cancer", "chemotherapy", "radiation"],
    "cardiology":    ["cardiology", "cardiac", "heart", "cardiothoracic"],
    "anesthesia":    ["anesthesia", "anaesthesia", "anesthesiology"],
    "pediatrics":    ["pediatrics", "paediatrics", "children", "child care", "neonatal"],
    "mental_health": ["mental health", "psychiatry", "psychiatrist", "psychology"],
    "dentistry":     ["dental", "dentist", "dentistry", "orthodontic"],
    "primary_care":  ["general medicine", "primary care", "general practitioner", "OPD", "outpatient"],
    "ophthalmology": ["ophthalmology", "eye", "retina", "cataract"],
    "orthopedics":   ["orthopedics", "orthopaedics", "bone", "joint", "fracture"],
    "dermatology":   ["dermatology", "skin", "dermatologist"],
    "runs_24_7":     ["24/7", "24 hours", "round the clock", "open all day"],
}


def _parse_capability_claims(raw: Any) -> list[str]:
    """Extract capability names with status confirmed or inferred."""
    if not raw:
        return []
    try:
        items = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(items, list):
            return []
        return [
            c["capability"] for c in items
            if isinstance(c, dict) and c.get("status") in {"confirmed", "inferred"}
        ]
    except (json.JSONDecodeError, TypeError, KeyError):
        return []


def _text_mentions_capability(text: str, capability: str) -> bool:
    keywords = _CAP_KEYWORDS.get(capability, [capability])
    tl = text.lower()
    return any(kw.lower() in tl for kw in keywords)


def verify_facility(
    facility_name: str,
    city: str,
    claimed_capabilities: list[str],
    *,
    client: TavilyClient | None = None,
) -> dict:
    """
    Search the web for a facility and check which claimed capabilities
    appear in real web sources.

    Returns a dict matching the spec shape.
    """
    if client is None:
        client = TavilyClient(api_key=TAVILY_API_KEY)

    query = f"{facility_name} {city} hospital clinic"

    try:
        resp = client.search(
            query=query,
            search_depth="basic",
            max_results=5,
        )
    except Exception as exc:
        return {
            "facility_name": facility_name,
            "city": city,
            "web_verified": False,
            "sources_found": 0,
            "web_sources": [],
            "capabilities_confirmed_by_web": [],
            "capabilities_not_found_on_web": claimed_capabilities,
            "verification_summary": f"Search failed: {exc}",
        }

    results: list[dict] = resp.get("results", [])

    # Combine all result text for capability scanning
    combined_text = " ".join(
        (r.get("title", "") + " " + r.get("content", ""))
        for r in results
    )

    # Check if the facility name appears in any result title/content
    name_lower = facility_name.lower()
    city_lower = city.lower()
    web_verified = any(
        name_lower in (r.get("title", "") + r.get("content", "")).lower()
        or city_lower in (r.get("title", "") + r.get("content", "")).lower()
        for r in results
        if name_lower in (r.get("title", "") + r.get("content", "")).lower()
    )

    web_sources = [r["url"] for r in results if r.get("url")]

    confirmed: list[str] = []
    not_found: list[str] = []
    for cap in claimed_capabilities:
        if _text_mentions_capability(combined_text, cap):
            confirmed.append(cap)
        else:
            not_found.append(cap)

    if not results:
        summary = f"No web results found for {facility_name} in {city}."
    elif not web_verified:
        summary = (
            f"Facility name not directly found in web results; "
            f"{len(confirmed)}/{len(claimed_capabilities)} capabilities have web evidence."
        )
    else:
        summary = (
            f"Facility found in {len(results)} web sources; "
            f"{len(confirmed)}/{len(claimed_capabilities)} claimed capabilities confirmed online."
        )

    return {
        "facility_name": facility_name,
        "city": city,
        "web_verified": web_verified,
        "sources_found": len(results),
        "web_sources": web_sources,
        "capabilities_confirmed_by_web": confirmed,
        "capabilities_not_found_on_web": not_found,
        "verification_summary": summary,
    }


def _recompute_trust(row: pd.Series) -> float:
    row_dict = row.to_dict()
    subs = {
        "internal_consistency":   internal_consistency_score(row_dict),
        "capability_plausibility": capability_plausibility_score(row_dict, row),
        "activity_signal":        activity_signal_score(row),
        "completeness":           completeness_score(row),
    }
    return round(compute_overall(subs), 1)


def batch_verify_top_suspicious(n: int = 20) -> list[dict]:
    """
    Load assessments, recompute trust scores, take the n lowest-trust
    facilities, run verify_facility on each, save to web_verifications.json,
    and print a summary table.
    """
    print(f"[tavily] Loading assessments from {ASSESSMENTS_PATH} …")
    df = pd.read_parquet(ASSESSMENTS_PATH)

    print("[tavily] Recomputing trust scores …")
    df["_trust"] = df.apply(_recompute_trust, axis=1)
    df = df.sort_values("_trust").head(n).reset_index(drop=True)

    client = TavilyClient(api_key=TAVILY_API_KEY)
    verifications: list[dict] = []

    print(f"\n[tavily] Verifying {len(df)} facilities (1 API call each) …\n")

    for i, row in df.iterrows():
        name = str(row.get("facility_name") or "")
        city = str(row.get("city") or "")
        caps = _parse_capability_claims(row.get("capability_claims", "[]"))
        trust = row["_trust"]

        print(f"  [{i+1:02d}/{len(df)}] {name} ({city})  trust={trust}")
        result = verify_facility(name, city, caps, client=client)
        result["trust_score"] = trust
        verifications.append(result)

        # Small pause to be kind to the API
        time.sleep(0.3)

    # Save
    VERIFICATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VERIFICATIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(verifications, f, indent=2, ensure_ascii=False)
    print(f"\n[tavily] Saved {len(verifications)} verifications -> {VERIFICATIONS_PATH}")

    # Summary table
    col_w = [40, 18, 7, 12, 8]
    header = (
        f"{'Facility':<{col_w[0]}}  {'City':<{col_w[1]}}  "
        f"{'Trust':>{col_w[2]}}  {'Web Verified':<{col_w[3]}}  {'Sources':>{col_w[4]}}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for v in verifications:
        verified_str = "YES" if v["web_verified"] else "NO "
        print(
            f"{v['facility_name'][:col_w[0]]:<{col_w[0]}}  "
            f"{v['city'][:col_w[1]]:<{col_w[1]}}  "
            f"{v['trust_score']:>{col_w[2]}.1f}  "
            f"{verified_str:<{col_w[3]}}  "
            f"{v['sources_found']:>{col_w[4]}}"
        )
    print(sep)

    return verifications


def _print_result(r: dict) -> None:
    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  {r['facility_name']}  ({r['city']})")
    print(sep)
    print(f"  web_verified           : {r['web_verified']}")
    print(f"  sources_found          : {r['sources_found']}")
    print(f"  web_sources            :")
    for url in r["web_sources"]:
        print(f"    - {url}")
    print(f"  confirmed_by_web       : {r['capabilities_confirmed_by_web']}")
    print(f"  not_found_on_web       : {r['capabilities_not_found_on_web']}")
    print(f"  summary                : {r['verification_summary']}")


if __name__ == "__main__":
    client = TavilyClient(api_key=TAVILY_API_KEY)

    test_cases = [
        ("1000 Smiles Dental Clinic",          "Hyderabad", ["dentistry", "primary_care"]),
        ("Krishna Homeopathy Research Hospital","Jaipur",    ["primary_care", "surgery", "emergency"]),
        ("7 Star Healthcare",                   "Delhi",     ["emergency", "icu", "surgery", "cardiology"]),
    ]

    print("=" * 60)
    print("  TAVILY SPOT-CHECK  (3 facilities)")
    print("=" * 60)

    for name, city, caps in test_cases:
        result = verify_facility(name, city, caps, client=client)
        _print_result(result)
        time.sleep(0.5)
