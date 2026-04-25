from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from backend.trust_score import (
        completeness_score,
        internal_consistency_score,
        capability_plausibility_score,
        activity_signal_score,
        compute_overall,
    )
except ImportError:
    from trust_score import (  # type: ignore[no-redef]
        completeness_score,
        internal_consistency_score,
        capability_plausibility_score,
        activity_signal_score,
        compute_overall,
    )

ASSESSMENTS_PATH = Path("backend/data/assessments_llm.parquet")
FACILITIES_PATH = Path("data/facilities_clean.parquet")
OUTPUT_PATH = Path("backend/data/districts.parquet")

DISTRICT_POPULATIONS: dict[str, int] = {
    "Mumbai": 12_000_000,
    "Delhi": 11_000_000,
    "Bangalore Urban": 8_000_000,
    "Hyderabad": 7_000_000,
    "Chennai": 7_000_000,
    "Kolkata": 5_000_000,
    "Pune": 3_500_000,
    "Jaipur": 3_000_000,
    "Lucknow": 2_800_000,
    "Ahmedabad": 5_500_000,
    "Guwahati": 1_000_000,
    "Patna": 2_500_000,
    "Bhopal": 1_800_000,
    "Dehradun": 600_000,
    "Ranchi": 1_100_000,
    "Raipur": 1_000_000,
    "Coimbatore": 1_600_000,
    "Nagpur": 2_400_000,
    "Indore": 2_000_000,
    "Varanasi": 1_200_000,
}
DEFAULT_POPULATION = 500_000

ALL_CAPABILITIES = [
    "emergency", "icu", "surgery", "obstetrics", "dialysis", "oncology",
    "cardiology", "anesthesia", "pediatrics", "mental_health", "dentistry",
    "primary_care", "ophthalmology", "orthopedics", "dermatology", "runs_24_7",
]

# ---------------------------------------------------------------------------
# State normalisation
# ---------------------------------------------------------------------------

# Two-letter (and short) ISO / common abbreviations → canonical
_ABBREV: dict[str, str] = {
    "ap": "Andhra Pradesh",
    "ar": "Arunachal Pradesh",
    "as": "Assam",
    "br": "Bihar",
    "cg": "Chhattisgarh",
    "ct": "Chhattisgarh",
    "ga": "Goa",
    "gj": "Gujarat",
    "hr": "Haryana",
    "hp": "Himachal Pradesh",
    "jh": "Jharkhand",
    "jk": "Jammu & Kashmir",
    "ka": "Karnataka",
    "kl": "Kerala",
    "la": "Ladakh",
    "mp": "Madhya Pradesh",
    "mh": "Maharashtra",
    "mn": "Manipur",
    "ml": "Meghalaya",
    "mz": "Mizoram",
    "nl": "Nagaland",
    "od": "Odisha",
    "or": "Odisha",
    "pb": "Punjab",
    "py": "Puducherry",
    "rj": "Rajasthan",
    "sk": "Sikkim",
    "tn": "Tamil Nadu",
    "ts": "Telangana",
    "tg": "Telangana",
    "tr": "Tripura",
    "up": "Uttar Pradesh",
    "uk": "Uttarakhand",
    "ut": "Uttarakhand",
    "wb": "West Bengal",
    "dl": "Delhi",
    "ch": "Chandigarh",
    "dn": "Dadra & Nagar Haveli and Daman & Diu",
    "dd": "Dadra & Nagar Haveli and Daman & Diu",
    "an": "Andaman & Nicobar Islands",
    "ld": "Lakshadweep",
    "u.p.": "Uttar Pradesh",
}

# Full / partial variant strings (lowercase key) → canonical
_VARIANTS: dict[str, str] = {
    # --- Andhra Pradesh ---
    "andhra pradesh": "Andhra Pradesh",
    "andhrapradesh": "Andhra Pradesh",
    "andhra": "Andhra Pradesh",
    "chittoor": "Andhra Pradesh",
    "kurnool": "Andhra Pradesh",
    "rajahmundry": "Andhra Pradesh",
    "prakasam district": "Andhra Pradesh",
    # --- Arunachal Pradesh ---
    "arunachal pradesh": "Arunachal Pradesh",
    "itanagar": "Arunachal Pradesh",
    # --- Assam ---
    "assam": "Assam",
    "guwahati": "Assam",
    "golaghat": "Assam",
    "darrang": "Assam",
    "sibsagar": "Assam",
    "silchar": "Assam",
    "barpeta, assam": "Assam",
    "barpeta": "Assam",
    # --- Bihar ---
    "bihar": "Bihar",
    "patna": "Bihar",
    "gaya": "Bihar",
    "supaul": "Bihar",
    "saran": "Bihar",
    "sitamarhi": "Bihar",
    "jehanabad, bihar": "Bihar",
    "jehanabad": "Bihar",
    "aurangabad-bihar": "Bihar",
    # Aurangabad is ambiguous (Bihar vs MH); hyphenated form is Bihar
    # --- Chhattisgarh ---
    "chhattisgarh": "Chhattisgarh",
    "chattisgarh": "Chhattisgarh",
    "raipur": "Chhattisgarh",
    "bhilai": "Chhattisgarh",
    "durg": "Chhattisgarh",
    # --- Goa ---
    "goa": "Goa",
    # --- Gujarat ---
    "gujarat": "Gujarat",
    "ahmedabad": "Gujarat",
    "surat": "Gujarat",
    "gandhinagar": "Gujarat",
    "rajkot": "Gujarat",
    "bharuch": "Gujarat",
    "mehsana": "Gujarat",
    "surendranagar district": "Gujarat",
    "surendranagar": "Gujarat",
    "veraval": "Gujarat",
    "vadodara": "Gujarat",
    "baroda": "Gujarat",
    # --- Haryana ---
    "haryana": "Haryana",
    "gurugram": "Haryana",
    "gurgaon": "Haryana",
    "faridabad": "Haryana",
    "kurukshetra": "Haryana",
    "nuh": "Haryana",
    "jhajjar": "Haryana",
    "charkhi dadri, haryana": "Haryana",
    "charkhi dadri": "Haryana",
    "fatehabad, haryana": "Haryana",
    "fatehabad": "Haryana",
    "sector 56": "Haryana",
    "zirakpur": "Haryana",   # actually Punjab/Mohali area, but listed under Haryana city
    # --- Himachal Pradesh ---
    "himachal pradesh": "Himachal Pradesh",
    "shimla": "Himachal Pradesh",
    "dharamshala": "Himachal Pradesh",
    # --- Jharkhand ---
    "jharkhand": "Jharkhand",
    "ranchi": "Jharkhand",
    "bokaro": "Jharkhand",
    "bokaro steel city, jharkhand": "Jharkhand",
    "bokaro steel city": "Jharkhand",
    "dhanbad": "Jharkhand",
    # --- Karnataka ---
    "karnataka": "Karnataka",
    "bengaluru": "Karnataka",
    "bangalore": "Karnataka",
    "belgaum": "Karnataka",
    "chikmagalur": "Karnataka",
    "udupi": "Karnataka",
    "hubli": "Karnataka",
    "mysuru": "Karnataka",
    "mysore": "Karnataka",
    "mangaluru": "Karnataka",
    "mangalore": "Karnataka",
    # --- Kerala ---
    "kerala": "Kerala",
    "kochi": "Kerala",
    "ernakulam": "Kerala",
    "kannur": "Kerala",
    "malappuram": "Kerala",
    "malappuram, kerala": "Kerala",
    "thrissur": "Kerala",
    "palakkad": "Kerala",
    "alappuzha": "Kerala",
    "pathanamthitta": "Kerala",
    "thiruvananthapuram": "Kerala",
    "chittur": "Kerala",
    "kozhikode": "Kerala",
    "calicut": "Kerala",
    # --- Ladakh ---
    "ladakh": "Ladakh",
    "leh": "Ladakh",
    "kargil": "Ladakh",
    # --- Madhya Pradesh ---
    "madhya pradesh": "Madhya Pradesh",
    "madhyapradesh": "Madhya Pradesh",
    "bhopal": "Madhya Pradesh",
    "indore": "Madhya Pradesh",
    "jabalpur": "Madhya Pradesh",
    "gwalior": "Madhya Pradesh",
    "singrauli": "Madhya Pradesh",
    "thatipur": "Madhya Pradesh",
    "guna, madhya pradesh": "Madhya Pradesh",
    "guna": "Madhya Pradesh",
    "dhar district, madhya pradesh": "Madhya Pradesh",
    "dhar district": "Madhya Pradesh",
    "dhar": "Madhya Pradesh",
    # --- Maharashtra ---
    "maharashtra": "Maharashtra",
    "mumbai": "Maharashtra",
    "navi mumbai": "Maharashtra",
    "navi mumbai, maharashtra": "Maharashtra",
    "pune": "Maharashtra",
    "pune, maharashtra": "Maharashtra",
    "pune-411044": "Maharashtra",
    "nagpur": "Maharashtra",
    "thane": "Maharashtra",
    "solapur": "Maharashtra",
    "mira bhayander": "Maharashtra",
    "pimpri-chinchwad": "Maharashtra",
    "pimpri chinchwad": "Maharashtra",
    "ambernath": "Maharashtra",
    "kalyan": "Maharashtra",
    "chinchwad": "Maharashtra",
    "jalgaon district": "Maharashtra",
    "jalgaon": "Maharashtra",
    "beed": "Maharashtra",
    "nashik": "Maharashtra",
    "aurangabad": "Maharashtra",
    "chandrapur": "Maharashtra",
    "dudhani": "Maharashtra",
    "mh": "Maharashtra",
    # --- Manipur ---
    "manipur": "Manipur",
    "imphal": "Manipur",
    # --- Meghalaya ---
    "meghalaya": "Meghalaya",
    "shillong": "Meghalaya",
    # --- Mizoram ---
    "mizoram": "Mizoram",
    "aizawl": "Mizoram",
    # --- Nagaland ---
    "nagaland": "Nagaland",
    "kohima": "Nagaland",
    # --- Odisha ---
    "odisha": "Odisha",
    "orissa": "Odisha",
    "bhubaneswar": "Odisha",
    # --- Punjab ---
    "punjab": "Punjab",
    "punjab region": "Punjab",
    "ludhiana": "Punjab",
    "amritsar": "Punjab",
    "patiala": "Punjab",
    "mohali": "Punjab",
    "gurdaspur": "Punjab",
    "sangrur": "Punjab",
    "ropar": "Punjab",
    "zirakpur": "Punjab",
    # --- Rajasthan ---
    "rajasthan": "Rajasthan",
    "jaipur": "Rajasthan",
    "jodhpur": "Rajasthan",
    "udaipur": "Rajasthan",
    "sikar": "Rajasthan",
    "churu": "Rajasthan",
    "pali-rajasthan": "Rajasthan",
    "pali": "Rajasthan",
    "rajsamand, rajasthan": "Rajasthan",
    "rajsamand": "Rajasthan",
    # --- Sikkim ---
    "sikkim": "Sikkim",
    "gangtok": "Sikkim",
    # --- Tamil Nadu ---
    "tamil nadu": "Tamil Nadu",
    "tamilnadu": "Tamil Nadu",
    "chennai": "Tamil Nadu",
    "coimbatore": "Tamil Nadu",
    "salem": "Tamil Nadu",
    "erode": "Tamil Nadu",
    "thanjavur": "Tamil Nadu",
    "vellore": "Tamil Nadu",
    "thoothukudi": "Tamil Nadu",
    "tiruvallur-602001": "Tamil Nadu",
    "tiruvallur": "Tamil Nadu",
    "tiruppur": "Tamil Nadu",
    "madurai": "Tamil Nadu",
    # --- Telangana ---
    "telangana": "Telangana",
    "telangana state": "Telangana",
    "hyderabad": "Telangana",
    "secunderabad": "Telangana",
    "karimnagar": "Telangana",
    "mandamarri": "Telangana",
    "warangal": "Telangana",
    # --- Tripura ---
    "tripura": "Tripura",
    "west tripura": "Tripura",
    "agartala": "Tripura",
    # --- Uttar Pradesh ---
    "uttar pradesh": "Uttar Pradesh",
    "lucknow": "Uttar Pradesh",
    "varanasi": "Uttar Pradesh",
    "allahabad": "Uttar Pradesh",
    "prayagraj": "Uttar Pradesh",
    "azamgarh": "Uttar Pradesh",
    "ghaziabad": "Uttar Pradesh",
    "gautam buddha nagar": "Uttar Pradesh",
    "noida": "Uttar Pradesh",
    "moradabad": "Uttar Pradesh",
    "kalyanpur kanpur": "Uttar Pradesh",
    "kanpur": "Uttar Pradesh",
    "ambedkar nagar": "Uttar Pradesh",
    "aligarh": "Uttar Pradesh",
    "faizabad": "Uttar Pradesh",
    "ayodhya": "Uttar Pradesh",
    "agra": "Uttar Pradesh",
    "meerut": "Uttar Pradesh",
    "u.p.": "Uttar Pradesh",
    # --- Uttarakhand ---
    "uttarakhand": "Uttarakhand",
    "uttaranchal": "Uttarakhand",
    "dehradun": "Uttarakhand",
    "mukteshwar": "Uttarakhand",
    "haridwar": "Uttarakhand",
    # --- West Bengal ---
    "west bengal": "West Bengal",
    "kolkata": "West Bengal",
    "hooghly": "West Bengal",
    "howrah": "West Bengal",
    "birbhum": "West Bengal",
    "murshidabad": "West Bengal",
    "north 24 parganas": "West Bengal",
    "kharagpur": "West Bengal",
    "paschim medinipur": "West Bengal",
    "puruliya": "West Bengal",
    "alipurduar": "West Bengal",
    "chakdah": "West Bengal",
    "dinajpur": "West Bengal",
    "khaira": "West Bengal",
    "rajarhat": "West Bengal",
    "durgapur": "West Bengal",
    "durgapura": "West Bengal",
    # --- Delhi ---
    "delhi": "Delhi",
    "new delhi": "Delhi",
    "delhi division": "Delhi",
    "delhi ncr": "Delhi",
    "ncr": "Delhi",
    "nct": "Delhi",
    "national capital territory of delhi": "Delhi",
    "west delhi": "Delhi",
    "north west delhi": "Delhi",
    "safdarjung enclave": "Delhi",
    # --- Jammu & Kashmir ---
    "jammu & kashmir": "Jammu & Kashmir",
    "jammu and kashmir": "Jammu & Kashmir",
    "j&k": "Jammu & Kashmir",
    "ganderbal": "Jammu & Kashmir",
    "anantnag": "Jammu & Kashmir",
    "kupwara": "Jammu & Kashmir",
    "srinagar": "Jammu & Kashmir",
    # --- Chandigarh ---
    "chandigarh": "Chandigarh",
    # --- Puducherry ---
    "puducherry": "Puducherry",
    "pondicherry": "Puducherry",
    # --- Dadra & Nagar Haveli and Daman & Diu ---
    "dadra & nagar haveli and daman & diu": "Dadra & Nagar Haveli and Daman & Diu",
    "daman and diu": "Dadra & Nagar Haveli and Daman & Diu",
    "daman & diu": "Dadra & Nagar Haveli and Daman & Diu",
    "ut of dadra & nagar haveli and daman diu": "Dadra & Nagar Haveli and Daman & Diu",
    # --- Andaman & Nicobar ---
    "andaman and nicobar islands": "Andaman & Nicobar Islands",
    "andaman & nicobar islands": "Andaman & Nicobar Islands",
    # Catch-all abbreviations also here for safety
    "gj": "Gujarat",
    "ka": "Karnataka",
    "mh": "Maharashtra",
}

# Substring fragments that unambiguously identify a state (for fallback)
_SUBSTR_TO_STATE: list[tuple[str, str]] = [
    ("andhra pradesh", "Andhra Pradesh"),
    ("arunachal", "Arunachal Pradesh"),
    ("chhattisgarh", "Chhattisgarh"),
    ("chattisgarh", "Chhattisgarh"),
    ("himachal", "Himachal Pradesh"),
    ("jammu", "Jammu & Kashmir"),
    ("kashmir", "Jammu & Kashmir"),
    ("jharkhand", "Jharkhand"),
    ("karnataka", "Karnataka"),
    ("madhya pradesh", "Madhya Pradesh"),
    ("madhyapradesh", "Madhya Pradesh"),
    ("maharashtra", "Maharashtra"),
    ("meghalaya", "Meghalaya"),
    ("mizoram", "Mizoram"),
    ("nagaland", "Nagaland"),
    ("rajasthan", "Rajasthan"),
    ("tamil nadu", "Tamil Nadu"),
    ("tamilnadu", "Tamil Nadu"),
    ("telangana", "Telangana"),
    ("uttarakhand", "Uttarakhand"),
    ("uttaranchal", "Uttarakhand"),
    ("uttar pradesh", "Uttar Pradesh"),
    ("west bengal", "West Bengal"),
    ("assam", "Assam"),
    ("bihar", "Bihar"),
    ("gujarat", "Gujarat"),
    ("haryana", "Haryana"),
    ("kerala", "Kerala"),
    ("manipur", "Manipur"),
    ("odisha", "Odisha"),
    ("punjab", "Punjab"),
    ("sikkim", "Sikkim"),
    ("tripura", "Tripura"),
    ("ladakh", "Ladakh"),
    ("delhi", "Delhi"),
    ("goa", "Goa"),
    ("chandigarh", "Chandigarh"),
    ("puducherry", "Puducherry"),
    ("pondicherry", "Puducherry"),
    ("lakshadweep", "Lakshadweep"),
    ("andaman", "Andaman & Nicobar Islands"),
]


def normalize_state(raw: Any) -> str:
    """Map a messy state/city/abbreviation string to one of India's 36 official
    states/UTs. Returns 'Unknown' when no match is found."""
    if raw is None:
        return "Unknown"
    try:
        import pandas as _pd
        if _pd.isna(raw):
            return "Unknown"
    except (TypeError, ValueError):
        pass
    s = str(raw).strip()
    if not s or s.lower() in {"", "nan", "none", "null"}:
        return "Unknown"

    lower = s.lower()

    # 1. Exact variant match
    if lower in _VARIANTS:
        return _VARIANTS[lower]

    # 2. Short abbreviation (≤3 chars)
    if len(lower) <= 3 and lower in _ABBREV:
        return _ABBREV[lower]

    # 3. "City, State" or "City State" compound — try the last comma-separated part
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        for part in reversed(parts):
            candidate = normalize_state(part)
            if candidate != "Unknown":
                return candidate

    # 4. Substring scan (catches misspellings that still contain the state name)
    for fragment, canonical in _SUBSTR_TO_STATE:
        if fragment in lower:
            return canonical

    return "Unknown"


# ---------------------------------------------------------------------------
# Capability extraction
# ---------------------------------------------------------------------------

def _active_caps(capability_claims_json: Any) -> set[str]:
    """Return the set of confirmed/inferred capabilities for one facility."""
    try:
        claims = json.loads(capability_claims_json) if isinstance(capability_claims_json, str) else capability_claims_json
        if not isinstance(claims, list):
            return set()
        return {c["capability"] for c in claims if c.get("status") in {"confirmed", "inferred"}}
    except (json.JSONDecodeError, TypeError, KeyError):
        return set()


# ---------------------------------------------------------------------------
# Population lookup
# ---------------------------------------------------------------------------

_POPULATION_LOWER = {k.lower(): v for k, v in DISTRICT_POPULATIONS.items()}


def _population(district: str) -> int:
    return _POPULATION_LOWER.get(district.lower(), DEFAULT_POPULATION)


# ---------------------------------------------------------------------------
# Desert score
# ---------------------------------------------------------------------------

def _desert_score(trustworthy_count: int, population: int) -> float:
    return min(100.0, (trustworthy_count / max(1, population / 100_000)) * 20)


# ---------------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------------

def _make_facility_id(name: Any, city: Any) -> str:
    n = "" if (name is None or str(name).lower() in {"nan", "none"}) else str(name)
    c = "" if (city is None or str(city).lower() in {"nan", "none"}) else str(city)
    slug = re.sub(r"[^a-z0-9]+", "-", f"{n} {c}".lower())
    return slug.strip("-")


def _compute_trust_score(row: pd.Series) -> float:
    """Recompute the overall trust score using trust_score.py functions."""
    ic = internal_consistency_score(row.to_dict())
    cp = capability_plausibility_score(row.to_dict(), row)
    act = activity_signal_score(row)
    comp = completeness_score(row)
    return compute_overall({
        "internal_consistency": ic,
        "capability_plausibility": cp,
        "activity_signal": act,
        "completeness": comp,
    })


def _build_districts(merged: pd.DataFrame) -> pd.DataFrame:
    # Recompute trust scores (LLM-assigned scores are all 50; use deterministic functions)
    print("  Computing trust scores for all facilities...")
    merged["computed_trust_score"] = merged.apply(_compute_trust_score, axis=1)

    # Add indicator column for each capability
    for cap in ALL_CAPABILITIES:
        col = f"cap_{cap}"
        merged[col] = merged["active_caps"].apply(lambda caps: int(cap in caps))

    merged["is_trustworthy"] = (merged["computed_trust_score"] >= 60).astype(int)

    # Group by (state_clean, district)
    cap_cols = [f"cap_{c}" for c in ALL_CAPABILITIES]
    agg: dict[str, Any] = {
        "computed_trust_score": ["count", "mean"],
        "is_trustworthy": "sum",
    }
    for col in cap_cols:
        agg[col] = "sum"

    grouped = merged.groupby(["state_clean", "district"], as_index=False).agg(agg)

    # Flatten multi-level columns
    grouped.columns = ["state_clean", "district",
                       "total_facilities", "avg_trust_score", "trustworthy_count"] + ALL_CAPABILITIES

    grouped["avg_trust_score"] = grouped["avg_trust_score"].round(1)

    # Population & desert score
    grouped["population"] = grouped["district"].apply(_population)
    grouped["desert_score"] = grouped.apply(
        lambda r: round(_desert_score(r["trustworthy_count"], r["population"]), 1), axis=1
    )

    # top_gaps: 3 capabilities with lowest count
    def _top_gaps(row: pd.Series) -> str:
        cap_counts = {cap: row[cap] for cap in ALL_CAPABILITIES}
        sorted_caps = sorted(cap_counts.items(), key=lambda x: (x[1], x[0]))
        return json.dumps([c for c, _ in sorted_caps[:3]])

    grouped["top_gaps"] = grouped.apply(_top_gaps, axis=1)

    # Rename capability columns to cap_ prefix for parquet clarity
    rename = {cap: f"cap_{cap}" for cap in ALL_CAPABILITIES}
    grouped.rename(columns=rename, inplace=True)

    return grouped.sort_values(["state_clean", "district"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _print_section(title: str, df: pd.DataFrame, cols: list[str]) -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 30)
    pd.set_option("display.float_format", "{:.1f}".format)
    print(f"\n{'-' * 70}")
    print(title)
    print("-" * 70)
    print(df[cols].to_string(index=False))


def _worst_cap_deserts(df: pd.DataFrame, cap: str, n: int = 5) -> pd.DataFrame:
    col = f"cap_{cap}"
    return (
        df[df["total_facilities"] >= 3]
        .sort_values(col)
        .head(n)
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    assessments = pd.read_parquet(ASSESSMENTS_PATH)
    facilities = pd.read_parquet(FACILITIES_PATH)

    facilities["facility_id"] = facilities.apply(
        lambda r: _make_facility_id(r.get("name"), r.get("address_city")), axis=1
    )

    merged = assessments.merge(facilities, on="facility_id", how="inner", suffixes=("_asmt", "_raw"))
    print(f"Loaded: {len(assessments):,} assessments | {len(facilities):,} facilities | {len(merged):,} joined")

    # Normalize state (prefer assessments' state field over facilities'; both are equally messy)
    merged["state_clean"] = merged["state"].apply(normalize_state)
    merged["district"] = merged["city"]

    # Drop rows with no usable state/district
    before = len(merged)
    merged = merged[merged["state_clean"] != "Unknown"].copy()
    skipped = before - len(merged)
    if skipped:
        print(f"  Skipped {skipped} rows with unresolvable state -> 'Unknown'")

    # Extract active capabilities per row
    merged["active_caps"] = merged["capability_claims"].apply(_active_caps)

    # Build district table
    districts = _build_districts(merged)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    districts.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved -> {OUTPUT_PATH}  ({len(districts):,} districts)")

    # ── Summary stats ────────────────────────────────────────────────────
    unique_states = districts["state_clean"].nunique()
    unique_districts = len(districts)
    print(f"\nUnique districts : {unique_districts:,}")
    print(f"Unique states    : {unique_states}  (official count: 36)")

    base_cols = ["district", "state_clean", "desert_score", "total_facilities", "population", "avg_trust_score"]

    # ── Top 10 worst (lowest desert_score) ───────────────────────────────
    worst = districts.nsmallest(10, "desert_score")
    _print_section("Top 10 WORST Healthcare Deserts (lowest desert_score = fewest trustworthy facilities per capita)", worst, base_cols)

    # ── Top 10 best ───────────────────────────────────────────────────────
    best = districts.nlargest(10, "desert_score")
    _print_section("Top 10 BEST Healthcare Coverage (highest desert_score)", best, base_cols)

    # ── Specific capability deserts ───────────────────────────────────────
    for cap in ("dialysis", "emergency", "oncology"):
        cap_col = f"cap_{cap}"
        worst_cap = _worst_cap_deserts(districts, cap)
        cols = ["district", "state_clean", cap_col, "total_facilities", "desert_score"]
        _print_section(f"Top 5 Worst {cap.upper()} Deserts (fewest confirmed/inferred facilities, min 3 total facilities in district)", worst_cap, cols)


if __name__ == "__main__":
    main()
