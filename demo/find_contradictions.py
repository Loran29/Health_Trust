import json
import re
from pathlib import Path

import pandas as pd


DATASET = Path("data/facilities_clean.parquet")
OUTPUT = Path("demo/planted_contradictions.json")
ARRAY_COLS = {
    "phone_numbers",
    "websites",
    "affiliationtypeids",
    "specialties",
    "procedure",
    "equipment",
    "capability",
}


def as_list(value):
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    if isinstance(value, list):
        return [str(v) for v in value if v is not None and str(v).strip()]
    if isinstance(value, tuple):
        return [str(v) for v in value if v is not None and str(v).strip()]
    if hasattr(value, "tolist"):
        return [str(v) for v in value.tolist() if v is not None and str(v).strip()]
    text = str(value).strip()
    return [] if text.lower() in {"", "nan", "none", "null", "<na>"} else [text]


def as_text(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def is_missing(value):
    return as_text(value).strip().lower() in {"", "nan", "none", "null", "<na>", "nat"}


def list_contains(items, terms):
    haystack = " | ".join(as_list(items)).lower()
    return any(term.lower() in haystack for term in terms)


def list_empty(items):
    return len(as_list(items)) == 0


DIRECTORY_OR_SOCIAL_DOMAINS = [
    "facebook.",
    "instagram.",
    "linkedin.",
    "twitter.",
    "x.com",
    "justdial.",
    "practo.",
    "indiamart.",
    "sulekha.",
    "quickerala.",
    "lybrate.",
    "credihealth.",
    "localo.site",
    "grotal.",
    "asklaila.",
]


def has_no_owned_website(row):
    if not is_missing(row.get("officialwebsite")):
        return False
    websites = as_list(row.get("websites"))
    if not websites:
        return True
    for website in websites:
        site = website.lower()
        if not any(domain in site for domain in DIRECTORY_OR_SOCIAL_DOMAINS):
            return False
    return True


def bool_false(value):
    if is_missing(value):
        return False
    return as_text(value).strip().lower() in {"0", "0.0", "false", "no"}


def n_float(value):
    if is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def row_view(row):
    return {
        "name": as_text(row.get("name")),
        "city": as_text(row.get("address_city")),
        "type": as_text(row.get("facilitytypeid")),
        "specialties": as_list(row.get("specialties")),
        "equipment": as_list(row.get("equipment")),
        "capability": as_list(row.get("capability")),
    }


SPECIALTY_GROUPS = {
    "mental_health": ["psychiatry", "psychology", "addiction"],
    "bone_muscle": ["orthopedic", "orthopedics", "sportsmedicine", "rheumatology"],
    "heart": ["cardiology", "cardiac", "cardiothoracic"],
    "dental": ["dentistry", "periodontics", "endodontics", "orthodontics"],
    "eye": ["ophthalmology", "retina", "glaucoma", "cataract"],
    "women_fertility": ["gynecology", "obstetrics", "infertility", "reproductive"],
    "skin_hair": ["dermatology", "hairandnail"],
    "child": ["pediatrics", "neonatology"],
    "cancer": ["oncology"],
    "urinary": ["urology", "nephrology"],
    "surgery": ["surgery"],
    "general": ["familymedicine", "internalmedicine"],
}


def specialty_groups(specs):
    joined = " | ".join(as_list(specs)).lower()
    found = []
    for group, needles in SPECIALTY_GROUPS.items():
        if any(needle in joined for needle in needles):
            found.append(group)
    return found


def very_different(specs):
    spec_list = as_list(specs)
    found = specialty_groups(spec_list)
    return len(spec_list) >= 5 and len(found) >= 4


def main():
    df = pd.read_parquet(DATASET)

    q1 = df[
        df["facilitytypeid"].astype(str).str.lower().isin(["dentist", "clinic"])
        & df["specialties"].apply(
            lambda v: list_contains(v, ["familyMedicine", "internalMedicine", "cardiology"])
        )
    ].copy()

    q2 = df[df["specialties"].apply(very_different)].copy()
    q2["_group_count"] = q2["specialties"].apply(lambda v: len(specialty_groups(v)))
    q2["_spec_count"] = q2["specialties"].apply(lambda v: len(as_list(v)))
    q2 = q2.sort_values(["_group_count", "_spec_count"], ascending=False)

    q3 = df[
        df["description"].apply(
            lambda v: bool(re.search(r"\b(icu|emergency|surgery)\b", as_text(v), re.I))
        )
        & df["equipment"].apply(list_empty)
    ].copy()

    q4 = df[
        df["engagement_metrics_n_followers"].apply(
            lambda v: (n_float(v) is not None and n_float(v) < 50)
        )
        & df["custom_logo_presence"].apply(bool_false)
        & df.apply(has_no_owned_website, axis=1)
    ].copy()

    q5 = df[
        df["capability"].apply(lambda v: list_contains(v, ["24/7", "24x7", "advanced"]))
        & df["numberdoctors"].apply(lambda v: is_missing(v) or n_float(v) == 1)
    ].copy()

    queries = [
        (
            "type_specialty_mismatch",
            q1,
            "Dentist/clinic claims broad medical specialties such as family medicine, internal medicine, or cardiology.",
        ),
        (
            "specialty_sprawl",
            q2,
            "Facility lists five or more specialties spanning several unrelated clinical families.",
        ),
        (
            "serious_description_no_equipment",
            q3,
            "Description mentions ICU, emergency, or surgery, but no equipment is listed.",
        ),
        (
            "low_social_no_brand_or_site",
            q4,
            "Low follower count, no custom logo signal, and no owned website make the facility hard to verify.",
        ),
        (
            "big_capability_tiny_staff",
            q5,
            "Capability claims 24/7 or advanced services, but doctor count is missing or only one.",
        ),
    ]

    demo = {}
    for key, query_df, note in queries:
        entries = []
        for _, row in query_df.head(3).iterrows():
            item = row_view(row)
            item["note"] = note
            entries.append(item)
        demo[key] = entries

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(demo, indent=2, ensure_ascii=False), encoding="utf-8")

    for title, query_df, _ in queries:
        print(f"\n=== {title} | matches: {len(query_df)} | showing 5 ===")
        for i, (_, row) in enumerate(query_df.head(5).iterrows(), start=1):
            item = row_view(row)
            print(f"[{i}] name: {item['name']}")
            print(f"    city: {item['city']}")
            print(f"    type: {item['type']}")
            print(f"    specialties: {item['specialties']}")
            print(f"    equipment: {item['equipment']}")
            print(f"    capability: {item['capability']}")

    print(f"\nSAVED {OUTPUT.as_posix()}")


if __name__ == "__main__":
    main()
