from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/facilities_clean.parquet")
CONTRADICTIONS_PATH = Path("demo/planted_contradictions.json")
PRIORITY_OUTPUT_PATH = Path("data/priority_rows.parquet")
NON_PRIORITY_OUTPUT_PATH = Path("data/non_priority_rows.parquet")
TARGET_ROWS = 5_000
COMPLETENESS_ROWS = 1_500
RANDOM_STATE = 42

DEMO_FACILITIES = [
    ("1000 Smiles Dental Clinic", "Hyderabad"),
    ("Krishna Homeopathy Research Hospital", "Jaipur"),
    ("Aastha Children Hospital", "Dehri"),
    ("City Health Clinic", "Guwahati"),
    ("7 Star Healthcare", "Delhi"),
]

CRITICAL_FIELDS = [
    "specialties",
    "procedure",
    "equipment",
    "capability",
    "description",
    "numberdoctors",
    "capacity",
]


def text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def norm(value) -> str:
    return text(value).casefold()


def list_len(value) -> int:
    if value is None:
        return 0
    try:
        if pd.isna(value):
            return 0
    except (TypeError, ValueError):
        pass
    if isinstance(value, list):
        return len([item for item in value if text(item)])
    if isinstance(value, tuple):
        return len([item for item in value if text(item)])
    if hasattr(value, "tolist"):
        return len([item for item in value.tolist() if text(item)])
    return 1 if text(value) else 0


def is_present(value) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, (list, tuple)) or hasattr(value, "tolist"):
        return list_len(value) > 0
    return bool(text(value))


def add_source(selection_source: pd.Series, indexes: list[int], source: str) -> int:
    added = 0
    for idx in indexes:
        if selection_source.at[idx] is None:
            selection_source.at[idx] = source
            added += 1
    return added


def find_demo_indexes(df: pd.DataFrame) -> tuple[list[int], list[tuple[str, str, bool]]]:
    found_indexes: list[int] = []
    statuses: list[tuple[str, str, bool]] = []

    for name_fragment, city in DEMO_FACILITIES:
        mask = (
            df["name"].map(norm).str.contains(name_fragment.casefold(), regex=False, na=False)
            & (df["address_city"].map(norm) == city.casefold())
        )
        matches = df[mask].copy()

        if matches.empty:
            statuses.append((name_fragment, city, False))
            continue

        if name_fragment == "City Health Clinic":
            preferred = matches[
                matches["name"].map(norm).str.contains("& diagnostic", regex=False, na=False)
            ]
            if not preferred.empty:
                matches = preferred

        found_indexes.append(int(matches.index[0]))
        statuses.append((name_fragment, city, True))

    return found_indexes, statuses


def contradiction_indexes(df: pd.DataFrame) -> list[int]:
    with CONTRADICTIONS_PATH.open("r", encoding="utf-8") as handle:
        contradiction_data = json.load(handle)

    name_city_pairs: set[tuple[str, str]] = set()
    for entries in contradiction_data.values():
        for entry in entries:
            name_city_pairs.add((norm(entry.get("name")), norm(entry.get("city"))))

    indexes: list[int] = []
    for name, city in sorted(name_city_pairs):
        matches = df[(df["name"].map(norm) == name) & (df["address_city"].map(norm) == city)]
        indexes.extend(int(idx) for idx in matches.index)

    return indexes


def completeness_score(row: pd.Series) -> int:
    return sum(1 for field in CRITICAL_FIELDS if field in row.index and is_present(row[field]))


def stratified_sample_indexes(df: pd.DataFrame, needed: int) -> list[int]:
    if needed <= 0 or df.empty:
        return []

    if needed >= len(df):
        return [int(idx) for idx in df.index]

    states = df["address_stateorregion"].map(lambda value: text(value) or "Unknown")
    counts = states.value_counts().sort_index()
    exact = counts / counts.sum() * needed
    allocation = exact.apply(int)
    allocation = allocation.clip(upper=counts)

    while int(allocation.sum()) < needed:
        remaining_capacity = counts - allocation
        candidates = remaining_capacity[remaining_capacity > 0].index
        fractional = (exact - allocation).loc[candidates].sort_values(ascending=False)
        if fractional.empty:
            break
        allocation.loc[fractional.index[0]] += 1

    sampled_indexes: list[int] = []
    for state, sample_size in allocation.items():
        sample_size = int(sample_size)
        if sample_size <= 0:
            continue
        state_indexes = states[states == state].index
        sampled = df.loc[state_indexes].sample(n=sample_size, random_state=RANDOM_STATE)
        sampled_indexes.extend(int(idx) for idx in sampled.index)

    if len(sampled_indexes) < needed:
        already = set(sampled_indexes)
        extra_pool = df.loc[[idx for idx in df.index if idx not in already]]
        extra = extra_pool.sample(n=needed - len(sampled_indexes), random_state=RANDOM_STATE)
        sampled_indexes.extend(int(idx) for idx in extra.index)

    return sampled_indexes[:needed]


def main() -> None:
    df = pd.read_parquet(INPUT_PATH).reset_index(drop=True)
    selection_source = pd.Series([None] * len(df), index=df.index, dtype=object)

    demo_indexes, demo_statuses = find_demo_indexes(df)
    add_source(selection_source, demo_indexes, "demo")

    print("Demo facility lookup:")
    for name, city, found in demo_statuses:
        status = "FOUND" if found else "WARNING: NOT FOUND"
        print(f"  {status}: {name} in {city}")

    contradiction_count = add_source(selection_source, contradiction_indexes(df), "contradictions")

    remaining = df[selection_source.isna()].copy()
    remaining["_completeness_score"] = remaining.apply(completeness_score, axis=1)
    completeness_indexes = (
        remaining.sort_values(["_completeness_score", "name", "address_city"], ascending=[False, True, True])
        .head(COMPLETENESS_ROWS)
        .index
        .tolist()
    )
    completeness_count = add_source(selection_source, completeness_indexes, "completeness")

    needed_random = TARGET_ROWS - int(selection_source.notna().sum())
    remaining = df[selection_source.isna()]
    random_indexes = stratified_sample_indexes(remaining, needed_random)
    random_count = add_source(selection_source, random_indexes, "random")

    selected_mask = selection_source.notna()
    selected = df[selected_mask].copy()
    non_priority = df[~selected_mask].copy()

    selected.to_parquet(PRIORITY_OUTPUT_PATH, index=False)
    non_priority.to_parquet(NON_PRIORITY_OUTPUT_PATH, index=False)

    source_counts = selection_source[selected_mask].value_counts().reindex(
        ["demo", "contradictions", "completeness", "random"], fill_value=0
    )
    present_demo_names = []
    for name_fragment, city in DEMO_FACILITIES:
        mask = (
            selected["name"].map(norm).str.contains(name_fragment.casefold(), regex=False, na=False)
            & (selected["address_city"].map(norm) == city.casefold())
        )
        if mask.any():
            present_demo_names.append(f"{name_fragment} in {city}")

    state_counts = selected["address_stateorregion"].map(lambda value: text(value) or "Unknown").value_counts()

    print("\nPriority selection summary:")
    print(f"  Total selected: {len(selected):,}")
    print("\nBreakdown by source:")
    for source, count in source_counts.items():
        print(f"  {source}: {int(count):,}")
    print(f"\n  Contradiction rows newly added after demo de-dupe: {contradiction_count:,}")
    print(f"  Completeness rows added: {completeness_count:,}")
    print(f"  Random rows added: {random_count:,}")

    print("\nDemo facilities present in priority_rows.parquet:")
    for item in present_demo_names:
        print(f"  {item}")
    missing_demo_count = len(DEMO_FACILITIES) - len(present_demo_names)
    if missing_demo_count:
        print(f"  WARNING: {missing_demo_count} demo facilities missing from selected output")

    print(f"\nUnique states represented: {state_counts.size:,}")
    print("\nTop 10 most-represented states:")
    for state, count in state_counts.head(10).items():
        print(f"  {state}: {int(count):,}")

    print("\nOutput files:")
    print(f"  {PRIORITY_OUTPUT_PATH.as_posix()} ({len(selected):,} rows)")
    print(f"  {NON_PRIORITY_OUTPUT_PATH.as_posix()} ({len(non_priority):,} rows)")

    if len(selected) != TARGET_ROWS:
        print(f"\nWARNING: Expected {TARGET_ROWS:,} selected rows, got {len(selected):,}")


if __name__ == "__main__":
    main()
