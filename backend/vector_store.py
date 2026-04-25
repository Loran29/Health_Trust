from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import pandas as pd

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
FACILITIES_PATH = Path("data/facilities_clean.parquet")
CHROMA_PATH = Path("backend/data/chroma_db")
COLLECTION_NAME = "health_trust_facilities"
EMBED_BATCH_SIZE = 128


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _make_facility_id(name: Any, city: Any) -> str:
    n = "" if (name is None or str(name).lower() in {"nan", "none"}) else str(name)
    c = "" if (city is None or str(city).lower() in {"nan", "none"}) else str(city)
    slug = re.sub(r"[^a-z0-9]+", "-", f"{n} {c}".lower())
    return slug.strip("-")


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, list):
        return ", ".join(str(v) for v in value if v is not None)
    if hasattr(value, "tolist"):
        return ", ".join(str(v) for v in value.tolist() if v is not None)
    s = str(value).strip()
    return "" if s.lower() in {"nan", "none", "null", "<na>"} else s


def _build_document(row: pd.Series) -> str:
    """Concatenate all searchable text fields into one blob."""
    parts: list[str] = []
    for field in ("facility_name", "city", "state"):
        v = _to_text(row.get(field))
        if v:
            parts.append(v)
    for field in ("description", "specialties", "equipment", "procedure", "capability"):
        v = _to_text(row.get(field))
        if v:
            parts.append(v)
    return " | ".join(parts)


def _active_capabilities(capability_claims_json: Any) -> str:
    try:
        claims = (
            json.loads(capability_claims_json)
            if isinstance(capability_claims_json, str)
            else capability_claims_json
        )
        if not isinstance(claims, list):
            return ""
        return ", ".join(
            c["capability"] for c in claims if c.get("status") in {"confirmed", "inferred"}
        )
    except (json.JSONDecodeError, TypeError, KeyError):
        return ""


def _compute_trust_score(row: pd.Series) -> float:
    row_dict = row.to_dict()
    subscores = {
        "internal_consistency": internal_consistency_score(row_dict),
        "capability_plausibility": capability_plausibility_score(row_dict, row),
        "activity_signal": activity_signal_score(row),
        "completeness": completeness_score(row),
    }
    return round(compute_overall(subscores), 1)


# ---------------------------------------------------------------------------
# Collection builder
# ---------------------------------------------------------------------------

def build_collection(
    client: chromadb.ClientAPI,
    merged: pd.DataFrame,
) -> chromadb.Collection:
    ef = DefaultEmbeddingFunction()

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Cleared existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    # Deduplicate: keep first occurrence of each facility_id
    seen_ids: set[str] = set()
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []

    print(f"  Building documents and computing trust scores for {len(merged):,} rows...")
    for idx, (_, row) in enumerate(merged.iterrows()):
        fid = str(row.get("facility_id", "") or "")
        doc_id = fid if fid else f"anon_{idx}"

        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)

        doc = _build_document(row)
        if not doc.strip():
            continue

        trust = _compute_trust_score(row)
        caps = _active_capabilities(row.get("capability_claims", ""))

        documents.append(doc)
        metadatas.append(
            {
                "facility_id": fid,
                "state": _to_text(row.get("state")) or "Unknown",
                "district": _to_text(row.get("city")) or "Unknown",
                "trust_score": float(trust),
                "capabilities": caps,
            }
        )
        ids.append(doc_id)

    total = len(documents)
    print(f"  Embedding and indexing {total:,} documents (batch={EMBED_BATCH_SIZE})...")
    for start in range(0, total, EMBED_BATCH_SIZE):
        end = min(start + EMBED_BATCH_SIZE, total)
        collection.add(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        print(f"    {end:,} / {total:,} indexed", end="\r")
    print()

    return collection


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    assessments = pd.read_parquet(ASSESSMENTS_PATH)
    facilities = pd.read_parquet(FACILITIES_PATH)

    facilities["facility_id"] = facilities.apply(
        lambda r: _make_facility_id(r.get("name"), r.get("address_city")), axis=1
    )

    merged = assessments.merge(
        facilities, on="facility_id", how="inner", suffixes=("_asmt", "_raw")
    )
    print(
        f"Loaded: {len(assessments):,} assessments | "
        f"{len(facilities):,} facilities | "
        f"{len(merged):,} joined"
    )

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    collection = build_collection(client, merged)

    count = collection.count()
    storage = CHROMA_PATH.resolve()

    print(f"\nCollection  : {COLLECTION_NAME}")
    print(f"Documents   : {count:,}")
    print(f"Storage     : {storage}")
    print(f"Model       : all-MiniLM-L6-v2  (ONNX via chromadb, 384 dims)")

    # Test query
    query = "dialysis in rural Bihar"
    print(f"\nTest query  : '{query}'")
    print("-" * 60)
    results = collection.query(query_texts=[query], n_results=3)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
        name_part = doc.split(" | ")[0] if " | " in doc else doc[:50]
        print(f"\n  #{rank}  {name_part}")
        print(f"       district    : {meta.get('district')} / {meta.get('state')}")
        print(f"       trust_score : {meta.get('trust_score')}")
        print(f"       capabilities: {meta.get('capabilities') or '(none)'}")
        print(f"       distance    : {dist:.4f}")


if __name__ == "__main__":
    main()
