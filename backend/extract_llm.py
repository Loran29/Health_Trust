from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import tiktoken
from dotenv import load_dotenv
from pydantic import ValidationError
from tqdm.asyncio import tqdm_asyncio

try:
    from backend.schemas import (
        Capability,
        CapabilityClaim,
        CapabilityStatus,
        Contradiction,
        ContradictionType,
        FacilityAssessment,
        TrustSubscores,
    )
except ImportError:
    from schemas import (
        Capability,
        CapabilityClaim,
        CapabilityStatus,
        Contradiction,
        ContradictionType,
        FacilityAssessment,
        TrustSubscores,
    )


API_URL = "https://api.openai.com/v1/chat/completions"
MODEL = "gpt-4o-mini"
INPUT_COST_PER_1M = 0.15
OUTPUT_COST_PER_1M = 0.60
PRIORITY_ROWS_PATH = Path("data/priority_rows.parquet")
FAILURE_LOG_PATH = Path("backend/data/extraction_failures.log")
RANDOM_STATE = 42

SCHEMA_RULE_BLOCK = """CRITICAL JSON SCHEMA RULES:
- capability_claims is a JSON array. Each element is an object with EXACTLY these keys: "capability", "status", "evidence_field", "evidence_snippet". No other keys.
- status MUST be one of exactly these strings: "confirmed", "inferred", "contradicted", "unknown". Never put a contradiction type here.
- capability MUST be one of exactly these strings (any other value will be rejected): emergency, icu, surgery, obstetrics, dialysis, oncology, cardiology, anesthesia, pediatrics, mental_health, dentistry, primary_care, ophthalmology, orthopedics, dermatology, runs_24_7
- contradictions is a JSON array of objects with EXACTLY these keys: "contradiction_type", "field_name", "claim", "why_contradictory", "severity".
- contradiction_type MUST be one of: type_specialty_mismatch, specialty_sprawl, missing_equipment, missing_staff, missing_brand, capability_overreach, other
- severity is an integer 1-5.
- confidence_interval is a 2-element array of integers [low, high] where 0 <= low <= high <= 100.
- trust_subscores is an object with EXACTLY these 4 integer keys: internal_consistency, capability_plausibility, activity_signal, completeness — each 0-100.

EVIDENCE SNIPPET RULES:
- evidence_snippet is the literal CONTENT from the field, not the field name or its rendering.
- Do NOT prefix with "specialties:" or "[...]". Quote the relevant text directly.
- Good: "familyMedicine, periodontics, dentistry"
- Bad: "specialties: [\\"familyMedicine\\", \\"periodontics\\"]"
- Max 200 chars.

EXTRACTION RICHNESS:
- For facilities with many listed specialties, output one capability_claim per relevant Capability enum value that is implied or contradicted. Do not collapse multiple specialties into a single primary_care claim.
- When a specialty maps to a Capability and the facility cannot plausibly support it (no equipment, no staff, wrong facility type), prefer status "contradicted" over "inferred".
- Aim for 3-8 capability_claims for facilities with rich specialty lists."""

SYSTEM_PROMPT = """You are an expert auditor of Indian healthcare facility data. Your job is to assess each facility for what it can actually deliver versus what it claims.

Known issues in this dataset:
- Roughly 63% of rows have type-specialty mismatches (e.g. dental clinics tagged with "familyMedicine")
- Many facilities claim 5+ unrelated specialties without supporting infrastructure
- Equipment lists are often empty even when claimed capabilities require equipment
- Staff fields (numberDoctors, capacity) are 90%+ null — treat their absence as missing data, not as zero
- Description fields can contain marketing language ("advanced", "24/7", "world-class") without supporting evidence

For each facility, return a FacilityAssessment as JSON.

CAPABILITY CLAIMS — for each capability the facility claims or implies, set status to:
- "confirmed": explicitly evidenced in the equipment, procedure, or capability fields
- "inferred": reasonably implied by specialties or description but not directly evidenced
- "contradicted": claimed in specialties but actively undermined by other fields (e.g. cardiology claimed by a 1-doctor homeopathy clinic with empty equipment)
- "unknown": no signal either way

CRITICAL — capability values MUST be EXACTLY one of these strings (no other values allowed):
emergency, icu, surgery, obstetrics, dialysis, oncology, cardiology, anesthesia, pediatrics, mental_health, dentistry, primary_care, ophthalmology, orthopedics, dermatology, runs_24_7

Mapping common dataset specialty names to our capability values:
- "familyMedicine", "internalMedicine", "generalPractice" → primary_care
- "physiotherapy", "physicalTherapy", "rehabilitation" → orthopedics
- "diabetes", "endocrinology" → primary_care
- "speechTherapy", "audiology", "hearing" → primary_care
- "psychiatry", "psychology" → mental_health
- "neurology", "neurosurgery" → surgery (if surgical context) or primary_care
- "ent", "otolaryngology" → primary_care
- "ayurveda", "homeopathy", "naturopathy" → primary_care
- "infertility", "ivf", "fertility" → obstetrics
- "radiology", "imaging" → primary_care

If a specialty has no clear mapping, do NOT include a capability_claim for it. Better to omit than to invent a value.

Always include evidence_field (which input field justified the status, e.g. "specialties" or "equipment" or "description") and evidence_snippet quoting actual text from that field (max 200 chars). Do not paraphrase the snippet — quote it directly from the input.

CONTRADICTIONS — you MUST actively find and report contradictions. A facility with empty equipment claiming high-acuity capabilities (NICU, PICU, surgery, ICU, oncology) ALWAYS gets a missing_equipment contradiction. A facility claiming "24/7" without supporting staff or equipment ALWAYS gets a capability_overreach contradiction. Do not be lenient — flag every issue.

Look for and report ALL that apply:
1. type_specialty_mismatch — clinic/dentist type claiming family medicine, internal medicine, or unrelated specialties (severity 2-3)
2. specialty_sprawl — 5+ unrelated specialties without supporting infrastructure (severity 4-5)
3. missing_equipment — high-acuity capability claimed (surgery, ICU, NICU, PICU, oncology, dialysis) with empty or null equipment field (severity 3-5)
4. missing_staff — advanced capabilities claimed AND numberDoctors is non-null AND ≤1 (severity 3-4). If numberDoctors is null, do NOT flag this.
5. capability_overreach — "24/7", "advanced", "emergency", "round the clock" claims without supporting staff or equipment (severity 2-4)

Each Contradiction needs: contradiction_type (from the 5 above or "other"), field_name (which input field), claim (what the facility claims), why_contradictory (one sentence), severity (1-5).

contradiction_type MUST be EXACTLY one of: type_specialty_mismatch, specialty_sprawl, missing_equipment, missing_staff, missing_brand, capability_overreach, other

SCORING — leave trust_subscores all at 50, overall_trust_score at 50, confidence_interval at [40, 60]. Scoring is computed in a separate pass.

trust_subscores MUST have EXACTLY these four integer keys, all set to 50:
{"internal_consistency": 50, "capability_plausibility": 50, "activity_signal": 50, "completeness": 50}

Do NOT use any other key names. Do NOT add extra fields.

REASONING SUMMARY — one punchy sentence under 200 chars summarizing the assessment.

evidence_snippet MUST be 200 characters or fewer. Truncate quoted text if needed and add "..." at the end.

OUTPUT — return ONLY valid JSON matching the FacilityAssessment schema. No markdown, no preamble. Required top-level keys: facility_id, facility_name, city, state, latitude, longitude, facility_type, capability_claims, contradictions, trust_subscores, overall_trust_score, confidence_interval, reasoning_summary.""" + "\n\n" + SCHEMA_RULE_BLOCK

DEMO_FACILITIES = [
    ("1000 Smiles Dental Clinic", "Hyderabad"),
    ("Krishna Homeopathy Research Hospital", "Jaipur"),
    ("Aastha Children Hospital", "Dehri"),
    ("City Health Clinic", "Guwahati"),
    ("7 Star Healthcare", "Delhi"),
]


def make_facility_id(name: str, city: str) -> str:
    slug_source = f"{name} {city}".lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug_source)
    return slug.strip("-")


def is_null(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"", "nan", "none", "null", "<na>", "nat"}


def to_list(value: Any) -> list[Any]:
    if is_null(value):
        return []
    if isinstance(value, list):
        return [item for item in value if not is_null(item)]
    if isinstance(value, tuple):
        return [item for item in value if not is_null(item)]
    if hasattr(value, "tolist"):
        return [item for item in value.tolist() if not is_null(item)]
    return [value]


def scalar_or_null(value: Any) -> str:
    if is_null(value):
        return "null"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def int_or_null(value: Any) -> str:
    if is_null(value):
        return "null"
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return "null"


def bool_or_null(value: Any) -> str:
    if is_null(value):
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    try:
        return "true" if float(value) != 0 else "false"
    except (TypeError, ValueError):
        text = str(value).strip().lower()
        if text in {"true", "yes", "y"}:
            return "true"
        if text in {"false", "no", "n"}:
            return "false"
    return "null"


def list_or_null(value: Any, empty_as_null: bool = True) -> str:
    values = to_list(value)
    if not values:
        return "(none)"
    return ", ".join(str(item) for item in values)


def log_validation_failure(row: pd.Series, attempt: str, error: Exception, raw_output: str) -> None:
    FAILURE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    name = scalar_or_null(row.get("name"))
    city = scalar_or_null(row.get("address_city"))
    with FAILURE_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write("=" * 100 + "\n")
        handle.write(f"facility: {name} | city: {city} | attempt: {attempt}\n")
        handle.write("validation_error:\n")
        handle.write(f"{error}\n")
        handle.write("raw_model_output:\n")
        handle.write(raw_output)
        handle.write("\n")


def build_user_prompt(row: pd.Series) -> str:
    name = scalar_or_null(row.get("name"))
    city = scalar_or_null(row.get("address_city"))
    facility_id = make_facility_id("" if name == "null" else name, "" if city == "null" else city)

    fields = [
        ("facility_id", facility_id),
        ("name", name),
        ("city", city),
        ("state", scalar_or_null(row.get("address_stateorregion"))),
        ("latitude", scalar_or_null(row.get("latitude"))),
        ("longitude", scalar_or_null(row.get("longitude"))),
        ("facility_type", scalar_or_null(row.get("facilitytypeid"))),
        ("specialties", list_or_null(row.get("specialties"), empty_as_null=False)),
        ("procedure", list_or_null(row.get("procedure"))),
        ("equipment", list_or_null(row.get("equipment"))),
        ("capability", list_or_null(row.get("capability"))),
        ("description", scalar_or_null(row.get("description"))),
        ("numberDoctors", scalar_or_null(row.get("numberdoctors"))),
        ("capacity", scalar_or_null(row.get("capacity"))),
        ("custom_logo_presence", bool_or_null(row.get("custom_logo_presence"))),
        (
            "distinct_social_media_presence_count",
            int_or_null(row.get("distinct_social_media_presence_count")),
        ),
        ("engagement_metrics_n_followers", int_or_null(row.get("engagement_metrics_n_followers"))),
        ("recency_of_page_update", scalar_or_null(row.get("recency_of_page_update"))),
    ]
    return "\n".join(f"{key}: {value}" for key, value in fields)


def usage_from_response(response_json: dict[str, Any]) -> tuple[int, int]:
    usage = response_json.get("usage") or {}
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


def response_content(response_json: dict[str, Any]) -> str:
    return response_json["choices"][0]["message"]["content"]


def fallback_assessment(row: pd.Series, error: Exception | str) -> FacilityAssessment:
    name = scalar_or_null(row.get("name"))
    city = scalar_or_null(row.get("address_city"))
    print(f"Extraction failed for {name}: {error}")
    return FacilityAssessment(
        facility_id=make_facility_id("" if name == "null" else name, "" if city == "null" else city),
        facility_name="" if name == "null" else name,
        city="" if city == "null" else city,
        state="" if is_null(row.get("address_stateorregion")) else str(row.get("address_stateorregion")),
        latitude=0.0 if is_null(row.get("latitude")) else float(row.get("latitude")),
        longitude=0.0 if is_null(row.get("longitude")) else float(row.get("longitude")),
        facility_type="" if is_null(row.get("facilitytypeid")) else str(row.get("facilitytypeid")),
        capability_claims=[
            CapabilityClaim(
                capability=Capability.PRIMARY_CARE,
                status=CapabilityStatus.UNKNOWN,
                evidence_field="extraction_failed",
                evidence_snippet="extraction_failed",
            )
        ],
        contradictions=[
            Contradiction(
                contradiction_type=ContradictionType.OTHER,
                field_name="extraction",
                claim="LLM extraction failed.",
                why_contradictory="The extractor could not produce valid schema-compliant JSON.",
                severity=1,
            )
        ],
        trust_subscores=TrustSubscores(
            internal_consistency=50,
            capability_plausibility=50,
            activity_signal=50,
            completeness=50,
        ),
        overall_trust_score=50,
        confidence_interval=(40, 60),
        reasoning_summary="Extraction failed — using fallback assessment",
    )


async def post_chat_completion(
    client: httpx.AsyncClient,
    api_key: str,
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    response = await client.post(
        API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": MODEL,
            "messages": messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "max_tokens": 3500,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


async def extract_one(
    client: httpx.AsyncClient,
    row: pd.Series,
    semaphore: asyncio.Semaphore,
) -> tuple[FacilityAssessment, dict[str, Any]]:
    async with semaphore:
        facility_name = scalar_or_null(row.get("name"))
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return fallback_assessment(row, "OPENAI_API_KEY is missing"), {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "validation_retries": 0,
                "facility_name": facility_name,
            }

        user_prompt = build_user_prompt(row)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        total_prompt_tokens = 0
        total_completion_tokens = 0

        try:
            response_json = await post_chat_completion(client, api_key, messages)
            prompt_tokens, completion_tokens = usage_from_response(response_json)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            content = response_content(response_json)
            try:
                return FacilityAssessment.model_validate_json(content), {
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "validation_retries": 0,
                    "facility_name": facility_name,
                }
            except (ValidationError, json.JSONDecodeError) as first_error:
                log_validation_failure(row, "first", first_error, content)
                messages.extend(
                    [
                        {"role": "assistant", "content": content},
                        {
                            "role": "user",
                            "content": (
                                "Your previous JSON failed validation:\n"
                                f"{first_error}\n\n"
                                "Common fixes:\n"
                                "- capability values must match the enum exactly "
                                "(lowercase, underscores, see system prompt)\n"
                                "- trust_subscores must have exactly the 4 keys listed "
                                "in the system prompt\n"
                                "- contradiction_type must match the enum exactly\n"
                                "- evidence_snippet must be ≤200 chars\n\n"
                                f"{SCHEMA_RULE_BLOCK}\n\n"
                                "Return ONLY corrected JSON. No markdown, no preamble."
                            ),
                        },
                    ]
                )
                retry_json = await post_chat_completion(client, api_key, messages)
                prompt_tokens, completion_tokens = usage_from_response(retry_json)
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                retry_content = response_content(retry_json)
                try:
                    return FacilityAssessment.model_validate_json(retry_content), {
                        "prompt_tokens": total_prompt_tokens,
                        "completion_tokens": total_completion_tokens,
                        "validation_retries": 1,
                        "facility_name": facility_name,
                    }
                except (ValidationError, json.JSONDecodeError) as second_error:
                    log_validation_failure(row, "retry", second_error, retry_content)
                    return fallback_assessment(row, second_error), {
                        "prompt_tokens": total_prompt_tokens,
                        "completion_tokens": total_completion_tokens,
                        "validation_retries": 1,
                        "facility_name": facility_name,
                    }
        except Exception as error:
            return fallback_assessment(row, error), {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "validation_retries": 0,
                "facility_name": facility_name,
            }


def find_demo_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name_fragment, city in DEMO_FACILITIES:
        mask = (
            df["name"].astype(str).str.casefold().str.contains(name_fragment.casefold(), regex=False)
            & (df["address_city"].astype(str).str.casefold() == city.casefold())
        )
        matches = df[mask]
        if name_fragment == "City Health Clinic" and not matches.empty:
            preferred = matches[
                matches["name"].astype(str).str.casefold().str.contains("& diagnostic", regex=False)
            ]
            if not preferred.empty:
                matches = preferred
        if matches.empty:
            print(f"WARNING: demo row not found: {name_fragment} in {city}")
        else:
            rows.append(matches.iloc[[0]])
    return pd.concat(rows) if rows else df.iloc[[]]


def select_test_rows(df: pd.DataFrame) -> pd.DataFrame:
    demo = find_demo_rows(df)
    demo_indexes = set(demo.index)
    remaining = df[~df.index.isin(demo_indexes)]
    random_rows = remaining.sample(n=15, random_state=RANDOM_STATE)
    return pd.concat([demo, random_rows]).reset_index(drop=True)


def contradiction_types(assessment: FacilityAssessment) -> str:
    types = [item.contradiction_type.value for item in assessment.contradictions]
    return ",".join(types) if types else "none"


def print_summary_table(assessments: list[FacilityAssessment]) -> None:
    rows = []
    for assessment in assessments:
        rows.append(
            {
                "facility_name": assessment.facility_name,
                "city": assessment.city,
                "num_capability_claims": len(assessment.capability_claims),
                "num_contradictions": len(assessment.contradictions),
                "contradiction_types_found": contradiction_types(assessment),
                "reasoning_summary": assessment.reasoning_summary,
            }
        )
    summary = pd.DataFrame(rows)
    print("\nSummary table:")
    print(summary.to_string(index=False))


def print_named_assessment(assessments: list[FacilityAssessment], name_fragment: str) -> None:
    for assessment in assessments:
        if name_fragment.casefold() in assessment.facility_name.casefold():
            print(f"\nFull assessment: {assessment.facility_name}")
            print(assessment.model_dump_json(indent=2))
            return
    print(f"\nWARNING: assessment not found for {name_fragment}")


def count_prompt_tokens_for_visibility(row: pd.Series) -> int:
    try:
        encoding = tiktoken.encoding_for_model(MODEL)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")
    return len(encoding.encode(SYSTEM_PROMPT + "\n" + build_user_prompt(row)))


async def run_test() -> None:
    load_dotenv()
    df = pd.read_parquet(PRIORITY_ROWS_PATH)
    test_rows = select_test_rows(df)
    semaphore = asyncio.Semaphore(10)

    estimated_prompt_tokens = sum(count_prompt_tokens_for_visibility(row) for _, row in test_rows.iterrows())
    print(f"Selected {len(test_rows)} rows for LLM extraction test.")
    print("Critical demo rows in test set:")
    for name_fragment, city in DEMO_FACILITIES:
        in_test = (
            test_rows["name"].astype(str).str.casefold().str.contains(name_fragment.casefold(), regex=False)
            & (test_rows["address_city"].astype(str).str.casefold() == city.casefold())
        ).any()
        status = "yes" if in_test else "NO"
        print(f"  {name_fragment} in {city}: {status}")
    print(f"Estimated local prompt tokens before API call: {estimated_prompt_tokens}")

    async with httpx.AsyncClient() as client:
        tasks = [extract_one(client, row, semaphore) for _, row in test_rows.iterrows()]
        results = await tqdm_asyncio.gather(*tasks, desc="Extracting")

    assessments = [assessment for assessment, _ in results]
    for assessment in assessments:
        FacilityAssessment.model_validate(assessment.model_dump())

    prompt_tokens = sum(usage["prompt_tokens"] for _, usage in results)
    completion_tokens = sum(usage["completion_tokens"] for _, usage in results)
    validation_retries = sum(int(usage.get("validation_retries", 0)) for _, usage in results)
    aastha_retries = sum(
        int(usage.get("validation_retries", 0))
        for _, usage in results
        if "aastha children hospital" in str(usage.get("facility_name", "")).casefold()
    )
    fallback_count = sum(
        1
        for assessment in assessments
        if any(claim.evidence_field == "extraction_failed" for claim in assessment.capability_claims)
    )
    total_cost = (prompt_tokens / 1_000_000 * INPUT_COST_PER_1M) + (
        completion_tokens / 1_000_000 * OUTPUT_COST_PER_1M
    )

    print_summary_table(assessments)
    print_named_assessment(assessments, "Krishna Homeopathy")
    print_named_assessment(assessments, "Aastha Children")
    print_named_assessment(assessments, "1000 Smiles")
    print("\nToken usage:")
    print(f"  prompt_tokens: {prompt_tokens}")
    print(f"  completion_tokens: {completion_tokens}")
    print(f"  total_tokens: {prompt_tokens + completion_tokens}")
    print(f"\nValidation retries attempted: {validation_retries}")
    print(f"Aastha Children Hospital validation retries: {aastha_retries}")
    print(f"\nFallback rows: {fallback_count}")
    print(f"\nTotal estimated cost for test: ${total_cost:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenAI LLM extraction for TrustMap facilities.")
    parser.add_argument("--test", action="store_true", help="Run the 20-row extraction quality test.")
    args = parser.parse_args()

    if not args.test:
        raise SystemExit("Only --test mode is implemented right now.")

    asyncio.run(run_test())


if __name__ == "__main__":
    main()
