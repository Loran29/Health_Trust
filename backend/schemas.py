"""Pydantic contracts for TrustMap India's trust scoring and search pipeline.

The most reliable source fields from the current dataset are name, city, state,
type, specialties, procedure, equipment, capability, latitude, and longitude.
Sparse fields such as numberDoctors, capacity, and yearEstablished should be
treated as bonus signals rather than required evidence.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Capability(str, Enum):
    """Canonical healthcare capabilities that TrustMap can extract and score."""

    EMERGENCY = "emergency"
    ICU = "icu"
    SURGERY = "surgery"
    OBSTETRICS = "obstetrics"
    DIALYSIS = "dialysis"
    ONCOLOGY = "oncology"
    CARDIOLOGY = "cardiology"
    ANESTHESIA = "anesthesia"
    PEDIATRICS = "pediatrics"
    MENTAL_HEALTH = "mental_health"
    DENTISTRY = "dentistry"
    PRIMARY_CARE = "primary_care"
    OPHTHALMOLOGY = "ophthalmology"
    ORTHOPEDICS = "orthopedics"
    DERMATOLOGY = "dermatology"
    RUNS_24_7 = "runs_24_7"


class CapabilityStatus(str, Enum):
    """Validation state for a capability claim after evidence checks."""

    CONFIRMED = "confirmed"
    INFERRED = "inferred"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


class ContradictionType(str, Enum):
    """Standard contradiction categories used by scoring and demos."""

    TYPE_SPECIALTY_MISMATCH = "type_specialty_mismatch"
    SPECIALTY_SPRAWL = "specialty_sprawl"
    MISSING_EQUIPMENT = "missing_equipment"
    MISSING_STAFF = "missing_staff"
    MISSING_BRAND = "missing_brand"
    CAPABILITY_OVERREACH = "capability_overreach"
    OTHER = "other"


class Contradiction(BaseModel):
    """A single trust issue found in one facility record."""

    contradiction_type: ContradictionType = Field(
        description="Machine-readable category for the contradiction."
    )
    field_name: str = Field(description="Dataset field where the suspect claim appears.")
    claim: str = Field(description="What the facility claims or implies.")
    why_contradictory: str = Field(
        description="One sentence explaining why the claim looks inconsistent."
    )
    severity: int = Field(
        ge=1,
        le=5,
        description="Impact score from 1, low concern, to 5, high concern.",
    )


class CapabilityClaim(BaseModel):
    """A normalized capability with the evidence used to justify its status."""

    capability: Capability = Field(description="Canonical capability being assessed.")
    status: CapabilityStatus = Field(description="Evidence-backed status of the capability.")
    evidence_field: str = Field(
        description='Source field that justified the claim, such as "specialties" or "equipment".'
    )
    evidence_snippet: str = Field(
        max_length=200,
        description="Actual text from the source field, capped for API display.",
    )


class TrustSubscores(BaseModel):
    """Component scores that explain the overall facility trust score."""

    internal_consistency: int = Field(
        ge=0,
        le=100,
        description="How consistent the facility type, specialties, and claims are.",
    )
    capability_plausibility: int = Field(
        ge=0,
        le=100,
        description="How plausible the claimed capabilities are given supporting evidence.",
    )
    activity_signal: int = Field(
        ge=0,
        le=100,
        description="Strength of external activity signals such as website, social, and freshness.",
    )
    completeness: int = Field(
        ge=0,
        le=100,
        description="How complete the reliable core profile fields are.",
    )


class FacilityAssessment(BaseModel):
    """Full trust assessment for one facility, shared by scoring, agent, and API layers."""

    facility_id: str = Field(description="Stable facility identifier from the dataset or index.")
    facility_name: str = Field(description="Facility display name.")
    city: str = Field(description="Facility city.")
    state: str = Field(description="Facility state or region.")
    latitude: float = Field(description="Facility latitude.")
    longitude: float = Field(description="Facility longitude.")
    facility_type: str = Field(description="Raw or normalized facility type.")
    capability_claims: list[CapabilityClaim] = Field(
        default_factory=list,
        description="Capabilities extracted from reliable fields and assessed for support.",
    )
    contradictions: list[Contradiction] = Field(
        default_factory=list,
        description="Trust issues found during consistency and plausibility checks.",
    )
    trust_subscores: TrustSubscores = Field(description="Explanatory trust score components.")
    overall_trust_score: int = Field(
        ge=0,
        le=100,
        description="Single trust score from 0, least trustworthy, to 100, most trustworthy.",
    )
    confidence_interval: tuple[int, int] = Field(
        description="Lower and upper confidence bounds for the trust score."
    )
    reasoning_summary: str = Field(
        max_length=200,
        description="Short public explanation of the trust assessment.",
    )

    @field_validator("confidence_interval")
    @classmethod
    def validate_confidence_interval(cls, value: tuple[int, int]) -> tuple[int, int]:
        """Ensure confidence intervals are ordered score bounds."""
        low, high = value
        if not (0 <= low <= high <= 100):
            raise ValueError("confidence_interval must satisfy 0 <= low <= high <= 100")
        return value


class QueryPlan(BaseModel):
    """Agent planner output used to turn natural language into retrieval filters."""

    location_filters: dict[str, str | None] = Field(
        default_factory=lambda: {"state": None, "city": None, "region_type": None},
        description='Location filters with keys "state", "city", and "region_type".',
    )
    capability_filters: list[Capability] = Field(
        default_factory=list,
        description="Capabilities required or requested by the user query.",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Additional natural-language constraints from the query.",
    )

    @field_validator("location_filters", mode="before")
    @classmethod
    def normalize_location_filters(cls, value: Any) -> dict[str, str | None]:
        """Guarantee that the planner contract always exposes the same location keys."""
        provided = dict(value or {})
        return {
            "state": provided.get("state"),
            "city": provided.get("city"),
            "region_type": provided.get("region_type"),
        }


class FacilitySearchResult(BaseModel):
    """One ranked facility returned by natural-language search."""

    facility_id: str = Field(description="Stable facility identifier from the dataset or index.")
    facility_name: str = Field(description="Facility display name.")
    city: str = Field(description="Facility city.")
    state: str = Field(description="Facility state or region.")
    latitude: float = Field(description="Facility latitude.")
    longitude: float = Field(description="Facility longitude.")
    overall_trust_score: int = Field(
        ge=0,
        le=100,
        description="Trust score used for ranking and display.",
    )
    top_contradiction: str | None = Field(
        default=None,
        description="Most important contradiction to show in the result card, if any.",
    )
    match_reason: str = Field(description="Short explanation of why this result matched the query.")


class QueryResponse(BaseModel):
    """Complete API response for a plain-English facility search."""

    results: list[FacilitySearchResult] = Field(
        default_factory=list,
        description="Ranked facility matches.",
    )
    reasoning_steps: list[str] = Field(
        default_factory=list,
        description="Public planner-to-retriever-to-validator reasoning trace.",
    )
    confidence_interval: tuple[int, int] = Field(
        description="Lower and upper confidence bounds for the query response."
    )
    query_plan: QueryPlan = Field(description="Structured interpretation of the user query.")

    @field_validator("confidence_interval")
    @classmethod
    def validate_confidence_interval(cls, value: tuple[int, int]) -> tuple[int, int]:
        """Ensure confidence intervals are ordered score bounds."""
        low, high = value
        if not (0 <= low <= high <= 100):
            raise ValueError("confidence_interval must satisfy 0 <= low <= high <= 100")
        return value


if __name__ == "__main__":
    assessment = FacilityAssessment(
        facility_id="krishna-homeopathy-research-hospital-jaipur",
        facility_name="Krishna Homeopathy Research Hospital",
        city="Jaipur",
        state="Rajasthan",
        latitude=26.9124,
        longitude=75.7873,
        facility_type="clinic",
        capability_claims=[
            CapabilityClaim(
                capability=Capability.PRIMARY_CARE,
                status=CapabilityStatus.INFERRED,
                evidence_field="specialties",
                evidence_snippet="familyMedicine, internalMedicine",
            ),
            CapabilityClaim(
                capability=Capability.MENTAL_HEALTH,
                status=CapabilityStatus.INFERRED,
                evidence_field="specialties",
                evidence_snippet="psychiatry",
            ),
            CapabilityClaim(
                capability=Capability.CARDIOLOGY,
                status=CapabilityStatus.CONTRADICTED,
                evidence_field="specialties",
                evidence_snippet="cardiology listed alongside homeopathy clinic with no equipment",
            ),
            CapabilityClaim(
                capability=Capability.ONCOLOGY,
                status=CapabilityStatus.CONTRADICTED,
                evidence_field="specialties",
                evidence_snippet="medicalOncology listed with no equipment or staffing support",
            ),
            CapabilityClaim(
                capability=Capability.OPHTHALMOLOGY,
                status=CapabilityStatus.CONTRADICTED,
                evidence_field="specialties",
                evidence_snippet="ophthalmology listed without equipment support",
            ),
            CapabilityClaim(
                capability=Capability.DENTISTRY,
                status=CapabilityStatus.CONTRADICTED,
                evidence_field="specialties",
                evidence_snippet="dentistry listed in a homeopathy clinic profile",
            ),
        ],
        contradictions=[
            Contradiction(
                contradiction_type=ContradictionType.SPECIALTY_SPRAWL,
                field_name="specialties",
                claim=(
                    "Claims urology, psychiatry, cardiology, oncology, ophthalmology, "
                    "dentistry, and more."
                ),
                why_contradictory=(
                    "A homeopathy clinic with one listed doctor is unlikely to support "
                    "so many unrelated specialties."
                ),
                severity=5,
            ),
            Contradiction(
                contradiction_type=ContradictionType.MISSING_EQUIPMENT,
                field_name="equipment",
                claim="No equipment is listed.",
                why_contradictory=(
                    "Claims like cardiology, oncology, ophthalmology, and dentistry "
                    "usually require supporting equipment."
                ),
                severity=4,
            ),
            Contradiction(
                contradiction_type=ContradictionType.MISSING_STAFF,
                field_name="numberDoctors",
                claim="Only 1 doctor listed on staff.",
                why_contradictory=(
                    "One listed doctor does not plausibly support the claimed specialty breadth."
                ),
                severity=4,
            ),
        ],
        trust_subscores=TrustSubscores(
            internal_consistency=20,
            capability_plausibility=15,
            activity_signal=35,
            completeness=70,
        ),
        overall_trust_score=28,
        confidence_interval=(20, 38),
        reasoning_summary=(
            "Classic specialty sprawl: many unrelated specialties, one listed doctor, "
            "and no equipment evidence."
        ),
    )

    first_json = assessment.model_dump_json(indent=2)
    print(first_json)

    round_tripped = FacilityAssessment.model_validate_json(first_json)
    second_json = round_tripped.model_dump_json(indent=2)
    assert first_json == second_json
    print("\nRound-trip validation passed.")
