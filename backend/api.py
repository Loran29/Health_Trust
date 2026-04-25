from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from .schemas import (
        Capability,
        CapabilityClaim,
        CapabilityStatus,
        Contradiction,
        ContradictionType,
        FacilityAssessment,
        FacilitySearchResult,
        QueryPlan,
        QueryResponse,
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
        FacilitySearchResult,
        QueryPlan,
        QueryResponse,
        TrustSubscores,
    )


app = FastAPI(title="TrustMap India API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    """Plain-English query submitted by the frontend."""

    text: str = Field(min_length=1, description="Natural-language search query.")


def claim(
    capability: Capability,
    status: CapabilityStatus,
    evidence_field: str,
    evidence_snippet: str,
) -> CapabilityClaim:
    return CapabilityClaim(
        capability=capability,
        status=status,
        evidence_field=evidence_field,
        evidence_snippet=evidence_snippet[:200],
    )


def contradiction(
    contradiction_type: ContradictionType,
    field_name: str,
    claim_text: str,
    why: str,
    severity: int,
) -> Contradiction:
    return Contradiction(
        contradiction_type=contradiction_type,
        field_name=field_name,
        claim=claim_text,
        why_contradictory=why,
        severity=severity,
    )


FACILITIES: dict[str, FacilityAssessment] = {
    "1000-smiles-dental-clinic-hyderabad": FacilityAssessment(
        facility_id="1000-smiles-dental-clinic-hyderabad",
        facility_name="1000 Smiles Dental Clinic",
        city="Hyderabad",
        state="Telangana",
        latitude=17.397739,
        longitude=78.482681,
        facility_type="clinic",
        capability_claims=[
            claim(
                Capability.DENTISTRY,
                CapabilityStatus.CONFIRMED,
                "specialties",
                "periodontics, endodontics, dentistry, aestheticDentistry",
            ),
            claim(
                Capability.PRIMARY_CARE,
                CapabilityStatus.CONTRADICTED,
                "specialties",
                "familyMedicine appears on a dental clinic profile",
            ),
        ],
        contradictions=[
            contradiction(
                ContradictionType.TYPE_SPECIALTY_MISMATCH,
                "specialties",
                "Dental clinic also lists familyMedicine.",
                "A dental clinic profile claiming family medicine needs stronger supporting evidence.",
                3,
            ),
            contradiction(
                ContradictionType.MISSING_EQUIPMENT,
                "equipment",
                "No equipment is listed.",
                "Dental specialties usually need chair, imaging, or procedure equipment evidence.",
                2,
            ),
        ],
        trust_subscores=TrustSubscores(
            internal_consistency=58,
            capability_plausibility=62,
            activity_signal=45,
            completeness=70,
        ),
        overall_trust_score=55,
        confidence_interval=(46, 64),
        reasoning_summary="Credible dental signals, but family medicine and missing equipment lower trust.",
    ),
    "aastha-children-hospital-dehri": FacilityAssessment(
        facility_id="aastha-children-hospital-dehri",
        facility_name="Aastha Children Hospital",
        city="Dehri",
        state="Bihar",
        latitude=24.9028,
        longitude=84.1821,
        facility_type="hospital",
        capability_claims=[
            claim(
                Capability.PEDIATRICS,
                CapabilityStatus.INFERRED,
                "name",
                "Aastha Children Hospital",
            ),
            claim(
                Capability.EMERGENCY,
                CapabilityStatus.CONTRADICTED,
                "capability",
                "only 24/7 pediatric emergency hospital in Dehri-on-Sone",
            ),
            claim(
                Capability.ICU,
                CapabilityStatus.CONTRADICTED,
                "capability",
                "Has NICU and PICU",
            ),
        ],
        contradictions=[
            contradiction(
                ContradictionType.MISSING_EQUIPMENT,
                "equipment",
                "Claims pediatric emergency, NICU, and PICU but lists no equipment.",
                "Emergency and ICU claims need equipment evidence such as monitors or ventilators.",
                5,
            ),
            contradiction(
                ContradictionType.MISSING_STAFF,
                "numberDoctors",
                "No doctor count is available.",
                "A 24/7 pediatric emergency claim is hard to verify without staffing signals.",
                4,
            ),
        ],
        trust_subscores=TrustSubscores(
            internal_consistency=48,
            capability_plausibility=32,
            activity_signal=38,
            completeness=57,
        ),
        overall_trust_score=42,
        confidence_interval=(29, 55),
        reasoning_summary="Pediatric emergency claims are important, but staff and equipment evidence is sparse.",
    ),
    "krishna-homeopathy-research-hospital-jaipur": FacilityAssessment(
        facility_id="krishna-homeopathy-research-hospital-jaipur",
        facility_name="Krishna Homeopathy Research Hospital",
        city="Jaipur",
        state="Rajasthan",
        latitude=26.9124,
        longitude=75.7873,
        facility_type="clinic",
        capability_claims=[
            claim(Capability.PRIMARY_CARE, CapabilityStatus.INFERRED, "specialties", "familyMedicine"),
            claim(Capability.MENTAL_HEALTH, CapabilityStatus.INFERRED, "specialties", "psychiatry"),
            claim(Capability.CARDIOLOGY, CapabilityStatus.CONTRADICTED, "specialties", "cardiology"),
            claim(Capability.ONCOLOGY, CapabilityStatus.CONTRADICTED, "specialties", "medicalOncology"),
            claim(Capability.OPHTHALMOLOGY, CapabilityStatus.CONTRADICTED, "specialties", "ophthalmology"),
            claim(Capability.DENTISTRY, CapabilityStatus.CONTRADICTED, "specialties", "dentistry"),
        ],
        contradictions=[
            contradiction(
                ContradictionType.SPECIALTY_SPRAWL,
                "specialties",
                "Lists urology, psychiatry, cardiology, oncology, ophthalmology, dentistry, and more.",
                "A homeopathy clinic with one listed doctor is unlikely to support so many fields.",
                5,
            ),
            contradiction(
                ContradictionType.MISSING_EQUIPMENT,
                "equipment",
                "No equipment is listed.",
                "Specialties like cardiology, oncology, ophthalmology, and dentistry need equipment support.",
                4,
            ),
            contradiction(
                ContradictionType.MISSING_STAFF,
                "numberDoctors",
                "Only 1 doctor is listed.",
                "One doctor does not plausibly cover the claimed breadth of specialties.",
                4,
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
        reasoning_summary="Classic specialty sprawl: many unrelated specialties, one doctor, no equipment evidence.",
    ),
    "city-health-clinic-guwahati": FacilityAssessment(
        facility_id="city-health-clinic-guwahati",
        facility_name="City Health Clinic & Diagnostic",
        city="Guwahati",
        state="Assam",
        latitude=26.1445,
        longitude=91.7362,
        facility_type="clinic",
        capability_claims=[
            claim(Capability.CARDIOLOGY, CapabilityStatus.CONFIRMED, "equipment", "ECG, Echo, TMT"),
            claim(Capability.SURGERY, CapabilityStatus.INFERRED, "capability", "on-site operating theatre"),
            claim(Capability.ICU, CapabilityStatus.INFERRED, "capability", "Has ICU"),
            claim(Capability.ONCOLOGY, CapabilityStatus.CONTRADICTED, "specialties", "medicalOncology"),
            claim(Capability.DENTISTRY, CapabilityStatus.CONTRADICTED, "specialties", "dentistry"),
        ],
        contradictions=[
            contradiction(
                ContradictionType.SPECIALTY_SPRAWL,
                "specialties",
                "Lists more than twenty specialties across unrelated clinical families.",
                "The specialty breadth is more hospital-like than clinic-like and needs strong verification.",
                4,
            ),
            contradiction(
                ContradictionType.TYPE_SPECIALTY_MISMATCH,
                "facilitytypeid",
                "Facility is typed as clinic while claiming ICU and operating theatre.",
                "Clinic type conflicts with high-acuity inpatient capability claims.",
                4,
            ),
        ],
        trust_subscores=TrustSubscores(
            internal_consistency=38,
            capability_plausibility=48,
            activity_signal=52,
            completeness=72,
        ),
        overall_trust_score=45,
        confidence_interval=(35, 56),
        reasoning_summary="Strong diagnostics evidence, but clinic type conflicts with ICU and broad specialty claims.",
    ),
    "7-star-healthcare-delhi": FacilityAssessment(
        facility_id="7-star-healthcare-delhi",
        facility_name="7 Star Healthcare (Hospital)",
        city="Delhi",
        state="Delhi",
        latitude=28.5612,
        longitude=77.2848,
        facility_type="hospital",
        capability_claims=[
            claim(Capability.RUNS_24_7, CapabilityStatus.CONFIRMED, "capability", "Always open"),
            claim(Capability.CARDIOLOGY, CapabilityStatus.INFERRED, "specialties", "cardiacSurgery"),
            claim(Capability.OBSTETRICS, CapabilityStatus.UNKNOWN, "specialties", "infertility services"),
            claim(Capability.SURGERY, CapabilityStatus.INFERRED, "specialties", "cardiacSurgery"),
        ],
        contradictions=[
            contradiction(
                ContradictionType.CAPABILITY_OVERREACH,
                "capability",
                "Claims 24x7 ultrasound and many specialties under one roof.",
                "The claim is plausible in Delhi but needs more staff and department evidence.",
                3,
            ),
            contradiction(
                ContradictionType.MISSING_STAFF,
                "numberDoctors",
                "No reliable doctor count is available.",
                "Round-the-clock services require staffing evidence to be highly trusted.",
                3,
            ),
        ],
        trust_subscores=TrustSubscores(
            internal_consistency=61,
            capability_plausibility=58,
            activity_signal=64,
            completeness=68,
        ),
        overall_trust_score=60,
        confidence_interval=(51, 70),
        reasoning_summary="Good urban activity signals, but 24x7 and surgical claims need staffing support.",
    ),
}


def result(
    facility_id: str,
    match_reason: str,
    top_contradiction: str | None = None,
) -> FacilitySearchResult:
    facility = FACILITIES[facility_id]
    return FacilitySearchResult(
        facility_id=facility.facility_id,
        facility_name=facility.facility_name,
        city=facility.city,
        state=facility.state,
        latitude=facility.latitude,
        longitude=facility.longitude,
        overall_trust_score=facility.overall_trust_score,
        top_contradiction=top_contradiction,
        match_reason=match_reason,
    )


def generic_results() -> list[FacilitySearchResult]:
    return [
        FacilitySearchResult(
            facility_id="apollo-clinic-pune",
            facility_name="Apollo Clinic Pune",
            city="Pune",
            state="Maharashtra",
            latitude=18.5204,
            longitude=73.8567,
            overall_trust_score=82,
            top_contradiction=None,
            match_reason="Urban clinic with clear primary-care and diagnostic signals.",
        ),
        FacilitySearchResult(
            facility_id="narayana-health-bengaluru",
            facility_name="Narayana Health Bengaluru",
            city="Bengaluru",
            state="Karnataka",
            latitude=12.8452,
            longitude=77.6602,
            overall_trust_score=88,
            top_contradiction=None,
            match_reason="Strong specialty coverage and urban activity signals.",
        ),
        FacilitySearchResult(
            facility_id="district-hospital-malkangiri",
            facility_name="District Hospital Malkangiri",
            city="Malkangiri",
            state="Odisha",
            latitude=18.3436,
            longitude=81.8825,
            overall_trust_score=64,
            top_contradiction="Sparse staffing and equipment fields widen uncertainty.",
            match_reason="Rural public hospital candidate with relevant baseline services.",
        ),
    ]


def confidence_for_query(text: str, default: tuple[int, int]) -> tuple[int, int]:
    lowered = text.lower()
    rural_terms = ["rural", "village", "bihar", "odisha", "northeast", "assam", "dehri"]
    urban_terms = ["mumbai", "delhi", "bengaluru", "bangalore", "hyderabad", "pune", "jaipur"]
    if any(term in lowered for term in rural_terms):
        return (max(0, default[0] - 10), min(100, default[1] + 12))
    if any(term in lowered for term in urban_terms):
        midpoint = round((default[0] + default[1]) / 2)
        return (max(0, midpoint - 7), min(100, midpoint + 7))
    return default


def response_for_query(text: str) -> QueryResponse:
    lowered = text.lower()

    if "dental" in lowered or "dentist" in lowered:
        plan = QueryPlan(
            location_filters={"state": "Telangana", "city": "Hyderabad", "region_type": "urban"},
            capability_filters=[Capability.DENTISTRY],
            constraints=["prefer facilities with verified dental capability"],
        )
        return QueryResponse(
            results=[
                result(
                    "1000-smiles-dental-clinic-hyderabad",
                    "Matches dentistry terms and has dental specialties.",
                    "Dental clinic also lists familyMedicine.",
                ),
                result(
                    "krishna-homeopathy-research-hospital-jaipur",
                    "Includes dentistry in specialties, but validator flags this as weak evidence.",
                    "Specialty sprawl across unrelated fields.",
                ),
                result(
                    "city-health-clinic-guwahati",
                    "Large multi-specialty profile includes dentistry among many services.",
                    "Clinic claims hospital-like breadth.",
                ),
            ],
            reasoning_steps=[
                "Decomposed query into location=Hyderabad, capability=dentistry, constraint=verified dental care",
                "Retrieved 31 candidate facilities matching dental or dentist terms",
                "Filtered by capability=dentistry -> 12 candidates",
                "Validator flagged clinic records where dental claims conflict with broad medical specialties",
                "Ranked by trust score, contradiction severity, and location fit",
            ],
            confidence_interval=confidence_for_query(text, (68, 84)),
            query_plan=plan,
        )

    if "emergency" in lowered or "c-section" in lowered or "obstetrics" in lowered:
        plan = QueryPlan(
            location_filters={"state": "Bihar", "city": "Dehri", "region_type": "semi_urban"},
            capability_filters=[Capability.EMERGENCY, Capability.OBSTETRICS, Capability.ANESTHESIA],
            constraints=["prefer 24/7 care with staff and equipment evidence"],
        )
        return QueryResponse(
            results=[
                result(
                    "aastha-children-hospital-dehri",
                    "Matches pediatric emergency language and rural Bihar location.",
                    "Claims NICU/PICU and emergency care but lists no equipment.",
                ),
                result(
                    "7-star-healthcare-delhi",
                    "Urban hospital stub with 24x7 availability and surgical-adjacent claims.",
                    "No reliable doctor count for 24x7 service claims.",
                ),
                result(
                    "city-health-clinic-guwahati",
                    "Has ICU and operating theatre language, but facility type is clinic.",
                    "Clinic type conflicts with ICU and OT claims.",
                ),
            ],
            reasoning_steps=[
                "Decomposed query into location=Bihar, capability=emergency/obstetrics, constraint=24/7 support",
                "Retrieved 23 candidate facilities matching emergency or obstetrics terms",
                "Filtered by capability=emergency -> 8 candidates",
                "Validator checked equipment and staffing signals for high-acuity claims",
                "Ranked by trust score while widening confidence for sparse district data",
            ],
            confidence_interval=confidence_for_query(text, (42, 68)),
            query_plan=plan,
        )

    if "homeopathy" in lowered or "specialty" in lowered or "sprawl" in lowered:
        plan = QueryPlan(
            location_filters={"state": "Rajasthan", "city": "Jaipur", "region_type": "urban"},
            capability_filters=[
                Capability.PRIMARY_CARE,
                Capability.CARDIOLOGY,
                Capability.ONCOLOGY,
                Capability.OPHTHALMOLOGY,
            ],
            constraints=["surface suspicious specialty breadth"],
        )
        return QueryResponse(
            results=[
                result(
                    "krishna-homeopathy-research-hospital-jaipur",
                    "Direct match for homeopathy and broad specialty claims.",
                    "Specialty sprawl with one listed doctor and no equipment.",
                ),
                result(
                    "city-health-clinic-guwahati",
                    "Another broad specialty profile useful for comparison.",
                    "Clinic claims ICU, OT, and many hospital-like departments.",
                ),
                result(
                    "1000-smiles-dental-clinic-hyderabad",
                    "Smaller mismatch example with one broad medical specialty on a dental profile.",
                    "Dental clinic also lists familyMedicine.",
                ),
            ],
            reasoning_steps=[
                "Decomposed query into location=Jaipur, capability=broad specialties, constraint=find sprawl",
                "Retrieved 44 candidate facilities with five or more specialties",
                "Filtered for unrelated clinical families such as oncology, cardiology, psychiatry, dentistry",
                "Validator scored missing equipment and missing staff as contradiction amplifiers",
                "Ranked by contradiction severity and demo clarity",
            ],
            confidence_interval=confidence_for_query(text, (62, 78)),
            query_plan=plan,
        )

    plan = QueryPlan(
        location_filters={"state": None, "city": None, "region_type": None},
        capability_filters=[Capability.PRIMARY_CARE],
        constraints=["balanced nationwide search"],
    )
    return QueryResponse(
        results=generic_results(),
        reasoning_steps=[
            "Decomposed query into location=all India, capability=primary_care, constraint=broad match",
            "Retrieved 118 candidate facilities from multiple states",
            "Filtered for usable name, city, state, type, specialties, and coordinates",
            "Validator downranked records with sparse capability evidence",
            "Ranked by trust score and geographic diversity",
        ],
        confidence_interval=confidence_for_query(text, (58, 76)),
        query_plan=plan,
    )


DISTRICTS: list[dict[str, Any]] = [
    {
        "district": "Mumbai",
        "state": "Maharashtra",
        "desert_score": 92,
        "population": 3085411,
        "top_capability_gaps": ["rural outreach", "affordable dialysis"],
        "num_facilities": 620,
        "avg_trust_score": 81,
    },
    {
        "district": "Bengaluru Urban",
        "state": "Karnataka",
        "desert_score": 90,
        "population": 9621551,
        "top_capability_gaps": ["public emergency beds", "low-cost oncology"],
        "num_facilities": 710,
        "avg_trust_score": 83,
    },
    {
        "district": "New Delhi",
        "state": "Delhi",
        "desert_score": 86,
        "population": 142004,
        "top_capability_gaps": ["transparent staffing", "verified 24/7 care"],
        "num_facilities": 210,
        "avg_trust_score": 77,
    },
    {
        "district": "Pune",
        "state": "Maharashtra",
        "desert_score": 84,
        "population": 9429408,
        "top_capability_gaps": ["peri-urban obstetrics", "mental health"],
        "num_facilities": 540,
        "avg_trust_score": 76,
    },
    {
        "district": "Hyderabad",
        "state": "Telangana",
        "desert_score": 82,
        "population": 3943323,
        "top_capability_gaps": ["verified dental equipment", "public ICU beds"],
        "num_facilities": 480,
        "avg_trust_score": 74,
    },
    {
        "district": "Patna",
        "state": "Bihar",
        "desert_score": 61,
        "population": 5838465,
        "top_capability_gaps": ["anesthesia support", "trusted obstetrics"],
        "num_facilities": 260,
        "avg_trust_score": 59,
    },
    {
        "district": "Gaya",
        "state": "Bihar",
        "desert_score": 44,
        "population": 4391418,
        "top_capability_gaps": ["emergency surgery", "ICU", "dialysis"],
        "num_facilities": 118,
        "avg_trust_score": 46,
    },
    {
        "district": "Malkangiri",
        "state": "Odisha",
        "desert_score": 26,
        "population": 613192,
        "top_capability_gaps": ["obstetrics", "emergency transport", "blood bank"],
        "num_facilities": 38,
        "avg_trust_score": 39,
    },
    {
        "district": "Nabarangpur",
        "state": "Odisha",
        "desert_score": 29,
        "population": 1220946,
        "top_capability_gaps": ["dialysis", "pediatrics", "ICU"],
        "num_facilities": 42,
        "avg_trust_score": 41,
    },
    {
        "district": "Kalahandi",
        "state": "Odisha",
        "desert_score": 35,
        "population": 1576869,
        "top_capability_gaps": ["oncology", "emergency surgery", "mental health"],
        "num_facilities": 55,
        "avg_trust_score": 44,
    },
    {
        "district": "Dhemaji",
        "state": "Assam",
        "desert_score": 31,
        "population": 686133,
        "top_capability_gaps": ["flood-season emergency care", "obstetrics"],
        "num_facilities": 34,
        "avg_trust_score": 42,
    },
    {
        "district": "Ukhrul",
        "state": "Manipur",
        "desert_score": 24,
        "population": 183998,
        "top_capability_gaps": ["specialists", "surgery", "reliable transport"],
        "num_facilities": 18,
        "avg_trust_score": 37,
    },
    {
        "district": "Mon",
        "state": "Nagaland",
        "desert_score": 22,
        "population": 250260,
        "top_capability_gaps": ["emergency care", "obstetrics", "diagnostics"],
        "num_facilities": 16,
        "avg_trust_score": 35,
    },
    {
        "district": "West Kameng",
        "state": "Arunachal Pradesh",
        "desert_score": 21,
        "population": 83947,
        "top_capability_gaps": ["ICU", "specialist access", "road-linked emergency care"],
        "num_facilities": 12,
        "avg_trust_score": 34,
    },
    {
        "district": "Gadchiroli",
        "state": "Maharashtra",
        "desert_score": 33,
        "population": 1072942,
        "top_capability_gaps": ["maternal emergency care", "dialysis", "critical care"],
        "num_facilities": 46,
        "avg_trust_score": 43,
    },
]


FACILITY_PINS: list[dict[str, Any]] = [
    {
        "facility_id": facility.facility_id,
        "name": facility.facility_name,
        "latitude": facility.latitude,
        "longitude": facility.longitude,
        "trust_score": facility.overall_trust_score,
        "has_contradictions": bool(facility.contradictions),
    }
    for facility in FACILITIES.values()
] + [
    {"facility_id": "apollo-clinic-pune", "name": "Apollo Clinic Pune", "latitude": 18.5204, "longitude": 73.8567, "trust_score": 82, "has_contradictions": False},
    {"facility_id": "narayana-health-bengaluru", "name": "Narayana Health Bengaluru", "latitude": 12.8452, "longitude": 77.6602, "trust_score": 88, "has_contradictions": False},
    {"facility_id": "district-hospital-malkangiri", "name": "District Hospital Malkangiri", "latitude": 18.3436, "longitude": 81.8825, "trust_score": 64, "has_contradictions": True},
    {"facility_id": "patna-medical-centre", "name": "Patna Medical Centre", "latitude": 25.5941, "longitude": 85.1376, "trust_score": 66, "has_contradictions": True},
    {"facility_id": "gaya-emergency-clinic", "name": "Gaya Emergency Clinic", "latitude": 24.7914, "longitude": 85.0002, "trust_score": 49, "has_contradictions": True},
    {"facility_id": "nabarangpur-district-hospital", "name": "Nabarangpur District Hospital", "latitude": 19.2281, "longitude": 82.5483, "trust_score": 52, "has_contradictions": True},
    {"facility_id": "kalahandi-care-centre", "name": "Kalahandi Care Centre", "latitude": 19.9137, "longitude": 83.1649, "trust_score": 47, "has_contradictions": True},
    {"facility_id": "dhemaji-community-hospital", "name": "Dhemaji Community Hospital", "latitude": 27.4811, "longitude": 94.5570, "trust_score": 43, "has_contradictions": True},
    {"facility_id": "ukhrul-health-centre", "name": "Ukhrul Health Centre", "latitude": 25.0952, "longitude": 94.3610, "trust_score": 36, "has_contradictions": True},
    {"facility_id": "mon-district-clinic", "name": "Mon District Clinic", "latitude": 26.7358, "longitude": 95.0584, "trust_score": 34, "has_contradictions": True},
    {"facility_id": "bomdila-general-hospital", "name": "Bomdila General Hospital", "latitude": 27.2648, "longitude": 92.4249, "trust_score": 41, "has_contradictions": True},
    {"facility_id": "gadchiroli-rural-hospital", "name": "Gadchiroli Rural Hospital", "latitude": 20.1849, "longitude": 80.0035, "trust_score": 45, "has_contradictions": True},
    {"facility_id": "mumbai-heart-institute", "name": "Mumbai Heart Institute", "latitude": 19.0760, "longitude": 72.8777, "trust_score": 91, "has_contradictions": False},
    {"facility_id": "thane-maternity-centre", "name": "Thane Maternity Centre", "latitude": 19.2183, "longitude": 72.9781, "trust_score": 78, "has_contradictions": False},
    {"facility_id": "surat-dialysis-centre", "name": "Surat Dialysis Centre", "latitude": 21.1702, "longitude": 72.8311, "trust_score": 74, "has_contradictions": False},
    {"facility_id": "ahmedabad-eye-care", "name": "Ahmedabad Eye Care", "latitude": 23.0225, "longitude": 72.5714, "trust_score": 81, "has_contradictions": False},
    {"facility_id": "jaipur-ortho-clinic", "name": "Jaipur Ortho Clinic", "latitude": 26.9124, "longitude": 75.7873, "trust_score": 69, "has_contradictions": True},
    {"facility_id": "jodhpur-cancer-care", "name": "Jodhpur Cancer Care", "latitude": 26.2389, "longitude": 73.0243, "trust_score": 58, "has_contradictions": True},
    {"facility_id": "lucknow-childrens-hospital", "name": "Lucknow Childrens Hospital", "latitude": 26.8467, "longitude": 80.9462, "trust_score": 72, "has_contradictions": False},
    {"facility_id": "kanpur-emergency-care", "name": "Kanpur Emergency Care", "latitude": 26.4499, "longitude": 80.3319, "trust_score": 57, "has_contradictions": True},
    {"facility_id": "varanasi-trauma-centre", "name": "Varanasi Trauma Centre", "latitude": 25.3176, "longitude": 82.9739, "trust_score": 62, "has_contradictions": True},
    {"facility_id": "ranchi-multispeciality", "name": "Ranchi Multispeciality", "latitude": 23.3441, "longitude": 85.3096, "trust_score": 71, "has_contradictions": False},
    {"facility_id": "kolkata-skin-laser", "name": "Kolkata Skin & Laser", "latitude": 22.5726, "longitude": 88.3639, "trust_score": 76, "has_contradictions": False},
    {"facility_id": "bhubaneswar-neuro-care", "name": "Bhubaneswar Neuro Care", "latitude": 20.2961, "longitude": 85.8245, "trust_score": 79, "has_contradictions": False},
    {"facility_id": "cuttack-surgery-centre", "name": "Cuttack Surgery Centre", "latitude": 20.4625, "longitude": 85.8828, "trust_score": 67, "has_contradictions": True},
    {"facility_id": "visakhapatnam-oncology", "name": "Visakhapatnam Oncology", "latitude": 17.6868, "longitude": 83.2185, "trust_score": 73, "has_contradictions": False},
    {"facility_id": "vijayawada-family-clinic", "name": "Vijayawada Family Clinic", "latitude": 16.5062, "longitude": 80.6480, "trust_score": 70, "has_contradictions": False},
    {"facility_id": "chennai-eye-hospital", "name": "Chennai Eye Hospital", "latitude": 13.0827, "longitude": 80.2707, "trust_score": 86, "has_contradictions": False},
    {"facility_id": "madurai-emergency-hospital", "name": "Madurai Emergency Hospital", "latitude": 9.9252, "longitude": 78.1198, "trust_score": 68, "has_contradictions": True},
    {"facility_id": "coimbatore-dental-care", "name": "Coimbatore Dental Care", "latitude": 11.0168, "longitude": 76.9558, "trust_score": 80, "has_contradictions": False},
    {"facility_id": "kochi-maternity-care", "name": "Kochi Maternity Care", "latitude": 9.9312, "longitude": 76.2673, "trust_score": 84, "has_contradictions": False},
    {"facility_id": "thiruvananthapuram-medical", "name": "Thiruvananthapuram Medical", "latitude": 8.5241, "longitude": 76.9366, "trust_score": 83, "has_contradictions": False},
    {"facility_id": "mysuru-primary-care", "name": "Mysuru Primary Care", "latitude": 12.2958, "longitude": 76.6394, "trust_score": 75, "has_contradictions": False},
    {"facility_id": "hubballi-heart-care", "name": "Hubballi Heart Care", "latitude": 15.3647, "longitude": 75.1240, "trust_score": 77, "has_contradictions": False},
    {"facility_id": "raipur-critical-care", "name": "Raipur Critical Care", "latitude": 21.2514, "longitude": 81.6296, "trust_score": 61, "has_contradictions": True},
    {"facility_id": "nagpur-orthopedics", "name": "Nagpur Orthopedics", "latitude": 21.1458, "longitude": 79.0882, "trust_score": 78, "has_contradictions": False},
    {"facility_id": "indore-child-care", "name": "Indore Child Care", "latitude": 22.7196, "longitude": 75.8577, "trust_score": 74, "has_contradictions": False},
    {"facility_id": "bhopal-mental-health", "name": "Bhopal Mental Health", "latitude": 23.2599, "longitude": 77.4126, "trust_score": 71, "has_contradictions": False},
    {"facility_id": "chandigarh-diagnostics", "name": "Chandigarh Diagnostics", "latitude": 30.7333, "longitude": 76.7794, "trust_score": 87, "has_contradictions": False},
    {"facility_id": "amritsar-emergency-hospital", "name": "Amritsar Emergency Hospital", "latitude": 31.6340, "longitude": 74.8723, "trust_score": 65, "has_contradictions": True},
    {"facility_id": "srinagar-general-hospital", "name": "Srinagar General Hospital", "latitude": 34.0837, "longitude": 74.7973, "trust_score": 63, "has_contradictions": True},
    {"facility_id": "leh-community-health", "name": "Leh Community Health", "latitude": 34.1526, "longitude": 77.5771, "trust_score": 55, "has_contradictions": True},
    {"facility_id": "shillong-family-clinic", "name": "Shillong Family Clinic", "latitude": 25.5788, "longitude": 91.8933, "trust_score": 59, "has_contradictions": True},
    {"facility_id": "aizawl-district-hospital", "name": "Aizawl District Hospital", "latitude": 23.7271, "longitude": 92.7176, "trust_score": 54, "has_contradictions": True},
    {"facility_id": "agartala-womens-care", "name": "Agartala Womens Care", "latitude": 23.8315, "longitude": 91.2868, "trust_score": 60, "has_contradictions": True},
]


@app.get("/")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "TrustMap India"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    return response_for_query(request.text)


@app.get("/facility/{facility_id}", response_model=FacilityAssessment)
def get_facility(facility_id: str) -> FacilityAssessment:
    facility = FACILITIES.get(facility_id)
    if facility is None:
        raise HTTPException(status_code=404, detail="Facility not found")
    return facility


@app.get("/districts")
def get_districts() -> list[dict[str, Any]]:
    return DISTRICTS


@app.get("/facility-pins")
def get_facility_pins() -> list[dict[str, Any]]:
    return FACILITY_PINS


def run_self_test() -> None:
    from fastapi.testclient import TestClient

    client = TestClient(app)
    checks = [
        ("GET /", client.get("/")),
        ("POST /query dental", client.post("/query", json={"text": "best dental clinic in Hyderabad"})),
        (
            "POST /query emergency",
            client.post("/query", json={"text": "24/7 emergency obstetrics in rural Bihar"}),
        ),
        (
            "POST /query homeopathy",
            client.post("/query", json={"text": "show homeopathy specialty sprawl in Jaipur"}),
        ),
        (
            "GET /facility/{id}",
            client.get("/facility/krishna-homeopathy-research-hospital-jaipur"),
        ),
        ("GET /districts", client.get("/districts")),
        ("GET /facility-pins", client.get("/facility-pins")),
    ]

    for label, response in checks:
        assert response.status_code == 200, f"{label} returned {response.status_code}"
        payload = response.json()
        if label.startswith("POST /query"):
            QueryResponse.model_validate(payload)
        elif label == "GET /facility/{id}":
            FacilityAssessment.model_validate(payload)
        elif label == "GET /districts":
            assert isinstance(payload, list) and len(payload) == 15
        elif label == "GET /facility-pins":
            assert isinstance(payload, list) and len(payload) == 50
        print(f"{label}: 200 OK")

    not_found = client.get("/facility/not-a-real-id")
    assert not_found.status_code == 404
    print("GET /facility/not-a-real-id: 404 OK")


if __name__ == "__main__":
    run_self_test()
