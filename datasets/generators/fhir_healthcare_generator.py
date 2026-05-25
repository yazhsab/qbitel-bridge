"""
HL7 FHIR R4 Healthcare Resource & Security Instruction Generator

Generates realistic HL7 FHIR R4 JSON resources and healthcare-security
instruction pairs for training PQC-aware healthcare ML models.

Part A - FHIR R4 Resources (JSON + metadata):
  Patient, Observation, MedicationRequest, Condition,
  DiagnosticReport, Immunization

Part B - Healthcare Security Instruction Pairs (JSONL):
  hipaa_pqc, fhir_security, medical_device,
  ehr_sharing, homomorphic_analytics
"""

import json
import random
import uuid
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


class FHIRHealthcareGenerator:
    """Generate realistic FHIR R4 resources and healthcare-security instruction pairs."""

    # ------------------------------------------------------------------
    # Shared Constants
    # ------------------------------------------------------------------

    FIRST_NAMES = [
        "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
        "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
        "Joseph", "Jessica", "Thomas", "Sarah", "Christopher", "Karen",
    ]
    LAST_NAMES = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
        "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Wilson",
        "Anderson", "Taylor", "Thomas", "Moore", "Jackson", "White", "Harris",
    ]
    STREETS = [
        "Main St", "Oak Ave", "Maple Dr", "Elm Rd", "Park Blvd",
        "Cedar Ln", "Pine St", "Washington Ave", "Lincoln Rd", "Liberty Dr",
    ]
    CITIES = [
        "Springfield", "Franklin", "Clinton", "Madison", "Georgetown",
        "Salem", "Fairview", "Bristol", "Oxford", "Arlington",
    ]
    STATES = ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI"]

    # LOINC vital-sign codes  (code -> display, unit, normal_low, normal_high)
    VITAL_SIGNS = {
        "8867-4":  ("Heart rate", "beats/min", 60, 100),
        "8480-6":  ("Systolic blood pressure", "mmHg", 90, 140),
        "8462-4":  ("Diastolic blood pressure", "mmHg", 60, 90),
        "8310-5":  ("Body temperature", "degC", 36.1, 37.2),
        "2708-6":  ("Oxygen saturation (SpO2)", "%", 95, 100),
        "9279-1":  ("Respiratory rate", "breaths/min", 12, 20),
    }

    # ICD-10 / SNOMED condition codes
    CONDITION_CODES = [
        {"icd10": "I10",     "snomed": "38341003",  "display": "Essential hypertension"},
        {"icd10": "E11.9",   "snomed": "44054006",  "display": "Type 2 diabetes mellitus"},
        {"icd10": "J06.9",   "snomed": "54150009",  "display": "Upper respiratory infection"},
        {"icd10": "M54.5",   "snomed": "279039007", "display": "Low back pain"},
        {"icd10": "K21.0",   "snomed": "235595009", "display": "Gastroesophageal reflux disease"},
        {"icd10": "F32.9",   "snomed": "35489007",  "display": "Major depressive disorder"},
        {"icd10": "J45.909", "snomed": "195967001", "display": "Asthma, unspecified"},
        {"icd10": "N39.0",   "snomed": "68566005",  "display": "Urinary tract infection"},
        {"icd10": "G43.909", "snomed": "37796009",  "display": "Migraine, unspecified"},
        {"icd10": "J18.9",   "snomed": "233604007", "display": "Pneumonia, unspecified"},
    ]

    # RxNorm medication codes
    MEDICATIONS = [
        {"rxnorm": "197361", "display": "Lisinopril 10 MG Oral Tablet"},
        {"rxnorm": "860975", "display": "Metformin 500 MG Oral Tablet"},
        {"rxnorm": "311989", "display": "Atorvastatin 20 MG Oral Tablet"},
        {"rxnorm": "197696", "display": "Omeprazole 20 MG Oral Capsule"},
        {"rxnorm": "310965", "display": "Amoxicillin 500 MG Oral Capsule"},
        {"rxnorm": "198440", "display": "Sertraline 50 MG Oral Tablet"},
        {"rxnorm": "312961", "display": "Ibuprofen 200 MG Oral Tablet"},
        {"rxnorm": "197379", "display": "Amlodipine 5 MG Oral Tablet"},
        {"rxnorm": "311368", "display": "Albuterol 90 MCG/ACT Inhalant"},
        {"rxnorm": "197591", "display": "Levothyroxine 50 MCG Oral Tablet"},
    ]

    # CVX vaccine codes
    VACCINES = [
        {"cvx": "207", "display": "COVID-19 mRNA (Moderna)"},
        {"cvx": "208", "display": "COVID-19 mRNA (Pfizer-BioNTech)"},
        {"cvx": "140", "display": "Influenza, seasonal, injectable, preservative free"},
        {"cvx": "03",  "display": "MMR (Measles, Mumps, Rubella)"},
        {"cvx": "21",  "display": "Varicella (Chickenpox)"},
        {"cvx": "113", "display": "Tdap (Tetanus, Diphtheria, Pertussis)"},
        {"cvx": "33",  "display": "Pneumococcal polysaccharide PPV23"},
        {"cvx": "187", "display": "Recombinant Zoster (Shingrix)"},
        {"cvx": "52",  "display": "Hepatitis A, adult dosage"},
        {"cvx": "43",  "display": "Hepatitis B, adult dosage"},
    ]

    # LOINC codes for DiagnosticReport panels
    DIAGNOSTIC_PANELS = [
        {"code": "58410-2", "display": "Complete blood count (CBC) panel"},
        {"code": "24323-8", "display": "Comprehensive metabolic panel"},
        {"code": "24362-6", "display": "Renal function panel"},
        {"code": "57021-8", "display": "Lipid panel with direct LDL"},
        {"code": "24357-6", "display": "Urinalysis macro (dipstick) panel"},
    ]

    HIPAA_PHI_IDENTIFIERS = [
        "name", "geographic_data", "dates", "phone_number", "fax_number",
        "email_address", "ssn", "mrn", "health_plan_id", "account_number",
        "certificate_number", "vehicle_identifier", "device_identifier",
        "url", "ip_address", "biometric_id", "photo", "other_unique_id",
    ]

    INSTRUCTION_CATEGORIES = [
        "hipaa_pqc",
        "fhir_security",
        "medical_device",
        "ehr_sharing",
        "homomorphic_analytics",
    ]

    DIFFICULTIES = ["basic", "intermediate", "advanced"]

    # ------------------------------------------------------------------
    # Init & Helpers
    # ------------------------------------------------------------------

    def __init__(self, seed: Optional[int] = None):
        if seed:
            random.seed(seed)
        self._patient_counter = random.randint(100000, 999999)

    def _uuid(self) -> str:
        return str(uuid.uuid4())

    def _next_patient_id(self) -> str:
        self._patient_counter += 1
        return f"PAT-{self._patient_counter}"

    def _random_datetime(self, days_back: int = 90) -> str:
        dt = datetime.now() - timedelta(
            days=random.randint(0, days_back),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )
        return dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    def _random_date(self, start_year: int = 1940, end_year: int = 2010) -> str:
        y = random.randint(start_year, end_year)
        m = random.randint(1, 12)
        d = random.randint(1, 28)
        return f"{y}-{m:02d}-{d:02d}"

    def _random_phone(self) -> str:
        return f"+1-{random.randint(200,999)}-{random.randint(200,999)}-{random.randint(1000,9999)}"

    def _random_address(self) -> Dict:
        return {
            "use": "home",
            "line": [f"{random.randint(100,9999)} {random.choice(self.STREETS)}"],
            "city": random.choice(self.CITIES),
            "state": random.choice(self.STATES),
            "postalCode": f"{random.randint(10000,99999)}",
            "country": "US",
        }

    # ------------------------------------------------------------------
    # Part A  -  FHIR R4 Resource Generators
    # ------------------------------------------------------------------

    def generate_patient(self) -> Dict:
        """Generate a FHIR R4 Patient resource."""
        pid = self._next_patient_id()
        rid = self._uuid()
        first = random.choice(self.FIRST_NAMES)
        last = random.choice(self.LAST_NAMES)
        gender = random.choice(["male", "female"])
        marital = random.choice(["M", "S", "D", "W", "UNK"])

        resource = {
            "resourceType": "Patient",
            "id": rid,
            "meta": {"versionId": "1", "lastUpdated": self._random_datetime(days_back=1)},
            "identifier": [
                {"use": "usual", "type": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v2-0203", "code": "MR"}]},
                 "system": "urn:oid:1.2.3.4.5", "value": pid}
            ],
            "active": True,
            "name": [{"use": "official", "family": last, "given": [first]}],
            "telecom": [
                {"system": "phone", "value": self._random_phone(), "use": "home"},
                {"system": "email", "value": f"{first.lower()}.{last.lower()}@example.com", "use": "home"},
            ],
            "gender": gender,
            "birthDate": self._random_date(),
            "address": [self._random_address()],
            "maritalStatus": {
                "coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-MaritalStatus", "code": marital}]
            },
        }

        metadata = {
            "resource_type": "Patient",
            "resource_id": rid,
            "patient_id": pid,
            "phi_fields": ["name", "birthDate", "address", "telecom", "identifier"],
            "timestamp": datetime.now().isoformat(),
            "hash": hashlib.sha256(json.dumps(resource, sort_keys=True).encode()).hexdigest(),
        }
        return resource, metadata

    def generate_observation(self, patient_ref: str = None) -> Dict:
        """Generate a FHIR R4 Observation (vital sign) resource."""
        rid = self._uuid()
        if not patient_ref:
            patient_ref = f"Patient/{self._uuid()}"

        code, (display, unit, lo, hi) = random.choice(list(self.VITAL_SIGNS.items()))

        # 80 % normal, 20 % abnormal
        if random.random() < 0.8:
            value = round(random.uniform(lo, hi), 1)
        else:
            value = round(random.choice([
                random.uniform(lo * 0.7, lo - 0.1),
                random.uniform(hi + 0.1, hi * 1.3),
            ]), 1)

        interpretation = "N"
        if value < lo:
            interpretation = "L"
        elif value > hi:
            interpretation = "H"

        resource = {
            "resourceType": "Observation",
            "id": rid,
            "status": random.choice(["final", "amended", "preliminary"]),
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                       "code": "vital-signs", "display": "Vital Signs"}]}],
            "code": {"coding": [{"system": "http://loinc.org", "code": code, "display": display}]},
            "subject": {"reference": patient_ref},
            "effectiveDateTime": self._random_datetime(days_back=30),
            "valueQuantity": {"value": value, "unit": unit, "system": "http://unitsofmeasure.org"},
            "interpretation": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                                             "code": interpretation}]}],
            "referenceRange": [{"low": {"value": lo, "unit": unit}, "high": {"value": hi, "unit": unit}}],
        }

        metadata = {
            "resource_type": "Observation",
            "resource_id": rid,
            "loinc_code": code,
            "loinc_display": display,
            "value": value,
            "unit": unit,
            "interpretation": interpretation,
            "normal_range": f"{lo}-{hi}",
            "timestamp": datetime.now().isoformat(),
            "hash": hashlib.sha256(json.dumps(resource, sort_keys=True).encode()).hexdigest(),
        }
        return resource, metadata

    def generate_medication_request(self, patient_ref: str = None) -> Dict:
        """Generate a FHIR R4 MedicationRequest resource."""
        rid = self._uuid()
        if not patient_ref:
            patient_ref = f"Patient/{self._uuid()}"

        med = random.choice(self.MEDICATIONS)
        freq = random.choice([1, 2, 3])
        route_map = {"Oral Tablet": "26643006", "Oral Capsule": "26643006",
                     "Inhalant": "447694001", "MCG Oral": "26643006"}
        route_code = "26643006"
        route_display = "Oral route"
        for k, v in route_map.items():
            if k in med["display"]:
                route_code = v
                route_display = "Oral route" if v == "26643006" else "Respiratory tract route"
                break

        resource = {
            "resourceType": "MedicationRequest",
            "id": rid,
            "status": random.choice(["active", "completed", "stopped", "on-hold"]),
            "intent": "order",
            "medicationCodeableConcept": {
                "coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                            "code": med["rxnorm"], "display": med["display"]}],
            },
            "subject": {"reference": patient_ref},
            "authoredOn": self._random_datetime(days_back=180),
            "requester": {"reference": f"Practitioner/{self._uuid()}",
                          "display": f"Dr. {random.choice(self.LAST_NAMES)}"},
            "dosageInstruction": [{
                "sequence": 1,
                "text": f"Take {freq} time(s) daily",
                "timing": {"repeat": {"frequency": freq, "period": 1, "periodUnit": "d"}},
                "route": {"coding": [{"system": "http://snomed.info/sct",
                                      "code": route_code, "display": route_display}]},
            }],
        }

        metadata = {
            "resource_type": "MedicationRequest",
            "resource_id": rid,
            "rxnorm_code": med["rxnorm"],
            "medication_display": med["display"],
            "frequency_per_day": freq,
            "timestamp": datetime.now().isoformat(),
            "hash": hashlib.sha256(json.dumps(resource, sort_keys=True).encode()).hexdigest(),
        }
        return resource, metadata

    def generate_condition(self, patient_ref: str = None) -> Dict:
        """Generate a FHIR R4 Condition resource."""
        rid = self._uuid()
        if not patient_ref:
            patient_ref = f"Patient/{self._uuid()}"

        cond = random.choice(self.CONDITION_CODES)
        clinical_status = random.choice(["active", "recurrence", "relapse", "inactive", "remission", "resolved"])
        verification = random.choice(["confirmed", "provisional", "differential", "unconfirmed"])

        resource = {
            "resourceType": "Condition",
            "id": rid,
            "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                                            "code": clinical_status}]},
            "verificationStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                                                "code": verification}]},
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-category",
                                       "code": "encounter-diagnosis", "display": "Encounter Diagnosis"}]}],
            "code": {
                "coding": [
                    {"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": cond["icd10"], "display": cond["display"]},
                    {"system": "http://snomed.info/sct", "code": cond["snomed"], "display": cond["display"]},
                ],
            },
            "subject": {"reference": patient_ref},
            "onsetDateTime": self._random_datetime(days_back=365),
        }

        metadata = {
            "resource_type": "Condition",
            "resource_id": rid,
            "icd10_code": cond["icd10"],
            "snomed_code": cond["snomed"],
            "display": cond["display"],
            "clinical_status": clinical_status,
            "verification_status": verification,
            "timestamp": datetime.now().isoformat(),
            "hash": hashlib.sha256(json.dumps(resource, sort_keys=True).encode()).hexdigest(),
        }
        return resource, metadata

    def generate_diagnostic_report(self, patient_ref: str = None) -> Dict:
        """Generate a FHIR R4 DiagnosticReport resource."""
        rid = self._uuid()
        if not patient_ref:
            patient_ref = f"Patient/{self._uuid()}"

        panel = random.choice(self.DIAGNOSTIC_PANELS)
        num_results = random.randint(2, 5)
        result_refs = [{"reference": f"Observation/{self._uuid()}"} for _ in range(num_results)]
        conclusions = [
            "All values within normal limits.",
            "Mild elevation in glucose; recommend repeat fasting glucose in 3 months.",
            "Elevated LDL cholesterol; recommend dietary modifications.",
            "Low hemoglobin; possible iron deficiency anemia.",
            "Elevated creatinine; further renal workup advised.",
        ]

        resource = {
            "resourceType": "DiagnosticReport",
            "id": rid,
            "status": random.choice(["final", "preliminary", "amended"]),
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                                       "code": "LAB", "display": "Laboratory"}]}],
            "code": {"coding": [{"system": "http://loinc.org",
                                  "code": panel["code"], "display": panel["display"]}]},
            "subject": {"reference": patient_ref},
            "effectiveDateTime": self._random_datetime(days_back=14),
            "issued": self._random_datetime(days_back=7),
            "result": result_refs,
            "conclusion": random.choice(conclusions),
        }

        metadata = {
            "resource_type": "DiagnosticReport",
            "resource_id": rid,
            "panel_code": panel["code"],
            "panel_display": panel["display"],
            "result_count": num_results,
            "timestamp": datetime.now().isoformat(),
            "hash": hashlib.sha256(json.dumps(resource, sort_keys=True).encode()).hexdigest(),
        }
        return resource, metadata

    def generate_immunization(self, patient_ref: str = None) -> Dict:
        """Generate a FHIR R4 Immunization resource."""
        rid = self._uuid()
        if not patient_ref:
            patient_ref = f"Patient/{self._uuid()}"

        vax = random.choice(self.VACCINES)
        sites = [
            {"code": "LA", "display": "Left arm"},
            {"code": "RA", "display": "Right arm"},
            {"code": "LT", "display": "Left thigh"},
        ]
        routes = [
            {"code": "IM", "display": "Intramuscular"},
            {"code": "SC", "display": "Subcutaneous"},
        ]
        site = random.choice(sites)
        route = random.choice(routes)

        resource = {
            "resourceType": "Immunization",
            "id": rid,
            "status": "completed",
            "vaccineCode": {
                "coding": [{"system": "http://hl7.org/fhir/sid/cvx",
                            "code": vax["cvx"], "display": vax["display"]}],
            },
            "patient": {"reference": patient_ref},
            "occurrenceDateTime": self._random_datetime(days_back=365),
            "lotNumber": f"LOT-{random.randint(100000,999999)}",
            "site": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ActSite",
                                  "code": site["code"], "display": site["display"]}]},
            "route": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-RouteOfAdministration",
                                   "code": route["code"], "display": route["display"]}]},
            "performer": [{"actor": {"reference": f"Practitioner/{self._uuid()}",
                                     "display": f"Dr. {random.choice(self.LAST_NAMES)}"}}],
        }

        metadata = {
            "resource_type": "Immunization",
            "resource_id": rid,
            "cvx_code": vax["cvx"],
            "vaccine_display": vax["display"],
            "site": site["display"],
            "route": route["display"],
            "timestamp": datetime.now().isoformat(),
            "hash": hashlib.sha256(json.dumps(resource, sort_keys=True).encode()).hexdigest(),
        }
        return resource, metadata

    # ------------------------------------------------------------------
    # Part B  -  Healthcare Security Instruction Pairs
    # ------------------------------------------------------------------

    def _generate_hipaa_pqc_pairs(self, count: int) -> List[Dict]:
        """HIPAA-compliant PQC encryption for PHI."""
        instructions = [
            "Design a PQC encryption strategy for protecting the 18 HIPAA PHI identifiers at rest.",
            "How should ML-KEM-768 be applied to encrypt HL7 FHIR bundles containing PHI?",
            "Create a de-identification pipeline that uses PQC-signed attestation tokens.",
            "What is the minimum necessary standard under HIPAA and how does PQC enforce it?",
            "Design a PQC key hierarchy for a multi-facility health system's PHI vault.",
            "How do you rotate ML-KEM keys for an EHR database without downtime?",
            "Draft a HIPAA Security Rule crosswalk mapping each safeguard to a PQC control.",
            "Explain how CRYSTALS-Dilithium signatures ensure non-repudiation for PHI access logs.",
        ]
        contexts = [
            "Health system: 12 hospitals, 200 clinics, Epic EHR, 8 million patient records. HIPAA compliance audit in 60 days.",
            "Community hospital migrating from on-prem to AWS GovCloud. Must maintain BAA compliance during hybrid phase.",
            "Research institution sharing de-identified datasets with 5 university partners under a DUA.",
            "Telehealth startup transmitting PHI over WebRTC. 500k monthly consultations across all 50 states.",
        ]
        responses = [
            "## PQC Encryption Strategy for HIPAA PHI\n\n### Key Hierarchy\n1. **Root KEM**: ML-KEM-1024 stored in FIPS 140-3 HSM\n2. **Facility KEK**: ML-KEM-768 derived per facility\n3. **Record DEK**: AES-256-GCM wrapped by facility KEK\n\n### PHI Field-Level Encryption\n- **Direct identifiers** (name, SSN, MRN): AES-256-GCM with ML-KEM-768 encapsulated DEK\n- **Quasi-identifiers** (DOB, ZIP): Format-preserving encryption for analytics utility\n- **Clinical data**: Encrypted at rest; decrypted only in authorized session context\n\n### HIPAA Mapping\n- 164.312(a)(2)(iv) Encryption at rest: ML-KEM + AES-256-GCM\n- 164.312(e)(1) Transmission security: TLS 1.3 with ML-KEM hybrid\n- 164.312(c)(1) Integrity: CRYSTALS-Dilithium signatures on all PHI writes\n- 164.312(d) Authentication: PQC-signed SAML assertions for clinician access",
            "## Minimum Necessary with PQC Enforcement\n\n### Principle\nHIPAA 164.502(b) requires covered entities to limit PHI use/disclosure to the minimum necessary.\n\n### PQC Implementation\n1. **Attribute-Based Encryption**: Encrypt PHI fields with CP-ABE policies tied to roles\n2. **PQC Access Tokens**: ML-DSA-signed JWT tokens encode permitted PHI field sets\n3. **Envelope Encryption**: Each role receives a KEM-encapsulated key that decrypts only authorized fields\n4. **Audit Trail**: Every decryption operation logged with Dilithium-signed timestamp\n\n### Example\n- Billing clerk: decrypts {name, DOB, insurance_id, procedure_codes}\n- Nurse: decrypts {name, DOB, vitals, medications, allergies}\n- Researcher: receives de-identified set with k-anonymity verification signed by PQC attestation",
        ]

        pairs = []
        for _ in range(count):
            pairs.append({
                "pair_id": self._uuid(),
                "category": "hipaa_pqc",
                "difficulty": random.choice(self.DIFFICULTIES),
                "instruction": random.choice(instructions),
                "context": random.choice(contexts),
                "response": random.choice(responses),
            })
        return pairs

    def _generate_fhir_security_pairs(self, count: int) -> List[Dict]:
        """SMART on FHIR + PQC, OAuth2 token encryption, FHIR AuditEvent."""
        instructions = [
            "Design a SMART on FHIR authorization flow that uses PQC-signed access tokens.",
            "How should FHIR AuditEvent resources be signed with CRYSTALS-Dilithium?",
            "Create a PQC-secured OAuth2 token exchange for FHIR bulk data export.",
            "What FHIR CapabilityStatement extensions are needed to advertise PQC support?",
            "Design a mutual-TLS + ML-KEM handshake for FHIR server-to-server communication.",
            "How do you protect FHIR Subscription notifications with post-quantum encryption?",
            "Draft a FHIR OperationOutcome response for PQC certificate validation failure.",
            "Explain the threat model for FHIR REST APIs under harvest-now-decrypt-later attacks.",
        ]
        contexts = [
            "FHIR R4 server (HAPI) serving 50 EHR client apps via SMART on FHIR. Authorization server is Keycloak 22.",
            "Health information exchange (HIE) connecting 30 organizations. Currently uses RSA-2048 for token signing.",
            "Patient-facing FHIR portal (Blue Button 2.0) handling 1M API calls/day with OAuth2 bearer tokens.",
            "Clinical trial platform using FHIR Bulk Data to export 500 GB of de-identified research data nightly.",
        ]
        responses = [
            "## SMART on FHIR with PQC Tokens\n\n### Flow\n1. **Authorization Request**: Client redirects to auth server with PKCE (S256)\n2. **Authentication**: Clinician authenticates with WebAuthn + PQC-backed FIDO2 key\n3. **Token Issuance**: Auth server issues JWT signed with ML-DSA-65\n   - Header: `{\"alg\": \"ML-DSA-65\", \"typ\": \"JWT\"}`\n   - Claims include SMART scopes: `patient/*.read`, `launch/patient`\n4. **Token Validation**: FHIR server verifies ML-DSA-65 signature using PQC JWKS endpoint\n5. **API Access**: Bearer token in Authorization header; TLS 1.3 with X25519ML-KEM-768\n\n### PQC AuditEvent\nEvery token use generates a FHIR AuditEvent with Dilithium-signed provenance:\n- agent: authenticated clinician\n- entity: accessed Patient/Observation resources\n- outcome: success/failure with PQC signature over event bundle",
            "## Harvest-Now-Decrypt-Later Threat Model for FHIR\n\n### Threat\nAdversary captures TLS-encrypted FHIR API traffic today, stores it, and decrypts after quantum computer availability (estimated 2030-2035).\n\n### Affected Data\n- Patient demographics (PHI lifetime: permanent)\n- Clinical observations and diagnoses\n- Medication history\n- Genomic data (lifetime value exceeds 50 years)\n\n### Mitigations\n1. **Hybrid TLS**: X25519 + ML-KEM-768 for session keys (NIST SP 800-227)\n2. **PQC Token Signing**: ML-DSA-65 replaces RS256/ES256 in OAuth2 tokens\n3. **Encrypted Search**: Searchable encryption over FHIR indices prevents plaintext exposure\n4. **Forward Secrecy**: Ephemeral ML-KEM encapsulation per session; no long-term decryption key",
        ]

        pairs = []
        for _ in range(count):
            pairs.append({
                "pair_id": self._uuid(),
                "category": "fhir_security",
                "difficulty": random.choice(self.DIFFICULTIES),
                "instruction": random.choice(instructions),
                "context": random.choice(contexts),
                "response": random.choice(responses),
            })
        return pairs

    def _generate_medical_device_pairs(self, count: int) -> List[Dict]:
        """Constrained device PQC for pacemakers, insulin pumps, etc."""
        instructions = [
            "Design a PQC firmware update verification scheme for implanted cardiac pacemakers.",
            "How can ML-KEM-512 fit within the memory constraints of an insulin pump microcontroller?",
            "Create a battery-aware PQC authentication protocol for continuous glucose monitors.",
            "What PQC algorithms are feasible on ARM Cortex-M4 devices with 256 KB flash?",
            "Design a PQC key provisioning flow for IoMT devices during manufacturing.",
            "How should SPHINCS+ be used for long-lived medical device firmware signatures?",
            "Evaluate the energy cost of Kyber key encapsulation vs. ECDH on a pacemaker SoC.",
            "Design a certificate chain for medical device identity using Dilithium and SPHINCS+.",
        ]
        contexts = [
            "Cardiac pacemaker: ARM Cortex-M0+, 64 KB flash, 8 KB RAM, BLE 5.2, 10-year battery life. FDA Class III device.",
            "Insulin pump: ARM Cortex-M4, 256 KB flash, 64 KB RAM, BLE + NFC, 7-day battery. Communicates with CGM every 5 min.",
            "Infusion pump fleet: 2,000 devices across 8 hospitals, Wi-Fi connected, OTA update capability. Current auth: TLS 1.2 with ECDSA-P256.",
            "Surgical robot: Real-time OS, 50 ms command latency budget, Ethernet connected, receives encrypted control commands from surgeon console.",
        ]
        responses = [
            "## PQC Firmware Verification for Pacemakers\n\n### Constraints\n- 64 KB flash (firmware ~40 KB, leaves 24 KB for crypto)\n- 8 KB RAM (peak crypto usage must stay under 4 KB)\n- Battery: every mJ counts; target < 5 mJ per verification\n\n### Recommended Algorithm: SPHINCS+-128s (SHA-256)\n- **Signature size**: 7,856 bytes (fits in flash staging area)\n- **Verification RAM**: ~2.5 KB (within 4 KB budget)\n- **Verification time**: ~50 ms on Cortex-M0+ @ 48 MHz\n- **Energy**: ~3.2 mJ per verification (acceptable for quarterly updates)\n\n### Why Not Dilithium?\n- Dilithium-2 needs ~30 KB RAM for verification (exceeds 8 KB)\n- SPHINCS+ is hash-based, stateless, and conservative for 10+ year device lifetime\n\n### Key Provisioning\n- Public key burned into OTP fuse during manufacturing\n- Firmware images signed by manufacturer HSM with SPHINCS+-128s\n- Device verifies signature before applying update; rollback protection via monotonic counter",
            "## Battery-Aware PQC for Continuous Glucose Monitors\n\n### Protocol Design\n1. **Session Setup** (once per sensor session, ~10 days):\n   - ML-KEM-512 key encapsulation: ~1.8 mJ\n   - Establishes shared AES-128-GCM session key\n2. **Per-Reading Auth** (every 5 minutes):\n   - AES-128-GCM encrypt + MAC: ~0.02 mJ per reading\n   - No asymmetric crypto per reading (battery savings)\n3. **Daily Re-Key**:\n   - Derive new AES key from session key via HKDF\n   - No additional KEM operation needed\n\n### Energy Budget\n- Sensor lifetime: 10 days = 14,400 minutes = 2,880 readings\n- PQC overhead: 1.8 mJ (setup) + 2,880 x 0.02 mJ (readings) = 59.4 mJ total\n- Battery capacity: ~200 mAh @ 3V = 2,160 J\n- PQC fraction: 0.003% of total energy budget\n\n### Fallback\n- If KEM fails (memory pressure), fall back to pre-shared key with AES-CCM",
        ]

        pairs = []
        for _ in range(count):
            pairs.append({
                "pair_id": self._uuid(),
                "category": "medical_device",
                "difficulty": random.choice(self.DIFFICULTIES),
                "instruction": random.choice(instructions),
                "context": random.choice(contexts),
                "response": random.choice(responses),
            })
        return pairs

    def _generate_ehr_sharing_pairs(self, count: int) -> List[Dict]:
        """Proxy re-encryption for EHR, consent management, break-glass access."""
        instructions = [
            "Design a proxy re-encryption scheme for sharing EHRs between hospitals using ML-KEM.",
            "How should PQC-based consent management work for patient-directed data sharing?",
            "Implement a break-glass emergency access protocol with PQC audit guarantees.",
            "Create a multi-party consent framework for sharing pediatric EHRs across state lines.",
            "Design a PQC revocation mechanism for expired patient consent directives.",
            "How does proxy re-encryption reduce key management burden in a health information exchange?",
            "Explain the trust model for a PQC-based consent management smart contract.",
            "Design an emergency department break-glass flow that logs PQC-signed justifications.",
        ]
        contexts = [
            "HIE network: 30 hospitals, 5 million patients. Current sharing via Direct messaging with S/MIME (RSA-2048). Migration budget: 18 months.",
            "Pediatric specialty network: 3 children's hospitals across CA, TX, FL. Parents must consent to cross-state sharing. COPPA + HIPAA requirements.",
            "Emergency department: 80,000 visits/year. Average door-to-doc time 18 min. Cannot tolerate auth delays > 2 seconds for break-glass.",
            "Accountable care organization (ACO) with 200 provider practices sharing care plans. Current consent is paper-based.",
        ]
        responses = [
            "## Proxy Re-Encryption for EHR Sharing\n\n### Architecture\n1. **Patient Key Pair**: ML-KEM-768 generated at enrollment\n   - Private key held on patient FIDO2 hardware token\n   - Public key registered with HIE identity provider\n2. **Encryption**: Originating hospital encrypts EHR bundle under patient public key\n3. **Re-Encryption Key**: Patient generates re-encryption key `rk(A->B)` = transform from hospital A to hospital B\n   - Uses lattice-based PRE (LWE-based unidirectional scheme)\n   - Re-encryption key stored at HIE proxy; proxy cannot decrypt\n4. **Sharing**: HIE proxy transforms ciphertext from A's key to B's key\n5. **Decryption**: Hospital B decrypts with its ML-KEM private key\n\n### Consent Integration\n- Re-encryption key generation requires patient's FIDO2 authentication\n- Key includes expiry timestamp and scope (resource types, date range)\n- Revocation: patient instructs proxy to delete re-encryption key\n- Audit: every re-encryption operation logged with Dilithium-signed event",
            "## Break-Glass Emergency Access with PQC\n\n### Protocol\n1. **Trigger**: Clinician requests emergency access to locked patient record\n2. **Authentication**: Clinician presents hospital badge + PIN (no patient consent available)\n3. **Authorization**: System grants time-limited access (4 hours)\n4. **Decryption**: Emergency DEK released from HSM, wrapped under clinician's ML-KEM public key\n5. **Audit**: Break-glass event signed with CRYSTALS-Dilithium-3:\n   - Clinician identity, patient ID, timestamp, justification code\n   - Countersigned by department supervisor within 24 hours\n\n### Justification Codes (mandatory)\n- EMERG_LIFE_THREAT, EMERG_UNCONSCIOUS, EMERG_PSYCH_CRISIS, EMERG_DISASTER\n\n### Post-Access Review\n- Automated alert to Privacy Officer within 1 hour\n- Patient notified within 48 hours per state law\n- If unjustified: key revoked, HR investigation initiated\n\n### Latency Budget\n- Badge + PIN: 800 ms | HSM key release: 200 ms | KEM encapsulation: 50 ms | Total: < 1.1 seconds",
        ]

        pairs = []
        for _ in range(count):
            pairs.append({
                "pair_id": self._uuid(),
                "category": "ehr_sharing",
                "difficulty": random.choice(self.DIFFICULTIES),
                "instruction": random.choice(instructions),
                "context": random.choice(contexts),
                "response": random.choice(responses),
            })
        return pairs

    def _generate_homomorphic_analytics_pairs(self, count: int) -> List[Dict]:
        """Privacy-preserving vital sign analytics, population health."""
        instructions = [
            "Design a homomorphic encryption pipeline for computing population-level vital sign statistics.",
            "How can FHE be applied to detect sepsis risk scores without decrypting patient data?",
            "Create a privacy-preserving federated learning architecture for hospital vital sign models.",
            "What lattice-based FHE schemes are practical for real-time ICU monitoring analytics?",
            "Design a secure multi-party computation protocol for cross-hospital mortality benchmarking.",
            "How do you validate FHE computation results without revealing plaintext patient data?",
            "Explain the trade-offs between CKKS and BFV schemes for encrypted clinical analytics.",
            "Design an encrypted search index over FHIR Observation resources for population health queries.",
        ]
        contexts = [
            "Population health platform: 2M patients across 50 clinics. Need to compute HbA1c trends, BP distributions, BMI percentiles without centralizing PHI.",
            "ICU monitoring: 200-bed unit, 6 vital signs per patient sampled every 30 seconds. Sepsis prediction model needs encrypted inference within 500 ms.",
            "Clinical research network: 10 hospitals contribute encrypted patient cohort data for drug efficacy study. No raw data leaves any site.",
            "Public health agency requesting encrypted aggregate COVID-19 hospitalization statistics from 100 hospitals. Individual hospital counts must remain confidential.",
        ]
        responses = [
            "## Homomorphic Vital Sign Analytics\n\n### Scheme: CKKS (Approximate HE)\n- **Why CKKS**: Vital signs are continuous floating-point values; CKKS supports approximate arithmetic natively\n- **Parameters**: N=32768, log(q)=218 bits, 128-bit security (lattice-based, PQC-ready)\n\n### Pipeline\n1. **Encryption**: Each clinic encrypts patient vital signs into CKKS ciphertexts\n   - Blood pressure, heart rate, SpO2, temperature batched into SIMD slots\n   - One ciphertext holds ~16,384 patient readings\n2. **Aggregation**: Central server computes encrypted mean, variance, percentiles\n   - Mean: homomorphic addition + scalar multiply (depth 1)\n   - Variance: requires squaring (depth 2) + bootstrapping\n3. **Decryption**: Only the requesting public health authority holds the secret key\n   - Decrypts aggregate statistics, never individual readings\n\n### Performance\n- Encryption: 45 ms per batch of 16K readings\n- Encrypted mean: 2 ms\n- Encrypted variance: 180 ms (includes bootstrapping)\n- Decryption: 30 ms\n\n### Privacy Guarantee\n- Individual patient values computationally hidden under Ring-LWE assumption\n- Combined with differential privacy (epsilon=1.0) for statistical disclosure control",
            "## Encrypted Sepsis Risk Scoring\n\n### Model\n- Logistic regression over 6 vital signs + 4 lab values\n- Coefficients trained on plaintext data; inference on encrypted patient data\n\n### FHE Implementation (BFV Scheme)\n1. **Feature Encoding**: Quantize vital signs to 16-bit integers; pack into BFV plaintext slots\n2. **Encrypted Inference**:\n   - Dot product: homomorphic multiply + rotate-and-sum (depth 1)\n   - Sigmoid approximation: degree-7 polynomial (depth 3)\n3. **Result**: Encrypted risk score [0, 1] returned to treating clinician\n4. **Decryption**: Only clinician's private key can decrypt the score\n\n### Latency\n- Encryption: 15 ms | Inference: 320 ms | Decryption: 10 ms | Total: 345 ms (within 500 ms budget)\n\n### Validation\n- Zero-knowledge proof of correct computation attached to encrypted result\n- Clinician verifies ZKP before trusting the score\n- Audit trail: Dilithium-signed log of model version, input ciphertext hash, output ciphertext hash",
        ]

        pairs = []
        for _ in range(count):
            pairs.append({
                "pair_id": self._uuid(),
                "category": "homomorphic_analytics",
                "difficulty": random.choice(self.DIFFICULTIES),
                "instruction": random.choice(instructions),
                "context": random.choice(contexts),
                "response": random.choice(responses),
            })
        return pairs

    # ------------------------------------------------------------------
    # Dataset Generation
    # ------------------------------------------------------------------

    def generate_dataset(self, num_samples: int, output_dir: str,
                         num_instruction_pairs: int = 200) -> Dict:
        """Generate FHIR R4 resource samples and healthcare security instruction pairs.

        Args:
            num_samples: Number of FHIR resource samples to generate.
            output_dir: Root output directory.
            num_instruction_pairs: Number of instruction pairs to generate.

        Returns:
            Combined dataset metadata dictionary.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # ---- Part A: FHIR Resources ----
        resource_generators = [
            ("patient",            self.generate_patient,            0.20),
            ("observation",        self.generate_observation,        0.25),
            ("medication_request", self.generate_medication_request, 0.15),
            ("condition",          self.generate_condition,          0.15),
            ("diagnostic_report",  self.generate_diagnostic_report,  0.15),
            ("immunization",       self.generate_immunization,       0.10),
        ]

        samples_by_type: Dict[str, int] = {}
        sample_idx = 0

        for res_type, generator, ratio in resource_generators:
            count = int(num_samples * ratio)
            samples_by_type[res_type] = count

            for _ in range(count):
                resource, metadata = generator()
                metadata["sample_index"] = sample_idx

                # Save resource JSON
                res_path = output_path / f"fhir_{res_type}_{sample_idx:06d}.json"
                with open(res_path, "w") as f:
                    json.dump(resource, f, indent=2, default=str)

                # Save metadata
                meta_path = output_path / f"fhir_{res_type}_{sample_idx:06d}.meta.json"
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2, default=str)

                sample_idx += 1

        # ---- Part B: Instruction Pairs ----
        category_generators = {
            "hipaa_pqc":             self._generate_hipaa_pqc_pairs,
            "fhir_security":         self._generate_fhir_security_pairs,
            "medical_device":        self._generate_medical_device_pairs,
            "ehr_sharing":           self._generate_ehr_sharing_pairs,
            "homomorphic_analytics": self._generate_homomorphic_analytics_pairs,
        }

        pairs_per_cat = num_instruction_pairs // len(self.INSTRUCTION_CATEGORIES)
        remainder = num_instruction_pairs % len(self.INSTRUCTION_CATEGORIES)

        all_pairs: List[Dict] = []
        pairs_by_category: Dict[str, int] = {}

        for idx, category in enumerate(self.INSTRUCTION_CATEGORIES):
            cat_count = pairs_per_cat + (1 if idx < remainder else 0)
            pairs = category_generators[category](cat_count)
            all_pairs.extend(pairs)
            pairs_by_category[category] = cat_count

        random.shuffle(all_pairs)

        jsonl_path = output_path / "healthcare_security_instructions.jsonl"
        with open(jsonl_path, "w") as f:
            for pair in all_pairs:
                f.write(json.dumps(pair, default=str) + "\n")

        # ---- Dataset Metadata ----
        dataset_metadata = {
            "protocol": "fhir_r4_healthcare",
            "version": "4.0.1",
            "total_resource_samples": sample_idx,
            "resource_samples_by_type": samples_by_type,
            "total_instruction_pairs": len(all_pairs),
            "instruction_pairs_by_category": pairs_by_category,
            "instruction_difficulty_distribution": {
                d: sum(1 for p in all_pairs if p["difficulty"] == d)
                for d in self.DIFFICULTIES
            },
            "generated_at": datetime.now().isoformat(),
        }

        with open(output_path / "dataset_metadata.json", "w") as f:
            json.dump(dataset_metadata, f, indent=2)

        return dataset_metadata


def main():
    """Generate FHIR R4 healthcare dataset."""
    generator = FHIRHealthcareGenerator(seed=42)
    output_dir = Path(__file__).parent.parent / "healthcare" / "fhir_r4"

    print("Generating FHIR R4 healthcare dataset...")
    metadata = generator.generate_dataset(
        num_samples=1000,
        output_dir=str(output_dir),
        num_instruction_pairs=200,
    )

    print(f"Generated {metadata['total_resource_samples']} FHIR resource samples")
    for res_type, count in metadata["resource_samples_by_type"].items():
        print(f"  - {res_type}: {count} samples")

    print(f"\nGenerated {metadata['total_instruction_pairs']} instruction pairs")
    for category, count in metadata["instruction_pairs_by_category"].items():
        print(f"  - {category}: {count} pairs")

    print(f"\nDifficulty distribution:")
    for difficulty, count in metadata["instruction_difficulty_distribution"].items():
        print(f"  - {difficulty}: {count} pairs")

    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
