"""
HL7 v2.x Message Generator

Generates realistic HL7 v2.x messages for training
the protocol discovery and field detection models.

HL7 v2.x Structure:
- Segments separated by carriage return (0x0D)
- Fields separated by pipe (|)
- Components separated by caret (^)
- Subcomponents separated by ampersand (&)
- Repetitions separated by tilde (~)
"""

import json
import random
import string
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import hashlib


class HL7Generator:
    """Generate realistic HL7 v2.x messages for ML training."""

    # Message types
    MESSAGE_TYPES = {
        "ADT": {
            "A01": "Admit/Visit Notification",
            "A02": "Transfer a Patient",
            "A03": "Discharge/End Visit",
            "A04": "Register a Patient",
            "A08": "Update Patient Information",
            "A11": "Cancel Admit",
            "A13": "Cancel Discharge",
        },
        "ORM": {
            "O01": "Order Message",
        },
        "ORU": {
            "R01": "Unsolicited Observation Result",
        },
        "SIU": {
            "S12": "Notification of New Appointment Booking",
            "S13": "Notification of Appointment Rescheduling",
            "S14": "Notification of Appointment Modification",
            "S15": "Notification of Appointment Cancellation",
        },
        "MDM": {
            "T01": "Original Document Notification",
            "T02": "Original Document Notification and Content",
        },
    }

    # Common patient names
    FIRST_NAMES = ["James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph",
                   "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica"]
    LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
                  "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson"]

    # Medical codes
    DIAGNOSIS_CODES = ["I10", "E11.9", "J06.9", "M54.5", "K21.0", "F32.9", "J18.9", "N39.0",
                       "R10.9", "Z00.00", "Z23", "J45.909", "G43.909", "M79.3"]

    PROCEDURE_CODES = ["99213", "99214", "99215", "99203", "99204", "36415", "85025", "80053",
                       "81001", "71046", "93000", "90471", "96372"]

    LOINC_CODES = {
        "2345-7": {"name": "Glucose", "unit": "mg/dL", "range": (70, 200)},
        "2160-0": {"name": "Creatinine", "unit": "mg/dL", "range": (0.6, 1.5)},
        "3094-0": {"name": "BUN", "unit": "mg/dL", "range": (7, 25)},
        "2951-2": {"name": "Sodium", "unit": "mEq/L", "range": (136, 145)},
        "2823-3": {"name": "Potassium", "unit": "mEq/L", "range": (3.5, 5.1)},
        "718-7": {"name": "Hemoglobin", "unit": "g/dL", "range": (12, 17)},
        "4544-3": {"name": "Hematocrit", "unit": "%", "range": (36, 50)},
        "6690-2": {"name": "WBC", "unit": "10*3/uL", "range": (4.5, 11)},
        "777-3": {"name": "Platelet", "unit": "10*3/uL", "range": (150, 400)},
    }

    FACILITY_NAMES = ["General Hospital", "Medical Center", "Regional Medical", "Community Hospital",
                      "University Hospital", "Memorial Hospital", "St. Mary's Hospital"]

    def __init__(self, seed: Optional[int] = None):
        if seed:
            random.seed(seed)
        self.message_control_id = random.randint(100000, 999999)

    def _generate_id(self, prefix: str = "", length: int = 10) -> str:
        """Generate a random ID."""
        chars = string.ascii_uppercase + string.digits
        return prefix + ''.join(random.choices(chars, k=length))

    def _generate_datetime(self, days_back: int = 30) -> str:
        """Generate HL7 datetime format (YYYYMMDDHHMMSS)."""
        dt = datetime.now() - timedelta(
            days=random.randint(0, days_back),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        return dt.strftime("%Y%m%d%H%M%S")

    def _generate_date(self, days_back: int = 30) -> str:
        """Generate HL7 date format (YYYYMMDD)."""
        dt = datetime.now() - timedelta(days=random.randint(0, days_back))
        return dt.strftime("%Y%m%d")

    def _generate_patient_name(self) -> str:
        """Generate HL7 formatted patient name (Last^First^Middle)."""
        last = random.choice(self.LAST_NAMES)
        first = random.choice(self.FIRST_NAMES)
        middle = random.choice(string.ascii_uppercase)
        return f"{last}^{first}^{middle}"

    def _generate_address(self) -> str:
        """Generate HL7 formatted address."""
        street_num = random.randint(100, 9999)
        streets = ["Main St", "Oak Ave", "Maple Dr", "Park Rd", "Washington Blvd", "Lincoln Ave"]
        cities = ["Springfield", "Franklin", "Clinton", "Madison", "Georgetown", "Salem"]
        states = ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI"]

        street = f"{street_num} {random.choice(streets)}"
        city = random.choice(cities)
        state = random.choice(states)
        zip_code = f"{random.randint(10000, 99999)}"

        return f"{street}^^{city}^{state}^{zip_code}^USA"

    def _generate_phone(self) -> str:
        """Generate phone number."""
        area = random.randint(200, 999)
        exchange = random.randint(200, 999)
        number = random.randint(1000, 9999)
        return f"({area}){exchange}-{number}"

    def _get_next_control_id(self) -> str:
        """Get next message control ID."""
        self.message_control_id += 1
        return str(self.message_control_id)

    def generate_msh_segment(self, message_type: str, trigger_event: str) -> Tuple[str, Dict]:
        """Generate MSH (Message Header) segment."""
        control_id = self._get_next_control_id()
        timestamp = self._generate_datetime(days_back=0)
        sending_app = random.choice(["HIS", "LIS", "RIS", "ADT", "LAB"])
        sending_facility = random.choice(self.FACILITY_NAMES).replace(" ", "")
        receiving_app = random.choice(["EMR", "EHR", "HIE", "CDR"])
        receiving_facility = "MAINLAB"

        segment = (
            f"MSH|^~\\&|{sending_app}|{sending_facility}|{receiving_app}|{receiving_facility}|"
            f"{timestamp}||{message_type}^{trigger_event}|{control_id}|P|2.5.1|||AL|NE"
        )

        metadata = {
            "segment": "MSH",
            "fields": [
                {"name": "field_separator", "position": 1, "value": "|"},
                {"name": "encoding_characters", "position": 2, "value": "^~\\&"},
                {"name": "sending_application", "position": 3, "value": sending_app},
                {"name": "sending_facility", "position": 4, "value": sending_facility},
                {"name": "receiving_application", "position": 5, "value": receiving_app},
                {"name": "receiving_facility", "position": 6, "value": receiving_facility},
                {"name": "datetime", "position": 7, "value": timestamp},
                {"name": "message_type", "position": 9, "value": f"{message_type}^{trigger_event}"},
                {"name": "message_control_id", "position": 10, "value": control_id},
                {"name": "processing_id", "position": 11, "value": "P"},
                {"name": "version_id", "position": 12, "value": "2.5.1"},
            ]
        }

        return segment, metadata

    def generate_pid_segment(self) -> Tuple[str, Dict]:
        """Generate PID (Patient Identification) segment."""
        patient_id = self._generate_id("PAT", 8)
        mrn = self._generate_id("MRN", 10)
        name = self._generate_patient_name()
        dob = f"{random.randint(1940, 2010)}{random.randint(1,12):02d}{random.randint(1,28):02d}"
        gender = random.choice(["M", "F"])
        address = self._generate_address()
        phone = self._generate_phone()
        ssn = f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}"

        segment = (
            f"PID|1||{patient_id}^^^HOSP^MR~{mrn}^^^HOSP^MRN||{name}||{dob}|{gender}|||"
            f"{address}||{phone}|||{random.choice(['M', 'S', 'D', 'W'])}||{ssn}"
        )

        metadata = {
            "segment": "PID",
            "fields": [
                {"name": "set_id", "position": 1, "value": "1"},
                {"name": "patient_id", "position": 3, "value": patient_id},
                {"name": "patient_name", "position": 5, "value": name},
                {"name": "date_of_birth", "position": 7, "value": dob},
                {"name": "sex", "position": 8, "value": gender},
                {"name": "address", "position": 11, "value": address},
                {"name": "phone", "position": 13, "value": phone},
                {"name": "ssn", "position": 19, "value": ssn},
            ]
        }

        return segment, metadata

    def generate_pv1_segment(self) -> Tuple[str, Dict]:
        """Generate PV1 (Patient Visit) segment."""
        visit_num = self._generate_id("V", 10)
        patient_class = random.choice(["I", "O", "E", "P", "R"])  # Inpatient, Outpatient, Emergency, etc.
        location = f"{random.choice(['ICU', 'MED', 'SURG', 'ER', 'OBS'])}^{random.randint(100, 999)}^{random.randint(1, 4)}"
        admit_type = random.choice(["E", "U", "R", "N"])
        attending_doc = f"{random.randint(10000, 99999)}^{random.choice(self.LAST_NAMES)}^{random.choice(self.FIRST_NAMES)}^MD"
        admit_datetime = self._generate_datetime(days_back=30)

        segment = (
            f"PV1|1|{patient_class}|{location}|||{location}|{attending_doc}||"
            f"{random.choice(['MED', 'SUR', 'OBS', 'PED'])}|||{admit_type}|||{attending_doc}||"
            f"{visit_num}|||||||||||||||||||||||{admit_datetime}"
        )

        metadata = {
            "segment": "PV1",
            "fields": [
                {"name": "set_id", "position": 1, "value": "1"},
                {"name": "patient_class", "position": 2, "value": patient_class},
                {"name": "assigned_location", "position": 3, "value": location},
                {"name": "attending_doctor", "position": 7, "value": attending_doc},
                {"name": "visit_number", "position": 19, "value": visit_num},
                {"name": "admit_datetime", "position": 44, "value": admit_datetime},
            ]
        }

        return segment, metadata

    def generate_obr_segment(self, set_id: int = 1) -> Tuple[str, Dict]:
        """Generate OBR (Observation Request) segment."""
        placer_order = self._generate_id("ORD", 10)
        filler_order = self._generate_id("FIL", 10)
        loinc = random.choice(list(self.LOINC_CODES.keys()))
        loinc_info = self.LOINC_CODES[loinc]
        order_datetime = self._generate_datetime(days_back=7)
        observation_datetime = self._generate_datetime(days_back=1)
        ordering_provider = f"{random.randint(10000, 99999)}^{random.choice(self.LAST_NAMES)}^{random.choice(self.FIRST_NAMES)}^MD"

        segment = (
            f"OBR|{set_id}|{placer_order}|{filler_order}|{loinc}^{loinc_info['name']}^LN|||"
            f"{order_datetime}|||||||{observation_datetime}|||{ordering_provider}||||||"
            f"{observation_datetime}|||F"
        )

        metadata = {
            "segment": "OBR",
            "fields": [
                {"name": "set_id", "position": 1, "value": str(set_id)},
                {"name": "placer_order_number", "position": 2, "value": placer_order},
                {"name": "filler_order_number", "position": 3, "value": filler_order},
                {"name": "universal_service_id", "position": 4, "value": f"{loinc}^{loinc_info['name']}^LN"},
                {"name": "requested_datetime", "position": 7, "value": order_datetime},
                {"name": "observation_datetime", "position": 14, "value": observation_datetime},
                {"name": "ordering_provider", "position": 16, "value": ordering_provider},
                {"name": "result_status", "position": 25, "value": "F"},
            ]
        }

        return segment, metadata

    def generate_obx_segment(self, set_id: int = 1, obr_loinc: str = None) -> Tuple[str, Dict]:
        """Generate OBX (Observation Result) segment."""
        if obr_loinc and obr_loinc in self.LOINC_CODES:
            loinc = obr_loinc
        else:
            loinc = random.choice(list(self.LOINC_CODES.keys()))

        loinc_info = self.LOINC_CODES[loinc]
        value_type = "NM"  # Numeric
        value = round(random.uniform(loinc_info["range"][0], loinc_info["range"][1]), 1)
        units = loinc_info["unit"]
        ref_range = f"{loinc_info['range'][0]}-{loinc_info['range'][1]}"
        abnormal_flag = ""
        if value < loinc_info["range"][0]:
            abnormal_flag = "L"
        elif value > loinc_info["range"][1]:
            abnormal_flag = "H"

        observation_datetime = self._generate_datetime(days_back=1)

        segment = (
            f"OBX|{set_id}|{value_type}|{loinc}^{loinc_info['name']}^LN||{value}|{units}|"
            f"{ref_range}|{abnormal_flag}|||F|||{observation_datetime}"
        )

        metadata = {
            "segment": "OBX",
            "fields": [
                {"name": "set_id", "position": 1, "value": str(set_id)},
                {"name": "value_type", "position": 2, "value": value_type},
                {"name": "observation_identifier", "position": 3, "value": f"{loinc}^{loinc_info['name']}^LN"},
                {"name": "observation_value", "position": 5, "value": str(value)},
                {"name": "units", "position": 6, "value": units},
                {"name": "reference_range", "position": 7, "value": ref_range},
                {"name": "abnormal_flags", "position": 8, "value": abnormal_flag},
                {"name": "observation_result_status", "position": 11, "value": "F"},
                {"name": "observation_datetime", "position": 14, "value": observation_datetime},
            ]
        }

        return segment, metadata

    def generate_dg1_segment(self, set_id: int = 1) -> Tuple[str, Dict]:
        """Generate DG1 (Diagnosis) segment."""
        diagnosis_code = random.choice(self.DIAGNOSIS_CODES)
        diagnosis_datetime = self._generate_datetime(days_back=30)
        diagnosis_type = random.choice(["A", "W", "F"])  # Admitting, Working, Final
        diagnosing_clinician = f"{random.randint(10000, 99999)}^{random.choice(self.LAST_NAMES)}^{random.choice(self.FIRST_NAMES)}^MD"

        segment = (
            f"DG1|{set_id}|ICD10|{diagnosis_code}^Diagnosis Description^ICD10|||{diagnosis_type}||||"
            f"{diagnosing_clinician}||{diagnosis_datetime}"
        )

        metadata = {
            "segment": "DG1",
            "fields": [
                {"name": "set_id", "position": 1, "value": str(set_id)},
                {"name": "diagnosis_coding_method", "position": 2, "value": "ICD10"},
                {"name": "diagnosis_code", "position": 3, "value": diagnosis_code},
                {"name": "diagnosis_type", "position": 6, "value": diagnosis_type},
                {"name": "diagnosing_clinician", "position": 16, "value": diagnosing_clinician},
                {"name": "diagnosis_datetime", "position": 19, "value": diagnosis_datetime},
            ]
        }

        return segment, metadata

    def generate_adt_a01(self) -> Tuple[bytes, Dict]:
        """Generate ADT^A01 (Admit) message."""
        segments = []
        metadata = {
            "protocol": "hl7_v2",
            "message_type": "ADT^A01",
            "timestamp": datetime.now().isoformat(),
            "trigger_event": "A01",
            "segments": [],
            "fields": []
        }

        # MSH
        msh, msh_meta = self.generate_msh_segment("ADT", "A01")
        segments.append(msh)
        metadata["segments"].append(msh_meta)

        # EVN
        evn = f"EVN|A01|{self._generate_datetime(days_back=0)}"
        segments.append(evn)
        metadata["segments"].append({"segment": "EVN"})

        # PID
        pid, pid_meta = self.generate_pid_segment()
        segments.append(pid)
        metadata["segments"].append(pid_meta)

        # PV1
        pv1, pv1_meta = self.generate_pv1_segment()
        segments.append(pv1)
        metadata["segments"].append(pv1_meta)

        # DG1 (1-3 diagnoses)
        for i in range(random.randint(1, 3)):
            dg1, dg1_meta = self.generate_dg1_segment(i + 1)
            segments.append(dg1)
            metadata["segments"].append(dg1_meta)

        message = "\r".join(segments) + "\r"
        metadata["segment_count"] = len(segments)
        metadata["hash"] = hashlib.sha256(message.encode()).hexdigest()

        return message.encode("ascii"), metadata

    def generate_oru_r01(self) -> Tuple[bytes, Dict]:
        """Generate ORU^R01 (Lab Result) message."""
        segments = []
        metadata = {
            "protocol": "hl7_v2",
            "message_type": "ORU^R01",
            "timestamp": datetime.now().isoformat(),
            "trigger_event": "R01",
            "segments": [],
            "fields": []
        }

        # MSH
        msh, msh_meta = self.generate_msh_segment("ORU", "R01")
        segments.append(msh)
        metadata["segments"].append(msh_meta)

        # PID
        pid, pid_meta = self.generate_pid_segment()
        segments.append(pid)
        metadata["segments"].append(pid_meta)

        # PV1
        pv1, pv1_meta = self.generate_pv1_segment()
        segments.append(pv1)
        metadata["segments"].append(pv1_meta)

        # OBR + OBX groups (1-5 test results)
        for i in range(random.randint(1, 5)):
            obr, obr_meta = self.generate_obr_segment(i + 1)
            segments.append(obr)
            metadata["segments"].append(obr_meta)

            # 1-3 OBX per OBR
            for j in range(random.randint(1, 3)):
                obx, obx_meta = self.generate_obx_segment(j + 1)
                segments.append(obx)
                metadata["segments"].append(obx_meta)

        message = "\r".join(segments) + "\r"
        metadata["segment_count"] = len(segments)
        metadata["hash"] = hashlib.sha256(message.encode()).hexdigest()

        return message.encode("ascii"), metadata

    def generate_orm_o01(self) -> Tuple[bytes, Dict]:
        """Generate ORM^O01 (Order) message."""
        segments = []
        metadata = {
            "protocol": "hl7_v2",
            "message_type": "ORM^O01",
            "timestamp": datetime.now().isoformat(),
            "trigger_event": "O01",
            "segments": [],
            "fields": []
        }

        # MSH
        msh, msh_meta = self.generate_msh_segment("ORM", "O01")
        segments.append(msh)
        metadata["segments"].append(msh_meta)

        # PID
        pid, pid_meta = self.generate_pid_segment()
        segments.append(pid)
        metadata["segments"].append(pid_meta)

        # PV1
        pv1, pv1_meta = self.generate_pv1_segment()
        segments.append(pv1)
        metadata["segments"].append(pv1_meta)

        # ORC + OBR (1-3 orders)
        for i in range(random.randint(1, 3)):
            orc_order = self._generate_id("ORD", 10)
            orc = f"ORC|NW|{orc_order}||||||{self._generate_datetime()}|||{random.randint(10000, 99999)}^{random.choice(self.LAST_NAMES)}^{random.choice(self.FIRST_NAMES)}^MD"
            segments.append(orc)
            metadata["segments"].append({"segment": "ORC", "fields": [{"name": "order_control", "value": "NW"}]})

            obr, obr_meta = self.generate_obr_segment(i + 1)
            segments.append(obr)
            metadata["segments"].append(obr_meta)

        message = "\r".join(segments) + "\r"
        metadata["segment_count"] = len(segments)
        metadata["hash"] = hashlib.sha256(message.encode()).hexdigest()

        return message.encode("ascii"), metadata

    def generate_dataset(self, num_samples: int, output_dir: str) -> Dict:
        """Generate a complete dataset of HL7 messages."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        generators = [
            ("adt_a01", self.generate_adt_a01, 0.40),
            ("oru_r01", self.generate_oru_r01, 0.40),
            ("orm_o01", self.generate_orm_o01, 0.20),
        ]

        dataset_metadata = {
            "protocol": "hl7_v2",
            "version": "2.5.1",
            "total_samples": num_samples,
            "samples_by_type": {},
            "generated_at": datetime.now().isoformat(),
        }

        sample_idx = 0
        for msg_type, generator, ratio in generators:
            count = int(num_samples * ratio)
            dataset_metadata["samples_by_type"][msg_type] = count

            for i in range(count):
                message, metadata = generator()
                metadata["sample_index"] = sample_idx

                bin_path = output_path / f"hl7_{msg_type}_{sample_idx:06d}.bin"
                with open(bin_path, "wb") as f:
                    f.write(message)

                meta_path = output_path / f"hl7_{msg_type}_{sample_idx:06d}.json"
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2, default=str)

                sample_idx += 1

        with open(output_path / "dataset_metadata.json", "w") as f:
            json.dump(dataset_metadata, f, indent=2)

        return dataset_metadata


def main():
    """Generate HL7 v2.x dataset."""
    generator = HL7Generator(seed=42)
    output_dir = Path(__file__).parent.parent / "protocols" / "hl7"

    print("Generating HL7 v2.x dataset...")
    metadata = generator.generate_dataset(num_samples=1000, output_dir=str(output_dir))

    print(f"Generated {metadata['total_samples']} samples")
    print(f"Output directory: {output_dir}")
    for msg_type, count in metadata["samples_by_type"].items():
        print(f"  - {msg_type}: {count} samples")


if __name__ == "__main__":
    main()
