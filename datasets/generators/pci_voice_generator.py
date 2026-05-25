"""
PCI Voice Compliance Event Generator

Generates labeled PCI-DSS voice channel compliance events for training
PCI compliance detection and enforcement ML models. Produces JSONL output.

Event types generated (from pci_voice.py):
1. dtmf_payment (40%): DTMF-based payment capture events with card data,
   masking mode, digit timing sequences
2. recording_pause_resume (30%): Call recording state transitions during
   PCI scope (ACTIVE -> PAUSED -> ACTIVE)
3. pan_detection (20%): PAN (Primary Account Number) detection alerts from
   real-time data stream analysis
4. compliance_report (10%): Periodic compliance scoring/audit events

Includes:
- Luhn-valid card numbers for Visa/MC/Amex/Discover/JCB
- DTMF digit sequences with per-digit timing
- Recording state transitions with timestamps
- Compliance scoring (0-100 scale)
- Both compliant and non-compliant events for training
"""

import json
import random
import hashlib
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


class PCIVoiceGenerator:
    """Generate labeled PCI voice compliance events for ML training."""

    # Card brands with BIN prefixes and lengths (from pci_voice.py)
    CARD_BRANDS = {
        "visa": {"prefixes": ["4"], "length": 16},
        "mastercard": {"prefixes": ["51", "52", "53", "54", "55"], "length": 16},
        "amex": {"prefixes": ["34", "37"], "length": 15},
        "discover": {"prefixes": ["6011", "65"], "length": 16},
        "jcb": {"prefixes": ["3528", "3529", "353"], "length": 16},
    }

    # DTMF masking modes (from pci_voice.py DTMFMaskingMode)
    MASKING_MODES = ["clamp", "flat_tone", "silence", "replace"]

    # Recording states (from pci_voice.py RecordingState)
    RECORDING_STATES = ["ACTIVE", "PAUSED", "STOPPED", "FAILED", "NOT_STARTED"]

    # PCI scope states (from pci_voice.py PCIScopeState)
    PCI_SCOPE_STATES = ["OUT_OF_SCOPE", "ENTERING_SCOPE", "IN_SCOPE", "EXITING_SCOPE"]

    # Masking targets (from pci_voice.py MaskingTarget)
    MASKING_TARGETS = [
        "CARD_NUMBER", "EXPIRY_DATE", "CVV", "CARDHOLDER_NAME",
        "BILLING_ADDRESS", "ZIP_CODE", "ACCOUNT_NUMBER", "ROUTING_NUMBER",
    ]

    # Compliance violation types
    VIOLATION_TYPES = [
        "recording_not_paused",
        "dtmf_not_masked",
        "pan_in_recording",
        "agent_screen_not_masked",
        "scope_timeout_exceeded",
        "card_data_in_logs",
        "unencrypted_dtmf_relay",
        "missing_pause_event",
    ]

    # Agent / tenant pools
    QUEUES = ["sales", "support", "billing", "retention", "collections"]

    def __init__(self, seed: Optional[int] = None):
        if seed:
            random.seed(seed)
        self.agent_ids = [f"agent_{i:04d}" for i in range(1, 201)]
        self.tenant_ids = [f"tenant_{i:03d}" for i in range(1, 21)]

    def _luhn_checksum(self, partial: str) -> str:
        """Calculate Luhn check digit."""
        digits = [int(d) for d in partial]
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        total = sum(odd_digits)
        for d in even_digits:
            total += sum(divmod(d * 2, 10))
        check = (10 - (total % 10)) % 10
        return str(check)

    def _generate_card_number(self, brand: str = None) -> Dict:
        """Generate a Luhn-valid card number with brand info."""
        if brand is None:
            brand = random.choice(list(self.CARD_BRANDS.keys()))

        info = self.CARD_BRANDS[brand]
        prefix = random.choice(info["prefixes"])
        length = info["length"]

        partial = prefix + "".join(
            str(random.randint(0, 9)) for _ in range(length - len(prefix) - 1)
        )
        full_number = partial + self._luhn_checksum(partial)

        # Masked version (first 6, last 4)
        masked = full_number[:6] + "*" * (length - 10) + full_number[-4:]

        return {
            "card_number_masked": masked,
            "card_brand": brand,
            "card_length": length,
            "bin_prefix": full_number[:6],
            "last_four": full_number[-4:],
            "is_luhn_valid": True,
        }

    def _generate_dtmf_sequence(self, length: int) -> List[Dict]:
        """Generate a DTMF digit sequence with per-digit timing."""
        sequence = []
        base_time = datetime.now() - timedelta(seconds=random.randint(10, 120))
        current_time = base_time

        for i in range(length):
            digit = str(random.randint(0, 9))
            inter_digit_gap = random.uniform(0.3, 2.5)
            duration = random.uniform(0.06, 0.25)
            current_time += timedelta(seconds=inter_digit_gap)

            sequence.append({
                "digit_index": i,
                "digit": digit,
                "timestamp": current_time.isoformat(),
                "duration_ms": round(duration * 1000, 1),
                "inter_digit_gap_ms": round(inter_digit_gap * 1000, 1),
                "low_frequency": random.choice([697, 770, 852, 941]),
                "high_frequency": random.choice([1209, 1336, 1477]),
                "is_masked": True,
            })
            current_time += timedelta(seconds=duration)

        return sequence

    # ------------------------------------------------------------------
    # Event generators
    # ------------------------------------------------------------------

    def generate_dtmf_payment(self) -> Dict:
        """Generate a DTMF-based payment capture event."""
        call_id = f"call-{uuid.uuid4().hex[:12]}"
        agent = random.choice(self.agent_ids)
        tenant = random.choice(self.tenant_ids)
        queue = random.choice(self.QUEUES)
        masking_mode = random.choice(self.MASKING_MODES)
        card_info = self._generate_card_number()

        # Generate DTMF sequences for card number, expiry, CVV
        card_digits = self._generate_dtmf_sequence(card_info["card_length"])
        expiry_digits = self._generate_dtmf_sequence(4)
        cvv_length = 4 if card_info["card_brand"] == "amex" else 3
        cvv_digits = self._generate_dtmf_sequence(cvv_length)

        # Payment result
        is_approved = random.random() < 0.85
        amount = round(random.uniform(1.00, 5000.00), 2)

        # Compliance check
        is_compliant = random.random() < 0.90
        violations = []
        if not is_compliant:
            violations = random.sample(
                self.VIOLATION_TYPES, k=random.randint(1, 3)
            )

        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": "dtmf_payment",
            "timestamp": datetime.now().isoformat(),
            "call_id": call_id,
            "agent_id": agent,
            "tenant_id": tenant,
            "queue": queue,
            "masking_mode": masking_mode,
            "card_info": card_info,
            "dtmf_card_digits": card_digits,
            "dtmf_expiry_digits": expiry_digits,
            "dtmf_cvv_digits": cvv_digits,
            "payment": {
                "amount": amount,
                "currency": "USD",
                "is_approved": is_approved,
                "authorization_code": (
                    f"AUTH{random.randint(100000, 999999)}" if is_approved else ""
                ),
                "decline_reason": (
                    "" if is_approved else random.choice([
                        "insufficient_funds", "card_expired",
                        "invalid_cvv", "do_not_honor", "lost_stolen",
                    ])
                ),
            },
            "pci_scope": {
                "scope_state": "IN_SCOPE",
                "recording_paused": is_compliant or random.random() < 0.5,
                "dtmf_masked": is_compliant or random.random() < 0.5,
                "agent_screen_masked": is_compliant or random.random() < 0.5,
                "scope_duration_seconds": round(random.uniform(15, 120), 1),
            },
            "is_compliant": is_compliant,
            "violations": violations,
        }
        return event

    def generate_recording_pause_resume(self) -> Dict:
        """Generate a recording pause/resume event."""
        call_id = f"call-{uuid.uuid4().hex[:12]}"
        agent = random.choice(self.agent_ids)
        tenant = random.choice(self.tenant_ids)

        # Generate a state transition sequence
        transitions = []
        base_time = datetime.now() - timedelta(seconds=random.randint(30, 300))
        current_state = "ACTIVE"
        current_time = base_time

        # Normal flow: ACTIVE -> PAUSED -> ACTIVE
        # Abnormal: might have FAILED or missing RESUME
        num_transitions = random.randint(2, 6)
        is_compliant = random.random() < 0.88

        for i in range(num_transitions):
            gap = random.uniform(1.0, 60.0)
            current_time += timedelta(seconds=gap)

            if current_state == "ACTIVE":
                next_state = "PAUSED"
                trigger = random.choice([
                    "pci_scope_enter", "agent_manual", "api_request",
                    "dtmf_detect", "ivr_payment_flow",
                ])
            elif current_state == "PAUSED":
                if not is_compliant and random.random() < 0.3:
                    next_state = "FAILED"
                    trigger = "system_error"
                else:
                    next_state = "ACTIVE"
                    trigger = random.choice([
                        "pci_scope_exit", "agent_manual", "api_request",
                        "timeout", "call_ended",
                    ])
            elif current_state == "FAILED":
                next_state = "ACTIVE"
                trigger = "recovery"
            else:
                next_state = "PAUSED"
                trigger = "pci_scope_enter"

            transitions.append({
                "transition_index": i,
                "timestamp": current_time.isoformat(),
                "from_state": current_state,
                "to_state": next_state,
                "trigger": trigger,
                "duration_in_state_seconds": round(gap, 1),
            })
            current_state = next_state

        # Check compliance
        violations = []
        if not is_compliant:
            total_pause_time = sum(
                t["duration_in_state_seconds"]
                for t in transitions if t["from_state"] == "PAUSED"
            )
            if total_pause_time > 120:
                violations.append("excessive_pause_duration")
            if any(t["to_state"] == "FAILED" for t in transitions):
                violations.append("recording_failure_during_pci_scope")
            if not violations:
                violations.append("missing_pause_event")

        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": "recording_pause_resume",
            "timestamp": datetime.now().isoformat(),
            "call_id": call_id,
            "agent_id": agent,
            "tenant_id": tenant,
            "recording_transitions": transitions,
            "total_transitions": len(transitions),
            "final_state": current_state,
            "total_pause_duration_seconds": round(sum(
                t["duration_in_state_seconds"]
                for t in transitions if t["from_state"] == "PAUSED"
            ), 1),
            "is_compliant": is_compliant,
            "violations": violations,
        }
        return event

    def generate_pan_detection(self) -> Dict:
        """Generate a PAN detection alert event."""
        call_id = f"call-{uuid.uuid4().hex[:12]}"
        agent = random.choice(self.agent_ids)
        tenant = random.choice(self.tenant_ids)

        card_info = self._generate_card_number()
        detection_source = random.choice([
            "audio_transcript", "screen_capture", "chat_message",
            "crm_field", "network_trace", "log_file",
        ])

        # True positive or false positive
        is_true_positive = random.random() < 0.75
        confidence = round(
            random.uniform(0.85, 0.99) if is_true_positive
            else random.uniform(0.50, 0.84), 3
        )

        # Action taken
        actions = ["alert_generated", "data_redacted", "session_terminated",
                    "agent_notified", "supervisor_escalated", "logged_only"]
        action_taken = random.choice(actions[:3]) if is_true_positive else random.choice(actions[3:])

        is_compliant = detection_source not in ["log_file", "network_trace"]

        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": "pan_detection",
            "timestamp": datetime.now().isoformat(),
            "call_id": call_id,
            "agent_id": agent,
            "tenant_id": tenant,
            "detection": {
                "source": detection_source,
                "card_info": card_info,
                "confidence": confidence,
                "is_true_positive": is_true_positive,
                "pattern_matched": random.choice([
                    "luhn_16_digit", "luhn_15_digit",
                    "bin_prefix_match", "regex_card_pattern",
                    "ml_classifier",
                ]),
                "context_window": random.choice([
                    "payment_processing", "account_verification",
                    "general_conversation", "data_entry",
                ]),
            },
            "action_taken": action_taken,
            "data_exposed_duration_seconds": (
                round(random.uniform(0.1, 30.0), 1) if is_true_positive
                else 0.0
            ),
            "masking_target": random.choice(self.MASKING_TARGETS),
            "is_compliant": is_compliant,
            "violations": (
                [f"pan_in_{detection_source}"] if not is_compliant else []
            ),
        }
        return event

    def generate_compliance_report(self) -> Dict:
        """Generate a periodic compliance scoring/audit event."""
        tenant = random.choice(self.tenant_ids)
        report_period = random.choice(["hourly", "daily", "weekly", "monthly"])

        # Generate compliance metrics
        total_calls = random.randint(50, 5000)
        pci_scope_calls = int(total_calls * random.uniform(0.05, 0.30))
        compliant_calls = int(pci_scope_calls * random.uniform(0.80, 1.0))

        # Scoring (0-100)
        dtmf_masking_score = random.randint(70, 100)
        recording_pause_score = random.randint(65, 100)
        screen_masking_score = random.randint(70, 100)
        data_protection_score = random.randint(60, 100)
        overall_score = round(
            (dtmf_masking_score + recording_pause_score +
             screen_masking_score + data_protection_score) / 4.0, 1
        )

        # Violations summary
        violation_counts = {}
        for vtype in random.sample(self.VIOLATION_TYPES, k=random.randint(0, 4)):
            violation_counts[vtype] = random.randint(1, 20)

        is_compliant = overall_score >= 85 and not violation_counts

        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": "compliance_report",
            "timestamp": datetime.now().isoformat(),
            "tenant_id": tenant,
            "report_period": report_period,
            "period_start": (
                datetime.now() - timedelta(
                    hours=1 if report_period == "hourly" else
                    24 if report_period == "daily" else
                    168 if report_period == "weekly" else 720
                )
            ).isoformat(),
            "period_end": datetime.now().isoformat(),
            "call_metrics": {
                "total_calls": total_calls,
                "pci_scope_calls": pci_scope_calls,
                "compliant_calls": compliant_calls,
                "non_compliant_calls": pci_scope_calls - compliant_calls,
                "compliance_rate": round(
                    compliant_calls / max(pci_scope_calls, 1) * 100, 1
                ),
            },
            "scores": {
                "overall": overall_score,
                "dtmf_masking": dtmf_masking_score,
                "recording_pause": recording_pause_score,
                "screen_masking": screen_masking_score,
                "data_protection": data_protection_score,
            },
            "violations": violation_counts,
            "total_violation_count": sum(violation_counts.values()),
            "recommendations": self._generate_recommendations(
                dtmf_masking_score, recording_pause_score,
                screen_masking_score, data_protection_score,
            ),
            "is_compliant": is_compliant,
        }
        return event

    def _generate_recommendations(
        self, dtmf: int, recording: int, screen: int, data: int
    ) -> List[str]:
        """Generate compliance improvement recommendations."""
        recs = []
        if dtmf < 90:
            recs.append("Improve DTMF masking coverage - enable flat_tone mode for all PCI scope calls")
        if recording < 90:
            recs.append("Review recording pause/resume automation - ensure API triggers are reliable")
        if screen < 90:
            recs.append("Enable agent screen masking for all CRM fields during PCI scope")
        if data < 85:
            recs.append("Audit data protection controls - check for PAN in logs and network traces")
        if not recs:
            recs.append("All compliance metrics within acceptable range - continue monitoring")
        return recs

    # ------------------------------------------------------------------
    # Dataset generation
    # ------------------------------------------------------------------

    def generate_dataset(self, num_samples: int, output_dir: str) -> Dict:
        """Generate a complete dataset of PCI voice events."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        generators = [
            ("dtmf_payment", self.generate_dtmf_payment, 0.40),
            ("recording_pause_resume", self.generate_recording_pause_resume, 0.30),
            ("pan_detection", self.generate_pan_detection, 0.20),
            ("compliance_report", self.generate_compliance_report, 0.10),
        ]

        dataset_metadata = {
            "protocol": "pci_voice",
            "version": "PCI-DSS 4.0",
            "total_samples": num_samples,
            "samples_by_type": {},
            "generated_at": datetime.now().isoformat(),
        }

        events_file = output_path / "events.jsonl"
        sample_idx = 0

        with open(events_file, "w") as f:
            for event_type, generator, ratio in generators:
                count = int(num_samples * ratio)
                dataset_metadata["samples_by_type"][event_type] = count

                for i in range(count):
                    event = generator()
                    event["sample_index"] = sample_idx
                    f.write(json.dumps(event, default=str) + "\n")
                    sample_idx += 1

        with open(output_path / "dataset_metadata.json", "w") as f:
            json.dump(dataset_metadata, f, indent=2)

        return dataset_metadata


def main():
    """Generate PCI voice compliance dataset."""
    generator = PCIVoiceGenerator(seed=42)
    output_dir = Path(__file__).parent.parent / "security_events" / "bpo_pci_voice"

    print("Generating PCI Voice compliance dataset...")
    metadata = generator.generate_dataset(num_samples=3000, output_dir=str(output_dir))

    print(f"Generated {metadata['total_samples']} events")
    print(f"Output directory: {output_dir}")
    for event_type, count in metadata["samples_by_type"].items():
        print(f"  - {event_type}: {count} events")


if __name__ == "__main__":
    main()
