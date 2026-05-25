"""
Toll Fraud CDR (Call Detail Record) Generator

Generates labeled Call Detail Records for training toll fraud detection
ML models. Produces JSONL output with realistic normal and fraudulent
call patterns observed in BPO / call center environments.

Fraud types modeled (from toll_fraud.py FraudPatternType):
- IRSF: International Revenue Share Fraud
- PBX_HACK: Unauthorized PBX access
- CALL_TRANSFER: Transfer to premium destination
- WANGIRI: Callback fraud (missed call bait)
- SUBSCRIPTION: Fake account fraud
- CALL_PUMPING: Artificial traffic inflation
- ARBITRAGE: Rate arbitrage exploitation
- BYPASS: SIM box / gateway bypass
- CLIP_MANIPULATION: Caller ID spoofing
- TOLL_FREE_ABUSE: Toll-free number abuse

Distribution: 80% normal traffic, 20% fraud (2% per fraud type)
"""

import json
import random
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import uuid


class TollFraudCDRGenerator:
    """Generate labeled Call Detail Records for toll fraud detection ML training."""

    # -----------------------------------------------------------------------
    # Premium rate prefix database (from toll_fraud.py)
    # -----------------------------------------------------------------------

    PREMIUM_RATE_PREFIXES: Dict[str, List[str]] = {
        "cuba": ["+53"],
        "jamaica": ["+1876", "+1658"],
        "somalia": ["+252"],
        "sierra_leone": ["+232"],
        "guinea": ["+224"],
        "tuvalu": ["+688"],
        "nauru": ["+674"],
        "kiribati": ["+686"],
        "satellite": ["+870", "+871", "+872", "+873"],
        "iprn_europe": ["+388"],
    }

    # Flattened premium prefixes for fast lookup
    ALL_PREMIUM_PREFIXES: List[str] = sorted(
        [p for prefixes in PREMIUM_RATE_PREFIXES.values() for p in prefixes],
        key=len,
        reverse=True,
    )

    # Country names keyed by prefix (for destination_country field)
    PREFIX_TO_COUNTRY: Dict[str, str] = {
        "+53": "CU", "+1876": "JM", "+1658": "JM",
        "+252": "SO", "+232": "SL", "+224": "GN",
        "+688": "TV", "+674": "NR", "+686": "KI",
        "+870": "SAT", "+871": "SAT", "+872": "SAT", "+873": "SAT",
        "+388": "IPRN",
    }

    # US area codes used for normal domestic traffic
    US_AREA_CODES: List[str] = [
        "212", "310", "415", "312", "713", "404", "305", "206",
        "617", "602", "503", "919", "720", "512", "614",
    ]

    # Queue and skill groups
    QUEUE_NAMES: List[str] = ["sales", "support", "billing", "retention", "collections", "tech"]
    SKILL_GROUPS: List[str] = ["english", "spanish", "billing", "tech", "sales"]

    # Toll-free prefixes (US)
    TOLL_FREE_PREFIXES: List[str] = ["+1800", "+1888", "+1877", "+1866"]

    # Arbitrage destination prefixes (countries with rate mismatches)
    ARBITRAGE_PREFIXES: Dict[str, str] = {
        "+91": "IN",    # India
        "+63": "PH",    # Philippines
        "+234": "NG",   # Nigeria
        "+880": "BD",   # Bangladesh
        "+92": "PK",    # Pakistan
    }

    # Fraud type labels
    FRAUD_TYPES: List[str] = [
        "IRSF", "PBX_HACK", "CALL_TRANSFER", "WANGIRI", "SUBSCRIPTION",
        "CALL_PUMPING", "ARBITRAGE", "BYPASS", "CLIP_MANIPULATION",
        "TOLL_FREE_ABUSE",
    ]

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the generator with an optional random seed.

        Args:
            seed: Random seed for reproducibility.
        """
        if seed is not None:
            random.seed(seed)
        self._base_time = datetime(2025, 1, 6, 8, 0, 0)  # Monday 8am

    # -----------------------------------------------------------------------
    # Number generation helpers
    # -----------------------------------------------------------------------

    def _generate_us_number(self) -> str:
        """Generate a random US phone number."""
        area_code = random.choice(self.US_AREA_CODES)
        subscriber = f"{random.randint(2000000, 9999999)}"
        return f"+1{area_code}{subscriber}"

    def _generate_internal_extension(self) -> str:
        """Generate an internal extension number."""
        return f"+1800555{random.randint(1000, 9999)}"

    def _generate_premium_number(self) -> str:
        """Generate a number with a premium rate prefix."""
        prefix = random.choice(self.ALL_PREMIUM_PREFIXES)
        suffix = "".join([str(random.randint(0, 9)) for _ in range(7)])
        return f"{prefix}{suffix}"

    def _generate_toll_free_number(self) -> str:
        """Generate a US toll-free number."""
        prefix = random.choice(self.TOLL_FREE_PREFIXES)
        suffix = f"{random.randint(1000000, 9999999)}"
        return f"{prefix}{suffix}"

    def _get_country_for_prefix(self, number: str) -> str:
        """Look up the country code for a given phone number."""
        for prefix, country in self.PREFIX_TO_COUNTRY.items():
            if number.startswith(prefix):
                return country
        for prefix, country in self.ARBITRAGE_PREFIXES.items():
            if number.startswith(prefix):
                return country
        if number.startswith("+1"):
            return "US"
        return "UNKNOWN"

    # -----------------------------------------------------------------------
    # Timestamp helpers
    # -----------------------------------------------------------------------

    def _generate_business_hours_time(self) -> datetime:
        """Generate a timestamp weighted toward business hours (8am-8pm)."""
        day_offset = random.randint(0, 89)  # ~3 months of data
        base = self._base_time + timedelta(days=day_offset)

        # 80% chance of business hours
        if random.random() < 0.80:
            hour = random.randint(8, 19)
        else:
            hour = random.choice(list(range(0, 8)) + list(range(20, 24)))

        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        return base.replace(hour=hour, minute=minute, second=second)

    def _generate_off_hours_time(self) -> datetime:
        """Generate a timestamp during off-hours (midnight-5am)."""
        day_offset = random.randint(0, 89)
        base = self._base_time + timedelta(days=day_offset)
        hour = random.randint(0, 4)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        return base.replace(hour=hour, minute=minute, second=second)

    # -----------------------------------------------------------------------
    # Normal CDR generation
    # -----------------------------------------------------------------------

    def _generate_normal_cdr(self) -> Dict:
        """
        Generate a normal (non-fraudulent) CDR.

        Normal traffic distribution:
        - 70% inbound, 30% outbound
        - Business hours weighted
        - Duration mostly 60-600 seconds
        - US domestic numbers predominantly
        """
        call_id = f"call-{uuid.uuid4()}"
        agent_id = f"agent-{random.randint(1, 500):03d}"
        tenant_id = f"tenant-{random.randint(1, 50):03d}"

        is_inbound = random.random() < 0.70
        direction = "inbound" if is_inbound else "outbound"

        start_time = self._generate_business_hours_time()

        # Duration: weighted toward 60-600s with some longer calls
        duration_roll = random.random()
        if duration_roll < 0.05:
            duration_seconds = random.randint(5, 30)       # very short
        elif duration_roll < 0.70:
            duration_seconds = random.randint(60, 600)     # typical
        elif duration_roll < 0.90:
            duration_seconds = random.randint(600, 1200)   # longer
        else:
            duration_seconds = random.randint(1200, 1800)  # long

        end_time = start_time + timedelta(seconds=duration_seconds)

        if is_inbound:
            source_number = self._generate_us_number()
            destination_number = self._generate_internal_extension()
        else:
            source_number = self._generate_internal_extension()
            destination_number = self._generate_us_number()

        cost_per_minute = round(random.uniform(0.01, 0.05), 4)
        total_cost = round((duration_seconds / 60.0) * cost_per_minute, 4)

        return {
            "call_id": call_id,
            "agent_id": agent_id,
            "tenant_id": tenant_id,
            "direction": direction,
            "source_number": source_number,
            "destination_number": destination_number,
            "destination_country": "US",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration_seconds,
            "trunk_id": f"trunk-{random.randint(1, 20):03d}",
            "was_transferred": False,
            "transfer_destination": None,
            "sip_response_code": 200,
            "cost_per_minute": cost_per_minute,
            "total_cost": total_cost,
            "queue_name": random.choice(self.QUEUE_NAMES),
            "skill_group": random.choice(self.SKILL_GROUPS),
            "recording_id": f"rec-{uuid.uuid4()}",
            "is_fraud": False,
            "fraud_type": None,
            "fraud_confidence": 0.0,
            "fraud_indicators": [],
        }

    # -----------------------------------------------------------------------
    # Fraud CDR generators (one per fraud type)
    # -----------------------------------------------------------------------

    def _generate_irsf_cdr(self) -> Dict:
        """
        Generate an IRSF (International Revenue Share Fraud) CDR.

        Outbound calls to premium rate prefixes. High cost per minute,
        off-hours timing, destinations in Cuba, Somalia, Pacific islands, etc.
        """
        cdr = self._generate_normal_cdr()
        premium_dest = self._generate_premium_number()

        cdr["direction"] = "outbound"
        cdr["source_number"] = self._generate_internal_extension()
        cdr["destination_number"] = premium_dest
        cdr["destination_country"] = self._get_country_for_prefix(premium_dest)

        # Off-hours timing
        start_time = self._generate_off_hours_time()
        duration_seconds = random.randint(1, 300)
        end_time = start_time + timedelta(seconds=duration_seconds)

        cdr["start_time"] = start_time.isoformat()
        cdr["end_time"] = end_time.isoformat()
        cdr["duration_seconds"] = duration_seconds

        # High cost
        cost_per_minute = round(random.uniform(2.0, 15.0), 4)
        total_cost = round((duration_seconds / 60.0) * cost_per_minute, 4)
        cdr["cost_per_minute"] = cost_per_minute
        cdr["total_cost"] = total_cost

        cdr["sip_response_code"] = 200

        # Fraud labels
        cdr["is_fraud"] = True
        cdr["fraud_type"] = "IRSF"
        cdr["fraud_confidence"] = round(random.uniform(0.85, 0.99), 2)
        cdr["fraud_indicators"] = [
            "premium_rate_destination",
            "high_cost_per_minute",
            "off_hours",
        ]

        return cdr

    def _generate_pbx_hack_cdr(self) -> Dict:
        """
        Generate a PBX_HACK CDR.

        Rapid sequential outbound calls at odd hours, typically 5+
        calls in 5 minutes, all international. Agent ID often reused.
        """
        cdr = self._generate_normal_cdr()

        cdr["direction"] = "outbound"
        cdr["source_number"] = self._generate_internal_extension()
        cdr["destination_number"] = self._generate_premium_number()
        cdr["destination_country"] = self._get_country_for_prefix(
            cdr["destination_number"]
        )

        # Off-hours, midnight-5am
        start_time = self._generate_off_hours_time()
        duration_seconds = random.randint(10, 180)
        end_time = start_time + timedelta(seconds=duration_seconds)

        cdr["start_time"] = start_time.isoformat()
        cdr["end_time"] = end_time.isoformat()
        cdr["duration_seconds"] = duration_seconds

        # Use a small pool of agent IDs to simulate reuse
        cdr["agent_id"] = f"agent-{random.choice([101, 102, 103, 204, 205]):03d}"

        cost_per_minute = round(random.uniform(1.5, 10.0), 4)
        total_cost = round((duration_seconds / 60.0) * cost_per_minute, 4)
        cdr["cost_per_minute"] = cost_per_minute
        cdr["total_cost"] = total_cost

        cdr["is_fraud"] = True
        cdr["fraud_type"] = "PBX_HACK"
        cdr["fraud_confidence"] = round(random.uniform(0.80, 0.97), 2)
        cdr["fraud_indicators"] = [
            "rapid_sequential_calls",
            "off_hours_international",
            "unusual_agent_pattern",
        ]

        return cdr

    def _generate_call_transfer_cdr(self) -> Dict:
        """
        Generate a CALL_TRANSFER fraud CDR.

        Normal inbound call that gets transferred to a premium number.
        """
        cdr = self._generate_normal_cdr()

        # Start as normal inbound call
        cdr["direction"] = "inbound"
        cdr["source_number"] = self._generate_us_number()
        cdr["destination_number"] = self._generate_internal_extension()
        cdr["destination_country"] = "US"

        start_time = self._generate_business_hours_time()
        duration_seconds = random.randint(30, 300)
        end_time = start_time + timedelta(seconds=duration_seconds)

        cdr["start_time"] = start_time.isoformat()
        cdr["end_time"] = end_time.isoformat()
        cdr["duration_seconds"] = duration_seconds

        # The transfer to premium destination is the fraud
        premium_dest = self._generate_premium_number()
        cdr["was_transferred"] = True
        cdr["transfer_destination"] = premium_dest

        cost_per_minute = round(random.uniform(2.0, 12.0), 4)
        total_cost = round((duration_seconds / 60.0) * cost_per_minute, 4)
        cdr["cost_per_minute"] = cost_per_minute
        cdr["total_cost"] = total_cost

        cdr["is_fraud"] = True
        cdr["fraud_type"] = "CALL_TRANSFER"
        cdr["fraud_confidence"] = round(random.uniform(0.88, 0.99), 2)
        cdr["fraud_indicators"] = [
            "transfer_to_premium",
            "high_transfer_cost",
        ]

        return cdr

    def _generate_wangiri_cdr(self) -> Dict:
        """
        Generate a WANGIRI (callback fraud) CDR.

        Short ring (1-3s) from premium number, then a callback.
        We generate the callback CDR (the outbound leg that costs money).
        """
        cdr = self._generate_normal_cdr()

        premium_number = self._generate_premium_number()

        # Randomly produce the missed-call leg or the callback leg
        if random.random() < 0.4:
            # Missed call leg: very short inbound ring from premium source
            cdr["direction"] = "inbound"
            cdr["source_number"] = premium_number
            cdr["destination_number"] = self._generate_internal_extension()
            cdr["destination_country"] = "US"

            duration_seconds = random.randint(1, 3)
            cdr["sip_response_code"] = 487  # Request Terminated (cancelled)
        else:
            # Callback leg: outbound call to the premium number
            cdr["direction"] = "outbound"
            cdr["source_number"] = self._generate_internal_extension()
            cdr["destination_number"] = premium_number
            cdr["destination_country"] = self._get_country_for_prefix(premium_number)

            duration_seconds = random.randint(30, 600)
            cdr["sip_response_code"] = 200

        start_time = self._generate_business_hours_time()
        end_time = start_time + timedelta(seconds=duration_seconds)

        cdr["start_time"] = start_time.isoformat()
        cdr["end_time"] = end_time.isoformat()
        cdr["duration_seconds"] = duration_seconds

        cost_per_minute = round(random.uniform(2.0, 10.0), 4)
        total_cost = round((duration_seconds / 60.0) * cost_per_minute, 4)
        cdr["cost_per_minute"] = cost_per_minute
        cdr["total_cost"] = total_cost

        cdr["is_fraud"] = True
        cdr["fraud_type"] = "WANGIRI"
        cdr["fraud_confidence"] = round(random.uniform(0.75, 0.95), 2)
        cdr["fraud_indicators"] = [
            "short_ring_premium_source",
            "callback_to_premium",
        ]

        return cdr

    def _generate_subscription_cdr(self) -> Dict:
        """
        Generate a SUBSCRIPTION fraud CDR.

        New agent ID making premium calls immediately. High agent
        numbers (agent-9xx) indicate recently created accounts.
        """
        cdr = self._generate_normal_cdr()

        # New account: agent-9xx
        cdr["agent_id"] = f"agent-9{random.randint(0, 99):02d}"

        # First call is international premium within 30 minutes of "login"
        cdr["direction"] = "outbound"
        cdr["source_number"] = self._generate_internal_extension()

        premium_dest = self._generate_premium_number()
        cdr["destination_number"] = premium_dest
        cdr["destination_country"] = self._get_country_for_prefix(premium_dest)

        # Very recent start time (simulating first minutes after login)
        start_time = self._generate_business_hours_time()
        # Shift to be early in a shift
        start_time = start_time.replace(
            hour=random.choice([8, 9, 16, 17]),
            minute=random.randint(0, 29),
        )
        duration_seconds = random.randint(30, 600)
        end_time = start_time + timedelta(seconds=duration_seconds)

        cdr["start_time"] = start_time.isoformat()
        cdr["end_time"] = end_time.isoformat()
        cdr["duration_seconds"] = duration_seconds

        cost_per_minute = round(random.uniform(2.0, 12.0), 4)
        total_cost = round((duration_seconds / 60.0) * cost_per_minute, 4)
        cdr["cost_per_minute"] = cost_per_minute
        cdr["total_cost"] = total_cost

        cdr["is_fraud"] = True
        cdr["fraud_type"] = "SUBSCRIPTION"
        cdr["fraud_confidence"] = round(random.uniform(0.78, 0.96), 2)
        cdr["fraud_indicators"] = [
            "new_account_premium_call",
            "immediate_international",
        ]

        return cdr

    def _generate_call_pumping_cdr(self) -> Dict:
        """
        Generate a CALL_PUMPING CDR.

        Extremely long calls to the same destination, inflating traffic
        volumes. Duration: 3600-14400 seconds (1-4 hours).
        """
        cdr = self._generate_normal_cdr()

        # Pick a repeated destination (a small pool)
        pump_destinations = [
            f"+1876{random.randint(1000000, 1000009)}",
            f"+252{random.randint(1000000, 1000009)}",
            f"+232{random.randint(1000000, 1000009)}",
        ]
        destination = random.choice(pump_destinations)

        cdr["direction"] = "outbound"
        cdr["source_number"] = self._generate_internal_extension()
        cdr["destination_number"] = destination
        cdr["destination_country"] = self._get_country_for_prefix(destination)

        start_time = self._generate_business_hours_time()
        duration_seconds = random.randint(3600, 14400)  # 1-4 hours
        end_time = start_time + timedelta(seconds=duration_seconds)

        cdr["start_time"] = start_time.isoformat()
        cdr["end_time"] = end_time.isoformat()
        cdr["duration_seconds"] = duration_seconds

        cost_per_minute = round(random.uniform(0.50, 5.0), 4)
        total_cost = round((duration_seconds / 60.0) * cost_per_minute, 4)
        cdr["cost_per_minute"] = cost_per_minute
        cdr["total_cost"] = total_cost

        cdr["is_fraud"] = True
        cdr["fraud_type"] = "CALL_PUMPING"
        cdr["fraud_confidence"] = round(random.uniform(0.70, 0.92), 2)
        cdr["fraud_indicators"] = [
            "abnormal_duration",
            "repeated_destination",
            "inflated_traffic",
        ]

        return cdr

    def _generate_arbitrage_cdr(self) -> Dict:
        """
        Generate an ARBITRAGE CDR.

        Calls exploiting rate differentials between carriers. Routed
        through specific trunks to countries with rate mismatches.
        """
        cdr = self._generate_normal_cdr()

        # Pick an arbitrage destination
        arb_prefix = random.choice(list(self.ARBITRAGE_PREFIXES.keys()))
        arb_country = self.ARBITRAGE_PREFIXES[arb_prefix]
        suffix = "".join([str(random.randint(0, 9)) for _ in range(7)])
        destination = f"{arb_prefix}{suffix}"

        cdr["direction"] = "outbound"
        cdr["source_number"] = self._generate_internal_extension()
        cdr["destination_number"] = destination
        cdr["destination_country"] = arb_country

        # Specific trunk pattern suggesting bypass routing
        cdr["trunk_id"] = f"trunk-{random.choice([18, 19, 20]):03d}"

        start_time = self._generate_business_hours_time()
        duration_seconds = random.randint(60, 1800)
        end_time = start_time + timedelta(seconds=duration_seconds)

        cdr["start_time"] = start_time.isoformat()
        cdr["end_time"] = end_time.isoformat()
        cdr["duration_seconds"] = duration_seconds

        # Cost reflects rate differential: billing at higher rate
        cost_per_minute = round(random.uniform(0.08, 0.25), 4)
        total_cost = round((duration_seconds / 60.0) * cost_per_minute, 4)
        cdr["cost_per_minute"] = cost_per_minute
        cdr["total_cost"] = total_cost

        cdr["is_fraud"] = True
        cdr["fraud_type"] = "ARBITRAGE"
        cdr["fraud_confidence"] = round(random.uniform(0.65, 0.88), 2)
        cdr["fraud_indicators"] = [
            "rate_differential",
            "unusual_routing",
            "specific_trunk_pattern",
        ]

        return cdr

    def _generate_bypass_cdr(self) -> Dict:
        """
        Generate a BYPASS (SIM box / gateway bypass) CDR.

        International destinations but unusually low cost, specific
        trunk patterns suggesting a SIM box is terminating the call.
        """
        cdr = self._generate_normal_cdr()

        # International destination
        intl_prefixes = ["+44", "+49", "+33", "+34", "+39", "+81", "+82"]
        prefix = random.choice(intl_prefixes)
        suffix = "".join([str(random.randint(0, 9)) for _ in range(8)])
        destination = f"{prefix}{suffix}"

        country_map = {
            "+44": "GB", "+49": "DE", "+33": "FR", "+34": "ES",
            "+39": "IT", "+81": "JP", "+82": "KR",
        }

        cdr["direction"] = "outbound"
        cdr["source_number"] = self._generate_internal_extension()
        cdr["destination_number"] = destination
        cdr["destination_country"] = country_map.get(prefix, "UNKNOWN")

        # SIM box trunk patterns
        cdr["trunk_id"] = f"trunk-{random.choice([15, 16, 17]):03d}"

        start_time = self._generate_business_hours_time()
        duration_seconds = random.randint(30, 900)
        end_time = start_time + timedelta(seconds=duration_seconds)

        cdr["start_time"] = start_time.isoformat()
        cdr["end_time"] = end_time.isoformat()
        cdr["duration_seconds"] = duration_seconds

        # Below market rate (SIM box routes are cheap)
        cost_per_minute = round(random.uniform(0.005, 0.02), 4)
        total_cost = round((duration_seconds / 60.0) * cost_per_minute, 4)
        cdr["cost_per_minute"] = cost_per_minute
        cdr["total_cost"] = total_cost

        cdr["is_fraud"] = True
        cdr["fraud_type"] = "BYPASS"
        cdr["fraud_confidence"] = round(random.uniform(0.68, 0.90), 2)
        cdr["fraud_indicators"] = [
            "sim_box_routing",
            "below_market_rate",
            "quality_anomaly",
        ]

        return cdr

    def _generate_clip_manipulation_cdr(self) -> Dict:
        """
        Generate a CLIP_MANIPULATION (caller ID spoofing) CDR.

        Source number does not match the trunk origin. Caller ID
        format inconsistencies reveal spoofing.
        """
        cdr = self._generate_normal_cdr()

        # Spoofed caller ID: number appears domestic but trunk is international
        spoofed_source = self._generate_us_number()
        real_destination = self._generate_us_number()

        cdr["direction"] = "outbound"
        cdr["source_number"] = spoofed_source
        cdr["destination_number"] = real_destination
        cdr["destination_country"] = "US"

        # Trunk mismatch: international trunk used for "domestic" call
        cdr["trunk_id"] = f"trunk-{random.choice([19, 20]):03d}"

        start_time = self._generate_business_hours_time()
        duration_seconds = random.randint(30, 600)
        end_time = start_time + timedelta(seconds=duration_seconds)

        cdr["start_time"] = start_time.isoformat()
        cdr["end_time"] = end_time.isoformat()
        cdr["duration_seconds"] = duration_seconds

        cost_per_minute = round(random.uniform(0.01, 0.05), 4)
        total_cost = round((duration_seconds / 60.0) * cost_per_minute, 4)
        cdr["cost_per_minute"] = cost_per_minute
        cdr["total_cost"] = total_cost

        # Insert format anomalies into the source number to signal spoofing
        if random.random() < 0.5:
            # Missing country code format
            cdr["source_number"] = spoofed_source.lstrip("+1")
        else:
            # Source doesn't match internal extension pattern
            cdr["source_number"] = f"+1{random.randint(100, 199)}{random.randint(1000000, 9999999)}"

        cdr["is_fraud"] = True
        cdr["fraud_type"] = "CLIP_MANIPULATION"
        cdr["fraud_confidence"] = round(random.uniform(0.72, 0.94), 2)
        cdr["fraud_indicators"] = [
            "cli_mismatch",
            "spoofed_caller_id",
            "format_anomaly",
        ]

        return cdr

    def _generate_toll_free_abuse_cdr(self) -> Dict:
        """
        Generate a TOLL_FREE_ABUSE CDR.

        Excessive inbound calls to toll-free numbers generating
        revenue for the attacker. Very high volume from the same
        source, short durations.
        """
        cdr = self._generate_normal_cdr()

        # Same source calling toll-free numbers repeatedly
        abuse_source_pool = [
            f"+1{random.choice(['900', '901', '902'])}{random.randint(1000000, 1000009)}"
            for _ in range(3)
        ]
        source = random.choice(abuse_source_pool)

        cdr["direction"] = "inbound"
        cdr["source_number"] = source
        cdr["destination_number"] = self._generate_toll_free_number()
        cdr["destination_country"] = "US"

        start_time = self._generate_business_hours_time()
        # Short durations (pumping volume)
        duration_seconds = random.randint(5, 60)
        end_time = start_time + timedelta(seconds=duration_seconds)

        cdr["start_time"] = start_time.isoformat()
        cdr["end_time"] = end_time.isoformat()
        cdr["duration_seconds"] = duration_seconds

        cost_per_minute = round(random.uniform(0.02, 0.06), 4)
        total_cost = round((duration_seconds / 60.0) * cost_per_minute, 4)
        cdr["cost_per_minute"] = cost_per_minute
        cdr["total_cost"] = total_cost

        cdr["is_fraud"] = True
        cdr["fraud_type"] = "TOLL_FREE_ABUSE"
        cdr["fraud_confidence"] = round(random.uniform(0.70, 0.93), 2)
        cdr["fraud_indicators"] = [
            "toll_free_high_volume",
            "same_source_pattern",
            "revenue_generation",
        ]

        return cdr

    # -----------------------------------------------------------------------
    # Dataset generation
    # -----------------------------------------------------------------------

    def generate_dataset(self, num_samples: int, output_dir: str) -> Dict:
        """
        Generate a complete labeled CDR dataset in JSONL format.

        Distribution: 80% normal, 20% fraud (2% per fraud type).

        Args:
            num_samples: Total number of CDR records to generate.
            output_dir: Directory to write output files into.

        Returns:
            Metadata dictionary describing the generated dataset.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Calculate sample counts
        fraud_per_type = max(1, int(num_samples * 0.02))
        total_fraud = fraud_per_type * len(self.FRAUD_TYPES)
        total_normal = num_samples - total_fraud

        # Split normal traffic into inbound/outbound for metadata tracking
        normal_inbound_count = int(total_normal * 0.70)
        normal_outbound_count = total_normal - normal_inbound_count

        # Build the generation plan
        generation_plan: List[tuple] = []

        # Normal CDRs (the generator internally handles inbound/outbound split)
        for _ in range(total_normal):
            generation_plan.append(("normal", self._generate_normal_cdr))

        # Fraud CDRs: map each type to its generator
        fraud_generators: Dict[str, callable] = {
            "IRSF": self._generate_irsf_cdr,
            "PBX_HACK": self._generate_pbx_hack_cdr,
            "CALL_TRANSFER": self._generate_call_transfer_cdr,
            "WANGIRI": self._generate_wangiri_cdr,
            "SUBSCRIPTION": self._generate_subscription_cdr,
            "CALL_PUMPING": self._generate_call_pumping_cdr,
            "ARBITRAGE": self._generate_arbitrage_cdr,
            "BYPASS": self._generate_bypass_cdr,
            "CLIP_MANIPULATION": self._generate_clip_manipulation_cdr,
            "TOLL_FREE_ABUSE": self._generate_toll_free_abuse_cdr,
        }

        for fraud_type in self.FRAUD_TYPES:
            generator_fn = fraud_generators[fraud_type]
            for _ in range(fraud_per_type):
                generation_plan.append((fraud_type, generator_fn))

        # Shuffle to mix normal and fraud records
        random.shuffle(generation_plan)

        # Track actual counts per type
        samples_by_type: Dict[str, int] = {
            "normal_inbound": 0,
            "normal_outbound": 0,
        }
        for ft in self.FRAUD_TYPES:
            samples_by_type[ft] = 0

        # Generate and write JSONL
        jsonl_path = output_path / "cdrs.jsonl"
        with open(jsonl_path, "w") as f:
            for idx, (label, generator_fn) in enumerate(generation_plan):
                cdr = generator_fn()
                cdr["sample_index"] = idx

                # Compute a record hash for integrity verification
                record_str = json.dumps(cdr, sort_keys=True)
                cdr["record_hash"] = hashlib.sha256(
                    record_str.encode("utf-8")
                ).hexdigest()[:16]

                # Write one JSON record per line
                f.write(json.dumps(cdr, default=str) + "\n")

                # Update counts
                if label == "normal":
                    if cdr["direction"] == "inbound":
                        samples_by_type["normal_inbound"] += 1
                    else:
                        samples_by_type["normal_outbound"] += 1
                else:
                    samples_by_type[label] += 1

        # Calculate actual fraud ratio
        actual_total_fraud = sum(
            v for k, v in samples_by_type.items()
            if k not in ("normal_inbound", "normal_outbound")
        )
        actual_fraud_ratio = round(actual_total_fraud / num_samples, 4) if num_samples > 0 else 0.0

        # Build metadata
        dataset_metadata = {
            "protocol": "toll_fraud_cdr",
            "total_samples": num_samples,
            "samples_by_type": samples_by_type,
            "fraud_ratio": actual_fraud_ratio,
            "generated_at": datetime.now().isoformat(),
        }

        # Save dataset metadata
        meta_path = output_path / "dataset_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(dataset_metadata, f, indent=2)

        return dataset_metadata


def main():
    """Generate toll fraud CDR dataset."""
    generator = TollFraudCDRGenerator(seed=42)

    output_dir = Path(__file__).parent.parent / "security_events" / "bpo_toll_fraud"

    print("Generating Toll Fraud CDR dataset...")
    metadata = generator.generate_dataset(
        num_samples=5000,
        output_dir=str(output_dir),
    )

    print(f"Generated {metadata['total_samples']} samples")
    print(f"Output directory: {output_dir}")
    print(f"Fraud ratio: {metadata['fraud_ratio']:.1%}")
    print("Samples by type:")
    for msg_type, count in metadata["samples_by_type"].items():
        print(f"  - {msg_type}: {count} samples")


if __name__ == "__main__":
    main()
