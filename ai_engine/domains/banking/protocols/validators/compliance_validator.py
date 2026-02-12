"""
Compliance Validators

Banking compliance validation including:
- AML (Anti-Money Laundering) checks
- Sanctions screening
- OFAC compliance
- Transaction monitoring rules
"""

import re
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

from ai_engine.domains.banking.protocols.validators.base_validator import (
    BaseValidator,
    ValidationResult,
    ValidationSeverity,
)


class ComplianceValidator(BaseValidator):
    """
    Base compliance validator with common checks.

    Provides foundation for:
    - Amount threshold monitoring
    - Pattern detection
    - Geographic risk assessment
    """

    # Standard thresholds (USD)
    CTR_THRESHOLD = 10_000  # Currency Transaction Report threshold
    SAR_THRESHOLD = 5_000  # Suspicious Activity Report review threshold
    WIRE_REPORTING_THRESHOLD = 3_000  # Wire transfer reporting threshold

    # High-risk countries (sample list - should be maintained externally)
    HIGH_RISK_COUNTRIES = {
        "KP",
        "IR",
        "SY",
        "CU",
        "VE",
        "MM",
        "BY",
        "RU",
        "AF",
        "YE",
        "LY",
        "SO",
        "SD",
        "ZW",
    }

    def __init__(self, strict: bool = True, custom_thresholds: Dict[str, float] = None):
        """
        Initialize the compliance validator.

        Args:
            strict: If True, treat warnings as errors
            custom_thresholds: Override default thresholds
        """
        super().__init__(strict)
        self.thresholds = {
            "ctr": self.CTR_THRESHOLD,
            "sar": self.SAR_THRESHOLD,
            "wire": self.WIRE_REPORTING_THRESHOLD,
        }
        if custom_thresholds:
            self.thresholds.update(custom_thresholds)

    @property
    def name(self) -> str:
        return "ComplianceValidator"

    @property
    def version(self) -> str:
        return "1.0"

    def validate(self, data: Any) -> ValidationResult:
        """
        Validate transaction for compliance requirements.

        Args:
            data: Transaction data (dict or object)

        Returns:
            ValidationResult with any compliance flags
        """
        result = self._create_result()

        # Extract transaction details
        amount = self._extract_amount(data)
        parties = self._extract_parties(data)
        tx_type = self._extract_type(data)

        # Amount threshold checks
        if amount is not None:
            self._check_amount_thresholds(amount, tx_type, result)

        # Geographic risk checks
        if parties:
            self._check_geographic_risk(parties, result)

        # Pattern checks (structuring detection)
        self._check_patterns(data, result)

        return result

    def _extract_amount(self, data: Any) -> Optional[float]:
        """Extract amount from transaction data."""
        if isinstance(data, dict):
            return data.get("amount")
        elif hasattr(data, "amount"):
            amount = data.amount
            if isinstance(amount, Decimal):
                return float(amount)
            return amount
        return None

    def _extract_parties(self, data: Any) -> List[Dict]:
        """Extract party information from transaction data."""
        parties = []

        if isinstance(data, dict):
            for key in ["originator", "beneficiary", "sender", "receiver"]:
                if key in data:
                    parties.append(data[key] if isinstance(data[key], dict) else {"name": str(data[key])})
        elif hasattr(data, "originator") or hasattr(data, "beneficiary"):
            if hasattr(data, "originator") and data.originator:
                parties.append(
                    {
                        "name": getattr(data.originator, "name", ""),
                        "country": getattr(getattr(data.originator, "address", None), "country", ""),
                    }
                )
            if hasattr(data, "beneficiary") and data.beneficiary:
                parties.append(
                    {
                        "name": getattr(data.beneficiary, "name", ""),
                        "country": getattr(getattr(data.beneficiary, "address", None), "country", ""),
                    }
                )

        return parties

    def _extract_type(self, data: Any) -> str:
        """Extract transaction type."""
        if isinstance(data, dict):
            return data.get("type", "unknown")
        elif hasattr(data, "business_function_code"):
            bfc = data.business_function_code
            return bfc.code if hasattr(bfc, "code") else str(bfc)
        return "unknown"

    def _check_amount_thresholds(self, amount: float, tx_type: str, result: ValidationResult) -> None:
        """Check amount against reporting thresholds."""
        # CTR threshold
        if amount >= self.thresholds["ctr"]:
            result.add_warning(
                "COMPLIANCE_CTR_THRESHOLD",
                f"Transaction amount ${amount:,.2f} meets CTR threshold (${self.thresholds['ctr']:,.2f})",
                field_name="amount",
                threshold="CTR",
                amount=amount,
            )
            result.metadata["requires_ctr"] = True

        # SAR review threshold
        if amount >= self.thresholds["sar"] and amount < self.thresholds["ctr"]:
            result.add_warning(
                "COMPLIANCE_SAR_REVIEW",
                f"Transaction amount ${amount:,.2f} may require SAR review",
                field_name="amount",
                threshold="SAR",
                amount=amount,
            )
            result.metadata["sar_review_recommended"] = True

        # Wire reporting threshold
        if tx_type in ["wire", "fedwire", "CTR", "CTP", "COV"]:
            if amount >= self.thresholds["wire"]:
                result.add_warning(
                    "COMPLIANCE_WIRE_REPORTING",
                    f"Wire transfer ${amount:,.2f} meets reporting threshold",
                    field_name="amount",
                    threshold="WIRE",
                    amount=amount,
                )
                result.metadata["requires_wire_reporting"] = True

    def _check_geographic_risk(self, parties: List[Dict], result: ValidationResult) -> None:
        """Check parties for geographic risk factors."""
        for i, party in enumerate(parties):
            country = party.get("country", "").upper()

            if country in self.HIGH_RISK_COUNTRIES:
                result.add_warning(
                    "COMPLIANCE_HIGH_RISK_COUNTRY",
                    f"Party involves high-risk country: {country}",
                    field=f"parties[{i}]/country",
                    country=country,
                    risk_level="HIGH",
                )
                result.metadata["high_risk_country_involved"] = True

    def _check_patterns(self, data: Any, result: ValidationResult) -> None:
        """Check for suspicious patterns (stub for pattern detection)."""
        # This would typically integrate with a pattern detection system
        # For now, just check for basic structuring indicators
        amount = self._extract_amount(data)

        if amount is not None:
            # Check for amounts just below reporting threshold (potential structuring)
            ctr_threshold = self.thresholds["ctr"]
            if ctr_threshold * 0.9 <= amount < ctr_threshold:
                result.add_warning(
                    "COMPLIANCE_POTENTIAL_STRUCTURING",
                    f"Amount ${amount:,.2f} is just below CTR threshold - potential structuring",
                    field_name="amount",
                    pattern="STRUCTURING_INDICATOR",
                )


class AMLValidator(ComplianceValidator):
    """
    Anti-Money Laundering (AML) specific validator.

    Enhanced checks for:
    - Transaction structuring detection
    - Velocity monitoring
    - Risk scoring
    """

    def __init__(self, strict: bool = True, velocity_window_hours: int = 24):
        """
        Initialize the AML validator.

        Args:
            strict: If True, treat warnings as errors
            velocity_window_hours: Time window for velocity checks
        """
        super().__init__(strict)
        self.velocity_window = timedelta(hours=velocity_window_hours)

    @property
    def name(self) -> str:
        return "AMLValidator"

    def validate(self, data: Any, transaction_history: List[Dict] = None) -> ValidationResult:
        """
        Validate transaction for AML compliance.

        Args:
            data: Transaction data
            transaction_history: Optional list of recent transactions for velocity checks

        Returns:
            ValidationResult with AML flags
        """
        # Run base compliance checks
        result = super().validate(data)

        # Enhanced AML checks
        if transaction_history:
            self._check_velocity(data, transaction_history, result)
            self._check_structuring_pattern(data, transaction_history, result)

        # Calculate risk score
        risk_score = self._calculate_risk_score(data, result)
        result.metadata["aml_risk_score"] = risk_score

        if risk_score >= 70:
            result.add_warning(
                "AML_HIGH_RISK_SCORE",
                f"Transaction has high AML risk score: {risk_score}",
                risk_score=risk_score,
            )
        elif risk_score >= 50:
            result.add_warning(
                "AML_ELEVATED_RISK_SCORE",
                f"Transaction has elevated AML risk score: {risk_score}",
                risk_score=risk_score,
            )

        return result

    def _check_velocity(self, data: Any, history: List[Dict], result: ValidationResult) -> None:
        """Check transaction velocity against recent history."""
        current_amount = self._extract_amount(data) or 0
        current_time = datetime.now()

        # Sum amounts in velocity window
        total_amount = current_amount
        tx_count = 1

        for tx in history:
            tx_time = tx.get("timestamp")
            if tx_time:
                if isinstance(tx_time, str):
                    tx_time = datetime.fromisoformat(tx_time)

                if current_time - tx_time <= self.velocity_window:
                    total_amount += tx.get("amount", 0)
                    tx_count += 1

        # Check if aggregate exceeds CTR threshold
        if total_amount >= self.thresholds["ctr"]:
            result.add_warning(
                "AML_VELOCITY_THRESHOLD",
                f"Aggregate amount ${total_amount:,.2f} in {self.velocity_window.total_seconds() / 3600:.0f}h exceeds CTR threshold",
                field_name="velocity",
                aggregate_amount=total_amount,
                transaction_count=tx_count,
            )
            result.metadata["velocity_ctr_exceeded"] = True

    def _check_structuring_pattern(self, data: Any, history: List[Dict], result: ValidationResult) -> None:
        """Detect potential structuring patterns."""
        current_amount = self._extract_amount(data) or 0
        ctr_threshold = self.thresholds["ctr"]

        # Count transactions just below threshold
        below_threshold_count = 0
        if ctr_threshold * 0.8 <= current_amount < ctr_threshold:
            below_threshold_count = 1

        for tx in history:
            tx_amount = tx.get("amount", 0)
            if ctr_threshold * 0.8 <= tx_amount < ctr_threshold:
                below_threshold_count += 1

        # Multiple transactions just below threshold is suspicious
        if below_threshold_count >= 3:
            result.add_warning(
                "AML_STRUCTURING_PATTERN",
                f"Detected {below_threshold_count} transactions just below CTR threshold - potential structuring",
                pattern="MULTI_BELOW_THRESHOLD",
                count=below_threshold_count,
            )
            result.metadata["structuring_pattern_detected"] = True

    def _calculate_risk_score(self, data: Any, result: ValidationResult) -> int:
        """Calculate AML risk score (0-100)."""
        score = 0

        # Amount-based factors
        amount = self._extract_amount(data) or 0
        if amount >= self.thresholds["ctr"]:
            score += 20
        elif amount >= self.thresholds["sar"]:
            score += 10

        # Geographic risk
        if result.metadata.get("high_risk_country_involved"):
            score += 30

        # Velocity concerns
        if result.metadata.get("velocity_ctr_exceeded"):
            score += 20

        # Structuring indicators
        if result.metadata.get("structuring_pattern_detected"):
            score += 25
        elif any(w.code == "COMPLIANCE_POTENTIAL_STRUCTURING" for w in result.warnings):
            score += 10

        return min(score, 100)


class SanctionsValidator(BaseValidator):
    """
    Sanctions screening validator.

    Checks transactions against:
    - OFAC SDN list
    - EU sanctions list
    - UN sanctions list
    - Other country-specific lists
    """

    # Sample sanctioned terms (in production, use actual sanctions lists)
    SAMPLE_BLOCKED_TERMS = {
        # These are examples - real implementation would use OFAC data
        "BLOCKED_ENTITY",
        "SANCTIONED_BANK",
        "PROHIBITED_PERSON",
    }

    # Sanctioned jurisdictions
    SANCTIONED_JURISDICTIONS = {
        "KP": "North Korea",
        "IR": "Iran",
        "SY": "Syria",
        "CU": "Cuba",
    }

    def __init__(self, strict: bool = True, sanctions_lists: List[str] = None):
        """
        Initialize the sanctions validator.

        Args:
            strict: If True, treat matches as errors (not warnings)
            sanctions_lists: List of sanctions lists to check against
        """
        super().__init__(strict)
        self.sanctions_lists = sanctions_lists or ["OFAC", "EU", "UN"]

    @property
    def name(self) -> str:
        return "SanctionsValidator"

    @property
    def version(self) -> str:
        return "1.0"

    def validate(self, data: Any) -> ValidationResult:
        """
        Screen transaction against sanctions lists.

        Args:
            data: Transaction data

        Returns:
            ValidationResult with any sanctions matches
        """
        result = self._create_result()

        # Extract parties for screening
        parties = self._extract_all_parties(data)

        # Screen each party
        for party_type, party_info in parties.items():
            matches = self._screen_party(party_info)

            for match in matches:
                severity = ValidationSeverity.ERROR if self.strict else ValidationSeverity.WARNING
                result.add_error(
                    "SANCTIONS_MATCH",
                    f"Potential sanctions match for {party_type}: {match['match_type']}",
                    field=party_type,
                    severity=severity,
                    match_type=match["match_type"],
                    matched_value=match["matched_value"],
                    list_name=match.get("list_name", "Unknown"),
                )
                result.metadata[f"sanctions_match_{party_type}"] = True

        # Check jurisdictions
        self._check_jurisdictions(data, result)

        # Set overall sanctions status
        result.metadata["sanctions_screened"] = True
        result.metadata["sanctions_lists_checked"] = self.sanctions_lists
        result.metadata["screening_timestamp"] = datetime.now().isoformat()

        return result

    def _extract_all_parties(self, data: Any) -> Dict[str, Dict]:
        """Extract all party information for screening."""
        parties = {}

        if isinstance(data, dict):
            for key in [
                "originator",
                "beneficiary",
                "sender",
                "receiver",
                "originator_fi",
                "beneficiary_fi",
                "intermediary_fi",
            ]:
                if key in data and data[key]:
                    parties[key] = data[key] if isinstance(data[key], dict) else {"name": str(data[key])}
        else:
            # Handle message objects
            for attr in [
                "originator",
                "beneficiary",
                "sender",
                "receiver",
                "originator_fi",
                "beneficiary_fi",
                "intermediary_fi",
            ]:
                obj = getattr(data, attr, None)
                if obj:
                    parties[attr] = {
                        "name": getattr(obj, "name", ""),
                        "identifier": getattr(obj, "identifier", ""),
                        "address": self._extract_address(obj),
                    }

        return parties

    def _extract_address(self, obj: Any) -> Dict:
        """Extract address from party object."""
        address = getattr(obj, "address", None)
        if address:
            return {
                "line1": getattr(address, "line1", ""),
                "line2": getattr(address, "line2", ""),
                "city": getattr(address, "city", ""),
                "country": getattr(address, "country", ""),
            }
        return {}

    def _screen_party(self, party_info: Dict) -> List[Dict]:
        """Screen party against sanctions lists."""
        matches = []

        # Get searchable text
        name = party_info.get("name", "").upper()
        identifier = party_info.get("identifier", "").upper()
        address = party_info.get("address", {})

        # Check name against blocked terms (simplified)
        for term in self.SAMPLE_BLOCKED_TERMS:
            if term in name:
                matches.append(
                    {
                        "match_type": "NAME_MATCH",
                        "matched_value": name,
                        "matched_term": term,
                        "list_name": "SAMPLE_LIST",
                    }
                )

        # Check country
        country = address.get("country", "").upper()
        if country in self.SANCTIONED_JURISDICTIONS:
            matches.append(
                {
                    "match_type": "JURISDICTION_MATCH",
                    "matched_value": country,
                    "matched_term": self.SANCTIONED_JURISDICTIONS[country],
                    "list_name": "JURISDICTION_LIST",
                }
            )

        return matches

    def _check_jurisdictions(self, data: Any, result: ValidationResult) -> None:
        """Check for sanctioned jurisdictions in transaction."""
        # Extract all country references
        countries = set()

        if isinstance(data, dict):
            self._extract_countries_dict(data, countries)
        else:
            # Handle message objects
            for attr in ["originator", "beneficiary", "sender", "receiver", "originator_fi", "beneficiary_fi"]:
                obj = getattr(data, attr, None)
                if obj:
                    address = getattr(obj, "address", None)
                    if address:
                        country = getattr(address, "country", "")
                        if country:
                            countries.add(country.upper())

        # Check against sanctioned jurisdictions
        for country in countries:
            if country in self.SANCTIONED_JURISDICTIONS:
                result.add_error(
                    "SANCTIONS_JURISDICTION",
                    f"Transaction involves sanctioned jurisdiction: {self.SANCTIONED_JURISDICTIONS[country]}",
                    field_name="jurisdiction",
                    country=country,
                    country_name=self.SANCTIONED_JURISDICTIONS[country],
                )

    def _extract_countries_dict(self, data: Dict, countries: Set[str]) -> None:
        """Recursively extract country codes from dictionary."""
        for key, value in data.items():
            if key.lower() == "country" and isinstance(value, str):
                countries.add(value.upper())
            elif isinstance(value, dict):
                self._extract_countries_dict(value, countries)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        self._extract_countries_dict(item, countries)
