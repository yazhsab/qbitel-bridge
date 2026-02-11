"""
SEPA Message Data Structures

Core data structures for SEPA payment messages including:
- Credit Transfers (SCT, SCT Inst)
- Direct Debits (SDD Core, SDD B2B)
- Mandates
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
from typing import Dict, List, Optional
import uuid

from ai_engine.domains.banking.protocols.payments.domestic.sepa.sepa_codes import (
    SEPAScheme,
    SEPAServiceLevel,
    SEPALocalInstrument,
    SEPASequenceType,
    SEPACategoryPurpose,
    SEPAPurposeCode,
    SEPA_CURRENCY,
    SEPA_MAX_AMOUNT,
    SCT_INST_MAX_AMOUNT,
    SEPA_COUNTRIES,
)


@dataclass
class SEPAAccount:
    """SEPA Account identification using IBAN."""

    # IBAN (International Bank Account Number)
    iban: str = ""

    # Account currency (always EUR for SEPA)
    currency: str = SEPA_CURRENCY

    # Account type (optional)
    account_type: str = ""

    # Account name (optional)
    account_name: str = ""

    def validate(self) -> List[str]:
        """Validate account information."""
        errors = []

        if not self.iban:
            errors.append("IBAN is required")
        else:
            # Basic IBAN validation
            iban = self.iban.replace(" ", "").upper()
            if len(iban) < 15 or len(iban) > 34:
                errors.append("IBAN must be between 15 and 34 characters")
            elif not iban[:2].isalpha():
                errors.append("IBAN must start with 2-letter country code")
            elif iban[:2] not in SEPA_COUNTRIES:
                errors.append(f"IBAN country {iban[:2]} is not a SEPA country")

        return errors

    @property
    def country_code(self) -> str:
        """Extract country code from IBAN."""
        return self.iban[:2].upper() if self.iban else ""


@dataclass
class SEPAFinancialInstitution:
    """Financial Institution identification."""

    # BIC (Bank Identifier Code / SWIFT code)
    bic: str = ""

    # Institution name
    name: str = ""

    # Country
    country: str = ""

    # Clearing system member ID (alternative to BIC)
    clearing_system_id: str = ""
    member_id: str = ""

    def validate(self) -> List[str]:
        """Validate FI information."""
        errors = []

        if self.bic:
            bic = self.bic.upper().replace(" ", "")
            if len(bic) not in (8, 11):
                errors.append("BIC must be 8 or 11 characters")
            elif not bic[:4].isalpha():
                errors.append("BIC bank code must be 4 letters")
            elif not bic[4:6].isalpha():
                errors.append("BIC country code must be 2 letters")
        elif not self.clearing_system_id:
            errors.append("Either BIC or clearing system ID is required")

        return errors


@dataclass
class SEPAParty:
    """Party information (debtor or creditor)."""

    # Name (required)
    name: str = ""

    # Account
    account: SEPAAccount = field(default_factory=SEPAAccount)

    # Financial Institution
    agent: SEPAFinancialInstitution = field(default_factory=SEPAFinancialInstitution)

    # Address
    street_name: str = ""
    building_number: str = ""
    postal_code: str = ""
    town_name: str = ""
    country: str = ""

    # Identification
    organisation_id: str = ""  # For companies
    private_id: str = ""  # For individuals
    id_type: str = ""  # CUST, EMPL, etc.

    # Contact
    email: str = ""
    phone: str = ""

    def validate(self) -> List[str]:
        """Validate party information."""
        errors = []

        if not self.name:
            errors.append("Party name is required")
        elif len(self.name) > 70:
            errors.append("Party name must not exceed 70 characters")

        errors.extend(self.account.validate())

        if self.agent:
            errors.extend(self.agent.validate())

        return errors

    def get_structured_address(self) -> Dict:
        """Get address in structured format."""
        return {
            "street_name": self.street_name,
            "building_number": self.building_number,
            "postal_code": self.postal_code,
            "town_name": self.town_name,
            "country": self.country,
        }


@dataclass
class SEPAMandate:
    """
    SEPA Direct Debit Mandate.

    Represents the authorization from the debtor to the creditor
    to collect payments.
    """

    # Mandate Identification (unique reference)
    mandate_id: str = ""

    # Mandate signature date
    date_of_signature: Optional[date] = None

    # Amendment indicator
    amendment_indicator: bool = False

    # Original mandate reference (if amended)
    original_mandate_id: str = ""
    original_creditor_scheme_id: str = ""
    original_debtor_account: str = ""
    original_debtor_agent: str = ""

    # Electronic signature (optional)
    electronic_signature: str = ""

    def validate(self) -> List[str]:
        """Validate mandate information."""
        errors = []

        if not self.mandate_id:
            errors.append("Mandate ID is required")
        elif len(self.mandate_id) > 35:
            errors.append("Mandate ID must not exceed 35 characters")

        if not self.date_of_signature:
            errors.append("Date of signature is required")
        elif self.date_of_signature > date.today():
            errors.append("Date of signature cannot be in the future")

        if self.amendment_indicator:
            # At least one amendment field should be populated
            has_amendment = any([
                self.original_mandate_id,
                self.original_creditor_scheme_id,
                self.original_debtor_account,
                self.original_debtor_agent,
            ])
            if not has_amendment:
                errors.append("Amendment details required when amendment indicator is true")

        return errors


@dataclass
class SEPARemittanceInfo:
    """Remittance information for SEPA payments."""

    # Unstructured remittance (max 140 chars)
    unstructured: str = ""

    # Structured remittance
    reference_type: str = ""  # SCOR (ISO), CINV (Invoice)
    reference: str = ""
    reference_issuer: str = ""

    # Creditor Reference (ISO 11649)
    creditor_reference: str = ""

    def validate(self) -> List[str]:
        """Validate remittance information."""
        errors = []

        if self.unstructured and len(self.unstructured) > 140:
            errors.append("Unstructured remittance must not exceed 140 characters")

        if self.reference and len(self.reference) > 35:
            errors.append("Reference must not exceed 35 characters")

        return errors


@dataclass
class SEPACreditTransfer:
    """
    SEPA Credit Transfer message.

    Represents a payment instruction from debtor to creditor.
    Supports both SCT (standard) and SCT Inst (instant) schemes.
    """

    # Message identification
    message_id: str = ""
    creation_datetime: datetime = field(default_factory=datetime.now)

    # Payment identification
    payment_info_id: str = ""
    instruction_id: str = ""
    end_to_end_id: str = ""
    uetr: str = ""  # Universal End-to-End Transaction Reference

    # Scheme
    scheme: SEPAScheme = SEPAScheme.SCT
    service_level: SEPAServiceLevel = SEPAServiceLevel.SEPA
    local_instrument: Optional[SEPALocalInstrument] = None

    # Amount
    amount: Decimal = Decimal("0")
    currency: str = SEPA_CURRENCY

    # Execution date
    requested_execution_date: Optional[date] = None

    # Parties
    debtor: SEPAParty = field(default_factory=SEPAParty)
    creditor: SEPAParty = field(default_factory=SEPAParty)

    # Ultimate parties (optional - different from actual debtor/creditor)
    ultimate_debtor_name: str = ""
    ultimate_creditor_name: str = ""

    # Remittance
    remittance_info: SEPARemittanceInfo = field(default_factory=SEPARemittanceInfo)

    # Purpose
    category_purpose: Optional[SEPACategoryPurpose] = None
    purpose_code: Optional[SEPAPurposeCode] = None

    # Charges
    charge_bearer: str = "SLEV"  # Following Service Level

    # Batch information
    batch_booking: bool = True
    number_of_transactions: int = 1
    control_sum: Optional[Decimal] = None

    # Status
    status: str = "PDNG"  # Pending

    def __post_init__(self):
        """Initialize IDs if not provided."""
        if not self.message_id:
            self.message_id = self._generate_id()
        if not self.payment_info_id:
            self.payment_info_id = self._generate_id()
        if not self.end_to_end_id:
            self.end_to_end_id = self._generate_id()
        if not self.uetr:
            self.uetr = str(uuid.uuid4())

        # Set local instrument for instant payments
        if self.scheme == SEPAScheme.SCT_INST and not self.local_instrument:
            self.local_instrument = SEPALocalInstrument.INST

    def _generate_id(self) -> str:
        """Generate a unique ID."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique = uuid.uuid4().hex[:8].upper()
        return f"SCT{timestamp}{unique}"[:35]

    def validate(self) -> List[str]:
        """Validate the credit transfer."""
        errors = []

        # Message ID
        if not self.message_id:
            errors.append("Message ID is required")
        elif len(self.message_id) > 35:
            errors.append("Message ID must not exceed 35 characters")

        # End-to-end ID
        if not self.end_to_end_id:
            errors.append("End-to-end ID is required")
        elif len(self.end_to_end_id) > 35:
            errors.append("End-to-end ID must not exceed 35 characters")

        # Amount
        if self.amount <= 0:
            errors.append("Amount must be greater than zero")
        elif self.amount > Decimal(str(SEPA_MAX_AMOUNT)):
            errors.append(f"Amount exceeds SEPA maximum ({SEPA_MAX_AMOUNT})")

        # Instant payment specific
        if self.scheme == SEPAScheme.SCT_INST:
            if self.amount > Decimal(str(SCT_INST_MAX_AMOUNT)):
                errors.append(f"Amount exceeds SCT Inst maximum ({SCT_INST_MAX_AMOUNT})")

        # Currency
        if self.currency != SEPA_CURRENCY:
            errors.append(f"SEPA only supports {SEPA_CURRENCY} currency")

        # Party validation
        errors.extend([f"Debtor: {e}" for e in self.debtor.validate()])
        errors.extend([f"Creditor: {e}" for e in self.creditor.validate()])

        # Remittance validation
        errors.extend(self.remittance_info.validate())

        return errors

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "message_id": self.message_id,
            "scheme": self.scheme.code,
            "amount": str(self.amount),
            "currency": self.currency,
            "debtor": {
                "name": self.debtor.name,
                "iban": self.debtor.account.iban,
                "bic": self.debtor.agent.bic,
            },
            "creditor": {
                "name": self.creditor.name,
                "iban": self.creditor.account.iban,
                "bic": self.creditor.agent.bic,
            },
            "end_to_end_id": self.end_to_end_id,
            "requested_execution_date": (
                self.requested_execution_date.isoformat()
                if self.requested_execution_date else None
            ),
            "status": self.status,
        }


@dataclass
class SEPADirectDebit:
    """
    SEPA Direct Debit message.

    Represents a collection instruction from creditor to debtor.
    Supports SDD Core and SDD B2B schemes.
    """

    # Message identification
    message_id: str = ""
    creation_datetime: datetime = field(default_factory=datetime.now)

    # Payment identification
    payment_info_id: str = ""
    instruction_id: str = ""
    end_to_end_id: str = ""

    # Scheme
    scheme: SEPAScheme = SEPAScheme.SDD_CORE
    service_level: SEPAServiceLevel = SEPAServiceLevel.SEPA
    local_instrument: SEPALocalInstrument = SEPALocalInstrument.CORE

    # Sequence type
    sequence_type: SEPASequenceType = SEPASequenceType.RCUR

    # Amount
    amount: Decimal = Decimal("0")
    currency: str = SEPA_CURRENCY

    # Collection date
    requested_collection_date: Optional[date] = None

    # Creditor (collector)
    creditor: SEPAParty = field(default_factory=SEPAParty)
    creditor_scheme_id: str = ""  # Creditor Identifier

    # Debtor (payer)
    debtor: SEPAParty = field(default_factory=SEPAParty)

    # Ultimate parties
    ultimate_debtor_name: str = ""
    ultimate_creditor_name: str = ""

    # Mandate
    mandate: SEPAMandate = field(default_factory=SEPAMandate)

    # Remittance
    remittance_info: SEPARemittanceInfo = field(default_factory=SEPARemittanceInfo)

    # Purpose
    category_purpose: Optional[SEPACategoryPurpose] = None
    purpose_code: Optional[SEPAPurposeCode] = None

    # Batch information
    batch_booking: bool = True
    number_of_transactions: int = 1
    control_sum: Optional[Decimal] = None

    # Status
    status: str = "PDNG"

    def __post_init__(self):
        """Initialize IDs if not provided."""
        if not self.message_id:
            self.message_id = self._generate_id()
        if not self.payment_info_id:
            self.payment_info_id = self._generate_id()
        if not self.end_to_end_id:
            self.end_to_end_id = self._generate_id()

        # Set local instrument based on scheme
        if self.scheme == SEPAScheme.SDD_B2B:
            self.local_instrument = SEPALocalInstrument.B2B

    def _generate_id(self) -> str:
        """Generate a unique ID."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique = uuid.uuid4().hex[:8].upper()
        return f"SDD{timestamp}{unique}"[:35]

    def validate(self) -> List[str]:
        """Validate the direct debit."""
        errors = []

        # Message ID
        if not self.message_id:
            errors.append("Message ID is required")
        elif len(self.message_id) > 35:
            errors.append("Message ID must not exceed 35 characters")

        # End-to-end ID
        if not self.end_to_end_id:
            errors.append("End-to-end ID is required")
        elif len(self.end_to_end_id) > 35:
            errors.append("End-to-end ID must not exceed 35 characters")

        # Creditor Scheme ID (Creditor Identifier)
        if not self.creditor_scheme_id:
            errors.append("Creditor Scheme ID (Creditor Identifier) is required")
        elif len(self.creditor_scheme_id) > 35:
            errors.append("Creditor Scheme ID must not exceed 35 characters")

        # Amount
        if self.amount <= 0:
            errors.append("Amount must be greater than zero")
        elif self.amount > Decimal(str(SEPA_MAX_AMOUNT)):
            errors.append(f"Amount exceeds SEPA maximum ({SEPA_MAX_AMOUNT})")

        # Currency
        if self.currency != SEPA_CURRENCY:
            errors.append(f"SEPA only supports {SEPA_CURRENCY} currency")

        # Party validation
        errors.extend([f"Creditor: {e}" for e in self.creditor.validate()])
        errors.extend([f"Debtor: {e}" for e in self.debtor.validate()])

        # Mandate validation
        errors.extend([f"Mandate: {e}" for e in self.mandate.validate()])

        # Remittance validation
        errors.extend(self.remittance_info.validate())

        # Collection date validation
        if self.requested_collection_date:
            if self.requested_collection_date < date.today():
                errors.append("Requested collection date cannot be in the past")

        return errors

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "message_id": self.message_id,
            "scheme": self.scheme.code,
            "sequence_type": self.sequence_type.code,
            "amount": str(self.amount),
            "currency": self.currency,
            "creditor": {
                "name": self.creditor.name,
                "iban": self.creditor.account.iban,
                "creditor_id": self.creditor_scheme_id,
            },
            "debtor": {
                "name": self.debtor.name,
                "iban": self.debtor.account.iban,
            },
            "mandate": {
                "id": self.mandate.mandate_id,
                "date_of_signature": (
                    self.mandate.date_of_signature.isoformat()
                    if self.mandate.date_of_signature else None
                ),
            },
            "end_to_end_id": self.end_to_end_id,
            "requested_collection_date": (
                self.requested_collection_date.isoformat()
                if self.requested_collection_date else None
            ),
            "status": self.status,
        }
