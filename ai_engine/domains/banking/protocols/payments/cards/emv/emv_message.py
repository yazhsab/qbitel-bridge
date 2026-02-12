"""
EMV Message Data Structures

Data structures for EMV transactions including:
- Card data
- Terminal data
- Transaction data
- Cryptogram data
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ai_engine.domains.banking.protocols.payments.cards.emv.emv_codes import (
    EmvTag,
    EmvApplication,
    EmvTransactionType,
    EmvCryptogramType,
    EmvCvmResult,
    EmvTerminalType,
    EmvPosEntryMode,
    get_tag_name,
)
from ai_engine.domains.banking.protocols.payments.cards.emv.emv_tlv import (
    TlvData,
    TlvBuilder,
    parse_tlv,
)


@dataclass
class EmvCard:
    """EMV Card Data."""

    # Primary data
    pan: str = ""  # Tag 5A
    pan_sequence_number: str = "00"  # Tag 5F34
    expiration_date: str = ""  # Tag 5F24 (YYMMDD)
    effective_date: Optional[str] = None  # Tag 5F25
    cardholder_name: str = ""  # Tag 5F20
    service_code: str = ""  # Tag 5F30

    # Application data
    aid: str = ""  # Tag 9F06/84
    application_label: str = ""  # Tag 50
    application_preferred_name: str = ""  # Tag 9F12
    application_version: str = ""  # Tag 9F08

    # Track 2 equivalent
    track2_equivalent: str = ""  # Tag 57

    # Risk management
    aip: str = ""  # Tag 82 - Application Interchange Profile
    afl: str = ""  # Tag 94 - Application File Locator
    cvm_list: str = ""  # Tag 8E

    # Cryptogram data
    atc: str = ""  # Tag 9F36 - Application Transaction Counter
    issuer_application_data: str = ""  # Tag 9F10

    @property
    def masked_pan(self) -> str:
        """Get masked PAN (first 6, last 4)."""
        if len(self.pan) >= 10:
            return f"{self.pan[:6]}{'*' * (len(self.pan) - 10)}{self.pan[-4:]}"
        return "*" * len(self.pan)

    @property
    def application(self) -> Optional[EmvApplication]:
        """Get application from AID."""
        return EmvApplication.from_aid(self.aid) if self.aid else None

    @property
    def expiry_date(self) -> Optional[date]:
        """Get expiration as date object."""
        if len(self.expiration_date) >= 4:
            try:
                year = int(self.expiration_date[:2])
                month = int(self.expiration_date[2:4])
                year = 2000 + year if year < 50 else 1900 + year
                return date(year, month, 1)
            except ValueError:
                pass
        return None

    @classmethod
    def from_tlv(cls, tlv_data: List[TlvData]) -> "EmvCard":
        """Create from TLV data."""
        card = cls()

        tag_map = {tlv.tag_hex: tlv.value_hex for tlv in tlv_data}

        # Recursively flatten nested TLV
        def flatten(elements: List[TlvData], result: Dict[str, str]):
            for elem in elements:
                result[elem.tag_hex] = elem.value_hex
                if elem.children:
                    flatten(elem.children, result)

        flatten(tlv_data, tag_map)

        card.pan = tag_map.get("5A", "")
        card.pan_sequence_number = tag_map.get("5F34", "00")
        card.expiration_date = tag_map.get("5F24", "")
        card.effective_date = tag_map.get("5F25")
        card.service_code = tag_map.get("5F30", "")
        card.aid = tag_map.get("9F06", "") or tag_map.get("84", "")
        card.application_label = (
            bytes.fromhex(tag_map.get("50", "")).decode("ascii", errors="ignore") if tag_map.get("50") else ""
        )
        card.application_preferred_name = (
            bytes.fromhex(tag_map.get("9F12", "")).decode("ascii", errors="ignore") if tag_map.get("9F12") else ""
        )
        card.application_version = tag_map.get("9F08", "")
        card.track2_equivalent = tag_map.get("57", "")
        card.aip = tag_map.get("82", "")
        card.afl = tag_map.get("94", "")
        card.cvm_list = tag_map.get("8E", "")
        card.atc = tag_map.get("9F36", "")
        card.issuer_application_data = tag_map.get("9F10", "")
        card.cardholder_name = (
            bytes.fromhex(tag_map.get("5F20", "")).decode("ascii", errors="ignore") if tag_map.get("5F20") else ""
        )

        return card

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pan": self.masked_pan,
            "pan_sequence_number": self.pan_sequence_number,
            "expiration_date": self.expiration_date,
            "cardholder_name": self.cardholder_name,
            "aid": self.aid,
            "application": self.application.description if self.application else None,
            "application_label": self.application_label,
        }


@dataclass
class EmvTerminal:
    """EMV Terminal Data."""

    # Identification
    terminal_id: str = ""  # Tag 9F1C
    merchant_id: str = ""  # Tag 9F16
    acquirer_id: str = ""  # Tag 9F01
    merchant_name_location: str = ""  # Tag 9F4E
    ifd_serial_number: str = ""  # Tag 9F1E

    # Capabilities
    terminal_type: str = "22"  # Tag 9F35
    terminal_capabilities: str = "E0F8C8"  # Tag 9F33
    additional_terminal_capabilities: str = "FF00F0A001"  # Tag 9F40

    # Location
    terminal_country_code: str = "840"  # Tag 9F1A (US)
    merchant_category_code: str = "5999"  # Tag 9F15

    # Limits
    floor_limit: int = 0  # Tag 9F1B

    # Transaction qualifiers (contactless)
    ttq: str = ""  # Tag 9F66 - Terminal Transaction Qualifiers

    # Verification results
    tvr: str = "0000000000"  # Tag 95 - Terminal Verification Results
    tsi: str = "0000"  # Tag 9B - Transaction Status Information

    # CVM results
    cvm_results: str = "1F0002"  # Tag 9F34

    # Application version
    application_version_number: str = "0001"  # Tag 9F09

    @property
    def terminal_type_enum(self) -> Optional[EmvTerminalType]:
        """Get terminal type enum."""
        for tt in EmvTerminalType:
            if tt.code == self.terminal_type:
                return tt
        return None

    def to_tlv(self) -> bytes:
        """Build terminal TLV data."""
        builder = TlvBuilder()

        if self.terminal_id:
            builder.add_alphanumeric("9F1C", self.terminal_id, 8)
        if self.merchant_id:
            builder.add_alphanumeric("9F16", self.merchant_id, 15)
        builder.add("9F35", self.terminal_type)
        builder.add("9F33", self.terminal_capabilities)
        builder.add("9F40", self.additional_terminal_capabilities)
        builder.add_numeric("9F1A", int(self.terminal_country_code), 2)
        builder.add("95", self.tvr)
        builder.add("9B", self.tsi)
        builder.add("9F34", self.cvm_results)
        builder.add("9F09", self.application_version_number)

        if self.ttq:
            builder.add("9F66", self.ttq)

        return builder.build()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "terminal_id": self.terminal_id,
            "merchant_id": self.merchant_id,
            "terminal_type": self.terminal_type,
            "terminal_country_code": self.terminal_country_code,
            "tvr": self.tvr,
            "tsi": self.tsi,
            "cvm_results": self.cvm_results,
        }


@dataclass
class EmvCryptogram:
    """EMV Cryptogram Data."""

    # Cryptogram
    application_cryptogram: str = ""  # Tag 9F26
    cryptogram_information_data: str = ""  # Tag 9F27

    # Transaction data used for cryptogram
    atc: str = ""  # Tag 9F36
    unpredictable_number: str = ""  # Tag 9F37
    issuer_application_data: str = ""  # Tag 9F10

    # Authorization
    authorization_code: str = ""  # Tag 89
    authorization_response_code: str = ""  # Tag 8A
    issuer_authentication_data: str = ""  # Tag 91

    @property
    def cryptogram_type(self) -> Optional[EmvCryptogramType]:
        """Get cryptogram type from CID."""
        if self.cryptogram_information_data:
            return EmvCryptogramType.from_cid(self.cryptogram_information_data)
        return None

    @property
    def is_arqc(self) -> bool:
        """Check if ARQC (online request)."""
        return self.cryptogram_type == EmvCryptogramType.ARQC

    @property
    def is_tc(self) -> bool:
        """Check if TC (offline approval)."""
        return self.cryptogram_type == EmvCryptogramType.TC

    @property
    def is_aac(self) -> bool:
        """Check if AAC (decline)."""
        return self.cryptogram_type == EmvCryptogramType.AAC

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "application_cryptogram": self.application_cryptogram,
            "cryptogram_type": self.cryptogram_type.description if self.cryptogram_type else None,
            "atc": self.atc,
            "unpredictable_number": self.unpredictable_number,
            "authorization_code": self.authorization_code,
            "authorization_response_code": self.authorization_response_code,
        }


@dataclass
class EmvResponse:
    """EMV Authorization Response."""

    # Response codes
    authorization_code: str = ""  # Tag 89
    authorization_response_code: str = ""  # Tag 8A

    # Issuer authentication
    issuer_authentication_data: str = ""  # Tag 91

    # Scripts
    issuer_script_template_1: str = ""  # Tag 71
    issuer_script_template_2: str = ""  # Tag 72
    issuer_script_results: str = ""  # Tag 9F5B

    @property
    def is_approved(self) -> bool:
        """Check if response is approved."""
        return self.authorization_response_code in ("00", "Y1", "Y3")

    @property
    def is_declined(self) -> bool:
        """Check if response is declined."""
        return self.authorization_response_code in ("05", "51", "57", "61", "Z1", "Z3")

    @property
    def requires_referral(self) -> bool:
        """Check if response requires referral."""
        return self.authorization_response_code in ("01", "02")

    @classmethod
    def from_tlv(cls, tlv_data: List[TlvData]) -> "EmvResponse":
        """Create from TLV data."""
        response = cls()

        tag_map = {tlv.tag_hex: tlv.value_hex for tlv in tlv_data}

        response.authorization_code = tag_map.get("89", "")
        response.authorization_response_code = tag_map.get("8A", "")
        response.issuer_authentication_data = tag_map.get("91", "")
        response.issuer_script_template_1 = tag_map.get("71", "")
        response.issuer_script_template_2 = tag_map.get("72", "")
        response.issuer_script_results = tag_map.get("9F5B", "")

        return response

    def to_tlv(self) -> bytes:
        """Build response TLV data."""
        builder = TlvBuilder()

        if self.authorization_code:
            builder.add("89", self.authorization_code)
        if self.authorization_response_code:
            builder.add("8A", self.authorization_response_code)
        if self.issuer_authentication_data:
            builder.add("91", self.issuer_authentication_data)
        if self.issuer_script_template_1:
            builder.add("71", self.issuer_script_template_1)
        if self.issuer_script_template_2:
            builder.add("72", self.issuer_script_template_2)

        return builder.build()


@dataclass
class EmvTransaction:
    """Complete EMV Transaction."""

    # Transaction identification
    transaction_id: str = ""
    retrieval_reference_number: str = ""

    # Card data
    card: EmvCard = field(default_factory=EmvCard)

    # Terminal data
    terminal: EmvTerminal = field(default_factory=EmvTerminal)

    # Cryptogram
    cryptogram: EmvCryptogram = field(default_factory=EmvCryptogram)

    # Response
    response: Optional[EmvResponse] = None

    # Transaction data
    transaction_type: str = "00"  # Tag 9C
    amount_authorized: int = 0  # Tag 9F02 (in minor units)
    amount_other: int = 0  # Tag 9F03
    currency_code: str = "840"  # Tag 5F2A
    transaction_date: str = ""  # Tag 9A (YYMMDD)
    transaction_time: str = ""  # Tag 9F21 (HHMMSS)

    # POS entry
    pos_entry_mode: str = "05"  # Tag 9F39

    # Status
    is_contactless: bool = False
    is_online: bool = True
    is_completed: bool = False

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def transaction_type_enum(self) -> Optional[EmvTransactionType]:
        """Get transaction type enum."""
        return EmvTransactionType.from_code(self.transaction_type)

    @property
    def amount_decimal(self) -> Decimal:
        """Get amount as decimal."""
        # Assuming 2 decimal places for most currencies
        return Decimal(self.amount_authorized) / 100

    def build_cdol1_data(self) -> bytes:
        """Build CDOL1 data for GENERATE AC command."""
        builder = TlvBuilder()

        # Amount, Authorized
        builder.add_numeric("9F02", self.amount_authorized, 6)
        # Amount, Other
        builder.add_numeric("9F03", self.amount_other, 6)
        # Terminal Country Code
        builder.add_numeric("9F1A", int(self.terminal.terminal_country_code), 2)
        # Terminal Verification Results
        builder.add("95", self.terminal.tvr)
        # Transaction Currency Code
        builder.add_numeric("5F2A", int(self.currency_code), 2)
        # Transaction Date
        if self.transaction_date:
            builder.add("9A", self.transaction_date)
        else:
            builder.add_numeric("9A", int(datetime.now().strftime("%y%m%d")), 3)
        # Transaction Type
        builder.add("9C", self.transaction_type)
        # Unpredictable Number
        if self.cryptogram.unpredictable_number:
            builder.add("9F37", self.cryptogram.unpredictable_number)
        # Terminal Type
        builder.add("9F35", self.terminal.terminal_type)

        return builder.build()

    def build_authorization_request(self) -> Dict[str, str]:
        """Build authorization request data."""
        return {
            "9F26": self.cryptogram.application_cryptogram,
            "9F27": self.cryptogram.cryptogram_information_data,
            "9F10": self.cryptogram.issuer_application_data,
            "9F36": self.cryptogram.atc,
            "9F37": self.cryptogram.unpredictable_number,
            "95": self.terminal.tvr,
            "9A": self.transaction_date,
            "9C": self.transaction_type,
            "9F02": f"{self.amount_authorized:012d}",
            "5F2A": self.currency_code,
            "9F1A": self.terminal.terminal_country_code,
            "5A": self.card.pan,
            "5F34": self.card.pan_sequence_number,
            "82": self.card.aip,
            "9F34": self.terminal.cvm_results,
            "9F35": self.terminal.terminal_type,
            "9F33": self.terminal.terminal_capabilities,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "transaction_id": self.transaction_id,
            "transaction_type": (
                self.transaction_type_enum.description if self.transaction_type_enum else self.transaction_type
            ),
            "amount": str(self.amount_decimal),
            "currency_code": self.currency_code,
            "transaction_date": self.transaction_date,
            "transaction_time": self.transaction_time,
            "pos_entry_mode": self.pos_entry_mode,
            "is_contactless": self.is_contactless,
            "is_online": self.is_online,
            "card": self.card.to_dict(),
            "terminal": self.terminal.to_dict(),
            "cryptogram": self.cryptogram.to_dict(),
            "response": self.response.to_dict() if self.response else None,
            "created_at": self.created_at.isoformat(),
        }

    def __str__(self) -> str:
        return (
            f"EmvTransaction({self.card.masked_pan}, "
            f"{self.amount_decimal} {self.currency_code}, "
            f"{self.transaction_type_enum.description if self.transaction_type_enum else self.transaction_type})"
        )
