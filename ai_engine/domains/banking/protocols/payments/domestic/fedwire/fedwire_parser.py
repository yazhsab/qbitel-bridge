"""
FedWire Message Parser

Parses FedWire Funds Transfer messages from their wire format.
FedWire messages use a tag-based format with curly braces: {tag}value
"""

import logging
import re
from datetime import datetime, date
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional, Tuple

from ai_engine.domains.banking.protocols.payments.domestic.fedwire.fedwire_codes import (
    TypeCode,
    TypeSubCode,
    BusinessFunctionCode,
    FedWireTag,
    IDCode,
)
from ai_engine.domains.banking.protocols.payments.domestic.fedwire.fedwire_message import (
    FedWireMessage,
    SenderInfo,
    ReceiverInfo,
    BeneficiaryInfo,
    OriginatorInfo,
    IntermediaryInfo,
    FIInfo,
    Address,
    IMAD,
    SenderSuppliedInfo,
    OriginatorToBeneficiaryInfo,
    FIToFIInfo,
    Charges,
    RemittanceInfo,
)

logger = logging.getLogger(__name__)


class FedWireParseError(Exception):
    """Exception raised when FedWire message parsing fails."""

    def __init__(self, message: str, tag: str = "", position: int = 0):
        super().__init__(message)
        self.tag = tag
        self.position = position


class FedWireParser:
    """
    Parser for FedWire Funds Transfer messages.

    FedWire messages use a tag-value format where each field is
    identified by a 4-digit tag enclosed in curly braces.

    Format: {tag}value{tag}value...

    Example:
    {1500}30    P {1510}1000{1520}20231215BKRT1234AB000001{2000}000001000000...
    """

    # Regex pattern to extract tags and values
    TAG_PATTERN = re.compile(r"\{(\d{4})\}([^{]*)")

    def __init__(self, strict: bool = True):
        """
        Initialize the parser.

        Args:
            strict: If True, raise errors on invalid data. If False, log warnings.
        """
        self.strict = strict
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def parse(self, message_content: str) -> FedWireMessage:
        """
        Parse a FedWire message from wire format.

        Args:
            message_content: Raw FedWire message string

        Returns:
            Parsed FedWireMessage object

        Raises:
            FedWireParseError: If parsing fails and strict mode is enabled
        """
        self.errors = []
        self.warnings = []

        # Extract all tags and values
        tags = self._extract_tags(message_content)

        # Create message object
        message = FedWireMessage()
        message.raw_message = message_content

        # Parse mandatory fields
        self._parse_sender_supplied(tags, message)
        self._parse_type_subtype(tags, message)
        self._parse_imad(tags, message)
        self._parse_amount(tags, message)
        self._parse_sender_di(tags, message)
        self._parse_receiver_di(tags, message)
        self._parse_business_function(tags, message)

        # Parse optional fields
        self._parse_sender_reference(tags, message)
        self._parse_previous_message_id(tags, message)
        self._parse_beneficiary_fi(tags, message)
        self._parse_intermediary_fi(tags, message)
        self._parse_beneficiary(tags, message)
        self._parse_originator(tags, message)
        self._parse_originator_fi(tags, message)
        self._parse_instructing_fi(tags, message)
        self._parse_charges(tags, message)
        self._parse_instructed_amount(tags, message)
        self._parse_exchange_rate(tags, message)
        self._parse_account_debited(tags, message)
        self._parse_account_credited(tags, message)
        self._parse_originator_to_beneficiary(tags, message)
        self._parse_fi_to_fi_info(tags, message)
        self._parse_drawdown_advice(tags, message)
        self._parse_unstructured_addenda(tags, message)
        self._parse_remittance(tags, message)
        self._parse_service_message(tags, message)

        # Parse response fields if present
        self._parse_output_fields(tags, message)

        # Check for errors
        if self.errors and self.strict:
            raise FedWireParseError(
                f"Failed to parse FedWire message: {'; '.join(self.errors)}"
            )

        return message

    def _extract_tags(self, content: str) -> Dict[str, str]:
        """Extract all tags and their values from the message."""
        tags = {}
        for match in self.TAG_PATTERN.finditer(content):
            tag = match.group(1)
            value = match.group(2)
            tags[tag] = value
        return tags

    def _parse_sender_supplied(self, tags: Dict[str, str], message: FedWireMessage) -> None:
        """Parse Sender Supplied Information (1500)."""
        value = tags.get("1500", "")
        if not value:
            self.errors.append("Missing mandatory tag 1500 (Sender Supplied Information)")
            return

        ssi = SenderSuppliedInfo()

        if len(value) >= 2:
            ssi.format_version = value[:2]
        if len(value) >= 6:
            ssi.user_request_correlation = value[2:6]
        if len(value) >= 7:
            ssi.test_production = value[6]
        if len(value) >= 8:
            ssi.message_dup_code = value[7]

        message.sender_supplied = ssi

    def _parse_type_subtype(self, tags: Dict[str, str], message: FedWireMessage) -> None:
        """Parse Type and Subtype (1510)."""
        value = tags.get("1510", "")
        if not value:
            self.errors.append("Missing mandatory tag 1510 (Type/Subtype)")
            return

        if len(value) >= 2:
            type_code = TypeCode.from_code(value[:2])
            if type_code:
                message.type_code = type_code
            else:
                self.warnings.append(f"Unknown type code: {value[:2]}")

        if len(value) >= 4:
            subcode = TypeSubCode.from_code(value[2:4])
            if subcode:
                message.type_subcode = subcode
            else:
                self.warnings.append(f"Unknown type subcode: {value[2:4]}")

    def _parse_imad(self, tags: Dict[str, str], message: FedWireMessage) -> None:
        """Parse Input Message Accountability Data (1520)."""
        value = tags.get("1520", "")
        if not value:
            self.errors.append("Missing mandatory tag 1520 (IMAD)")
            return

        try:
            message.imad = IMAD.from_string(value)
        except ValueError as e:
            self.errors.append(f"Invalid IMAD format: {e}")

    def _parse_amount(self, tags: Dict[str, str], message: FedWireMessage) -> None:
        """Parse Amount (2000)."""
        value = tags.get("2000", "")
        if not value:
            self.errors.append("Missing mandatory tag 2000 (Amount)")
            return

        try:
            # Amount is in cents, 12 characters, zero-padded
            # Remove leading zeros and convert to decimal
            cents = int(value)
            message.amount = Decimal(cents) / 100
        except (ValueError, InvalidOperation) as e:
            self.errors.append(f"Invalid amount format: {e}")

    def _parse_sender_di(self, tags: Dict[str, str], message: FedWireMessage) -> None:
        """Parse Sender Depository Institution (3100)."""
        value = tags.get("3100", "")
        if not value:
            self.errors.append("Missing mandatory tag 3100 (Sender DI)")
            return

        sender = SenderInfo()

        # First 9 characters are routing number
        if len(value) >= 9:
            sender.routing_number = value[:9]

        # Rest is name/address, asterisk-delimited
        if len(value) > 9:
            parts = value[9:].split("*")
            if len(parts) >= 1:
                sender.short_name = parts[0].strip()
            if len(parts) >= 2:
                sender.address.line1 = parts[1].strip()
            if len(parts) >= 3:
                sender.address.line2 = parts[2].strip()
            if len(parts) >= 4:
                sender.address.line3 = parts[3].strip()

        message.sender = sender

    def _parse_receiver_di(self, tags: Dict[str, str], message: FedWireMessage) -> None:
        """Parse Receiver Depository Institution (3400)."""
        value = tags.get("3400", "")
        if not value:
            self.errors.append("Missing mandatory tag 3400 (Receiver DI)")
            return

        receiver = ReceiverInfo()

        # First 9 characters are routing number
        if len(value) >= 9:
            receiver.routing_number = value[:9]

        # Rest is name/address, asterisk-delimited
        if len(value) > 9:
            parts = value[9:].split("*")
            if len(parts) >= 1:
                receiver.short_name = parts[0].strip()
            if len(parts) >= 2:
                receiver.address.line1 = parts[1].strip()
            if len(parts) >= 3:
                receiver.address.line2 = parts[2].strip()
            if len(parts) >= 4:
                receiver.address.line3 = parts[3].strip()

        message.receiver = receiver

    def _parse_business_function(self, tags: Dict[str, str], message: FedWireMessage) -> None:
        """Parse Business Function Code (3600)."""
        value = tags.get("3600", "")
        if not value:
            self.errors.append("Missing mandatory tag 3600 (Business Function Code)")
            return

        bfc = BusinessFunctionCode.from_code(value[:3])
        if bfc:
            message.business_function_code = bfc
        else:
            self.warnings.append(f"Unknown business function code: {value[:3]}")

    def _parse_sender_reference(self, tags: Dict[str, str], message: FedWireMessage) -> None:
        """Parse Sender Reference (3320)."""
        value = tags.get("3320", "")
        if value:
            message.sender_reference = value.strip()

    def _parse_previous_message_id(self, tags: Dict[str, str], message: FedWireMessage) -> None:
        """Parse Previous Message Identifier (3500)."""
        value = tags.get("3500", "")
        if value:
            message.previous_message_id = value.strip()

    def _parse_party_info(self, value: str) -> Tuple[Optional[IDCode], str, str, Address]:
        """Parse party information (beneficiary, originator, etc.)."""
        parts = value.split("*")

        id_code = None
        identifier = ""
        name = ""
        address = Address()

        if parts:
            first_part = parts[0]
            # Check if first character is an ID code
            if first_part and first_part[0].isalpha():
                id_code = IDCode.from_code(first_part[0])
                identifier = first_part[1:].strip()
            else:
                identifier = first_part.strip()

        if len(parts) >= 2:
            name = parts[1].strip()
        if len(parts) >= 3:
            address.line1 = parts[2].strip()
        if len(parts) >= 4:
            address.line2 = parts[3].strip()

        return id_code, identifier, name, address

    def _parse_beneficiary_fi(self, tags: Dict[str, str], message: FedWireMessage) -> None:
        """Parse Beneficiary FI (4000)."""
        value = tags.get("4000", "")
        if value:
            id_code, identifier, name, address = self._parse_party_info(value)
            message.beneficiary_fi = FIInfo(
                id_code=id_code,
                identifier=identifier,
                name=name,
                address=address,
            )

    def _parse_intermediary_fi(self, tags: Dict[str, str], message: FedWireMessage) -> None:
        """Parse Intermediary FI (4100)."""
        value = tags.get("4100", "")
        if value:
            id_code, identifier, name, address = self._parse_party_info(value)
            message.intermediary_fi = IntermediaryInfo(
                id_code=id_code,
                identifier=identifier,
                name=name,
                address=address,
            )

    def _parse_beneficiary(self, tags: Dict[str, str], message: FedWireMessage) -> None:
        """Parse Beneficiary (4200)."""
        value = tags.get("4200", "")
        if value:
            id_code, identifier, name, address = self._parse_party_info(value)
            message.beneficiary = BeneficiaryInfo(
                id_code=id_code,
                identifier=identifier,
                name=name,
                address=address,
            )

    def _parse_originator(self, tags: Dict[str, str], message: FedWireMessage) -> None:
        """Parse Originator (5000)."""
        value = tags.get("5000", "")
        if value:
            id_code, identifier, name, address = self._parse_party_info(value)
            message.originator = OriginatorInfo(
                id_code=id_code,
                identifier=identifier,
                name=name,
                address=address,
            )

    def _parse_originator_fi(self, tags: Dict[str, str], message: FedWireMessage) -> None:
        """Parse Originator FI (5100)."""
        value = tags.get("5100", "")
        if value:
            id_code, identifier, name, address = self._parse_party_info(value)
            message.originator_fi = FIInfo(
                id_code=id_code,
                identifier=identifier,
                name=name,
                address=address,
            )

    def _parse_instructing_fi(self, tags: Dict[str, str], message: FedWireMessage) -> None:
        """Parse Instructing FI (5200)."""
        value = tags.get("5200", "")
        if value:
            id_code, identifier, name, address = self._parse_party_info(value)
            message.instructing_fi = FIInfo(
                id_code=id_code,
                identifier=identifier,
                name=name,
                address=address,
            )

    def _parse_charges(self, tags: Dict[str, str], message: FedWireMessage) -> None:
        """Parse Charges (3700)."""
        value = tags.get("3700", "")
        if value:
            charges = Charges()
            if value:
                charges.charge_details = value[0] if value else ""
                # Parse amount if present
                if len(value) > 4:
                    try:
                        charges.send_amount = Decimal(value[4:]) / 100
                    except (ValueError, InvalidOperation):
                        pass
            message.charges = charges

    def _parse_instructed_amount(self, tags: Dict[str, str], message: FedWireMessage) -> None:
        """Parse Instructed Amount (3710)."""
        value = tags.get("3710", "")
        if value:
            # Format: CCCnnnnnnnnnnn (3 char currency + amount)
            if len(value) >= 3:
                message.instructed_currency = value[:3]
            if len(value) > 3:
                try:
                    message.instructed_amount = Decimal(value[3:]) / 100
                except (ValueError, InvalidOperation):
                    pass

    def _parse_exchange_rate(self, tags: Dict[str, str], message: FedWireMessage) -> None:
        """Parse Exchange Rate (3720)."""
        value = tags.get("3720", "")
        if value:
            try:
                message.exchange_rate = Decimal(value)
            except (ValueError, InvalidOperation):
                pass

    def _parse_account_debited(self, tags: Dict[str, str], message: FedWireMessage) -> None:
        """Parse Account Debited in Drawdown (5400)."""
        value = tags.get("5400", "")
        if value:
            message.account_debited = value.strip()

    def _parse_account_credited(self, tags: Dict[str, str], message: FedWireMessage) -> None:
        """Parse Account Credited in Drawdown (5500)."""
        value = tags.get("5500", "")
        if value:
            message.account_credited = value.strip()

    def _parse_originator_to_beneficiary(self, tags: Dict[str, str],
                                          message: FedWireMessage) -> None:
        """Parse Originator to Beneficiary Information (6000)."""
        value = tags.get("6000", "")
        if value:
            parts = value.split("*")
            otb = OriginatorToBeneficiaryInfo()
            if len(parts) >= 1:
                otb.line1 = parts[0]
            if len(parts) >= 2:
                otb.line2 = parts[1]
            if len(parts) >= 3:
                otb.line3 = parts[2]
            if len(parts) >= 4:
                otb.line4 = parts[3]
            message.originator_to_beneficiary = otb

    def _parse_fi_to_fi_info(self, tags: Dict[str, str], message: FedWireMessage) -> None:
        """Parse FI to FI Information (6100)."""
        value = tags.get("6100", "")
        if value:
            parts = value.split("*")
            fif = FIToFIInfo()
            if len(parts) >= 1:
                fif.line1 = parts[0]
            if len(parts) >= 2:
                fif.line2 = parts[1]
            if len(parts) >= 3:
                fif.line3 = parts[2]
            if len(parts) >= 4:
                fif.line4 = parts[3]
            if len(parts) >= 5:
                fif.line5 = parts[4]
            if len(parts) >= 6:
                fif.line6 = parts[5]
            message.fi_to_fi_info = fif

    def _parse_drawdown_advice(self, tags: Dict[str, str], message: FedWireMessage) -> None:
        """Parse Drawdown Debit Account Advice (6200)."""
        value = tags.get("6200", "")
        if value:
            parts = value.split("*")
            if parts:
                message.drawdown_advice_code = parts[0][:3] if parts[0] else ""
                message.drawdown_advice_info = parts[1:] if len(parts) > 1 else []

    def _parse_unstructured_addenda(self, tags: Dict[str, str],
                                     message: FedWireMessage) -> None:
        """Parse Unstructured Addenda (7500)."""
        value = tags.get("7500", "")
        if value:
            message.unstructured_addenda = value

    def _parse_remittance(self, tags: Dict[str, str], message: FedWireMessage) -> None:
        """Parse Remittance Information (8200-8750)."""
        # Check if any remittance tags present
        remit_tags = ["8200", "8300", "8350", "8400", "8450", "8500",
                      "8550", "8600", "8650", "8700", "8750"]
        has_remittance = any(tag in tags for tag in remit_tags)

        if not has_remittance:
            return

        remit = RemittanceInfo()

        # Related remittance (8200)
        if "8200" in tags:
            # Complex format - simplified parsing
            remit.remittance_location_uri = tags["8200"]

        # Remittance originator (8300)
        if "8300" in tags:
            parts = tags["8300"].split("*")
            if parts:
                remit.originator_name = parts[0] if parts else ""

        # Remittance beneficiary (8350)
        if "8350" in tags:
            parts = tags["8350"].split("*")
            if parts:
                remit.beneficiary_name = parts[0] if parts else ""

        # Primary remittance document (8400)
        if "8400" in tags:
            parts = tags["8400"].split("*")
            if parts:
                remit.document_id = parts[0] if parts else ""

        # Actual amount paid (8450)
        if "8450" in tags:
            value = tags["8450"]
            if len(value) >= 3:
                remit.amount_paid_currency = value[:3]
            if len(value) > 3:
                try:
                    remit.amount_paid = Decimal(value[3:]) / 100
                except (ValueError, InvalidOperation):
                    pass

        # Gross amount (8500)
        if "8500" in tags:
            value = tags["8500"]
            if len(value) >= 3:
                remit.gross_amount_currency = value[:3]
            if len(value) > 3:
                try:
                    remit.gross_amount = Decimal(value[3:]) / 100
                except (ValueError, InvalidOperation):
                    pass

        # Document date (8650)
        if "8650" in tags:
            try:
                remit.document_date = datetime.strptime(
                    tags["8650"][:8], "%Y%m%d"
                ).date()
            except ValueError:
                pass

        # Free text (8750)
        if "8750" in tags:
            remit.free_text_lines = tags["8750"].split("*")

        message.remittance = remit

    def _parse_service_message(self, tags: Dict[str, str], message: FedWireMessage) -> None:
        """Parse Service Message (9000)."""
        value = tags.get("9000", "")
        if value:
            message.service_message = value

    def _parse_output_fields(self, tags: Dict[str, str], message: FedWireMessage) -> None:
        """Parse output/response fields."""
        # Message disposition (1100)
        if "1100" in tags:
            message.message_disposition = tags["1100"]

        # Receipt timestamp (1110)
        if "1110" in tags:
            try:
                message.receipt_timestamp = datetime.strptime(
                    tags["1110"][:14], "%Y%m%d%H%M%S"
                )
            except ValueError:
                pass

        # OMAD (1120)
        if "1120" in tags:
            message.omad = tags["1120"]

        # Error (1130)
        if "1130" in tags:
            message.error_info = tags["1130"]

    def get_errors(self) -> List[str]:
        """Get parsing errors."""
        return self.errors

    def get_warnings(self) -> List[str]:
        """Get parsing warnings."""
        return self.warnings
