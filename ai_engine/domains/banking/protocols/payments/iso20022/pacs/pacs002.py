"""
ISO 20022 pacs.002 - FI to FI Payment Status Report

This message is used by the instructed agent to inform the previous party
about the positive or negative status of a payment instruction.

Use Cases:
- Acknowledgment of payment receipt
- Notification of payment rejection
- Status updates during processing
- Settlement confirmation
"""

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional
import uuid

logger = logging.getLogger(__name__)


class TransactionStatus(Enum):
    """ISO 20022 Transaction Status Codes."""

    # Positive Statuses
    ACCP = ("ACCP", "Accepted Customer Profile", "Preceding check successful")
    ACSC = ("ACSC", "Accepted Settlement Completed", "Settlement completed")
    ACSP = ("ACSP", "Accepted Settlement In Process", "All preceding checks passed")
    ACTC = ("ACTC", "Accepted Technical Validation", "Technical validation successful")
    ACWC = ("ACWC", "Accepted With Change", "Instruction accepted with change")
    ACWP = ("ACWP", "Accepted Without Posting", "Credited without posting")

    # Pending Statuses
    PDNG = ("PDNG", "Pending", "Payment pending")
    RCVD = ("RCVD", "Received", "Instruction received")

    # Negative Statuses
    RJCT = ("RJCT", "Rejected", "Payment rejected")
    CANC = ("CANC", "Cancelled", "Payment cancelled")

    # Partial Status
    PART = ("PART", "Partially Accepted", "Some transactions accepted")

    def __init__(self, code: str, name: str, description: str):
        self.code = code
        self.status_name = name
        self.description = description

    @classmethod
    def from_code(cls, code: str) -> Optional["TransactionStatus"]:
        """Get status from code."""
        for status in cls:
            if status.code == code:
                return status
        return None

    @property
    def is_final(self) -> bool:
        """Check if this is a final status."""
        return self in [
            TransactionStatus.ACSC,
            TransactionStatus.RJCT,
            TransactionStatus.CANC,
        ]

    @property
    def is_positive(self) -> bool:
        """Check if this is a positive status."""
        return self.code.startswith("AC")


class StatusReasonCode(Enum):
    """ISO 20022 Status Reason Codes."""

    # Account Related
    AC01 = ("AC01", "Incorrect Account Number")
    AC02 = ("AC02", "Invalid Debtor Account Number")
    AC03 = ("AC03", "Invalid Creditor Account Number")
    AC04 = ("AC04", "Closed Account Number")
    AC05 = ("AC05", "Closed Debtor Account Number")
    AC06 = ("AC06", "Blocked Account")
    AC07 = ("AC07", "Closed Creditor Account Number")
    AC13 = ("AC13", "Invalid Debtor Account Type")
    AC14 = ("AC14", "Invalid Creditor Account Type")

    # Agent/Bank Issues
    AG01 = ("AG01", "Transaction Forbidden")
    AG02 = ("AG02", "Invalid Bank Operation Code")
    AG03 = ("AG03", "Transaction Not Supported")
    AG09 = ("AG09", "Payment Not Received")
    AG10 = ("AG10", "Agent Suspended")

    # Amount Issues
    AM01 = ("AM01", "Zero Amount")
    AM02 = ("AM02", "Not Allowed Amount")
    AM03 = ("AM03", "Not Allowed Currency")
    AM04 = ("AM04", "Insufficient Funds")
    AM05 = ("AM05", "Duplicate")
    AM06 = ("AM06", "Too Low Amount")
    AM07 = ("AM07", "Blocked Amount")
    AM09 = ("AM09", "Wrong Amount")
    AM10 = ("AM10", "Invalid Control Sum")

    # End Customer
    BE01 = ("BE01", "Inconsistent With End Customer")
    BE04 = ("BE04", "Missing Creditor Address")
    BE05 = ("BE05", "Unrecognised Initiating Party")
    BE06 = ("BE06", "Unknown End Customer")
    BE07 = ("BE07", "Missing Debtor Address")

    # Date/Time
    DT01 = ("DT01", "Invalid Date")
    DT02 = ("DT02", "Invalid Creation Date Time")
    DT04 = ("DT04", "Future Date Not Supported")

    # Format
    FF01 = ("FF01", "Invalid File Format")
    FF03 = ("FF03", "Invalid Payment Type Information")
    FF05 = ("FF05", "Invalid Local Instrument Code")

    # Regulatory
    RR01 = ("RR01", "Missing Debtor Account Or Identification")
    RR02 = ("RR02", "Missing Debtor Name Or Address")
    RR03 = ("RR03", "Missing Creditor Name Or Address")
    RR04 = ("RR04", "Regulatory Reason")

    # Technical
    TM01 = ("TM01", "Invalid Cut Off Time")
    TS01 = ("TS01", "Technical Problem")

    # Narrative
    NARR = ("NARR", "Narrative (See Additional Information)")

    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description

    @classmethod
    def from_code(cls, code: str) -> Optional["StatusReasonCode"]:
        """Get reason code from string."""
        for rc in cls:
            if rc.code == code:
                return rc
        return None


@dataclass
class StatusReasonInfo:
    """Status reason information."""

    # Reason code
    reason_code: Optional[StatusReasonCode] = None
    proprietary_code: str = ""

    # Additional information
    additional_info: List[str] = field(default_factory=list)

    # Originator (who reported the reason)
    originator: str = ""

    def to_dict(self) -> Dict:
        return {
            "reason_code": self.reason_code.code if self.reason_code else self.proprietary_code,
            "description": self.reason_code.description if self.reason_code else "",
            "additional_info": self.additional_info,
        }


@dataclass
class OriginalGroupInfo:
    """Original group information reference."""

    # Original message identification
    original_message_id: str = ""
    original_message_name_id: str = ""  # e.g., "pacs.008.001.08"
    original_creation_datetime: Optional[datetime] = None

    # Group status
    group_status: Optional[TransactionStatus] = None

    # Status reason (if rejected at group level)
    status_reason_info: List[StatusReasonInfo] = field(default_factory=list)

    # Counts
    number_of_transactions_per_status: Dict[str, int] = field(default_factory=dict)


@dataclass
class OriginalTransactionInfo:
    """Original transaction reference and status."""

    # Original payment identification
    original_instruction_id: str = ""
    original_end_to_end_id: str = ""
    original_transaction_id: str = ""
    original_uetr: str = ""

    # Transaction status
    transaction_status: TransactionStatus = TransactionStatus.ACSC

    # Status reason (if rejected)
    status_reason_info: List[StatusReasonInfo] = field(default_factory=list)

    # Acceptance datetime (if accepted)
    acceptance_datetime: Optional[datetime] = None

    # Settlement information
    clearing_system_ref: str = ""

    # Original amounts (for reference)
    original_instructed_amount: Optional[Decimal] = None
    original_instructed_currency: str = ""

    # Charges
    charges_amount: Optional[Decimal] = None
    charges_currency: str = ""

    def is_accepted(self) -> bool:
        """Check if transaction was accepted."""
        return self.transaction_status.is_positive

    def is_rejected(self) -> bool:
        """Check if transaction was rejected."""
        return self.transaction_status == TransactionStatus.RJCT

    def to_dict(self) -> Dict:
        return {
            "original_end_to_end_id": self.original_end_to_end_id,
            "original_transaction_id": self.original_transaction_id,
            "status": self.transaction_status.code,
            "status_name": self.transaction_status.status_name,
            "is_accepted": self.is_accepted(),
            "is_rejected": self.is_rejected(),
            "reasons": [r.to_dict() for r in self.status_reason_info],
        }


@dataclass
class Pacs002Message:
    """
    ISO 20022 pacs.002 - FI to FI Payment Status Report.

    Used to report status of payment instructions.
    """

    # Message identification
    message_id: str = ""
    creation_datetime: datetime = field(default_factory=datetime.now)

    # Instructing/Instructed agents
    instructing_agent_bic: str = ""
    instructed_agent_bic: str = ""

    # Original group information
    original_group_info: OriginalGroupInfo = field(default_factory=OriginalGroupInfo)

    # Transaction status information
    transaction_info_and_status: List[OriginalTransactionInfo] = field(default_factory=list)

    # Raw XML for reference
    raw_xml: str = ""

    def __post_init__(self):
        """Generate message ID if not provided."""
        if not self.message_id:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            unique = uuid.uuid4().hex[:8].upper()
            self.message_id = f"PACS002{timestamp}{unique}"[:35]

    def validate(self) -> List[str]:
        """Validate the message."""
        errors = []

        if not self.message_id:
            errors.append("Message ID is required")
        elif len(self.message_id) > 35:
            errors.append("Message ID must not exceed 35 characters")

        if not self.original_group_info.original_message_id:
            errors.append("Original message ID is required")

        return errors

    @property
    def overall_status(self) -> Optional[TransactionStatus]:
        """Get the overall status of the report."""
        if self.original_group_info.group_status:
            return self.original_group_info.group_status

        if not self.transaction_info_and_status:
            return None

        # Determine overall status from transactions
        statuses = set(tx.transaction_status for tx in self.transaction_info_and_status)

        if len(statuses) == 1:
            return list(statuses)[0]

        if TransactionStatus.RJCT in statuses:
            return TransactionStatus.PART

        if all(s.is_positive for s in statuses):
            return TransactionStatus.ACSC

        return TransactionStatus.PDNG

    @property
    def all_accepted(self) -> bool:
        """Check if all transactions were accepted."""
        if not self.transaction_info_and_status:
            return self.original_group_info.group_status and self.original_group_info.group_status.is_positive

        return all(tx.is_accepted() for tx in self.transaction_info_and_status)

    @property
    def any_rejected(self) -> bool:
        """Check if any transactions were rejected."""
        if self.original_group_info.group_status == TransactionStatus.RJCT:
            return True

        return any(tx.is_rejected() for tx in self.transaction_info_and_status)

    def get_rejected_transactions(self) -> List[OriginalTransactionInfo]:
        """Get list of rejected transactions."""
        return [tx for tx in self.transaction_info_and_status if tx.is_rejected()]

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "message_id": self.message_id,
            "creation_datetime": self.creation_datetime.isoformat(),
            "original_message_id": self.original_group_info.original_message_id,
            "overall_status": self.overall_status.code if self.overall_status else None,
            "all_accepted": self.all_accepted,
            "any_rejected": self.any_rejected,
            "transactions": [tx.to_dict() for tx in self.transaction_info_and_status],
        }


class Pacs002Parser:
    """Parser for pacs.002 Payment Status Report messages."""

    NAMESPACE = "urn:iso:std:iso:20022:tech:xsd:pacs.002.001.10"

    def __init__(self, strict: bool = True):
        self.strict = strict
        self.errors: List[str] = []

    def parse(self, xml_content: str) -> Pacs002Message:
        """
        Parse a pacs.002 XML message.

        Args:
            xml_content: XML string

        Returns:
            Parsed Pacs002Message
        """
        self.errors = []
        message = Pacs002Message()
        message.raw_xml = xml_content

        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            self.errors.append(f"XML parse error: {e}")
            return message

        # Find namespace
        ns = self._extract_namespace(root)
        ns_prefix = f"{{{ns}}}" if ns else ""

        # Find the main content
        pmt_sts_rpt = root.find(f".//{ns_prefix}FIToFIPmtStsRpt")
        if pmt_sts_rpt is None:
            pmt_sts_rpt = root

        # Parse Group Header
        grp_hdr = pmt_sts_rpt.find(f"{ns_prefix}GrpHdr")
        if grp_hdr is not None:
            self._parse_group_header(grp_hdr, ns_prefix, message)

        # Parse Original Group Information
        orgnl_grp_inf = pmt_sts_rpt.find(f"{ns_prefix}OrgnlGrpInfAndSts")
        if orgnl_grp_inf is not None:
            self._parse_original_group_info(orgnl_grp_inf, ns_prefix, message)

        # Parse Transaction Information and Status
        for tx_inf in pmt_sts_rpt.findall(f"{ns_prefix}TxInfAndSts"):
            tx_info = self._parse_transaction_info(tx_inf, ns_prefix)
            if tx_info:
                message.transaction_info_and_status.append(tx_info)

        return message

    def _extract_namespace(self, root: ET.Element) -> str:
        """Extract namespace from root element."""
        if root.tag.startswith("{"):
            return root.tag[1 : root.tag.index("}")]
        return ""

    def _parse_group_header(self, grp_hdr: ET.Element, ns: str, message: Pacs002Message) -> None:
        """Parse group header."""
        msg_id = grp_hdr.find(f"{ns}MsgId")
        if msg_id is not None and msg_id.text:
            message.message_id = msg_id.text

        cre_dt_tm = grp_hdr.find(f"{ns}CreDtTm")
        if cre_dt_tm is not None and cre_dt_tm.text:
            try:
                message.creation_datetime = datetime.fromisoformat(cre_dt_tm.text.replace("Z", "+00:00"))
            except ValueError:
                pass

        # Instructing Agent
        instg_agt = grp_hdr.find(f".//{ns}InstgAgt/{ns}FinInstnId/{ns}BICFI")
        if instg_agt is not None and instg_agt.text:
            message.instructing_agent_bic = instg_agt.text

        # Instructed Agent
        instd_agt = grp_hdr.find(f".//{ns}InstdAgt/{ns}FinInstnId/{ns}BICFI")
        if instd_agt is not None and instd_agt.text:
            message.instructed_agent_bic = instd_agt.text

    def _parse_original_group_info(self, orgnl_grp: ET.Element, ns: str, message: Pacs002Message) -> None:
        """Parse original group information."""
        og = message.original_group_info

        # Original Message ID
        orgnl_msg_id = orgnl_grp.find(f"{ns}OrgnlMsgId")
        if orgnl_msg_id is not None and orgnl_msg_id.text:
            og.original_message_id = orgnl_msg_id.text

        # Original Message Name ID
        orgnl_msg_nm_id = orgnl_grp.find(f"{ns}OrgnlMsgNmId")
        if orgnl_msg_nm_id is not None and orgnl_msg_nm_id.text:
            og.original_message_name_id = orgnl_msg_nm_id.text

        # Group Status
        grp_sts = orgnl_grp.find(f"{ns}GrpSts")
        if grp_sts is not None and grp_sts.text:
            og.group_status = TransactionStatus.from_code(grp_sts.text)

        # Status Reason
        for sts_rsn_inf in orgnl_grp.findall(f"{ns}StsRsnInf"):
            reason = self._parse_status_reason(sts_rsn_inf, ns)
            if reason:
                og.status_reason_info.append(reason)

    def _parse_transaction_info(self, tx_inf: ET.Element, ns: str) -> Optional[OriginalTransactionInfo]:
        """Parse transaction information and status."""
        tx = OriginalTransactionInfo()

        # Original Instruction ID
        orgnl_instr_id = tx_inf.find(f"{ns}OrgnlInstrId")
        if orgnl_instr_id is not None and orgnl_instr_id.text:
            tx.original_instruction_id = orgnl_instr_id.text

        # Original End-to-End ID
        orgnl_e2e_id = tx_inf.find(f"{ns}OrgnlEndToEndId")
        if orgnl_e2e_id is not None and orgnl_e2e_id.text:
            tx.original_end_to_end_id = orgnl_e2e_id.text

        # Original Transaction ID
        orgnl_tx_id = tx_inf.find(f"{ns}OrgnlTxId")
        if orgnl_tx_id is not None and orgnl_tx_id.text:
            tx.original_transaction_id = orgnl_tx_id.text

        # Original UETR
        orgnl_uetr = tx_inf.find(f"{ns}OrgnlUETR")
        if orgnl_uetr is not None and orgnl_uetr.text:
            tx.original_uetr = orgnl_uetr.text

        # Transaction Status
        tx_sts = tx_inf.find(f"{ns}TxSts")
        if tx_sts is not None and tx_sts.text:
            status = TransactionStatus.from_code(tx_sts.text)
            if status:
                tx.transaction_status = status

        # Status Reason
        for sts_rsn_inf in tx_inf.findall(f"{ns}StsRsnInf"):
            reason = self._parse_status_reason(sts_rsn_inf, ns)
            if reason:
                tx.status_reason_info.append(reason)

        # Acceptance DateTime
        acpt_dt_tm = tx_inf.find(f"{ns}AccptncDtTm")
        if acpt_dt_tm is not None and acpt_dt_tm.text:
            try:
                tx.acceptance_datetime = datetime.fromisoformat(acpt_dt_tm.text.replace("Z", "+00:00"))
            except ValueError:
                pass

        # Clearing System Reference
        clr_sys_ref = tx_inf.find(f"{ns}ClrSysRef")
        if clr_sys_ref is not None and clr_sys_ref.text:
            tx.clearing_system_ref = clr_sys_ref.text

        return tx

    def _parse_status_reason(self, sts_rsn_inf: ET.Element, ns: str) -> Optional[StatusReasonInfo]:
        """Parse status reason information."""
        reason = StatusReasonInfo()

        # Reason Code
        rsn = sts_rsn_inf.find(f"{ns}Rsn")
        if rsn is not None:
            cd = rsn.find(f"{ns}Cd")
            if cd is not None and cd.text:
                reason.reason_code = StatusReasonCode.from_code(cd.text)

            prtry = rsn.find(f"{ns}Prtry")
            if prtry is not None and prtry.text:
                reason.proprietary_code = prtry.text

        # Additional Information
        for addtl_inf in sts_rsn_inf.findall(f"{ns}AddtlInf"):
            if addtl_inf.text:
                reason.additional_info.append(addtl_inf.text)

        # Originator
        orgtr = sts_rsn_inf.find(f".//{ns}Orgtr/{ns}Nm")
        if orgtr is not None and orgtr.text:
            reason.originator = orgtr.text

        return reason

    def get_errors(self) -> List[str]:
        """Get parsing errors."""
        return self.errors


class Pacs002Builder:
    """Builder for pacs.002 Payment Status Report messages."""

    NAMESPACE = "urn:iso:std:iso:20022:tech:xsd:pacs.002.001.10"

    def __init__(self):
        self._reset()

    def _reset(self):
        """Reset builder state."""
        self._message_id: str = ""
        self._instructing_agent_bic: str = ""
        self._instructed_agent_bic: str = ""
        self._original_message_id: str = ""
        self._original_message_name_id: str = ""
        self._group_status: Optional[TransactionStatus] = None
        self._group_reason: Optional[StatusReasonCode] = None
        self._transactions: List[OriginalTransactionInfo] = []

    def set_message_id(self, message_id: str) -> "Pacs002Builder":
        """Set message ID."""
        self._message_id = message_id[:35]
        return self

    def set_agents(
        self,
        instructing_agent_bic: str,
        instructed_agent_bic: str,
    ) -> "Pacs002Builder":
        """Set instructing and instructed agents."""
        self._instructing_agent_bic = instructing_agent_bic
        self._instructed_agent_bic = instructed_agent_bic
        return self

    def set_original_message(
        self,
        original_message_id: str,
        original_message_name_id: str = "pacs.008.001.08",
    ) -> "Pacs002Builder":
        """Set original message reference."""
        self._original_message_id = original_message_id
        self._original_message_name_id = original_message_name_id
        return self

    def set_group_status(
        self,
        status: TransactionStatus,
        reason: Optional[StatusReasonCode] = None,
    ) -> "Pacs002Builder":
        """Set group-level status."""
        self._group_status = status
        self._group_reason = reason
        return self

    def add_accepted_transaction(
        self,
        original_end_to_end_id: str,
        original_transaction_id: str = "",
        clearing_system_ref: str = "",
    ) -> "Pacs002Builder":
        """Add an accepted transaction."""
        tx = OriginalTransactionInfo(
            original_end_to_end_id=original_end_to_end_id,
            original_transaction_id=original_transaction_id,
            transaction_status=TransactionStatus.ACSC,
            acceptance_datetime=datetime.now(),
            clearing_system_ref=clearing_system_ref,
        )
        self._transactions.append(tx)
        return self

    def add_rejected_transaction(
        self,
        original_end_to_end_id: str,
        reason_code: StatusReasonCode,
        additional_info: str = "",
        original_transaction_id: str = "",
    ) -> "Pacs002Builder":
        """Add a rejected transaction."""
        reason = StatusReasonInfo(
            reason_code=reason_code,
            additional_info=[additional_info] if additional_info else [],
        )
        tx = OriginalTransactionInfo(
            original_end_to_end_id=original_end_to_end_id,
            original_transaction_id=original_transaction_id,
            transaction_status=TransactionStatus.RJCT,
            status_reason_info=[reason],
        )
        self._transactions.append(tx)
        return self

    def build(self) -> Pacs002Message:
        """Build the pacs.002 message."""
        message = Pacs002Message(
            message_id=self._message_id,
            instructing_agent_bic=self._instructing_agent_bic,
            instructed_agent_bic=self._instructed_agent_bic,
            original_group_info=OriginalGroupInfo(
                original_message_id=self._original_message_id,
                original_message_name_id=self._original_message_name_id,
                group_status=self._group_status,
            ),
            transaction_info_and_status=self._transactions.copy(),
        )

        if self._group_reason:
            message.original_group_info.status_reason_info.append(StatusReasonInfo(reason_code=self._group_reason))

        self._reset()
        return message

    def to_xml(self, message: Pacs002Message) -> str:
        """Convert message to XML."""
        root = ET.Element("Document", xmlns=self.NAMESPACE)
        fi_pmt_sts = ET.SubElement(root, "FIToFIPmtStsRpt")

        # Group Header
        grp_hdr = ET.SubElement(fi_pmt_sts, "GrpHdr")
        ET.SubElement(grp_hdr, "MsgId").text = message.message_id
        ET.SubElement(grp_hdr, "CreDtTm").text = message.creation_datetime.isoformat()

        # Original Group Information
        orgnl_grp = ET.SubElement(fi_pmt_sts, "OrgnlGrpInfAndSts")
        ET.SubElement(orgnl_grp, "OrgnlMsgId").text = message.original_group_info.original_message_id
        ET.SubElement(orgnl_grp, "OrgnlMsgNmId").text = message.original_group_info.original_message_name_id

        if message.original_group_info.group_status:
            ET.SubElement(orgnl_grp, "GrpSts").text = message.original_group_info.group_status.code

        for reason in message.original_group_info.status_reason_info:
            self._add_status_reason(orgnl_grp, reason)

        # Transaction Information
        for tx in message.transaction_info_and_status:
            tx_inf = ET.SubElement(fi_pmt_sts, "TxInfAndSts")

            if tx.original_end_to_end_id:
                ET.SubElement(tx_inf, "OrgnlEndToEndId").text = tx.original_end_to_end_id
            if tx.original_transaction_id:
                ET.SubElement(tx_inf, "OrgnlTxId").text = tx.original_transaction_id

            ET.SubElement(tx_inf, "TxSts").text = tx.transaction_status.code

            for reason in tx.status_reason_info:
                self._add_status_reason(tx_inf, reason)

            if tx.acceptance_datetime:
                ET.SubElement(tx_inf, "AccptncDtTm").text = tx.acceptance_datetime.isoformat()

            if tx.clearing_system_ref:
                ET.SubElement(tx_inf, "ClrSysRef").text = tx.clearing_system_ref

        return ET.tostring(root, encoding="unicode", xml_declaration=True)

    def _add_status_reason(self, parent: ET.Element, reason: StatusReasonInfo) -> None:
        """Add status reason element."""
        sts_rsn_inf = ET.SubElement(parent, "StsRsnInf")
        rsn = ET.SubElement(sts_rsn_inf, "Rsn")

        if reason.reason_code:
            ET.SubElement(rsn, "Cd").text = reason.reason_code.code
        elif reason.proprietary_code:
            ET.SubElement(rsn, "Prtry").text = reason.proprietary_code

        for info in reason.additional_info:
            ET.SubElement(sts_rsn_inf, "AddtlInf").text = info
