"""
ISO 20022 pain.002 - Customer Payment Status Report

This message is sent by the Debtor Agent to the Initiating Party to
inform about the positive or negative status of a Customer Credit
Transfer Initiation message (pain.001).

Use Cases:
- Payment initiation acceptance/rejection
- Batch processing status
- Individual transaction status updates
"""

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional
import uuid

from ai_engine.domains.banking.protocols.payments.iso20022.pacs.pacs002 import (
    TransactionStatus,
    StatusReasonCode,
    StatusReasonInfo,
)

logger = logging.getLogger(__name__)


@dataclass
class OriginalPaymentInfo:
    """Original payment information reference and status."""

    # Original payment information identification
    original_payment_info_id: str = ""

    # Status at payment information level
    payment_info_status: Optional[TransactionStatus] = None

    # Status reasons
    status_reason_info: List[StatusReasonInfo] = field(default_factory=list)

    # Number of transactions
    number_of_transactions_per_status: Dict[str, int] = field(default_factory=dict)

    # Transaction statuses
    transaction_statuses: List["TransactionStatusInfo"] = field(default_factory=list)


@dataclass
class TransactionStatusInfo:
    """Individual transaction status information."""

    # Original identification
    original_instruction_id: str = ""
    original_end_to_end_id: str = ""
    original_uetr: str = ""

    # Transaction status
    transaction_status: TransactionStatus = TransactionStatus.ACSC

    # Status reasons
    status_reason_info: List[StatusReasonInfo] = field(default_factory=list)

    # Acceptance datetime
    acceptance_datetime: Optional[datetime] = None

    # Original amounts
    original_instructed_amount: Optional[Decimal] = None
    original_instructed_currency: str = ""

    # Charges deducted
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
            "original_instruction_id": self.original_instruction_id,
            "original_end_to_end_id": self.original_end_to_end_id,
            "status": self.transaction_status.code,
            "status_name": self.transaction_status.status_name,
            "is_accepted": self.is_accepted(),
            "is_rejected": self.is_rejected(),
            "reasons": [
                {
                    "code": r.reason_code.code if r.reason_code else r.proprietary_code,
                    "info": r.additional_info,
                }
                for r in self.status_reason_info
            ],
        }


@dataclass
class Pain002Message:
    """
    ISO 20022 pain.002 - Customer Payment Status Report.

    Reports status of payment initiation messages (pain.001).
    """

    # Message identification
    message_id: str = ""
    creation_datetime: datetime = field(default_factory=datetime.now)

    # Initiating party (original sender)
    initiating_party_name: str = ""
    initiating_party_id: str = ""

    # Debtor agent (reporting bank)
    debtor_agent_bic: str = ""

    # Original group information
    original_message_id: str = ""
    original_message_name_id: str = "pain.001.001.09"
    original_creation_datetime: Optional[datetime] = None

    # Group status
    group_status: Optional[TransactionStatus] = None
    group_status_reason: List[StatusReasonInfo] = field(default_factory=list)

    # Original payment information status
    original_payment_info_and_status: List[OriginalPaymentInfo] = field(default_factory=list)

    # Raw XML
    raw_xml: str = ""

    def __post_init__(self):
        """Generate message ID if not provided."""
        if not self.message_id:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            unique = uuid.uuid4().hex[:8].upper()
            self.message_id = f"PAIN002{timestamp}{unique}"[:35]

    def validate(self) -> List[str]:
        """Validate the message."""
        errors = []

        if not self.message_id:
            errors.append("Message ID is required")
        elif len(self.message_id) > 35:
            errors.append("Message ID must not exceed 35 characters")

        if not self.original_message_id:
            errors.append("Original message ID is required")

        return errors

    @property
    def overall_status(self) -> Optional[TransactionStatus]:
        """Get the overall status."""
        if self.group_status:
            return self.group_status

        # Derive from payment info statuses
        if not self.original_payment_info_and_status:
            return None

        statuses = set()
        for pmt_info in self.original_payment_info_and_status:
            if pmt_info.payment_info_status:
                statuses.add(pmt_info.payment_info_status)
            for tx in pmt_info.transaction_statuses:
                statuses.add(tx.transaction_status)

        if len(statuses) == 1:
            return list(statuses)[0]

        if TransactionStatus.RJCT in statuses:
            return TransactionStatus.PART

        return TransactionStatus.PDNG

    @property
    def all_accepted(self) -> bool:
        """Check if all transactions were accepted."""
        if self.group_status:
            return self.group_status.is_positive

        for pmt_info in self.original_payment_info_and_status:
            if pmt_info.payment_info_status and not pmt_info.payment_info_status.is_positive:
                return False
            for tx in pmt_info.transaction_statuses:
                if not tx.is_accepted():
                    return False
        return True

    @property
    def any_rejected(self) -> bool:
        """Check if any transactions were rejected."""
        if self.group_status == TransactionStatus.RJCT:
            return True

        for pmt_info in self.original_payment_info_and_status:
            if pmt_info.payment_info_status == TransactionStatus.RJCT:
                return True
            for tx in pmt_info.transaction_statuses:
                if tx.is_rejected():
                    return True
        return False

    def get_rejected_transactions(self) -> List[TransactionStatusInfo]:
        """Get all rejected transactions."""
        rejected = []
        for pmt_info in self.original_payment_info_and_status:
            for tx in pmt_info.transaction_statuses:
                if tx.is_rejected():
                    rejected.append(tx)
        return rejected

    def get_rejection_reasons(self) -> List[Dict]:
        """Get summary of all rejection reasons."""
        reasons = []

        # Group level reasons
        for reason in self.group_status_reason:
            reasons.append(
                {
                    "level": "group",
                    "code": reason.reason_code.code if reason.reason_code else reason.proprietary_code,
                    "description": reason.reason_code.description if reason.reason_code else "",
                    "additional_info": reason.additional_info,
                }
            )

        # Payment info level reasons
        for pmt_info in self.original_payment_info_and_status:
            for reason in pmt_info.status_reason_info:
                reasons.append(
                    {
                        "level": "payment_info",
                        "payment_info_id": pmt_info.original_payment_info_id,
                        "code": reason.reason_code.code if reason.reason_code else reason.proprietary_code,
                        "description": reason.reason_code.description if reason.reason_code else "",
                        "additional_info": reason.additional_info,
                    }
                )

            # Transaction level reasons
            for tx in pmt_info.transaction_statuses:
                for reason in tx.status_reason_info:
                    reasons.append(
                        {
                            "level": "transaction",
                            "end_to_end_id": tx.original_end_to_end_id,
                            "code": reason.reason_code.code if reason.reason_code else reason.proprietary_code,
                            "description": reason.reason_code.description if reason.reason_code else "",
                            "additional_info": reason.additional_info,
                        }
                    )

        return reasons

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "message_id": self.message_id,
            "creation_datetime": self.creation_datetime.isoformat(),
            "original_message_id": self.original_message_id,
            "overall_status": self.overall_status.code if self.overall_status else None,
            "all_accepted": self.all_accepted,
            "any_rejected": self.any_rejected,
            "rejection_reasons": self.get_rejection_reasons() if self.any_rejected else [],
            "payment_info": [
                {
                    "id": pi.original_payment_info_id,
                    "status": pi.payment_info_status.code if pi.payment_info_status else None,
                    "transactions": [tx.to_dict() for tx in pi.transaction_statuses],
                }
                for pi in self.original_payment_info_and_status
            ],
        }


class Pain002Parser:
    """Parser for pain.002 Customer Payment Status Report messages."""

    NAMESPACE = "urn:iso:std:iso:20022:tech:xsd:pain.002.001.10"

    def __init__(self, strict: bool = True):
        self.strict = strict
        self.errors: List[str] = []

    def parse(self, xml_content: str) -> Pain002Message:
        """
        Parse a pain.002 XML message.

        Args:
            xml_content: XML string

        Returns:
            Parsed Pain002Message
        """
        self.errors = []
        message = Pain002Message()
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
        cstmr_pmt_sts = root.find(f".//{ns_prefix}CstmrPmtStsRpt")
        if cstmr_pmt_sts is None:
            cstmr_pmt_sts = root

        # Parse Group Header
        grp_hdr = cstmr_pmt_sts.find(f"{ns_prefix}GrpHdr")
        if grp_hdr is not None:
            self._parse_group_header(grp_hdr, ns_prefix, message)

        # Parse Original Group Information
        orgnl_grp_inf = cstmr_pmt_sts.find(f"{ns_prefix}OrgnlGrpInfAndSts")
        if orgnl_grp_inf is not None:
            self._parse_original_group_info(orgnl_grp_inf, ns_prefix, message)

        # Parse Original Payment Information
        for orgnl_pmt_inf in cstmr_pmt_sts.findall(f"{ns_prefix}OrgnlPmtInfAndSts"):
            pmt_info = self._parse_original_payment_info(orgnl_pmt_inf, ns_prefix)
            if pmt_info:
                message.original_payment_info_and_status.append(pmt_info)

        return message

    def _extract_namespace(self, root: ET.Element) -> str:
        """Extract namespace from root element."""
        if root.tag.startswith("{"):
            return root.tag[1 : root.tag.index("}")]
        return ""

    def _parse_group_header(self, grp_hdr: ET.Element, ns: str, message: Pain002Message) -> None:
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

        # Initiating Party
        init_pty = grp_hdr.find(f"{ns}InitgPty")
        if init_pty is not None:
            nm = init_pty.find(f"{ns}Nm")
            if nm is not None and nm.text:
                message.initiating_party_name = nm.text

        # Debtor Agent
        dbtr_agt = grp_hdr.find(f".//{ns}DbtrAgt/{ns}FinInstnId/{ns}BICFI")
        if dbtr_agt is not None and dbtr_agt.text:
            message.debtor_agent_bic = dbtr_agt.text

    def _parse_original_group_info(self, orgnl_grp: ET.Element, ns: str, message: Pain002Message) -> None:
        """Parse original group information."""
        # Original Message ID
        orgnl_msg_id = orgnl_grp.find(f"{ns}OrgnlMsgId")
        if orgnl_msg_id is not None and orgnl_msg_id.text:
            message.original_message_id = orgnl_msg_id.text

        # Original Message Name ID
        orgnl_msg_nm_id = orgnl_grp.find(f"{ns}OrgnlMsgNmId")
        if orgnl_msg_nm_id is not None and orgnl_msg_nm_id.text:
            message.original_message_name_id = orgnl_msg_nm_id.text

        # Original Creation DateTime
        orgnl_cre_dt_tm = orgnl_grp.find(f"{ns}OrgnlCreDtTm")
        if orgnl_cre_dt_tm is not None and orgnl_cre_dt_tm.text:
            try:
                message.original_creation_datetime = datetime.fromisoformat(orgnl_cre_dt_tm.text.replace("Z", "+00:00"))
            except ValueError:
                pass

        # Group Status
        grp_sts = orgnl_grp.find(f"{ns}GrpSts")
        if grp_sts is not None and grp_sts.text:
            message.group_status = TransactionStatus.from_code(grp_sts.text)

        # Status Reason
        for sts_rsn_inf in orgnl_grp.findall(f"{ns}StsRsnInf"):
            reason = self._parse_status_reason(sts_rsn_inf, ns)
            if reason:
                message.group_status_reason.append(reason)

    def _parse_original_payment_info(self, orgnl_pmt: ET.Element, ns: str) -> Optional[OriginalPaymentInfo]:
        """Parse original payment information."""
        pmt_info = OriginalPaymentInfo()

        # Original Payment Info ID
        orgnl_pmt_inf_id = orgnl_pmt.find(f"{ns}OrgnlPmtInfId")
        if orgnl_pmt_inf_id is not None and orgnl_pmt_inf_id.text:
            pmt_info.original_payment_info_id = orgnl_pmt_inf_id.text

        # Payment Info Status
        pmt_inf_sts = orgnl_pmt.find(f"{ns}PmtInfSts")
        if pmt_inf_sts is not None and pmt_inf_sts.text:
            pmt_info.payment_info_status = TransactionStatus.from_code(pmt_inf_sts.text)

        # Status Reason
        for sts_rsn_inf in orgnl_pmt.findall(f"{ns}StsRsnInf"):
            reason = self._parse_status_reason(sts_rsn_inf, ns)
            if reason:
                pmt_info.status_reason_info.append(reason)

        # Transaction Information
        for tx_inf in orgnl_pmt.findall(f"{ns}TxInfAndSts"):
            tx = self._parse_transaction_info(tx_inf, ns)
            if tx:
                pmt_info.transaction_statuses.append(tx)

        return pmt_info

    def _parse_transaction_info(self, tx_inf: ET.Element, ns: str) -> Optional[TransactionStatusInfo]:
        """Parse transaction status information."""
        tx = TransactionStatusInfo()

        # Original Instruction ID
        orgnl_instr_id = tx_inf.find(f"{ns}OrgnlInstrId")
        if orgnl_instr_id is not None and orgnl_instr_id.text:
            tx.original_instruction_id = orgnl_instr_id.text

        # Original End-to-End ID
        orgnl_e2e_id = tx_inf.find(f"{ns}OrgnlEndToEndId")
        if orgnl_e2e_id is not None and orgnl_e2e_id.text:
            tx.original_end_to_end_id = orgnl_e2e_id.text

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

        # Original Instructed Amount
        orgnl_instd_amt = tx_inf.find(f"{ns}OrgnlInstdAmt")
        if orgnl_instd_amt is not None and orgnl_instd_amt.text:
            try:
                tx.original_instructed_amount = Decimal(orgnl_instd_amt.text)
                tx.original_instructed_currency = orgnl_instd_amt.get("Ccy", "")
            except Exception:
                pass

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

        return reason

    def get_errors(self) -> List[str]:
        """Get parsing errors."""
        return self.errors


class Pain002Builder:
    """Builder for pain.002 Customer Payment Status Report messages."""

    NAMESPACE = "urn:iso:std:iso:20022:tech:xsd:pain.002.001.10"

    def __init__(self):
        self._reset()

    def _reset(self):
        """Reset builder state."""
        self._message_id: str = ""
        self._initiating_party_name: str = ""
        self._debtor_agent_bic: str = ""
        self._original_message_id: str = ""
        self._original_message_name_id: str = "pain.001.001.09"
        self._group_status: Optional[TransactionStatus] = None
        self._group_reason: Optional[StatusReasonCode] = None
        self._payment_infos: List[OriginalPaymentInfo] = []

    def set_message_id(self, message_id: str) -> "Pain002Builder":
        """Set message ID."""
        self._message_id = message_id[:35]
        return self

    def set_initiating_party(self, name: str) -> "Pain002Builder":
        """Set initiating party name."""
        self._initiating_party_name = name
        return self

    def set_debtor_agent(self, bic: str) -> "Pain002Builder":
        """Set debtor agent BIC."""
        self._debtor_agent_bic = bic
        return self

    def set_original_message(
        self,
        original_message_id: str,
        original_message_name_id: str = "pain.001.001.09",
    ) -> "Pain002Builder":
        """Set original message reference."""
        self._original_message_id = original_message_id
        self._original_message_name_id = original_message_name_id
        return self

    def set_group_status(
        self,
        status: TransactionStatus,
        reason: Optional[StatusReasonCode] = None,
    ) -> "Pain002Builder":
        """Set group-level status."""
        self._group_status = status
        self._group_reason = reason
        return self

    def add_payment_info_accepted(
        self,
        original_payment_info_id: str,
        transactions: List[Dict] = None,
    ) -> "Pain002Builder":
        """Add an accepted payment information block."""
        pmt_info = OriginalPaymentInfo(
            original_payment_info_id=original_payment_info_id,
            payment_info_status=TransactionStatus.ACSC,
        )

        if transactions:
            for tx in transactions:
                tx_info = TransactionStatusInfo(
                    original_end_to_end_id=tx.get("end_to_end_id", ""),
                    original_instruction_id=tx.get("instruction_id", ""),
                    transaction_status=TransactionStatus.ACSC,
                    acceptance_datetime=datetime.now(),
                )
                pmt_info.transaction_statuses.append(tx_info)

        self._payment_infos.append(pmt_info)
        return self

    def add_payment_info_rejected(
        self,
        original_payment_info_id: str,
        reason_code: StatusReasonCode,
        additional_info: str = "",
        transactions: List[Dict] = None,
    ) -> "Pain002Builder":
        """Add a rejected payment information block."""
        pmt_info = OriginalPaymentInfo(
            original_payment_info_id=original_payment_info_id,
            payment_info_status=TransactionStatus.RJCT,
            status_reason_info=[
                StatusReasonInfo(
                    reason_code=reason_code,
                    additional_info=[additional_info] if additional_info else [],
                )
            ],
        )

        if transactions:
            for tx in transactions:
                tx_reason = tx.get("reason_code", reason_code)
                tx_info = TransactionStatusInfo(
                    original_end_to_end_id=tx.get("end_to_end_id", ""),
                    original_instruction_id=tx.get("instruction_id", ""),
                    transaction_status=TransactionStatus.RJCT,
                    status_reason_info=[
                        StatusReasonInfo(
                            reason_code=tx_reason,
                            additional_info=[tx.get("additional_info", "")] if tx.get("additional_info") else [],
                        )
                    ],
                )
                pmt_info.transaction_statuses.append(tx_info)

        self._payment_infos.append(pmt_info)
        return self

    def build(self) -> Pain002Message:
        """Build the pain.002 message."""
        message = Pain002Message(
            message_id=self._message_id,
            initiating_party_name=self._initiating_party_name,
            debtor_agent_bic=self._debtor_agent_bic,
            original_message_id=self._original_message_id,
            original_message_name_id=self._original_message_name_id,
            group_status=self._group_status,
            original_payment_info_and_status=self._payment_infos.copy(),
        )

        if self._group_reason:
            message.group_status_reason.append(StatusReasonInfo(reason_code=self._group_reason))

        self._reset()
        return message

    def to_xml(self, message: Pain002Message) -> str:
        """Convert message to XML."""
        root = ET.Element("Document", xmlns=self.NAMESPACE)
        cstmr_pmt_sts = ET.SubElement(root, "CstmrPmtStsRpt")

        # Group Header
        grp_hdr = ET.SubElement(cstmr_pmt_sts, "GrpHdr")
        ET.SubElement(grp_hdr, "MsgId").text = message.message_id
        ET.SubElement(grp_hdr, "CreDtTm").text = message.creation_datetime.isoformat()

        if message.initiating_party_name:
            init_pty = ET.SubElement(grp_hdr, "InitgPty")
            ET.SubElement(init_pty, "Nm").text = message.initiating_party_name

        # Original Group Information
        orgnl_grp = ET.SubElement(cstmr_pmt_sts, "OrgnlGrpInfAndSts")
        ET.SubElement(orgnl_grp, "OrgnlMsgId").text = message.original_message_id
        ET.SubElement(orgnl_grp, "OrgnlMsgNmId").text = message.original_message_name_id

        if message.group_status:
            ET.SubElement(orgnl_grp, "GrpSts").text = message.group_status.code

        # Original Payment Information
        for pmt_info in message.original_payment_info_and_status:
            orgnl_pmt = ET.SubElement(cstmr_pmt_sts, "OrgnlPmtInfAndSts")
            ET.SubElement(orgnl_pmt, "OrgnlPmtInfId").text = pmt_info.original_payment_info_id

            if pmt_info.payment_info_status:
                ET.SubElement(orgnl_pmt, "PmtInfSts").text = pmt_info.payment_info_status.code

            for reason in pmt_info.status_reason_info:
                self._add_status_reason(orgnl_pmt, reason)

            for tx in pmt_info.transaction_statuses:
                tx_inf = ET.SubElement(orgnl_pmt, "TxInfAndSts")

                if tx.original_instruction_id:
                    ET.SubElement(tx_inf, "OrgnlInstrId").text = tx.original_instruction_id
                if tx.original_end_to_end_id:
                    ET.SubElement(tx_inf, "OrgnlEndToEndId").text = tx.original_end_to_end_id

                ET.SubElement(tx_inf, "TxSts").text = tx.transaction_status.code

                for reason in tx.status_reason_info:
                    self._add_status_reason(tx_inf, reason)

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
