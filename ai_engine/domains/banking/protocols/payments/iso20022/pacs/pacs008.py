"""
ISO 20022 pacs.008 - FI to FI Customer Credit Transfer

This message is sent by the debtor agent to the creditor agent, directly or
through other agents and/or a payment clearing and settlement system.
It is used to move funds from a debtor account to a creditor.

Used by: SWIFT, FedWire, CHIPS, TARGET2, SEPA clearing
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET
import logging
import uuid

from ai_engine.domains.banking.protocols.payments.iso20022.base import (
    ISO20022Message,
    ISO20022MessageType,
    ISO20022Parser,
    ISO20022Builder,
    Amount,
    AccountIdentification,
    FinancialInstitution,
    PartyIdentification,
    TransactionStatus,
)

logger = logging.getLogger(__name__)


class SettlementMethod(str):
    """Settlement method codes."""

    INDA = "INDA"  # Instructed Agent
    INGA = "INGA"  # Instructing Agent
    COVE = "COVE"  # Cover Payment
    CLRG = "CLRG"  # Clearing System


class ClearingSystem(str):
    """Clearing system codes."""

    FEDWIRE = "FEDWIR"
    CHIPS = "CHIPS"
    TARGET2 = "TGT"
    SEPA = "SEPA"
    SWIFT = "SWIFT"


@dataclass
class SettlementInformation:
    """Settlement information."""

    settlement_method: str = SettlementMethod.CLRG
    clearing_system: Optional[str] = None
    settlement_account: Optional[AccountIdentification] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "settlement_method": self.settlement_method,
            "clearing_system": self.clearing_system,
        }


@dataclass
class PaymentTypeInformation:
    """Payment type information."""

    instruction_priority: str = "NORM"  # HIGH, NORM
    service_level: Optional[str] = None  # SEPA, URGP, etc.
    local_instrument: Optional[str] = None
    category_purpose: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instruction_priority": self.instruction_priority,
            "service_level": self.service_level,
            "local_instrument": self.local_instrument,
            "category_purpose": self.category_purpose,
        }


@dataclass
class FIToFICreditTransfer:
    """Individual FI to FI credit transfer transaction."""

    # Transaction identification
    instruction_id: str = ""
    end_to_end_id: str = ""
    transaction_id: str = ""  # UETR for SWIFT gpi
    uetr: Optional[str] = None  # Unique End-to-end Transaction Reference

    # Status
    status: Optional[TransactionStatus] = None

    # Amount
    interbank_settlement_amount: Optional[Amount] = None
    instructed_amount: Optional[Amount] = None
    exchange_rate: Optional[Decimal] = None

    # Dates
    interbank_settlement_date: Optional[date] = None
    acceptance_datetime: Optional[datetime] = None

    # Charge information
    charge_bearer: str = "SLEV"
    charges: List[Dict[str, Any]] = field(default_factory=list)

    # Agents
    instructing_agent: Optional[FinancialInstitution] = None
    instructed_agent: Optional[FinancialInstitution] = None
    intermediary_agents: List[FinancialInstitution] = field(default_factory=list)

    # Parties
    debtor: Optional[PartyIdentification] = None
    debtor_account: Optional[AccountIdentification] = None
    debtor_agent: Optional[FinancialInstitution] = None

    creditor: Optional[PartyIdentification] = None
    creditor_account: Optional[AccountIdentification] = None
    creditor_agent: Optional[FinancialInstitution] = None
    ultimate_creditor: Optional[PartyIdentification] = None

    # Payment details
    purpose: Optional[str] = None
    remittance_information: Optional[str] = None
    structured_remittance: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instruction_id": self.instruction_id,
            "end_to_end_id": self.end_to_end_id,
            "transaction_id": self.transaction_id,
            "uetr": self.uetr,
            "interbank_settlement_amount": {
                "value": str(self.interbank_settlement_amount.value) if self.interbank_settlement_amount else None,
                "currency": self.interbank_settlement_amount.currency if self.interbank_settlement_amount else None,
            },
            "interbank_settlement_date": (
                self.interbank_settlement_date.isoformat() if self.interbank_settlement_date else None
            ),
            "charge_bearer": self.charge_bearer,
            "debtor_name": self.debtor.name if self.debtor else None,
            "creditor_name": self.creditor.name if self.creditor else None,
            "remittance_information": self.remittance_information,
        }


@dataclass
class Pacs008Message(ISO20022Message):
    """
    ISO 20022 pacs.008 - FI to FI Customer Credit Transfer

    Structure:
    - Document
      - FIToFICstmrCdtTrf (FIToFICustomerCreditTransfer)
        - GrpHdr (GroupHeader)
        - CdtTrfTxInf (CreditTransferTransactionInformation) [1..n]
    """

    message_type: ISO20022MessageType = ISO20022MessageType.PACS_008

    # Group header
    number_of_transactions: int = 0
    total_interbank_settlement_amount: Optional[Amount] = None
    interbank_settlement_date: Optional[date] = None
    settlement_information: Optional[SettlementInformation] = None
    payment_type_information: Optional[PaymentTypeInformation] = None

    # Agents
    instructing_agent: Optional[FinancialInstitution] = None
    instructed_agent: Optional[FinancialInstitution] = None

    # Transactions
    credit_transfers: List[FIToFICreditTransfer] = field(default_factory=list)

    def validate(self) -> List[str]:
        """Validate pacs.008 message."""
        errors = []

        # Validate group header
        if not self.message_id:
            errors.append("Message ID is required")

        if not self.settlement_information:
            errors.append("Settlement information is required")

        if not self.credit_transfers:
            errors.append("At least one credit transfer is required")

        # Validate transactions
        total_amount = Decimal("0")

        for idx, txn in enumerate(self.credit_transfers):
            if not txn.instruction_id:
                errors.append(f"Transaction {idx}: Instruction ID is required")

            if not txn.end_to_end_id:
                errors.append(f"Transaction {idx}: End-to-end ID is required")

            if not txn.interbank_settlement_amount:
                errors.append(f"Transaction {idx}: Interbank settlement amount is required")
            else:
                if txn.interbank_settlement_amount.value <= 0:
                    errors.append(f"Transaction {idx}: Amount must be positive")
                total_amount += txn.interbank_settlement_amount.value

            if not txn.creditor:
                errors.append(f"Transaction {idx}: Creditor is required")

            if not txn.creditor_agent:
                errors.append(f"Transaction {idx}: Creditor agent is required")

        # Validate totals
        if self.total_interbank_settlement_amount:
            if self.total_interbank_settlement_amount.value != total_amount:
                errors.append(
                    f"Total amount mismatch: header={self.total_interbank_settlement_amount.value}, " f"actual={total_amount}"
                )

        return errors

    def to_xml(self) -> str:
        """Serialize to XML."""
        builder = Pacs008Builder()
        return builder.build_from_message(self)

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "number_of_transactions": self.number_of_transactions,
                "total_interbank_settlement_amount": {
                    "value": (
                        str(self.total_interbank_settlement_amount.value) if self.total_interbank_settlement_amount else None
                    ),
                    "currency": (
                        self.total_interbank_settlement_amount.currency if self.total_interbank_settlement_amount else None
                    ),
                },
                "interbank_settlement_date": (
                    self.interbank_settlement_date.isoformat() if self.interbank_settlement_date else None
                ),
                "settlement_information": self.settlement_information.to_dict() if self.settlement_information else None,
                "credit_transfers": [ct.to_dict() for ct in self.credit_transfers],
            }
        )
        return base


class Pacs008Parser(ISO20022Parser):
    """Parser for pacs.008 messages."""

    def parse(self, xml_content: str) -> Pacs008Message:
        """Parse pacs.008 XML to message object."""
        self.errors = []

        root = self._parse_xml(xml_content)
        message = Pacs008Message(raw_xml=xml_content)

        # Detect namespace
        ns = self._detect_namespace(root)

        # Find main element
        fi_to_fi = self._find_element(root, "FIToFICstmrCdtTrf", ns)
        if fi_to_fi is None:
            self.errors.append("FIToFICstmrCdtTrf element not found")
            return message

        # Parse group header
        grp_hdr = self._find_element(fi_to_fi, "GrpHdr", ns)
        if grp_hdr is not None:
            self._parse_group_header(grp_hdr, message, ns)

        # Parse credit transfers
        for cdt_trf_tx_inf in self._find_all_elements(fi_to_fi, "CdtTrfTxInf", ns):
            credit_transfer = self._parse_credit_transfer(cdt_trf_tx_inf, ns)
            message.credit_transfers.append(credit_transfer)

        message.number_of_transactions = len(message.credit_transfers)

        return message

    def _detect_namespace(self, root: ET.Element) -> Dict[str, str]:
        """Detect namespace from root element."""
        ns = {}
        tag = root.tag
        if "{" in tag:
            namespace = tag[1 : tag.index("}")]
            ns["ns"] = namespace
        return ns

    def _find_element(self, parent: ET.Element, name: str, ns: Dict[str, str]) -> Optional[ET.Element]:
        """Find element with namespace handling."""
        if ns:
            elem = parent.find(f"ns:{name}", ns)
            if elem is not None:
                return elem

        elem = parent.find(name)
        if elem is not None:
            return elem

        for child in parent:
            local_name = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if local_name == name:
                return child

        return None

    def _find_all_elements(self, parent: ET.Element, name: str, ns: Dict[str, str]) -> List[ET.Element]:
        """Find all elements with namespace handling."""
        results = []

        if ns:
            results = parent.findall(f"ns:{name}", ns)
            if results:
                return results

        results = parent.findall(name)
        if results:
            return results

        for child in parent:
            local_name = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if local_name == name:
                results.append(child)

        return results

    def _parse_group_header(self, grp_hdr: ET.Element, message: Pacs008Message, ns: Dict[str, str]):
        """Parse group header."""
        msg_id = self._find_element(grp_hdr, "MsgId", ns)
        message.message_id = self._get_text(msg_id)

        cre_dt_tm = self._find_element(grp_hdr, "CreDtTm", ns)
        message.creation_datetime = self._get_datetime(cre_dt_tm) or datetime.utcnow()

        nb_of_txs = self._find_element(grp_hdr, "NbOfTxs", ns)
        message.number_of_transactions = int(self._get_text(nb_of_txs, "0"))

        # Total amount
        ttl_intrbnk_sttlm_amt = self._find_element(grp_hdr, "TtlIntrBkSttlmAmt", ns)
        if ttl_intrbnk_sttlm_amt is not None:
            message.total_interbank_settlement_amount = Amount.from_xml(ttl_intrbnk_sttlm_amt)

        # Settlement date
        intrbnk_sttlm_dt = self._find_element(grp_hdr, "IntrBkSttlmDt", ns)
        if intrbnk_sttlm_dt is not None and intrbnk_sttlm_dt.text:
            try:
                message.interbank_settlement_date = date.fromisoformat(intrbnk_sttlm_dt.text)
            except ValueError:
                pass

        # Settlement information
        sttlm_inf = self._find_element(grp_hdr, "SttlmInf", ns)
        if sttlm_inf is not None:
            message.settlement_information = self._parse_settlement_info(sttlm_inf, ns)

        # Instructing agent
        instg_agt = self._find_element(grp_hdr, "InstgAgt", ns)
        if instg_agt is not None:
            fin_instn_id = self._find_element(instg_agt, "FinInstnId", ns)
            if fin_instn_id is not None:
                message.instructing_agent = FinancialInstitution.from_xml(fin_instn_id, ns)

        # Instructed agent
        instd_agt = self._find_element(grp_hdr, "InstdAgt", ns)
        if instd_agt is not None:
            fin_instn_id = self._find_element(instd_agt, "FinInstnId", ns)
            if fin_instn_id is not None:
                message.instructed_agent = FinancialInstitution.from_xml(fin_instn_id, ns)

    def _parse_settlement_info(self, sttlm_inf: ET.Element, ns: Dict[str, str]) -> SettlementInformation:
        """Parse settlement information."""
        info = SettlementInformation()

        sttlm_mtd = self._find_element(sttlm_inf, "SttlmMtd", ns)
        info.settlement_method = self._get_text(sttlm_mtd, "CLRG")

        clr_sys = self._find_element(sttlm_inf, "ClrSys", ns)
        if clr_sys is not None:
            cd = self._find_element(clr_sys, "Cd", ns)
            info.clearing_system = self._get_text(cd)

        return info

    def _parse_credit_transfer(self, cdt_trf_tx_inf: ET.Element, ns: Dict[str, str]) -> FIToFICreditTransfer:
        """Parse credit transfer transaction."""
        txn = FIToFICreditTransfer()

        # Payment ID
        pmt_id = self._find_element(cdt_trf_tx_inf, "PmtId", ns)
        if pmt_id is not None:
            instr_id = self._find_element(pmt_id, "InstrId", ns)
            txn.instruction_id = self._get_text(instr_id)

            end_to_end_id = self._find_element(pmt_id, "EndToEndId", ns)
            txn.end_to_end_id = self._get_text(end_to_end_id)

            tx_id = self._find_element(pmt_id, "TxId", ns)
            txn.transaction_id = self._get_text(tx_id)

            uetr = self._find_element(pmt_id, "UETR", ns)
            txn.uetr = self._get_text(uetr) or None

        # Interbank settlement amount
        intrbnk_sttlm_amt = self._find_element(cdt_trf_tx_inf, "IntrBkSttlmAmt", ns)
        if intrbnk_sttlm_amt is not None:
            txn.interbank_settlement_amount = Amount.from_xml(intrbnk_sttlm_amt)

        # Settlement date
        intrbnk_sttlm_dt = self._find_element(cdt_trf_tx_inf, "IntrBkSttlmDt", ns)
        if intrbnk_sttlm_dt is not None and intrbnk_sttlm_dt.text:
            try:
                txn.interbank_settlement_date = date.fromisoformat(intrbnk_sttlm_dt.text)
            except ValueError:
                pass

        # Charge bearer
        chrg_br = self._find_element(cdt_trf_tx_inf, "ChrgBr", ns)
        txn.charge_bearer = self._get_text(chrg_br, "SLEV")

        # Instructing agent
        instg_agt = self._find_element(cdt_trf_tx_inf, "InstgAgt", ns)
        if instg_agt is not None:
            fin_instn_id = self._find_element(instg_agt, "FinInstnId", ns)
            if fin_instn_id is not None:
                txn.instructing_agent = FinancialInstitution.from_xml(fin_instn_id, ns)

        # Instructed agent
        instd_agt = self._find_element(cdt_trf_tx_inf, "InstdAgt", ns)
        if instd_agt is not None:
            fin_instn_id = self._find_element(instd_agt, "FinInstnId", ns)
            if fin_instn_id is not None:
                txn.instructed_agent = FinancialInstitution.from_xml(fin_instn_id, ns)

        # Debtor
        dbtr = self._find_element(cdt_trf_tx_inf, "Dbtr", ns)
        if dbtr is not None:
            txn.debtor = PartyIdentification.from_xml(dbtr, ns)

        # Debtor account
        dbtr_acct = self._find_element(cdt_trf_tx_inf, "DbtrAcct", ns)
        if dbtr_acct is not None:
            acct_id = self._find_element(dbtr_acct, "Id", ns)
            if acct_id is not None:
                txn.debtor_account = AccountIdentification.from_xml(acct_id, ns)

        # Debtor agent
        dbtr_agt = self._find_element(cdt_trf_tx_inf, "DbtrAgt", ns)
        if dbtr_agt is not None:
            fin_instn_id = self._find_element(dbtr_agt, "FinInstnId", ns)
            if fin_instn_id is not None:
                txn.debtor_agent = FinancialInstitution.from_xml(fin_instn_id, ns)

        # Creditor agent
        cdtr_agt = self._find_element(cdt_trf_tx_inf, "CdtrAgt", ns)
        if cdtr_agt is not None:
            fin_instn_id = self._find_element(cdtr_agt, "FinInstnId", ns)
            if fin_instn_id is not None:
                txn.creditor_agent = FinancialInstitution.from_xml(fin_instn_id, ns)

        # Creditor
        cdtr = self._find_element(cdt_trf_tx_inf, "Cdtr", ns)
        if cdtr is not None:
            txn.creditor = PartyIdentification.from_xml(cdtr, ns)

        # Creditor account
        cdtr_acct = self._find_element(cdt_trf_tx_inf, "CdtrAcct", ns)
        if cdtr_acct is not None:
            acct_id = self._find_element(cdtr_acct, "Id", ns)
            if acct_id is not None:
                txn.creditor_account = AccountIdentification.from_xml(acct_id, ns)

        # Remittance information
        rmt_inf = self._find_element(cdt_trf_tx_inf, "RmtInf", ns)
        if rmt_inf is not None:
            ustrd = self._find_element(rmt_inf, "Ustrd", ns)
            txn.remittance_information = self._get_text(ustrd)

        return txn


class Pacs008Builder(ISO20022Builder):
    """Builder for pacs.008 messages."""

    def __init__(self, version: str = "02"):
        super().__init__(ISO20022MessageType.PACS_008, version)

    def build(
        self,
        message_id: str,
        settlement_method: str = "CLRG",
        clearing_system: Optional[str] = None,
        instructing_agent_bic: str = "",
        instructed_agent_bic: str = "",
        transactions: List[Dict[str, Any]] = None,
        settlement_date: Optional[date] = None,
    ) -> str:
        """Build pacs.008 message."""
        transactions = transactions or []

        root = self._create_root()

        # FI to FI Customer Credit Transfer
        fi_to_fi = self._add_element(root, "FIToFICstmrCdtTrf")

        # Group Header
        grp_hdr = self._add_element(fi_to_fi, "GrpHdr")
        self._add_element(grp_hdr, "MsgId", message_id)
        self._add_element(grp_hdr, "CreDtTm", self._format_datetime(datetime.utcnow()))
        self._add_element(grp_hdr, "NbOfTxs", str(len(transactions)))

        # Total amount
        if transactions:
            total = sum(Decimal(str(t.get("amount", 0))) for t in transactions)
            currency = transactions[0].get("currency", "USD")
            ttl_amt = self._add_element(grp_hdr, "TtlIntrBkSttlmAmt", f"{total:.2f}")
            ttl_amt.set("Ccy", currency)

        # Settlement date
        sttlm_dt = settlement_date or date.today()
        self._add_element(grp_hdr, "IntrBkSttlmDt", sttlm_dt.isoformat())

        # Settlement Information
        sttlm_inf = self._add_element(grp_hdr, "SttlmInf")
        self._add_element(sttlm_inf, "SttlmMtd", settlement_method)
        if clearing_system:
            clr_sys = self._add_element(sttlm_inf, "ClrSys")
            self._add_element(clr_sys, "Cd", clearing_system)

        # Instructing Agent
        if instructing_agent_bic:
            instg_agt = self._add_element(grp_hdr, "InstgAgt")
            fin_instn_id = self._add_element(instg_agt, "FinInstnId")
            self._add_element(fin_instn_id, "BICFI", instructing_agent_bic)

        # Instructed Agent
        if instructed_agent_bic:
            instd_agt = self._add_element(grp_hdr, "InstdAgt")
            fin_instn_id = self._add_element(instd_agt, "FinInstnId")
            self._add_element(fin_instn_id, "BICFI", instructed_agent_bic)

        # Credit Transfer Transactions
        for idx, txn in enumerate(transactions):
            cdt_trf_tx_inf = self._add_element(fi_to_fi, "CdtTrfTxInf")
            self._build_transaction(cdt_trf_tx_inf, txn, message_id, idx)

        return self._to_string(root)

    def build_from_message(self, message: Pacs008Message) -> str:
        """Build XML from Pacs008Message object."""
        root = self._create_root()

        fi_to_fi = self._add_element(root, "FIToFICstmrCdtTrf")

        # Group Header
        grp_hdr = self._add_element(fi_to_fi, "GrpHdr")
        self._add_element(grp_hdr, "MsgId", message.message_id)
        self._add_element(grp_hdr, "CreDtTm", self._format_datetime(message.creation_datetime))
        self._add_element(grp_hdr, "NbOfTxs", str(message.number_of_transactions))

        if message.total_interbank_settlement_amount:
            ttl_amt = self._add_element(grp_hdr, "TtlIntrBkSttlmAmt", message.total_interbank_settlement_amount.to_xml_value())
            ttl_amt.set("Ccy", message.total_interbank_settlement_amount.currency)

        if message.interbank_settlement_date:
            self._add_element(grp_hdr, "IntrBkSttlmDt", message.interbank_settlement_date.isoformat())

        # Settlement Information
        if message.settlement_information:
            sttlm_inf = self._add_element(grp_hdr, "SttlmInf")
            self._add_element(sttlm_inf, "SttlmMtd", message.settlement_information.settlement_method)
            if message.settlement_information.clearing_system:
                clr_sys = self._add_element(sttlm_inf, "ClrSys")
                self._add_element(clr_sys, "Cd", message.settlement_information.clearing_system)

        # Agents
        if message.instructing_agent and message.instructing_agent.bic:
            instg_agt = self._add_element(grp_hdr, "InstgAgt")
            fin_instn_id = self._add_element(instg_agt, "FinInstnId")
            self._add_element(fin_instn_id, "BICFI", message.instructing_agent.bic)

        if message.instructed_agent and message.instructed_agent.bic:
            instd_agt = self._add_element(grp_hdr, "InstdAgt")
            fin_instn_id = self._add_element(instd_agt, "FinInstnId")
            self._add_element(fin_instn_id, "BICFI", message.instructed_agent.bic)

        # Credit Transfers
        for txn in message.credit_transfers:
            cdt_trf_tx_inf = self._add_element(fi_to_fi, "CdtTrfTxInf")
            self._build_transfer_from_object(cdt_trf_tx_inf, txn)

        return self._to_string(root)

    def _build_transaction(
        self,
        parent: ET.Element,
        txn: Dict[str, Any],
        message_id: str,
        idx: int,
    ):
        """Build credit transfer transaction element from dict."""
        # Payment ID
        pmt_id = self._add_element(parent, "PmtId")
        self._add_element(pmt_id, "InstrId", txn.get("instruction_id", f"{message_id}-{idx+1:04d}"))
        self._add_element(pmt_id, "EndToEndId", txn.get("end_to_end_id", f"E2E{idx+1:08d}"))
        self._add_element(pmt_id, "TxId", txn.get("transaction_id", str(uuid.uuid4())))

        if txn.get("uetr"):
            self._add_element(pmt_id, "UETR", txn["uetr"])

        # Interbank Settlement Amount
        amt = self._add_element(parent, "IntrBkSttlmAmt", f"{Decimal(str(txn.get('amount', 0))):.2f}")
        amt.set("Ccy", txn.get("currency", "USD"))

        # Charge Bearer
        self._add_element(parent, "ChrgBr", txn.get("charge_bearer", "SLEV"))

        # Debtor Agent
        if txn.get("debtor_agent_bic"):
            dbtr_agt = self._add_element(parent, "DbtrAgt")
            fin_instn_id = self._add_element(dbtr_agt, "FinInstnId")
            self._add_element(fin_instn_id, "BICFI", txn["debtor_agent_bic"])

        # Debtor
        if txn.get("debtor_name"):
            dbtr = self._add_element(parent, "Dbtr")
            self._add_element(dbtr, "Nm", txn["debtor_name"])

        # Debtor Account
        if txn.get("debtor_account"):
            dbtr_acct = self._add_element(parent, "DbtrAcct")
            acct_id = self._add_element(dbtr_acct, "Id")
            self._add_element(acct_id, "IBAN", txn["debtor_account"])

        # Creditor Agent
        if txn.get("creditor_agent_bic"):
            cdtr_agt = self._add_element(parent, "CdtrAgt")
            fin_instn_id = self._add_element(cdtr_agt, "FinInstnId")
            self._add_element(fin_instn_id, "BICFI", txn["creditor_agent_bic"])

        # Creditor
        if txn.get("creditor_name"):
            cdtr = self._add_element(parent, "Cdtr")
            self._add_element(cdtr, "Nm", txn["creditor_name"])

        # Creditor Account
        if txn.get("creditor_account"):
            cdtr_acct = self._add_element(parent, "CdtrAcct")
            acct_id = self._add_element(cdtr_acct, "Id")
            self._add_element(acct_id, "IBAN", txn["creditor_account"])

        # Remittance Information
        if txn.get("remittance_info"):
            rmt_inf = self._add_element(parent, "RmtInf")
            self._add_element(rmt_inf, "Ustrd", txn["remittance_info"])

    def _build_transfer_from_object(self, parent: ET.Element, txn: FIToFICreditTransfer):
        """Build credit transfer transaction element from object."""
        # Payment ID
        pmt_id = self._add_element(parent, "PmtId")
        if txn.instruction_id:
            self._add_element(pmt_id, "InstrId", txn.instruction_id)
        self._add_element(pmt_id, "EndToEndId", txn.end_to_end_id)
        if txn.transaction_id:
            self._add_element(pmt_id, "TxId", txn.transaction_id)
        if txn.uetr:
            self._add_element(pmt_id, "UETR", txn.uetr)

        # Interbank Settlement Amount
        if txn.interbank_settlement_amount:
            amt = self._add_element(parent, "IntrBkSttlmAmt", txn.interbank_settlement_amount.to_xml_value())
            amt.set("Ccy", txn.interbank_settlement_amount.currency)

        # Settlement Date
        if txn.interbank_settlement_date:
            self._add_element(parent, "IntrBkSttlmDt", txn.interbank_settlement_date.isoformat())

        # Charge Bearer
        self._add_element(parent, "ChrgBr", txn.charge_bearer)

        # Agents and parties...
        if txn.debtor_agent and txn.debtor_agent.bic:
            dbtr_agt = self._add_element(parent, "DbtrAgt")
            fin_instn_id = self._add_element(dbtr_agt, "FinInstnId")
            self._add_element(fin_instn_id, "BICFI", txn.debtor_agent.bic)

        if txn.debtor and txn.debtor.name:
            dbtr = self._add_element(parent, "Dbtr")
            self._add_element(dbtr, "Nm", txn.debtor.name)

        if txn.debtor_account and txn.debtor_account.iban:
            dbtr_acct = self._add_element(parent, "DbtrAcct")
            acct_id = self._add_element(dbtr_acct, "Id")
            self._add_element(acct_id, "IBAN", txn.debtor_account.iban)

        if txn.creditor_agent and txn.creditor_agent.bic:
            cdtr_agt = self._add_element(parent, "CdtrAgt")
            fin_instn_id = self._add_element(cdtr_agt, "FinInstnId")
            self._add_element(fin_instn_id, "BICFI", txn.creditor_agent.bic)

        if txn.creditor and txn.creditor.name:
            cdtr = self._add_element(parent, "Cdtr")
            self._add_element(cdtr, "Nm", txn.creditor.name)

        if txn.creditor_account and txn.creditor_account.iban:
            cdtr_acct = self._add_element(parent, "CdtrAcct")
            acct_id = self._add_element(cdtr_acct, "Id")
            self._add_element(acct_id, "IBAN", txn.creditor_account.iban)

        if txn.remittance_information:
            rmt_inf = self._add_element(parent, "RmtInf")
            self._add_element(rmt_inf, "Ustrd", txn.remittance_information)
