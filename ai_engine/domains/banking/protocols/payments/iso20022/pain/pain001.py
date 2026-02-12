"""
ISO 20022 pain.001 - Customer Credit Transfer Initiation

This message is sent by the initiating party to the forwarding agent or
debtor agent. It is used to request movement of funds from the debtor
account to a creditor.

Used by: SWIFT, SEPA, FedNow, domestic payment systems
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET
import logging

from ai_engine.domains.banking.protocols.payments.iso20022.base import (
    ISO20022Message,
    ISO20022MessageType,
    ISO20022Parser,
    ISO20022Builder,
    Amount,
    AccountIdentification,
    FinancialInstitution,
    PartyIdentification,
)

logger = logging.getLogger(__name__)


@dataclass
class CreditTransferTransaction:
    """Individual credit transfer transaction."""

    # Transaction identification
    instruction_id: str = ""  # Unique instruction ID
    end_to_end_id: str = ""  # End-to-end reference
    uetr: Optional[str] = None  # Unique End-to-end Transaction Reference (SWIFT gpi)

    # Amount
    amount: Optional[Amount] = None

    # Creditor information
    creditor_name: Optional[str] = None
    creditor_account: Optional[AccountIdentification] = None
    creditor_agent: Optional[FinancialInstitution] = None
    creditor_address: Optional[Dict[str, str]] = None

    # Payment details
    remittance_information: Optional[str] = None
    purpose_code: Optional[str] = None
    category_purpose: Optional[str] = None

    # Charges
    charge_bearer: str = "SLEV"  # DEBT, CRED, SHAR, SLEV

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "instruction_id": self.instruction_id,
            "end_to_end_id": self.end_to_end_id,
            "uetr": self.uetr,
            "amount": {
                "value": str(self.amount.value) if self.amount else None,
                "currency": self.amount.currency if self.amount else None,
            },
            "creditor_name": self.creditor_name,
            "creditor_account": self.creditor_account.to_dict() if self.creditor_account else None,
            "remittance_information": self.remittance_information,
            "charge_bearer": self.charge_bearer,
        }


@dataclass
class PaymentInstruction:
    """Payment instruction containing multiple transactions."""

    # Instruction identification
    payment_info_id: str = ""
    payment_method: str = "TRF"  # TRF (Transfer), CHK (Cheque)
    batch_booking: bool = True

    # Number of transactions
    number_of_transactions: int = 0
    control_sum: Optional[Decimal] = None

    # Requested execution date
    requested_execution_date: Optional[date] = None

    # Debtor information
    debtor_name: Optional[str] = None
    debtor_account: Optional[AccountIdentification] = None
    debtor_agent: Optional[FinancialInstitution] = None

    # Transactions
    transactions: List[CreditTransferTransaction] = field(default_factory=list)

    def calculate_control_sum(self) -> Decimal:
        """Calculate sum of all transaction amounts."""
        total = Decimal("0")
        for txn in self.transactions:
            if txn.amount:
                total += txn.amount.value
        return total

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "payment_info_id": self.payment_info_id,
            "payment_method": self.payment_method,
            "batch_booking": self.batch_booking,
            "number_of_transactions": self.number_of_transactions,
            "control_sum": str(self.control_sum) if self.control_sum else None,
            "requested_execution_date": self.requested_execution_date.isoformat() if self.requested_execution_date else None,
            "debtor_name": self.debtor_name,
            "debtor_account": self.debtor_account.to_dict() if self.debtor_account else None,
            "transactions": [t.to_dict() for t in self.transactions],
        }


@dataclass
class Pain001Message(ISO20022Message):
    """
    ISO 20022 pain.001 - Customer Credit Transfer Initiation

    Structure:
    - Document
      - CstmrCdtTrfInitn (CustomerCreditTransferInitiation)
        - GrpHdr (GroupHeader)
        - PmtInf (PaymentInformation) [1..n]
          - CdtTrfTxInf (CreditTransferTransactionInformation) [1..n]
    """

    message_type: ISO20022MessageType = ISO20022MessageType.PAIN_001

    # Group header
    number_of_transactions: int = 0
    control_sum: Optional[Decimal] = None
    initiating_party: Optional[PartyIdentification] = None

    # Payment instructions
    payment_instructions: List[PaymentInstruction] = field(default_factory=list)

    def validate(self) -> List[str]:
        """Validate pain.001 message."""
        errors = []

        # Validate group header
        if not self.message_id:
            errors.append("Group header message ID is required")

        if not self.initiating_party or not self.initiating_party.name:
            errors.append("Initiating party name is required")

        # Validate payment instructions
        if not self.payment_instructions:
            errors.append("At least one payment instruction is required")

        total_transactions = 0
        total_amount = Decimal("0")

        for idx, pmt_inf in enumerate(self.payment_instructions):
            if not pmt_inf.payment_info_id:
                errors.append(f"Payment instruction {idx}: Payment info ID is required")

            if not pmt_inf.debtor_account:
                errors.append(f"Payment instruction {idx}: Debtor account is required")

            if not pmt_inf.transactions:
                errors.append(f"Payment instruction {idx}: At least one transaction is required")

            for txn_idx, txn in enumerate(pmt_inf.transactions):
                total_transactions += 1

                if not txn.end_to_end_id:
                    errors.append(f"Payment {idx}, Transaction {txn_idx}: End-to-end ID is required")

                if not txn.amount:
                    errors.append(f"Payment {idx}, Transaction {txn_idx}: Amount is required")
                else:
                    if txn.amount.value <= 0:
                        errors.append(f"Payment {idx}, Transaction {txn_idx}: Amount must be positive")
                    total_amount += txn.amount.value

                if not txn.creditor_name:
                    errors.append(f"Payment {idx}, Transaction {txn_idx}: Creditor name is required")

                if not txn.creditor_account:
                    errors.append(f"Payment {idx}, Transaction {txn_idx}: Creditor account is required")

        # Validate totals
        if self.number_of_transactions != total_transactions:
            errors.append(
                f"Number of transactions mismatch: header={self.number_of_transactions}, " f"actual={total_transactions}"
            )

        if self.control_sum and self.control_sum != total_amount:
            errors.append(f"Control sum mismatch: header={self.control_sum}, " f"actual={total_amount}")

        return errors

    def to_xml(self) -> str:
        """Serialize to XML."""
        builder = Pain001Builder()
        return builder.build_from_message(self)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base = super().to_dict()
        base.update(
            {
                "number_of_transactions": self.number_of_transactions,
                "control_sum": str(self.control_sum) if self.control_sum else None,
                "initiating_party": self.initiating_party.to_dict() if self.initiating_party else None,
                "payment_instructions": [pi.to_dict() for pi in self.payment_instructions],
            }
        )
        return base


class Pain001Parser(ISO20022Parser):
    """Parser for pain.001 messages."""

    def parse(self, xml_content: str) -> Pain001Message:
        """Parse pain.001 XML to message object."""
        self.errors = []

        root = self._parse_xml(xml_content)
        message = Pain001Message(raw_xml=xml_content)

        # Detect namespace
        ns = self._detect_namespace(root)

        # Find the main element
        cstmr_cdt_trf_initn = self._find_element(root, "CstmrCdtTrfInitn", ns)
        if cstmr_cdt_trf_initn is None:
            self.errors.append("CstmrCdtTrfInitn element not found")
            return message

        # Parse group header
        grp_hdr = self._find_element(cstmr_cdt_trf_initn, "GrpHdr", ns)
        if grp_hdr is not None:
            self._parse_group_header(grp_hdr, message, ns)

        # Parse payment instructions
        for pmt_inf in self._find_all_elements(cstmr_cdt_trf_initn, "PmtInf", ns):
            payment_instruction = self._parse_payment_instruction(pmt_inf, ns)
            message.payment_instructions.append(payment_instruction)

        return message

    def _detect_namespace(self, root: ET.Element) -> Dict[str, str]:
        """Detect namespace from root element."""
        ns = {}
        tag = root.tag
        if "{" in tag:
            namespace = tag[1 : tag.index("}")]
            ns["ns"] = namespace
        return ns

    def _find_element(
        self,
        parent: ET.Element,
        name: str,
        ns: Dict[str, str],
    ) -> Optional[ET.Element]:
        """Find element with namespace handling."""
        if ns:
            elem = parent.find(f"ns:{name}", ns)
            if elem is not None:
                return elem

        # Try without namespace
        elem = parent.find(name)
        if elem is not None:
            return elem

        # Try with wildcard
        for child in parent:
            local_name = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if local_name == name:
                return child

        return None

    def _find_all_elements(
        self,
        parent: ET.Element,
        name: str,
        ns: Dict[str, str],
    ) -> List[ET.Element]:
        """Find all elements with namespace handling."""
        results = []

        if ns:
            results = parent.findall(f"ns:{name}", ns)
            if results:
                return results

        results = parent.findall(name)
        if results:
            return results

        # Try with wildcard
        for child in parent:
            local_name = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if local_name == name:
                results.append(child)

        return results

    def _parse_group_header(
        self,
        grp_hdr: ET.Element,
        message: Pain001Message,
        ns: Dict[str, str],
    ):
        """Parse group header."""
        # Message ID
        msg_id = self._find_element(grp_hdr, "MsgId", ns)
        message.message_id = self._get_text(msg_id)

        # Creation datetime
        cre_dt_tm = self._find_element(grp_hdr, "CreDtTm", ns)
        message.creation_datetime = self._get_datetime(cre_dt_tm) or datetime.utcnow()

        # Number of transactions
        nb_of_txs = self._find_element(grp_hdr, "NbOfTxs", ns)
        message.number_of_transactions = int(self._get_text(nb_of_txs, "0"))

        # Control sum
        ctrl_sum = self._find_element(grp_hdr, "CtrlSum", ns)
        if ctrl_sum is not None and ctrl_sum.text:
            message.control_sum = Decimal(ctrl_sum.text)

        # Initiating party
        initg_pty = self._find_element(grp_hdr, "InitgPty", ns)
        if initg_pty is not None:
            message.initiating_party = PartyIdentification.from_xml(initg_pty, ns)

    def _parse_payment_instruction(
        self,
        pmt_inf: ET.Element,
        ns: Dict[str, str],
    ) -> PaymentInstruction:
        """Parse payment instruction."""
        instruction = PaymentInstruction()

        # Payment info ID
        pmt_inf_id = self._find_element(pmt_inf, "PmtInfId", ns)
        instruction.payment_info_id = self._get_text(pmt_inf_id)

        # Payment method
        pmt_mtd = self._find_element(pmt_inf, "PmtMtd", ns)
        instruction.payment_method = self._get_text(pmt_mtd, "TRF")

        # Batch booking
        btch_bookg = self._find_element(pmt_inf, "BtchBookg", ns)
        instruction.batch_booking = self._get_text(btch_bookg, "true").lower() == "true"

        # Number of transactions
        nb_of_txs = self._find_element(pmt_inf, "NbOfTxs", ns)
        instruction.number_of_transactions = int(self._get_text(nb_of_txs, "0"))

        # Control sum
        ctrl_sum = self._find_element(pmt_inf, "CtrlSum", ns)
        if ctrl_sum is not None and ctrl_sum.text:
            instruction.control_sum = Decimal(ctrl_sum.text)

        # Requested execution date
        reqd_exctn_dt = self._find_element(pmt_inf, "ReqdExctnDt", ns)
        if reqd_exctn_dt is not None:
            # Try to find Dt sub-element first (ISO 20022 2019)
            dt = self._find_element(reqd_exctn_dt, "Dt", ns)
            date_text = self._get_text(dt) if dt is not None else self._get_text(reqd_exctn_dt)
            if date_text:
                try:
                    instruction.requested_execution_date = date.fromisoformat(date_text)
                except ValueError:
                    pass

        # Debtor
        dbtr = self._find_element(pmt_inf, "Dbtr", ns)
        if dbtr is not None:
            name_elem = self._find_element(dbtr, "Nm", ns)
            instruction.debtor_name = self._get_text(name_elem)

        # Debtor account
        dbtr_acct = self._find_element(pmt_inf, "DbtrAcct", ns)
        if dbtr_acct is not None:
            acct_id = self._find_element(dbtr_acct, "Id", ns)
            if acct_id is not None:
                instruction.debtor_account = AccountIdentification.from_xml(acct_id, ns)

        # Debtor agent
        dbtr_agt = self._find_element(pmt_inf, "DbtrAgt", ns)
        if dbtr_agt is not None:
            fin_instn_id = self._find_element(dbtr_agt, "FinInstnId", ns)
            if fin_instn_id is not None:
                instruction.debtor_agent = FinancialInstitution.from_xml(fin_instn_id, ns)

        # Parse transactions
        for cdt_trf_tx_inf in self._find_all_elements(pmt_inf, "CdtTrfTxInf", ns):
            transaction = self._parse_transaction(cdt_trf_tx_inf, ns)
            instruction.transactions.append(transaction)

        instruction.number_of_transactions = len(instruction.transactions)

        return instruction

    def _parse_transaction(
        self,
        cdt_trf_tx_inf: ET.Element,
        ns: Dict[str, str],
    ) -> CreditTransferTransaction:
        """Parse credit transfer transaction."""
        transaction = CreditTransferTransaction()

        # Payment ID
        pmt_id = self._find_element(cdt_trf_tx_inf, "PmtId", ns)
        if pmt_id is not None:
            instr_id = self._find_element(pmt_id, "InstrId", ns)
            transaction.instruction_id = self._get_text(instr_id)

            end_to_end_id = self._find_element(pmt_id, "EndToEndId", ns)
            transaction.end_to_end_id = self._get_text(end_to_end_id)

            uetr = self._find_element(pmt_id, "UETR", ns)
            transaction.uetr = self._get_text(uetr) or None

        # Amount
        amt = self._find_element(cdt_trf_tx_inf, "Amt", ns)
        if amt is not None:
            instd_amt = self._find_element(amt, "InstdAmt", ns)
            if instd_amt is not None:
                transaction.amount = Amount.from_xml(instd_amt)

        # Charge bearer
        chrg_br = self._find_element(cdt_trf_tx_inf, "ChrgBr", ns)
        transaction.charge_bearer = self._get_text(chrg_br, "SLEV")

        # Creditor agent
        cdtr_agt = self._find_element(cdt_trf_tx_inf, "CdtrAgt", ns)
        if cdtr_agt is not None:
            fin_instn_id = self._find_element(cdtr_agt, "FinInstnId", ns)
            if fin_instn_id is not None:
                transaction.creditor_agent = FinancialInstitution.from_xml(fin_instn_id, ns)

        # Creditor
        cdtr = self._find_element(cdt_trf_tx_inf, "Cdtr", ns)
        if cdtr is not None:
            name_elem = self._find_element(cdtr, "Nm", ns)
            transaction.creditor_name = self._get_text(name_elem)

        # Creditor account
        cdtr_acct = self._find_element(cdt_trf_tx_inf, "CdtrAcct", ns)
        if cdtr_acct is not None:
            acct_id = self._find_element(cdtr_acct, "Id", ns)
            if acct_id is not None:
                transaction.creditor_account = AccountIdentification.from_xml(acct_id, ns)

        # Remittance information
        rmt_inf = self._find_element(cdt_trf_tx_inf, "RmtInf", ns)
        if rmt_inf is not None:
            ustrd = self._find_element(rmt_inf, "Ustrd", ns)
            transaction.remittance_information = self._get_text(ustrd)

        # Purpose
        purp = self._find_element(cdt_trf_tx_inf, "Purp", ns)
        if purp is not None:
            cd = self._find_element(purp, "Cd", ns)
            transaction.purpose_code = self._get_text(cd)

        return transaction


class Pain001Builder(ISO20022Builder):
    """Builder for pain.001 messages."""

    def __init__(self, version: str = "03"):
        super().__init__(ISO20022MessageType.PAIN_001, version)

    def build(
        self,
        message_id: str,
        initiating_party_name: str,
        debtor_name: str,
        debtor_iban: str,
        debtor_bic: str,
        transactions: List[Dict[str, Any]],
        requested_execution_date: Optional[date] = None,
    ) -> str:
        """
        Build pain.001 message.

        Args:
            message_id: Unique message identifier
            initiating_party_name: Name of the initiating party
            debtor_name: Name of the debtor
            debtor_iban: Debtor's IBAN
            debtor_bic: Debtor's bank BIC
            transactions: List of transaction dicts with keys:
                - end_to_end_id: str
                - amount: float
                - currency: str
                - creditor_name: str
                - creditor_iban: str
                - creditor_bic: str (optional)
                - remittance_info: str (optional)
            requested_execution_date: Requested execution date

        Returns:
            XML string
        """
        # Create document
        root = self._create_root()

        # Customer Credit Transfer Initiation
        cstmr_cdt_trf_initn = self._add_element(root, "CstmrCdtTrfInitn")

        # Group Header
        grp_hdr = self._add_element(cstmr_cdt_trf_initn, "GrpHdr")
        self._add_element(grp_hdr, "MsgId", message_id)
        self._add_element(grp_hdr, "CreDtTm", self._format_datetime(datetime.utcnow()))
        self._add_element(grp_hdr, "NbOfTxs", str(len(transactions)))

        # Calculate control sum
        control_sum = sum(Decimal(str(t["amount"])) for t in transactions)
        self._add_element(grp_hdr, "CtrlSum", f"{control_sum:.2f}")

        # Initiating Party
        initg_pty = self._add_element(grp_hdr, "InitgPty")
        self._add_element(initg_pty, "Nm", initiating_party_name)

        # Payment Information
        pmt_inf = self._add_element(cstmr_cdt_trf_initn, "PmtInf")
        self._add_element(pmt_inf, "PmtInfId", f"{message_id}-001")
        self._add_element(pmt_inf, "PmtMtd", "TRF")
        self._add_element(pmt_inf, "BtchBookg", "true")
        self._add_element(pmt_inf, "NbOfTxs", str(len(transactions)))
        self._add_element(pmt_inf, "CtrlSum", f"{control_sum:.2f}")

        # Payment Type Information
        pmt_tp_inf = self._add_element(pmt_inf, "PmtTpInf")
        svc_lvl = self._add_element(pmt_tp_inf, "SvcLvl")
        self._add_element(svc_lvl, "Cd", "SEPA")

        # Requested Execution Date
        reqd_exctn_dt = self._add_element(pmt_inf, "ReqdExctnDt")
        exec_date = requested_execution_date or date.today()
        self._add_element(reqd_exctn_dt, "Dt", exec_date.isoformat())

        # Debtor
        dbtr = self._add_element(pmt_inf, "Dbtr")
        self._add_element(dbtr, "Nm", debtor_name)

        # Debtor Account
        dbtr_acct = self._add_element(pmt_inf, "DbtrAcct")
        dbtr_acct_id = self._add_element(dbtr_acct, "Id")
        self._add_element(dbtr_acct_id, "IBAN", debtor_iban)

        # Debtor Agent
        dbtr_agt = self._add_element(pmt_inf, "DbtrAgt")
        dbtr_fin_instn_id = self._add_element(dbtr_agt, "FinInstnId")
        self._add_element(dbtr_fin_instn_id, "BICFI", debtor_bic)

        # Credit Transfer Transaction Information
        for idx, txn in enumerate(transactions):
            cdt_trf_tx_inf = self._add_element(pmt_inf, "CdtTrfTxInf")

            # Payment ID
            pmt_id = self._add_element(cdt_trf_tx_inf, "PmtId")
            self._add_element(pmt_id, "InstrId", f"{message_id}-{idx+1:04d}")
            self._add_element(pmt_id, "EndToEndId", txn["end_to_end_id"])

            # Amount
            amt = self._add_element(cdt_trf_tx_inf, "Amt")
            instd_amt = self._add_element(amt, "InstdAmt", f"{Decimal(str(txn['amount'])):.2f}")
            instd_amt.set("Ccy", txn.get("currency", "EUR"))

            # Charge Bearer
            self._add_element(cdt_trf_tx_inf, "ChrgBr", "SLEV")

            # Creditor Agent (if provided)
            if txn.get("creditor_bic"):
                cdtr_agt = self._add_element(cdt_trf_tx_inf, "CdtrAgt")
                cdtr_fin_instn_id = self._add_element(cdtr_agt, "FinInstnId")
                self._add_element(cdtr_fin_instn_id, "BICFI", txn["creditor_bic"])

            # Creditor
            cdtr = self._add_element(cdt_trf_tx_inf, "Cdtr")
            self._add_element(cdtr, "Nm", txn["creditor_name"])

            # Creditor Account
            cdtr_acct = self._add_element(cdt_trf_tx_inf, "CdtrAcct")
            cdtr_acct_id = self._add_element(cdtr_acct, "Id")
            self._add_element(cdtr_acct_id, "IBAN", txn["creditor_iban"])

            # Remittance Information (if provided)
            if txn.get("remittance_info"):
                rmt_inf = self._add_element(cdt_trf_tx_inf, "RmtInf")
                self._add_element(rmt_inf, "Ustrd", txn["remittance_info"])

        return self._to_string(root)

    def build_from_message(self, message: Pain001Message) -> str:
        """Build XML from Pain001Message object."""
        # Create document
        root = self._create_root()

        # Customer Credit Transfer Initiation
        cstmr_cdt_trf_initn = self._add_element(root, "CstmrCdtTrfInitn")

        # Group Header
        grp_hdr = self._add_element(cstmr_cdt_trf_initn, "GrpHdr")
        self._add_element(grp_hdr, "MsgId", message.message_id)
        self._add_element(grp_hdr, "CreDtTm", self._format_datetime(message.creation_datetime))
        self._add_element(grp_hdr, "NbOfTxs", str(message.number_of_transactions))

        if message.control_sum:
            self._add_element(grp_hdr, "CtrlSum", f"{message.control_sum:.2f}")

        # Initiating Party
        if message.initiating_party:
            initg_pty = self._add_element(grp_hdr, "InitgPty")
            if message.initiating_party.name:
                self._add_element(initg_pty, "Nm", message.initiating_party.name)

        # Payment Instructions
        for pmt_instr in message.payment_instructions:
            pmt_inf = self._add_element(cstmr_cdt_trf_initn, "PmtInf")
            self._build_payment_instruction(pmt_inf, pmt_instr)

        return self._to_string(root)

    def _build_payment_instruction(
        self,
        pmt_inf: ET.Element,
        instruction: PaymentInstruction,
    ):
        """Build payment instruction element."""
        self._add_element(pmt_inf, "PmtInfId", instruction.payment_info_id)
        self._add_element(pmt_inf, "PmtMtd", instruction.payment_method)
        self._add_element(pmt_inf, "BtchBookg", "true" if instruction.batch_booking else "false")
        self._add_element(pmt_inf, "NbOfTxs", str(len(instruction.transactions)))

        if instruction.control_sum:
            self._add_element(pmt_inf, "CtrlSum", f"{instruction.control_sum:.2f}")

        # Requested Execution Date
        if instruction.requested_execution_date:
            reqd_exctn_dt = self._add_element(pmt_inf, "ReqdExctnDt")
            self._add_element(reqd_exctn_dt, "Dt", instruction.requested_execution_date.isoformat())

        # Debtor
        if instruction.debtor_name:
            dbtr = self._add_element(pmt_inf, "Dbtr")
            self._add_element(dbtr, "Nm", instruction.debtor_name)

        # Debtor Account
        if instruction.debtor_account:
            dbtr_acct = self._add_element(pmt_inf, "DbtrAcct")
            dbtr_acct_id = self._add_element(dbtr_acct, "Id")
            if instruction.debtor_account.iban:
                self._add_element(dbtr_acct_id, "IBAN", instruction.debtor_account.iban)

        # Debtor Agent
        if instruction.debtor_agent:
            dbtr_agt = self._add_element(pmt_inf, "DbtrAgt")
            fin_instn_id = self._add_element(dbtr_agt, "FinInstnId")
            if instruction.debtor_agent.bic:
                self._add_element(fin_instn_id, "BICFI", instruction.debtor_agent.bic)

        # Transactions
        for txn in instruction.transactions:
            cdt_trf_tx_inf = self._add_element(pmt_inf, "CdtTrfTxInf")
            self._build_transaction(cdt_trf_tx_inf, txn)

    def _build_transaction(
        self,
        cdt_trf_tx_inf: ET.Element,
        txn: CreditTransferTransaction,
    ):
        """Build credit transfer transaction element."""
        # Payment ID
        pmt_id = self._add_element(cdt_trf_tx_inf, "PmtId")
        if txn.instruction_id:
            self._add_element(pmt_id, "InstrId", txn.instruction_id)
        self._add_element(pmt_id, "EndToEndId", txn.end_to_end_id)
        if txn.uetr:
            self._add_element(pmt_id, "UETR", txn.uetr)

        # Amount
        if txn.amount:
            amt = self._add_element(cdt_trf_tx_inf, "Amt")
            instd_amt = self._add_element(amt, "InstdAmt", txn.amount.to_xml_value())
            instd_amt.set("Ccy", txn.amount.currency)

        # Charge Bearer
        self._add_element(cdt_trf_tx_inf, "ChrgBr", txn.charge_bearer)

        # Creditor Agent
        if txn.creditor_agent and txn.creditor_agent.bic:
            cdtr_agt = self._add_element(cdt_trf_tx_inf, "CdtrAgt")
            fin_instn_id = self._add_element(cdtr_agt, "FinInstnId")
            self._add_element(fin_instn_id, "BICFI", txn.creditor_agent.bic)

        # Creditor
        if txn.creditor_name:
            cdtr = self._add_element(cdt_trf_tx_inf, "Cdtr")
            self._add_element(cdtr, "Nm", txn.creditor_name)

        # Creditor Account
        if txn.creditor_account:
            cdtr_acct = self._add_element(cdt_trf_tx_inf, "CdtrAcct")
            cdtr_acct_id = self._add_element(cdtr_acct, "Id")
            if txn.creditor_account.iban:
                self._add_element(cdtr_acct_id, "IBAN", txn.creditor_account.iban)

        # Remittance Information
        if txn.remittance_information:
            rmt_inf = self._add_element(cdt_trf_tx_inf, "RmtInf")
            self._add_element(rmt_inf, "Ustrd", txn.remittance_information)

        # Purpose
        if txn.purpose_code:
            purp = self._add_element(cdt_trf_tx_inf, "Purp")
            self._add_element(purp, "Cd", txn.purpose_code)
