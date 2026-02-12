"""
ISO 20022 camt.053 - Bank to Customer Statement

This message is sent by the account servicer to an account owner or
to a party authorised by the account owner to receive the message.
It is used to inform the account owner of the entries reported on their account.

Used by: All banks for account statements, treasury systems
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET
import logging

from ai_engine.domains.banking.protocols.payments.iso20022.base import (
    ISO20022Message,
    ISO20022MessageType,
    ISO20022Parser,
    Amount,
    AccountIdentification,
    FinancialInstitution,
    PartyIdentification,
)

logger = logging.getLogger(__name__)


class BalanceType(Enum):
    """Balance type codes."""

    OPENING_BOOKED = ("OPBD", "Opening Booked")
    CLOSING_BOOKED = ("CLBD", "Closing Booked")
    OPENING_AVAILABLE = ("OPAV", "Opening Available")
    CLOSING_AVAILABLE = ("CLAV", "Closing Available")
    FORWARD_AVAILABLE = ("FWAV", "Forward Available")
    INTERIM_BOOKED = ("ITBD", "Interim Booked")
    INTERIM_AVAILABLE = ("ITAV", "Interim Available")

    def __init__(self, code: str, description: str):
        self.type_code = code
        self.description = description

    @classmethod
    def from_code(cls, code: str) -> Optional["BalanceType"]:
        for bt in cls:
            if bt.type_code == code:
                return bt
        return None


class CreditDebitIndicator(Enum):
    """Credit/Debit indicator."""

    CREDIT = "CRDT"
    DEBIT = "DBIT"


class EntryStatus(Enum):
    """Entry status codes."""

    BOOKED = "BOOK"
    PENDING = "PDNG"
    INFORMATION = "INFO"


@dataclass
class Balance:
    """Account balance."""

    balance_type: BalanceType
    amount: Amount
    credit_debit: CreditDebitIndicator
    date: date

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.balance_type.type_code,
            "amount": str(self.amount.value),
            "currency": self.amount.currency,
            "credit_debit": self.credit_debit.value,
            "date": self.date.isoformat(),
        }


@dataclass
class TransactionDetails:
    """Transaction details within an entry."""

    # References
    instruction_id: Optional[str] = None
    end_to_end_id: Optional[str] = None
    transaction_id: Optional[str] = None
    mandate_id: Optional[str] = None

    # Amount
    amount: Optional[Amount] = None
    credit_debit: Optional[CreditDebitIndicator] = None

    # Parties
    debtor: Optional[PartyIdentification] = None
    debtor_account: Optional[AccountIdentification] = None
    creditor: Optional[PartyIdentification] = None
    creditor_account: Optional[AccountIdentification] = None

    # Agent
    debtor_agent: Optional[FinancialInstitution] = None
    creditor_agent: Optional[FinancialInstitution] = None

    # Remittance
    remittance_information: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instruction_id": self.instruction_id,
            "end_to_end_id": self.end_to_end_id,
            "transaction_id": self.transaction_id,
            "amount": str(self.amount.value) if self.amount else None,
            "currency": self.amount.currency if self.amount else None,
            "credit_debit": self.credit_debit.value if self.credit_debit else None,
            "debtor_name": self.debtor.name if self.debtor else None,
            "creditor_name": self.creditor.name if self.creditor else None,
            "remittance_information": self.remittance_information,
        }


@dataclass
class StatementEntry:
    """Statement entry (transaction)."""

    # Entry reference
    entry_reference: Optional[str] = None
    account_servicer_reference: Optional[str] = None

    # Amount
    amount: Optional[Amount] = None
    credit_debit: CreditDebitIndicator = CreditDebitIndicator.CREDIT

    # Status and dates
    status: EntryStatus = EntryStatus.BOOKED
    booking_date: Optional[date] = None
    value_date: Optional[date] = None

    # Bank transaction code
    domain_code: Optional[str] = None
    family_code: Optional[str] = None
    sub_family_code: Optional[str] = None

    # Details
    transaction_details: List[TransactionDetails] = field(default_factory=list)

    # Additional info
    additional_info: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_reference": self.entry_reference,
            "amount": str(self.amount.value) if self.amount else None,
            "currency": self.amount.currency if self.amount else None,
            "credit_debit": self.credit_debit.value,
            "status": self.status.value,
            "booking_date": self.booking_date.isoformat() if self.booking_date else None,
            "value_date": self.value_date.isoformat() if self.value_date else None,
            "domain_code": self.domain_code,
            "family_code": self.family_code,
            "transaction_details": [td.to_dict() for td in self.transaction_details],
        }


@dataclass
class BankStatement:
    """Bank statement for an account."""

    # Statement identification
    statement_id: str = ""
    sequence_number: Optional[int] = None
    creation_datetime: datetime = field(default_factory=datetime.utcnow)
    from_date: Optional[date] = None
    to_date: Optional[date] = None

    # Account
    account: Optional[AccountIdentification] = None
    account_owner: Optional[PartyIdentification] = None
    account_servicer: Optional[FinancialInstitution] = None

    # Balances
    balances: List[Balance] = field(default_factory=list)

    # Transaction summary
    number_of_entries: int = 0
    sum_of_credits: Optional[Amount] = None
    number_of_credits: int = 0
    sum_of_debits: Optional[Amount] = None
    number_of_debits: int = 0

    # Entries
    entries: List[StatementEntry] = field(default_factory=list)

    def get_opening_balance(self) -> Optional[Balance]:
        """Get opening booked balance."""
        for bal in self.balances:
            if bal.balance_type == BalanceType.OPENING_BOOKED:
                return bal
        return None

    def get_closing_balance(self) -> Optional[Balance]:
        """Get closing booked balance."""
        for bal in self.balances:
            if bal.balance_type == BalanceType.CLOSING_BOOKED:
                return bal
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statement_id": self.statement_id,
            "sequence_number": self.sequence_number,
            "creation_datetime": self.creation_datetime.isoformat(),
            "from_date": self.from_date.isoformat() if self.from_date else None,
            "to_date": self.to_date.isoformat() if self.to_date else None,
            "account_iban": self.account.iban if self.account else None,
            "number_of_entries": self.number_of_entries,
            "number_of_credits": self.number_of_credits,
            "number_of_debits": self.number_of_debits,
            "balances": [b.to_dict() for b in self.balances],
            "entries": [e.to_dict() for e in self.entries],
        }


@dataclass
class Camt053Message(ISO20022Message):
    """
    ISO 20022 camt.053 - Bank to Customer Statement

    Structure:
    - Document
      - BkToCstmrStmt (BankToCustomerStatement)
        - GrpHdr (GroupHeader)
        - Stmt (Statement) [1..n]
          - Bal (Balance) [1..n]
          - Ntry (Entry) [0..n]
    """

    message_type: ISO20022MessageType = ISO20022MessageType.CAMT_053

    # Statements
    statements: List[BankStatement] = field(default_factory=list)

    def validate(self) -> List[str]:
        """Validate camt.053 message."""
        errors = []

        if not self.message_id:
            errors.append("Message ID is required")

        if not self.statements:
            errors.append("At least one statement is required")

        for idx, stmt in enumerate(self.statements):
            if not stmt.statement_id:
                errors.append(f"Statement {idx}: Statement ID is required")

            if not stmt.account:
                errors.append(f"Statement {idx}: Account is required")

            # Must have at least opening and closing balance
            has_opening = any(b.balance_type == BalanceType.OPENING_BOOKED for b in stmt.balances)
            has_closing = any(b.balance_type == BalanceType.CLOSING_BOOKED for b in stmt.balances)

            if not has_opening:
                errors.append(f"Statement {idx}: Opening balance is required")
            if not has_closing:
                errors.append(f"Statement {idx}: Closing balance is required")

        return errors

    def to_xml(self) -> str:
        """Serialize to XML (not implemented - read-only message type)."""
        raise NotImplementedError("camt.053 is typically a bank-generated message")

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "statements": [s.to_dict() for s in self.statements],
            }
        )
        return base


class Camt053Parser(ISO20022Parser):
    """Parser for camt.053 messages."""

    def parse(self, xml_content: str) -> Camt053Message:
        """Parse camt.053 XML to message object."""
        self.errors = []

        root = self._parse_xml(xml_content)
        message = Camt053Message(raw_xml=xml_content)

        # Detect namespace
        ns = self._detect_namespace(root)

        # Find main element
        bk_to_cstmr_stmt = self._find_element(root, "BkToCstmrStmt", ns)
        if bk_to_cstmr_stmt is None:
            self.errors.append("BkToCstmrStmt element not found")
            return message

        # Parse group header
        grp_hdr = self._find_element(bk_to_cstmr_stmt, "GrpHdr", ns)
        if grp_hdr is not None:
            self._parse_group_header(grp_hdr, message, ns)

        # Parse statements
        for stmt in self._find_all_elements(bk_to_cstmr_stmt, "Stmt", ns):
            statement = self._parse_statement(stmt, ns)
            message.statements.append(statement)

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

    def _parse_group_header(self, grp_hdr: ET.Element, message: Camt053Message, ns: Dict[str, str]):
        """Parse group header."""
        msg_id = self._find_element(grp_hdr, "MsgId", ns)
        message.message_id = self._get_text(msg_id)

        cre_dt_tm = self._find_element(grp_hdr, "CreDtTm", ns)
        message.creation_datetime = self._get_datetime(cre_dt_tm) or datetime.utcnow()

    def _parse_statement(self, stmt: ET.Element, ns: Dict[str, str]) -> BankStatement:
        """Parse statement."""
        statement = BankStatement()

        # Statement ID
        stmt_id = self._find_element(stmt, "Id", ns)
        statement.statement_id = self._get_text(stmt_id)

        # Sequence number
        seq_nb = self._find_element(stmt, "ElctrncSeqNb", ns)
        if seq_nb is not None and seq_nb.text:
            try:
                statement.sequence_number = int(seq_nb.text)
            except ValueError:
                pass

        # Creation datetime
        cre_dt_tm = self._find_element(stmt, "CreDtTm", ns)
        statement.creation_datetime = self._get_datetime(cre_dt_tm) or datetime.utcnow()

        # From/To dates
        fr_to_dt = self._find_element(stmt, "FrToDt", ns)
        if fr_to_dt is not None:
            fr_dt = self._find_element(fr_to_dt, "FrDtTm", ns)
            if fr_dt is not None and fr_dt.text:
                try:
                    statement.from_date = date.fromisoformat(fr_dt.text[:10])
                except ValueError:
                    pass

            to_dt = self._find_element(fr_to_dt, "ToDtTm", ns)
            if to_dt is not None and to_dt.text:
                try:
                    statement.to_date = date.fromisoformat(to_dt.text[:10])
                except ValueError:
                    pass

        # Account
        acct = self._find_element(stmt, "Acct", ns)
        if acct is not None:
            acct_id = self._find_element(acct, "Id", ns)
            if acct_id is not None:
                statement.account = AccountIdentification.from_xml(acct_id, ns)

            # Account owner
            ownr = self._find_element(acct, "Ownr", ns)
            if ownr is not None:
                statement.account_owner = PartyIdentification.from_xml(ownr, ns)

            # Account servicer
            svcr = self._find_element(acct, "Svcr", ns)
            if svcr is not None:
                fin_instn_id = self._find_element(svcr, "FinInstnId", ns)
                if fin_instn_id is not None:
                    statement.account_servicer = FinancialInstitution.from_xml(fin_instn_id, ns)

        # Balances
        for bal in self._find_all_elements(stmt, "Bal", ns):
            balance = self._parse_balance(bal, ns)
            if balance:
                statement.balances.append(balance)

        # Transaction summary
        txs_sumry = self._find_element(stmt, "TxsSummry", ns)
        if txs_sumry is not None:
            ttl_ntries = self._find_element(txs_sumry, "TtlNtries", ns)
            if ttl_ntries is not None:
                nb_of_ntries = self._find_element(ttl_ntries, "NbOfNtries", ns)
                if nb_of_ntries is not None and nb_of_ntries.text:
                    statement.number_of_entries = int(nb_of_ntries.text)

            # Credits
            ttl_cdt_ntries = self._find_element(txs_sumry, "TtlCdtNtries", ns)
            if ttl_cdt_ntries is not None:
                nb = self._find_element(ttl_cdt_ntries, "NbOfNtries", ns)
                if nb is not None and nb.text:
                    statement.number_of_credits = int(nb.text)

            # Debits
            ttl_dbt_ntries = self._find_element(txs_sumry, "TtlDbtNtries", ns)
            if ttl_dbt_ntries is not None:
                nb = self._find_element(ttl_dbt_ntries, "NbOfNtries", ns)
                if nb is not None and nb.text:
                    statement.number_of_debits = int(nb.text)

        # Entries
        for ntry in self._find_all_elements(stmt, "Ntry", ns):
            entry = self._parse_entry(ntry, ns)
            statement.entries.append(entry)

        statement.number_of_entries = len(statement.entries)

        return statement

    def _parse_balance(self, bal: ET.Element, ns: Dict[str, str]) -> Optional[Balance]:
        """Parse balance."""
        # Type
        tp = self._find_element(bal, "Tp", ns)
        if tp is None:
            return None

        cd_or_prtry = self._find_element(tp, "CdOrPrtry", ns)
        if cd_or_prtry is None:
            return None

        cd = self._find_element(cd_or_prtry, "Cd", ns)
        if cd is None or not cd.text:
            return None

        balance_type = BalanceType.from_code(cd.text)
        if balance_type is None:
            return None

        # Amount
        amt = self._find_element(bal, "Amt", ns)
        if amt is None:
            return None
        amount = Amount.from_xml(amt)

        # Credit/Debit
        cdt_dbt_ind = self._find_element(bal, "CdtDbtInd", ns)
        cdt_dbt_text = self._get_text(cdt_dbt_ind, "CRDT")
        credit_debit = CreditDebitIndicator.CREDIT if cdt_dbt_text == "CRDT" else CreditDebitIndicator.DEBIT

        # Date
        dt = self._find_element(bal, "Dt", ns)
        balance_date = date.today()
        if dt is not None:
            dt_elem = self._find_element(dt, "Dt", ns)
            if dt_elem is not None and dt_elem.text:
                try:
                    balance_date = date.fromisoformat(dt_elem.text)
                except ValueError:
                    pass

        return Balance(
            balance_type=balance_type,
            amount=amount,
            credit_debit=credit_debit,
            date=balance_date,
        )

    def _parse_entry(self, ntry: ET.Element, ns: Dict[str, str]) -> StatementEntry:
        """Parse statement entry."""
        entry = StatementEntry()

        # Entry reference
        ntry_ref = self._find_element(ntry, "NtryRef", ns)
        entry.entry_reference = self._get_text(ntry_ref)

        # Account servicer reference
        acct_svcr_ref = self._find_element(ntry, "AcctSvcrRef", ns)
        entry.account_servicer_reference = self._get_text(acct_svcr_ref)

        # Amount
        amt = self._find_element(ntry, "Amt", ns)
        if amt is not None:
            entry.amount = Amount.from_xml(amt)

        # Credit/Debit
        cdt_dbt_ind = self._find_element(ntry, "CdtDbtInd", ns)
        cdt_dbt_text = self._get_text(cdt_dbt_ind, "CRDT")
        entry.credit_debit = CreditDebitIndicator.CREDIT if cdt_dbt_text == "CRDT" else CreditDebitIndicator.DEBIT

        # Status
        sts = self._find_element(ntry, "Sts", ns)
        sts_text = self._get_text(sts, "BOOK")
        entry.status = EntryStatus.BOOKED if sts_text == "BOOK" else EntryStatus.PENDING

        # Booking date
        bookg_dt = self._find_element(ntry, "BookgDt", ns)
        if bookg_dt is not None:
            dt = self._find_element(bookg_dt, "Dt", ns)
            if dt is not None and dt.text:
                try:
                    entry.booking_date = date.fromisoformat(dt.text)
                except ValueError:
                    pass

        # Value date
        val_dt = self._find_element(ntry, "ValDt", ns)
        if val_dt is not None:
            dt = self._find_element(val_dt, "Dt", ns)
            if dt is not None and dt.text:
                try:
                    entry.value_date = date.fromisoformat(dt.text)
                except ValueError:
                    pass

        # Bank transaction code
        bk_tx_cd = self._find_element(ntry, "BkTxCd", ns)
        if bk_tx_cd is not None:
            domn = self._find_element(bk_tx_cd, "Domn", ns)
            if domn is not None:
                cd = self._find_element(domn, "Cd", ns)
                entry.domain_code = self._get_text(cd)

                fmly = self._find_element(domn, "Fmly", ns)
                if fmly is not None:
                    cd = self._find_element(fmly, "Cd", ns)
                    entry.family_code = self._get_text(cd)

                    sub_fmly_cd = self._find_element(fmly, "SubFmlyCd", ns)
                    entry.sub_family_code = self._get_text(sub_fmly_cd)

        # Transaction details
        ntry_dtls = self._find_element(ntry, "NtryDtls", ns)
        if ntry_dtls is not None:
            for tx_dtls in self._find_all_elements(ntry_dtls, "TxDtls", ns):
                details = self._parse_transaction_details(tx_dtls, ns)
                entry.transaction_details.append(details)

        # Additional info
        addtl_ntry_inf = self._find_element(ntry, "AddtlNtryInf", ns)
        entry.additional_info = self._get_text(addtl_ntry_inf)

        return entry

    def _parse_transaction_details(self, tx_dtls: ET.Element, ns: Dict[str, str]) -> TransactionDetails:
        """Parse transaction details."""
        details = TransactionDetails()

        # References
        refs = self._find_element(tx_dtls, "Refs", ns)
        if refs is not None:
            instr_id = self._find_element(refs, "InstrId", ns)
            details.instruction_id = self._get_text(instr_id)

            end_to_end_id = self._find_element(refs, "EndToEndId", ns)
            details.end_to_end_id = self._get_text(end_to_end_id)

            tx_id = self._find_element(refs, "TxId", ns)
            details.transaction_id = self._get_text(tx_id)

            mndt_id = self._find_element(refs, "MndtId", ns)
            details.mandate_id = self._get_text(mndt_id)

        # Amount
        amt_dtls = self._find_element(tx_dtls, "AmtDtls", ns)
        if amt_dtls is not None:
            tx_amt = self._find_element(amt_dtls, "TxAmt", ns)
            if tx_amt is not None:
                amt = self._find_element(tx_amt, "Amt", ns)
                if amt is not None:
                    details.amount = Amount.from_xml(amt)

        # Related parties
        rltd_pties = self._find_element(tx_dtls, "RltdPties", ns)
        if rltd_pties is not None:
            dbtr = self._find_element(rltd_pties, "Dbtr", ns)
            if dbtr is not None:
                details.debtor = PartyIdentification.from_xml(dbtr, ns)

            dbtr_acct = self._find_element(rltd_pties, "DbtrAcct", ns)
            if dbtr_acct is not None:
                acct_id = self._find_element(dbtr_acct, "Id", ns)
                if acct_id is not None:
                    details.debtor_account = AccountIdentification.from_xml(acct_id, ns)

            cdtr = self._find_element(rltd_pties, "Cdtr", ns)
            if cdtr is not None:
                details.creditor = PartyIdentification.from_xml(cdtr, ns)

            cdtr_acct = self._find_element(rltd_pties, "CdtrAcct", ns)
            if cdtr_acct is not None:
                acct_id = self._find_element(cdtr_acct, "Id", ns)
                if acct_id is not None:
                    details.creditor_account = AccountIdentification.from_xml(acct_id, ns)

        # Remittance information
        rmt_inf = self._find_element(tx_dtls, "RmtInf", ns)
        if rmt_inf is not None:
            ustrd = self._find_element(rmt_inf, "Ustrd", ns)
            details.remittance_information = self._get_text(ustrd)

        return details
