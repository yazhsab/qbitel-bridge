"""
ACH/NACHA Code Definitions

Standard Entry Class codes, transaction codes, and other
NACHA-defined code values.
"""

from enum import Enum
from typing import Optional


class ServiceClassCode(Enum):
    """
    ACH Batch Service Class Codes.

    Indicates the type of entries in the batch.
    """

    MIXED = ("200", "Mixed Debits and Credits")
    CREDITS_ONLY = ("220", "Credits Only")
    DEBITS_ONLY = ("225", "Debits Only")
    AUTOMATED_ACCOUNTING_ADVICE = ("280", "Automated Accounting Advice")

    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description

    @classmethod
    def from_code(cls, code: str) -> Optional["ServiceClassCode"]:
        for scc in cls:
            if scc.code == code:
                return scc
        return None


class TransactionCode(Enum):
    """
    ACH Transaction Codes.

    Two-digit codes indicating the type of account and transaction.
    """

    # Checking Account - Credits
    CHECKING_CREDIT = ("22", "Checking Credit (Deposit)")
    CHECKING_CREDIT_PRENOTE = ("23", "Checking Credit Prenote")
    CHECKING_CREDIT_ZERO_DOLLAR = ("24", "Checking Credit Zero Dollar with Remittance")

    # Checking Account - Debits
    CHECKING_DEBIT = ("27", "Checking Debit (Payment)")
    CHECKING_DEBIT_PRENOTE = ("28", "Checking Debit Prenote")
    CHECKING_DEBIT_ZERO_DOLLAR = ("29", "Checking Debit Zero Dollar with Remittance")

    # Savings Account - Credits
    SAVINGS_CREDIT = ("32", "Savings Credit (Deposit)")
    SAVINGS_CREDIT_PRENOTE = ("33", "Savings Credit Prenote")
    SAVINGS_CREDIT_ZERO_DOLLAR = ("34", "Savings Credit Zero Dollar with Remittance")

    # Savings Account - Debits
    SAVINGS_DEBIT = ("37", "Savings Debit (Payment)")
    SAVINGS_DEBIT_PRENOTE = ("38", "Savings Debit Prenote")
    SAVINGS_DEBIT_ZERO_DOLLAR = ("39", "Savings Debit Zero Dollar with Remittance")

    # GL Account (General Ledger) - Credits
    GL_CREDIT = ("42", "General Ledger Credit")
    GL_CREDIT_PRENOTE = ("43", "General Ledger Credit Prenote")
    GL_CREDIT_ZERO_DOLLAR = ("44", "General Ledger Credit Zero Dollar")

    # GL Account - Debits
    GL_DEBIT = ("47", "General Ledger Debit")
    GL_DEBIT_PRENOTE = ("48", "General Ledger Debit Prenote")
    GL_DEBIT_ZERO_DOLLAR = ("49", "General Ledger Debit Zero Dollar")

    # Loan Account - Credits
    LOAN_CREDIT = ("52", "Loan Account Credit")
    LOAN_CREDIT_PRENOTE = ("53", "Loan Account Credit Prenote")
    LOAN_CREDIT_ZERO_DOLLAR = ("54", "Loan Account Credit Zero Dollar")

    # Loan Account - Debits (Reversal)
    LOAN_DEBIT = ("55", "Loan Account Debit (Reversal)")
    LOAN_DEBIT_PRENOTE = ("56", "Loan Account Debit Prenote")

    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description

    @property
    def is_credit(self) -> bool:
        """Check if transaction is a credit."""
        return self.code[1] in ("2", "3", "4")

    @property
    def is_debit(self) -> bool:
        """Check if transaction is a debit."""
        return self.code[1] in ("7", "8", "9", "5", "6")

    @property
    def is_prenote(self) -> bool:
        """Check if transaction is a prenote (zero-dollar test)."""
        return self.code[1] in ("3", "8")

    @property
    def account_type(self) -> str:
        """Get account type."""
        first_digit = self.code[0]
        types = {
            "2": "Checking",
            "3": "Savings",
            "4": "General Ledger",
            "5": "Loan",
        }
        return types.get(first_digit, "Unknown")

    @classmethod
    def from_code(cls, code: str) -> Optional["TransactionCode"]:
        for tc in cls:
            if tc.code == code:
                return tc
        return None


class SECCode(Enum):
    """
    Standard Entry Class (SEC) Codes.

    Three-letter codes identifying the type of ACH entry.
    """

    # Consumer Entries
    PPD = ("PPD", "Prearranged Payment and Deposit", "Consumer credits/debits with prior authorization")
    WEB = ("WEB", "Internet-Initiated Entry", "Consumer entries authorized via internet")
    TEL = ("TEL", "Telephone-Initiated Entry", "Consumer entries authorized via telephone")
    POP = ("POP", "Point of Purchase Entry", "Debit at point of sale using check")
    ARC = ("ARC", "Accounts Receivable Entry", "Consumer check converted to ACH")
    BOC = ("BOC", "Back Office Conversion", "Consumer check converted at back office")
    RCK = ("RCK", "Re-presented Check Entry", "Returned check re-presented as ACH")

    # Corporate Entries
    CCD = ("CCD", "Corporate Credit or Debit", "Business-to-business payments")
    CTX = ("CTX", "Corporate Trade Exchange", "Business payments with addenda")
    CIE = ("CIE", "Customer-Initiated Entry", "Corporate entry initiated by consumer")

    # Government Entries
    ACK = ("ACK", "Acknowledgment Entry", "Acknowledgment of CCD entry")
    ATX = ("ATX", "Acknowledgment Entry with Addenda", "Acknowledgment of CTX entry")

    # Financial Institution Entries
    IAT = ("IAT", "International ACH Transaction", "Cross-border payments")
    ENR = ("ENR", "Automated Enrollment Entry", "Enrollment for direct deposit")
    TRX = ("TRX", "Truncated Entry", "Truncated check entries")
    XCK = ("XCK", "Destroyed Check Entry", "Destroyed check to ACH")

    # Point of Sale
    POS = ("POS", "Point of Sale Entry", "Point of sale debit")
    SHR = ("SHR", "Shared Network Entry", "Shared network transaction")
    MTE = ("MTE", "Machine Transfer Entry", "ATM entry")

    # Direct Deposit/Payment
    DNE = ("DNE", "Death Notification Entry", "Notification of death")

    def __init__(self, code: str, name: str, description: str):
        self.code = code
        self.sec_name = name
        self.description = description

    @property
    def is_consumer(self) -> bool:
        """Check if SEC code is for consumer entries."""
        return self.code in ("PPD", "WEB", "TEL", "POP", "ARC", "BOC", "RCK")

    @property
    def is_corporate(self) -> bool:
        """Check if SEC code is for corporate entries."""
        return self.code in ("CCD", "CTX", "CIE")

    @property
    def requires_authorization(self) -> bool:
        """Check if SEC code requires explicit authorization."""
        return self.code in ("PPD", "CCD", "CTX", "IAT")

    @classmethod
    def from_code(cls, code: str) -> Optional["SECCode"]:
        code = code.upper()
        for sec in cls:
            if sec.code == code:
                return sec
        return None


class AddendaTypeCode(Enum):
    """
    Addenda Type Codes.

    Identifies the type of addenda record.
    """

    STANDARD = ("05", "Standard Addenda")
    PAYMENT_RELATED = ("05", "Payment Related Information")
    IAT_FIRST_ADDENDA = ("10", "IAT First Addenda - Transaction")
    IAT_SECOND_ADDENDA = ("11", "IAT Second Addenda - Originator")
    IAT_THIRD_ADDENDA = ("12", "IAT Third Addenda - Originator Address")
    IAT_FOURTH_ADDENDA = ("13", "IAT Fourth Addenda - RDFI")
    IAT_FIFTH_ADDENDA = ("14", "IAT Fifth Addenda - Receiver Address")
    IAT_SIXTH_ADDENDA = ("15", "IAT Sixth Addenda - Receiver ID")
    IAT_SEVENTH_ADDENDA = ("16", "IAT Seventh Addenda - Remittance")
    IAT_FOREIGN_CORRESPONDENT = ("17", "IAT Foreign Correspondent Bank")
    IAT_REMITTANCE = ("18", "IAT Remittance Information")
    RETURN = ("99", "Return Entry Addenda")
    NOC = ("98", "Notification of Change Addenda")
    CONTESTED_DISHONORED = ("99", "Contested/Dishonored Return")

    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description

    @classmethod
    def from_code(cls, code: str) -> Optional["AddendaTypeCode"]:
        for atc in cls:
            if atc.code == code:
                return atc
        return None


class ReturnReasonCode(Enum):
    """
    ACH Return Reason Codes.

    Codes indicating why an ACH entry was returned.
    """

    # Administrative Returns
    R01 = ("R01", "Insufficient Funds", "Available balance not sufficient")
    R02 = ("R02", "Account Closed", "Account closed by customer or RDFI")
    R03 = ("R03", "No Account/Unable to Locate", "Account number does not exist")
    R04 = ("R04", "Invalid Account Number", "Account number structure invalid")
    R05 = ("R05", "Unauthorized Debit to Consumer Account", "Consumer did not authorize")
    R06 = ("R06", "Returned per ODFIs Request", "ODFI requested return")
    R07 = ("R07", "Authorization Revoked by Customer", "Authorization revoked")
    R08 = ("R08", "Payment Stopped", "Stop payment on this item")
    R09 = ("R09", "Uncollected Funds", "Funds not yet collected")
    R10 = ("R10", "Customer Advises Unauthorized", "Customer claims unauthorized")

    # Account Status Returns
    R11 = ("R11", "Check Truncation Entry Return", "Check truncation return")
    R12 = ("R12", "Account Sold to Another DFI", "Account transferred")
    R13 = ("R13", "Invalid ACH Routing Number", "RDFI not ACH participant")
    R14 = ("R14", "Representative Payee Deceased", "Representative deceased")
    R15 = ("R15", "Beneficiary or Account Holder Deceased", "Account holder deceased")
    R16 = ("R16", "Account Frozen", "Account frozen - legal action")
    R17 = ("R17", "File Record Edit Criteria", "Entry failed edit criteria")
    R18 = ("R18", "Improper Effective Entry Date", "Effective date invalid")
    R19 = ("R19", "Amount Field Error", "Amount field error")
    R20 = ("R20", "Non-Transaction Account", "Account is non-transaction")

    # Regulatory Returns
    R21 = ("R21", "Invalid Company Identification", "Company ID invalid")
    R22 = ("R22", "Invalid Individual ID Number", "Individual ID invalid")
    R23 = ("R23", "Credit Entry Refused by Receiver", "Receiver refused credit")
    R24 = ("R24", "Duplicate Entry", "Duplicate entry")
    R25 = ("R25", "Addenda Error", "Addenda record error")
    R26 = ("R26", "Mandatory Field Error", "Required field missing")
    R27 = ("R27", "Trace Number Error", "Trace number error")
    R28 = ("R28", "Routing Number Check Digit Error", "Check digit wrong")
    R29 = ("R29", "Corporate Customer Advises Not Authorized", "Corporate unauthorized")
    R30 = ("R30", "RDFI Not Participant in Check Truncation", "Not truncation participant")

    R31 = ("R31", "Permissible Return Entry (CCD/CTX)", "Permissible return")
    R32 = ("R32", "RDFI Non-Settlement", "RDFI non-settlement")
    R33 = ("R33", "Return of XCK Entry", "XCK return")
    R34 = ("R34", "Limited Participation DFI", "Limited participation")
    R35 = ("R35", "Return of Improper Debit Entry", "Improper debit returned")
    R36 = ("R36", "Return of Improper Credit Entry", "Improper credit returned")

    def __init__(self, code: str, reason: str, description: str):
        self.code = code
        self.reason = reason
        self.description = description

    @property
    def is_administrative(self) -> bool:
        """Check if return is administrative."""
        return self.code in ("R01", "R02", "R03", "R04", "R06", "R08", "R09")

    @property
    def requires_correction(self) -> bool:
        """Check if return indicates data correction needed."""
        return self.code in ("R03", "R04", "R13", "R17", "R25", "R26", "R27", "R28")

    @property
    def is_unauthorized(self) -> bool:
        """Check if return indicates unauthorized transaction."""
        return self.code in ("R05", "R07", "R10", "R29")

    @classmethod
    def from_code(cls, code: str) -> Optional["ReturnReasonCode"]:
        code = code.upper()
        for rrc in cls:
            if rrc.code == code:
                return rrc
        return None
