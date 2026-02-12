"""
SEPA Code Definitions

Complete code definitions for SEPA payments including:
- Payment schemes (SCT, SCT Inst, SDD Core, SDD B2B)
- Service levels
- Local instruments
- Purpose codes
- Reject and return codes
"""

from enum import Enum, auto
from typing import Optional


class SEPAScheme(Enum):
    """SEPA Payment Schemes."""

    # Credit Transfer Schemes
    SCT = ("SCT", "SEPA Credit Transfer", "Standard credit transfer")
    SCT_INST = ("SCTI", "SEPA Instant Credit Transfer", "Real-time credit transfer")

    # Direct Debit Schemes
    SDD_CORE = ("CORE", "SEPA Direct Debit Core", "Consumer direct debit")
    SDD_B2B = ("B2B", "SEPA Direct Debit B2B", "Business-to-business direct debit")

    def __init__(self, code: str, name: str, description: str):
        self.code = code
        self.scheme_name = name
        self.description = description

    @classmethod
    def from_code(cls, code: str) -> Optional["SEPAScheme"]:
        """Get scheme from code."""
        for scheme in cls:
            if scheme.code == code:
                return scheme
        return None

    @property
    def is_instant(self) -> bool:
        """Check if this is an instant payment scheme."""
        return self == SEPAScheme.SCT_INST

    @property
    def is_direct_debit(self) -> bool:
        """Check if this is a direct debit scheme."""
        return self in [SEPAScheme.SDD_CORE, SEPAScheme.SDD_B2B]


class SEPAServiceLevel(Enum):
    """SEPA Service Level Codes."""

    SEPA = ("SEPA", "SEPA Scheme")
    URGP = ("URGP", "Urgent Payment")
    NURG = ("NURG", "Non-Urgent Payment")
    SDVA = ("SDVA", "Same Day Value")

    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description


class SEPALocalInstrument(Enum):
    """SEPA Local Instrument Codes (Direct Debit specific)."""

    # Direct Debit Types
    CORE = ("CORE", "SEPA Core Direct Debit")
    B2B = ("B2B", "SEPA B2B Direct Debit")
    COR1 = ("COR1", "SEPA Core Direct Debit (D-1)")

    # Credit Transfer Types
    INST = ("INST", "SEPA Instant Credit Transfer")

    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description


class SEPASequenceType(Enum):
    """SEPA Direct Debit Sequence Types."""

    FRST = ("FRST", "First", "First collection of a series")
    RCUR = ("RCUR", "Recurring", "Recurring collection")
    FNAL = ("FNAL", "Final", "Final collection of a series")
    OOFF = ("OOFF", "One-Off", "Single collection")

    def __init__(self, code: str, name: str, description: str):
        self.code = code
        self.seq_name = name
        self.description = description

    @classmethod
    def from_code(cls, code: str) -> Optional["SEPASequenceType"]:
        """Get sequence type from code."""
        for st in cls:
            if st.code == code:
                return st
        return None


class SEPACategoryPurpose(Enum):
    """SEPA Category Purpose Codes."""

    # Salary and Pension
    SALA = ("SALA", "Salary Payment")
    PENS = ("PENS", "Pension Payment")
    SSBE = ("SSBE", "Social Security Benefit")

    # Business
    SUPP = ("SUPP", "Supplier Payment")
    TRAD = ("TRAD", "Trade Settlement Payment")
    INTC = ("INTC", "Intra Company Payment")
    CASH = ("CASH", "Cash Management Transfer")
    CORT = ("CORT", "Trade Services Operation")

    # Tax
    TAXS = ("TAXS", "Tax Payment")
    VATX = ("VATX", "Value Added Tax Payment")
    WHLD = ("WHLD", "Withholding Tax")

    # Government
    GOVT = ("GOVT", "Government Payment")
    BONU = ("BONU", "Bonus Payment")

    # Other
    DIVI = ("DIVI", "Dividend")
    LOAN = ("LOAN", "Loan")
    OTHR = ("OTHR", "Other")

    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description


class SEPAPurposeCode(Enum):
    """SEPA Purpose Codes (ISO 20022 external codes)."""

    # Payments
    ACCT = ("ACCT", "Account Management")
    ADVA = ("ADVA", "Advance Payment")
    AGRT = ("AGRT", "Agricultural Transfer")
    AIRB = ("AIRB", "Air")
    ALMY = ("ALMY", "Alimony Payment")
    ANNI = ("ANNI", "Annuity")
    ANTS = ("ANTS", "Anesthesia Services")
    AREN = ("AREN", "Account Receivables Entry")

    # Salary/HR
    BEXP = ("BEXP", "Business Expenses")
    BOCE = ("BOCE", "Back Office Conversion Entry")
    BONU = ("BONU", "Bonus Payment")
    COMC = ("COMC", "Commercial")
    COMM = ("COMM", "Commission")

    # Bills and Utilities
    ELEC = ("ELEC", "Electricity Bill")
    GASB = ("GASB", "Gas Bill")
    WTER = ("WTER", "Water Bill")
    PHON = ("PHON", "Telephone Bill")
    CBFF = ("CBFF", "Capital Building Fringe Fortune")

    # Insurance
    INSU = ("INSU", "Insurance Premium")
    LIFI = ("LIFI", "Life Insurance")
    HLTI = ("HLTI", "Health Insurance")

    # Tax
    TAXS = ("TAXS", "Tax Payment")
    ESTX = ("ESTX", "Estate Tax")
    FWLV = ("FWLV", "Foreign Worker Levy")
    GSTX = ("GSTX", "Goods and Services Tax")
    HSTX = ("HSTX", "Housing Tax")
    INTX = ("INTX", "Income Tax")
    NITX = ("NITX", "Net Income Tax")
    PTXP = ("PTXP", "Property Tax")
    RDTX = ("RDTX", "Road Tax")
    VATX = ("VATX", "Value Added Tax")
    WHLD = ("WHLD", "Withholding")

    # Other
    RENT = ("RENT", "Rent")
    SALA = ("SALA", "Salary")
    PENS = ("PENS", "Pension")
    SSBE = ("SSBE", "Social Security Benefit")
    SUPP = ("SUPP", "Supplier Payment")

    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description


class SEPARejectCode(Enum):
    """SEPA Reject Reason Codes (for pacs.002/pain.002)."""

    # Account Issues
    AC01 = ("AC01", "Incorrect Account Number")
    AC04 = ("AC04", "Closed Account Number")
    AC06 = ("AC06", "Blocked Account")
    AC13 = ("AC13", "Invalid Debtor Account Type")

    # Agent Issues
    AG01 = ("AG01", "Transaction Forbidden")
    AG02 = ("AG02", "Invalid Bank Operation Code")

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

    # End Customer Issues
    BE01 = ("BE01", "Inconsistent With End Customer")
    BE04 = ("BE04", "Missing Creditor Address")
    BE05 = ("BE05", "Unrecognised Initiating Party")
    BE06 = ("BE06", "Unknown End Customer")
    BE07 = ("BE07", "Missing Debtor Address")

    # Date/Time Issues
    DT01 = ("DT01", "Invalid Date")

    # Format Issues
    FF01 = ("FF01", "Invalid File Format")
    FF05 = ("FF05", "Invalid Local Instrument Code")

    # Direct Debit Specific
    MD01 = ("MD01", "No Mandate")
    MD02 = ("MD02", "Missing Mandatory Mandated Related Information")
    MD06 = ("MD06", "Refund Request By End Customer")
    MD07 = ("MD07", "End Customer Deceased")

    # Mandate Issues
    MS02 = ("MS02", "Not Specified Reason Customer Generated")
    MS03 = ("MS03", "Not Specified Reason Agent Generated")

    # Regulatory
    RR01 = ("RR01", "Missing Debtor Account Or Identification")
    RR02 = ("RR02", "Missing Debtor Name Or Address")
    RR03 = ("RR03", "Missing Creditor Name Or Address")
    RR04 = ("RR04", "Regulatory Reason")

    # Service Level
    SL01 = ("SL01", "Specific Service Offered By Debtor Agent")
    SL02 = ("SL02", "Specific Service Offered By Creditor Agent")

    # Technical
    TECH = ("TECH", "Technical Problem")
    TM01 = ("TM01", "Cut Off Time")

    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description

    @classmethod
    def from_code(cls, code: str) -> Optional["SEPARejectCode"]:
        """Get reject code from string."""
        for rc in cls:
            if rc.code == code:
                return rc
        return None


class SEPAReturnCode(Enum):
    """SEPA Return Reason Codes (for pacs.004)."""

    # Account Related
    AC01 = ("AC01", "Incorrect Account Number")
    AC04 = ("AC04", "Closed Account")
    AC06 = ("AC06", "Blocked Account")

    # Amount Related
    AM04 = ("AM04", "Insufficient Funds")
    AM05 = ("AM05", "Duplicate")

    # Direct Debit Related
    MD01 = ("MD01", "No Mandate")
    MD06 = ("MD06", "Refund Request By End Customer")

    # Customer Request
    CUST = ("CUST", "Requested By Customer")
    FOCR = ("FOCR", "Following Cancellation Request")

    # Regulatory
    LEGL = ("LEGL", "Legal Reasons")
    RR04 = ("RR04", "Regulatory Reason")

    # Technical
    TECH = ("TECH", "Technical Problems")

    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description


# SEPA Constants
SEPA_CURRENCY = "EUR"  # SEPA only supports EUR
SEPA_MAX_AMOUNT = 999999999.99  # Maximum SEPA amount
SEPA_MESSAGE_ID_MAX_LENGTH = 35
SEPA_END_TO_END_ID_MAX_LENGTH = 35
SEPA_MANDATE_ID_MAX_LENGTH = 35
SEPA_CREDITOR_ID_MAX_LENGTH = 35

# SCT Inst specific limits
SCT_INST_MAX_AMOUNT = 100000.00  # Default SCT Inst limit (€100,000)
SCT_INST_TIMEOUT_SECONDS = 10  # Maximum 10 seconds for instant payment

# Direct Debit timelines (business days)
SDD_CORE_FIRST_D_DAYS = 5  # D-5 for first CORE collection
SDD_CORE_RECUR_D_DAYS = 2  # D-2 for recurring CORE collection
SDD_B2B_D_DAYS = 1  # D-1 for B2B collection
SDD_COR1_D_DAYS = 1  # D-1 for COR1 collection

# SEPA Countries (ISO 3166-1 alpha-2)
SEPA_COUNTRIES = {
    # EU Member States
    "AT",
    "BE",
    "BG",
    "HR",
    "CY",
    "CZ",
    "DK",
    "EE",
    "FI",
    "FR",
    "DE",
    "GR",
    "HU",
    "IE",
    "IT",
    "LV",
    "LT",
    "LU",
    "MT",
    "NL",
    "PL",
    "PT",
    "RO",
    "SK",
    "SI",
    "ES",
    "SE",
    # EEA Countries
    "IS",
    "LI",
    "NO",
    # Other SEPA Countries
    "AD",
    "MC",
    "SM",
    "VA",
    "CH",
    "GB",  # UK still in SEPA
}
