"""
SWIFT MT Message Codes and Constants

Defines:
- Message type codes (MT1xx, MT2xx, etc.)
- Block types (1-5)
- Field tags and definitions
- Character set rules
- Message categories
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


class SwiftMessageType(Enum):
    """SWIFT MT message types."""

    # Category 1: Customer Payments and Cheques
    MT103 = ("103", "Single Customer Credit Transfer", "customer_payment")
    MT103_STP = ("103_STP", "Single Customer Credit Transfer (STP)", "customer_payment")
    MT103_REMIT = ("103_REMIT", "Single Customer Credit Transfer (Remittance)", "customer_payment")
    MT104 = ("104", "Direct Debit and Request for Debit Transfer", "customer_payment")
    MT107 = ("107", "General Direct Debit Message", "customer_payment")
    MT110 = ("110", "Advice of Cheque(s)", "customer_payment")
    MT111 = ("111", "Request for Stop Payment of a Cheque", "customer_payment")
    MT112 = ("112", "Status of a Request for Stop Payment of a Cheque", "customer_payment")

    # Category 2: Financial Institution Transfers
    MT200 = ("200", "Financial Institution Transfer for Own Account", "fi_transfer")
    MT201 = ("201", "Multiple Financial Institution Transfer for Own Account", "fi_transfer")
    MT202 = ("202", "General Financial Institution Transfer", "fi_transfer")
    MT202_COV = ("202_COV", "General Financial Institution Transfer (Cover)", "fi_transfer")
    MT203 = ("203", "Multiple General Financial Institution Transfer", "fi_transfer")
    MT204 = ("204", "Financial Markets Direct Debit Message", "fi_transfer")
    MT205 = ("205", "Financial Institution Transfer Execution", "fi_transfer")
    MT205_COV = ("205_COV", "Financial Institution Transfer Execution (Cover)", "fi_transfer")
    MT210 = ("210", "Notice to Receive", "fi_transfer")

    # Category 3: Treasury Markets - Foreign Exchange, Money Markets, Derivatives
    MT300 = ("300", "Foreign Exchange Confirmation", "treasury")
    MT303 = ("303", "Forex/Currency Option Allocation Instruction", "treasury")
    MT304 = ("304", "Advice/Instruction of a Third Party Deal", "treasury")
    MT305 = ("305", "Foreign Currency Option Confirmation", "treasury")
    MT306 = ("306", "Foreign Currency Option Confirmation", "treasury")
    MT320 = ("320", "Fixed Loan/Deposit Confirmation", "treasury")
    MT330 = ("330", "Call/Notice Loan/Deposit Confirmation", "treasury")
    MT340 = ("340", "Forward Rate Agreement Confirmation", "treasury")
    MT350 = ("350", "Advice of Loan/Deposit Interest Payment", "treasury")
    MT360 = ("360", "Single Currency Interest Rate Derivative Confirmation", "treasury")
    MT361 = ("361", "Cross Currency Interest Rate Swap Confirmation", "treasury")
    MT362 = ("362", "Interest Rate Reset/Advice of Payment", "treasury")
    MT364 = ("364", "Single Currency Interest Rate Swap Termination", "treasury")
    MT365 = ("365", "Single Currency Interest Rate Swap Termination", "treasury")

    # Category 4: Collections and Cash Letters
    MT400 = ("400", "Advice of Payment", "collections")
    MT405 = ("405", "Clean Collection", "collections")
    MT410 = ("410", "Acknowledgement", "collections")
    MT412 = ("412", "Advice of Acceptance", "collections")
    MT416 = ("416", "Advice of Non-Payment/Non-Acceptance", "collections")
    MT420 = ("420", "Tracer", "collections")
    MT422 = ("422", "Advice of Fate and Request for Instructions", "collections")
    MT430 = ("430", "Amendment of Instructions", "collections")
    MT450 = ("450", "Cash Letter Credit Advice", "collections")
    MT455 = ("455", "Cash Letter Credit Adjustment Advice", "collections")
    MT456 = ("456", "Advice of Dishonour", "collections")

    # Category 5: Securities Markets
    MT502 = ("502", "Order to Buy or Sell", "securities")
    MT503 = ("503", "Collateral Claim", "securities")
    MT504 = ("504", "Collateral Proposal", "securities")
    MT505 = ("505", "Collateral Substitution", "securities")
    MT506 = ("506", "Collateral and Exposure Statement", "securities")
    MT507 = ("507", "Collateral Status and Processing Advice", "securities")
    MT508 = ("508", "Intra-Position Advice", "securities")
    MT509 = ("509", "Trade Status Message", "securities")
    MT513 = ("513", "Client Advice of Execution", "securities")
    MT514 = ("514", "Trade Allocation Instruction", "securities")
    MT515 = ("515", "Client Confirmation of Purchase or Sale", "securities")
    MT518 = ("518", "Market-Side Securities Trade Confirmation", "securities")
    MT535 = ("535", "Statement of Holdings", "securities")
    MT536 = ("536", "Statement of Transactions", "securities")
    MT537 = ("537", "Statement of Pending Transactions", "securities")
    MT538 = ("538", "Statement of Intra-Position Advices", "securities")
    MT540 = ("540", "Receive Free", "securities")
    MT541 = ("541", "Receive Against Payment", "securities")
    MT542 = ("542", "Deliver Free", "securities")
    MT543 = ("543", "Deliver Against Payment", "securities")
    MT544 = ("544", "Receive Free Confirmation", "securities")
    MT545 = ("545", "Receive Against Payment Confirmation", "securities")
    MT546 = ("546", "Deliver Free Confirmation", "securities")
    MT547 = ("547", "Deliver Against Payment Confirmation", "securities")
    MT548 = ("548", "Settlement Status and Processing Advice", "securities")
    MT564 = ("564", "Corporate Action Notification", "securities")
    MT565 = ("565", "Corporate Action Instruction", "securities")
    MT566 = ("566", "Corporate Action Confirmation", "securities")
    MT567 = ("567", "Corporate Action Status and Processing Advice", "securities")
    MT568 = ("568", "Corporate Action Narrative", "securities")
    MT569 = ("569", "Triparty Collateral Status and Processing Advice", "securities")
    MT574 = ("574", "IRS 1441 NRA", "securities")
    MT575 = ("575", "Report of Combined Activity", "securities")
    MT576 = ("576", "Statement of Open Orders", "securities")
    MT577 = ("577", "Statement of Numbers", "securities")
    MT578 = ("578", "Statement of Allegement", "securities")
    MT579 = ("579", "Certificate Numbers", "securities")
    MT581 = ("581", "Collateral Adjustment Message", "securities")
    MT582 = ("582", "Reimbursement Claim or Advice", "securities")
    MT584 = ("584", "Statement of Settlement Allegements", "securities")
    MT586 = ("586", "Statement of Settlement Allegements", "securities")
    MT587 = ("587", "Depositary Receipt Instruction", "securities")
    MT588 = ("588", "Depositary Receipt Confirmation", "securities")
    MT589 = ("589", "Depositary Receipt Status and Processing Advice", "securities")

    # Category 6: Treasury Markets - Precious Metals and Syndications
    MT600 = ("600", "Precious Metal Trade Confirmation", "precious_metals")
    MT601 = ("601", "Precious Metal Option Confirmation", "precious_metals")
    MT604 = ("604", "Precious Metal Transfer/Delivery Order", "precious_metals")
    MT605 = ("605", "Precious Metal Notice to Receive", "precious_metals")
    MT606 = ("606", "Precious Metal Debit Advice", "precious_metals")
    MT607 = ("607", "Precious Metal Credit Advice", "precious_metals")
    MT608 = ("608", "Statement of a Metal Account", "precious_metals")
    MT609 = ("609", "Statement of a Metal Account", "precious_metals")
    MT620 = ("620", "Metal Fixed Loan/Deposit Confirmation", "precious_metals")
    MT643 = ("643", "Notice of Drawdown/Renewal", "syndication")
    MT644 = ("644", "Advice of Rate and Amount Fixing", "syndication")
    MT645 = ("645", "Notice of Fee Due", "syndication")
    MT646 = ("646", "Payment of Principal and/or Interest", "syndication")
    MT649 = ("649", "General Syndicated Facility Message", "syndication")

    # Category 7: Documentary Credits and Guarantees
    MT700 = ("700", "Issue of a Documentary Credit", "documentary_credit")
    MT701 = ("701", "Issue of a Documentary Credit", "documentary_credit")
    MT705 = ("705", "Pre-Advice of a Documentary Credit", "documentary_credit")
    MT707 = ("707", "Amendment to a Documentary Credit", "documentary_credit")
    MT710 = ("710", "Advice of a Third Bank's Documentary Credit", "documentary_credit")
    MT711 = ("711", "Advice of a Third Bank's Documentary Credit", "documentary_credit")
    MT720 = ("720", "Transfer of a Documentary Credit", "documentary_credit")
    MT721 = ("721", "Transfer of a Documentary Credit", "documentary_credit")
    MT730 = ("730", "Acknowledgement", "documentary_credit")
    MT732 = ("732", "Advice of Discharge", "documentary_credit")
    MT734 = ("734", "Advice of Refusal", "documentary_credit")
    MT740 = ("740", "Authorisation to Reimburse", "documentary_credit")
    MT742 = ("742", "Reimbursement Claim", "documentary_credit")
    MT747 = ("747", "Amendment to an Authorisation to Reimburse", "documentary_credit")
    MT750 = ("750", "Advice of Discrepancy", "documentary_credit")
    MT752 = ("752", "Authorisation to Pay, Accept or Negotiate", "documentary_credit")
    MT754 = ("754", "Advice of Payment/Acceptance/Negotiation", "documentary_credit")
    MT756 = ("756", "Advice of Reimbursement or Payment", "documentary_credit")
    MT760 = ("760", "Guarantee / Standby Letter of Credit", "guarantee")
    MT767 = ("767", "Guarantee / Standby Letter of Credit Amendment", "guarantee")
    MT768 = ("768", "Acknowledgement of a Guarantee / Standby Letter of Credit", "guarantee")
    MT769 = ("769", "Advice of Reduction or Release", "guarantee")

    # Category 8: Travellers Cheques
    MT800 = ("800", "T/C Sales and Settlement Advice", "travellers_cheque")
    MT801 = ("801", "T/C Multiple Sales Advice", "travellers_cheque")
    MT802 = ("802", "T/C Settlement Advice", "travellers_cheque")
    MT810 = ("810", "T/C Refund Request", "travellers_cheque")
    MT812 = ("812", "T/C Refund Authorisation", "travellers_cheque")
    MT813 = ("813", "T/C Refund Confirmation", "travellers_cheque")
    MT820 = ("820", "Request for T/C Stock", "travellers_cheque")
    MT821 = ("821", "T/C Inventory Addition", "travellers_cheque")
    MT822 = ("822", "Trust Receipt Acknowledgement", "travellers_cheque")
    MT823 = ("823", "T/C Inventory Transfer", "travellers_cheque")
    MT824 = ("824", "T/C Inventory Destruction/Cancellation Notice", "travellers_cheque")

    # Category 9: Cash Management and Customer Status
    MT900 = ("900", "Confirmation of Debit", "cash_management")
    MT910 = ("910", "Confirmation of Credit", "cash_management")
    MT920 = ("920", "Request Message", "cash_management")
    MT940 = ("940", "Customer Statement Message", "cash_management")
    MT941 = ("941", "Balance Report", "cash_management")
    MT942 = ("942", "Interim Transaction Report", "cash_management")
    MT950 = ("950", "Statement Message", "cash_management")
    MT970 = ("970", "Netting Statement", "cash_management")
    MT971 = ("971", "Netting Balance Report", "cash_management")
    MT972 = ("972", "Netting Interim Statement", "cash_management")
    MT973 = ("973", "Netting Request Message", "cash_management")
    MT985 = ("985", "Status Enquiry", "cash_management")
    MT986 = ("986", "Status Report", "cash_management")

    # Category n: Common Group Messages
    MT190 = ("190", "Advice of Charges, Interest and Other Adjustments", "common")
    MT191 = ("191", "Request for Payment of Charges, Interest, etc.", "common")
    MT192 = ("192", "Request for Cancellation", "common")
    MT195 = ("195", "Queries", "common")
    MT196 = ("196", "Answers", "common")
    MT198 = ("198", "Proprietary Message", "common")
    MT199 = ("199", "Free Format Message", "common")
    MT290 = ("290", "Advice of Charges, Interest and Other Adjustments", "common")
    MT291 = ("291", "Request for Payment of Charges, Interest, etc.", "common")
    MT292 = ("292", "Request for Cancellation", "common")
    MT295 = ("295", "Queries", "common")
    MT296 = ("296", "Answers", "common")
    MT298 = ("298", "Proprietary Message", "common")
    MT299 = ("299", "Free Format Message", "common")

    def __init__(self, code: str, description: str, category: str):
        self._code = code
        self._description = description
        self._category = category

    @property
    def code(self) -> str:
        return self._code

    @property
    def description(self) -> str:
        return self._description

    @property
    def category(self) -> str:
        return self._category

    @classmethod
    def from_code(cls, code: str) -> Optional["SwiftMessageType"]:
        """Get message type from code."""
        code = code.upper().replace(" ", "_")
        for mt in cls:
            if mt.code == code or mt.code.replace("_", "") == code.replace("_", ""):
                return mt
        return None


class SwiftBlockType(Enum):
    """SWIFT message block types."""

    BASIC_HEADER = ("1", "Basic Header Block")
    APPLICATION_HEADER = ("2", "Application Header Block")
    USER_HEADER = ("3", "User Header Block")
    TEXT = ("4", "Text Block")
    TRAILER = ("5", "Trailer Block")

    def __init__(self, number: str, description: str):
        self._number = number
        self._description = description

    @property
    def number(self) -> str:
        return self._number

    @property
    def description(self) -> str:
        return self._description


class SwiftCharacterSet(Enum):
    """SWIFT character set types."""

    # Character set X - alphanumeric and special
    SWIFT_X = ("x", r"[A-Za-z0-9/\-?:().,'+ ]")
    # Character set Y - uppercase alphabetic only
    SWIFT_Y = ("y", r"[A-Z ]")
    # Character set Z - alphanumeric and more special
    SWIFT_Z = ("z", r"[A-Za-z0-9/\-?:().,'+ \n]")
    # Numeric only
    NUMERIC = ("n", r"[0-9]")
    # Decimal amount
    DECIMAL = ("d", r"[0-9,]")
    # Alphabetic uppercase
    ALPHA = ("a", r"[A-Z]")
    # Alphanumeric
    ALPHANUM = ("c", r"[A-Za-z0-9]")

    def __init__(self, code: str, pattern: str):
        self._code = code
        self._pattern = pattern

    @property
    def code(self) -> str:
        return self._code

    @property
    def pattern(self) -> str:
        return self._pattern


@dataclass
class FieldDefinition:
    """Definition of a SWIFT field."""

    tag: str
    name: str
    format: str
    description: str
    mandatory: bool = True
    repeatable: bool = False
    options: List[str] = None

    def __post_init__(self):
        if self.options is None:
            self.options = []


class SwiftFieldTag(Enum):
    """Common SWIFT field tags."""

    # Transaction Reference Number
    F20 = ("20", "Transaction Reference Number", "16x", True)
    # Related Reference
    F21 = ("21", "Related Reference", "16x", False)
    # Bank Operation Code
    F23B = ("23B", "Bank Operation Code", "4!c", True)
    # Instruction Code
    F23E = ("23E", "Instruction Code", "4!c[/30x]", False)
    # Transaction Type Code
    F26T = ("26T", "Transaction Type Code", "3!c", False)
    # Value Date/Currency/Amount
    F32A = ("32A", "Value Date/Currency/Interbank Settled Amount", "6!n3!a15d", True)
    # Currency/Instructed Amount
    F33B = ("33B", "Currency/Instructed Amount", "3!a15d", False)
    # Exchange Rate
    F36 = ("36", "Exchange Rate", "12d", False)
    # Ordering Customer
    F50 = ("50", "Ordering Customer", "4*35x", False)
    F50A = ("50A", "Ordering Customer - Account with BIC", "[/34x]4!a2!a2!c[3!c]", False)
    F50F = ("50F", "Ordering Customer - Name and Address", "35x[/35x][/35x][/35x]", False)
    F50K = ("50K", "Ordering Customer - Name and Address", "[/34x]4*35x", False)
    # Ordering Institution
    F52A = ("52A", "Ordering Institution - BIC", "[/1!a][/34x]4!a2!a2!c[3!c]", False)
    F52D = ("52D", "Ordering Institution - Name and Address", "[/1!a][/34x]4*35x", False)
    # Sender's Correspondent
    F53A = ("53A", "Sender's Correspondent - BIC", "[/1!a][/34x]4!a2!a2!c[3!c]", False)
    F53B = ("53B", "Sender's Correspondent - Location", "[/1!a][/34x][35x]", False)
    F53D = ("53D", "Sender's Correspondent - Name and Address", "[/1!a][/34x]4*35x", False)
    # Receiver's Correspondent
    F54A = ("54A", "Receiver's Correspondent - BIC", "[/1!a][/34x]4!a2!a2!c[3!c]", False)
    F54B = ("54B", "Receiver's Correspondent - Location", "[/1!a][/34x][35x]", False)
    F54D = ("54D", "Receiver's Correspondent - Name and Address", "[/1!a][/34x]4*35x", False)
    # Third Reimbursement Institution
    F55A = ("55A", "Third Reimbursement Institution - BIC", "[/1!a][/34x]4!a2!a2!c[3!c]", False)
    F55B = ("55B", "Third Reimbursement Institution - Location", "[/1!a][/34x][35x]", False)
    F55D = ("55D", "Third Reimbursement Institution - Name and Address", "[/1!a][/34x]4*35x", False)
    # Intermediary Institution
    F56A = ("56A", "Intermediary Institution - BIC", "[/1!a][/34x]4!a2!a2!c[3!c]", False)
    F56C = ("56C", "Intermediary Institution - Account", "/34x", False)
    F56D = ("56D", "Intermediary Institution - Name and Address", "[/1!a][/34x]4*35x", False)
    # Account With Institution
    F57A = ("57A", "Account With Institution - BIC", "[/1!a][/34x]4!a2!a2!c[3!c]", False)
    F57B = ("57B", "Account With Institution - Location", "[/1!a][/34x][35x]", False)
    F57C = ("57C", "Account With Institution - Account", "/34x", False)
    F57D = ("57D", "Account With Institution - Name and Address", "[/1!a][/34x]4*35x", False)
    # Beneficiary Institution
    F58A = ("58A", "Beneficiary Institution - BIC", "[/1!a][/34x]4!a2!a2!c[3!c]", False)
    F58D = ("58D", "Beneficiary Institution - Name and Address", "[/1!a][/34x]4*35x", False)
    # Beneficiary Customer
    F59 = ("59", "Beneficiary Customer", "[/34x]4*35x", True)
    F59A = ("59A", "Beneficiary Customer - BIC", "[/34x]4!a2!a2!c[3!c]", False)
    F59F = ("59F", "Beneficiary Customer - Name and Address", "35x[/35x][/35x][/35x]", False)
    # Remittance Information
    F70 = ("70", "Remittance Information", "4*35x", False)
    # Details of Charges
    F71A = ("71A", "Details of Charges", "3!a", True)
    # Sender's Charges
    F71F = ("71F", "Sender's Charges", "3!a15d", False)
    # Receiver's Charges
    F71G = ("71G", "Receiver's Charges", "3!a15d", False)
    # Sender to Receiver Information
    F72 = ("72", "Sender to Receiver Information", "6*35x", False)
    # Regulatory Reporting
    F77B = ("77B", "Regulatory Reporting", "3*35x", False)

    def __init__(self, tag: str, name: str, format: str, mandatory: bool):
        self._tag = tag
        self._name = name
        self._format = format
        self._mandatory = mandatory

    @property
    def tag(self) -> str:
        return self._tag

    @property
    def field_name(self) -> str:
        return self._name

    @property
    def format(self) -> str:
        return self._format

    @property
    def mandatory(self) -> bool:
        return self._mandatory

    @classmethod
    def from_tag(cls, tag: str) -> Optional["SwiftFieldTag"]:
        """Get field definition from tag."""
        tag = tag.upper()
        for field in cls:
            if field.tag == tag:
                return field
        return None


# Message categories
SWIFT_MESSAGE_CATEGORIES: Dict[str, str] = {
    "1": "Customer Payments and Cheques",
    "2": "Financial Institution Transfers",
    "3": "Treasury Markets - FX, MM, Derivatives",
    "4": "Collections and Cash Letters",
    "5": "Securities Markets",
    "6": "Treasury Markets - Precious Metals and Syndications",
    "7": "Documentary Credits and Guarantees",
    "8": "Travellers Cheques",
    "9": "Cash Management and Customer Status",
    "n": "Common Group Messages",
}


def get_message_category(message_type: str) -> Optional[str]:
    """Get category description for a message type."""
    if not message_type:
        return None

    # Extract category digit
    mt_code = message_type.replace("MT", "").replace("mt", "")
    if mt_code and mt_code[0].isdigit():
        category = mt_code[0]
        return SWIFT_MESSAGE_CATEGORIES.get(category)

    return None


# Field definitions for common message types
MT103_FIELDS: Dict[str, FieldDefinition] = {
    "20": FieldDefinition("20", "Sender's Reference", "16x", "Unique reference assigned by sender", True),
    "13C": FieldDefinition("13C", "Time Indication", "/8c/4!n1!x4!n", "Time indication", False),
    "23B": FieldDefinition("23B", "Bank Operation Code", "4!c", "Type of operation", True),
    "23E": FieldDefinition("23E", "Instruction Code", "4!c[/30x]", "Instruction code", False, True),
    "26T": FieldDefinition("26T", "Transaction Type Code", "3!c", "Transaction type", False),
    "32A": FieldDefinition("32A", "Value Date/Currency/Amount", "6!n3!a15d", "Settlement details", True),
    "33B": FieldDefinition("33B", "Currency/Instructed Amount", "3!a15d", "Original amount", False),
    "36": FieldDefinition("36", "Exchange Rate", "12d", "Exchange rate", False),
    "50A": FieldDefinition("50A", "Ordering Customer", "[/34x]4!a2!a2!c[3!c]", "Debtor with BIC", False),
    "50F": FieldDefinition("50F", "Ordering Customer", "35x4*35x", "Debtor party identification", False),
    "50K": FieldDefinition("50K", "Ordering Customer", "[/34x]4*35x", "Debtor name and address", False),
    "51A": FieldDefinition("51A", "Sending Institution", "[/1!a][/34x]4!a2!a2!c[3!c]", "Sending bank", False),
    "52A": FieldDefinition("52A", "Ordering Institution", "[/1!a][/34x]4!a2!a2!c[3!c]", "Debtor agent BIC", False),
    "52D": FieldDefinition("52D", "Ordering Institution", "[/1!a][/34x]4*35x", "Debtor agent name", False),
    "53A": FieldDefinition("53A", "Sender's Correspondent", "[/1!a][/34x]4!a2!a2!c[3!c]", "Sender correspondent BIC", False),
    "53B": FieldDefinition("53B", "Sender's Correspondent", "[/1!a][/34x][35x]", "Sender correspondent location", False),
    "53D": FieldDefinition("53D", "Sender's Correspondent", "[/1!a][/34x]4*35x", "Sender correspondent name", False),
    "54A": FieldDefinition(
        "54A", "Receiver's Correspondent", "[/1!a][/34x]4!a2!a2!c[3!c]", "Receiver correspondent BIC", False
    ),
    "54B": FieldDefinition("54B", "Receiver's Correspondent", "[/1!a][/34x][35x]", "Receiver correspondent location", False),
    "54D": FieldDefinition("54D", "Receiver's Correspondent", "[/1!a][/34x]4*35x", "Receiver correspondent name", False),
    "55A": FieldDefinition(
        "55A", "Third Reimbursement Institution", "[/1!a][/34x]4!a2!a2!c[3!c]", "Third reimbursement BIC", False
    ),
    "55B": FieldDefinition(
        "55B", "Third Reimbursement Institution", "[/1!a][/34x][35x]", "Third reimbursement location", False
    ),
    "55D": FieldDefinition("55D", "Third Reimbursement Institution", "[/1!a][/34x]4*35x", "Third reimbursement name", False),
    "56A": FieldDefinition("56A", "Intermediary Institution", "[/1!a][/34x]4!a2!a2!c[3!c]", "Intermediary BIC", False),
    "56C": FieldDefinition("56C", "Intermediary Institution", "/34x", "Intermediary account", False),
    "56D": FieldDefinition("56D", "Intermediary Institution", "[/1!a][/34x]4*35x", "Intermediary name", False),
    "57A": FieldDefinition("57A", "Account With Institution", "[/1!a][/34x]4!a2!a2!c[3!c]", "Creditor agent BIC", False),
    "57B": FieldDefinition("57B", "Account With Institution", "[/1!a][/34x][35x]", "Creditor agent location", False),
    "57C": FieldDefinition("57C", "Account With Institution", "/34x", "Creditor agent account", False),
    "57D": FieldDefinition("57D", "Account With Institution", "[/1!a][/34x]4*35x", "Creditor agent name", False),
    "59": FieldDefinition("59", "Beneficiary Customer", "[/34x]4*35x", "Creditor", True),
    "59A": FieldDefinition("59A", "Beneficiary Customer", "[/34x]4!a2!a2!c[3!c]", "Creditor with BIC", False),
    "59F": FieldDefinition("59F", "Beneficiary Customer", "35x4*35x", "Creditor party identification", False),
    "70": FieldDefinition("70", "Remittance Information", "4*35x", "Remittance information", False),
    "71A": FieldDefinition("71A", "Details of Charges", "3!a", "Charge bearer", True),
    "71F": FieldDefinition("71F", "Sender's Charges", "3!a15d", "Sender charges", False, True),
    "71G": FieldDefinition("71G", "Receiver's Charges", "3!a15d", "Receiver charges", False),
    "72": FieldDefinition("72", "Sender to Receiver Information", "6*35x", "Additional information", False),
    "77B": FieldDefinition("77B", "Regulatory Reporting", "3*35x", "Regulatory reporting", False),
    "77T": FieldDefinition("77T", "Envelope Contents", "9000z", "Envelope contents", False),
}

MT202_FIELDS: Dict[str, FieldDefinition] = {
    "20": FieldDefinition("20", "Transaction Reference Number", "16x", "Unique reference", True),
    "21": FieldDefinition("21", "Related Reference", "16x", "Related reference", True),
    "13C": FieldDefinition("13C", "Time Indication", "/8c/4!n1!x4!n", "Time indication", False),
    "32A": FieldDefinition("32A", "Value Date/Currency/Amount", "6!n3!a15d", "Settlement details", True),
    "52A": FieldDefinition("52A", "Ordering Institution", "[/1!a][/34x]4!a2!a2!c[3!c]", "Ordering bank BIC", False),
    "52D": FieldDefinition("52D", "Ordering Institution", "[/1!a][/34x]4*35x", "Ordering bank name", False),
    "53A": FieldDefinition("53A", "Sender's Correspondent", "[/1!a][/34x]4!a2!a2!c[3!c]", "Sender correspondent BIC", False),
    "53B": FieldDefinition("53B", "Sender's Correspondent", "[/1!a][/34x][35x]", "Sender correspondent location", False),
    "53D": FieldDefinition("53D", "Sender's Correspondent", "[/1!a][/34x]4*35x", "Sender correspondent name", False),
    "54A": FieldDefinition(
        "54A", "Receiver's Correspondent", "[/1!a][/34x]4!a2!a2!c[3!c]", "Receiver correspondent BIC", False
    ),
    "54B": FieldDefinition("54B", "Receiver's Correspondent", "[/1!a][/34x][35x]", "Receiver correspondent location", False),
    "54D": FieldDefinition("54D", "Receiver's Correspondent", "[/1!a][/34x]4*35x", "Receiver correspondent name", False),
    "56A": FieldDefinition("56A", "Intermediary", "[/1!a][/34x]4!a2!a2!c[3!c]", "Intermediary BIC", False),
    "56D": FieldDefinition("56D", "Intermediary", "[/1!a][/34x]4*35x", "Intermediary name", False),
    "57A": FieldDefinition("57A", "Account With Institution", "[/1!a][/34x]4!a2!a2!c[3!c]", "Account with BIC", False),
    "57B": FieldDefinition("57B", "Account With Institution", "[/1!a][/34x][35x]", "Account with location", False),
    "57D": FieldDefinition("57D", "Account With Institution", "[/1!a][/34x]4*35x", "Account with name", False),
    "58A": FieldDefinition("58A", "Beneficiary Institution", "[/1!a][/34x]4!a2!a2!c[3!c]", "Beneficiary bank BIC", True),
    "58D": FieldDefinition("58D", "Beneficiary Institution", "[/1!a][/34x]4*35x", "Beneficiary bank name", False),
    "72": FieldDefinition("72", "Sender to Receiver Information", "6*35x", "Additional information", False),
}


def get_field_definition(message_type: str, tag: str) -> Optional[FieldDefinition]:
    """Get field definition for a specific message type and tag."""
    message_type = message_type.upper().replace("MT", "")

    if message_type == "103" or message_type.startswith("103"):
        return MT103_FIELDS.get(tag)
    elif message_type == "202" or message_type.startswith("202"):
        return MT202_FIELDS.get(tag)

    # Try common field lookup
    field_enum = SwiftFieldTag.from_tag(f"F{tag}")
    if field_enum:
        return FieldDefinition(
            tag=field_enum.tag,
            name=field_enum.field_name,
            format=field_enum.format,
            description=field_enum.field_name,
            mandatory=field_enum.mandatory,
        )

    return None


# Bank Operation Codes for MT103
BANK_OPERATION_CODES: Dict[str, str] = {
    "CRED": "Credit Transfer",
    "CRTS": "Credit Transfer for Test Purposes",
    "SPAY": "Specific Payment",
    "SPRI": "Priority Payment",
    "SSTD": "Standard Payment",
}

# Charge codes
CHARGE_CODES: Dict[str, str] = {
    "BEN": "All charges are to be borne by the beneficiary",
    "OUR": "All charges are to be borne by the ordering customer",
    "SHA": "Charges are shared between ordering customer and beneficiary",
}

# Instruction codes for field 23E
INSTRUCTION_CODES: Dict[str, str] = {
    "CHQB": "Pay Beneficiary Only by Cheque",
    "CORT": "Correspondent Bank Transfer",
    "HOLD": "Hold for Ordering Customer",
    "INTC": "Intracompany Transfer",
    "PHOB": "Please Phone Beneficiary",
    "PHOI": "Please Phone Intermediary",
    "PHON": "Please Phone Ordering Institution",
    "REPA": "Repeat of Previously Sent Payment",
    "SDVA": "Same Day Value",
    "TELB": "Please Advise/Contact Beneficiary by Telephone",
    "TELE": "Please Advise/Contact Intermediary by Telephone",
    "TELI": "Please Advise/Contact Ordering Customer by Telephone",
}
