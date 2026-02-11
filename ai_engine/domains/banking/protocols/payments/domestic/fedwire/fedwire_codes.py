"""
FedWire Code Definitions

Complete code definitions for FedWire Funds Transfer messages including:
- Type Codes (Basic, Funds Transfer, Foreign Transfer, Settlement)
- Type Sub-Codes
- Business Function Codes
- Tag Definitions
- ID Codes for party identification
"""

from enum import Enum
from typing import Dict, Optional, Tuple


class TypeCode(Enum):
    """
    FedWire Type Codes - Primary classification of message type.

    The Type Code indicates the general category of the funds transfer.
    """

    # Basic Funds Transfers (10)
    BASIC_FUNDS_TRANSFER = ("10", "Basic Funds Transfer")

    # Request for Reversal (15)
    REQUEST_FOR_REVERSAL = ("15", "Request for Reversal")

    # Reversal of Transfer (16)
    REVERSAL_OF_TRANSFER = ("16", "Reversal of Transfer")

    # Request for Reversal of Prior Day Transfer (17)
    REQUEST_REVERSAL_PRIOR_DAY = ("17", "Request for Reversal of Prior Day Transfer")

    # Reversal of Prior Day Transfer (18)
    REVERSAL_PRIOR_DAY = ("18", "Reversal of Prior Day Transfer")

    # As-Of Adjustment (Debit) (25)
    AS_OF_ADJUSTMENT_DEBIT = ("25", "As-Of Adjustment (Debit)")

    # As-Of Adjustment (Credit) (26)
    AS_OF_ADJUSTMENT_CREDIT = ("26", "As-Of Adjustment (Credit)")

    # Service Message (90)
    SERVICE_MESSAGE = ("90", "Service Message")

    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description

    @classmethod
    def from_code(cls, code: str) -> Optional["TypeCode"]:
        """Get TypeCode from string code."""
        for tc in cls:
            if tc.code == code:
                return tc
        return None


class TypeSubCode(Enum):
    """
    FedWire Type Sub-Codes - Further classification of message type.

    Sub-codes provide additional detail about the type of transfer.
    """

    # Basic Transfer Sub-Codes
    BASIC_TRANSFER = ("00", "Basic Transfer")
    REQUEST_FOR_CREDIT = ("01", "Request for Credit (Drawdown)")
    FUNDS_TRANSFER_REQUEST_CREDIT = ("02", "Funds Transfer Honoring Request for Credit")
    REFUSAL_HONOR_REQUEST = ("03", "Refusal to Honor Request for Credit")

    # Settlement Sub-Codes
    SSI_SERVICE_MESSAGE = ("07", "SSI Service Message")

    # Request Codes
    REQUEST_FOR_DEBIT = ("08", "Request for Debit Authorization")
    FUNDS_TRANSFER_REQUEST_DEBIT = ("09", "Funds Transfer Honoring Request for Debit")
    REFUSAL_HONOR_DEBIT = ("10", "Refusal to Honor Request for Debit")

    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description

    @classmethod
    def from_code(cls, code: str) -> Optional["TypeSubCode"]:
        """Get TypeSubCode from string code."""
        for tsc in cls:
            if tsc.code == code:
                return tsc
        return None


class BusinessFunctionCode(Enum):
    """
    FedWire Business Function Codes.

    Identifies the business reason for the funds transfer.
    """

    # Bank Transfer (BTR) - Bank-to-bank transfer
    BANK_TRANSFER = ("BTR", "Bank Transfer", "Bank-to-bank transfer of own funds")

    # Customer Transfer (CTR) - Transfer on behalf of customer
    CUSTOMER_TRANSFER = ("CTR", "Customer Transfer", "Transfer on behalf of customer")

    # Customer Transfer Plus (CTP) - Customer transfer with additional info
    CUSTOMER_TRANSFER_PLUS = ("CTP", "Customer Transfer Plus",
                              "Customer transfer with originator/beneficiary info")

    # Drawdown Request (DRW) - Request to draw funds
    DRAWDOWN_REQUEST = ("DRW", "Drawdown Request", "Request for drawdown of funds")

    # Drawdown Transfer (DRT) - Transfer honoring drawdown request
    DRAWDOWN_TRANSFER = ("DRT", "Drawdown Transfer", "Transfer honoring drawdown request")

    # Drawdown Refusal (DRR) - Refusal of drawdown request
    DRAWDOWN_REFUSAL = ("DRR", "Drawdown Refusal", "Refusal of drawdown request")

    # Federal Reserve Bank (FRB) - Federal Reserve transaction
    FED_FUNDS = ("FFR", "Fed Funds", "Federal funds transaction")

    # Cover Payment (COV) - Cover payment for underlying transaction
    COVER_PAYMENT = ("COV", "Cover Payment", "Cover payment for underlying transaction")

    # Service Message (SVC) - Service/administrative message
    SERVICE_MESSAGE = ("SVC", "Service Message", "Service or administrative message")

    # Check Same Day Settlement (CKS) - Check clearing settlement
    CHECK_SAME_DAY = ("CKS", "Check Same Day Settlement",
                      "Check clearing same-day settlement")

    # Deposit to Sender's Account (DEP) - Deposit
    DEPOSIT = ("DEP", "Deposit to Sender's Account", "Deposit to sender's account")

    # FedNow Liquidity Transfer (FNS)
    FEDNOW_SETTLEMENT = ("FNS", "FedNow Settlement", "FedNow liquidity management")

    def __init__(self, code: str, short_name: str, description: str):
        self.code = code
        self.short_name = short_name
        self.description = description

    @classmethod
    def from_code(cls, code: str) -> Optional["BusinessFunctionCode"]:
        """Get BusinessFunctionCode from string code."""
        for bfc in cls:
            if bfc.code == code:
                return bfc
        return None


class FedWireTag(Enum):
    """
    FedWire Tag Definitions.

    Tags define the structure and content of FedWire messages.
    Each tag has a specific format and purpose.
    """

    # Mandatory Tags
    SENDER_SUPPLIED_INFO = ("1500", "Sender Supplied Information", True, 1, 35)
    TYPE_SUBTYPE = ("1510", "Type and Subtype", True, 4, 4)
    IMAD = ("1520", "Input Message Accountability Data", True, 22, 22)
    AMOUNT = ("2000", "Amount", True, 1, 12)
    SENDER_DI = ("3100", "Sender Depository Institution", True, 9, 43)
    RECEIVER_DI = ("3400", "Receiver Depository Institution", True, 9, 43)
    BUSINESS_FUNCTION_CODE = ("3600", "Business Function Code", True, 3, 3)

    # Conditional/Optional Tags - Sender/Receiver Info
    SENDER_REFERENCE = ("3320", "Sender Reference", False, 1, 16)
    PREVIOUS_MESSAGE_ID = ("3500", "Previous Message Identifier", False, 22, 22)

    # Beneficiary Tags
    BENEFICIARY_FI = ("4000", "Beneficiary FI", False, 9, 43)
    BENEFICIARY = ("4200", "Beneficiary", False, 1, 43)

    # Originator Tags
    ORIGINATOR = ("5000", "Originator", False, 1, 43)
    ORIGINATOR_FI = ("5100", "Originator FI", False, 9, 43)

    # Instructing/Instructed FI
    INSTRUCTING_FI = ("5200", "Instructing FI", False, 9, 43)

    # Account Numbers
    ACCOUNT_DEBITED = ("5400", "Account Debited in Drawdown", False, 1, 34)
    ACCOUNT_CREDITED = ("5500", "Account Credited in Drawdown", False, 1, 34)

    # Intermediary Tags
    INTERMEDIARY_FI = ("4100", "Intermediary FI", False, 9, 43)
    RECEIVER_FI = ("4100", "Receiver FI (Intermediary)", False, 9, 43)

    # Originator/Beneficiary Options
    ORIGINATOR_TO_BENEFICIARY_INFO = ("6000", "Originator to Beneficiary Info", False, 1, 140)
    FI_TO_FI_INFO = ("6100", "FI to FI Information", False, 1, 210)

    # Additional Fields
    CHARGES = ("3700", "Charges", False, 1, 11)
    INSTRUCTED_AMOUNT = ("3710", "Instructed Amount", False, 3, 31)
    EXCHANGE_RATE = ("3720", "Exchange Rate", False, 1, 12)

    # Beneficiary Intermediary
    BENEFICIARY_INTERMEDIARY_FI = ("4000", "Beneficiary Intermediary FI", False, 9, 43)

    # Drawdown Tags
    DRAWDOWN_DEBIT_ACCOUNT_ADVICE = ("6200", "Drawdown Debit Account Advice Info",
                                     False, 1, 200)

    # Remittance Data
    RELATED_REMITTANCE_INFO = ("8200", "Related Remittance Information", False, 1, 3000)
    REMITTANCE_ORIGINATOR = ("8300", "Remittance Originator", False, 1, 140)
    REMITTANCE_BENEFICIARY = ("8350", "Remittance Beneficiary", False, 1, 140)
    PRIMARY_REMITTANCE_DOC = ("8400", "Primary Remittance Document", False, 1, 280)
    ACTUAL_AMOUNT_PAID = ("8450", "Actual Amount Paid", False, 10, 27)
    GROSS_AMOUNT_REMIT = ("8500", "Gross Amount of Remittance Doc", False, 10, 27)
    AMOUNT_NEGOTIATED_DISCOUNT = ("8550", "Amt of Negotiated Discount", False, 10, 32)
    ADJUSTMENT = ("8600", "Adjustment", False, 1, 280)
    DATE_REMIT_DOC = ("8650", "Date of Remittance Document", False, 8, 8)
    SECONDARY_REMIT_DOC = ("8700", "Secondary Remittance Document", False, 1, 200)
    REMIT_FREE_TEXT = ("8750", "Remittance Free Text", False, 1, 840)

    # Service Message Tags
    SERVICE_MESSAGE_INFO = ("9000", "Service Message Information", False, 1, 200)
    MESSAGE_DISPOSITION = ("1100", "Message Disposition", False, 1, 80)
    RECEIPT_TIME_STAMP = ("1110", "Receipt Time Stamp", False, 14, 14)
    OUTPUT_MESSAGE_ACCOUNTABILITY = ("1120", "Output Message Accountability Data", False, 22, 22)
    ERROR = ("1130", "Error", False, 1, 80)

    # Unstructured Addenda
    UNSTRUCTURED_ADDENDA = ("7500", "Unstructured Addenda", False, 1, 9000)

    def __init__(self, tag: str, description: str, mandatory: bool,
                 min_length: int, max_length: int):
        self.tag = tag
        self.description = description
        self.mandatory = mandatory
        self.min_length = min_length
        self.max_length = max_length

    @classmethod
    def from_tag(cls, tag: str) -> Optional["FedWireTag"]:
        """Get FedWireTag from tag string."""
        for fwt in cls:
            if fwt.tag == tag:
                return fwt
        return None


class IDCode(Enum):
    """
    ID Codes for party identification in FedWire messages.
    """

    # Bank Identification
    ABA_ROUTING = ("D", "ABA Routing Number")
    SWIFT_BIC = ("B", "SWIFT BIC")
    CHIPS_PARTICIPANT = ("C", "CHIPS Participant ID")
    DDA_NUMBER = ("U", "Demand Deposit Account Number")
    FW_ID = ("F", "FedWire ID")

    # Tax Identification
    EIN = ("T", "Employer Identification Number")
    SSN_ITIN = ("S", "Social Security Number / ITIN")

    # Other IDs
    PASSPORT = ("1", "Passport Number")
    ALIEN_REG = ("2", "Alien Registration Number")
    CORPORATE_ID = ("3", "Corporate Identification")
    DRIVERS_LICENSE = ("4", "Driver's License Number")
    CUSTOMER_ID = ("5", "Customer Identification Number")

    # Unknown/Other
    OTHER = ("9", "Other Identification")

    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description

    @classmethod
    def from_code(cls, code: str) -> Optional["IDCode"]:
        """Get IDCode from code string."""
        for ic in cls:
            if ic.code == code:
                return ic
        return None


class DrawdownDebitAccountAdviceCode(Enum):
    """
    Drawdown Debit Account Advice Codes.
    """

    ADVICE_NO_LETTER = ("ADV", "Advise Account - No Letter")
    ADVICE_LETTER_FOLLOWS = ("LTR", "Advise Account - Letter Follows")
    NO_ADVICE_REQUIRED = ("NON", "No Advice Required")
    PHONE_CALL_REQUIRED = ("PHN", "Phone Call Required")
    WIRE_ADVICE = ("WRE", "Wire Advice")

    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description

    @classmethod
    def from_code(cls, code: str) -> Optional["DrawdownDebitAccountAdviceCode"]:
        """Get DrawdownDebitAccountAdviceCode from code string."""
        for dda in cls:
            if dda.code == code:
                return dda
        return None


# Validation constants
FEDWIRE_AMOUNT_MAX = 9_999_999_999_99  # $99,999,999,999.99
FEDWIRE_ABA_LENGTH = 9
FEDWIRE_BIC_LENGTH_MIN = 8
FEDWIRE_BIC_LENGTH_MAX = 11

# Tag categories for message structure validation
MANDATORY_TAGS = [
    FedWireTag.SENDER_SUPPLIED_INFO,
    FedWireTag.TYPE_SUBTYPE,
    FedWireTag.IMAD,
    FedWireTag.AMOUNT,
    FedWireTag.SENDER_DI,
    FedWireTag.RECEIVER_DI,
    FedWireTag.BUSINESS_FUNCTION_CODE,
]

# Business function code requirements
BFC_REQUIREMENTS: Dict[BusinessFunctionCode, Dict[str, bool]] = {
    BusinessFunctionCode.BANK_TRANSFER: {
        "beneficiary_required": False,
        "originator_required": False,
        "originator_fi_required": False,
    },
    BusinessFunctionCode.CUSTOMER_TRANSFER: {
        "beneficiary_required": True,
        "originator_required": False,
        "originator_fi_required": False,
    },
    BusinessFunctionCode.CUSTOMER_TRANSFER_PLUS: {
        "beneficiary_required": True,
        "originator_required": True,
        "originator_fi_required": False,
    },
    BusinessFunctionCode.DRAWDOWN_REQUEST: {
        "beneficiary_required": True,
        "originator_required": True,
        "originator_fi_required": True,
    },
    BusinessFunctionCode.DRAWDOWN_TRANSFER: {
        "beneficiary_required": True,
        "originator_required": True,
        "originator_fi_required": True,
    },
    BusinessFunctionCode.COVER_PAYMENT: {
        "beneficiary_required": True,
        "originator_required": True,
        "originator_fi_required": True,
    },
}
