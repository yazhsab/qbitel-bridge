"""
FedNow Code Definitions

Complete code definitions for FedNow Instant Payment Service including:
- Message types
- Reject/Return reason codes
- Participant types
- Service types
- Transaction statuses
"""

from enum import Enum, auto
from typing import Optional


class FedNowMessageType(Enum):
    """
    FedNow Message Types based on ISO 20022.

    FedNow uses a subset of ISO 20022 messages for instant payments.
    """

    # Credit Transfer Messages
    CREDIT_TRANSFER = ("pacs.008", "FI to FI Customer Credit Transfer")
    PAYMENT_STATUS_REPORT = ("pacs.002", "FI to FI Payment Status Report")
    PAYMENT_RETURN = ("pacs.004", "Payment Return")

    # Request for Payment Messages
    REQUEST_FOR_PAYMENT = ("pain.013", "Creditor Payment Activation Request")
    RFP_RESPONSE = ("pain.014", "Creditor Payment Activation Request Status Report")

    # Administrative Messages
    PAYMENT_CANCELLATION = ("camt.056", "FI to FI Payment Cancellation Request")
    RESOLUTION_OF_INVESTIGATION = ("camt.029", "Resolution of Investigation")

    # Account Management
    ACCOUNT_REPORTING_REQUEST = ("camt.060", "Account Reporting Request")
    BANK_TO_CUSTOMER_STATEMENT = ("camt.053", "Bank to Customer Statement")

    # Liquidity Management
    LIQUIDITY_CREDIT_TRANSFER = ("camt.050", "Liquidity Credit Transfer")
    LIQUIDITY_DEBIT_TRANSFER = ("camt.051", "Liquidity Debit Transfer")

    def __init__(self, iso_code: str, description: str):
        self.iso_code = iso_code
        self.description = description

    @classmethod
    def from_iso_code(cls, code: str) -> Optional["FedNowMessageType"]:
        """Get message type from ISO code."""
        for mt in cls:
            if mt.iso_code == code:
                return mt
        return None


class FedNowRejectCode(Enum):
    """
    FedNow Reject Reason Codes.

    Used in pacs.002 messages to indicate why a payment was rejected.
    """

    # Account Issues
    AC01 = ("AC01", "Incorrect Account Number", "Account number is invalid")
    AC02 = ("AC02", "Invalid Debtor Account Number", "Debtor account invalid")
    AC03 = ("AC03", "Invalid Creditor Account Number", "Creditor account invalid")
    AC04 = ("AC04", "Closed Account Number", "Account has been closed")
    AC05 = ("AC05", "Closed Debtor Account", "Debtor account closed")
    AC06 = ("AC06", "Blocked Account", "Account is blocked")
    AC07 = ("AC07", "Closed Creditor Account", "Creditor account closed")
    AC13 = ("AC13", "Invalid Debtor Account Type", "Wrong account type for debtor")
    AC14 = ("AC14", "Invalid Creditor Account Type", "Wrong account type for creditor")

    # Agent/Bank Issues
    AG01 = ("AG01", "Transaction Forbidden", "Transaction not allowed")
    AG02 = ("AG02", "Invalid Bank Operation Code", "Bank operation code invalid")
    AG03 = ("AG03", "Transaction Not Supported", "Transaction type not supported")
    AG09 = ("AG09", "Payment Not Received", "Expected payment not received")
    AG10 = ("AG10", "Agent Suspended", "Agent is suspended")

    # Amount Issues
    AM01 = ("AM01", "Zero Amount", "Amount is zero")
    AM02 = ("AM02", "Not Allowed Amount", "Amount not allowed")
    AM03 = ("AM03", "Not Allowed Currency", "Currency not allowed")
    AM04 = ("AM04", "Insufficient Funds", "Insufficient funds")
    AM05 = ("AM05", "Duplicate", "Duplicate payment")
    AM06 = ("AM06", "Too Low Amount", "Amount is too low")
    AM07 = ("AM07", "Blocked Amount", "Amount is blocked")
    AM09 = ("AM09", "Wrong Amount", "Amount is incorrect")
    AM10 = ("AM10", "Invalid Control Sum", "Control sum is invalid")

    # Technical Issues
    BE01 = ("BE01", "Inconsistent with End Customer", "Data inconsistent")
    BE04 = ("BE04", "Missing Creditor Address", "Creditor address required")
    BE05 = ("BE05", "Unrecognized Initiating Party", "Initiating party unknown")
    BE06 = ("BE06", "Unknown End Customer", "End customer not recognized")
    BE07 = ("BE07", "Missing Debtor Address", "Debtor address required")

    # Date/Time Issues
    DT01 = ("DT01", "Invalid Date", "Date is invalid")
    DT02 = ("DT02", "Invalid Creation Date Time", "Creation date/time invalid")
    DT03 = ("DT03", "Invalid Non-Processing Date", "Non-processing date invalid")
    DT04 = ("DT04", "Future Date Not Supported", "Future date not allowed")

    # Format Issues
    FF01 = ("FF01", "Invalid File Format", "File format is invalid")
    FF03 = ("FF03", "Invalid Payment Type Info", "Payment type info invalid")
    FF04 = ("FF04", "Invalid Service Level Code", "Service level code invalid")

    # Regulatory Issues
    RR01 = ("RR01", "Missing Debtor Account or ID", "Debtor info required")
    RR02 = ("RR02", "Missing Debtor Name or Address", "Debtor name/address required")
    RR03 = ("RR03", "Missing Creditor Name or Address", "Creditor name/address required")
    RR04 = ("RR04", "Regulatory Reason", "Rejected for regulatory reasons")

    # FedNow Specific
    FNOW01 = ("FNOW01", "Participant Not Active", "Participant is not active")
    FNOW02 = ("FNOW02", "Outside Operating Hours", "Outside FedNow operating hours")
    FNOW03 = ("FNOW03", "Message Timeout", "Message processing timeout")
    FNOW04 = ("FNOW04", "Liquidity Insufficient", "Insufficient liquidity")

    # Technical Validation
    TM01 = ("TM01", "Invalid Cut Off Time", "Cut off time has passed")
    TS01 = ("TS01", "Technical Issue", "Technical issue prevented processing")
    TS02 = ("TS02", "Invalid File Format", "File format is invalid")

    def __init__(self, code: str, name: str, description: str):
        self.code = code
        self.reason_name = name
        self.description = description

    @classmethod
    def from_code(cls, code: str) -> Optional["FedNowRejectCode"]:
        """Get reject code from string."""
        for rc in cls:
            if rc.code == code:
                return rc
        return None


class FedNowReturnCode(Enum):
    """
    FedNow Return Reason Codes.

    Used in pacs.004 messages for returning funds.
    """

    # Account Related Returns
    AC01 = ("AC01", "Incorrect Account Number")
    AC04 = ("AC04", "Closed Account")
    AC06 = ("AC06", "Blocked Account")

    # Amount Related Returns
    AM04 = ("AM04", "Insufficient Funds")
    AM05 = ("AM05", "Duplicate")
    AM09 = ("AM09", "Wrong Amount")

    # Beneficiary/Customer Returns
    BE04 = ("BE04", "Missing Creditor Address")
    CUST = ("CUST", "Requested by Customer")
    CUTA = ("CUTA", "Customer Cancellation Request")

    # Fraud Related
    FRAD = ("FRAD", "Fraudulent Origin")
    FRTR = ("FRTR", "Fraud Related Return")

    # Regulatory
    LEGL = ("LEGL", "Legal Reasons")
    RR04 = ("RR04", "Regulatory Reason")

    # Technical
    TECH = ("TECH", "Technical Problems")
    NARR = ("NARR", "Narrative - See Additional Info")

    # FedNow Specific
    FNRR = ("FNRR", "FedNow Return Requested")

    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description

    @classmethod
    def from_code(cls, code: str) -> Optional["FedNowReturnCode"]:
        """Get return code from string."""
        for rc in cls:
            if rc.code == code:
                return rc
        return None


class FedNowParticipantType(Enum):
    """Types of FedNow participants."""

    # Direct participants
    DIRECT_SEND_RECEIVE = ("DSR", "Direct Send and Receive")
    DIRECT_RECEIVE_ONLY = ("DRO", "Direct Receive Only")
    DIRECT_LIQUIDITY = ("DLQ", "Direct Liquidity Provider")

    # Through participants (via correspondent)
    CORRESPONDENT = ("COR", "Correspondent Banking")
    SERVICE_PROVIDER = ("SVP", "Service Provider")

    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description


class FedNowServiceType(Enum):
    """FedNow Service Types."""

    # Core Services
    INSTANT_PAYMENT = ("IPMT", "Instant Payment")
    REQUEST_FOR_PAYMENT = ("RFPY", "Request for Payment")
    PAYMENT_RETURN = ("PRTN", "Payment Return")

    # Liquidity Services
    LIQUIDITY_TRANSFER = ("LQTY", "Liquidity Transfer")
    JOINT_ACCOUNT = ("JACT", "Joint Account Access")

    # Value-Added Services
    ALIAS_SERVICE = ("ALIS", "Alias Directory Service")
    FRAUD_PREVENTION = ("FRPV", "Fraud Prevention")

    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description


class TransactionStatus(Enum):
    """Transaction status codes for FedNow."""

    # Positive Statuses
    ACCP = ("ACCP", "Accepted Customer Profile", "Preceding check passed")
    ACSC = ("ACSC", "Accepted Settlement Completed", "Settlement completed")
    ACSP = ("ACSP", "Accepted Settlement In Process", "Settlement in process")
    ACTC = ("ACTC", "Accepted Technical Validation", "Technical validation passed")
    ACWC = ("ACWC", "Accepted With Change", "Accepted with modification")

    # Pending Statuses
    PDNG = ("PDNG", "Pending", "Payment is pending")
    RCVD = ("RCVD", "Received", "Payment instruction received")

    # Negative Statuses
    RJCT = ("RJCT", "Rejected", "Payment rejected")
    CANC = ("CANC", "Cancelled", "Payment cancelled")

    # Return Related
    RTND = ("RTND", "Returned", "Payment has been returned")
    RTRQ = ("RTRQ", "Return Requested", "Return has been requested")

    def __init__(self, code: str, name: str, description: str):
        self.code = code
        self.status_name = name
        self.description = description

    @classmethod
    def from_code(cls, code: str) -> Optional["TransactionStatus"]:
        """Get status from code."""
        for ts in cls:
            if ts.code == code:
                return ts
        return None

    @property
    def is_final(self) -> bool:
        """Check if this is a final status."""
        return self in [
            TransactionStatus.ACSC,
            TransactionStatus.RJCT,
            TransactionStatus.CANC,
            TransactionStatus.RTND,
        ]

    @property
    def is_positive(self) -> bool:
        """Check if this is a positive status."""
        return self in [
            TransactionStatus.ACCP,
            TransactionStatus.ACSC,
            TransactionStatus.ACSP,
            TransactionStatus.ACTC,
            TransactionStatus.ACWC,
        ]


# FedNow Amount Limits
FEDNOW_MAX_AMOUNT = 500_000_00  # $500,000 in cents (default limit)
FEDNOW_MIN_AMOUNT = 1  # $0.01 in cents

# FedNow Timeout Values (in seconds)
FEDNOW_SEND_TIMEOUT = 20  # 20 seconds for initial response
FEDNOW_SETTLEMENT_TIMEOUT = 60  # 60 seconds for settlement confirmation

# FedNow Operating Parameters
FEDNOW_MESSAGE_ID_MAX_LENGTH = 35
FEDNOW_END_TO_END_ID_MAX_LENGTH = 35
FEDNOW_ROUTING_NUMBER_LENGTH = 9
