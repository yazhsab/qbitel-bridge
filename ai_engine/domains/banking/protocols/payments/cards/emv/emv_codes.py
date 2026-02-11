"""
EMV Protocol Codes and Constants

Defines:
- EMV Tags
- Application identifiers
- Transaction types
- Cryptogram types
- CVM results
- Terminal types
- Bitmap definitions (TVR, TSI, AIP)
"""

from enum import Enum
from typing import Dict, Optional, Tuple


class EmvTag(Enum):
    """Common EMV Tags."""

    # Application Data
    AID = ("9F06", "Application Identifier (AID)")
    APPLICATION_LABEL = ("50", "Application Label")
    APPLICATION_PREFERRED_NAME = ("9F12", "Application Preferred Name")
    APPLICATION_PRIORITY_INDICATOR = ("87", "Application Priority Indicator")
    APPLICATION_VERSION_NUMBER_CARD = ("9F08", "Application Version Number (Card)")
    APPLICATION_VERSION_NUMBER_TERMINAL = ("9F09", "Application Version Number (Terminal)")
    DEDICATED_FILE_NAME = ("84", "Dedicated File (DF) Name")

    # Card Data
    PAN = ("5A", "Application Primary Account Number (PAN)")
    PAN_SEQUENCE_NUMBER = ("5F34", "PAN Sequence Number")
    TRACK_2_EQUIVALENT_DATA = ("57", "Track 2 Equivalent Data")
    CARDHOLDER_NAME = ("5F20", "Cardholder Name")
    EXPIRATION_DATE = ("5F24", "Application Expiration Date")
    EFFECTIVE_DATE = ("5F25", "Application Effective Date")
    SERVICE_CODE = ("5F30", "Service Code")
    ISSUER_COUNTRY_CODE = ("5F28", "Issuer Country Code")
    CARD_RISK_MANAGEMENT_DATA_1 = ("9F50", "Card Risk Management Data 1")

    # Cryptogram Data
    APPLICATION_CRYPTOGRAM = ("9F26", "Application Cryptogram")
    CRYPTOGRAM_INFORMATION_DATA = ("9F27", "Cryptogram Information Data")
    APPLICATION_TRANSACTION_COUNTER = ("9F36", "Application Transaction Counter (ATC)")
    UNPREDICTABLE_NUMBER = ("9F37", "Unpredictable Number")
    ISSUER_APPLICATION_DATA = ("9F10", "Issuer Application Data")

    # Transaction Data
    AMOUNT_AUTHORIZED = ("9F02", "Amount, Authorized (Numeric)")
    AMOUNT_OTHER = ("9F03", "Amount, Other (Numeric)")
    TRANSACTION_CURRENCY_CODE = ("5F2A", "Transaction Currency Code")
    TRANSACTION_DATE = ("9A", "Transaction Date")
    TRANSACTION_TIME = ("9F21", "Transaction Time")
    TRANSACTION_TYPE = ("9C", "Transaction Type")
    TRANSACTION_CATEGORY_CODE = ("9F53", "Transaction Category Code")
    MERCHANT_CATEGORY_CODE = ("9F15", "Merchant Category Code")

    # Terminal Data
    TERMINAL_COUNTRY_CODE = ("9F1A", "Terminal Country Code")
    TERMINAL_IDENTIFICATION = ("9F1C", "Terminal Identification")
    TERMINAL_TYPE = ("9F35", "Terminal Type")
    TERMINAL_CAPABILITIES = ("9F33", "Terminal Capabilities")
    ADDITIONAL_TERMINAL_CAPABILITIES = ("9F40", "Additional Terminal Capabilities")
    TERMINAL_VERIFICATION_RESULTS = ("95", "Terminal Verification Results (TVR)")
    TRANSACTION_STATUS_INFORMATION = ("9B", "Transaction Status Information (TSI)")
    INTERFACE_DEVICE_SERIAL_NUMBER = ("9F1E", "Interface Device (IFD) Serial Number")

    # CVM Data
    CARDHOLDER_VERIFICATION_METHOD_RESULTS = ("9F34", "Cardholder Verification Method (CVM) Results")
    CVM_LIST = ("8E", "Cardholder Verification Method (CVM) List")
    PIN_TRY_COUNTER = ("9F17", "Personal Identification Number (PIN) Try Counter")

    # Risk Management
    APPLICATION_INTERCHANGE_PROFILE = ("82", "Application Interchange Profile (AIP)")
    APPLICATION_FILE_LOCATOR = ("94", "Application File Locator (AFL)")
    ISSUER_AUTHENTICATION_DATA = ("91", "Issuer Authentication Data")
    ISSUER_SCRIPT_TEMPLATE_1 = ("71", "Issuer Script Template 1")
    ISSUER_SCRIPT_TEMPLATE_2 = ("72", "Issuer Script Template 2")
    ISSUER_SCRIPT_RESULTS = ("9F5B", "Issuer Script Results")

    # Processing Options
    PDOL = ("9F38", "Processing Options Data Object List (PDOL)")
    CDOL_1 = ("8C", "Card Risk Management Data Object List 1 (CDOL1)")
    CDOL_2 = ("8D", "Card Risk Management Data Object List 2 (CDOL2)")

    # Authorization Response
    AUTHORIZATION_CODE = ("89", "Authorization Code")
    AUTHORIZATION_RESPONSE_CODE = ("8A", "Authorization Response Code")
    RESPONSE_MESSAGE_TEMPLATE_2 = ("77", "Response Message Template Format 2")

    # Kernel Specific
    KERNEL_IDENTIFIER = ("9F2A", "Kernel Identifier")
    CARD_TRANSACTION_QUALIFIERS = ("9F6C", "Card Transaction Qualifiers (CTQ)")
    TERMINAL_TRANSACTION_QUALIFIERS = ("9F66", "Terminal Transaction Qualifiers (TTQ)")

    # Form Factor Indicator
    FORM_FACTOR_INDICATOR = ("9F6E", "Form Factor Indicator")

    # Additional Tags
    ACQUIRER_IDENTIFIER = ("9F01", "Acquirer Identifier")
    MERCHANT_IDENTIFIER = ("9F16", "Merchant Identifier")
    MERCHANT_NAME_LOCATION = ("9F4E", "Merchant Name and Location")
    POINT_OF_SERVICE_ENTRY_MODE = ("9F39", "Point-of-Service (POS) Entry Mode")
    TERMINAL_FLOOR_LIMIT = ("9F1B", "Terminal Floor Limit")

    def __init__(self, tag: str, description: str):
        self._tag = tag
        self._description = description

    @property
    def tag(self) -> str:
        return self._tag

    @property
    def description(self) -> str:
        return self._description

    @property
    def tag_bytes(self) -> bytes:
        return bytes.fromhex(self._tag)

    @classmethod
    def from_tag(cls, tag: str) -> Optional["EmvTag"]:
        """Get EmvTag from tag value."""
        tag = tag.upper()
        for emv_tag in cls:
            if emv_tag.tag == tag:
                return emv_tag
        return None


class EmvApplication(Enum):
    """Common EMV Application Identifiers (AIDs)."""

    # Visa
    VISA_CREDIT = ("A0000000031010", "Visa Credit/Debit")
    VISA_ELECTRON = ("A0000000032010", "Visa Electron")
    VISA_VPAY = ("A0000000032020", "V Pay")
    VISA_PLUS = ("A0000000038010", "Plus")

    # Mastercard
    MASTERCARD_CREDIT = ("A0000000041010", "Mastercard Credit/Debit")
    MASTERCARD_MAESTRO = ("A0000000043060", "Maestro")
    MASTERCARD_CIRRUS = ("A0000000046000", "Cirrus")

    # American Express
    AMEX = ("A000000025010801", "American Express")

    # Discover
    DISCOVER = ("A0000001523010", "Discover")
    DINERS = ("A0000001524010", "Diners Club")

    # JCB
    JCB = ("A0000000651010", "JCB")

    # UnionPay
    UNIONPAY_DEBIT = ("A000000333010101", "UnionPay Debit")
    UNIONPAY_CREDIT = ("A000000333010102", "UnionPay Credit")

    # Interac (Canada)
    INTERAC = ("A0000002771010", "Interac")

    # RuPay (India)
    RUPAY = ("A0000005241010", "RuPay")

    # eftpos (Australia)
    EFTPOS = ("A0000003845010", "eftpos")

    def __init__(self, aid: str, description: str):
        self._aid = aid
        self._description = description

    @property
    def aid(self) -> str:
        return self._aid

    @property
    def description(self) -> str:
        return self._description

    @property
    def aid_bytes(self) -> bytes:
        return bytes.fromhex(self._aid)

    @classmethod
    def from_aid(cls, aid: str) -> Optional["EmvApplication"]:
        """Get application from AID."""
        aid = aid.upper().replace(" ", "")
        for app in cls:
            if aid.startswith(app.aid) or app.aid.startswith(aid):
                return app
        return None


class EmvTransactionType(Enum):
    """EMV Transaction Types (tag 9C)."""

    PURCHASE = ("00", "Purchase")
    CASH_ADVANCE = ("01", "Cash Advance")
    PURCHASE_CASHBACK = ("09", "Purchase with Cashback")
    REFUND = ("20", "Refund")
    BALANCE_INQUIRY = ("31", "Balance Inquiry")
    TRANSFER = ("40", "Transfer")
    PAYMENT = ("50", "Payment")
    ADMIN = ("60", "Administrative")

    def __init__(self, code: str, description: str):
        self._code = code
        self._description = description

    @property
    def code(self) -> str:
        return self._code

    @property
    def description(self) -> str:
        return self._description

    @classmethod
    def from_code(cls, code: str) -> Optional["EmvTransactionType"]:
        """Get transaction type from code."""
        for tt in cls:
            if tt.code == code:
                return tt
        return None


class EmvCryptogramType(Enum):
    """EMV Cryptogram Types."""

    AAC = ("00", "Application Authentication Cryptogram (Decline)")
    TC = ("40", "Transaction Certificate (Approval)")
    ARQC = ("80", "Authorization Request Cryptogram (Online)")
    AAR = ("C0", "Application Authentication Referral")  # RFU

    def __init__(self, code: str, description: str):
        self._code = code
        self._description = description

    @property
    def code(self) -> str:
        return self._code

    @property
    def code_int(self) -> int:
        return int(self._code, 16)

    @property
    def description(self) -> str:
        return self._description

    @classmethod
    def from_cid(cls, cid: str) -> Optional["EmvCryptogramType"]:
        """Get cryptogram type from Cryptogram Information Data (tag 9F27)."""
        # CID byte: bits 7-6 indicate cryptogram type
        cid_int = int(cid, 16) if isinstance(cid, str) else cid
        type_bits = cid_int & 0xC0

        for ct in cls:
            if ct.code_int == type_bits:
                return ct
        return None


class EmvCvmResult(Enum):
    """EMV Cardholder Verification Method Results."""

    NO_CVM = ("1F", "00", "No CVM Performed")
    SIGNATURE = ("1E", "00", "Signature")
    ONLINE_PIN = ("02", "00", "Online PIN")
    OFFLINE_PLAINTEXT_PIN = ("01", "00", "Offline Plaintext PIN")
    OFFLINE_ENCIPHERED_PIN = ("04", "00", "Offline Enciphered PIN")
    OFFLINE_PLAINTEXT_PIN_SIGNATURE = ("03", "00", "Offline Plaintext PIN + Signature")
    OFFLINE_ENCIPHERED_PIN_SIGNATURE = ("05", "00", "Offline Enciphered PIN + Signature")
    CDCVM = ("1F", "01", "Consumer Device CVM (CDCVM)")
    FAILED = ("3F", "01", "CVM Failed")

    def __init__(self, method: str, condition: str, description: str):
        self._method = method
        self._condition = condition
        self._description = description

    @property
    def method(self) -> str:
        return self._method

    @property
    def condition(self) -> str:
        return self._condition

    @property
    def description(self) -> str:
        return self._description

    @property
    def cvm_results(self) -> str:
        """Get 3-byte CVM Results value."""
        return f"{self._method}{self._condition}02"  # 02 = CVM successful


class EmvTerminalType(Enum):
    """EMV Terminal Types (tag 9F35)."""

    FINANCIAL_ATTENDED_ONLINE = ("11", "Financial Institution, Attended, Online Only")
    FINANCIAL_ATTENDED_OFFLINE = ("12", "Financial Institution, Attended, Offline with Online")
    FINANCIAL_ATTENDED_OFFLINE_ONLY = ("13", "Financial Institution, Attended, Offline Only")
    FINANCIAL_UNATTENDED_ONLINE = ("14", "Financial Institution, Unattended, Online Only")
    FINANCIAL_UNATTENDED_OFFLINE = ("15", "Financial Institution, Unattended, Offline with Online")
    FINANCIAL_UNATTENDED_OFFLINE_ONLY = ("16", "Financial Institution, Unattended, Offline Only")
    MERCHANT_ATTENDED_ONLINE = ("21", "Merchant, Attended, Online Only")
    MERCHANT_ATTENDED_OFFLINE = ("22", "Merchant, Attended, Offline with Online")
    MERCHANT_ATTENDED_OFFLINE_ONLY = ("23", "Merchant, Attended, Offline Only")
    MERCHANT_UNATTENDED_ONLINE = ("24", "Merchant, Unattended, Online Only")
    MERCHANT_UNATTENDED_OFFLINE = ("25", "Merchant, Unattended, Offline with Online")
    MERCHANT_UNATTENDED_OFFLINE_ONLY = ("26", "Merchant, Unattended, Offline Only")

    def __init__(self, code: str, description: str):
        self._code = code
        self._description = description

    @property
    def code(self) -> str:
        return self._code

    @property
    def description(self) -> str:
        return self._description


class EmvPosEntryMode(Enum):
    """POS Entry Modes (tag 9F39)."""

    MANUAL = ("01", "Manual entry")
    MAG_STRIPE = ("02", "Magnetic stripe")
    ICC_CVV = ("05", "ICC (chip) with CVV reliability check")
    CONTACTLESS_MAG = ("07", "Contactless using magnetic stripe data")
    CONTACTLESS_ICC = ("07", "Contactless using EMV chip data")
    ICC = ("05", "ICC (chip)")
    FALLBACK_MAG = ("80", "Fallback to magnetic stripe")
    ECOMMERCE = ("81", "E-commerce")
    CREDENTIALS_ON_FILE = ("10", "Credentials on file")

    def __init__(self, code: str, description: str):
        self._code = code
        self._description = description

    @property
    def code(self) -> str:
        return self._code

    @property
    def description(self) -> str:
        return self._description


# Terminal Verification Results (TVR) - 5 bytes
TVR_BITS: Dict[Tuple[int, int], str] = {
    # Byte 1
    (1, 7): "Offline data authentication was not performed",
    (1, 6): "SDA failed",
    (1, 5): "ICC data missing",
    (1, 4): "Card appears on terminal exception file",
    (1, 3): "DDA failed",
    (1, 2): "CDA failed",
    (1, 1): "SDA selected",
    (1, 0): "RFU",
    # Byte 2
    (2, 7): "ICC and terminal have different application versions",
    (2, 6): "Expired application",
    (2, 5): "Application not yet effective",
    (2, 4): "Requested service not allowed for card product",
    (2, 3): "New card",
    (2, 2): "RFU",
    (2, 1): "RFU",
    (2, 0): "RFU",
    # Byte 3
    (3, 7): "Cardholder verification was not successful",
    (3, 6): "Unrecognised CVM",
    (3, 5): "PIN Try Limit exceeded",
    (3, 4): "PIN entry required and PIN pad not present or not working",
    (3, 3): "PIN entry required, PIN pad present, but PIN was not entered",
    (3, 2): "Online PIN entered",
    (3, 1): "RFU",
    (3, 0): "RFU",
    # Byte 4
    (4, 7): "Transaction exceeds floor limit",
    (4, 6): "Lower consecutive offline limit exceeded",
    (4, 5): "Upper consecutive offline limit exceeded",
    (4, 4): "Transaction selected randomly for online processing",
    (4, 3): "Merchant forced transaction online",
    (4, 2): "RFU",
    (4, 1): "RFU",
    (4, 0): "RFU",
    # Byte 5
    (5, 7): "Default TDOL used",
    (5, 6): "Issuer authentication failed",
    (5, 5): "Script processing failed before final GENERATE AC",
    (5, 4): "Script processing failed after final GENERATE AC",
    (5, 3): "RFU",
    (5, 2): "RFU",
    (5, 1): "RFU",
    (5, 0): "RFU",
}

# Transaction Status Information (TSI) - 2 bytes
TSI_BITS: Dict[Tuple[int, int], str] = {
    # Byte 1
    (1, 7): "Offline data authentication was performed",
    (1, 6): "Cardholder verification was performed",
    (1, 5): "Card risk management was performed",
    (1, 4): "Issuer authentication was performed",
    (1, 3): "Terminal risk management was performed",
    (1, 2): "Script processing was performed",
    (1, 1): "RFU",
    (1, 0): "RFU",
    # Byte 2 (all RFU)
    (2, 7): "RFU",
    (2, 6): "RFU",
    (2, 5): "RFU",
    (2, 4): "RFU",
    (2, 3): "RFU",
    (2, 2): "RFU",
    (2, 1): "RFU",
    (2, 0): "RFU",
}

# Application Interchange Profile (AIP) - 2 bytes
AIP_BITS: Dict[Tuple[int, int], str] = {
    # Byte 1
    (1, 7): "RFU",
    (1, 6): "SDA supported",
    (1, 5): "DDA supported",
    (1, 4): "Cardholder verification is supported",
    (1, 3): "Terminal risk management is to be performed",
    (1, 2): "Issuer authentication is supported",
    (1, 1): "RFU",
    (1, 0): "CDA supported",
    # Byte 2
    (2, 7): "RFU",
    (2, 6): "RFU",
    (2, 5): "RFU",
    (2, 4): "RFU",
    (2, 3): "RFU",
    (2, 2): "RFU",
    (2, 1): "RFU",
    (2, 0): "RFU",
}


# Tag name lookup
EMV_TAG_NAMES: Dict[str, str] = {tag.tag: tag.description for tag in EmvTag}


def get_tag_name(tag: str) -> str:
    """Get the name of an EMV tag."""
    tag = tag.upper()
    return EMV_TAG_NAMES.get(tag, f"Unknown({tag})")


# Tag format lookup (n=numeric, an=alphanumeric, b=binary, cn=compressed numeric)
EMV_TAG_FORMATS: Dict[str, str] = {
    "5A": "cn",  # PAN
    "57": "cn",  # Track 2
    "5F20": "ans",  # Cardholder Name
    "5F24": "n",  # Expiration Date
    "5F25": "n",  # Effective Date
    "5F28": "n",  # Issuer Country Code
    "5F2A": "n",  # Transaction Currency Code
    "5F34": "n",  # PAN Sequence Number
    "82": "b",  # AIP
    "84": "b",  # DF Name
    "87": "b",  # Application Priority Indicator
    "8E": "b",  # CVM List
    "91": "b",  # Issuer Authentication Data
    "95": "b",  # TVR
    "9A": "n",  # Transaction Date
    "9B": "b",  # TSI
    "9C": "n",  # Transaction Type
    "9F02": "n",  # Amount Authorized
    "9F03": "n",  # Amount Other
    "9F06": "b",  # AID
    "9F10": "b",  # IAD
    "9F1A": "n",  # Terminal Country Code
    "9F1B": "b",  # Terminal Floor Limit
    "9F21": "n",  # Transaction Time
    "9F26": "b",  # Application Cryptogram
    "9F27": "b",  # Cryptogram Information Data
    "9F33": "b",  # Terminal Capabilities
    "9F34": "b",  # CVM Results
    "9F35": "n",  # Terminal Type
    "9F36": "b",  # ATC
    "9F37": "b",  # Unpredictable Number
}


def get_tag_format(tag: str) -> str:
    """Get the format of an EMV tag."""
    tag = tag.upper()
    return EMV_TAG_FORMATS.get(tag, "b")  # Default to binary
