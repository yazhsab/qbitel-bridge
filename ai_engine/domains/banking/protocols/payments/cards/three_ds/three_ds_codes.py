"""
3D Secure Protocol Codes and Constants

Defines:
- Protocol versions
- Message types
- Transaction status values
- Challenge indicators
- Device channels
- Authentication types
- ECI values
"""

from enum import Enum
from typing import Dict, Optional, Tuple


class ThreeDSVersion(Enum):
    """3D Secure Protocol Versions."""

    V1_0 = ("1.0.0", "3DS 1.0")
    V1_0_2 = ("1.0.2", "3DS 1.0.2")
    V2_1_0 = ("2.1.0", "3DS 2.1")
    V2_2_0 = ("2.2.0", "3DS 2.2")
    V2_3_0 = ("2.3.0", "3DS 2.3")

    def __init__(self, version: str, description: str):
        self._version = version
        self._description = description

    @property
    def version(self) -> str:
        return self._version

    @property
    def description(self) -> str:
        return self._description

    @property
    def is_v1(self) -> bool:
        return self._version.startswith("1.")

    @property
    def is_v2(self) -> bool:
        return self._version.startswith("2.")

    @classmethod
    def from_string(cls, version: str) -> Optional["ThreeDSVersion"]:
        """Get version from string."""
        for v in cls:
            if v.version == version:
                return v
        return None


class ThreeDSMessageType(Enum):
    """3D Secure Message Types."""

    # Version 2.x messages
    AREQ = ("AReq", "Authentication Request")
    ARES = ("ARes", "Authentication Response")
    CREQ = ("CReq", "Challenge Request")
    CRES = ("CRes", "Challenge Response")
    RREQ = ("RReq", "Results Request")
    RRES = ("RRes", "Results Response")
    PREQ = ("PReq", "Preparation Request")
    PRES = ("PRes", "Preparation Response")
    ERRO = ("Erro", "Error Message")

    # Version 1.x messages
    PAREQ = ("PAReq", "Payer Authentication Request (1.x)")
    PARES = ("PARes", "Payer Authentication Response (1.x)")
    VEREQ = ("VEReq", "Verify Enrollment Request (1.x)")
    VERES = ("VERes", "Verify Enrollment Response (1.x)")

    def __init__(self, code: str, description: str):
        self._code = code
        self._description = description

    @property
    def code(self) -> str:
        return self._code

    @property
    def description(self) -> str:
        return self._description


class ThreeDSTransactionStatus(Enum):
    """3D Secure Transaction Status Values."""

    # Successful statuses
    Y = ("Y", "Authentication/Verification Successful")
    A = ("A", "Attempts Processing Performed")
    C = ("C", "Challenge Required")
    D = ("D", "Challenge Required; Decoupled Authentication")
    I = ("I", "Informational Only")

    # Unsuccessful statuses
    N = ("N", "Not Authenticated/Account Not Verified")
    U = ("U", "Authentication/Account Verification Could Not Be Performed")
    R = ("R", "Authentication/Account Verification Rejected")

    def __init__(self, code: str, description: str):
        self._code = code
        self._description = description

    @property
    def code(self) -> str:
        return self._code

    @property
    def description(self) -> str:
        return self._description

    @property
    def is_successful(self) -> bool:
        """Check if authentication was successful."""
        return self._code in ("Y", "A")

    @property
    def requires_challenge(self) -> bool:
        """Check if challenge is required."""
        return self._code in ("C", "D")

    @property
    def is_frictionless(self) -> bool:
        """Check if frictionless flow completed."""
        return self._code in ("Y", "A", "N", "U", "R", "I")

    @classmethod
    def from_code(cls, code: str) -> Optional["ThreeDSTransactionStatus"]:
        """Get status from code."""
        for status in cls:
            if status.code == code:
                return status
        return None


class ThreeDSChallengeIndicator(Enum):
    """3D Secure Challenge Preference Indicator."""

    NO_PREFERENCE = ("01", "No preference")
    NO_CHALLENGE_REQUESTED = ("02", "No challenge requested")
    CHALLENGE_REQUESTED_3DS_PREFERENCE = ("03", "Challenge requested: 3DS Requestor preference")
    CHALLENGE_REQUESTED_MANDATE = ("04", "Challenge requested: Mandate")
    NO_CHALLENGE_RISK_ANALYSIS = ("05", "No challenge requested (transactional risk analysis)")
    NO_CHALLENGE_DATA_SHARE_ONLY = ("06", "No challenge requested (Data share only)")
    NO_CHALLENGE_SCA_ALREADY_PERFORMED = ("07", "No challenge requested (SCA already performed)")
    NO_CHALLENGE_WHITELIST = ("08", "No challenge requested (whitelist exemption)")
    CHALLENGE_WHITELIST_PROMPT = ("09", "Challenge requested (whitelist prompt requested)")

    def __init__(self, code: str, description: str):
        self._code = code
        self._description = description

    @property
    def code(self) -> str:
        return self._code

    @property
    def description(self) -> str:
        return self._description


class ThreeDSDeviceChannel(Enum):
    """3D Secure Device Channels."""

    APP = ("01", "App-based")
    BROWSER = ("02", "Browser")
    THREE_DS_REQUESTOR = ("03", "3DS Requestor Initiated (3RI)")

    def __init__(self, code: str, description: str):
        self._code = code
        self._description = description

    @property
    def code(self) -> str:
        return self._code

    @property
    def description(self) -> str:
        return self._description


class ThreeDSAuthenticationType(Enum):
    """3D Secure Authentication Types."""

    STATIC_PASSCODE = ("01", "Static Passcode")
    SMS_OTP = ("02", "SMS OTP")
    KEY_FOB = ("03", "Key fob or EMV card reader OTP")
    APP_OTP = ("04", "App-based OTP")
    APP_LOGIN = ("05", "OTP via App Login")
    APP_RBA = ("06", "App RBA")
    BIOMETRIC = ("07", "Biometric")

    def __init__(self, code: str, description: str):
        self._code = code
        self._description = description

    @property
    def code(self) -> str:
        return self._code

    @property
    def description(self) -> str:
        return self._description


class ThreeDSMessageCategory(Enum):
    """3D Secure Message Categories."""

    PAYMENT = ("01", "Payment Authentication")
    NON_PAYMENT = ("02", "Non-Payment Authentication")

    def __init__(self, code: str, description: str):
        self._code = code
        self._description = description

    @property
    def code(self) -> str:
        return self._code

    @property
    def description(self) -> str:
        return self._description


class ThreeDSACSRenderingType(Enum):
    """ACS Rendering Types for Challenge."""

    TEXT = ("01", "Text")
    SINGLE_SELECT = ("02", "Single Select")
    MULTI_SELECT = ("03", "Multi Select")
    OOB = ("04", "Out-of-Band")
    HTML_OTHER = ("05", "HTML Other")

    def __init__(self, code: str, description: str):
        self._code = code
        self._description = description

    @property
    def code(self) -> str:
        return self._code

    @property
    def description(self) -> str:
        return self._description


class ThreeDSChallengeWindowSize(Enum):
    """Challenge Window Sizes."""

    SIZE_250X400 = ("01", "250 x 400")
    SIZE_390X400 = ("02", "390 x 400")
    SIZE_500X600 = ("03", "500 x 600")
    SIZE_600X400 = ("04", "600 x 400")
    SIZE_FULL_SCREEN = ("05", "Full screen")

    def __init__(self, code: str, description: str):
        self._code = code
        self._description = description

    @property
    def code(self) -> str:
        return self._code

    @property
    def description(self) -> str:
        return self._description


# ECI (Electronic Commerce Indicator) Values
ECI_VALUES: Dict[str, Dict[str, str]] = {
    "visa": {
        "05": "Fully authenticated (3DS 2.x)",
        "06": "Attempted authentication (3DS 2.x)",
        "07": "Non-3DS transaction or authentication failed",
    },
    "mastercard": {
        "02": "Fully authenticated (3DS 2.x)",
        "01": "Attempted authentication (3DS 2.x)",
        "00": "Non-3DS transaction or authentication failed",
    },
    "amex": {
        "05": "Fully authenticated",
        "06": "Attempted authentication",
        "07": "Non-3DS transaction",
    },
    "discover": {
        "05": "Fully authenticated",
        "06": "Attempted authentication",
        "07": "Non-3DS transaction",
    },
}


def get_eci_description(eci: str, card_brand: str = "visa") -> str:
    """
    Get ECI value description.

    Args:
        eci: ECI value
        card_brand: Card brand (visa, mastercard, amex, discover)

    Returns:
        Description of the ECI value
    """
    brand = card_brand.lower()
    if brand in ECI_VALUES:
        return ECI_VALUES[brand].get(eci, f"Unknown ECI: {eci}")
    return f"Unknown ECI: {eci}"


# Authentication Value types
AUTHENTICATION_VALUE_TYPES: Dict[str, str] = {
    "CAVV": "Cardholder Authentication Verification Value (Visa)",
    "AAV": "Accountholder Authentication Value (Mastercard)",
    "AEVV": "American Express Verification Value",
}

# Error codes
THREE_DS_ERROR_CODES: Dict[str, str] = {
    "101": "Message received invalid",
    "102": "Message version number not supported",
    "103": "Message could not be processed",
    "201": "Required data element missing",
    "202": "Format of data element invalid",
    "203": "Duplicate data element",
    "204": "Message too large",
    "301": "Transaction ID not recognized",
    "302": "Data decryption failure",
    "303": "Access denied",
    "304": "ISO code invalid",
    "305": "Transaction data not valid",
    "306": "Merchant category code not valid for payment system",
    "307": "Serial number not valid",
    "402": "Transaction timed out",
    "403": "Suspected fraud",
    "404": "ACS technical issue",
    "405": "DS technical issue",
}
