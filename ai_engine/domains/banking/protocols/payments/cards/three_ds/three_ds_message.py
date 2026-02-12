"""
3D Secure Message Data Structures

Data structures for 3D Secure authentication including:
- Authentication request/response
- Challenge flow messages
- Device and browser information
- Merchant and cardholder information
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
import uuid
import json
import base64


from ai_engine.domains.banking.protocols.payments.cards.three_ds.three_ds_codes import (
    ThreeDSVersion,
    ThreeDSMessageType,
    ThreeDSTransactionStatus,
    ThreeDSChallengeIndicator,
    ThreeDSDeviceChannel,
    ThreeDSAuthenticationType,
    ThreeDSMessageCategory,
)


@dataclass
class BrowserInfo:
    """Browser Information for 3DS 2.x."""

    # Required fields
    accept_header: str = "*/*"
    java_enabled: bool = False
    javascript_enabled: bool = True
    language: str = "en-US"
    color_depth: int = 24
    screen_height: int = 1080
    screen_width: int = 1920
    time_zone: int = 0  # Offset in minutes from UTC
    user_agent: str = ""

    # Optional
    ip_address: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API request."""
        return {
            "browserAcceptHeader": self.accept_header,
            "browserJavaEnabled": self.java_enabled,
            "browserJavascriptEnabled": self.javascript_enabled,
            "browserLanguage": self.language,
            "browserColorDepth": str(self.color_depth),
            "browserScreenHeight": str(self.screen_height),
            "browserScreenWidth": str(self.screen_width),
            "browserTZ": str(self.time_zone),
            "browserUserAgent": self.user_agent,
            "browserIP": self.ip_address,
        }


@dataclass
class DeviceInfo:
    """Device Information for 3DS 2.x App-based."""

    # SDK Info
    sdk_app_id: str = ""
    sdk_enc_data: str = ""
    sdk_ephem_pub_key: Dict[str, str] = field(default_factory=dict)
    sdk_max_timeout: int = 5  # Minutes
    sdk_reference_number: str = ""
    sdk_trans_id: str = ""

    # Device data
    device_render_options: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API request."""
        return {
            "sdkAppID": self.sdk_app_id,
            "sdkEncData": self.sdk_enc_data,
            "sdkEphemPubKey": self.sdk_ephem_pub_key,
            "sdkMaxTimeout": str(self.sdk_max_timeout).zfill(2),
            "sdkReferenceNumber": self.sdk_reference_number,
            "sdkTransID": self.sdk_trans_id,
            "deviceRenderOptions": self.device_render_options,
        }


@dataclass
class MerchantInfo:
    """Merchant Information."""

    # Required
    acquirer_bin: str = ""
    acquirer_merchant_id: str = ""
    merchant_name: str = ""
    merchant_country_code: str = ""  # ISO 3166-1 numeric
    mcc: str = ""  # Merchant Category Code

    # Optional
    merchant_url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API request."""
        result = {
            "acquirerBIN": self.acquirer_bin,
            "acquirerMerchantID": self.acquirer_merchant_id,
            "merchantName": self.merchant_name,
            "merchantCountryCode": self.merchant_country_code,
            "mcc": self.mcc,
        }
        if self.merchant_url:
            result["merchantURL"] = self.merchant_url
        return result


@dataclass
class CardholderInfo:
    """Cardholder Information."""

    # Account info
    cardholder_name: str = ""
    email: str = ""
    phone_country: str = ""
    phone_number: str = ""

    # Address
    billing_address_line1: str = ""
    billing_address_line2: str = ""
    billing_city: str = ""
    billing_state: str = ""
    billing_postal_code: str = ""
    billing_country_code: str = ""  # ISO 3166-1 numeric

    # Shipping address (if different)
    shipping_address_line1: str = ""
    shipping_address_line2: str = ""
    shipping_city: str = ""
    shipping_state: str = ""
    shipping_postal_code: str = ""
    shipping_country_code: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API request."""
        result = {}

        if self.cardholder_name:
            result["cardholderName"] = self.cardholder_name
        if self.email:
            result["email"] = self.email
        if self.phone_country and self.phone_number:
            result["homePhone"] = {
                "cc": self.phone_country,
                "subscriber": self.phone_number,
            }

        # Billing address
        if self.billing_address_line1:
            result["billAddrLine1"] = self.billing_address_line1
        if self.billing_address_line2:
            result["billAddrLine2"] = self.billing_address_line2
        if self.billing_city:
            result["billAddrCity"] = self.billing_city
        if self.billing_state:
            result["billAddrState"] = self.billing_state
        if self.billing_postal_code:
            result["billAddrPostCode"] = self.billing_postal_code
        if self.billing_country_code:
            result["billAddrCountry"] = self.billing_country_code

        # Shipping address
        if self.shipping_address_line1:
            result["shipAddrLine1"] = self.shipping_address_line1
        if self.shipping_address_line2:
            result["shipAddrLine2"] = self.shipping_address_line2
        if self.shipping_city:
            result["shipAddrCity"] = self.shipping_city
        if self.shipping_state:
            result["shipAddrState"] = self.shipping_state
        if self.shipping_postal_code:
            result["shipAddrPostCode"] = self.shipping_postal_code
        if self.shipping_country_code:
            result["shipAddrCountry"] = self.shipping_country_code

        return result


@dataclass
class ThreeDSAuthRequest:
    """
    3D Secure Authentication Request (AReq).

    Used to initiate authentication with the ACS.
    """

    # Message identification
    threeds_server_trans_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ds_trans_id: str = ""
    acs_trans_id: str = ""

    # Version
    message_version: str = "2.2.0"

    # Transaction
    device_channel: str = "02"  # Browser
    message_category: str = "01"  # Payment
    purchase_amount: int = 0  # Minor units
    purchase_currency: str = "840"  # ISO 4217 numeric
    purchase_date: str = ""  # YYYYMMDDHHMMSS
    purchase_exponent: str = "2"

    # Card data
    acct_number: str = ""  # PAN
    card_expiry_date: str = ""  # YYMM

    # Merchant
    merchant_info: MerchantInfo = field(default_factory=MerchantInfo)

    # Cardholder
    cardholder_info: CardholderInfo = field(default_factory=CardholderInfo)

    # Device/Browser
    browser_info: Optional[BrowserInfo] = None
    device_info: Optional[DeviceInfo] = None

    # Challenge preference
    three_ds_requestor_challenge_ind: str = "01"  # No preference

    # URLs
    notification_url: str = ""  # Where to send challenge result

    # Risk indicators
    acct_type: str = ""  # Account type
    addr_match: str = ""  # Address match indicator
    purchase_instal_data: str = ""  # Installment data
    recurring_expiry: str = ""  # Recurring expiry date
    recurring_frequency: str = ""  # Recurring frequency

    @property
    def version(self) -> Optional[ThreeDSVersion]:
        """Get protocol version."""
        return ThreeDSVersion.from_string(self.message_version)

    @property
    def channel(self) -> Optional[ThreeDSDeviceChannel]:
        """Get device channel enum."""
        for ch in ThreeDSDeviceChannel:
            if ch.code == self.device_channel:
                return ch
        return None

    @property
    def amount_decimal(self) -> Decimal:
        """Get amount as decimal."""
        exp = int(self.purchase_exponent)
        return Decimal(self.purchase_amount) / (10**exp)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API request."""
        result = {
            "messageType": "AReq",
            "messageVersion": self.message_version,
            "threeDSServerTransID": self.threeds_server_trans_id,
            "deviceChannel": self.device_channel,
            "messageCategory": self.message_category,
            "acctNumber": self.acct_number,
            "cardExpiryDate": self.card_expiry_date,
            "purchaseAmount": str(self.purchase_amount),
            "purchaseCurrency": self.purchase_currency,
            "purchaseExponent": self.purchase_exponent,
            "purchaseDate": self.purchase_date or datetime.utcnow().strftime("%Y%m%d%H%M%S"),
            "threeDSRequestorChallengeInd": self.three_ds_requestor_challenge_ind,
        }

        if self.notification_url:
            result["notificationURL"] = self.notification_url

        # Add merchant info
        result.update(self.merchant_info.to_dict())

        # Add cardholder info
        result.update(self.cardholder_info.to_dict())

        # Add browser or device info
        if self.device_channel == "02" and self.browser_info:
            result.update(self.browser_info.to_dict())
        elif self.device_channel == "01" and self.device_info:
            result.update(self.device_info.to_dict())

        return result

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


@dataclass
class ThreeDSAuthResponse:
    """
    3D Secure Authentication Response (ARes).

    Response from ACS after authentication request.
    """

    # Message identification
    threeds_server_trans_id: str = ""
    ds_trans_id: str = ""
    acs_trans_id: str = ""

    # Version
    message_version: str = "2.2.0"

    # Transaction status
    trans_status: str = ""  # Y, N, U, A, C, D, R, I

    # Authentication value (if authenticated)
    authentication_value: str = ""  # CAVV/AAV
    eci: str = ""  # Electronic Commerce Indicator

    # Challenge data (if challenge required)
    acs_url: str = ""
    acs_challenge_mandated: str = ""  # Y/N
    challenge_req: str = ""  # Encoded challenge request

    # Error data
    acs_operator_id: str = ""
    ds_reference_number: str = ""

    # Additional data
    trans_status_reason: str = ""
    authentication_type: str = ""
    white_list_status: str = ""
    white_list_status_source: str = ""

    @property
    def status(self) -> Optional[ThreeDSTransactionStatus]:
        """Get transaction status enum."""
        return ThreeDSTransactionStatus.from_code(self.trans_status)

    @property
    def is_authenticated(self) -> bool:
        """Check if authentication was successful."""
        return self.trans_status in ("Y", "A")

    @property
    def requires_challenge(self) -> bool:
        """Check if challenge is required."""
        return self.trans_status in ("C", "D")

    @property
    def is_frictionless(self) -> bool:
        """Check if frictionless flow completed."""
        return self.trans_status in ("Y", "A", "N", "U", "R", "I")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThreeDSAuthResponse":
        """Create from dictionary."""
        return cls(
            threeds_server_trans_id=data.get("threeDSServerTransID", ""),
            ds_trans_id=data.get("dsTransID", ""),
            acs_trans_id=data.get("acsTransID", ""),
            message_version=data.get("messageVersion", "2.2.0"),
            trans_status=data.get("transStatus", ""),
            authentication_value=data.get("authenticationValue", ""),
            eci=data.get("eci", ""),
            acs_url=data.get("acsURL", ""),
            acs_challenge_mandated=data.get("acsChallengeMandated", ""),
            trans_status_reason=data.get("transStatusReason", ""),
            authentication_type=data.get("authenticationType", ""),
            white_list_status=data.get("whiteListStatus", ""),
            white_list_status_source=data.get("whiteListStatusSource", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "messageType": "ARes",
            "messageVersion": self.message_version,
            "threeDSServerTransID": self.threeds_server_trans_id,
            "dsTransID": self.ds_trans_id,
            "acsTransID": self.acs_trans_id,
            "transStatus": self.trans_status,
        }

        if self.authentication_value:
            result["authenticationValue"] = self.authentication_value
        if self.eci:
            result["eci"] = self.eci
        if self.acs_url:
            result["acsURL"] = self.acs_url
        if self.trans_status_reason:
            result["transStatusReason"] = self.trans_status_reason

        return result


@dataclass
class ThreeDSChallengeRequest:
    """
    3D Secure Challenge Request (CReq).

    Sent by 3DS Requestor to ACS when challenge is required.
    """

    # Message identification
    threeds_server_trans_id: str = ""
    acs_trans_id: str = ""

    # Version
    message_version: str = "2.2.0"

    # Challenge
    challenge_window_size: str = "05"  # Full screen

    # SDK data (for app-based)
    sdk_trans_id: str = ""
    sdk_counter_s_to_a: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "messageType": "CReq",
            "messageVersion": self.message_version,
            "threeDSServerTransID": self.threeds_server_trans_id,
            "acsTransID": self.acs_trans_id,
            "challengeWindowSize": self.challenge_window_size,
        }

        if self.sdk_trans_id:
            result["sdkTransID"] = self.sdk_trans_id
        if self.sdk_counter_s_to_a:
            result["sdkCounterStoA"] = self.sdk_counter_s_to_a

        return result

    def to_base64(self) -> str:
        """Convert to base64-encoded JSON for form post."""
        json_str = json.dumps(self.to_dict())
        return base64.b64encode(json_str.encode()).decode()


@dataclass
class ThreeDSChallengeResponse:
    """
    3D Secure Challenge Response (CRes).

    Response from ACS after challenge completion.
    """

    # Message identification
    threeds_server_trans_id: str = ""
    acs_trans_id: str = ""

    # Version
    message_version: str = "2.2.0"

    # Transaction status
    trans_status: str = ""

    # Authentication result
    authentication_value: str = ""
    eci: str = ""

    # Challenge data
    challenge_completion_ind: str = ""  # Y = completed

    # SDK data
    acs_counter_a_to_s: str = ""
    sdk_trans_id: str = ""

    @property
    def status(self) -> Optional[ThreeDSTransactionStatus]:
        """Get transaction status enum."""
        return ThreeDSTransactionStatus.from_code(self.trans_status)

    @property
    def is_completed(self) -> bool:
        """Check if challenge was completed."""
        return self.challenge_completion_ind == "Y"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThreeDSChallengeResponse":
        """Create from dictionary."""
        return cls(
            threeds_server_trans_id=data.get("threeDSServerTransID", ""),
            acs_trans_id=data.get("acsTransID", ""),
            message_version=data.get("messageVersion", "2.2.0"),
            trans_status=data.get("transStatus", ""),
            authentication_value=data.get("authenticationValue", ""),
            eci=data.get("eci", ""),
            challenge_completion_ind=data.get("challengeCompletionInd", ""),
        )

    @classmethod
    def from_base64(cls, encoded: str) -> "ThreeDSChallengeResponse":
        """Create from base64-encoded JSON."""
        json_str = base64.b64decode(encoded).decode()
        data = json.loads(json_str)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "messageType": "CRes",
            "messageVersion": self.message_version,
            "threeDSServerTransID": self.threeds_server_trans_id,
            "acsTransID": self.acs_trans_id,
            "transStatus": self.trans_status,
            "authenticationValue": self.authentication_value,
            "eci": self.eci,
            "challengeCompletionInd": self.challenge_completion_ind,
        }


@dataclass
class ThreeDSResultRequest:
    """
    3D Secure Results Request (RReq).

    Sent by ACS to DS/3DS Server with final authentication result.
    """

    # Message identification
    threeds_server_trans_id: str = ""
    ds_trans_id: str = ""
    acs_trans_id: str = ""

    # Version
    message_version: str = "2.2.0"

    # Result
    trans_status: str = ""
    authentication_value: str = ""
    eci: str = ""
    authentication_type: str = ""

    # Challenge result
    challenge_cancel: str = ""  # Reason if cancelled
    interaction_counter: str = ""

    @property
    def status(self) -> Optional[ThreeDSTransactionStatus]:
        """Get transaction status enum."""
        return ThreeDSTransactionStatus.from_code(self.trans_status)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "messageType": "RReq",
            "messageVersion": self.message_version,
            "threeDSServerTransID": self.threeds_server_trans_id,
            "dsTransID": self.ds_trans_id,
            "acsTransID": self.acs_trans_id,
            "transStatus": self.trans_status,
        }

        if self.authentication_value:
            result["authenticationValue"] = self.authentication_value
        if self.eci:
            result["eci"] = self.eci
        if self.authentication_type:
            result["authenticationType"] = self.authentication_type
        if self.challenge_cancel:
            result["challengeCancel"] = self.challenge_cancel
        if self.interaction_counter:
            result["interactionCounter"] = self.interaction_counter

        return result


@dataclass
class ThreeDSResult:
    """
    Complete 3D Secure Authentication Result.

    Aggregates all authentication data for authorization.
    """

    # Protocol info
    version: str = "2.2.0"
    ds_trans_id: str = ""
    acs_trans_id: str = ""
    threeds_server_trans_id: str = ""

    # Authentication result
    trans_status: str = ""
    authentication_value: str = ""  # CAVV/AAV
    eci: str = ""
    authentication_type: str = ""

    # Timestamps
    authentication_timestamp: str = ""

    # Flow info
    was_challenged: bool = False
    challenge_completion_ind: str = ""

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def status(self) -> Optional[ThreeDSTransactionStatus]:
        """Get transaction status enum."""
        return ThreeDSTransactionStatus.from_code(self.trans_status)

    @property
    def is_authenticated(self) -> bool:
        """Check if authentication was successful."""
        return self.trans_status in ("Y", "A")

    @property
    def is_liability_shift(self) -> bool:
        """Check if liability shift applies."""
        # Liability shifts to issuer for successful auth or attempted auth
        return self.trans_status in ("Y", "A")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/transmission."""
        return {
            "version": self.version,
            "dsTransID": self.ds_trans_id,
            "acsTransID": self.acs_trans_id,
            "threeDSServerTransID": self.threeds_server_trans_id,
            "transStatus": self.trans_status,
            "transStatusDescription": self.status.description if self.status else None,
            "authenticationValue": self.authentication_value,
            "eci": self.eci,
            "authenticationType": self.authentication_type,
            "wasChallenged": self.was_challenged,
            "isAuthenticated": self.is_authenticated,
            "isLiabilityShift": self.is_liability_shift,
            "createdAt": self.created_at.isoformat(),
        }

    @classmethod
    def from_auth_response(cls, auth_response: ThreeDSAuthResponse) -> "ThreeDSResult":
        """Create from authentication response (frictionless flow)."""
        return cls(
            version=auth_response.message_version,
            ds_trans_id=auth_response.ds_trans_id,
            acs_trans_id=auth_response.acs_trans_id,
            threeds_server_trans_id=auth_response.threeds_server_trans_id,
            trans_status=auth_response.trans_status,
            authentication_value=auth_response.authentication_value,
            eci=auth_response.eci,
            authentication_type=auth_response.authentication_type,
            was_challenged=False,
        )

    @classmethod
    def from_challenge_response(
        cls,
        auth_response: ThreeDSAuthResponse,
        challenge_response: ThreeDSChallengeResponse,
    ) -> "ThreeDSResult":
        """Create from challenge response."""
        return cls(
            version=challenge_response.message_version,
            ds_trans_id=auth_response.ds_trans_id,
            acs_trans_id=challenge_response.acs_trans_id,
            threeds_server_trans_id=challenge_response.threeds_server_trans_id,
            trans_status=challenge_response.trans_status,
            authentication_value=challenge_response.authentication_value,
            eci=challenge_response.eci,
            was_challenged=True,
            challenge_completion_ind=challenge_response.challenge_completion_ind,
        )

    def __str__(self) -> str:
        return f"ThreeDSResult(status={self.trans_status}, " f"eci={self.eci}, " f"auth={bool(self.authentication_value)})"
