"""
3D Secure Validator

Validates 3D Secure messages and authentication data.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from ai_engine.domains.banking.protocols.validators.base_validator import (
    BaseValidator,
    ValidationResult,
    ValidationSeverity,
)
from ai_engine.domains.banking.protocols.payments.cards.three_ds.three_ds_codes import (
    ThreeDSVersion,
    ThreeDSTransactionStatus,
    ThreeDSDeviceChannel,
    ThreeDSMessageCategory,
    ECI_VALUES,
)
from ai_engine.domains.banking.protocols.payments.cards.three_ds.three_ds_message import (
    ThreeDSAuthRequest,
    ThreeDSAuthResponse,
    ThreeDSChallengeRequest,
    ThreeDSChallengeResponse,
    ThreeDSResult,
    BrowserInfo,
)


class ThreeDSValidator(BaseValidator):
    """
    Validator for 3D Secure messages.

    Validates:
    - Authentication requests
    - Authentication responses
    - Challenge flow
    - Field formats and values
    """

    UUID_PATTERN = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        re.IGNORECASE,
    )

    def __init__(
        self,
        strict: bool = True,
        version: str = "2.2.0",
    ):
        """
        Initialize the 3DS validator.

        Args:
            strict: If True, treat warnings as errors
            version: Expected 3DS version
        """
        super().__init__(strict)
        self.expected_version = version

    @property
    def name(self) -> str:
        return "ThreeDSValidator"

    @property
    def version(self) -> str:
        return "1.0"

    def validate(self, data: Any) -> ValidationResult:
        """
        Validate 3DS data.

        Args:
            data: ThreeDSAuthRequest, ThreeDSAuthResponse, ThreeDSResult, or dict

        Returns:
            ValidationResult with any errors or warnings
        """
        result = self._create_result()

        if isinstance(data, ThreeDSAuthRequest):
            self._validate_auth_request(data, result)
        elif isinstance(data, ThreeDSAuthResponse):
            self._validate_auth_response(data, result)
        elif isinstance(data, ThreeDSChallengeRequest):
            self._validate_challenge_request(data, result)
        elif isinstance(data, ThreeDSChallengeResponse):
            self._validate_challenge_response(data, result)
        elif isinstance(data, ThreeDSResult):
            self._validate_result(data, result)
        elif isinstance(data, dict):
            self._validate_dict(data, result)
        else:
            result.add_error(
                "3DS_INVALID_INPUT",
                "Input must be a 3DS message or dictionary",
                severity=ValidationSeverity.CRITICAL,
            )

        return result

    def _validate_auth_request(
        self, req: ThreeDSAuthRequest, result: ValidationResult
    ) -> None:
        """Validate authentication request."""
        # Transaction IDs
        if not req.threeds_server_trans_id:
            result.add_error(
                "3DS_MISSING_SERVER_TRANS_ID",
                "threeDSServerTransID is required",
                field="threeDSServerTransID",
            )
        elif not self.UUID_PATTERN.match(req.threeds_server_trans_id):
            result.add_error(
                "3DS_INVALID_SERVER_TRANS_ID",
                "threeDSServerTransID must be a valid UUID",
                field="threeDSServerTransID",
            )

        # Version
        if not req.message_version:
            result.add_error(
                "3DS_MISSING_VERSION",
                "messageVersion is required",
                field="messageVersion",
            )
        elif not ThreeDSVersion.from_string(req.message_version):
            result.add_warning(
                "3DS_UNKNOWN_VERSION",
                f"Unknown message version: {req.message_version}",
                field="messageVersion",
            )

        # Device channel
        if not req.device_channel:
            result.add_error(
                "3DS_MISSING_DEVICE_CHANNEL",
                "deviceChannel is required",
                field="deviceChannel",
            )
        elif req.device_channel not in ("01", "02", "03"):
            result.add_error(
                "3DS_INVALID_DEVICE_CHANNEL",
                f"Invalid deviceChannel: {req.device_channel}",
                field="deviceChannel",
            )

        # Message category
        if not req.message_category:
            result.add_error(
                "3DS_MISSING_MESSAGE_CATEGORY",
                "messageCategory is required",
                field="messageCategory",
            )
        elif req.message_category not in ("01", "02"):
            result.add_error(
                "3DS_INVALID_MESSAGE_CATEGORY",
                f"Invalid messageCategory: {req.message_category}",
                field="messageCategory",
            )

        # Card data
        if not req.acct_number:
            result.add_error(
                "3DS_MISSING_ACCT_NUMBER",
                "acctNumber (PAN) is required",
                field="acctNumber",
            )
        elif not req.acct_number.isdigit() or len(req.acct_number) < 13:
            result.add_error(
                "3DS_INVALID_ACCT_NUMBER",
                "acctNumber must be a valid PAN",
                field="acctNumber",
            )

        if not req.card_expiry_date:
            result.add_error(
                "3DS_MISSING_EXPIRY",
                "cardExpiryDate is required",
                field="cardExpiryDate",
            )
        elif len(req.card_expiry_date) != 4 or not req.card_expiry_date.isdigit():
            result.add_error(
                "3DS_INVALID_EXPIRY",
                "cardExpiryDate must be YYMM format",
                field="cardExpiryDate",
            )

        # Purchase data (for payment authentication)
        if req.message_category == "01":  # Payment
            if req.purchase_amount < 0:
                result.add_error(
                    "3DS_NEGATIVE_AMOUNT",
                    "purchaseAmount cannot be negative",
                    field="purchaseAmount",
                )

            if not req.purchase_currency:
                result.add_error(
                    "3DS_MISSING_CURRENCY",
                    "purchaseCurrency is required for payment",
                    field="purchaseCurrency",
                )
            elif not req.purchase_currency.isdigit() or len(req.purchase_currency) != 3:
                result.add_error(
                    "3DS_INVALID_CURRENCY",
                    "purchaseCurrency must be 3-digit ISO 4217 code",
                    field="purchaseCurrency",
                )

        # Merchant info
        if not req.merchant_info.acquirer_bin:
            result.add_error(
                "3DS_MISSING_ACQUIRER_BIN",
                "acquirerBIN is required",
                field="acquirerBIN",
            )

        if not req.merchant_info.acquirer_merchant_id:
            result.add_error(
                "3DS_MISSING_MERCHANT_ID",
                "acquirerMerchantID is required",
                field="acquirerMerchantID",
            )

        if not req.merchant_info.mcc:
            result.add_error(
                "3DS_MISSING_MCC",
                "mcc (Merchant Category Code) is required",
                field="mcc",
            )

        # Browser info (for browser channel)
        if req.device_channel == "02":
            self._validate_browser_info(req.browser_info, result)

        # Notification URL
        if not req.notification_url:
            result.add_warning(
                "3DS_MISSING_NOTIFICATION_URL",
                "notificationURL is recommended",
                field="notificationURL",
            )
        elif not req.notification_url.startswith("https://"):
            result.add_warning(
                "3DS_INSECURE_NOTIFICATION_URL",
                "notificationURL should use HTTPS",
                field="notificationURL",
            )

    def _validate_browser_info(
        self, browser: Optional[BrowserInfo], result: ValidationResult
    ) -> None:
        """Validate browser information."""
        if not browser:
            result.add_error(
                "3DS_MISSING_BROWSER_INFO",
                "Browser information is required for browser channel",
                field="browserInfo",
            )
            return

        if not browser.user_agent:
            result.add_error(
                "3DS_MISSING_USER_AGENT",
                "browserUserAgent is required",
                field="browserUserAgent",
            )

        if not browser.accept_header:
            result.add_warning(
                "3DS_MISSING_ACCEPT_HEADER",
                "browserAcceptHeader is recommended",
                field="browserAcceptHeader",
            )

        if browser.color_depth not in (1, 4, 8, 15, 16, 24, 32, 48):
            result.add_warning(
                "3DS_INVALID_COLOR_DEPTH",
                f"Unusual browserColorDepth: {browser.color_depth}",
                field="browserColorDepth",
            )

        if browser.screen_width <= 0 or browser.screen_height <= 0:
            result.add_error(
                "3DS_INVALID_SCREEN_SIZE",
                "browserScreenWidth and browserScreenHeight must be positive",
                field="browserScreenWidth",
            )

    def _validate_auth_response(
        self, res: ThreeDSAuthResponse, result: ValidationResult
    ) -> None:
        """Validate authentication response."""
        # Transaction IDs
        if not res.threeds_server_trans_id:
            result.add_error(
                "3DS_MISSING_SERVER_TRANS_ID",
                "threeDSServerTransID is required",
                field="threeDSServerTransID",
            )

        if not res.acs_trans_id:
            result.add_warning(
                "3DS_MISSING_ACS_TRANS_ID",
                "acsTransID is expected in response",
                field="acsTransID",
            )

        # Transaction status
        if not res.trans_status:
            result.add_error(
                "3DS_MISSING_TRANS_STATUS",
                "transStatus is required",
                field="transStatus",
            )
        elif not ThreeDSTransactionStatus.from_code(res.trans_status):
            result.add_error(
                "3DS_INVALID_TRANS_STATUS",
                f"Invalid transStatus: {res.trans_status}",
                field="transStatus",
            )

        # Authentication value for successful auth
        if res.trans_status in ("Y", "A"):
            if not res.authentication_value:
                result.add_error(
                    "3DS_MISSING_AUTH_VALUE",
                    "authenticationValue required for successful authentication",
                    field="authenticationValue",
                )

            if not res.eci:
                result.add_error(
                    "3DS_MISSING_ECI",
                    "eci required for successful authentication",
                    field="eci",
                )

        # Challenge data
        if res.trans_status in ("C", "D"):
            if not res.acs_url:
                result.add_error(
                    "3DS_MISSING_ACS_URL",
                    "acsURL required when challenge is needed",
                    field="acsURL",
                )
            elif not res.acs_url.startswith("https://"):
                result.add_warning(
                    "3DS_INSECURE_ACS_URL",
                    "acsURL should use HTTPS",
                    field="acsURL",
                )

        # ECI validation
        if res.eci:
            if res.eci not in ("00", "01", "02", "05", "06", "07"):
                result.add_warning(
                    "3DS_UNKNOWN_ECI",
                    f"Unknown ECI value: {res.eci}",
                    field="eci",
                )

    def _validate_challenge_request(
        self, req: ThreeDSChallengeRequest, result: ValidationResult
    ) -> None:
        """Validate challenge request."""
        if not req.threeds_server_trans_id:
            result.add_error(
                "3DS_MISSING_SERVER_TRANS_ID",
                "threeDSServerTransID is required",
                field="threeDSServerTransID",
            )

        if not req.acs_trans_id:
            result.add_error(
                "3DS_MISSING_ACS_TRANS_ID",
                "acsTransID is required",
                field="acsTransID",
            )

        if req.challenge_window_size not in ("01", "02", "03", "04", "05"):
            result.add_warning(
                "3DS_INVALID_WINDOW_SIZE",
                f"Invalid challengeWindowSize: {req.challenge_window_size}",
                field="challengeWindowSize",
            )

    def _validate_challenge_response(
        self, res: ThreeDSChallengeResponse, result: ValidationResult
    ) -> None:
        """Validate challenge response."""
        if not res.threeds_server_trans_id:
            result.add_error(
                "3DS_MISSING_SERVER_TRANS_ID",
                "threeDSServerTransID is required",
                field="threeDSServerTransID",
            )

        if not res.acs_trans_id:
            result.add_error(
                "3DS_MISSING_ACS_TRANS_ID",
                "acsTransID is required",
                field="acsTransID",
            )

        if not res.trans_status:
            result.add_error(
                "3DS_MISSING_TRANS_STATUS",
                "transStatus is required",
                field="transStatus",
            )

        if res.challenge_completion_ind not in ("Y", "N", ""):
            result.add_error(
                "3DS_INVALID_COMPLETION_IND",
                f"Invalid challengeCompletionInd: {res.challenge_completion_ind}",
                field="challengeCompletionInd",
            )

        # Authentication value for completed challenge
        if res.trans_status == "Y":
            if not res.authentication_value:
                result.add_error(
                    "3DS_MISSING_AUTH_VALUE",
                    "authenticationValue required for successful challenge",
                    field="authenticationValue",
                )

    def _validate_result(
        self, result_obj: ThreeDSResult, result: ValidationResult
    ) -> None:
        """Validate complete 3DS result."""
        # Transaction IDs
        if not result_obj.threeds_server_trans_id:
            result.add_error(
                "3DS_MISSING_SERVER_TRANS_ID",
                "threeDSServerTransID is required",
                field="threeDSServerTransID",
            )

        # Status
        if not result_obj.trans_status:
            result.add_error(
                "3DS_MISSING_TRANS_STATUS",
                "transStatus is required",
                field="transStatus",
            )
        elif not ThreeDSTransactionStatus.from_code(result_obj.trans_status):
            result.add_error(
                "3DS_INVALID_TRANS_STATUS",
                f"Invalid transStatus: {result_obj.trans_status}",
                field="transStatus",
            )

        # Authentication value for successful auth
        if result_obj.trans_status in ("Y", "A"):
            if not result_obj.authentication_value:
                result.add_warning(
                    "3DS_MISSING_AUTH_VALUE",
                    "authenticationValue expected for successful authentication",
                    field="authenticationValue",
                )
            else:
                # CAVV should be 28 or 32 base64 characters
                if len(result_obj.authentication_value) not in (28, 32, 40):
                    result.add_warning(
                        "3DS_INVALID_AUTH_VALUE_LENGTH",
                        f"authenticationValue has unusual length: {len(result_obj.authentication_value)}",
                        field="authenticationValue",
                    )

            if not result_obj.eci:
                result.add_warning(
                    "3DS_MISSING_ECI",
                    "eci expected for successful authentication",
                    field="eci",
                )

        # Consistency checks
        if result_obj.was_challenged and not result_obj.challenge_completion_ind:
            result.add_warning(
                "3DS_MISSING_CHALLENGE_COMPLETION",
                "challengeCompletionInd expected when challenge was performed",
                field="challengeCompletionInd",
            )

    def _validate_dict(self, data: Dict, result: ValidationResult) -> None:
        """Validate 3DS data from dictionary."""
        # Determine message type
        msg_type = data.get("messageType", "")

        if msg_type == "AReq":
            # Validate as auth request
            if not data.get("threeDSServerTransID"):
                result.add_error(
                    "3DS_MISSING_SERVER_TRANS_ID",
                    "threeDSServerTransID is required",
                    field="threeDSServerTransID",
                )

            if not data.get("acctNumber"):
                result.add_error(
                    "3DS_MISSING_ACCT_NUMBER",
                    "acctNumber is required",
                    field="acctNumber",
                )

        elif msg_type == "ARes":
            # Validate as auth response
            if not data.get("transStatus"):
                result.add_error(
                    "3DS_MISSING_TRANS_STATUS",
                    "transStatus is required",
                    field="transStatus",
                )

        # Common validations
        version = data.get("messageVersion", "")
        if version and not ThreeDSVersion.from_string(version):
            result.add_warning(
                "3DS_UNKNOWN_VERSION",
                f"Unknown message version: {version}",
                field="messageVersion",
            )

        trans_status = data.get("transStatus", "")
        if trans_status and not ThreeDSTransactionStatus.from_code(trans_status):
            result.add_error(
                "3DS_INVALID_TRANS_STATUS",
                f"Invalid transStatus: {trans_status}",
                field="transStatus",
            )
