"""
3D Secure Protocol Implementation

3D Secure (3DS) authentication protocol support for:
- 3DS 1.0 (legacy)
- 3DS 2.0/2.1/2.2 (EMV 3D Secure)

Components:
- Authentication Request/Response
- Challenge Flow
- Risk-Based Authentication (RBA)
- Frictionless Flow
- Device Data Collection
"""

from ai_engine.domains.banking.protocols.payments.cards.three_ds.three_ds_codes import (
    ThreeDSVersion,
    ThreeDSMessageType,
    ThreeDSTransactionStatus,
    ThreeDSChallengeIndicator,
    ThreeDSDeviceChannel,
    ThreeDSAuthenticationType,
    ThreeDSMessageCategory,
    ECI_VALUES,
    get_eci_description,
)
from ai_engine.domains.banking.protocols.payments.cards.three_ds.three_ds_message import (
    ThreeDSAuthRequest,
    ThreeDSAuthResponse,
    ThreeDSChallengeRequest,
    ThreeDSChallengeResponse,
    ThreeDSResultRequest,
    ThreeDSResult,
    DeviceInfo,
    BrowserInfo,
    MerchantInfo,
    CardholderInfo,
)
from ai_engine.domains.banking.protocols.payments.cards.three_ds.three_ds_validator import (
    ThreeDSValidator,
)

__all__ = [
    # Codes and enums
    "ThreeDSVersion",
    "ThreeDSMessageType",
    "ThreeDSTransactionStatus",
    "ThreeDSChallengeIndicator",
    "ThreeDSDeviceChannel",
    "ThreeDSAuthenticationType",
    "ThreeDSMessageCategory",
    "ECI_VALUES",
    "get_eci_description",
    # Message structures
    "ThreeDSAuthRequest",
    "ThreeDSAuthResponse",
    "ThreeDSChallengeRequest",
    "ThreeDSChallengeResponse",
    "ThreeDSResultRequest",
    "ThreeDSResult",
    "DeviceInfo",
    "BrowserInfo",
    "MerchantInfo",
    "CardholderInfo",
    # Validator
    "ThreeDSValidator",
]
