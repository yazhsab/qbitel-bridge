"""
Industrial OT/ICS Domain PQC Module

Post-quantum cryptography for industrial control systems:
- SCADA (Supervisory Control and Data Acquisition)
- PLC (Programmable Logic Controllers)
- DCS (Distributed Control Systems)
- SIS (Safety Instrumented Systems)

Key challenges:
- 20+ year equipment lifecycles
- Real-time requirements (< 1ms for some protocols)
- Legacy protocol compatibility (Modbus, DNP3, etc.)
- Safety certification (IEC 61508, IEC 62443)

Supported standards:
- IEC 62351: Power systems security
- IEC 62443: Industrial automation security
- NERC CIP: Critical infrastructure protection
"""

from .lightweight_scada import (
    LightweightScadaPQC,
    ScadaSecurityLevel,
    ScadaKeyExchange,
    ScadaSessionContext,
    ScadaMessageAuthenticator,
)

from .iec62351_profile import (
    IEC62351Profile,
    IEC62351KeyManagement,
    IEC62351MessageProtection,
    SecurityObjective,
    ProtectionLevel,
    GooseSecurityProfile,
    SvSecurityProfile,
)

from .modbus_secure import (
    SecureModbusClient,
    SecureModbusServer,
    ModbusSecurityProfile,
    ModbusAuthenticator,
    ModbusFunctionCode,
    ModbusSecureMessage,
)

from .plc_authenticator import (
    PLCAuthenticator,
    PLCSecurityContext,
    PLCKeyManager,
    FirmwareValidator,
    PLCVendor,
    PLCSecurityMode,
)

from .safety_instrumented import (
    SISSecurityManager,
    SafetyFunction,
    SILLevel,
    SafetyKeyManager,
    SISMessageAuthenticator,
    SISAuditLogger,
)

__all__ = [
    # SCADA
    "LightweightScadaPQC",
    "ScadaSecurityLevel",
    "ScadaKeyExchange",
    "ScadaSessionContext",
    "ScadaMessageAuthenticator",
    # IEC 62351
    "IEC62351Profile",
    "IEC62351KeyManagement",
    "IEC62351MessageProtection",
    "SecurityObjective",
    "ProtectionLevel",
    "GooseSecurityProfile",
    "SvSecurityProfile",
    # Modbus
    "SecureModbusClient",
    "SecureModbusServer",
    "ModbusSecurityProfile",
    "ModbusAuthenticator",
    "ModbusFunctionCode",
    "ModbusSecureMessage",
    # PLC
    "PLCAuthenticator",
    "PLCSecurityContext",
    "PLCKeyManager",
    "FirmwareValidator",
    "PLCVendor",
    "PLCSecurityMode",
    # SIS
    "SISSecurityManager",
    "SafetyFunction",
    "SILLevel",
    "SafetyKeyManager",
    "SISMessageAuthenticator",
    "SISAuditLogger",
]
