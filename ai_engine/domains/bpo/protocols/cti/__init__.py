"""
CTI Bridge Security

Provides security monitoring and validation for Computer Telephony Integration
(CTI) bridge protocols used in BPO contact center environments. Enforces agent
state machine integrity, validates call routing to prevent queue jumping and
toll fraud, and detects shared credential usage and abnormal call patterns.

Supports:
- CTI message and event data structures (CSTA/TAPI compatible)
- Agent state transition validation
- Call routing integrity enforcement
- Agent login pattern monitoring
- Transfer and conference request validation
- Abnormal call pattern detection
- Permission-level enforcement for agent operations
"""

from typing import List

__all__: List[str] = [
    "cti_message",
    "cti_validator",
]
