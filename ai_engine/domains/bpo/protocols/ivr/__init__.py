"""
IVR Protocol Security

Provides security monitoring and validation for Interactive Voice Response
protocol interactions in BPO contact center environments. Enforces PCI DSS
compliance for DTMF-based payment capture, detects automated probing and
brute-force navigation attacks, and prevents toll fraud via IVR transfer
exploitation.

Supports:
- IVR message and event data structures
- DTMF sequence analysis with PCI-compliant masking
- IVR navigation path validation
- Rate limiting and automated attack detection
- Transfer destination validation for toll fraud prevention
- Full interaction audit logging for compliance
"""

from typing import List

__all__: List[str] = [
    "ivr_message",
    "ivr_validator",
]
