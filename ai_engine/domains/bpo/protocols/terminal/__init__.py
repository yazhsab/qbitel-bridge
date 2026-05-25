"""
TN3270e Terminal Protocol Security

Provides security monitoring and validation for TN3270e terminal emulation
sessions used in BPO mainframe access scenarios. Detects unauthorized access
patterns, data exfiltration via screen scraping, credential stuffing attacks,
and protected field tampering in IBM 3270 terminal data streams.

Supports:
- TN3270e message parsing and serialization (RFC 2355)
- 3270 data stream interpretation (structured fields, orders, attributes)
- Screen buffer analysis for sensitive data detection
- Session security validation and audit logging
- Protected field integrity verification
"""

from typing import List

__all__: List[str] = [
    "tn3270e_message",
    "tn3270e_parser",
    "tn3270e_validator",
]
