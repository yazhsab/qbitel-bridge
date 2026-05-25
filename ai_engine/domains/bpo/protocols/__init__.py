"""
BPO Protocol Handlers

Protocol handlers for telephony, terminal, and contact center protocols:
- SIP (Session Initiation Protocol) - Voice signaling
- TN3270e - IBM mainframe terminal emulation
- IVR (Interactive Voice Response) - VoiceXML/MRCP
- CTI (Computer Telephony Integration) - CSTA/TAPI
"""

from typing import List

__all__: List[str] = [
    "sip",
    "terminal",
    "ivr",
    "cti",
]
