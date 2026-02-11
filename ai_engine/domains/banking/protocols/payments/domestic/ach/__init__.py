"""
ACH/NACHA Protocol Handler

US Automated Clearing House file format parser and builder.
Implements NACHA Operating Rules for ACH transactions.

Volume: ~$72 trillion annually, 30+ billion transactions
"""

from ai_engine.domains.banking.protocols.payments.domestic.ach.nacha_parser import (
    NACHAParser,
    NACHAFile,
    FileHeader,
    FileControl,
    BatchHeader,
    BatchControl,
    EntryDetail,
    Addenda,
)
from ai_engine.domains.banking.protocols.payments.domestic.ach.nacha_builder import (
    NACHABuilder,
)
from ai_engine.domains.banking.protocols.payments.domestic.ach.ach_codes import (
    ServiceClassCode,
    TransactionCode,
    SECCode,
    AddendaTypeCode,
    ReturnReasonCode,
)

__all__ = [
    # Parser
    "NACHAParser",
    "NACHAFile",
    "FileHeader",
    "FileControl",
    "BatchHeader",
    "BatchControl",
    "EntryDetail",
    "Addenda",
    # Builder
    "NACHABuilder",
    # Codes
    "ServiceClassCode",
    "TransactionCode",
    "SECCode",
    "AddendaTypeCode",
    "ReturnReasonCode",
]
