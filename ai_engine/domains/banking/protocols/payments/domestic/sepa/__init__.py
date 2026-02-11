"""
SEPA (Single Euro Payments Area) Protocol Implementation

SEPA enables euro-denominated payments across 36 European countries
using standardized payment instruments.

Payment Schemes:
- SCT (SEPA Credit Transfer) - Credit transfers
- SCT Inst (SEPA Instant Credit Transfer) - Real-time credit transfers
- SDD (SEPA Direct Debit) - Direct debits
  - SDD Core - Consumer direct debits
  - SDD B2B - Business-to-business direct debits

Message Formats:
- pain.001 - Customer Credit Transfer Initiation
- pain.002 - Customer Payment Status Report
- pain.008 - Customer Direct Debit Initiation
- pacs.008 - FI to FI Customer Credit Transfer
- pacs.003 - FI to FI Customer Direct Debit
- camt.053 - Bank to Customer Statement
- camt.054 - Bank to Customer Debit Credit Notification
"""

from typing import List

from ai_engine.domains.banking.protocols.payments.domestic.sepa.sepa_codes import (
    SEPAScheme,
    SEPAServiceLevel,
    SEPALocalInstrument,
    SEPASequenceType,
    SEPACategoryPurpose,
    SEPAPurposeCode,
    SEPARejectCode,
    SEPAReturnCode,
)
from ai_engine.domains.banking.protocols.payments.domestic.sepa.sepa_message import (
    SEPACreditTransfer,
    SEPADirectDebit,
    SEPAMandate,
    SEPAParty,
    SEPAAccount,
    SEPAFinancialInstitution,
)
from ai_engine.domains.banking.protocols.payments.domestic.sepa.sepa_builder import (
    SEPABuilder,
    create_sepa_credit_transfer,
    create_sepa_direct_debit,
)
from ai_engine.domains.banking.protocols.payments.domestic.sepa.sepa_validator import (
    SEPAValidator,
)

__all__: List[str] = [
    # Codes
    "SEPAScheme",
    "SEPAServiceLevel",
    "SEPALocalInstrument",
    "SEPASequenceType",
    "SEPACategoryPurpose",
    "SEPAPurposeCode",
    "SEPARejectCode",
    "SEPAReturnCode",
    # Message structures
    "SEPACreditTransfer",
    "SEPADirectDebit",
    "SEPAMandate",
    "SEPAParty",
    "SEPAAccount",
    "SEPAFinancialInstitution",
    # Builder
    "SEPABuilder",
    "create_sepa_credit_transfer",
    "create_sepa_direct_debit",
    # Validator
    "SEPAValidator",
]
