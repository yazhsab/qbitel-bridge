"""
FedNow Message Builder

Builder classes for constructing FedNow messages in ISO 20022 format.
"""

import logging
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Optional

from ai_engine.domains.banking.protocols.payments.domestic.fednow.fednow_codes import (
    FedNowMessageType,
    FedNowRejectCode,
    FedNowReturnCode,
    TransactionStatus,
)
from ai_engine.domains.banking.protocols.payments.domestic.fednow.fednow_message import (
    FedNowMessage,
    FedNowCreditTransfer,
    FedNowPaymentStatus,
    FedNowPaymentReturn,
    FedNowRequestForPayment,
    FedNowParticipant,
    FedNowParty,
    FedNowAccount,
    FedNowRemittanceInfo,
)

logger = logging.getLogger(__name__)


class FedNowBuildError(Exception):
    """Exception raised when FedNow message building fails."""

    pass


class FedNowBuilder:
    """
    Builder for FedNow messages.

    Supports building:
    - Credit Transfer (pacs.008)
    - Payment Status (pacs.002)
    - Payment Return (pacs.004)
    - Request for Payment (pain.013)

    Example usage:
        builder = FedNowBuilder()
        msg = (builder
            .set_amount(Decimal("100.00"))
            .set_debtor("JOHN DOE", "123456789", "091000019")
            .set_creditor("ACME CORP", "987654321", "021000089")
            .set_remittance_info("Invoice 12345")
            .build_credit_transfer())
    """

    # FedNow ISO 20022 namespaces
    NAMESPACES = {
        "pacs.008": "urn:iso:std:iso:20022:tech:xsd:pacs.008.001.08",
        "pacs.002": "urn:iso:std:iso:20022:tech:xsd:pacs.002.001.10",
        "pacs.004": "urn:iso:std:iso:20022:tech:xsd:pacs.004.001.09",
        "pain.013": "urn:iso:std:iso:20022:tech:xsd:pain.013.001.07",
        "head": "urn:iso:std:iso:20022:tech:xsd:head.001.001.02",
    }

    def __init__(self):
        """Initialize the builder."""
        self._reset()

    def _reset(self):
        """Reset builder state."""
        self._message_id: str = ""
        self._end_to_end_id: str = ""
        self._transaction_id: str = ""
        self._uetr: str = ""

        self._amount: Decimal = Decimal("0")
        self._currency: str = "USD"

        self._debtor: Optional[FedNowParty] = None
        self._creditor: Optional[FedNowParty] = None

        self._instructing_agent: Optional[FedNowParticipant] = None
        self._instructed_agent: Optional[FedNowParticipant] = None

        self._remittance_info: Optional[FedNowRemittanceInfo] = None

        self._purpose_code: str = ""
        self._category_purpose: str = ""

        # For status/return messages
        self._original_message_id: str = ""
        self._original_end_to_end_id: str = ""
        self._original_uetr: str = ""
        self._status: TransactionStatus = TransactionStatus.ACSC
        self._reject_code: Optional[FedNowRejectCode] = None
        self._return_code: Optional[FedNowReturnCode] = None

    def set_message_id(self, message_id: str) -> "FedNowBuilder":
        """Set the message ID."""
        self._message_id = message_id[:35]
        return self

    def set_end_to_end_id(self, e2e_id: str) -> "FedNowBuilder":
        """Set the end-to-end ID."""
        self._end_to_end_id = e2e_id[:35]
        return self

    def set_amount(self, amount: Decimal, currency: str = "USD") -> "FedNowBuilder":
        """Set the payment amount."""
        self._amount = amount
        self._currency = currency
        return self

    def set_debtor(
        self,
        name: str,
        account_number: str,
        routing_number: str,
        account_type: str = "CACC",
    ) -> "FedNowBuilder":
        """Set the debtor (payer) information."""
        self._debtor = FedNowParty(
            name=name,
            account=FedNowAccount(
                account_number=account_number,
                routing_number=routing_number,
                account_type=account_type,
            ),
        )
        return self

    def set_debtor_address(
        self,
        address_line1: str,
        city: str,
        state: str,
        postal_code: str,
        country: str = "US",
    ) -> "FedNowBuilder":
        """Set the debtor address."""
        if self._debtor:
            self._debtor.address_line1 = address_line1
            self._debtor.city = city
            self._debtor.state = state
            self._debtor.postal_code = postal_code
            self._debtor.country = country
        return self

    def set_creditor(
        self,
        name: str,
        account_number: str,
        routing_number: str,
        account_type: str = "CACC",
    ) -> "FedNowBuilder":
        """Set the creditor (payee) information."""
        self._creditor = FedNowParty(
            name=name,
            account=FedNowAccount(
                account_number=account_number,
                routing_number=routing_number,
                account_type=account_type,
            ),
        )
        return self

    def set_creditor_address(
        self,
        address_line1: str,
        city: str,
        state: str,
        postal_code: str,
        country: str = "US",
    ) -> "FedNowBuilder":
        """Set the creditor address."""
        if self._creditor:
            self._creditor.address_line1 = address_line1
            self._creditor.city = city
            self._creditor.state = state
            self._creditor.postal_code = postal_code
            self._creditor.country = country
        return self

    def set_instructing_agent(
        self,
        routing_number: str,
        name: str = "",
    ) -> "FedNowBuilder":
        """Set the instructing agent (sender's bank)."""
        self._instructing_agent = FedNowParticipant(
            routing_number=routing_number,
            name=name,
        )
        return self

    def set_instructed_agent(
        self,
        routing_number: str,
        name: str = "",
    ) -> "FedNowBuilder":
        """Set the instructed agent (receiver's bank)."""
        self._instructed_agent = FedNowParticipant(
            routing_number=routing_number,
            name=name,
        )
        return self

    def set_remittance_info(
        self,
        unstructured: str = "",
        reference_number: str = "",
        reference_type: str = "",
    ) -> "FedNowBuilder":
        """Set remittance information."""
        self._remittance_info = FedNowRemittanceInfo(
            unstructured=unstructured,
            reference_number=reference_number,
            reference_type=reference_type,
        )
        return self

    def set_purpose(
        self,
        purpose_code: str = "",
        category_purpose: str = "",
    ) -> "FedNowBuilder":
        """Set payment purpose codes."""
        self._purpose_code = purpose_code
        self._category_purpose = category_purpose
        return self

    def set_original_message(
        self,
        message_id: str,
        end_to_end_id: str = "",
        uetr: str = "",
    ) -> "FedNowBuilder":
        """Set original message reference (for status/return)."""
        self._original_message_id = message_id
        self._original_end_to_end_id = end_to_end_id
        self._original_uetr = uetr
        return self

    def set_status(
        self,
        status: TransactionStatus,
        reject_code: Optional[FedNowRejectCode] = None,
    ) -> "FedNowBuilder":
        """Set transaction status (for status messages)."""
        self._status = status
        self._reject_code = reject_code
        return self

    def set_return_code(self, return_code: FedNowReturnCode) -> "FedNowBuilder":
        """Set return reason code (for return messages)."""
        self._return_code = return_code
        return self

    def build_credit_transfer(self, validate: bool = True) -> FedNowCreditTransfer:
        """
        Build a FedNow Credit Transfer message.

        Args:
            validate: If True, validate before returning

        Returns:
            FedNowCreditTransfer object

        Raises:
            FedNowBuildError: If validation fails
        """
        msg = FedNowCreditTransfer(
            message_id=self._message_id,
            end_to_end_id=self._end_to_end_id,
            transaction_id=self._transaction_id or self._end_to_end_id,
            uetr=self._uetr or str(uuid.uuid4()),
            amount=self._amount,
            currency=self._currency,
            debtor=self._debtor or FedNowParty(),
            creditor=self._creditor or FedNowParty(),
            purpose_code=self._purpose_code,
            category_purpose=self._category_purpose,
            remittance_info=self._remittance_info or FedNowRemittanceInfo(),
        )

        if self._instructing_agent:
            msg.instructing_agent = self._instructing_agent
        elif self._debtor:
            msg.instructing_agent = FedNowParticipant(routing_number=self._debtor.account.routing_number)

        if self._instructed_agent:
            msg.instructed_agent = self._instructed_agent
        elif self._creditor:
            msg.instructed_agent = FedNowParticipant(routing_number=self._creditor.account.routing_number)

        if validate:
            errors = msg.validate()
            if errors:
                raise FedNowBuildError(f"Validation failed: {'; '.join(errors)}")

        self._reset()
        return msg

    def build_payment_status(
        self,
        validate: bool = True,
    ) -> FedNowPaymentStatus:
        """
        Build a FedNow Payment Status message.

        Args:
            validate: If True, validate before returning

        Returns:
            FedNowPaymentStatus object
        """
        msg = FedNowPaymentStatus(
            message_id=self._message_id,
            original_message_id=self._original_message_id,
            original_end_to_end_id=self._original_end_to_end_id,
            original_uetr=self._original_uetr,
            transaction_status=self._status,
            reject_code=self._reject_code,
            original_amount=self._amount if self._amount > 0 else None,
        )

        if self._instructing_agent:
            msg.instructing_agent = self._instructing_agent
        if self._instructed_agent:
            msg.instructed_agent = self._instructed_agent

        if validate:
            errors = msg.validate()
            if errors:
                raise FedNowBuildError(f"Validation failed: {'; '.join(errors)}")

        self._reset()
        return msg

    def build_payment_return(
        self,
        validate: bool = True,
    ) -> FedNowPaymentReturn:
        """
        Build a FedNow Payment Return message.

        Args:
            validate: If True, validate before returning

        Returns:
            FedNowPaymentReturn object
        """
        msg = FedNowPaymentReturn(
            message_id=self._message_id,
            original_message_id=self._original_message_id,
            original_end_to_end_id=self._original_end_to_end_id,
            original_uetr=self._original_uetr,
            return_code=self._return_code,
            returned_amount=self._amount,
            return_debtor=self._debtor or FedNowParty(),
            return_creditor=self._creditor or FedNowParty(),
        )

        if self._instructing_agent:
            msg.instructing_agent = self._instructing_agent
        if self._instructed_agent:
            msg.instructed_agent = self._instructed_agent

        if validate:
            errors = msg.validate()
            if errors:
                raise FedNowBuildError(f"Validation failed: {'; '.join(errors)}")

        self._reset()
        return msg

    def build_request_for_payment(
        self,
        expiry_hours: int = 24,
        validate: bool = True,
    ) -> FedNowRequestForPayment:
        """
        Build a FedNow Request for Payment message.

        Args:
            expiry_hours: Hours until the request expires
            validate: If True, validate before returning

        Returns:
            FedNowRequestForPayment object
        """
        msg = FedNowRequestForPayment(
            message_id=self._message_id,
            end_to_end_id=self._end_to_end_id,
            requested_amount=self._amount,
            currency=self._currency,
            creditor=self._creditor or FedNowParty(),
            debtor=self._debtor or FedNowParty(),
            expiry_datetime=datetime.now() + timedelta(hours=expiry_hours),
            purpose_code=self._purpose_code,
            category_purpose=self._category_purpose,
            remittance_info=self._remittance_info or FedNowRemittanceInfo(),
        )

        if self._instructing_agent:
            msg.instructing_agent = self._instructing_agent
        if self._instructed_agent:
            msg.instructed_agent = self._instructed_agent

        if validate:
            errors = msg.validate()
            if errors:
                raise FedNowBuildError(f"Validation failed: {'; '.join(errors)}")

        self._reset()
        return msg

    def to_xml(self, message: FedNowMessage) -> str:
        """
        Convert a FedNow message to ISO 20022 XML format.

        Args:
            message: FedNow message object

        Returns:
            XML string representation
        """
        if isinstance(message, FedNowCreditTransfer):
            return self._build_pacs008_xml(message)
        elif isinstance(message, FedNowPaymentStatus):
            return self._build_pacs002_xml(message)
        elif isinstance(message, FedNowPaymentReturn):
            return self._build_pacs004_xml(message)
        elif isinstance(message, FedNowRequestForPayment):
            return self._build_pain013_xml(message)
        else:
            raise FedNowBuildError(f"Unknown message type: {type(message)}")

    def _build_pacs008_xml(self, msg: FedNowCreditTransfer) -> str:
        """Build pacs.008 XML."""
        ns = self.NAMESPACES["pacs.008"]

        root = ET.Element("Document", xmlns=ns)
        fi_cdt_trf = ET.SubElement(root, "FIToFICstmrCdtTrf")

        # Group Header
        grp_hdr = ET.SubElement(fi_cdt_trf, "GrpHdr")
        ET.SubElement(grp_hdr, "MsgId").text = msg.message_id
        ET.SubElement(grp_hdr, "CreDtTm").text = msg.creation_datetime.isoformat()
        ET.SubElement(grp_hdr, "NbOfTxs").text = "1"

        # Settlement Information
        sttlm_inf = ET.SubElement(grp_hdr, "SttlmInf")
        ET.SubElement(sttlm_inf, "SttlmMtd").text = "CLRG"
        clr_sys = ET.SubElement(sttlm_inf, "ClrSys")
        ET.SubElement(clr_sys, "Cd").text = "FDN"  # FedNow

        # Credit Transfer Transaction Information
        cdt_trf_tx_inf = ET.SubElement(fi_cdt_trf, "CdtTrfTxInf")

        # Payment ID
        pmt_id = ET.SubElement(cdt_trf_tx_inf, "PmtId")
        ET.SubElement(pmt_id, "EndToEndId").text = msg.end_to_end_id
        ET.SubElement(pmt_id, "TxId").text = msg.transaction_id
        ET.SubElement(pmt_id, "UETR").text = msg.uetr

        # Amount
        amt = ET.SubElement(cdt_trf_tx_inf, "IntrBkSttlmAmt", Ccy=msg.currency)
        amt.text = str(msg.amount)

        # Settlement Date
        ET.SubElement(cdt_trf_tx_inf, "IntrBkSttlmDt").text = date.today().isoformat()

        # Charge Bearer
        ET.SubElement(cdt_trf_tx_inf, "ChrgBr").text = msg.charge_bearer

        # Instructing Agent
        instg_agt = ET.SubElement(cdt_trf_tx_inf, "InstgAgt")
        fin_instn_id = ET.SubElement(instg_agt, "FinInstnId")
        clr_sys_mmb_id = ET.SubElement(fin_instn_id, "ClrSysMmbId")
        ET.SubElement(clr_sys_mmb_id, "MmbId").text = msg.instructing_agent.routing_number

        # Instructed Agent
        instd_agt = ET.SubElement(cdt_trf_tx_inf, "InstdAgt")
        fin_instn_id = ET.SubElement(instd_agt, "FinInstnId")
        clr_sys_mmb_id = ET.SubElement(fin_instn_id, "ClrSysMmbId")
        ET.SubElement(clr_sys_mmb_id, "MmbId").text = msg.instructed_agent.routing_number

        # Debtor
        dbtr = ET.SubElement(cdt_trf_tx_inf, "Dbtr")
        ET.SubElement(dbtr, "Nm").text = msg.debtor.name

        # Debtor Account
        dbtr_acct = ET.SubElement(cdt_trf_tx_inf, "DbtrAcct")
        dbtr_id = ET.SubElement(dbtr_acct, "Id")
        ET.SubElement(dbtr_id, "Othr").text = msg.debtor.account.account_number

        # Debtor Agent
        dbtr_agt = ET.SubElement(cdt_trf_tx_inf, "DbtrAgt")
        fin_instn_id = ET.SubElement(dbtr_agt, "FinInstnId")
        clr_sys_mmb_id = ET.SubElement(fin_instn_id, "ClrSysMmbId")
        ET.SubElement(clr_sys_mmb_id, "MmbId").text = msg.debtor.account.routing_number

        # Creditor Agent
        cdtr_agt = ET.SubElement(cdt_trf_tx_inf, "CdtrAgt")
        fin_instn_id = ET.SubElement(cdtr_agt, "FinInstnId")
        clr_sys_mmb_id = ET.SubElement(fin_instn_id, "ClrSysMmbId")
        ET.SubElement(clr_sys_mmb_id, "MmbId").text = msg.creditor.account.routing_number

        # Creditor
        cdtr = ET.SubElement(cdt_trf_tx_inf, "Cdtr")
        ET.SubElement(cdtr, "Nm").text = msg.creditor.name

        # Creditor Account
        cdtr_acct = ET.SubElement(cdt_trf_tx_inf, "CdtrAcct")
        cdtr_id = ET.SubElement(cdtr_acct, "Id")
        ET.SubElement(cdtr_id, "Othr").text = msg.creditor.account.account_number

        # Remittance Information
        if msg.remittance_info and msg.remittance_info.unstructured:
            rmt_inf = ET.SubElement(cdt_trf_tx_inf, "RmtInf")
            ET.SubElement(rmt_inf, "Ustrd").text = msg.remittance_info.unstructured

        return ET.tostring(root, encoding="unicode", xml_declaration=True)

    def _build_pacs002_xml(self, msg: FedNowPaymentStatus) -> str:
        """Build pacs.002 XML."""
        ns = self.NAMESPACES["pacs.002"]

        root = ET.Element("Document", xmlns=ns)
        fi_pmt_sts = ET.SubElement(root, "FIToFIPmtStsRpt")

        # Group Header
        grp_hdr = ET.SubElement(fi_pmt_sts, "GrpHdr")
        ET.SubElement(grp_hdr, "MsgId").text = msg.message_id
        ET.SubElement(grp_hdr, "CreDtTm").text = msg.creation_datetime.isoformat()

        # Original Group Information
        orgnl_grp_inf = ET.SubElement(fi_pmt_sts, "OrgnlGrpInfAndSts")
        ET.SubElement(orgnl_grp_inf, "OrgnlMsgId").text = msg.original_message_id
        ET.SubElement(orgnl_grp_inf, "OrgnlMsgNmId").text = "pacs.008.001.08"

        # Transaction Information and Status
        tx_inf_sts = ET.SubElement(fi_pmt_sts, "TxInfAndSts")
        ET.SubElement(tx_inf_sts, "OrgnlEndToEndId").text = msg.original_end_to_end_id
        ET.SubElement(tx_inf_sts, "TxSts").text = msg.transaction_status.code

        # Status Reason (if rejected)
        if msg.reject_code:
            sts_rsn_inf = ET.SubElement(tx_inf_sts, "StsRsnInf")
            rsn = ET.SubElement(sts_rsn_inf, "Rsn")
            ET.SubElement(rsn, "Cd").text = msg.reject_code.code
            if msg.reject_reason:
                ET.SubElement(sts_rsn_inf, "AddtlInf").text = msg.reject_reason

        return ET.tostring(root, encoding="unicode", xml_declaration=True)

    def _build_pacs004_xml(self, msg: FedNowPaymentReturn) -> str:
        """Build pacs.004 XML."""
        ns = self.NAMESPACES["pacs.004"]

        root = ET.Element("Document", xmlns=ns)
        pmt_rtr = ET.SubElement(root, "PmtRtr")

        # Group Header
        grp_hdr = ET.SubElement(pmt_rtr, "GrpHdr")
        ET.SubElement(grp_hdr, "MsgId").text = msg.message_id
        ET.SubElement(grp_hdr, "CreDtTm").text = msg.creation_datetime.isoformat()
        ET.SubElement(grp_hdr, "NbOfTxs").text = "1"

        # Transaction Information
        tx_inf = ET.SubElement(pmt_rtr, "TxInf")
        ET.SubElement(tx_inf, "RtrId").text = msg.return_id
        ET.SubElement(tx_inf, "OrgnlEndToEndId").text = msg.original_end_to_end_id

        # Returned Amount
        rtrd_amt = ET.SubElement(tx_inf, "RtrdIntrBkSttlmAmt", Ccy="USD")
        rtrd_amt.text = str(msg.returned_amount)

        # Return Reason
        if msg.return_code:
            rtr_rsn_inf = ET.SubElement(tx_inf, "RtrRsnInf")
            rsn = ET.SubElement(rtr_rsn_inf, "Rsn")
            ET.SubElement(rsn, "Cd").text = msg.return_code.code

        return ET.tostring(root, encoding="unicode", xml_declaration=True)

    def _build_pain013_xml(self, msg: FedNowRequestForPayment) -> str:
        """Build pain.013 XML."""
        ns = self.NAMESPACES["pain.013"]

        root = ET.Element("Document", xmlns=ns)
        cdtr_pmt_actvtn_req = ET.SubElement(root, "CdtrPmtActvtnReq")

        # Group Header
        grp_hdr = ET.SubElement(cdtr_pmt_actvtn_req, "GrpHdr")
        ET.SubElement(grp_hdr, "MsgId").text = msg.message_id
        ET.SubElement(grp_hdr, "CreDtTm").text = msg.creation_datetime.isoformat()
        ET.SubElement(grp_hdr, "NbOfTxs").text = "1"

        # Payment Information
        pmt_inf = ET.SubElement(cdtr_pmt_actvtn_req, "PmtInf")
        ET.SubElement(pmt_inf, "PmtInfId").text = msg.request_id
        ET.SubElement(pmt_inf, "PmtMtd").text = "TRF"

        # Requested Amount
        amt = ET.SubElement(pmt_inf, "ReqdAmt", Ccy=msg.currency)
        amt.text = str(msg.requested_amount)

        # Expiry
        if msg.expiry_datetime:
            ET.SubElement(pmt_inf, "XpryDt").text = msg.expiry_datetime.date().isoformat()

        # Creditor (requester)
        cdtr = ET.SubElement(pmt_inf, "Cdtr")
        ET.SubElement(cdtr, "Nm").text = msg.creditor.name

        cdtr_acct = ET.SubElement(pmt_inf, "CdtrAcct")
        cdtr_id = ET.SubElement(cdtr_acct, "Id")
        ET.SubElement(cdtr_id, "Othr").text = msg.creditor.account.account_number

        # Debtor (payer)
        dbtr = ET.SubElement(pmt_inf, "Dbtr")
        ET.SubElement(dbtr, "Nm").text = msg.debtor.name

        dbtr_acct = ET.SubElement(pmt_inf, "DbtrAcct")
        dbtr_id = ET.SubElement(dbtr_acct, "Id")
        ET.SubElement(dbtr_id, "Othr").text = msg.debtor.account.account_number

        return ET.tostring(root, encoding="unicode", xml_declaration=True)


def create_instant_payment(
    debtor_name: str,
    debtor_account: str,
    debtor_routing: str,
    creditor_name: str,
    creditor_account: str,
    creditor_routing: str,
    amount: Decimal,
    remittance_info: str = "",
) -> FedNowCreditTransfer:
    """
    Create a simple FedNow instant payment.

    Args:
        debtor_name: Name of the payer
        debtor_account: Payer's account number
        debtor_routing: Payer's bank routing number
        creditor_name: Name of the payee
        creditor_account: Payee's account number
        creditor_routing: Payee's bank routing number
        amount: Payment amount
        remittance_info: Optional payment reference/description

    Returns:
        FedNowCreditTransfer message
    """
    builder = FedNowBuilder()

    builder.set_amount(amount)
    builder.set_debtor(debtor_name, debtor_account, debtor_routing)
    builder.set_creditor(creditor_name, creditor_account, creditor_routing)

    if remittance_info:
        builder.set_remittance_info(unstructured=remittance_info)

    return builder.build_credit_transfer()


def create_request_for_payment(
    creditor_name: str,
    creditor_account: str,
    creditor_routing: str,
    debtor_name: str,
    debtor_account: str,
    debtor_routing: str,
    amount: Decimal,
    reference: str = "",
    expiry_hours: int = 24,
) -> FedNowRequestForPayment:
    """
    Create a FedNow Request for Payment.

    Args:
        creditor_name: Name of the requester (payee)
        creditor_account: Requester's account number
        creditor_routing: Requester's bank routing number
        debtor_name: Name of the payer
        debtor_account: Payer's account number
        debtor_routing: Payer's bank routing number
        amount: Requested amount
        reference: Payment reference/description
        expiry_hours: Hours until request expires

    Returns:
        FedNowRequestForPayment message
    """
    builder = FedNowBuilder()

    builder.set_amount(amount)
    builder.set_creditor(creditor_name, creditor_account, creditor_routing)
    builder.set_debtor(debtor_name, debtor_account, debtor_routing)

    if reference:
        builder.set_remittance_info(unstructured=reference)

    return builder.build_request_for_payment(expiry_hours=expiry_hours)
