"""
SEPA Message Builder

Builder classes for constructing SEPA payment messages in ISO 20022 format.
"""

import logging
import xml.etree.ElementTree as ET
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Optional
import uuid

from ai_engine.domains.banking.protocols.payments.domestic.sepa.sepa_codes import (
    SEPAScheme,
    SEPAServiceLevel,
    SEPALocalInstrument,
    SEPASequenceType,
    SEPACategoryPurpose,
    SEPAPurposeCode,
    SEPA_CURRENCY,
)
from ai_engine.domains.banking.protocols.payments.domestic.sepa.sepa_message import (
    SEPACreditTransfer,
    SEPADirectDebit,
    SEPAMandate,
    SEPAParty,
    SEPAAccount,
    SEPAFinancialInstitution,
    SEPARemittanceInfo,
)

logger = logging.getLogger(__name__)


class SEPABuildError(Exception):
    """Exception raised when SEPA message building fails."""
    pass


class SEPABuilder:
    """
    Builder for SEPA payment messages.

    Supports building:
    - SCT (SEPA Credit Transfer)
    - SCT Inst (SEPA Instant Credit Transfer)
    - SDD Core (SEPA Direct Debit Core)
    - SDD B2B (SEPA Direct Debit B2B)

    Example usage:
        builder = SEPABuilder()
        msg = (builder
            .set_scheme(SEPAScheme.SCT)
            .set_amount(Decimal("100.00"))
            .set_debtor("John Doe", "DE89370400440532013000", "COBADEFFXXX")
            .set_creditor("ACME Corp", "FR7630006000011234567890189", "BNPAFRPP")
            .set_remittance_info("Invoice 12345")
            .build_credit_transfer())
    """

    # ISO 20022 namespaces for SEPA
    NAMESPACES = {
        "pain.001": "urn:iso:std:iso:20022:tech:xsd:pain.001.001.09",
        "pain.008": "urn:iso:std:iso:20022:tech:xsd:pain.008.001.08",
        "pacs.008": "urn:iso:std:iso:20022:tech:xsd:pacs.008.001.08",
    }

    def __init__(self):
        """Initialize the builder."""
        self._reset()

    def _reset(self):
        """Reset builder state."""
        self._scheme: SEPAScheme = SEPAScheme.SCT
        self._service_level: SEPAServiceLevel = SEPAServiceLevel.SEPA
        self._local_instrument: Optional[SEPALocalInstrument] = None

        self._message_id: str = ""
        self._end_to_end_id: str = ""

        self._amount: Decimal = Decimal("0")
        self._currency: str = SEPA_CURRENCY

        self._debtor: Optional[SEPAParty] = None
        self._creditor: Optional[SEPAParty] = None

        self._remittance_info: Optional[SEPARemittanceInfo] = None

        self._category_purpose: Optional[SEPACategoryPurpose] = None
        self._purpose_code: Optional[SEPAPurposeCode] = None

        self._requested_execution_date: Optional[date] = None
        self._requested_collection_date: Optional[date] = None

        # Direct Debit specific
        self._mandate: Optional[SEPAMandate] = None
        self._creditor_scheme_id: str = ""
        self._sequence_type: SEPASequenceType = SEPASequenceType.RCUR

    def set_scheme(self, scheme: SEPAScheme) -> "SEPABuilder":
        """Set the SEPA payment scheme."""
        self._scheme = scheme
        return self

    def set_message_id(self, message_id: str) -> "SEPABuilder":
        """Set the message ID."""
        self._message_id = message_id[:35]
        return self

    def set_end_to_end_id(self, e2e_id: str) -> "SEPABuilder":
        """Set the end-to-end ID."""
        self._end_to_end_id = e2e_id[:35]
        return self

    def set_amount(self, amount: Decimal, currency: str = SEPA_CURRENCY) -> "SEPABuilder":
        """Set the payment amount."""
        self._amount = amount
        self._currency = currency
        return self

    def set_debtor(
        self,
        name: str,
        iban: str,
        bic: str = "",
    ) -> "SEPABuilder":
        """Set the debtor (payer) information."""
        self._debtor = SEPAParty(
            name=name,
            account=SEPAAccount(iban=iban),
            agent=SEPAFinancialInstitution(bic=bic) if bic else SEPAFinancialInstitution(),
        )
        return self

    def set_debtor_address(
        self,
        street_name: str,
        postal_code: str,
        town_name: str,
        country: str,
    ) -> "SEPABuilder":
        """Set the debtor address."""
        if self._debtor:
            self._debtor.street_name = street_name
            self._debtor.postal_code = postal_code
            self._debtor.town_name = town_name
            self._debtor.country = country
        return self

    def set_creditor(
        self,
        name: str,
        iban: str,
        bic: str = "",
    ) -> "SEPABuilder":
        """Set the creditor (payee) information."""
        self._creditor = SEPAParty(
            name=name,
            account=SEPAAccount(iban=iban),
            agent=SEPAFinancialInstitution(bic=bic) if bic else SEPAFinancialInstitution(),
        )
        return self

    def set_creditor_address(
        self,
        street_name: str,
        postal_code: str,
        town_name: str,
        country: str,
    ) -> "SEPABuilder":
        """Set the creditor address."""
        if self._creditor:
            self._creditor.street_name = street_name
            self._creditor.postal_code = postal_code
            self._creditor.town_name = town_name
            self._creditor.country = country
        return self

    def set_remittance_info(
        self,
        unstructured: str = "",
        reference: str = "",
        reference_type: str = "",
    ) -> "SEPABuilder":
        """Set remittance information."""
        self._remittance_info = SEPARemittanceInfo(
            unstructured=unstructured[:140] if unstructured else "",
            reference=reference,
            reference_type=reference_type,
        )
        return self

    def set_purpose(
        self,
        category_purpose: Optional[SEPACategoryPurpose] = None,
        purpose_code: Optional[SEPAPurposeCode] = None,
    ) -> "SEPABuilder":
        """Set payment purpose codes."""
        self._category_purpose = category_purpose
        self._purpose_code = purpose_code
        return self

    def set_execution_date(self, execution_date: date) -> "SEPABuilder":
        """Set requested execution date (for credit transfers)."""
        self._requested_execution_date = execution_date
        return self

    def set_collection_date(self, collection_date: date) -> "SEPABuilder":
        """Set requested collection date (for direct debits)."""
        self._requested_collection_date = collection_date
        return self

    # Direct Debit specific methods

    def set_creditor_scheme_id(self, creditor_id: str) -> "SEPABuilder":
        """Set creditor scheme identification (Creditor ID for direct debits)."""
        self._creditor_scheme_id = creditor_id[:35]
        return self

    def set_sequence_type(self, sequence_type: SEPASequenceType) -> "SEPABuilder":
        """Set sequence type for direct debit."""
        self._sequence_type = sequence_type
        return self

    def set_mandate(
        self,
        mandate_id: str,
        date_of_signature: date,
        amendment_indicator: bool = False,
    ) -> "SEPABuilder":
        """Set mandate information for direct debit."""
        self._mandate = SEPAMandate(
            mandate_id=mandate_id,
            date_of_signature=date_of_signature,
            amendment_indicator=amendment_indicator,
        )
        return self

    def build_credit_transfer(self, validate: bool = True) -> SEPACreditTransfer:
        """
        Build a SEPA Credit Transfer message.

        Args:
            validate: If True, validate before returning

        Returns:
            SEPACreditTransfer object
        """
        msg = SEPACreditTransfer(
            message_id=self._message_id,
            end_to_end_id=self._end_to_end_id,
            scheme=self._scheme,
            service_level=self._service_level,
            local_instrument=self._local_instrument,
            amount=self._amount,
            currency=self._currency,
            debtor=self._debtor or SEPAParty(),
            creditor=self._creditor or SEPAParty(),
            remittance_info=self._remittance_info or SEPARemittanceInfo(),
            category_purpose=self._category_purpose,
            purpose_code=self._purpose_code,
            requested_execution_date=self._requested_execution_date,
        )

        if validate:
            errors = msg.validate()
            if errors:
                raise SEPABuildError(f"Validation failed: {'; '.join(errors)}")

        self._reset()
        return msg

    def build_direct_debit(self, validate: bool = True) -> SEPADirectDebit:
        """
        Build a SEPA Direct Debit message.

        Args:
            validate: If True, validate before returning

        Returns:
            SEPADirectDebit object
        """
        msg = SEPADirectDebit(
            message_id=self._message_id,
            end_to_end_id=self._end_to_end_id,
            scheme=self._scheme if self._scheme.is_direct_debit else SEPAScheme.SDD_CORE,
            service_level=self._service_level,
            sequence_type=self._sequence_type,
            amount=self._amount,
            currency=self._currency,
            creditor=self._creditor or SEPAParty(),
            creditor_scheme_id=self._creditor_scheme_id,
            debtor=self._debtor or SEPAParty(),
            mandate=self._mandate or SEPAMandate(),
            remittance_info=self._remittance_info or SEPARemittanceInfo(),
            category_purpose=self._category_purpose,
            purpose_code=self._purpose_code,
            requested_collection_date=self._requested_collection_date,
        )

        if validate:
            errors = msg.validate()
            if errors:
                raise SEPABuildError(f"Validation failed: {'; '.join(errors)}")

        self._reset()
        return msg

    def to_xml(self, message) -> str:
        """
        Convert a SEPA message to ISO 20022 XML format.

        Args:
            message: SEPA message object

        Returns:
            XML string representation
        """
        if isinstance(message, SEPACreditTransfer):
            return self._build_pain001_xml(message)
        elif isinstance(message, SEPADirectDebit):
            return self._build_pain008_xml(message)
        else:
            raise SEPABuildError(f"Unknown message type: {type(message)}")

    def _build_pain001_xml(self, msg: SEPACreditTransfer) -> str:
        """Build pain.001 XML for SEPA Credit Transfer."""
        ns = self.NAMESPACES["pain.001"]

        root = ET.Element("Document", xmlns=ns)
        cstmr_cdt_trf = ET.SubElement(root, "CstmrCdtTrfInitn")

        # Group Header
        grp_hdr = ET.SubElement(cstmr_cdt_trf, "GrpHdr")
        ET.SubElement(grp_hdr, "MsgId").text = msg.message_id
        ET.SubElement(grp_hdr, "CreDtTm").text = msg.creation_datetime.isoformat()
        ET.SubElement(grp_hdr, "NbOfTxs").text = str(msg.number_of_transactions)
        if msg.control_sum:
            ET.SubElement(grp_hdr, "CtrlSum").text = str(msg.control_sum)

        # Initiating Party
        init_pty = ET.SubElement(grp_hdr, "InitgPty")
        ET.SubElement(init_pty, "Nm").text = msg.debtor.name

        # Payment Information
        pmt_inf = ET.SubElement(cstmr_cdt_trf, "PmtInf")
        ET.SubElement(pmt_inf, "PmtInfId").text = msg.payment_info_id
        ET.SubElement(pmt_inf, "PmtMtd").text = "TRF"
        ET.SubElement(pmt_inf, "BtchBookg").text = "true" if msg.batch_booking else "false"
        ET.SubElement(pmt_inf, "NbOfTxs").text = str(msg.number_of_transactions)

        # Payment Type Information
        pmt_tp_inf = ET.SubElement(pmt_inf, "PmtTpInf")
        svc_lvl = ET.SubElement(pmt_tp_inf, "SvcLvl")
        ET.SubElement(svc_lvl, "Cd").text = msg.service_level.code

        if msg.local_instrument:
            lcl_instrm = ET.SubElement(pmt_tp_inf, "LclInstrm")
            ET.SubElement(lcl_instrm, "Cd").text = msg.local_instrument.code

        if msg.category_purpose:
            ctgy_purp = ET.SubElement(pmt_tp_inf, "CtgyPurp")
            ET.SubElement(ctgy_purp, "Cd").text = msg.category_purpose.code

        # Requested Execution Date
        reqd_exctn_dt = ET.SubElement(pmt_inf, "ReqdExctnDt")
        exec_date = msg.requested_execution_date or date.today()
        ET.SubElement(reqd_exctn_dt, "Dt").text = exec_date.isoformat()

        # Debtor
        dbtr = ET.SubElement(pmt_inf, "Dbtr")
        ET.SubElement(dbtr, "Nm").text = msg.debtor.name
        if msg.debtor.country:
            pstl_adr = ET.SubElement(dbtr, "PstlAdr")
            ET.SubElement(pstl_adr, "Ctry").text = msg.debtor.country

        # Debtor Account
        dbtr_acct = ET.SubElement(pmt_inf, "DbtrAcct")
        dbtr_id = ET.SubElement(dbtr_acct, "Id")
        ET.SubElement(dbtr_id, "IBAN").text = msg.debtor.account.iban

        # Debtor Agent
        dbtr_agt = ET.SubElement(pmt_inf, "DbtrAgt")
        fin_instn_id = ET.SubElement(dbtr_agt, "FinInstnId")
        if msg.debtor.agent.bic:
            ET.SubElement(fin_instn_id, "BICFI").text = msg.debtor.agent.bic

        # Charge Bearer
        ET.SubElement(pmt_inf, "ChrgBr").text = msg.charge_bearer

        # Credit Transfer Transaction Information
        cdt_trf_tx_inf = ET.SubElement(pmt_inf, "CdtTrfTxInf")

        # Payment ID
        pmt_id = ET.SubElement(cdt_trf_tx_inf, "PmtId")
        if msg.instruction_id:
            ET.SubElement(pmt_id, "InstrId").text = msg.instruction_id
        ET.SubElement(pmt_id, "EndToEndId").text = msg.end_to_end_id

        # Amount
        amt = ET.SubElement(cdt_trf_tx_inf, "Amt")
        instd_amt = ET.SubElement(amt, "InstdAmt", Ccy=msg.currency)
        instd_amt.text = f"{msg.amount:.2f}"

        # Creditor Agent
        cdtr_agt = ET.SubElement(cdt_trf_tx_inf, "CdtrAgt")
        fin_instn_id = ET.SubElement(cdtr_agt, "FinInstnId")
        if msg.creditor.agent.bic:
            ET.SubElement(fin_instn_id, "BICFI").text = msg.creditor.agent.bic

        # Creditor
        cdtr = ET.SubElement(cdt_trf_tx_inf, "Cdtr")
        ET.SubElement(cdtr, "Nm").text = msg.creditor.name
        if msg.creditor.country:
            pstl_adr = ET.SubElement(cdtr, "PstlAdr")
            ET.SubElement(pstl_adr, "Ctry").text = msg.creditor.country

        # Creditor Account
        cdtr_acct = ET.SubElement(cdt_trf_tx_inf, "CdtrAcct")
        cdtr_id = ET.SubElement(cdtr_acct, "Id")
        ET.SubElement(cdtr_id, "IBAN").text = msg.creditor.account.iban

        # Purpose
        if msg.purpose_code:
            purp = ET.SubElement(cdt_trf_tx_inf, "Purp")
            ET.SubElement(purp, "Cd").text = msg.purpose_code.code

        # Remittance Information
        if msg.remittance_info:
            rmt_inf = ET.SubElement(cdt_trf_tx_inf, "RmtInf")
            if msg.remittance_info.unstructured:
                ET.SubElement(rmt_inf, "Ustrd").text = msg.remittance_info.unstructured
            elif msg.remittance_info.reference:
                strd = ET.SubElement(rmt_inf, "Strd")
                cdtr_ref_inf = ET.SubElement(strd, "CdtrRefInf")
                tp = ET.SubElement(cdtr_ref_inf, "Tp")
                cd_or_prtry = ET.SubElement(tp, "CdOrPrtry")
                ET.SubElement(cd_or_prtry, "Cd").text = msg.remittance_info.reference_type or "SCOR"
                ET.SubElement(cdtr_ref_inf, "Ref").text = msg.remittance_info.reference

        return ET.tostring(root, encoding="unicode", xml_declaration=True)

    def _build_pain008_xml(self, msg: SEPADirectDebit) -> str:
        """Build pain.008 XML for SEPA Direct Debit."""
        ns = self.NAMESPACES["pain.008"]

        root = ET.Element("Document", xmlns=ns)
        cstmr_drct_dbt = ET.SubElement(root, "CstmrDrctDbtInitn")

        # Group Header
        grp_hdr = ET.SubElement(cstmr_drct_dbt, "GrpHdr")
        ET.SubElement(grp_hdr, "MsgId").text = msg.message_id
        ET.SubElement(grp_hdr, "CreDtTm").text = msg.creation_datetime.isoformat()
        ET.SubElement(grp_hdr, "NbOfTxs").text = str(msg.number_of_transactions)

        # Initiating Party
        init_pty = ET.SubElement(grp_hdr, "InitgPty")
        ET.SubElement(init_pty, "Nm").text = msg.creditor.name

        # Payment Information
        pmt_inf = ET.SubElement(cstmr_drct_dbt, "PmtInf")
        ET.SubElement(pmt_inf, "PmtInfId").text = msg.payment_info_id
        ET.SubElement(pmt_inf, "PmtMtd").text = "DD"
        ET.SubElement(pmt_inf, "NbOfTxs").text = str(msg.number_of_transactions)

        # Payment Type Information
        pmt_tp_inf = ET.SubElement(pmt_inf, "PmtTpInf")
        svc_lvl = ET.SubElement(pmt_tp_inf, "SvcLvl")
        ET.SubElement(svc_lvl, "Cd").text = msg.service_level.code
        lcl_instrm = ET.SubElement(pmt_tp_inf, "LclInstrm")
        ET.SubElement(lcl_instrm, "Cd").text = msg.local_instrument.code
        ET.SubElement(pmt_tp_inf, "SeqTp").text = msg.sequence_type.code

        # Requested Collection Date
        coll_date = msg.requested_collection_date or (date.today() + timedelta(days=5))
        ET.SubElement(pmt_inf, "ReqdColltnDt").text = coll_date.isoformat()

        # Creditor
        cdtr = ET.SubElement(pmt_inf, "Cdtr")
        ET.SubElement(cdtr, "Nm").text = msg.creditor.name

        # Creditor Account
        cdtr_acct = ET.SubElement(pmt_inf, "CdtrAcct")
        cdtr_id = ET.SubElement(cdtr_acct, "Id")
        ET.SubElement(cdtr_id, "IBAN").text = msg.creditor.account.iban

        # Creditor Agent
        cdtr_agt = ET.SubElement(pmt_inf, "CdtrAgt")
        fin_instn_id = ET.SubElement(cdtr_agt, "FinInstnId")
        if msg.creditor.agent.bic:
            ET.SubElement(fin_instn_id, "BICFI").text = msg.creditor.agent.bic

        # Creditor Scheme Identification
        cdtr_schme_id = ET.SubElement(pmt_inf, "CdtrSchmeId")
        id_elem = ET.SubElement(cdtr_schme_id, "Id")
        prvt_id = ET.SubElement(id_elem, "PrvtId")
        othr = ET.SubElement(prvt_id, "Othr")
        ET.SubElement(othr, "Id").text = msg.creditor_scheme_id
        schme_nm = ET.SubElement(othr, "SchmeNm")
        ET.SubElement(schme_nm, "Prtry").text = "SEPA"

        # Direct Debit Transaction Information
        drct_dbt_tx_inf = ET.SubElement(pmt_inf, "DrctDbtTxInf")

        # Payment ID
        pmt_id = ET.SubElement(drct_dbt_tx_inf, "PmtId")
        ET.SubElement(pmt_id, "EndToEndId").text = msg.end_to_end_id

        # Amount
        instd_amt = ET.SubElement(drct_dbt_tx_inf, "InstdAmt", Ccy=msg.currency)
        instd_amt.text = f"{msg.amount:.2f}"

        # Mandate Related Information
        drct_dbt_tx = ET.SubElement(drct_dbt_tx_inf, "DrctDbtTx")
        mndt_rltd_inf = ET.SubElement(drct_dbt_tx, "MndtRltdInf")
        ET.SubElement(mndt_rltd_inf, "MndtId").text = msg.mandate.mandate_id
        if msg.mandate.date_of_signature:
            ET.SubElement(mndt_rltd_inf, "DtOfSgntr").text = msg.mandate.date_of_signature.isoformat()

        # Debtor Agent
        dbtr_agt = ET.SubElement(drct_dbt_tx_inf, "DbtrAgt")
        fin_instn_id = ET.SubElement(dbtr_agt, "FinInstnId")
        if msg.debtor.agent.bic:
            ET.SubElement(fin_instn_id, "BICFI").text = msg.debtor.agent.bic

        # Debtor
        dbtr = ET.SubElement(drct_dbt_tx_inf, "Dbtr")
        ET.SubElement(dbtr, "Nm").text = msg.debtor.name

        # Debtor Account
        dbtr_acct = ET.SubElement(drct_dbt_tx_inf, "DbtrAcct")
        dbtr_id = ET.SubElement(dbtr_acct, "Id")
        ET.SubElement(dbtr_id, "IBAN").text = msg.debtor.account.iban

        # Remittance Information
        if msg.remittance_info and msg.remittance_info.unstructured:
            rmt_inf = ET.SubElement(drct_dbt_tx_inf, "RmtInf")
            ET.SubElement(rmt_inf, "Ustrd").text = msg.remittance_info.unstructured

        return ET.tostring(root, encoding="unicode", xml_declaration=True)


def create_sepa_credit_transfer(
    debtor_name: str,
    debtor_iban: str,
    debtor_bic: str,
    creditor_name: str,
    creditor_iban: str,
    creditor_bic: str,
    amount: Decimal,
    remittance_info: str = "",
    instant: bool = False,
) -> SEPACreditTransfer:
    """
    Create a simple SEPA Credit Transfer.

    Args:
        debtor_name: Name of the payer
        debtor_iban: Payer's IBAN
        debtor_bic: Payer's bank BIC
        creditor_name: Name of the payee
        creditor_iban: Payee's IBAN
        creditor_bic: Payee's bank BIC
        amount: Payment amount
        remittance_info: Optional payment reference
        instant: If True, create an instant payment (SCT Inst)

    Returns:
        SEPACreditTransfer message
    """
    builder = SEPABuilder()

    scheme = SEPAScheme.SCT_INST if instant else SEPAScheme.SCT
    builder.set_scheme(scheme)
    builder.set_amount(amount)
    builder.set_debtor(debtor_name, debtor_iban, debtor_bic)
    builder.set_creditor(creditor_name, creditor_iban, creditor_bic)

    if remittance_info:
        builder.set_remittance_info(unstructured=remittance_info)

    return builder.build_credit_transfer()


def create_sepa_direct_debit(
    creditor_name: str,
    creditor_iban: str,
    creditor_bic: str,
    creditor_id: str,
    debtor_name: str,
    debtor_iban: str,
    debtor_bic: str,
    amount: Decimal,
    mandate_id: str,
    mandate_date: date,
    remittance_info: str = "",
    sequence_type: SEPASequenceType = SEPASequenceType.RCUR,
    scheme: SEPAScheme = SEPAScheme.SDD_CORE,
) -> SEPADirectDebit:
    """
    Create a SEPA Direct Debit.

    Args:
        creditor_name: Name of the collector
        creditor_iban: Collector's IBAN
        creditor_bic: Collector's bank BIC
        creditor_id: Creditor Identifier
        debtor_name: Name of the payer
        debtor_iban: Payer's IBAN
        debtor_bic: Payer's bank BIC
        amount: Collection amount
        mandate_id: Mandate reference
        mandate_date: Date mandate was signed
        remittance_info: Optional payment reference
        sequence_type: First, Recurring, Final, or One-off
        scheme: SDD Core or SDD B2B

    Returns:
        SEPADirectDebit message
    """
    builder = SEPABuilder()

    builder.set_scheme(scheme)
    builder.set_amount(amount)
    builder.set_creditor(creditor_name, creditor_iban, creditor_bic)
    builder.set_creditor_scheme_id(creditor_id)
    builder.set_debtor(debtor_name, debtor_iban, debtor_bic)
    builder.set_mandate(mandate_id, mandate_date)
    builder.set_sequence_type(sequence_type)

    if remittance_info:
        builder.set_remittance_info(unstructured=remittance_info)

    return builder.build_direct_debit()
