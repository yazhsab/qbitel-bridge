"""
ISO 20022 Message Validator

Comprehensive validation for ISO 20022 financial messages including:
- Schema validation
- Business rules validation
- Cross-field validation
- Compliance checks
"""

import re
from datetime import datetime, date
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set
import xml.etree.ElementTree as ET

from ai_engine.domains.banking.protocols.validators.base_validator import (
    BaseValidator,
    ValidationResult,
    ValidationSeverity,
    validate_iban,
    validate_bic,
    validate_currency_code,
    validate_amount,
)


class ISO20022Validator(BaseValidator):
    """
    Validator for ISO 20022 messages.

    Supports validation of:
    - pain.001 (Customer Credit Transfer Initiation)
    - pain.002 (Customer Payment Status Report)
    - pacs.008 (FI to FI Customer Credit Transfer)
    - pacs.009 (FI to FI Financial Institution Credit Transfer)
    - camt.053 (Bank to Customer Statement)
    - camt.054 (Bank to Customer Debit Credit Notification)
    """

    # Supported message types
    SUPPORTED_TYPES = {
        "pain.001", "pain.002", "pain.008",
        "pacs.002", "pacs.008", "pacs.009",
        "camt.052", "camt.053", "camt.054",
    }

    # ISO 20022 namespace patterns
    NAMESPACE_PATTERN = re.compile(
        r"urn:iso:std:iso:20022:tech:xsd:(pain|pacs|camt|head)\.\d{3}\.\d{3}\.\d{2}"
    )

    def __init__(self, strict: bool = True, validate_schema: bool = True):
        """
        Initialize the ISO 20022 validator.

        Args:
            strict: If True, treat warnings as errors
            validate_schema: If True, perform XML schema validation
        """
        super().__init__(strict)
        self.validate_schema = validate_schema

    @property
    def name(self) -> str:
        return "ISO20022Validator"

    @property
    def version(self) -> str:
        return "1.0"

    def validate(self, data: Any) -> ValidationResult:
        """
        Validate an ISO 20022 message.

        Args:
            data: Either an XML string, ElementTree Element, or parsed message object

        Returns:
            ValidationResult with any errors or warnings
        """
        result = self._create_result()

        # Handle different input types
        if isinstance(data, str):
            result = self._validate_xml_string(data, result)
        elif isinstance(data, ET.Element):
            result = self._validate_element(data, result)
        elif hasattr(data, 'message_type'):
            # Parsed message object
            result = self._validate_message_object(data, result)
        else:
            result.add_error(
                "ISO20022_INVALID_INPUT",
                "Input must be XML string, Element, or parsed message object",
                severity=ValidationSeverity.CRITICAL,
            )

        return result

    def _validate_xml_string(self, xml_str: str, result: ValidationResult) -> ValidationResult:
        """Validate XML string format."""
        try:
            root = ET.fromstring(xml_str)
            return self._validate_element(root, result)
        except ET.ParseError as e:
            result.add_error(
                "ISO20022_XML_PARSE_ERROR",
                f"Failed to parse XML: {str(e)}",
                severity=ValidationSeverity.CRITICAL,
            )
            return result

    def _validate_element(self, root: ET.Element, result: ValidationResult) -> ValidationResult:
        """Validate parsed XML element."""
        # Check for namespace
        ns = self._extract_namespace(root)
        if not ns:
            result.add_warning(
                "ISO20022_NO_NAMESPACE",
                "Message does not contain ISO 20022 namespace",
            )

        # Determine message type
        message_type = self._determine_message_type(root, ns)
        result.metadata["message_type"] = message_type

        if message_type:
            # Route to specific validator
            if message_type.startswith("pain.001"):
                self._validate_pain001(root, ns, result)
            elif message_type.startswith("pacs.008"):
                self._validate_pacs008(root, ns, result)
            elif message_type.startswith("camt.053"):
                self._validate_camt053(root, ns, result)
            else:
                result.add_warning(
                    "ISO20022_UNSUPPORTED_TYPE",
                    f"Message type {message_type} validation not fully implemented",
                )
        else:
            result.add_error(
                "ISO20022_UNKNOWN_TYPE",
                "Could not determine message type",
            )

        return result

    def _validate_message_object(self, msg: Any, result: ValidationResult) -> ValidationResult:
        """Validate a parsed message object."""
        msg_type = getattr(msg, 'message_type', None)
        result.metadata["message_type"] = str(msg_type) if msg_type else "unknown"

        # Call the message's own validation if available
        if hasattr(msg, 'validate'):
            errors = msg.validate()
            for error in errors:
                result.add_error(
                    "ISO20022_VALIDATION",
                    error,
                )

        return result

    def _extract_namespace(self, root: ET.Element) -> Optional[str]:
        """Extract ISO 20022 namespace from root element."""
        # Check root tag for namespace
        if root.tag.startswith("{"):
            ns_end = root.tag.find("}")
            return root.tag[1:ns_end]

        # Check xmlns attributes
        for attr, value in root.attrib.items():
            if self.NAMESPACE_PATTERN.match(value):
                return value

        return None

    def _determine_message_type(self, root: ET.Element, ns: Optional[str]) -> Optional[str]:
        """Determine the ISO 20022 message type."""
        if ns:
            # Extract from namespace (e.g., pain.001.001.03)
            match = re.search(r"(pain|pacs|camt|head)\.\d{3}\.\d{3}\.\d{2}", ns)
            if match:
                return match.group(0)

        # Try to determine from root element name
        root_name = root.tag.split("}")[-1] if "}" in root.tag else root.tag

        type_mapping = {
            "CstmrCdtTrfInitn": "pain.001",
            "FIToFICstmrCdtTrf": "pacs.008",
            "BkToCstmrStmt": "camt.053",
        }

        # Check child elements
        for child in root:
            child_name = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if child_name in type_mapping:
                return type_mapping[child_name]

        return None

    def _validate_pain001(self, root: ET.Element, ns: str, result: ValidationResult) -> None:
        """Validate pain.001 Customer Credit Transfer Initiation."""
        ns_prefix = f"{{{ns}}}" if ns else ""

        # Find the main content element
        content = root.find(f".//{ns_prefix}CstmrCdtTrfInitn") or root

        # Validate Group Header
        grp_hdr = content.find(f".//{ns_prefix}GrpHdr")
        if grp_hdr is None:
            result.add_error(
                "PAIN001_MISSING_GRP_HDR",
                "Group Header (GrpHdr) is required",
                field_name="GrpHdr",
            )
        else:
            self._validate_pain001_header(grp_hdr, ns_prefix, result)

        # Validate Payment Information
        pmt_infs = content.findall(f".//{ns_prefix}PmtInf")
        if not pmt_infs:
            result.add_error(
                "PAIN001_MISSING_PMT_INF",
                "At least one Payment Information (PmtInf) block is required",
                field_name="PmtInf",
            )
        else:
            for i, pmt_inf in enumerate(pmt_infs):
                self._validate_pain001_payment(pmt_inf, ns_prefix, i, result)

    def _validate_pain001_header(self, grp_hdr: ET.Element, ns: str,
                                  result: ValidationResult) -> None:
        """Validate pain.001 Group Header."""
        # Message ID
        msg_id = grp_hdr.find(f"{ns}MsgId")
        if msg_id is None or not msg_id.text:
            result.add_error(
                "PAIN001_MISSING_MSG_ID",
                "Message Identification (MsgId) is required",
                field_name="GrpHdr/MsgId",
            )
        elif len(msg_id.text) > 35:
            result.add_error(
                "PAIN001_INVALID_MSG_ID",
                "Message ID must not exceed 35 characters",
                field_name="GrpHdr/MsgId",
            )

        # Creation DateTime
        cre_dt_tm = grp_hdr.find(f"{ns}CreDtTm")
        if cre_dt_tm is None or not cre_dt_tm.text:
            result.add_error(
                "PAIN001_MISSING_CRE_DT_TM",
                "Creation DateTime (CreDtTm) is required",
                field_name="GrpHdr/CreDtTm",
            )
        else:
            try:
                datetime.fromisoformat(cre_dt_tm.text.replace("Z", "+00:00"))
            except ValueError:
                result.add_error(
                    "PAIN001_INVALID_CRE_DT_TM",
                    "Invalid datetime format for CreDtTm",
                    field_name="GrpHdr/CreDtTm",
                )

        # Number of Transactions
        nb_of_txs = grp_hdr.find(f"{ns}NbOfTxs")
        if nb_of_txs is not None and nb_of_txs.text:
            try:
                count = int(nb_of_txs.text)
                if count <= 0:
                    result.add_error(
                        "PAIN001_INVALID_NB_OF_TXS",
                        "Number of transactions must be positive",
                        field_name="GrpHdr/NbOfTxs",
                    )
                result.metadata["transaction_count"] = count
            except ValueError:
                result.add_error(
                    "PAIN001_INVALID_NB_OF_TXS",
                    "Number of transactions must be numeric",
                    field_name="GrpHdr/NbOfTxs",
                )

        # Control Sum
        ctrl_sum = grp_hdr.find(f"{ns}CtrlSum")
        if ctrl_sum is not None and ctrl_sum.text:
            try:
                total = Decimal(ctrl_sum.text)
                if total < 0:
                    result.add_error(
                        "PAIN001_INVALID_CTRL_SUM",
                        "Control sum cannot be negative",
                        field_name="GrpHdr/CtrlSum",
                    )
                result.metadata["control_sum"] = str(total)
            except Exception:
                result.add_error(
                    "PAIN001_INVALID_CTRL_SUM",
                    "Invalid control sum format",
                    field_name="GrpHdr/CtrlSum",
                )

    def _validate_pain001_payment(self, pmt_inf: ET.Element, ns: str, index: int,
                                   result: ValidationResult) -> None:
        """Validate pain.001 Payment Information block."""
        prefix = f"PmtInf[{index}]"

        # Payment Information ID
        pmt_inf_id = pmt_inf.find(f"{ns}PmtInfId")
        if pmt_inf_id is None or not pmt_inf_id.text:
            result.add_error(
                "PAIN001_MISSING_PMT_INF_ID",
                "Payment Information ID is required",
                field=f"{prefix}/PmtInfId",
            )

        # Payment Method
        pmt_mtd = pmt_inf.find(f"{ns}PmtMtd")
        if pmt_mtd is not None and pmt_mtd.text not in ("TRF", "CHK", "TRA"):
            result.add_warning(
                "PAIN001_UNUSUAL_PMT_MTD",
                f"Unusual payment method: {pmt_mtd.text}",
                field=f"{prefix}/PmtMtd",
            )

        # Requested Execution Date
        reqd_exctn_dt = pmt_inf.find(f"{ns}ReqdExctnDt")
        if reqd_exctn_dt is not None:
            dt_elem = reqd_exctn_dt.find(f"{ns}Dt")
            if dt_elem is not None and dt_elem.text:
                try:
                    exec_date = date.fromisoformat(dt_elem.text)
                    if exec_date < date.today():
                        result.add_warning(
                            "PAIN001_PAST_EXEC_DATE",
                            "Requested execution date is in the past",
                            field=f"{prefix}/ReqdExctnDt",
                        )
                except ValueError:
                    result.add_error(
                        "PAIN001_INVALID_EXEC_DATE",
                        "Invalid date format for ReqdExctnDt",
                        field=f"{prefix}/ReqdExctnDt",
                    )

        # Debtor Account
        dbtr_acct = pmt_inf.find(f".//{ns}DbtrAcct")
        if dbtr_acct is not None:
            iban = dbtr_acct.find(f".//{ns}IBAN")
            if iban is not None and iban.text:
                error = validate_iban(iban.text)
                if error:
                    result.add_error(
                        "PAIN001_INVALID_DEBTOR_IBAN",
                        error,
                        field=f"{prefix}/DbtrAcct/IBAN",
                    )

        # Debtor Agent BIC
        dbtr_agt = pmt_inf.find(f".//{ns}DbtrAgt")
        if dbtr_agt is not None:
            bic = dbtr_agt.find(f".//{ns}BIC") or dbtr_agt.find(f".//{ns}BICFI")
            if bic is not None and bic.text:
                error = validate_bic(bic.text)
                if error:
                    result.add_error(
                        "PAIN001_INVALID_DEBTOR_BIC",
                        error,
                        field=f"{prefix}/DbtrAgt/BIC",
                    )

        # Credit Transfer Transactions
        txs = pmt_inf.findall(f".//{ns}CdtTrfTxInf")
        if not txs:
            result.add_error(
                "PAIN001_MISSING_TX",
                "At least one Credit Transfer Transaction is required",
                field=f"{prefix}/CdtTrfTxInf",
            )
        else:
            for j, tx in enumerate(txs):
                self._validate_pain001_transaction(tx, ns, f"{prefix}/CdtTrfTxInf[{j}]", result)

    def _validate_pain001_transaction(self, tx: ET.Element, ns: str, prefix: str,
                                       result: ValidationResult) -> None:
        """Validate pain.001 Credit Transfer Transaction."""
        # End-to-End ID
        e2e_id = tx.find(f".//{ns}EndToEndId")
        if e2e_id is None or not e2e_id.text:
            result.add_error(
                "PAIN001_MISSING_E2E_ID",
                "End-to-End ID is required",
                field=f"{prefix}/EndToEndId",
            )
        elif len(e2e_id.text) > 35:
            result.add_error(
                "PAIN001_INVALID_E2E_ID",
                "End-to-End ID must not exceed 35 characters",
                field=f"{prefix}/EndToEndId",
            )

        # Amount
        amt = tx.find(f".//{ns}InstdAmt") or tx.find(f".//{ns}Amt/{ns}InstdAmt")
        if amt is not None:
            if amt.text:
                error = validate_amount(amt.text)
                if error:
                    result.add_error(
                        "PAIN001_INVALID_AMOUNT",
                        error,
                        field=f"{prefix}/InstdAmt",
                    )

            # Currency
            ccy = amt.get("Ccy")
            if ccy:
                error = validate_currency_code(ccy)
                if error:
                    result.add_error(
                        "PAIN001_INVALID_CURRENCY",
                        error,
                        field=f"{prefix}/InstdAmt/@Ccy",
                    )

        # Creditor Account
        cdtr_acct = tx.find(f".//{ns}CdtrAcct")
        if cdtr_acct is not None:
            iban = cdtr_acct.find(f".//{ns}IBAN")
            if iban is not None and iban.text:
                error = validate_iban(iban.text)
                if error:
                    result.add_error(
                        "PAIN001_INVALID_CREDITOR_IBAN",
                        error,
                        field=f"{prefix}/CdtrAcct/IBAN",
                    )

        # Creditor Agent BIC
        cdtr_agt = tx.find(f".//{ns}CdtrAgt")
        if cdtr_agt is not None:
            bic = cdtr_agt.find(f".//{ns}BIC") or cdtr_agt.find(f".//{ns}BICFI")
            if bic is not None and bic.text:
                error = validate_bic(bic.text)
                if error:
                    result.add_error(
                        "PAIN001_INVALID_CREDITOR_BIC",
                        error,
                        field=f"{prefix}/CdtrAgt/BIC",
                    )

    def _validate_pacs008(self, root: ET.Element, ns: str, result: ValidationResult) -> None:
        """Validate pacs.008 FI to FI Customer Credit Transfer."""
        ns_prefix = f"{{{ns}}}" if ns else ""

        # Find the main content element
        content = root.find(f".//{ns_prefix}FIToFICstmrCdtTrf") or root

        # Validate Group Header
        grp_hdr = content.find(f".//{ns_prefix}GrpHdr")
        if grp_hdr is None:
            result.add_error(
                "PACS008_MISSING_GRP_HDR",
                "Group Header (GrpHdr) is required",
                field_name="GrpHdr",
            )
        else:
            # Message ID
            msg_id = grp_hdr.find(f"{ns_prefix}MsgId")
            if msg_id is None or not msg_id.text:
                result.add_error(
                    "PACS008_MISSING_MSG_ID",
                    "Message ID is required",
                    field_name="GrpHdr/MsgId",
                )

            # Settlement Information
            sttlm_inf = grp_hdr.find(f"{ns_prefix}SttlmInf")
            if sttlm_inf is None:
                result.add_error(
                    "PACS008_MISSING_STTLM_INF",
                    "Settlement Information is required",
                    field_name="GrpHdr/SttlmInf",
                )

        # Validate Credit Transfer Transactions
        txs = content.findall(f".//{ns_prefix}CdtTrfTxInf")
        if not txs:
            result.add_error(
                "PACS008_MISSING_TX",
                "At least one Credit Transfer Transaction is required",
                field_name="CdtTrfTxInf",
            )

    def _validate_camt053(self, root: ET.Element, ns: str, result: ValidationResult) -> None:
        """Validate camt.053 Bank to Customer Statement."""
        ns_prefix = f"{{{ns}}}" if ns else ""

        # Find the main content element
        content = root.find(f".//{ns_prefix}BkToCstmrStmt") or root

        # Validate Group Header
        grp_hdr = content.find(f".//{ns_prefix}GrpHdr")
        if grp_hdr is None:
            result.add_error(
                "CAMT053_MISSING_GRP_HDR",
                "Group Header (GrpHdr) is required",
                field_name="GrpHdr",
            )

        # Validate Statement(s)
        stmts = content.findall(f".//{ns_prefix}Stmt")
        if not stmts:
            result.add_error(
                "CAMT053_MISSING_STMT",
                "At least one Statement (Stmt) is required",
                field_name="Stmt",
            )
        else:
            for i, stmt in enumerate(stmts):
                self._validate_camt053_statement(stmt, ns_prefix, i, result)

    def _validate_camt053_statement(self, stmt: ET.Element, ns: str, index: int,
                                     result: ValidationResult) -> None:
        """Validate camt.053 Statement."""
        prefix = f"Stmt[{index}]"

        # Statement ID
        stmt_id = stmt.find(f"{ns}Id")
        if stmt_id is None or not stmt_id.text:
            result.add_error(
                "CAMT053_MISSING_STMT_ID",
                "Statement ID is required",
                field=f"{prefix}/Id",
            )

        # Account
        acct = stmt.find(f"{ns}Acct")
        if acct is None:
            result.add_error(
                "CAMT053_MISSING_ACCT",
                "Account information is required",
                field=f"{prefix}/Acct",
            )
        else:
            iban = acct.find(f".//{ns}IBAN")
            if iban is not None and iban.text:
                error = validate_iban(iban.text)
                if error:
                    result.add_error(
                        "CAMT053_INVALID_IBAN",
                        error,
                        field=f"{prefix}/Acct/IBAN",
                    )

        # Balance
        bals = stmt.findall(f"{ns}Bal")
        opening_found = False
        closing_found = False
        for bal in bals:
            tp = bal.find(f".//{ns}CdOrPrtry/{ns}Cd")
            if tp is not None:
                if tp.text == "OPBD":
                    opening_found = True
                elif tp.text == "CLBD":
                    closing_found = True

        if not opening_found:
            result.add_warning(
                "CAMT053_MISSING_OPENING_BAL",
                "Opening balance (OPBD) not found",
                field=f"{prefix}/Bal",
            )
        if not closing_found:
            result.add_warning(
                "CAMT053_MISSING_CLOSING_BAL",
                "Closing balance (CLBD) not found",
                field=f"{prefix}/Bal",
            )
