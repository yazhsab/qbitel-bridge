"""
SWIFT/ISO 20022 Banking Message Generator

Generates realistic SWIFT MT messages and ISO 20022 XML messages for training
the protocol discovery, field detection, and PQC security models.

Part A - Protocol Samples (binary + metadata):
  - MT103: Single Customer Credit Transfer
  - MT202: General Financial Institution Transfer
  - pacs.008: FIToFICustomerCreditTransfer (ISO 20022)
  - pacs.009: FinancialInstitutionCreditTransfer (ISO 20022)
  - camt.053: BankToCustomerStatement (ISO 20022)

Part B - Instruction Pairs (JSONL):
  - swift_security: PQC integration for SWIFT messaging
  - transaction_monitoring: Anomalous transaction detection, AML screening
  - regulatory_compliance: Basel III/IV, PCI-DSS, PSD2 compliance with PQC
  - correspondent_banking: Multi-hop proxy re-encryption, chain-of-trust
"""

import json
import random
import uuid
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class SwiftBankingGenerator:
    """Generate realistic SWIFT/ISO 20022 messages and banking security instruction pairs."""

    # ----------------------------------------------------------------
    # Constants
    # ----------------------------------------------------------------

    BIC_CODES = [
        "BOFAUS3N", "CHASUS33", "DEUTDEFF", "BNPAFRPP", "HSBCGB2L",
        "CITIUS33", "WFBIUS6S", "BARCGB22", "SCBLSGSG", "UBSWCHZH",
        "ANZBAU3M", "NWBKGB2L", "LOYDGB2L", "SOGEGB2L", "MIDLGB22",
        "COBADEFF", "BNKAUS33", "MABORUMM", "ABNAPL2A", "INGBNL2A",
        "RBOSGB2L", "USBKUS44", "PNBPUS3N", "MHCBUS33", "MELNGB2X",
    ]

    CURRENCY_CODES = ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "SGD", "HKD", "SEK"]

    CURRENCY_DECIMAL = {
        "USD": 2, "EUR": 2, "GBP": 2, "JPY": 0, "CHF": 2,
        "AUD": 2, "CAD": 2, "SGD": 2, "HKD": 2, "SEK": 2,
    }

    CHARGE_CODES = ["SHA", "OUR", "BEN"]

    BANK_OP_CODES = ["CRED", "SPAY", "SSTD"]

    CUSTOMER_NAMES = [
        "Northwind Trading Corp", "Contoso Financial Ltd", "Fabrikam Industries AG",
        "Alpine Commodities GmbH", "Pacific Rim Holdings Pte", "Atlantic Bridge Capital LLC",
        "Meridian Exports SA", "Quantum Ventures BV", "Pinnacle Resources Inc",
        "Sterling Partners LLP", "Evergreen Logistics KK", "Vanguard Shipping Co",
        "Summit Advisory Group", "Azure Dynamics Corp", "Golden Gate Enterprises",
        "Crescent Energy Ltd", "Horizon Pharmaceuticals AG", "Olympus Trading House",
        "Silverline Manufacturing Pty", "Redwood Capital Partners",
    ]

    CUSTOMER_ACCOUNTS = [
        "DE89370400440532013000", "GB29NWBK60161331926819", "FR7630006000011234567890189",
        "CH9300762011623852957", "US64SVBKUS6S3300958879", "NL91ABNA0417164300",
        "SG68OCBC5832038571288", "AU32ANZBAU3M001234567", "JP82MUFGJPJT0123456789",
        "HK82HSBC12345678901234", "SE3550000000054910000003", "CA73ROYA00000260000037",
    ]

    ADDRESSES = [
        "1 Wall Street, New York, NY 10005, US",
        "25 Canada Square, London E14 5LQ, GB",
        "Taunusanlage 12, 60325 Frankfurt, DE",
        "16 Boulevard des Italiens, 75009 Paris, FR",
        "8 Marina Boulevard, Singapore 018981, SG",
        "Paradeplatz 8, 8001 Zurich, CH",
        "Level 28, 385 Bourke Street, Melbourne VIC 3000, AU",
        "1-3 Marunouchi, Chiyoda-ku, Tokyo 100-8388, JP",
    ]

    MESSAGE_TYPES = {
        "mt103": "Single Customer Credit Transfer",
        "mt202": "General Financial Institution Transfer",
        "pacs008": "FIToFICustomerCreditTransfer",
        "pacs009": "FinancialInstitutionCreditTransfer",
        "camt053": "BankToCustomerStatement",
    }

    INSTRUCTION_CATEGORIES = [
        "swift_security",
        "transaction_monitoring",
        "regulatory_compliance",
        "correspondent_banking",
    ]

    DIFFICULTIES = ["basic", "intermediate", "advanced"]

    def __init__(self, seed: Optional[int] = None):
        if seed:
            random.seed(seed)
        self.ref_counter = random.randint(100000, 999999)

    # ----------------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------------

    def _next_ref(self, prefix: str = "REF") -> str:
        """Generate next sequential reference number."""
        self.ref_counter += 1
        return f"{prefix}{self.ref_counter:010d}"

    def _random_amount(self, low: float = 100.0, high: float = 50_000_000.0) -> float:
        """Generate a realistic transaction amount with log-normal-ish distribution."""
        # Most transactions are small; a few are very large
        bucket = random.random()
        if bucket < 0.50:
            return round(random.uniform(low, 10_000.0), 2)
        elif bucket < 0.80:
            return round(random.uniform(10_000.0, 500_000.0), 2)
        elif bucket < 0.95:
            return round(random.uniform(500_000.0, 5_000_000.0), 2)
        else:
            return round(random.uniform(5_000_000.0, high), 2)

    def _random_date(self, days_back: int = 60) -> datetime:
        """Generate a random datetime within the given window."""
        return datetime.now() - timedelta(
            days=random.randint(0, days_back),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59),
        )

    def _swift_date(self, dt: datetime) -> str:
        """Format date as SWIFT YYMMDD."""
        return dt.strftime("%y%m%d")

    def _iso_datetime(self, dt: datetime) -> str:
        """Format datetime as ISO 20022 ISODateTime."""
        return dt.strftime("%Y-%m-%dT%H:%M:%S")

    def _format_amount(self, amount: float, currency: str) -> str:
        """Format amount with proper decimal places for SWIFT."""
        decimals = self.CURRENCY_DECIMAL.get(currency, 2)
        if decimals == 0:
            return str(int(amount))
        return f"{amount:.{decimals}f}".replace(".", ",")

    def _pick_bic_pair(self) -> Tuple[str, str]:
        """Pick two distinct BIC codes."""
        pair = random.sample(self.BIC_CODES, 2)
        return pair[0], pair[1]

    # ----------------------------------------------------------------
    # Part A: SWIFT MT message generators
    # ----------------------------------------------------------------

    def generate_mt103(self) -> Tuple[bytes, Dict]:
        """Generate MT103 Single Customer Credit Transfer."""
        dt = self._random_date(days_back=30)
        sender_bic, receiver_bic = self._pick_bic_pair()
        currency = random.choice(self.CURRENCY_CODES)
        amount = self._random_amount(100.0, 10_000_000.0)
        txn_ref = self._next_ref("FT")
        ordering_name = random.choice(self.CUSTOMER_NAMES)
        ordering_acct = random.choice(self.CUSTOMER_ACCOUNTS)
        ordering_addr = random.choice(self.ADDRESSES)
        beneficiary_name = random.choice(self.CUSTOMER_NAMES)
        beneficiary_acct = random.choice(self.CUSTOMER_ACCOUNTS)
        beneficiary_addr = random.choice(self.ADDRESSES)
        bank_op = random.choice(self.BANK_OP_CODES)
        charges = random.choice(self.CHARGE_CODES)

        # Build SWIFT block structure
        block1 = f"{{1:F01{sender_bic}XXXX0000000000}}"
        block2 = f"{{2:I103{receiver_bic}XXXXN}}"
        block3 = "{3:{108:" + txn_ref + "}}"

        amount_str = self._format_amount(amount, currency)
        value_date = self._swift_date(dt)

        block4_lines = [
            f":20:{txn_ref}",
            f":23B:{bank_op}",
            f":32A:{value_date}{currency}{amount_str}",
            f":50K:/{ordering_acct}",
            f"{ordering_name}",
            f"{ordering_addr}",
            f":59:/{beneficiary_acct}",
            f"{beneficiary_name}",
            f"{beneficiary_addr}",
            f":71A:{charges}",
        ]

        # Optional remittance information
        if random.random() < 0.6:
            remit_ref = self._next_ref("INV")
            block4_lines.append(f":70:/INV/{remit_ref}")

        # Optional intermediary bank
        intermediary_bic = None
        if random.random() < 0.3:
            intermediary_bic = random.choice(
                [b for b in self.BIC_CODES if b != sender_bic and b != receiver_bic]
            )
            block4_lines.insert(6, f":56A:{intermediary_bic}")

        block4_text = "\r\n".join(block4_lines)
        block4 = "{4:\r\n" + block4_text + "\r\n-}"

        block5 = "{5:{MAC:00000000}{CHK:000000000000}}"

        message_text = block1 + block2 + block3 + block4 + block5
        message_bytes = message_text.encode("ascii")

        metadata = {
            "protocol": "swift_mt",
            "message_type": "MT103",
            "description": "Single Customer Credit Transfer",
            "timestamp": datetime.now().isoformat(),
            "message_length": len(message_bytes),
            "fields": {
                "transaction_ref": txn_ref,
                "bank_operation_code": bank_op,
                "value_date": value_date,
                "currency": currency,
                "amount": amount,
                "sender_bic": sender_bic,
                "receiver_bic": receiver_bic,
                "ordering_customer": ordering_name,
                "ordering_account": ordering_acct,
                "beneficiary": beneficiary_name,
                "beneficiary_account": beneficiary_acct,
                "charges": charges,
                "intermediary_bic": intermediary_bic,
            },
            "hash": hashlib.sha256(message_bytes).hexdigest(),
        }

        return message_bytes, metadata

    def generate_mt202(self) -> Tuple[bytes, Dict]:
        """Generate MT202 General Financial Institution Transfer."""
        dt = self._random_date(days_back=30)
        sender_bic, receiver_bic = self._pick_bic_pair()
        currency = random.choice(self.CURRENCY_CODES)
        amount = self._random_amount(50_000.0, 50_000_000.0)
        txn_ref = self._next_ref("COV")
        related_ref = self._next_ref("FT")
        ordering_inst = random.choice(self.BIC_CODES)
        beneficiary_inst_bic = random.choice(
            [b for b in self.BIC_CODES if b != sender_bic and b != receiver_bic]
        )

        value_date = self._swift_date(dt)
        amount_str = self._format_amount(amount, currency)

        block1 = f"{{1:F01{sender_bic}XXXX0000000000}}"
        block2 = f"{{2:I202{receiver_bic}XXXXN}}"
        block3 = "{3:{108:" + txn_ref + "}}"

        block4_lines = [
            f":20:{txn_ref}",
            f":21:{related_ref}",
            f":32A:{value_date}{currency}{amount_str}",
            f":52A:{ordering_inst}",
            f":58A:{beneficiary_inst_bic}",
        ]

        # Optional account with institution
        if random.random() < 0.4:
            acct_inst = random.choice(
                [b for b in self.BIC_CODES if b not in (sender_bic, receiver_bic, beneficiary_inst_bic)]
            )
            block4_lines.insert(4, f":57A:{acct_inst}")

        block4_text = "\r\n".join(block4_lines)
        block4 = "{4:\r\n" + block4_text + "\r\n-}"
        block5 = "{5:{MAC:00000000}{CHK:000000000000}}"

        message_text = block1 + block2 + block3 + block4 + block5
        message_bytes = message_text.encode("ascii")

        metadata = {
            "protocol": "swift_mt",
            "message_type": "MT202",
            "description": "General Financial Institution Transfer",
            "timestamp": datetime.now().isoformat(),
            "message_length": len(message_bytes),
            "fields": {
                "transaction_ref": txn_ref,
                "related_ref": related_ref,
                "value_date": value_date,
                "currency": currency,
                "amount": amount,
                "sender_bic": sender_bic,
                "receiver_bic": receiver_bic,
                "ordering_institution": ordering_inst,
                "beneficiary_institution": beneficiary_inst_bic,
            },
            "hash": hashlib.sha256(message_bytes).hexdigest(),
        }

        return message_bytes, metadata

    # ----------------------------------------------------------------
    # Part A: ISO 20022 XML message generators
    # ----------------------------------------------------------------

    def generate_pacs008(self) -> Tuple[bytes, Dict]:
        """Generate pacs.008 FIToFICustomerCreditTransfer (ISO 20022 XML)."""
        dt = self._random_date(days_back=30)
        msg_id = self._next_ref("MSG")
        instr_id = self._next_ref("INSTR")
        end_to_end_id = self._next_ref("E2E")
        tx_id = self._next_ref("TXN")

        sender_bic, receiver_bic = self._pick_bic_pair()
        currency = random.choice(self.CURRENCY_CODES)
        amount = self._random_amount(100.0, 10_000_000.0)
        debtor_name = random.choice(self.CUSTOMER_NAMES)
        debtor_acct = random.choice(self.CUSTOMER_ACCOUNTS)
        creditor_name = random.choice(self.CUSTOMER_NAMES)
        creditor_acct = random.choice(self.CUSTOMER_ACCOUNTS)
        debtor_addr = random.choice(self.ADDRESSES)
        creditor_addr = random.choice(self.ADDRESSES)
        charge_bearer = random.choice(["SLEV", "SHAR", "DEBT", "CRED"])

        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Document xmlns="urn:iso:std:iso:20022:tech:xsd:pacs.008.001.08">
  <FIToFICstmrCdtTrf>
    <GrpHdr>
      <MsgId>{msg_id}</MsgId>
      <CreDtTm>{self._iso_datetime(dt)}</CreDtTm>
      <NbOfTxs>1</NbOfTxs>
      <TtlIntrBkSttlmAmt Ccy="{currency}">{amount:.2f}</TtlIntrBkSttlmAmt>
      <IntrBkSttlmDt>{dt.strftime('%Y-%m-%d')}</IntrBkSttlmDt>
      <SttlmInf>
        <SttlmMtd>CLRG</SttlmMtd>
      </SttlmInf>
      <InstgAgt>
        <FinInstnId><BICFI>{sender_bic}</BICFI></FinInstnId>
      </InstgAgt>
      <InstdAgt>
        <FinInstnId><BICFI>{receiver_bic}</BICFI></FinInstnId>
      </InstdAgt>
    </GrpHdr>
    <CdtTrfTxInf>
      <PmtId>
        <InstrId>{instr_id}</InstrId>
        <EndToEndId>{end_to_end_id}</EndToEndId>
        <TxId>{tx_id}</TxId>
      </PmtId>
      <IntrBkSttlmAmt Ccy="{currency}">{amount:.2f}</IntrBkSttlmAmt>
      <ChrgBr>{charge_bearer}</ChrgBr>
      <Dbtr>
        <Nm>{debtor_name}</Nm>
        <PstlAdr><AdrLine>{debtor_addr}</AdrLine></PstlAdr>
      </Dbtr>
      <DbtrAcct>
        <Id><IBAN>{debtor_acct}</IBAN></Id>
      </DbtrAcct>
      <DbtrAgt>
        <FinInstnId><BICFI>{sender_bic}</BICFI></FinInstnId>
      </DbtrAgt>
      <CdtrAgt>
        <FinInstnId><BICFI>{receiver_bic}</BICFI></FinInstnId>
      </CdtrAgt>
      <Cdtr>
        <Nm>{creditor_name}</Nm>
        <PstlAdr><AdrLine>{creditor_addr}</AdrLine></PstlAdr>
      </Cdtr>
      <CdtrAcct>
        <Id><IBAN>{creditor_acct}</IBAN></Id>
      </CdtrAcct>
      <RmtInf>
        <Ustrd>Payment ref {end_to_end_id}</Ustrd>
      </RmtInf>
    </CdtTrfTxInf>
  </FIToFICstmrCdtTrf>
</Document>"""

        message_bytes = xml.encode("utf-8")

        metadata = {
            "protocol": "iso20022",
            "message_type": "pacs.008.001.08",
            "description": "FIToFICustomerCreditTransfer",
            "timestamp": datetime.now().isoformat(),
            "message_length": len(message_bytes),
            "fields": {
                "msg_id": msg_id,
                "instruction_id": instr_id,
                "end_to_end_id": end_to_end_id,
                "transaction_id": tx_id,
                "creation_datetime": self._iso_datetime(dt),
                "settlement_date": dt.strftime("%Y-%m-%d"),
                "currency": currency,
                "amount": amount,
                "sender_bic": sender_bic,
                "receiver_bic": receiver_bic,
                "debtor": debtor_name,
                "debtor_account": debtor_acct,
                "creditor": creditor_name,
                "creditor_account": creditor_acct,
                "charge_bearer": charge_bearer,
            },
            "hash": hashlib.sha256(message_bytes).hexdigest(),
        }

        return message_bytes, metadata

    def generate_pacs009(self) -> Tuple[bytes, Dict]:
        """Generate pacs.009 FinancialInstitutionCreditTransfer (ISO 20022 XML)."""
        dt = self._random_date(days_back=30)
        msg_id = self._next_ref("FI2FI")
        instr_id = self._next_ref("INSTR")
        end_to_end_id = self._next_ref("E2E")

        sender_bic, receiver_bic = self._pick_bic_pair()
        debtor_bic = sender_bic
        creditor_bic = random.choice(
            [b for b in self.BIC_CODES if b != sender_bic and b != receiver_bic]
        )
        currency = random.choice(self.CURRENCY_CODES)
        amount = self._random_amount(100_000.0, 50_000_000.0)

        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Document xmlns="urn:iso:std:iso:20022:tech:xsd:pacs.009.001.08">
  <FICdtTrf>
    <GrpHdr>
      <MsgId>{msg_id}</MsgId>
      <CreDtTm>{self._iso_datetime(dt)}</CreDtTm>
      <NbOfTxs>1</NbOfTxs>
      <SttlmInf>
        <SttlmMtd>INDA</SttlmMtd>
      </SttlmInf>
      <InstgAgt>
        <FinInstnId><BICFI>{sender_bic}</BICFI></FinInstnId>
      </InstgAgt>
      <InstdAgt>
        <FinInstnId><BICFI>{receiver_bic}</BICFI></FinInstnId>
      </InstdAgt>
    </GrpHdr>
    <CdtTrfTxInf>
      <PmtId>
        <InstrId>{instr_id}</InstrId>
        <EndToEndId>{end_to_end_id}</EndToEndId>
      </PmtId>
      <IntrBkSttlmAmt Ccy="{currency}">{amount:.2f}</IntrBkSttlmAmt>
      <IntrBkSttlmDt>{dt.strftime('%Y-%m-%d')}</IntrBkSttlmDt>
      <Dbtr>
        <FinInstnId><BICFI>{debtor_bic}</BICFI></FinInstnId>
      </Dbtr>
      <Cdtr>
        <FinInstnId><BICFI>{creditor_bic}</BICFI></FinInstnId>
      </Cdtr>
    </CdtTrfTxInf>
  </FICdtTrf>
</Document>"""

        message_bytes = xml.encode("utf-8")

        metadata = {
            "protocol": "iso20022",
            "message_type": "pacs.009.001.08",
            "description": "FinancialInstitutionCreditTransfer",
            "timestamp": datetime.now().isoformat(),
            "message_length": len(message_bytes),
            "fields": {
                "msg_id": msg_id,
                "instruction_id": instr_id,
                "end_to_end_id": end_to_end_id,
                "creation_datetime": self._iso_datetime(dt),
                "settlement_date": dt.strftime("%Y-%m-%d"),
                "currency": currency,
                "amount": amount,
                "sender_bic": sender_bic,
                "receiver_bic": receiver_bic,
                "debtor_bic": debtor_bic,
                "creditor_bic": creditor_bic,
            },
            "hash": hashlib.sha256(message_bytes).hexdigest(),
        }

        return message_bytes, metadata

    def generate_camt053(self) -> Tuple[bytes, Dict]:
        """Generate camt.053 BankToCustomerStatement (ISO 20022 XML)."""
        dt = self._random_date(days_back=7)
        msg_id = self._next_ref("STMT")
        stmt_id = self._next_ref("ST")
        account_iban = random.choice(self.CUSTOMER_ACCOUNTS)
        account_bic = random.choice(self.BIC_CODES)
        account_name = random.choice(self.CUSTOMER_NAMES)
        currency = random.choice(self.CURRENCY_CODES)

        # Generate statement entries
        num_entries = random.randint(3, 12)
        opening_balance = round(random.uniform(10_000.0, 5_000_000.0), 2)
        balance = opening_balance
        entry_xmls = []
        entry_metadata = []

        for i in range(num_entries):
            is_credit = random.random() < 0.5
            entry_amount = round(random.uniform(100.0, 500_000.0), 2)
            entry_ref = self._next_ref("NTRY")
            counterparty = random.choice(self.CUSTOMER_NAMES)

            if is_credit:
                balance += entry_amount
                cd_indicator = "CRDT"
            else:
                balance -= entry_amount
                cd_indicator = "DBIT"

            entry_dt = dt - timedelta(days=random.randint(0, 5))

            entry_xmls.append(f"""      <Ntry>
        <Amt Ccy="{currency}">{entry_amount:.2f}</Amt>
        <CdtDbtInd>{cd_indicator}</CdtDbtInd>
        <Sts>BOOK</Sts>
        <BookgDt><Dt>{entry_dt.strftime('%Y-%m-%d')}</Dt></BookgDt>
        <AcctSvcrRef>{entry_ref}</AcctSvcrRef>
        <NtryDtls>
          <TxDtls>
            <RltdPties>
              <Dbtr><Nm>{counterparty if cd_indicator == 'CRDT' else account_name}</Nm></Dbtr>
              <Cdtr><Nm>{account_name if cd_indicator == 'CRDT' else counterparty}</Nm></Cdtr>
            </RltdPties>
            <RmtInf><Ustrd>Payment {entry_ref}</Ustrd></RmtInf>
          </TxDtls>
        </NtryDtls>
      </Ntry>""")

            entry_metadata.append({
                "ref": entry_ref,
                "amount": entry_amount,
                "direction": cd_indicator,
                "counterparty": counterparty,
                "booking_date": entry_dt.strftime("%Y-%m-%d"),
            })

        closing_balance = round(balance, 2)
        entries_block = "\n".join(entry_xmls)

        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Document xmlns="urn:iso:std:iso:20022:tech:xsd:camt.053.001.08">
  <BkToCstmrStmt>
    <GrpHdr>
      <MsgId>{msg_id}</MsgId>
      <CreDtTm>{self._iso_datetime(dt)}</CreDtTm>
    </GrpHdr>
    <Stmt>
      <Id>{stmt_id}</Id>
      <CreDtTm>{self._iso_datetime(dt)}</CreDtTm>
      <Acct>
        <Id><IBAN>{account_iban}</IBAN></Id>
        <Ccy>{currency}</Ccy>
        <Ownr><Nm>{account_name}</Nm></Ownr>
        <Svcr>
          <FinInstnId><BICFI>{account_bic}</BICFI></FinInstnId>
        </Svcr>
      </Acct>
      <Bal>
        <Tp><CdOrPrtry><Cd>OPBD</Cd></CdOrPrtry></Tp>
        <Amt Ccy="{currency}">{opening_balance:.2f}</Amt>
        <CdtDbtInd>CRDT</CdtDbtInd>
        <Dt><Dt>{dt.strftime('%Y-%m-%d')}</Dt></Dt>
      </Bal>
      <Bal>
        <Tp><CdOrPrtry><Cd>CLBD</Cd></CdOrPrtry></Tp>
        <Amt Ccy="{currency}">{closing_balance:.2f}</Amt>
        <CdtDbtInd>{"CRDT" if closing_balance >= 0 else "DBIT"}</CdtDbtInd>
        <Dt><Dt>{dt.strftime('%Y-%m-%d')}</Dt></Dt>
      </Bal>
{entries_block}
    </Stmt>
  </BkToCstmrStmt>
</Document>"""

        message_bytes = xml.encode("utf-8")

        metadata = {
            "protocol": "iso20022",
            "message_type": "camt.053.001.08",
            "description": "BankToCustomerStatement",
            "timestamp": datetime.now().isoformat(),
            "message_length": len(message_bytes),
            "fields": {
                "msg_id": msg_id,
                "statement_id": stmt_id,
                "account_iban": account_iban,
                "account_bic": account_bic,
                "account_owner": account_name,
                "currency": currency,
                "opening_balance": opening_balance,
                "closing_balance": closing_balance,
                "num_entries": num_entries,
                "entries": entry_metadata,
            },
            "hash": hashlib.sha256(message_bytes).hexdigest(),
        }

        return message_bytes, metadata

    # ----------------------------------------------------------------
    # Part B: Instruction pair generators
    # ----------------------------------------------------------------

    def _generate_swift_security_pairs(self, count: int) -> List[Dict]:
        """Generate PQC integration for SWIFT messaging instruction pairs."""
        templates = [
            {
                "difficulty": "basic",
                "instructions": [
                    "Explain how ML-KEM (Kyber) key encapsulation can protect SWIFT MT message confidentiality.",
                    "What is the role of ML-DSA (Dilithium) digital signatures in SWIFT FIN message authentication?",
                    "Describe how a bank should plan for the post-quantum migration of SWIFT Alliance Lite2 gateways.",
                    "What PQC algorithms are recommended by NIST for protecting interbank message integrity?",
                    "How should HSM key hierarchies be restructured to support hybrid RSA + ML-KEM key exchange in SWIFT?",
                ],
                "contexts": [
                    "Bank operates SWIFT Alliance Access with 500+ daily MT103/MT202 messages. Current PKI uses RSA-2048 for signing and ECDH for key agreement.",
                    "Regional bank processing EUR/USD correspondent payments. HSM fleet: 12 Thales Luna Network HSMs. TLS 1.2 for SWIFT connectivity.",
                ],
                "responses": [
                    "## ML-KEM Key Encapsulation for SWIFT MT Messages\n\n### Overview\nML-KEM (formerly CRYSTALS-Kyber) provides quantum-resistant key encapsulation that can secure the symmetric session keys used to encrypt SWIFT FIN messages in transit.\n\n### Integration Points\n1. **Session Key Establishment**: Replace ECDH key agreement with ML-KEM-768 encapsulation at the SWIFT Alliance Access gateway level\n2. **Hybrid Mode**: Run ML-KEM-768 in parallel with ECDH-P256 during the transition period; derive the session key from both shared secrets using HKDF\n3. **Message Encryption**: The encapsulated shared secret feeds AES-256-GCM for FIN block 4 encryption\n4. **HSM Integration**: Store ML-KEM private keys in FIPS 140-3 Level 3 HSMs; use PKCS#11 v3.1 interface\n\n### Key Sizes\n- ML-KEM-768 public key: 1,184 bytes\n- ML-KEM-768 ciphertext: 1,088 bytes\n- Shared secret: 32 bytes\n\n### Migration Steps\n1. Firmware-update HSMs to support ML-KEM\n2. Generate hybrid key pairs (ECDH + ML-KEM)\n3. Distribute ML-KEM public keys via SWIFT RMA exchange\n4. Enable hybrid encapsulation on outbound messages\n5. Validate decapsulation on inbound messages\n6. Monitor performance impact (expect <5ms overhead per message)",
                ],
            },
            {
                "difficulty": "intermediate",
                "instructions": [
                    "Design a hybrid PQC signature scheme for SWIFT gpi Tracker confirmations that maintains backward compatibility.",
                    "How should banks implement crypto-agility in their SWIFT infrastructure to allow rapid algorithm rotation?",
                    "Propose a key management lifecycle for ML-DSA signing keys used in ISO 20022 pacs.008 messages.",
                    "Evaluate the bandwidth impact of replacing ECDSA with ML-DSA-65 signatures on SWIFT FIN messages over SWIFTNet.",
                    "How can SWIFT's PKI be extended to issue hybrid X.509 certificates containing both RSA and ML-DSA public keys?",
                ],
                "contexts": [
                    "Global systemically important bank (G-SIB) with 50,000+ daily SWIFT messages. Must comply with SWIFT CSCF v2025. Operating SWIFT Alliance Gateway with Oracle DB backend.",
                    "Central bank upgrading RTGS system to ISO 20022. Processes domestic high-value payments and cross-border settlements. Current throughput: 1.2M messages/day.",
                ],
                "responses": [
                    "## Hybrid PQC Signature Scheme for SWIFT gpi Tracker\n\n### Architecture\nComposite signature combining ECDSA P-256 with ML-DSA-65 (Dilithium3) as specified in IETF draft-ounsworth-pq-composite-sigs.\n\n### Signature Flow\n1. **Message Preparation**: Canonicalize the gpi Tracker JSON payload (RFC 8785 JCS)\n2. **Dual Signing**:\n   - Sign with ECDSA P-256 (legacy compatibility)\n   - Sign with ML-DSA-65 (quantum resistance)\n3. **Composite Encoding**: Wrap both signatures in a CompositeSignature ASN.1 structure\n4. **Header Tagging**: Add X-PQC-Sig: composite-ecdsa-mldsa65 header to the gpi API call\n\n### Backward Compatibility\n- Legacy validators extract and verify only the ECDSA component\n- PQC-capable validators verify both; reject if either fails\n- Transition flag in RMA profile: pqc_capable=true\n\n### Signature Sizes\n- ECDSA P-256: 64 bytes\n- ML-DSA-65: 3,293 bytes\n- Composite overhead: ~3,400 bytes per message\n- Impact on gpi API payload: <2% size increase for typical confirmations\n\n### Performance\n- ML-DSA-65 sign: ~0.8ms (Thales Luna HSM)\n- ML-DSA-65 verify: ~0.3ms\n- Total added latency per gpi confirmation: <2ms\n\n### Rollout Phases\n1. **Phase 1**: Generate hybrid key pairs, distribute via SWIFT RMA\n2. **Phase 2**: Sign with composite, verify ECDSA only (shadow mode)\n3. **Phase 3**: Full composite verification mandatory\n4. **Phase 4**: Deprecate standalone ECDSA after 24-month overlap",
                ],
            },
            {
                "difficulty": "advanced",
                "instructions": [
                    "Design a zero-trust architecture for SWIFT message authentication that uses PQC at every trust boundary.",
                    "How would a harvest-now-decrypt-later attack against recorded SWIFTNet traffic be mitigated by ML-KEM?",
                    "Architect a PQC key ceremony procedure for SWIFT HSMs that complies with FIPS 140-3 Level 3 and eIDAS.",
                    "Design a PQC-secured SWIFT Alliance Gateway cluster with active-active failover and sub-millisecond signing latency.",
                ],
                "contexts": [
                    "Tier-1 investment bank operating SWIFT Alliance Gateway cluster across three data centers. 200,000+ messages/day. Must support FIN, InterAct, and FileAct. Budget approved for full PQC migration.",
                ],
                "responses": [
                    "## Zero-Trust PQC Architecture for SWIFT Message Authentication\n\n### Trust Boundaries\n1. **Application-to-HSM**: Mutual TLS 1.3 with ML-KEM-1024 key exchange\n2. **HSM-to-HSM (cluster sync)**: ML-KEM-768 encrypted replication channel\n3. **Gateway-to-SWIFTNet**: Hybrid ECDH+ML-KEM session, ML-DSA-87 message signatures\n4. **Gateway-to-Backend**: mTLS with composite certificates (RSA-4096 + ML-DSA-65)\n5. **Operator-to-Gateway**: FIDO2 + ML-DSA-44 challenge-response authentication\n\n### Message Flow\n```\nApp -> [mTLS+ML-KEM] -> Alliance Gateway -> [HSM: ML-DSA-87 sign] -> \n[Hybrid TLS] -> SWIFTNet -> [HSM: ML-DSA-87 verify] -> Counterparty\n```\n\n### Key Hierarchy\n- Root CA: ML-DSA-87 (offline, air-gapped ceremony)\n- Issuing CA: ML-DSA-65 (HSM-resident, dual-control)\n- Message Signing: ML-DSA-65 (HSM-resident, per-gateway)\n- Session Keys: ML-KEM-768 ephemeral (per-connection)\n- Data-at-rest: AES-256-GCM with ML-KEM-wrapped DEKs\n\n### Continuous Verification\n- Every FIN block re-authenticated at each trust boundary\n- No implicit trust inheritance between zones\n- Anomaly detection on signature timing and key usage patterns\n- Real-time revocation via CRL + OCSP with ML-DSA-44 signed responses",
                ],
            },
        ]

        pairs = []
        for _ in range(count):
            template = random.choice(templates)
            pairs.append({
                "pair_id": str(uuid.uuid4()),
                "category": "swift_security",
                "difficulty": template["difficulty"],
                "instruction": random.choice(template["instructions"]),
                "context": random.choice(template["contexts"]),
                "response": random.choice(template["responses"]),
            })
        return pairs

    def _generate_transaction_monitoring_pairs(self, count: int) -> List[Dict]:
        """Generate transaction monitoring and AML screening instruction pairs."""
        instructions = [
            "Analyze this SWIFT MT103 transfer for AML red flags and explain your risk assessment.",
            "Design a real-time transaction monitoring rule set for detecting structuring in SWIFT payments.",
            "How should a correspondent bank implement sanctions screening on pacs.008 messages?",
            "Identify typology indicators in this sequence of MT202 cover payments.",
            "What machine learning features are most predictive of trade-based money laundering in SWIFT traffic?",
            "Design an anomaly detection model for detecting unusual correspondent banking patterns.",
            "How should banks screen ISO 20022 messages against OFAC, EU, and UN sanctions lists?",
            "Analyze this customer's transaction pattern for potential layering activity.",
            "What additional due diligence should be triggered by this cross-border payment pattern?",
            "Design a post-quantum-secure audit trail for AML investigation evidence preservation.",
        ]

        contexts = [
            (
                "MT103: $4,950,000 USD from BOFAUS3N to DEUTDEFF. Ordering customer: 'Global Trading Corp' "
                "(registered in BVI, 6 months old). Beneficiary: 'Pinnacle Advisory SA' in Panama. "
                "No prior transaction history between parties. Field 70: /INV/CONSULTING FEES Q4."
            ),
            (
                "Pattern: 12 MT103 transfers in 48 hours, all from same ordering customer, amounts: "
                "$9,800, $9,500, $9,900, $8,700, $9,600, $9,950, $9,100, $9,400, $9,750, $9,300, $9,850, $9,200. "
                "Total: $114,050. All to different beneficiaries in 4 countries. Charge code: OUR."
            ),
            (
                "Correspondent chain: BOFAUS3N -> CHASUS33 -> HSBCGB2L -> SCBLSGSG -> final beneficiary. "
                "4-hop MT202 cover chain for a $2.3M payment. Originator in Russia, beneficiary in UAE. "
                "Purpose: 'equipment purchase'. No underlying trade documentation."
            ),
            (
                "camt.053 statement shows: 150 inbound credits (avg $45,000) from 80 different originators "
                "in 30 days, immediately followed by 3 large outbound wires totaling $6.2M to a single "
                "beneficiary in Hong Kong. Account opened 90 days ago. KYC shows import/export firm."
            ),
        ]

        responses = [
            (
                "## AML Risk Assessment: MT103 Transfer\n\n"
                "### Red Flags Identified\n"
                "1. **High Value**: $4.95M single transfer - above enhanced due diligence threshold\n"
                "2. **Jurisdiction Risk**: BVI originator (high-risk) to Panama beneficiary (high-risk)\n"
                "3. **Entity Age**: Ordering customer incorporated only 6 months ago\n"
                "4. **No History**: First transaction between these counterparties\n"
                "5. **Vague Purpose**: 'CONSULTING FEES Q4' - generic description for $5M payment\n"
                "6. **Round Amount**: $4.95M suggests possible structuring below $5M reporting threshold\n\n"
                "### Risk Score: 87/100 (HIGH)\n\n"
                "### Required Actions\n"
                "- File Suspicious Activity Report (SAR/STR) within 24 hours\n"
                "- Request source of funds documentation from ordering bank\n"
                "- Verify beneficial ownership of both entities via registry lookup\n"
                "- Check counterparties against OFAC SDN, EU consolidated, UN sanctions lists\n"
                "- Escalate to MLRO for potential transaction hold\n"
                "- Review all historical transactions from Global Trading Corp"
            ),
            (
                "## AML Risk Assessment: Structuring Pattern Detected\n\n"
                "### Typology: Smurfing / Structuring\n"
                "12 transfers in 48 hours, all below $10,000 reporting threshold.\n\n"
                "### Statistical Indicators\n"
                "- Mean: $9,504 | Std Dev: $380 | Range: $8,700-$9,950\n"
                "- All amounts cluster tightly below $10,000 CTR threshold\n"
                "- 12 different beneficiaries in 4 countries = fan-out pattern\n"
                "- Charge code OUR (sender pays all fees) = hiding true amounts\n"
                "- Aggregate $114,050 over 48 hours from single originator\n\n"
                "### Risk Score: 95/100 (CRITICAL)\n\n"
                "### Actions\n"
                "- Immediate SAR filing (structuring is itself a criminal offense)\n"
                "- Aggregate CTR for total $114,050\n"
                "- Freeze outbound transactions pending investigation\n"
                "- Request explanation from account holder\n"
                "- Cross-reference beneficiaries for common ownership"
            ),
        ]

        pairs = []
        for _ in range(count):
            pairs.append({
                "pair_id": str(uuid.uuid4()),
                "category": "transaction_monitoring",
                "difficulty": random.choice(self.DIFFICULTIES),
                "instruction": random.choice(instructions),
                "context": random.choice(contexts),
                "response": random.choice(responses),
            })
        return pairs

    def _generate_regulatory_compliance_pairs(self, count: int) -> List[Dict]:
        """Generate regulatory compliance instruction pairs (Basel, PCI-DSS, PSD2)."""
        instructions = [
            "How does Basel III's operational risk framework apply to PQC migration in banking infrastructure?",
            "Assess PCI-DSS 4.0 compliance requirements for quantum-safe encryption of cardholder data in SWIFT messages.",
            "What PSD2 Strong Customer Authentication requirements must be updated for PQC algorithms?",
            "Design a regulatory reporting framework for PQC readiness under ECB DORA requirements.",
            "How should a bank document PQC migration for Basel IV operational resilience requirements?",
            "Evaluate the impact of quantum computing on SWIFT CSCF mandatory controls.",
            "What changes to the SWIFT Customer Security Programme are needed to address quantum threats?",
            "How should banks update their ICT risk management frameworks to include PQC under DORA?",
            "Design an audit trail architecture for PQC key lifecycle events that satisfies MAS TRM requirements.",
            "What operational risk capital charges apply to the quantum threat under Basel III standardized approach?",
        ]

        contexts = [
            (
                "European G-SIB subject to ECB supervision, Basel III/CRR II, PSD2, DORA, and SWIFT CSCF v2025. "
                "Operating 5 SWIFT Alliance Gateways. Annual SWIFT message volume: 18M. Current crypto: RSA-2048 + AES-256."
            ),
            (
                "US bank holding company subject to Fed SR 11-7, OCC heightened standards, PCI-DSS Level 1 (processes "
                "card data in SWIFT messages). SWIFT CSCF architecture type: A3. Dodd-Frank stress testing participant."
            ),
            (
                "APAC bank regulated by MAS (Singapore). Subject to MAS TRM Guidelines 2021, MAS Notice 644 "
                "(technology risk management). SWIFT participant with 2,000 daily messages. Operating ISO 20022 migration."
            ),
        ]

        responses = [
            (
                "## Basel III Operational Risk & PQC Migration\n\n"
                "### Regulatory Mapping\n"
                "Basel III's operational risk framework (BCBS d424) requires banks to identify, assess, and "
                "mitigate risks from technology changes. PQC migration creates operational risk across:\n\n"
                "1. **ICT Risk (Pillar 2)**: Quantum threat to cryptographic infrastructure\n"
                "   - Classify as 'technology obsolescence' risk event type\n"
                "   - Include in ICAAP scenario analysis\n"
                "   - Quantify potential loss from cryptographic failure\n\n"
                "2. **Operational Resilience (BCBS d516)**:\n"
                "   - Map critical SWIFT payment services to tolerance thresholds\n"
                "   - PQC migration is a 'change to critical third-party service'\n"
                "   - Requires board-approved migration timeline\n\n"
                "3. **Capital Implications**:\n"
                "   - Standardized approach: PQC migration costs are operational expenses (not risk-weighted)\n"
                "   - Internal models: Add quantum threat scenario to loss distribution\n"
                "   - Stress testing: Include 'cryptographic compromise' scenario\n\n"
                "### Compliance Actions\n"
                "- Document PQC migration in annual ICAAP submission\n"
                "- Include quantum risk in RCSA (Risk Control Self-Assessment)\n"
                "- Report PQC readiness metrics to board risk committee quarterly\n"
                "- Engage external auditor to validate PQC migration controls"
            ),
            (
                "## PSD2 SCA Updates for Post-Quantum Cryptography\n\n"
                "### Current SCA Requirements (EBA RTS)\n"
                "PSD2 mandates Strong Customer Authentication using 2+ of: knowledge, possession, inherence.\n"
                "Cryptographic binding between authentication factors relies on classical algorithms.\n\n"
                "### Required PQC Updates\n"
                "1. **Dynamic Linking (Art. 97)**: Transaction signing currently uses ECDSA\n"
                "   - Migrate to ML-DSA-65 for transaction-specific authentication codes\n"
                "   - Hybrid ECDSA + ML-DSA during transition\n\n"
                "2. **Secure Communication (Art. 34)**: TLS for API channels\n"
                "   - Upgrade to TLS 1.3 with hybrid ML-KEM + ECDH key exchange\n"
                "   - Update eIDAS certificates to include PQC public keys\n\n"
                "3. **Authentication Code Generation**:\n"
                "   - Replace HMAC-SHA256 OTPs with HMAC-SHA3-256\n"
                "   - Hardware tokens: firmware update for ML-DSA signing\n\n"
                "4. **Exemption Thresholds**: No change (not crypto-dependent)\n\n"
                "### Timeline\n"
                "- EBA expected to issue PQC guidance by 2026\n"
                "- Recommend proactive hybrid deployment now\n"
                "- Full PQC-only mode target: 2030 (aligned with NIST timeline)"
            ),
        ]

        pairs = []
        for _ in range(count):
            pairs.append({
                "pair_id": str(uuid.uuid4()),
                "category": "regulatory_compliance",
                "difficulty": random.choice(self.DIFFICULTIES),
                "instruction": random.choice(instructions),
                "context": random.choice(contexts),
                "response": random.choice(responses),
            })
        return pairs

    def _generate_correspondent_banking_pairs(self, count: int) -> List[Dict]:
        """Generate correspondent banking PQC instruction pairs."""
        instructions = [
            "Design a multi-hop proxy re-encryption scheme for correspondent banking MT202 cover payments.",
            "How can PQC lattice-based proxy re-encryption protect payment messages across a 4-bank correspondent chain?",
            "Explain chain-of-trust verification for SWIFT gpi payments using ML-DSA aggregate signatures.",
            "Design a PQC-secured nostro/vostro reconciliation protocol between correspondent banks.",
            "How should banks implement threshold signatures for multi-party payment authorization in SWIFT?",
            "Propose a decentralized key management architecture for correspondent banking PQC credentials.",
            "Design a privacy-preserving sanctions screening protocol using homomorphic encryption over SWIFT messages.",
            "How can verifiable credentials (W3C VC) with PQC signatures replace traditional SWIFT RMA key exchange?",
            "Architect a PQC-secured payment status tracking system for multi-leg correspondent transfers.",
            "Design a quantum-resistant secure channel for SWIFT Y-Copy topology in correspondent banking.",
        ]

        contexts = [
            (
                "Correspondent chain: US Bank (BOFAUS3N) -> UK Intermediary (HSBCGB2L) -> "
                "Singapore Beneficiary Bank (SCBLSGSG). MT103/MT202 cover method. "
                "Each bank must verify payment authenticity without seeing the full payment details of other legs."
            ),
            (
                "Nostro reconciliation between DEUTDEFF and CHASUS33. Daily volume: 5,000 payments, "
                "$2.1B aggregate. Current process: end-of-day MT950 statements. "
                "Target: real-time reconciliation with PQC integrity guarantees."
            ),
            (
                "Regional bank network: 12 banks in Southeast Asia sharing a common correspondent (HSBCSGSG). "
                "Need PQC-secured bilateral key agreements for each pair. "
                "Total key pairs required: 66 bilateral + 12 hub-spoke = 78 key relationships."
            ),
        ]

        responses = [
            (
                "## Multi-Hop Proxy Re-Encryption for Correspondent Banking\n\n"
                "### Problem\n"
                "In a 3-bank correspondent chain (A -> B -> C), Bank B must forward the payment message "
                "without having the ability to decrypt the full originator/beneficiary details, while each "
                "bank needs to verify the payment's authenticity.\n\n"
                "### PQC Proxy Re-Encryption Scheme\n"
                "Using lattice-based proxy re-encryption (LB-PRE) built on ML-KEM:\n\n"
                "1. **Key Setup**:\n"
                "   - Each bank generates ML-KEM-768 key pair: (pk_i, sk_i)\n"
                "   - Re-encryption keys computed: rk_{A->B} = ReKeyGen(sk_A, pk_B)\n"
                "   - Re-encryption keys computed: rk_{B->C} = ReKeyGen(sk_B, pk_C)\n\n"
                "2. **Encryption (Bank A)**:\n"
                "   - Encrypt full MT103 under pk_A: c_A = Enc(pk_A, MT103)\n"
                "   - Generate payment routing header (unencrypted): {sender_BIC, receiver_BIC, amount, currency}\n"
                "   - Sign entire message with ML-DSA-65: sig_A = Sign(sk_A^{sig}, c_A || header)\n\n"
                "3. **Re-Encryption (Bank B / Intermediary)**:\n"
                "   - Verify sig_A\n"
                "   - Apply re-encryption: c_B = ReEnc(rk_{A->B}, c_A) -- B can verify routing but NOT read originator details\n"
                "   - Add own signature: sig_B = Sign(sk_B^{sig}, c_B || header)\n\n"
                "4. **Decryption (Bank C)**:\n"
                "   - Verify both sig_A and sig_B (chain-of-trust)\n"
                "   - Decrypt: MT103 = Dec(sk_C, ReEnc(rk_{B->C}, c_B))\n\n"
                "### Security Properties\n"
                "- **Quantum Resistance**: ML-KEM-768 provides 192-bit post-quantum security\n"
                "- **Non-Transitive**: Bank B cannot delegate decryption to unauthorized parties\n"
                "- **Unidirectional**: Re-encryption keys work only in one direction\n"
                "- **Collusion Resistant**: Proxy + delegatee cannot recover delegator's secret key"
            ),
            (
                "## PQC-Secured Nostro/Vostro Reconciliation\n\n"
                "### Architecture\n"
                "Replace end-of-day MT950 reconciliation with real-time PQC-authenticated state synchronization.\n\n"
                "### Protocol Design\n"
                "1. **Shared State Commitment**:\n"
                "   - After each settlement, both banks compute a Merkle tree of transaction hashes\n"
                "   - Root hash signed with ML-DSA-65 and exchanged bilaterally\n"
                "   - Disagreements detected in real-time (not end-of-day)\n\n"
                "2. **Secure Channel**:\n"
                "   - ML-KEM-768 authenticated key exchange for the reconciliation channel\n"
                "   - AES-256-GCM for bulk data encryption\n"
                "   - Perfect forward secrecy via ephemeral ML-KEM key pairs per session\n\n"
                "3. **Reconciliation Messages** (ISO 20022 camt.053 based):\n"
                "   - Each entry signed with ML-DSA-44 (fast verification)\n"
                "   - Statement-level signature with ML-DSA-65\n"
                "   - Cross-signed by both banks for non-repudiation\n\n"
                "### Operational Improvements\n"
                "- Detection time: End-of-day -> Real-time (<30 seconds)\n"
                "- Dispute resolution: Days -> Hours (cryptographic proof of each transaction)\n"
                "- Quantum safety: Protected against harvest-now-decrypt-later on reconciliation data"
            ),
        ]

        pairs = []
        for _ in range(count):
            pairs.append({
                "pair_id": str(uuid.uuid4()),
                "category": "correspondent_banking",
                "difficulty": random.choice(self.DIFFICULTIES),
                "instruction": random.choice(instructions),
                "context": random.choice(contexts),
                "response": random.choice(responses),
            })
        return pairs

    # ----------------------------------------------------------------
    # Dataset generation
    # ----------------------------------------------------------------

    def generate_dataset(self, num_protocol_samples: int, num_instruction_pairs: int,
                         output_dir: str) -> Dict:
        """Generate the complete SWIFT banking dataset.

        Args:
            num_protocol_samples: Number of protocol binary+metadata samples (Part A).
            num_instruction_pairs: Number of instruction pairs (Part B).
            output_dir: Root output directory.

        Returns:
            Dataset metadata dictionary.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # ---- Part A: Protocol Samples ----
        protocol_dir = output_path / "protocols"
        protocol_dir.mkdir(parents=True, exist_ok=True)

        protocol_generators = [
            ("mt103", self.generate_mt103, 0.30),
            ("mt202", self.generate_mt202, 0.20),
            ("pacs008", self.generate_pacs008, 0.25),
            ("pacs009", self.generate_pacs009, 0.10),
            ("camt053", self.generate_camt053, 0.15),
        ]

        protocol_counts = {}
        sample_idx = 0
        for msg_type, generator, ratio in protocol_generators:
            count = int(num_protocol_samples * ratio)
            protocol_counts[msg_type] = count

            for _ in range(count):
                message, metadata = generator()
                metadata["sample_index"] = sample_idx
                metadata["message_subtype"] = msg_type

                bin_path = protocol_dir / f"swift_{msg_type}_{sample_idx:06d}.bin"
                with open(bin_path, "wb") as f:
                    f.write(message)

                meta_path = protocol_dir / f"swift_{msg_type}_{sample_idx:06d}.json"
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2, default=str)

                sample_idx += 1

        # ---- Part B: Instruction Pairs ----
        pairs_dir = output_path / "instruction_pairs"
        pairs_dir.mkdir(parents=True, exist_ok=True)

        pairs_per_category = num_instruction_pairs // len(self.INSTRUCTION_CATEGORIES)
        remainder = num_instruction_pairs % len(self.INSTRUCTION_CATEGORIES)

        category_generators = {
            "swift_security": self._generate_swift_security_pairs,
            "transaction_monitoring": self._generate_transaction_monitoring_pairs,
            "regulatory_compliance": self._generate_regulatory_compliance_pairs,
            "correspondent_banking": self._generate_correspondent_banking_pairs,
        }

        all_pairs = []
        instruction_counts = {}
        for idx, category in enumerate(self.INSTRUCTION_CATEGORIES):
            cat_count = pairs_per_category + (1 if idx < remainder else 0)
            generator = category_generators[category]
            pairs = generator(cat_count)
            all_pairs.extend(pairs)
            instruction_counts[category] = cat_count

        random.shuffle(all_pairs)

        # Write JSONL file
        jsonl_path = pairs_dir / "swift_banking_instructions.jsonl"
        with open(jsonl_path, "w") as f:
            for pair in all_pairs:
                f.write(json.dumps(pair, default=str) + "\n")

        # ---- Dataset Metadata ----
        dataset_metadata = {
            "dataset": "swift_banking",
            "version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "protocol_samples": {
                "total": sample_idx,
                "by_type": protocol_counts,
                "output_dir": str(protocol_dir),
            },
            "instruction_pairs": {
                "total": len(all_pairs),
                "by_category": instruction_counts,
                "difficulties": {
                    d: sum(1 for p in all_pairs if p["difficulty"] == d)
                    for d in self.DIFFICULTIES
                },
                "output_file": str(jsonl_path),
            },
        }

        with open(output_path / "dataset_metadata.json", "w") as f:
            json.dump(dataset_metadata, f, indent=2)

        return dataset_metadata


def main():
    """Generate SWIFT banking dataset."""
    generator = SwiftBankingGenerator(seed=42)

    output_dir = Path(__file__).parent.parent / "protocols" / "swift_banking"

    print("Generating SWIFT/ISO 20022 banking dataset...")
    metadata = generator.generate_dataset(
        num_protocol_samples=1000,
        num_instruction_pairs=200,
        output_dir=str(output_dir),
    )

    print(f"\nPart A - Protocol Samples: {metadata['protocol_samples']['total']}")
    for msg_type, count in metadata["protocol_samples"]["by_type"].items():
        print(f"  - {msg_type}: {count} samples")

    print(f"\nPart B - Instruction Pairs: {metadata['instruction_pairs']['total']}")
    for category, count in metadata["instruction_pairs"]["by_category"].items():
        print(f"  - {category}: {count} pairs")

    print(f"\nDifficulty distribution:")
    for difficulty, count in metadata["instruction_pairs"]["difficulties"].items():
        print(f"  - {difficulty}: {count} pairs")

    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
