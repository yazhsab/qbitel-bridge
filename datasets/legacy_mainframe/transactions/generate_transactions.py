#!/usr/bin/env python3
"""
QBITEL - Mainframe Transaction Log Generator

Generates realistic mainframe transaction logs for training and RAG.
Includes banking, insurance, and general business transactions.
"""

import json
import random
import struct
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple
from decimal import Decimal
import hashlib

# =============================================================================
# Configuration
# =============================================================================

TRANSACTION_TYPES = {
    "banking": {
        "0100": {"name": "Authorization Request", "weight": 30},
        "0110": {"name": "Authorization Response", "weight": 30},
        "0200": {"name": "Financial Request", "weight": 20},
        "0210": {"name": "Financial Response", "weight": 20},
        "0400": {"name": "Reversal Request", "weight": 5},
        "0410": {"name": "Reversal Response", "weight": 5},
        "0800": {"name": "Network Management Request", "weight": 3},
        "0810": {"name": "Network Management Response", "weight": 3},
    },
    "cics": {
        "INQY": {"name": "Account Inquiry", "weight": 40},
        "XFER": {"name": "Transfer Funds", "weight": 25},
        "DPST": {"name": "Deposit", "weight": 15},
        "WDRW": {"name": "Withdrawal", "weight": 15},
        "STMT": {"name": "Statement Request", "weight": 5},
    },
    "batch": {
        "EOD": {"name": "End of Day Processing", "weight": 30},
        "INT": {"name": "Interest Calculation", "weight": 25},
        "FEE": {"name": "Fee Assessment", "weight": 20},
        "RPT": {"name": "Report Generation", "weight": 15},
        "ARC": {"name": "Archival", "weight": 10},
    }
}

RESPONSE_CODES = {
    "00": "Approved",
    "01": "Refer to issuer",
    "03": "Invalid merchant",
    "05": "Do not honor",
    "12": "Invalid transaction",
    "13": "Invalid amount",
    "14": "Invalid card number",
    "51": "Insufficient funds",
    "54": "Expired card",
    "55": "Invalid PIN",
    "61": "Exceeds withdrawal limit",
    "91": "Issuer unavailable",
}


class TransactionGenerator:
    """Generates realistic mainframe transaction data."""

    def __init__(self, seed: int = None):
        if seed:
            random.seed(seed)
        self.transaction_counter = 0

    def generate_account_number(self) -> str:
        """Generate realistic account number."""
        return f"{random.randint(1000, 9999)}{random.randint(100000, 999999)}{random.randint(0, 9)}"

    def generate_card_number(self) -> str:
        """Generate test card number (Luhn-valid pattern)."""
        prefix = random.choice(["4", "5", "37", "6011"])
        length = 16 if prefix != "37" else 15
        partial = prefix + "".join(str(random.randint(0, 9)) for _ in range(length - len(prefix) - 1))
        # Add check digit (simplified)
        return partial + str(random.randint(0, 9))

    def generate_amount(self, trans_type: str) -> Decimal:
        """Generate transaction amount based on type."""
        if trans_type in ["INQY", "STMT", "RPT"]:
            return Decimal("0.00")
        elif trans_type in ["0800", "0810"]:
            return Decimal("0.00")
        else:
            # Weighted distribution: most transactions are small
            weights = [(100, 50), (500, 25), (1000, 15), (5000, 7), (10000, 3)]
            max_amount = random.choices([w[0] for w in weights], [w[1] for w in weights])[0]
            amount = Decimal(str(random.uniform(1, max_amount))).quantize(Decimal("0.01"))
            return amount

    def generate_timestamp(self, base_date: datetime = None) -> Dict[str, str]:
        """Generate COBOL-style timestamp."""
        if base_date is None:
            base_date = datetime.now() - timedelta(days=random.randint(0, 365))

        # Add random time within the day
        base_date = base_date.replace(
            hour=random.randint(0, 23),
            minute=random.randint(0, 59),
            second=random.randint(0, 59),
            microsecond=random.randint(0, 999999)
        )

        return {
            "iso": base_date.isoformat(),
            "cobol_date": base_date.strftime("%Y%m%d"),
            "cobol_time": base_date.strftime("%H%M%S"),
            "julian": base_date.strftime("%Y%j"),
            "timestamp26": base_date.strftime("%Y-%m-%d-%H.%M.%S.%f")
        }

    def generate_iso8583_message(self) -> Dict[str, Any]:
        """Generate ISO 8583 financial message."""
        self.transaction_counter += 1

        mti_weights = [(mti, data["weight"]) for mti, data in TRANSACTION_TYPES["banking"].items()]
        mti = random.choices([m[0] for m in mti_weights], [m[1] for m in mti_weights])[0]

        is_response = mti[2] == "1"
        timestamp = self.generate_timestamp()

        message = {
            "message_type": "iso8583",
            "mti": mti,
            "mti_description": TRANSACTION_TYPES["banking"][mti]["name"],
            "fields": {
                "2": self.generate_card_number(),  # Primary Account Number
                "3": f"{random.randint(0, 99):02d}0000",  # Processing Code
                "4": str(self.generate_amount(mti)),  # Transaction Amount
                "7": timestamp["cobol_date"] + timestamp["cobol_time"],  # Transmission Date/Time
                "11": f"{self.transaction_counter:06d}",  # System Trace Audit Number
                "12": timestamp["cobol_time"],  # Local Transaction Time
                "13": timestamp["cobol_date"][4:8],  # Local Transaction Date
                "22": f"{random.randint(0, 9):01d}{random.randint(0, 9):01d}1",  # POS Entry Mode
                "23": f"{random.randint(0, 999):03d}",  # Card Sequence Number
                "32": f"{random.randint(100000, 999999)}",  # Acquiring Institution ID
                "37": f"{random.randint(100000000000, 999999999999)}",  # Retrieval Reference Number
                "41": f"TERM{random.randint(1000, 9999):04d}",  # Card Acceptor Terminal ID
                "42": f"MERCH{random.randint(100000000000, 999999999999)}",  # Merchant ID
                "43": self._generate_merchant_name(),  # Card Acceptor Name/Location
                "49": "840",  # Currency Code (USD)
            },
            "timestamp": timestamp,
            "trace_id": hashlib.md5(f"{self.transaction_counter}{timestamp['iso']}".encode()).hexdigest()[:16]
        }

        if is_response:
            response_code = random.choices(
                list(RESPONSE_CODES.keys()),
                weights=[70, 2, 1, 3, 2, 2, 3, 8, 2, 2, 3, 2]  # Mostly approvals
            )[0]
            message["fields"]["39"] = response_code
            message["response_description"] = RESPONSE_CODES[response_code]

        return message

    def generate_cics_transaction(self) -> Dict[str, Any]:
        """Generate CICS online transaction."""
        self.transaction_counter += 1

        trans_weights = [(t, data["weight"]) for t, data in TRANSACTION_TYPES["cics"].items()]
        trans_id = random.choices([t[0] for t in trans_weights], [t[1] for t in trans_weights])[0]

        timestamp = self.generate_timestamp()
        account = self.generate_account_number()

        transaction = {
            "message_type": "cics",
            "transaction_id": trans_id,
            "transaction_name": TRANSACTION_TYPES["cics"][trans_id]["name"],
            "terminal_id": f"T{random.randint(100, 999):03d}",
            "user_id": f"USR{random.randint(10000, 99999)}",
            "account_number": account,
            "amount": str(self.generate_amount(trans_id)),
            "timestamp": timestamp,
            "response_time_ms": random.randint(10, 500),
            "abend_code": None if random.random() > 0.01 else random.choice(["ASRA", "AICA", "AEY7"]),
            "commarea": {
                "function_code": trans_id,
                "account": account,
                "return_code": "00" if random.random() > 0.05 else f"{random.randint(1, 99):02d}",
            },
            "trace_id": hashlib.md5(f"CICS{self.transaction_counter}{timestamp['iso']}".encode()).hexdigest()[:16]
        }

        if trans_id == "XFER":
            transaction["to_account"] = self.generate_account_number()

        return transaction

    def generate_batch_record(self) -> Dict[str, Any]:
        """Generate batch processing record."""
        self.transaction_counter += 1

        job_weights = [(j, data["weight"]) for j, data in TRANSACTION_TYPES["batch"].items()]
        job_type = random.choices([j[0] for j in job_weights], [j[1] for j in job_weights])[0]

        timestamp = self.generate_timestamp()

        record = {
            "message_type": "batch",
            "job_name": f"{job_type}JOB{random.randint(100, 999):03d}",
            "job_type": job_type,
            "job_description": TRANSACTION_TYPES["batch"][job_type]["name"],
            "step_name": f"STEP{random.randint(1, 5):02d}",
            "program_name": f"{job_type}PGM{random.randint(10, 99):02d}",
            "start_time": timestamp,
            "records_read": random.randint(1000, 1000000),
            "records_written": random.randint(900, 999000),
            "records_rejected": random.randint(0, 1000),
            "return_code": random.choices([0, 4, 8, 12, 16], weights=[80, 10, 5, 3, 2])[0],
            "cpu_time_seconds": random.uniform(1, 3600),
            "elapsed_time_seconds": random.uniform(60, 7200),
            "trace_id": hashlib.md5(f"BATCH{self.transaction_counter}{timestamp['iso']}".encode()).hexdigest()[:16]
        }

        if job_type in ["INT", "FEE"]:
            record["total_amount"] = str(Decimal(str(random.uniform(10000, 10000000))).quantize(Decimal("0.01")))

        return record

    def generate_fixed_width_record(self, record_type: str = "customer") -> Tuple[bytes, Dict]:
        """Generate fixed-width mainframe record with metadata."""
        if record_type == "customer":
            # Customer master record (200 bytes)
            account = self.generate_account_number()
            name = self._generate_person_name()
            balance = Decimal(str(random.uniform(-10000, 1000000))).quantize(Decimal("0.01"))

            # Build fixed-width record
            record = (
                account.ljust(12)[:12] +                    # Bytes 0-11: Account number
                name["full"].ljust(30)[:30] +              # Bytes 12-41: Name
                self._generate_address().ljust(50)[:50] +  # Bytes 42-91: Address
                self._generate_city().ljust(20)[:20] +     # Bytes 92-111: City
                random.choice(["CA", "NY", "TX", "FL", "IL"]) +  # Bytes 112-113: State
                f"{random.randint(10000, 99999)}" +        # Bytes 114-118: ZIP
                datetime.now().strftime("%Y%m%d") +         # Bytes 119-126: Open date
                random.choice(["A", "I", "C", "S"]) +       # Byte 127: Status
                f"{int(abs(balance) * 100):015d}" +        # Bytes 128-142: Balance (packed would be smaller)
                ("+" if balance >= 0 else "-") +           # Byte 143: Sign
                " " * 56                                    # Bytes 144-199: Filler
            )

            metadata = {
                "record_type": "customer",
                "account": account,
                "name": name,
                "balance": str(balance),
                "layout": {
                    "account": {"start": 0, "length": 12, "type": "alphanumeric"},
                    "name": {"start": 12, "length": 30, "type": "alphanumeric"},
                    "address": {"start": 42, "length": 50, "type": "alphanumeric"},
                    "city": {"start": 92, "length": 20, "type": "alphanumeric"},
                    "state": {"start": 112, "length": 2, "type": "alphanumeric"},
                    "zip": {"start": 114, "length": 5, "type": "numeric"},
                    "open_date": {"start": 119, "length": 8, "type": "numeric"},
                    "status": {"start": 127, "length": 1, "type": "alphanumeric"},
                    "balance": {"start": 128, "length": 15, "type": "numeric"},
                    "balance_sign": {"start": 143, "length": 1, "type": "alphanumeric"},
                }
            }

            return record.encode('ascii'), metadata

        return b"", {}

    def _generate_person_name(self) -> Dict[str, str]:
        """Generate random person name."""
        first_names = ["JAMES", "MARY", "JOHN", "PATRICIA", "ROBERT", "JENNIFER",
                      "MICHAEL", "LINDA", "WILLIAM", "ELIZABETH", "DAVID", "BARBARA"]
        last_names = ["SMITH", "JOHNSON", "WILLIAMS", "BROWN", "JONES", "GARCIA",
                     "MILLER", "DAVIS", "RODRIGUEZ", "MARTINEZ", "HERNANDEZ", "LOPEZ"]

        first = random.choice(first_names)
        last = random.choice(last_names)

        return {"first": first, "last": last, "full": f"{first} {last}"}

    def _generate_address(self) -> str:
        """Generate random street address."""
        return f"{random.randint(100, 9999)} {random.choice(['MAIN', 'OAK', 'MAPLE', 'CEDAR', 'ELM'])} {random.choice(['ST', 'AVE', 'BLVD', 'DR'])}"

    def _generate_city(self) -> str:
        """Generate random city name."""
        return random.choice(["NEW YORK", "LOS ANGELES", "CHICAGO", "HOUSTON", "PHOENIX",
                             "PHILADELPHIA", "SAN ANTONIO", "SAN DIEGO", "DALLAS", "SAN JOSE"])

    def _generate_merchant_name(self) -> str:
        """Generate merchant name and location."""
        merchants = ["WALMART", "TARGET", "AMAZON", "COSTCO", "KROGER", "WALGREENS",
                    "CVS", "HOME DEPOT", "LOWES", "BEST BUY", "MCDONALDS", "STARBUCKS"]
        cities = ["NEW YORK", "LOS ANGELES", "CHICAGO", "HOUSTON", "PHOENIX"]

        return f"{random.choice(merchants)}          {random.choice(cities)}    US"


def generate_dataset(output_dir: Path, num_records: int = 10000):
    """Generate complete transaction dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = TransactionGenerator(seed=42)

    # Generate ISO 8583 messages
    iso8583_records = []
    for i in range(num_records // 3):
        iso8583_records.append(generator.generate_iso8583_message())

    with open(output_dir / "iso8583_transactions.json", 'w') as f:
        json.dump({"transactions": iso8583_records, "count": len(iso8583_records)}, f, indent=2)
    print(f"Generated {len(iso8583_records)} ISO 8583 transactions")

    # Generate CICS transactions
    cics_records = []
    for i in range(num_records // 3):
        cics_records.append(generator.generate_cics_transaction())

    with open(output_dir / "cics_transactions.json", 'w') as f:
        json.dump({"transactions": cics_records, "count": len(cics_records)}, f, indent=2)
    print(f"Generated {len(cics_records)} CICS transactions")

    # Generate batch records
    batch_records = []
    for i in range(num_records // 3):
        batch_records.append(generator.generate_batch_record())

    with open(output_dir / "batch_transactions.json", 'w') as f:
        json.dump({"transactions": batch_records, "count": len(batch_records)}, f, indent=2)
    print(f"Generated {len(batch_records)} batch records")

    # Generate fixed-width customer records
    customer_records = []
    customer_binary = b""
    for i in range(1000):
        record_bytes, metadata = generator.generate_fixed_width_record("customer")
        customer_binary += record_bytes
        customer_records.append(metadata)

    with open(output_dir / "customer_records.dat", 'wb') as f:
        f.write(customer_binary)

    with open(output_dir / "customer_records_metadata.json", 'w') as f:
        json.dump({"records": customer_records, "count": len(customer_records), "record_length": 200}, f, indent=2)
    print(f"Generated {len(customer_records)} customer records")

    # Generate dataset metadata
    metadata = {
        "generated_date": datetime.now().isoformat(),
        "total_records": num_records + 1000,
        "files": {
            "iso8583_transactions.json": {"count": len(iso8583_records), "type": "ISO 8583 financial messages"},
            "cics_transactions.json": {"count": len(cics_records), "type": "CICS online transactions"},
            "batch_transactions.json": {"count": len(batch_records), "type": "Batch job records"},
            "customer_records.dat": {"count": len(customer_records), "type": "Fixed-width customer master"},
        }
    }

    with open(output_dir / "dataset_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDataset generation complete!")
    print(f"Total records: {num_records + 1000}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    import sys

    output_dir = Path(__file__).parent
    num_records = int(sys.argv[1]) if len(sys.argv) > 1 else 10000

    generate_dataset(output_dir, num_records)
