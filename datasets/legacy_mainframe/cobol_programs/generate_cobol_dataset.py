#!/usr/bin/env python3
"""
QBITEL - COBOL Program Dataset Generator

Generates a comprehensive dataset of COBOL programs for training and RAG.
Covers various business domains, patterns, and complexity levels.

This generator creates:
1. Complete COBOL programs with all divisions
2. Copybooks (reusable data structures)
3. Annotated metadata for each program
4. Complexity scores and pattern classifications
"""

import json
import random
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple
import string

# =============================================================================
# COBOL Templates and Patterns
# =============================================================================

BUSINESS_DOMAINS = [
    "banking", "insurance", "healthcare", "retail", "manufacturing",
    "telecommunications", "government", "utilities", "transportation", "finance"
]

PROGRAM_TYPES = [
    "batch_processing", "online_transaction", "report_generation",
    "file_maintenance", "data_validation", "calculation_engine",
    "interface_program", "conversion_utility", "audit_trail", "archival"
]

COBOL_VERBS = [
    "MOVE", "ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", "COMPUTE",
    "IF", "EVALUATE", "PERFORM", "GO TO", "CALL", "STRING", "UNSTRING",
    "INSPECT", "READ", "WRITE", "REWRITE", "DELETE", "START", "OPEN", "CLOSE",
    "ACCEPT", "DISPLAY", "INITIALIZE", "SET", "SEARCH", "SORT", "MERGE"
]

DATA_TYPES = [
    ("PIC 9", "numeric_display", 1, 18),
    ("PIC S9", "signed_numeric", 1, 18),
    ("PIC X", "alphanumeric", 1, 256),
    ("PIC A", "alphabetic", 1, 100),
    ("PIC 9 COMP", "binary", 1, 18),
    ("PIC S9 COMP", "signed_binary", 1, 18),
    ("PIC S9 COMP-3", "packed_decimal", 1, 18),
    ("PIC 9V9", "decimal_display", 1, 18),
    ("PIC S9V9 COMP-3", "signed_packed_decimal", 1, 18),
]

FILE_ORGANIZATIONS = [
    "SEQUENTIAL", "INDEXED", "RELATIVE", "LINE SEQUENTIAL"
]

ACCESS_MODES = [
    "SEQUENTIAL", "RANDOM", "DYNAMIC"
]


class COBOLProgramGenerator:
    """Generates realistic COBOL programs for training datasets."""

    def __init__(self, seed: int = None):
        if seed:
            random.seed(seed)
        self.program_counter = 0

    def generate_program_id(self, domain: str, program_type: str) -> str:
        """Generate a realistic COBOL program ID (8 characters max)."""
        prefixes = {
            "banking": ["BNK", "ACT", "TRX", "LED", "CUS"],
            "insurance": ["INS", "POL", "CLM", "UND", "PRM"],
            "healthcare": ["HLT", "PAT", "CLN", "MED", "BIL"],
            "retail": ["RTL", "INV", "POS", "ORD", "SKU"],
            "manufacturing": ["MFG", "BOM", "WIP", "QTY", "PRD"],
            "telecommunications": ["TEL", "CDR", "NET", "SVC", "BIL"],
            "government": ["GOV", "TAX", "SSN", "BEN", "REG"],
            "utilities": ["UTL", "MTR", "BIL", "USG", "RAT"],
            "transportation": ["TRN", "SHP", "FRT", "RTE", "TRK"],
            "finance": ["FIN", "TRD", "SEC", "PRT", "VAL"],
        }
        type_codes = {
            "batch_processing": "B", "online_transaction": "O",
            "report_generation": "R", "file_maintenance": "M",
            "data_validation": "V", "calculation_engine": "C",
            "interface_program": "I", "conversion_utility": "U",
            "audit_trail": "A", "archival": "X"
        }
        prefix = random.choice(prefixes.get(domain, ["PRG"]))
        type_code = type_codes.get(program_type, "P")
        num = random.randint(100, 999)
        return f"{prefix}{type_code}{num}"

    def generate_identification_division(self, program_id: str, author: str,
                                         domain: str, description: str) -> str:
        """Generate the IDENTIFICATION DIVISION."""
        date_written = (datetime.now() - timedelta(days=random.randint(365, 15000))).strftime("%Y-%m-%d")
        date_compiled = datetime.now().strftime("%Y-%m-%d")

        return f"""       IDENTIFICATION DIVISION.
       PROGRAM-ID. {program_id}.
       AUTHOR. {author}.
       INSTALLATION. QBITEL-MAINFRAME.
       DATE-WRITTEN. {date_written}.
       DATE-COMPILED. {date_compiled}.
       SECURITY. CONFIDENTIAL.
      *
      * {description}
      * Domain: {domain.upper()}
      * Generated for QBITEL Training Dataset
      *
"""

    def generate_environment_division(self, files: List[Dict]) -> str:
        """Generate the ENVIRONMENT DIVISION."""
        env = """       ENVIRONMENT DIVISION.
       CONFIGURATION SECTION.
       SOURCE-COMPUTER. IBM-ZOS.
       OBJECT-COMPUTER. IBM-ZOS.
       SPECIAL-NAMES.
           DECIMAL-POINT IS COMMA.
      *
       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
"""
        for f in files:
            org = f.get("organization", "SEQUENTIAL")
            access = f.get("access_mode", "SEQUENTIAL")
            env += f"""           SELECT {f['name']}
               ASSIGN TO {f['ddname']}
               ORGANIZATION IS {org}
"""
            if org == "INDEXED":
                env += f"""               ACCESS MODE IS {access}
               RECORD KEY IS {f['record_key']}
               FILE STATUS IS {f['status_var']}.
"""
            else:
                env += f"""               FILE STATUS IS {f['status_var']}.
"""
        return env

    def generate_data_division(self, files: List[Dict], working_storage: List[Dict],
                               linkage: List[Dict] = None) -> str:
        """Generate the DATA DIVISION."""
        data = """       DATA DIVISION.
       FILE SECTION.
"""
        # File descriptions
        for f in files:
            data += f"""       FD {f['name']}
           RECORDING MODE IS F
           BLOCK CONTAINS 0 RECORDS
           RECORD CONTAINS {f['record_length']} CHARACTERS.
       01 {f['record_name']}.
"""
            for field in f['fields']:
                data += self._format_field(field, 10)

        # Working storage
        data += """      *
       WORKING-STORAGE SECTION.
      *--- PROGRAM CONSTANTS ---
       01 WS-CONSTANTS.
           05 WS-PROGRAM-NAME      PIC X(8) VALUE '{}'.
           05 WS-PROGRAM-VERSION   PIC X(6) VALUE '01.00'.
      *
      *--- FILE STATUS VARIABLES ---
       01 WS-FILE-STATUS-VARS.
""".format(files[0]['name'][:8] if files else "PROGRAM")

        for f in files:
            data += f"""           05 {f['status_var']}     PIC XX VALUE SPACES.
"""

        data += """      *
      *--- WORKING VARIABLES ---
       01 WS-WORK-AREAS.
"""
        for var in working_storage:
            data += self._format_field(var, 10)

        data += """      *
      *--- COUNTERS AND ACCUMULATORS ---
       01 WS-COUNTERS.
           05 WS-RECORDS-READ      PIC 9(9) COMP VALUE 0.
           05 WS-RECORDS-WRITTEN   PIC 9(9) COMP VALUE 0.
           05 WS-RECORDS-UPDATED   PIC 9(9) COMP VALUE 0.
           05 WS-RECORDS-DELETED   PIC 9(9) COMP VALUE 0.
           05 WS-ERROR-COUNT       PIC 9(9) COMP VALUE 0.
      *
      *--- FLAGS AND SWITCHES ---
       01 WS-FLAGS.
           05 WS-EOF-FLAG          PIC 9 VALUE 0.
              88 END-OF-FILE       VALUE 1.
              88 NOT-END-OF-FILE   VALUE 0.
           05 WS-ERROR-FLAG        PIC 9 VALUE 0.
              88 ERROR-OCCURRED    VALUE 1.
              88 NO-ERROR          VALUE 0.
           05 WS-FIRST-TIME        PIC 9 VALUE 1.
              88 IS-FIRST-TIME     VALUE 1.
              88 NOT-FIRST-TIME    VALUE 0.
      *
      *--- DATE AND TIME ---
       01 WS-DATE-TIME.
           05 WS-CURRENT-DATE.
              10 WS-YEAR           PIC 9(4).
              10 WS-MONTH          PIC 9(2).
              10 WS-DAY            PIC 9(2).
           05 WS-CURRENT-TIME.
              10 WS-HOUR           PIC 9(2).
              10 WS-MINUTE         PIC 9(2).
              10 WS-SECOND         PIC 9(2).
"""

        # Linkage section if provided
        if linkage:
            data += """      *
       LINKAGE SECTION.
       01 LS-PARAMETERS.
"""
            for var in linkage:
                data += self._format_field(var, 10)

        return data

    def generate_procedure_division(self, program_type: str,
                                    files: List[Dict],
                                    has_linkage: bool = False) -> str:
        """Generate the PROCEDURE DIVISION."""
        if has_linkage:
            proc = """       PROCEDURE DIVISION USING LS-PARAMETERS.
"""
        else:
            proc = """       PROCEDURE DIVISION.
"""

        proc += """      *===============================================================
      * MAIN PROGRAM LOGIC
      *===============================================================
       0000-MAIN-LOGIC.
           PERFORM 1000-INITIALIZATION
           PERFORM 2000-PROCESS-MAIN UNTIL END-OF-FILE
           PERFORM 9000-TERMINATION
           STOP RUN.
      *
      *===============================================================
      * INITIALIZATION
      *===============================================================
       1000-INITIALIZATION.
           PERFORM 1100-INIT-VARIABLES
           PERFORM 1200-OPEN-FILES
           PERFORM 1300-READ-FIRST-RECORD.
      *
       1100-INIT-VARIABLES.
           INITIALIZE WS-COUNTERS
           INITIALIZE WS-FLAGS
           MOVE FUNCTION CURRENT-DATE TO WS-DATE-TIME.
      *
       1200-OPEN-FILES.
"""
        for f in files:
            mode = f.get("open_mode", "INPUT")
            proc += f"""           OPEN {mode} {f['name']}
           IF {f['status_var']} NOT = '00'
               DISPLAY 'ERROR OPENING {f['name']}: ' {f['status_var']}
               SET ERROR-OCCURRED TO TRUE
               PERFORM 9999-ABORT-PROGRAM
           END-IF.
"""

        proc += """      *
       1300-READ-FIRST-RECORD.
           PERFORM 2100-READ-RECORD.
      *
      *===============================================================
      * MAIN PROCESSING
      *===============================================================
       2000-PROCESS-MAIN.
           PERFORM 2200-PROCESS-RECORD
           PERFORM 2100-READ-RECORD.
      *
       2100-READ-RECORD.
"""
        if files:
            f = files[0]
            proc += f"""           READ {f['name']}
               AT END
                   SET END-OF-FILE TO TRUE
               NOT AT END
                   ADD 1 TO WS-RECORDS-READ
           END-READ.
      *
"""

        proc += self._generate_processing_logic(program_type, files)

        proc += """      *
      *===============================================================
      * TERMINATION
      *===============================================================
       9000-TERMINATION.
           PERFORM 9100-CLOSE-FILES
           PERFORM 9200-DISPLAY-STATISTICS.
      *
       9100-CLOSE-FILES.
"""
        for f in files:
            proc += f"""           CLOSE {f['name']}.
"""

        proc += """      *
       9200-DISPLAY-STATISTICS.
           DISPLAY '========================================='
           DISPLAY 'PROGRAM STATISTICS'
           DISPLAY '========================================='
           DISPLAY 'RECORDS READ:    ' WS-RECORDS-READ
           DISPLAY 'RECORDS WRITTEN: ' WS-RECORDS-WRITTEN
           DISPLAY 'RECORDS UPDATED: ' WS-RECORDS-UPDATED
           DISPLAY 'ERRORS:          ' WS-ERROR-COUNT
           DISPLAY '========================================='.
      *
       9999-ABORT-PROGRAM.
           DISPLAY 'PROGRAM ABORTED DUE TO ERROR'
           PERFORM 9100-CLOSE-FILES
           STOP RUN.
"""
        return proc

    def _generate_processing_logic(self, program_type: str, files: List[Dict]) -> str:
        """Generate processing logic based on program type."""
        logic = """       2200-PROCESS-RECORD.
"""
        if program_type == "batch_processing":
            logic += """           EVALUATE TRUE
               WHEN WS-RECORD-TYPE = 'H'
                   PERFORM 3000-PROCESS-HEADER
               WHEN WS-RECORD-TYPE = 'D'
                   PERFORM 3100-PROCESS-DETAIL
               WHEN WS-RECORD-TYPE = 'T'
                   PERFORM 3200-PROCESS-TRAILER
               WHEN OTHER
                   ADD 1 TO WS-ERROR-COUNT
                   PERFORM 8000-HANDLE-ERROR
           END-EVALUATE.
      *
       3000-PROCESS-HEADER.
           CONTINUE.
      *
       3100-PROCESS-DETAIL.
           PERFORM 3110-VALIDATE-DATA
           IF NO-ERROR
               PERFORM 3120-APPLY-BUSINESS-RULES
               PERFORM 3130-UPDATE-ACCUMULATORS
           END-IF.
      *
       3110-VALIDATE-DATA.
           SET NO-ERROR TO TRUE
           IF WS-FIELD-1 = SPACES
               SET ERROR-OCCURRED TO TRUE
               ADD 1 TO WS-ERROR-COUNT
           END-IF.
      *
       3120-APPLY-BUSINESS-RULES.
           CONTINUE.
      *
       3130-UPDATE-ACCUMULATORS.
           ADD 1 TO WS-RECORDS-WRITTEN.
      *
       3200-PROCESS-TRAILER.
           CONTINUE.
      *
       8000-HANDLE-ERROR.
           DISPLAY 'ERROR IN RECORD: ' WS-RECORDS-READ.
"""
        elif program_type == "calculation_engine":
            logic += """           PERFORM 3000-CALCULATE-VALUES
           PERFORM 3100-VALIDATE-RESULTS
           IF NO-ERROR
               PERFORM 3200-STORE-RESULTS
           END-IF.
      *
       3000-CALCULATE-VALUES.
           COMPUTE WS-RESULT = WS-AMOUNT * WS-RATE / 100
           COMPUTE WS-TAX = WS-RESULT * WS-TAX-RATE / 100
           COMPUTE WS-TOTAL = WS-RESULT + WS-TAX.
      *
       3100-VALIDATE-RESULTS.
           SET NO-ERROR TO TRUE
           IF WS-TOTAL < 0
               SET ERROR-OCCURRED TO TRUE
               ADD 1 TO WS-ERROR-COUNT
           END-IF.
      *
       3200-STORE-RESULTS.
           ADD 1 TO WS-RECORDS-WRITTEN.
"""
        elif program_type == "data_validation":
            logic += """           PERFORM 3000-VALIDATE-FIELDS
           IF NO-ERROR
               PERFORM 3100-WRITE-VALID-RECORD
           ELSE
               PERFORM 3200-WRITE-ERROR-RECORD
           END-IF.
      *
       3000-VALIDATE-FIELDS.
           SET NO-ERROR TO TRUE
           PERFORM 3010-VALIDATE-NUMERIC-FIELDS
           PERFORM 3020-VALIDATE-DATE-FIELDS
           PERFORM 3030-VALIDATE-CODE-FIELDS.
      *
       3010-VALIDATE-NUMERIC-FIELDS.
           IF WS-AMOUNT NOT NUMERIC
               SET ERROR-OCCURRED TO TRUE
               ADD 1 TO WS-ERROR-COUNT
           END-IF.
      *
       3020-VALIDATE-DATE-FIELDS.
           IF WS-DATE < 19000101 OR WS-DATE > 99991231
               SET ERROR-OCCURRED TO TRUE
               ADD 1 TO WS-ERROR-COUNT
           END-IF.
      *
       3030-VALIDATE-CODE-FIELDS.
           EVALUATE WS-STATUS-CODE
               WHEN 'A' CONTINUE
               WHEN 'I' CONTINUE
               WHEN 'C' CONTINUE
               WHEN OTHER
                   SET ERROR-OCCURRED TO TRUE
                   ADD 1 TO WS-ERROR-COUNT
           END-EVALUATE.
      *
       3100-WRITE-VALID-RECORD.
           ADD 1 TO WS-RECORDS-WRITTEN.
      *
       3200-WRITE-ERROR-RECORD.
           ADD 1 TO WS-ERROR-COUNT.
"""
        else:
            logic += """           PERFORM 3000-APPLY-BUSINESS-LOGIC
           ADD 1 TO WS-RECORDS-WRITTEN.
      *
       3000-APPLY-BUSINESS-LOGIC.
           CONTINUE.
"""
        return logic

    def _format_field(self, field: Dict, indent: int) -> str:
        """Format a COBOL field definition."""
        spaces = " " * indent
        level = field.get("level", "05")
        name = field.get("name", "FILLER")
        pic = field.get("pic", "X(10)")
        value = field.get("value")

        line = f"{spaces}{level} {name}"
        if pic:
            line += f" PIC {pic}"
        if value is not None:
            if isinstance(value, str):
                line += f" VALUE '{value}'"
            else:
                line += f" VALUE {value}"
        line += ".\n"
        return line

    def generate_complete_program(self, domain: str = None,
                                  program_type: str = None,
                                  complexity: str = "medium") -> Dict[str, Any]:
        """Generate a complete COBOL program with metadata."""
        self.program_counter += 1

        if not domain:
            domain = random.choice(BUSINESS_DOMAINS)
        if not program_type:
            program_type = random.choice(PROGRAM_TYPES)

        program_id = self.generate_program_id(domain, program_type)
        author = random.choice([
            "J. SMITH", "M. JOHNSON", "R. WILLIAMS", "S. BROWN",
            "D. JONES", "T. DAVIS", "C. MILLER", "P. WILSON"
        ])

        descriptions = {
            "batch_processing": f"Batch processing program for {domain} transactions",
            "online_transaction": f"Online transaction processor for {domain} system",
            "report_generation": f"Report generator for {domain} analysis",
            "file_maintenance": f"File maintenance utility for {domain} master files",
            "data_validation": f"Data validation program for {domain} input",
            "calculation_engine": f"Calculation engine for {domain} computations",
            "interface_program": f"Interface program for {domain} external systems",
            "conversion_utility": f"Data conversion utility for {domain} migration",
            "audit_trail": f"Audit trail generator for {domain} compliance",
            "archival": f"Data archival program for {domain} retention",
        }

        # Generate files based on complexity
        num_files = {"simple": 1, "medium": 2, "complex": 4}.get(complexity, 2)
        files = self._generate_files(domain, num_files)

        # Generate working storage variables
        working_storage = self._generate_working_storage(domain, complexity)

        # Assemble the program
        program = ""
        program += self.generate_identification_division(
            program_id, author, domain, descriptions.get(program_type, "General processing")
        )
        program += self.generate_environment_division(files)
        program += self.generate_data_division(files, working_storage)
        program += self.generate_procedure_division(program_type, files)

        # Calculate complexity metrics
        lines = program.count('\n')
        verbs_used = sum(1 for verb in COBOL_VERBS if verb in program)
        complexity_score = min(1.0, (lines / 500) * 0.4 + (verbs_used / 20) * 0.3 + (num_files / 4) * 0.3)

        # Generate metadata
        metadata = {
            "program_id": program_id,
            "domain": domain,
            "program_type": program_type,
            "author": author,
            "lines_of_code": lines,
            "complexity_level": complexity,
            "complexity_score": round(complexity_score, 3),
            "files_used": num_files,
            "verbs_used": verbs_used,
            "has_indexed_files": any(f.get("organization") == "INDEXED" for f in files),
            "has_packed_decimal": "COMP-3" in program,
            "has_binary_fields": "COMP " in program,
            "has_perform_loops": "PERFORM" in program,
            "has_evaluate": "EVALUATE" in program,
            "has_call_statements": "CALL " in program,
            "generated_date": datetime.now().isoformat(),
            "hash": hashlib.md5(program.encode()).hexdigest()
        }

        return {
            "source_code": program,
            "metadata": metadata,
            "files": files
        }

    def _generate_files(self, domain: str, num_files: int) -> List[Dict]:
        """Generate file definitions."""
        files = []
        file_types = {
            "banking": ["CUSTOMER", "ACCOUNT", "TRANSACT", "BALANCE", "AUDIT"],
            "insurance": ["POLICY", "CLAIM", "PREMIUM", "BENEFIC", "COVERAGE"],
            "healthcare": ["PATIENT", "MEDICAL", "BILLING", "PROVIDER", "CLAIM"],
            "retail": ["PRODUCT", "INVENTRY", "ORDER", "CUSTOMER", "SALES"],
            "default": ["MASTER", "DETAIL", "HISTORY", "REPORT", "ERROR"]
        }

        prefixes = file_types.get(domain, file_types["default"])

        for i in range(num_files):
            prefix = prefixes[i % len(prefixes)]
            org = random.choice(FILE_ORGANIZATIONS[:2])  # Mostly SEQUENTIAL or INDEXED
            access = "DYNAMIC" if org == "INDEXED" else "SEQUENTIAL"

            file_def = {
                "name": f"{prefix}-FILE",
                "ddname": f"DD{prefix[:6]}",
                "record_name": f"{prefix}-RECORD",
                "record_length": random.choice([80, 100, 200, 500, 1000]),
                "organization": org,
                "access_mode": access,
                "status_var": f"WS-{prefix[:4]}-STATUS",
                "open_mode": "INPUT" if i == 0 else random.choice(["INPUT", "OUTPUT", "I-O"]),
                "fields": self._generate_record_fields(prefix, domain)
            }

            if org == "INDEXED":
                file_def["record_key"] = f"{prefix}-KEY"

            files.append(file_def)

        return files

    def _generate_record_fields(self, prefix: str, domain: str) -> List[Dict]:
        """Generate record field definitions."""
        fields = [
            {"level": "05", "name": f"{prefix}-KEY", "pic": "X(10)"},
        ]

        domain_fields = {
            "banking": [
                {"level": "05", "name": f"{prefix}-ACCOUNT-NO", "pic": "9(12)"},
                {"level": "05", "name": f"{prefix}-AMOUNT", "pic": "S9(11)V99 COMP-3"},
                {"level": "05", "name": f"{prefix}-DATE", "pic": "9(8)"},
                {"level": "05", "name": f"{prefix}-STATUS", "pic": "X(1)"},
            ],
            "insurance": [
                {"level": "05", "name": f"{prefix}-POLICY-NO", "pic": "X(15)"},
                {"level": "05", "name": f"{prefix}-PREMIUM", "pic": "S9(9)V99 COMP-3"},
                {"level": "05", "name": f"{prefix}-EFF-DATE", "pic": "9(8)"},
                {"level": "05", "name": f"{prefix}-EXP-DATE", "pic": "9(8)"},
            ],
            "healthcare": [
                {"level": "05", "name": f"{prefix}-PATIENT-ID", "pic": "X(12)"},
                {"level": "05", "name": f"{prefix}-MRN", "pic": "9(10)"},
                {"level": "05", "name": f"{prefix}-DOS", "pic": "9(8)"},
                {"level": "05", "name": f"{prefix}-CHARGE", "pic": "S9(7)V99 COMP-3"},
            ],
        }

        fields.extend(domain_fields.get(domain, [
            {"level": "05", "name": f"{prefix}-ID", "pic": "9(10)"},
            {"level": "05", "name": f"{prefix}-DATA", "pic": "X(50)"},
            {"level": "05", "name": f"{prefix}-AMOUNT", "pic": "S9(9)V99 COMP-3"},
        ]))

        fields.append({"level": "05", "name": "FILLER", "pic": "X(20)"})

        return fields

    def _generate_working_storage(self, domain: str, complexity: str) -> List[Dict]:
        """Generate working storage variables."""
        vars = [
            {"level": "05", "name": "WS-RECORD-TYPE", "pic": "X(1)"},
            {"level": "05", "name": "WS-FIELD-1", "pic": "X(20)"},
            {"level": "05", "name": "WS-AMOUNT", "pic": "S9(11)V99 COMP-3", "value": 0},
            {"level": "05", "name": "WS-RATE", "pic": "S9(3)V9(4) COMP-3", "value": 0},
            {"level": "05", "name": "WS-TAX-RATE", "pic": "S9(3)V9(4) COMP-3", "value": 0},
            {"level": "05", "name": "WS-RESULT", "pic": "S9(13)V99 COMP-3", "value": 0},
            {"level": "05", "name": "WS-TAX", "pic": "S9(11)V99 COMP-3", "value": 0},
            {"level": "05", "name": "WS-TOTAL", "pic": "S9(13)V99 COMP-3", "value": 0},
            {"level": "05", "name": "WS-DATE", "pic": "9(8)", "value": 0},
            {"level": "05", "name": "WS-STATUS-CODE", "pic": "X(1)"},
        ]

        if complexity in ["medium", "complex"]:
            vars.extend([
                {"level": "05", "name": "WS-PREV-KEY", "pic": "X(10)"},
                {"level": "05", "name": "WS-SAVE-AREA", "pic": "X(100)"},
                {"level": "05", "name": "WS-RETURN-CODE", "pic": "S9(4) COMP", "value": 0},
            ])

        return vars


def generate_dataset(output_dir: Path, num_programs: int = 1000):
    """Generate the complete COBOL dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = COBOLProgramGenerator(seed=42)

    programs = []
    metadata_list = []

    # Generate diverse programs
    for i in range(num_programs):
        complexity = random.choices(
            ["simple", "medium", "complex"],
            weights=[0.3, 0.5, 0.2]
        )[0]

        result = generator.generate_complete_program(complexity=complexity)

        # Save individual program
        program_file = output_dir / f"{result['metadata']['program_id']}.cbl"
        with open(program_file, 'w') as f:
            f.write(result['source_code'])

        # Save metadata
        meta_file = output_dir / f"{result['metadata']['program_id']}.json"
        with open(meta_file, 'w') as f:
            json.dump(result['metadata'], f, indent=2)

        metadata_list.append(result['metadata'])

        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_programs} programs...")

    # Save combined metadata
    combined_meta = output_dir / "dataset_metadata.json"
    with open(combined_meta, 'w') as f:
        json.dump({
            "total_programs": num_programs,
            "generated_date": datetime.now().isoformat(),
            "domains": BUSINESS_DOMAINS,
            "program_types": PROGRAM_TYPES,
            "programs": metadata_list
        }, f, indent=2)

    print(f"\nDataset generation complete!")
    print(f"Total programs: {num_programs}")
    print(f"Output directory: {output_dir}")

    # Print statistics
    complexity_counts = {}
    domain_counts = {}
    type_counts = {}

    for meta in metadata_list:
        complexity_counts[meta['complexity_level']] = complexity_counts.get(meta['complexity_level'], 0) + 1
        domain_counts[meta['domain']] = domain_counts.get(meta['domain'], 0) + 1
        type_counts[meta['program_type']] = type_counts.get(meta['program_type'], 0) + 1

    print(f"\nComplexity distribution: {complexity_counts}")
    print(f"Domain distribution: {domain_counts}")
    print(f"Program type distribution: {type_counts}")


if __name__ == "__main__":
    import sys

    output_dir = Path(__file__).parent
    num_programs = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    generate_dataset(output_dir, num_programs)
