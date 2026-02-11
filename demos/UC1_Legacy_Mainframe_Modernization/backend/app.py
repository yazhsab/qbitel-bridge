"""
QBITEL - UC1 Legacy Mainframe Modernization Demo
Complete working demonstration of mainframe modernization capabilities.
"""

import asyncio
import json
import os
import sys
import time
import uuid
import struct
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field, asdict

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Demo directory
DEMO_DIR = Path(__file__).parent.parent
COBOL_DIR = DEMO_DIR / "cobol_samples"
DATA_DIR = DEMO_DIR / "data"
OUTPUT_DIR = DEMO_DIR / "output"

app = FastAPI(
    title="QBITEL Bridge - Legacy Mainframe Modernization Demo",
    description="UC1: Demonstrates AI-powered legacy mainframe modernization capabilities",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Data Models
# ============================================================================

class SystemStatus(str, Enum):
    ACTIVE = "active"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"

class ModernizationApproach(str, Enum):
    REHOST = "rehost"
    REPLATFORM = "replatform"
    REFACTOR = "refactor"
    REARCHITECT = "rearchitect"
    REBUILD = "rebuild"
    REPLACE = "replace"

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class LegacySystem:
    system_id: str
    name: str
    type: str
    platform: str
    language: str
    lines_of_code: int
    age_years: int
    business_criticality: str
    status: SystemStatus
    last_modified: datetime
    dependencies: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class COBOLProgram:
    program_id: str
    name: str
    file_path: str
    lines_of_code: int
    copybooks: List[str]
    data_divisions: int
    procedure_divisions: int
    complexity_score: float
    last_modified: datetime
    analysis: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProtocolField:
    name: str
    offset: int
    length: int
    field_type: str
    description: str
    pic_clause: str = ""
    is_numeric: bool = False
    is_signed: bool = False
    decimal_places: int = 0

@dataclass
class ModernizationPlan:
    plan_id: str
    system_id: str
    approach: ModernizationApproach
    risk_level: RiskLevel
    estimated_effort_days: int
    phases: List[Dict[str, Any]]
    generated_code: Dict[str, str]
    documentation: str
    created_at: datetime

# Pydantic models for API
class AnalyzeRequest(BaseModel):
    cobol_file: str
    include_metrics: bool = True

class ModernizeRequest(BaseModel):
    system_id: str
    approach: str = "refactor"
    target_language: str = "python"
    target_framework: str = "fastapi"

class ProtocolAnalyzeRequest(BaseModel):
    protocol_data: str  # hex encoded
    system_context: str = ""

# ============================================================================
# Simulated Legacy Mainframe Environment
# ============================================================================

class MainframeSimulator:
    """Simulates a legacy IBM mainframe environment."""

    def __init__(self):
        self.systems: Dict[str, LegacySystem] = {}
        self.programs: Dict[str, COBOLProgram] = {}
        self.transactions: List[Dict] = []
        self.job_queue: List[Dict] = []
        self._initialize_demo_systems()

    def _initialize_demo_systems(self):
        """Initialize demo legacy systems."""
        # Main banking core system
        self.systems["SYS001"] = LegacySystem(
            system_id="SYS001",
            name="Core Banking System",
            type="transaction_processing",
            platform="IBM z/OS",
            language="COBOL",
            lines_of_code=2_500_000,
            age_years=38,
            business_criticality="critical",
            status=SystemStatus.ACTIVE,
            last_modified=datetime(2024, 6, 15),
            dependencies=["DB2", "CICS", "MQ Series", "IMS"],
            metrics={
                "daily_transactions": 15_000_000,
                "avg_response_time_ms": 45,
                "uptime_percent": 99.97,
                "batch_jobs_per_day": 2500
            }
        )

        # Customer master system
        self.systems["SYS002"] = LegacySystem(
            system_id="SYS002",
            name="Customer Master System",
            type="master_data",
            platform="IBM z/OS",
            language="COBOL",
            lines_of_code=850_000,
            age_years=32,
            business_criticality="high",
            status=SystemStatus.ACTIVE,
            last_modified=datetime(2023, 11, 20),
            dependencies=["DB2", "VSAM", "CICS"],
            metrics={
                "total_records": 45_000_000,
                "daily_updates": 250_000,
                "avg_query_time_ms": 12
            }
        )

        # Account processing system
        self.systems["SYS003"] = LegacySystem(
            system_id="SYS003",
            name="Account Processing Batch",
            type="batch_processing",
            platform="IBM z/OS",
            language="COBOL",
            lines_of_code=1_200_000,
            age_years=29,
            business_criticality="high",
            status=SystemStatus.DEGRADED,
            last_modified=datetime(2024, 1, 10),
            dependencies=["DB2", "VSAM", "JCL"],
            metrics={
                "nightly_batch_duration_hours": 4.5,
                "records_processed": 120_000_000,
                "failure_rate_percent": 0.02
            }
        )

    def get_system(self, system_id: str) -> Optional[LegacySystem]:
        return self.systems.get(system_id)

    def get_all_systems(self) -> List[LegacySystem]:
        return list(self.systems.values())

    def submit_job(self, job_name: str, job_type: str, parameters: Dict) -> str:
        """Submit a batch job to the mainframe."""
        job_id = f"JOB{int(time.time())}"
        self.job_queue.append({
            "job_id": job_id,
            "job_name": job_name,
            "job_type": job_type,
            "parameters": parameters,
            "status": "queued",
            "submitted_at": datetime.now().isoformat()
        })
        return job_id

# ============================================================================
# COBOL Analyzer
# ============================================================================

class COBOLAnalyzer:
    """Analyzes COBOL programs for modernization."""

    def __init__(self):
        self.pic_patterns = {
            "9": "numeric",
            "X": "alphanumeric",
            "A": "alphabetic",
            "S": "signed",
            "V": "decimal",
            "P": "assumed_decimal",
            "Z": "zero_suppressed",
            "COMP": "binary",
            "COMP-3": "packed_decimal"
        }

    def analyze_cobol_file(self, file_path: str) -> COBOLProgram:
        """Analyze a COBOL source file."""
        with open(file_path, 'r') as f:
            content = f.read()

        lines = content.split('\n')
        loc = len([l for l in lines if l.strip() and not l.strip().startswith('*')])

        # Extract program info
        program_id = self._extract_program_id(content)
        copybooks = self._extract_copybooks(content)
        data_divisions = self._count_data_divisions(content)
        procedure_divisions = self._count_procedure_divisions(content)
        complexity = self._calculate_complexity(content)

        # Deep analysis
        analysis = {
            "file_sections": self._analyze_file_sections(content),
            "working_storage": self._analyze_working_storage(content),
            "procedures": self._analyze_procedures(content),
            "data_flow": self._analyze_data_flow(content),
            "control_flow": self._analyze_control_flow(content),
            "legacy_patterns": self._identify_legacy_patterns(content),
            "modernization_opportunities": self._identify_modernization_opportunities(content)
        }

        return COBOLProgram(
            program_id=program_id,
            name=Path(file_path).stem,
            file_path=file_path,
            lines_of_code=loc,
            copybooks=copybooks,
            data_divisions=data_divisions,
            procedure_divisions=procedure_divisions,
            complexity_score=complexity,
            last_modified=datetime.fromtimestamp(os.path.getmtime(file_path)),
            analysis=analysis
        )

    def _extract_program_id(self, content: str) -> str:
        for line in content.split('\n'):
            if 'PROGRAM-ID' in line.upper():
                parts = line.split('.')
                if len(parts) >= 1:
                    return parts[0].split()[-1].strip()
        return "UNKNOWN"

    def _extract_copybooks(self, content: str) -> List[str]:
        copybooks = []
        for line in content.split('\n'):
            upper_line = line.upper()
            if 'COPY' in upper_line:
                parts = upper_line.split('COPY')
                if len(parts) > 1:
                    copybook = parts[1].strip().split()[0].replace('.', '')
                    copybooks.append(copybook)
        return copybooks

    def _count_data_divisions(self, content: str) -> int:
        return content.upper().count('DATA DIVISION')

    def _count_procedure_divisions(self, content: str) -> int:
        return content.upper().count('PROCEDURE DIVISION')

    def _calculate_complexity(self, content: str) -> float:
        """Calculate McCabe-like complexity for COBOL."""
        complexity = 1.0

        control_keywords = [
            'IF', 'EVALUATE', 'PERFORM', 'PERFORM UNTIL',
            'PERFORM VARYING', 'GO TO', 'CALL'
        ]

        for keyword in control_keywords:
            complexity += content.upper().count(keyword) * 0.5

        # Nested structures increase complexity
        complexity += content.count('END-IF') * 0.3
        complexity += content.count('END-EVALUATE') * 0.5
        complexity += content.count('END-PERFORM') * 0.3

        return round(complexity, 2)

    def _analyze_file_sections(self, content: str) -> List[Dict]:
        """Analyze FD (File Description) sections."""
        files = []
        lines = content.split('\n')
        current_file = None

        for i, line in enumerate(lines):
            upper_line = line.upper().strip()
            if upper_line.startswith('FD '):
                if current_file:
                    files.append(current_file)
                file_name = upper_line.split()[1].replace('.', '')
                current_file = {
                    "name": file_name,
                    "type": "unknown",
                    "records": []
                }
            elif upper_line.startswith('SELECT '):
                parts = upper_line.split()
                for j, part in enumerate(parts):
                    if part == 'ASSIGN':
                        if j + 2 < len(parts):
                            assign_to = parts[j + 2]
                            for f in files:
                                if f["name"] in upper_line:
                                    f["assign_to"] = assign_to

        if current_file:
            files.append(current_file)

        return files

    def _analyze_working_storage(self, content: str) -> Dict:
        """Analyze WORKING-STORAGE SECTION."""
        variables = []
        lines = content.split('\n')
        in_working_storage = False

        for line in lines:
            upper_line = line.upper()
            if 'WORKING-STORAGE SECTION' in upper_line:
                in_working_storage = True
                continue
            if in_working_storage:
                if 'PROCEDURE DIVISION' in upper_line or 'LINKAGE SECTION' in upper_line:
                    break

                # Parse variable definitions
                stripped = line.strip()
                if stripped and not stripped.startswith('*'):
                    level_match = stripped.split()
                    if level_match and level_match[0].isdigit():
                        level = int(level_match[0])
                        name = level_match[1] if len(level_match) > 1 else ""
                        pic = ""
                        if 'PIC' in upper_line:
                            pic_start = upper_line.find('PIC') + 4
                            pic_end = upper_line.find(' ', pic_start)
                            if pic_end == -1:
                                pic_end = upper_line.find('.', pic_start)
                            pic = upper_line[pic_start:pic_end].strip() if pic_end > pic_start else ""

                        variables.append({
                            "level": level,
                            "name": name.replace('.', ''),
                            "pic": pic,
                            "type": self._pic_to_type(pic)
                        })

        return {
            "variable_count": len(variables),
            "variables": variables[:50],  # First 50 for demo
            "has_88_levels": any(v["level"] == 88 for v in variables),
            "has_redefines": "REDEFINES" in content.upper()
        }

    def _pic_to_type(self, pic: str) -> str:
        """Convert PIC clause to data type."""
        if not pic:
            return "group"
        pic = pic.upper()
        if '9' in pic:
            if 'V' in pic:
                return "decimal"
            return "integer"
        if 'X' in pic:
            return "string"
        if 'A' in pic:
            return "alpha"
        return "unknown"

    def _analyze_procedures(self, content: str) -> List[Dict]:
        """Analyze procedure paragraphs."""
        procedures = []
        lines = content.split('\n')
        in_procedure_division = False
        current_para = None

        for line in lines:
            upper_line = line.upper()
            if 'PROCEDURE DIVISION' in upper_line:
                in_procedure_division = True
                continue

            if in_procedure_division:
                stripped = line.strip()
                # Check for paragraph header (ends with period, no PERFORM, etc.)
                if stripped and not stripped.startswith('*'):
                    if stripped.endswith('.') and ' ' not in stripped.replace('.', ''):
                        if current_para:
                            procedures.append(current_para)
                        current_para = {
                            "name": stripped.replace('.', ''),
                            "statements": [],
                            "calls_to": [],
                            "performs": []
                        }
                    elif current_para:
                        if 'PERFORM' in upper_line:
                            parts = upper_line.split('PERFORM')
                            if len(parts) > 1:
                                target_parts = parts[1].strip().split()
                                if target_parts:
                                    target = target_parts[0]
                                    current_para["performs"].append(target)
                        if 'CALL' in upper_line:
                            parts = upper_line.split('CALL')
                            if len(parts) > 1:
                                target_parts = parts[1].strip().split()
                                if target_parts:
                                    target = target_parts[0].replace("'", "").replace('"', '')
                                    current_para["calls_to"].append(target)

        if current_para:
            procedures.append(current_para)

        return procedures

    def _analyze_data_flow(self, content: str) -> Dict:
        """Analyze data flow patterns."""
        return {
            "move_statements": content.upper().count('MOVE '),
            "compute_statements": content.upper().count('COMPUTE '),
            "add_statements": content.upper().count('ADD '),
            "subtract_statements": content.upper().count('SUBTRACT '),
            "multiply_statements": content.upper().count('MULTIPLY '),
            "divide_statements": content.upper().count('DIVIDE '),
            "string_operations": content.upper().count('STRING ') + content.upper().count('UNSTRING '),
            "inspect_operations": content.upper().count('INSPECT ')
        }

    def _analyze_control_flow(self, content: str) -> Dict:
        """Analyze control flow patterns."""
        return {
            "if_statements": content.upper().count(' IF '),
            "evaluate_statements": content.upper().count('EVALUATE '),
            "perform_statements": content.upper().count('PERFORM '),
            "go_to_statements": content.upper().count('GO TO '),
            "call_statements": content.upper().count('CALL '),
            "exit_statements": content.upper().count('EXIT '),
            "stop_run": content.upper().count('STOP RUN')
        }

    def _identify_legacy_patterns(self, content: str) -> List[Dict]:
        """Identify legacy coding patterns."""
        patterns = []

        if content.upper().count('GO TO ') > 5:
            patterns.append({
                "pattern": "excessive_goto",
                "severity": "high",
                "description": "Excessive use of GO TO statements makes code hard to maintain",
                "recommendation": "Refactor to structured PERFORM statements"
            })

        if 'ALTER' in content.upper():
            patterns.append({
                "pattern": "alter_statement",
                "severity": "critical",
                "description": "ALTER statement modifies code flow at runtime",
                "recommendation": "Remove ALTER and use conditional logic"
            })

        if content.upper().count('WORKING-STORAGE') > 1:
            patterns.append({
                "pattern": "multiple_working_storage",
                "severity": "medium",
                "description": "Multiple working storage sections",
                "recommendation": "Consolidate into single section"
            })

        if 'COPY' in content.upper() and 'REPLACING' in content.upper():
            patterns.append({
                "pattern": "copy_replacing",
                "severity": "low",
                "description": "COPY with REPLACING used for code reuse",
                "recommendation": "Consider modular design patterns"
            })

        return patterns

    def _identify_modernization_opportunities(self, content: str) -> List[Dict]:
        """Identify modernization opportunities."""
        opportunities = []

        # Check for DB2 access
        if 'EXEC SQL' in content.upper():
            opportunities.append({
                "area": "database_access",
                "current": "Embedded SQL (DB2)",
                "modern": "ORM (SQLAlchemy) or async database access",
                "effort": "medium",
                "benefit": "Improved maintainability and performance"
            })

        # Check for CICS
        if 'EXEC CICS' in content.upper():
            opportunities.append({
                "area": "transaction_processing",
                "current": "CICS transactions",
                "modern": "REST APIs with FastAPI/Flask",
                "effort": "high",
                "benefit": "Cloud-native scalability"
            })

        # Check for file I/O
        if 'READ ' in content.upper() or 'WRITE ' in content.upper():
            opportunities.append({
                "area": "file_processing",
                "current": "Sequential/Indexed file I/O",
                "modern": "Modern file formats (JSON, Parquet) or databases",
                "effort": "medium",
                "benefit": "Better integration capabilities"
            })

        # Check for batch processing patterns
        if 'PERFORM UNTIL' in content.upper():
            opportunities.append({
                "area": "batch_processing",
                "current": "COBOL batch loops",
                "modern": "Apache Spark or streaming pipelines",
                "effort": "high",
                "benefit": "Parallel processing, real-time capabilities"
            })

        return opportunities

# ============================================================================
# Protocol Analyzer
# ============================================================================

class MainframeProtocolAnalyzer:
    """Analyzes mainframe binary protocols."""

    def __init__(self):
        self.known_protocols = {
            "3270": self._analyze_3270,
            "EBCDIC": self._analyze_ebcdic,
            "CICS": self._analyze_cics,
            "MQ": self._analyze_mq
        }

    def analyze_protocol(self, data: bytes, context: str = "") -> Dict:
        """Analyze protocol data."""
        analysis = {
            "raw_length": len(data),
            "encoding": self._detect_encoding(data),
            "structure": self._detect_structure(data),
            "fields": self._detect_fields(data),
            "patterns": self._detect_patterns(data),
            "recommendations": []
        }

        # Add modernization recommendations
        if analysis["encoding"] == "EBCDIC":
            analysis["recommendations"].append({
                "issue": "EBCDIC encoding detected",
                "solution": "Convert to UTF-8 for modern systems",
                "code_snippet": self._generate_conversion_code("EBCDIC", "UTF-8")
            })

        if analysis["structure"]["type"] == "fixed_length":
            analysis["recommendations"].append({
                "issue": "Fixed-length records",
                "solution": "Convert to JSON or Protocol Buffers",
                "code_snippet": self._generate_parser_code(analysis["fields"])
            })

        return analysis

    def _detect_encoding(self, data: bytes) -> str:
        """Detect data encoding."""
        # Simple heuristic: EBCDIC has different byte patterns
        ascii_count = sum(1 for b in data if 32 <= b <= 126)
        ebcdic_count = sum(1 for b in data if 64 <= b <= 249)

        ascii_ratio = ascii_count / len(data) if data else 0

        if ascii_ratio > 0.8:
            return "ASCII"
        elif ebcdic_count / len(data) > 0.5 if data else False:
            return "EBCDIC"
        return "BINARY"

    def _detect_structure(self, data: bytes) -> Dict:
        """Detect record structure."""
        # Check for common mainframe record formats
        if len(data) in [80, 132, 256, 512, 1024]:
            return {
                "type": "fixed_length",
                "record_length": len(data),
                "format": "FB" if len(data) == 80 else "VB"
            }

        # Check for variable length with RDW
        if len(data) >= 4:
            rdw_length = struct.unpack('>H', data[:2])[0]
            if rdw_length == len(data):
                return {
                    "type": "variable_length",
                    "has_rdw": True,
                    "record_length": rdw_length
                }

        return {
            "type": "unknown",
            "record_length": len(data)
        }

    def _detect_fields(self, data: bytes) -> List[Dict]:
        """Detect field boundaries."""
        fields = []

        # Simple boundary detection based on patterns
        offset = 0
        field_num = 1

        # Look for numeric fields (packed decimal patterns)
        i = 0
        while i < len(data):
            # Check for packed decimal (COMP-3)
            if i + 4 <= len(data):
                sample = data[i:i+4]
                if self._is_packed_decimal(sample):
                    fields.append({
                        "name": f"FIELD_{field_num:03d}",
                        "offset": i,
                        "length": 4,
                        "type": "packed_decimal",
                        "description": "Packed decimal number (COMP-3)"
                    })
                    field_num += 1
                    i += 4
                    continue

            # Check for character data
            if data[i] >= 64 and data[i] <= 249:  # EBCDIC printable range
                start = i
                while i < len(data) and data[i] >= 64 and data[i] <= 249:
                    i += 1
                if i - start >= 2:
                    fields.append({
                        "name": f"FIELD_{field_num:03d}",
                        "offset": start,
                        "length": i - start,
                        "type": "character",
                        "description": "Character field"
                    })
                    field_num += 1
                continue

            i += 1

        return fields

    def _is_packed_decimal(self, data: bytes) -> bool:
        """Check if data looks like packed decimal."""
        if len(data) < 2:
            return False
        # Last nibble should be sign (C, D, or F)
        last_nibble = data[-1] & 0x0F
        return last_nibble in [0x0C, 0x0D, 0x0F]

    def _detect_patterns(self, data: bytes) -> List[Dict]:
        """Detect common patterns in data."""
        patterns = []

        # Check for magic bytes
        if data[:4] == b'\x00\x00\x00\x00':
            patterns.append({
                "pattern": "null_header",
                "offset": 0,
                "description": "Null header bytes"
            })

        # Check for length prefix
        if len(data) >= 2:
            prefix_len = struct.unpack('>H', data[:2])[0]
            if prefix_len == len(data) or prefix_len == len(data) - 2:
                patterns.append({
                    "pattern": "length_prefix",
                    "offset": 0,
                    "description": "Big-endian length prefix"
                })

        # Check for repeating structures
        chunk_sizes = [10, 20, 40, 80, 100]
        for size in chunk_sizes:
            if len(data) >= size * 2:
                chunk1 = data[:size]
                chunk2 = data[size:size*2]
                similarity = sum(1 for a, b in zip(chunk1, chunk2) if a == b) / size
                if similarity > 0.7:
                    patterns.append({
                        "pattern": "repeating_structure",
                        "chunk_size": size,
                        "description": f"Repeating {size}-byte structures detected"
                    })
                    break

        return patterns

    def _generate_conversion_code(self, from_enc: str, to_enc: str) -> str:
        """Generate encoding conversion code."""
        return f'''
def convert_{from_enc.lower()}_to_{to_enc.lower()}(data: bytes) -> str:
    """Convert {from_enc} encoded data to {to_enc}."""
    import codecs
    # EBCDIC to ASCII mapping for IBM CP037
    return data.decode('cp037').encode('{to_enc.lower()}').decode('{to_enc.lower()}')

# Usage:
# modern_text = convert_{from_enc.lower()}_to_{to_enc.lower()}(legacy_data)
'''

    def _generate_parser_code(self, fields: List[Dict]) -> str:
        """Generate parser code for detected fields."""
        code_lines = [
            "from dataclasses import dataclass",
            "from typing import Optional",
            "import struct",
            "",
            "@dataclass",
            "class MainframeRecord:",
            '    """Auto-generated from mainframe protocol analysis."""',
            ""
        ]

        for field in fields:
            field_type = "str" if field["type"] == "character" else "float"
            code_lines.append(f"    {field['name'].lower()}: {field_type}")

        code_lines.extend([
            "",
            "    @classmethod",
            "    def from_bytes(cls, data: bytes) -> 'MainframeRecord':",
            '        """Parse mainframe record from bytes."""',
        ])

        for field in fields:
            if field["type"] == "packed_decimal":
                code_lines.append(
                    f"        {field['name'].lower()} = unpack_decimal(data[{field['offset']}:{field['offset'] + field['length']}])"
                )
            else:
                code_lines.append(
                    f"        {field['name'].lower()} = data[{field['offset']}:{field['offset'] + field['length']}].decode('cp037').strip()"
                )

        code_lines.append(f"        return cls({', '.join(f['name'].lower() for f in fields)})")

        return '\n'.join(code_lines)

    def _analyze_3270(self, data: bytes) -> Dict:
        return {"protocol": "3270", "description": "IBM 3270 terminal protocol"}

    def _analyze_ebcdic(self, data: bytes) -> Dict:
        return {"protocol": "EBCDIC", "description": "Extended Binary Coded Decimal"}

    def _analyze_cics(self, data: bytes) -> Dict:
        return {"protocol": "CICS", "description": "Customer Information Control System"}

    def _analyze_mq(self, data: bytes) -> Dict:
        return {"protocol": "MQ", "description": "IBM MQ Message"}

# ============================================================================
# Code Generator
# ============================================================================

class ModernCodeGenerator:
    """Generates modern code from legacy analysis."""

    def __init__(self):
        self.templates = {}

    def generate_python_model(self, cobol_analysis: Dict) -> str:
        """Generate Python dataclass from COBOL analysis."""
        working_storage = cobol_analysis.get("working_storage", {})
        variables = working_storage.get("variables", [])

        code_lines = [
            '"""',
            'Auto-generated Python models from COBOL analysis.',
            'Generated by QBITEL Legacy Modernization',
            '"""',
            '',
            'from dataclasses import dataclass, field',
            'from typing import Optional, List',
            'from decimal import Decimal',
            'from datetime import datetime',
            '',
        ]

        # Group variables by level
        level_01_vars = [v for v in variables if v["level"] == 1]

        for var in level_01_vars:
            class_name = self._to_python_class_name(var["name"])
            code_lines.extend([
                '@dataclass',
                f'class {class_name}:',
                f'    """Generated from COBOL record: {var["name"]}"""',
                ''
            ])

            # Find child fields (levels 05, 10, etc.)
            child_vars = [v for v in variables if v["level"] > 1 and v["level"] < 88]

            for child in child_vars[:20]:  # Limit for demo
                py_name = self._to_python_name(child["name"])
                py_type = self._cobol_to_python_type(child["type"], child.get("pic", ""))
                code_lines.append(f'    {py_name}: {py_type} = None')

            code_lines.append('')

        return '\n'.join(code_lines)

    def generate_fastapi_endpoints(self, cobol_analysis: Dict) -> str:
        """Generate FastAPI endpoints from COBOL procedures."""
        procedures = cobol_analysis.get("procedures", [])

        code_lines = [
            '"""',
            'Auto-generated FastAPI endpoints from COBOL procedures.',
            'Generated by QBITEL Legacy Modernization',
            '"""',
            '',
            'from fastapi import APIRouter, HTTPException',
            'from pydantic import BaseModel',
            'from typing import Optional, List',
            '',
            'router = APIRouter(prefix="/api/v1", tags=["modernized"])',
            '',
        ]

        for proc in procedures[:10]:  # Limit for demo
            endpoint_name = self._to_python_name(proc["name"]).replace('_', '-')
            func_name = self._to_python_name(proc["name"])

            code_lines.extend([
                f'@router.post("/{endpoint_name}")',
                f'async def {func_name}():',
                f'    """',
                f'    Modernized endpoint for COBOL paragraph: {proc["name"]}',
                f'    Original performs: {", ".join(proc.get("performs", [])[:5])}',
                f'    """',
                '    # TODO: Implement business logic',
                '    return {"status": "success", "message": "Endpoint implemented"}',
                '',
            ])

        return '\n'.join(code_lines)

    def generate_sql_schema(self, cobol_analysis: Dict) -> str:
        """Generate SQL schema from COBOL file definitions."""
        file_sections = cobol_analysis.get("file_sections", [])

        sql_lines = [
            '-- Auto-generated SQL schema from COBOL file definitions',
            '-- Generated by QBITEL Legacy Modernization',
            '',
        ]

        for file_def in file_sections:
            table_name = self._to_sql_table_name(file_def["name"])
            sql_lines.extend([
                f'CREATE TABLE {table_name} (',
                '    id SERIAL PRIMARY KEY,',
                '    -- Fields extracted from COBOL record definition',
                '    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,',
                '    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
                ');',
                '',
            ])

        return '\n'.join(sql_lines)

    def _to_python_class_name(self, cobol_name: str) -> str:
        """Convert COBOL name to Python class name."""
        parts = cobol_name.replace('-', '_').split('_')
        return ''.join(part.capitalize() for part in parts)

    def _to_python_name(self, cobol_name: str) -> str:
        """Convert COBOL name to Python variable name."""
        return cobol_name.replace('-', '_').lower()

    def _to_sql_table_name(self, cobol_name: str) -> str:
        """Convert COBOL name to SQL table name."""
        return cobol_name.replace('-', '_').lower()

    def _cobol_to_python_type(self, cobol_type: str, pic: str) -> str:
        """Convert COBOL type to Python type."""
        type_map = {
            "integer": "int",
            "decimal": "Decimal",
            "string": "str",
            "alpha": "str",
            "group": "dict",
            "unknown": "str"
        }
        return f'Optional[{type_map.get(cobol_type, "str")}]'

# ============================================================================
# Modernization Planner
# ============================================================================

class ModernizationPlanner:
    """Creates comprehensive modernization plans."""

    def __init__(self):
        self.code_generator = ModernCodeGenerator()

    def create_plan(
        self,
        system: LegacySystem,
        cobol_analysis: Dict,
        approach: ModernizationApproach,
        target_language: str = "python"
    ) -> ModernizationPlan:
        """Create a comprehensive modernization plan."""

        # Generate phases based on approach
        phases = self._generate_phases(approach, system)

        # Calculate effort
        effort = self._estimate_effort(system, approach, cobol_analysis)

        # Assess risk
        risk = self._assess_risk(system, approach)

        # Generate code artifacts
        generated_code = {
            "models": self.code_generator.generate_python_model(cobol_analysis),
            "api": self.code_generator.generate_fastapi_endpoints(cobol_analysis),
            "schema": self.code_generator.generate_sql_schema(cobol_analysis)
        }

        # Generate documentation
        documentation = self._generate_documentation(system, approach, phases)

        return ModernizationPlan(
            plan_id=str(uuid.uuid4())[:8],
            system_id=system.system_id,
            approach=approach,
            risk_level=risk,
            estimated_effort_days=effort,
            phases=phases,
            generated_code=generated_code,
            documentation=documentation,
            created_at=datetime.now()
        )

    def _generate_phases(self, approach: ModernizationApproach, system: LegacySystem) -> List[Dict]:
        """Generate project phases."""
        base_phases = [
            {
                "phase": 1,
                "name": "Discovery & Assessment",
                "description": "Analyze existing system, document dependencies, identify risks",
                "duration_weeks": 4,
                "deliverables": [
                    "System inventory",
                    "Dependency map",
                    "Risk assessment",
                    "Modernization roadmap"
                ]
            },
            {
                "phase": 2,
                "name": "Architecture Design",
                "description": "Design target architecture, define APIs, plan data migration",
                "duration_weeks": 6,
                "deliverables": [
                    "Target architecture document",
                    "API specifications",
                    "Data migration plan",
                    "Security requirements"
                ]
            }
        ]

        if approach == ModernizationApproach.REFACTOR:
            base_phases.extend([
                {
                    "phase": 3,
                    "name": "Code Transformation",
                    "description": "Transform COBOL to modern language with AI assistance",
                    "duration_weeks": 12,
                    "deliverables": [
                        "Transformed codebase",
                        "Unit tests",
                        "Integration tests",
                        "Code documentation"
                    ]
                },
                {
                    "phase": 4,
                    "name": "Testing & Validation",
                    "description": "Comprehensive testing and validation against original",
                    "duration_weeks": 8,
                    "deliverables": [
                        "Test results report",
                        "Performance benchmarks",
                        "Regression test suite",
                        "Validation sign-off"
                    ]
                },
                {
                    "phase": 5,
                    "name": "Deployment & Cutover",
                    "description": "Deploy to production, execute cutover plan",
                    "duration_weeks": 4,
                    "deliverables": [
                        "Production deployment",
                        "Runbook",
                        "Monitoring setup",
                        "Support documentation"
                    ]
                }
            ])
        elif approach == ModernizationApproach.REPLATFORM:
            base_phases.extend([
                {
                    "phase": 3,
                    "name": "Platform Migration",
                    "description": "Migrate to cloud platform with minimal code changes",
                    "duration_weeks": 8,
                    "deliverables": [
                        "Cloud infrastructure",
                        "Migrated application",
                        "Configuration files"
                    ]
                }
            ])

        return base_phases

    def _estimate_effort(
        self,
        system: LegacySystem,
        approach: ModernizationApproach,
        analysis: Dict
    ) -> int:
        """Estimate effort in person-days."""
        base_effort = system.lines_of_code / 500  # ~500 LOC per day

        # Adjust for approach
        approach_multiplier = {
            ModernizationApproach.REHOST: 0.3,
            ModernizationApproach.REPLATFORM: 0.5,
            ModernizationApproach.REFACTOR: 1.0,
            ModernizationApproach.REARCHITECT: 1.5,
            ModernizationApproach.REBUILD: 2.0,
            ModernizationApproach.REPLACE: 0.8
        }

        # Adjust for complexity
        complexity = analysis.get("procedures", [])
        complexity_factor = 1 + (len(complexity) / 100)

        # Adjust for dependencies
        dependency_factor = 1 + (len(system.dependencies) * 0.1)

        return int(
            base_effort *
            approach_multiplier.get(approach, 1.0) *
            complexity_factor *
            dependency_factor
        )

    def _assess_risk(self, system: LegacySystem, approach: ModernizationApproach) -> RiskLevel:
        """Assess modernization risk level."""
        risk_score = 0

        # System criticality
        if system.business_criticality == "critical":
            risk_score += 3
        elif system.business_criticality == "high":
            risk_score += 2

        # System age
        if system.age_years > 30:
            risk_score += 2
        elif system.age_years > 20:
            risk_score += 1

        # Code size
        if system.lines_of_code > 1_000_000:
            risk_score += 2
        elif system.lines_of_code > 500_000:
            risk_score += 1

        # Approach risk
        approach_risk = {
            ModernizationApproach.REHOST: 1,
            ModernizationApproach.REPLATFORM: 2,
            ModernizationApproach.REFACTOR: 3,
            ModernizationApproach.REARCHITECT: 4,
            ModernizationApproach.REBUILD: 4,
            ModernizationApproach.REPLACE: 3
        }
        risk_score += approach_risk.get(approach, 2)

        if risk_score >= 10:
            return RiskLevel.CRITICAL
        elif risk_score >= 7:
            return RiskLevel.HIGH
        elif risk_score >= 4:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _generate_documentation(
        self,
        system: LegacySystem,
        approach: ModernizationApproach,
        phases: List[Dict]
    ) -> str:
        """Generate modernization documentation."""
        return f"""
# Modernization Plan: {system.name}

## Executive Summary
This document outlines the modernization plan for {system.name},
a {system.age_years}-year-old {system.language} system running on {system.platform}.

## System Overview
- **System ID:** {system.system_id}
- **Lines of Code:** {system.lines_of_code:,}
- **Business Criticality:** {system.business_criticality}
- **Current Status:** {system.status.value}

## Dependencies
{chr(10).join(f'- {dep}' for dep in system.dependencies)}

## Modernization Approach: {approach.value.upper()}

### Rationale
The {approach.value} approach was selected based on:
- System criticality and risk tolerance
- Available timeline and resources
- Business requirements for continuity

## Project Phases

{chr(10).join(self._format_phase(p) for p in phases)}

## Risk Mitigation
- Comprehensive testing at each phase
- Parallel running during cutover
- Rollback procedures documented
- Monitoring and alerting in place

## Success Criteria
- 100% functional parity with original system
- Performance within 10% of original
- All regulatory requirements maintained
- Zero data loss during migration

---
*Generated by QBITEL Legacy Mainframe Modernization*
*Date: {datetime.now().strftime('%Y-%m-%d')}*
"""

    def _format_phase(self, phase: Dict) -> str:
        """Format a phase for documentation."""
        deliverables = '\n'.join(f'  - {d}' for d in phase.get('deliverables', []))
        return f"""
### Phase {phase['phase']}: {phase['name']}
**Duration:** {phase['duration_weeks']} weeks

{phase['description']}

**Deliverables:**
{deliverables}
"""

# ============================================================================
# Initialize Global Objects
# ============================================================================

mainframe = MainframeSimulator()
cobol_analyzer = COBOLAnalyzer()
protocol_analyzer = MainframeProtocolAnalyzer()
modernization_planner = ModernizationPlanner()

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Demo home page."""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>QBITEL Bridge - Legacy Mainframe Modernization Demo</title>
        <style>
            body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 0; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); min-height: 100vh; color: #fff; }
            .container { max-width: 1200px; margin: 0 auto; padding: 40px 20px; }
            h1 { font-size: 2.5em; margin-bottom: 10px; background: linear-gradient(90deg, #00d9ff, #00ff88); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            .subtitle { font-size: 1.2em; color: #8892b0; margin-bottom: 40px; }
            .card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 24px; }
            .card { background: rgba(255,255,255,0.05); border-radius: 16px; padding: 24px; border: 1px solid rgba(255,255,255,0.1); transition: all 0.3s; }
            .card:hover { transform: translateY(-5px); border-color: #00d9ff; box-shadow: 0 10px 40px rgba(0,217,255,0.2); }
            .card h3 { color: #00d9ff; margin-top: 0; font-size: 1.3em; }
            .card p { color: #8892b0; line-height: 1.6; }
            .endpoint { background: rgba(0,0,0,0.3); padding: 8px 12px; border-radius: 6px; font-family: monospace; margin: 8px 0; display: inline-block; }
            .badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; margin-right: 8px; }
            .badge-get { background: #10b981; color: #fff; }
            .badge-post { background: #3b82f6; color: #fff; }
            a { color: #00d9ff; text-decoration: none; }
            a:hover { text-decoration: underline; }
            .stats { display: flex; gap: 40px; margin: 30px 0; flex-wrap: wrap; }
            .stat { text-align: center; }
            .stat-value { font-size: 2em; font-weight: bold; color: #00ff88; }
            .stat-label { color: #8892b0; font-size: 0.9em; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>QBITEL</h1>
            <div class="subtitle">UC1: Legacy Mainframe Modernization Demo</div>

            <div class="stats">
                <div class="stat">
                    <div class="stat-value">3</div>
                    <div class="stat-label">Legacy Systems</div>
                </div>
                <div class="stat">
                    <div class="stat-value">4.5M</div>
                    <div class="stat-label">Lines of COBOL</div>
                </div>
                <div class="stat">
                    <div class="stat-value">38</div>
                    <div class="stat-label">Years Legacy</div>
                </div>
                <div class="stat">
                    <div class="stat-value">99.97%</div>
                    <div class="stat-label">Uptime</div>
                </div>
            </div>

            <div class="card-grid">
                <div class="card">
                    <h3>Legacy System Inventory</h3>
                    <p>View and manage legacy mainframe systems, including IBM z/OS COBOL applications.</p>
                    <span class="badge badge-get">GET</span>
                    <span class="endpoint">/api/systems</span>
                    <br><br>
                    <a href="/api/systems">View Systems →</a>
                </div>

                <div class="card">
                    <h3>COBOL Analyzer</h3>
                    <p>AI-powered COBOL code analysis with complexity scoring, pattern detection, and modernization recommendations.</p>
                    <span class="badge badge-get">GET</span>
                    <span class="endpoint">/api/analyze/cobol</span>
                    <br><br>
                    <a href="/api/analyze/cobol/list">Analyze COBOL Files →</a>
                </div>

                <div class="card">
                    <h3>Protocol Analyzer</h3>
                    <p>Reverse engineer mainframe binary protocols including EBCDIC, 3270, and proprietary formats.</p>
                    <span class="badge badge-post">POST</span>
                    <span class="endpoint">/api/analyze/protocol</span>
                    <br><br>
                    <a href="/docs#/default/analyze_protocol_api_analyze_protocol_post">Try Protocol Analysis →</a>
                </div>

                <div class="card">
                    <h3>Modernization Planner</h3>
                    <p>Generate comprehensive modernization plans with code generation, effort estimation, and risk assessment.</p>
                    <span class="badge badge-post">POST</span>
                    <span class="endpoint">/api/modernize</span>
                    <br><br>
                    <a href="/docs#/default/create_modernization_plan_api_modernize_post">Create Plan →</a>
                </div>

                <div class="card">
                    <h3>Code Generator</h3>
                    <p>Auto-generate modern Python/FastAPI code, SQL schemas, and API specifications from legacy analysis.</p>
                    <span class="badge badge-post">POST</span>
                    <span class="endpoint">/api/generate</span>
                    <br><br>
                    <a href="/docs#/default/generate_code_api_generate_post">Generate Code →</a>
                </div>

                <div class="card">
                    <h3>Interactive Demo</h3>
                    <p>Step-by-step walkthrough of the complete mainframe modernization process.</p>
                    <span class="badge badge-get">GET</span>
                    <span class="endpoint">/demo</span>
                    <br><br>
                    <a href="/demo">Launch Demo →</a>
                </div>
            </div>

            <div style="margin-top: 40px; text-align: center; color: #8892b0;">
                <p>API Documentation: <a href="/docs">Swagger UI</a> | <a href="/redoc">ReDoc</a></p>
                <p style="font-size: 0.9em;">Powered by QBITEL Engine</p>
            </div>
        </div>
    </body>
    </html>
    """)

@app.get("/api/systems")
async def get_systems():
    """Get all legacy systems."""
    systems = mainframe.get_all_systems()
    return {
        "systems": [asdict(s) for s in systems],
        "total": len(systems)
    }

@app.get("/api/systems/{system_id}")
async def get_system(system_id: str):
    """Get specific legacy system details."""
    system = mainframe.get_system(system_id)
    if not system:
        raise HTTPException(status_code=404, detail="System not found")
    return asdict(system)

@app.get("/api/analyze/cobol/list")
async def list_cobol_files():
    """List available COBOL files for analysis."""
    cobol_files = list(COBOL_DIR.glob("*.cbl"))
    return {
        "files": [
            {
                "name": f.name,
                "path": str(f),
                "size_bytes": f.stat().st_size,
                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
            }
            for f in cobol_files
        ],
        "total": len(cobol_files)
    }

@app.get("/api/analyze/cobol/{filename}")
async def analyze_cobol_file(filename: str):
    """Analyze a specific COBOL file."""
    file_path = COBOL_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="COBOL file not found")

    program = cobol_analyzer.analyze_cobol_file(str(file_path))
    return asdict(program)

@app.post("/api/analyze/protocol")
async def analyze_protocol(request: ProtocolAnalyzeRequest):
    """Analyze mainframe protocol data."""
    try:
        data = bytes.fromhex(request.protocol_data)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid hex data")

    analysis = protocol_analyzer.analyze_protocol(data, request.system_context)
    return analysis

@app.post("/api/modernize")
async def create_modernization_plan(request: ModernizeRequest):
    """Create a comprehensive modernization plan."""
    system = mainframe.get_system(request.system_id)
    if not system:
        raise HTTPException(status_code=404, detail="System not found")

    # Get COBOL analysis for the system
    cobol_files = list(COBOL_DIR.glob("*.cbl"))
    if cobol_files:
        cobol_analysis = cobol_analyzer.analyze_cobol_file(str(cobol_files[0])).analysis
    else:
        cobol_analysis = {}

    try:
        approach = ModernizationApproach(request.approach)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid modernization approach")

    plan = modernization_planner.create_plan(
        system=system,
        cobol_analysis=cobol_analysis,
        approach=approach,
        target_language=request.target_language
    )

    return asdict(plan)

@app.post("/api/generate")
async def generate_code(request: AnalyzeRequest):
    """Generate modern code from COBOL analysis."""
    file_path = COBOL_DIR / request.cobol_file
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="COBOL file not found")

    program = cobol_analyzer.analyze_cobol_file(str(file_path))
    generator = ModernCodeGenerator()

    return {
        "source_file": request.cobol_file,
        "generated_code": {
            "python_models": generator.generate_python_model(program.analysis),
            "fastapi_endpoints": generator.generate_fastapi_endpoints(program.analysis),
            "sql_schema": generator.generate_sql_schema(program.analysis)
        },
        "analysis_summary": {
            "lines_of_code": program.lines_of_code,
            "complexity_score": program.complexity_score,
            "procedures_count": len(program.analysis.get("procedures", [])),
            "modernization_opportunities": len(program.analysis.get("modernization_opportunities", []))
        }
    }

@app.get("/demo")
async def interactive_demo():
    """Interactive demo page."""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>QBITEL - Interactive Demo</title>
        <style>
            * { box-sizing: border-box; }
            body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 0; background: #0a0a0f; color: #fff; }
            .header { background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%); padding: 20px 40px; border-bottom: 1px solid rgba(255,255,255,0.1); }
            .header h1 { margin: 0; font-size: 1.5em; background: linear-gradient(90deg, #00d9ff, #00ff88); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            .main { display: flex; height: calc(100vh - 70px); }
            .sidebar { width: 300px; background: #111; border-right: 1px solid rgba(255,255,255,0.1); padding: 20px; overflow-y: auto; }
            .content { flex: 1; padding: 20px; overflow-y: auto; }
            .step { padding: 15px; margin-bottom: 10px; background: rgba(255,255,255,0.05); border-radius: 8px; cursor: pointer; border: 1px solid transparent; transition: all 0.2s; }
            .step:hover { border-color: #00d9ff; }
            .step.active { border-color: #00ff88; background: rgba(0,255,136,0.1); }
            .step h4 { margin: 0 0 5px 0; color: #00d9ff; }
            .step p { margin: 0; font-size: 0.9em; color: #8892b0; }
            .step-number { display: inline-block; width: 24px; height: 24px; background: #00d9ff; color: #000; border-radius: 50%; text-align: center; line-height: 24px; font-size: 0.8em; font-weight: bold; margin-right: 10px; }
            .panel { background: rgba(255,255,255,0.05); border-radius: 12px; padding: 24px; margin-bottom: 20px; }
            .panel h2 { margin-top: 0; color: #00d9ff; }
            pre { background: #000; padding: 15px; border-radius: 8px; overflow-x: auto; font-size: 0.85em; }
            code { font-family: 'Fira Code', 'Consolas', monospace; }
            .btn { display: inline-block; padding: 10px 20px; background: linear-gradient(90deg, #00d9ff, #00ff88); color: #000; border: none; border-radius: 8px; font-weight: bold; cursor: pointer; transition: all 0.2s; }
            .btn:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(0,217,255,0.3); }
            .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
            .output { background: #000; border-radius: 8px; padding: 15px; margin-top: 15px; max-height: 400px; overflow-y: auto; }
            .loading { display: inline-block; width: 20px; height: 20px; border: 2px solid rgba(255,255,255,0.3); border-top-color: #00d9ff; border-radius: 50%; animation: spin 1s linear infinite; margin-right: 10px; }
            @keyframes spin { to { transform: rotate(360deg); } }
            .metric { display: inline-block; padding: 8px 16px; background: rgba(0,217,255,0.2); border-radius: 20px; margin: 4px; }
            .metric-label { color: #8892b0; font-size: 0.85em; }
            .metric-value { color: #00ff88; font-weight: bold; }
            .code-tabs { display: flex; gap: 10px; margin-bottom: 15px; }
            .code-tab { padding: 8px 16px; background: rgba(255,255,255,0.05); border-radius: 8px 8px 0 0; cursor: pointer; border: 1px solid transparent; border-bottom: none; }
            .code-tab.active { background: #000; border-color: rgba(255,255,255,0.2); }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>QBITEL Bridge - Legacy Mainframe Modernization Demo</h1>
        </div>
        <div class="main">
            <div class="sidebar">
                <h3 style="color: #8892b0; margin-top: 0;">Demo Steps</h3>
                <div class="step active" onclick="showStep(1)">
                    <span class="step-number">1</span>
                    <h4>System Discovery</h4>
                    <p>Identify legacy mainframe systems</p>
                </div>
                <div class="step" onclick="showStep(2)">
                    <span class="step-number">2</span>
                    <h4>COBOL Analysis</h4>
                    <p>Analyze COBOL source code</p>
                </div>
                <div class="step" onclick="showStep(3)">
                    <span class="step-number">3</span>
                    <h4>Protocol Analysis</h4>
                    <p>Reverse engineer data formats</p>
                </div>
                <div class="step" onclick="showStep(4)">
                    <span class="step-number">4</span>
                    <h4>Code Generation</h4>
                    <p>Generate modern code</p>
                </div>
                <div class="step" onclick="showStep(5)">
                    <span class="step-number">5</span>
                    <h4>Modernization Plan</h4>
                    <p>Create comprehensive plan</p>
                </div>
            </div>
            <div class="content" id="content">
                <!-- Content loaded dynamically -->
            </div>
        </div>

        <script>
            let currentStep = 1;

            function showStep(step) {
                currentStep = step;
                document.querySelectorAll('.step').forEach((el, i) => {
                    el.classList.toggle('active', i + 1 === step);
                });
                loadStepContent(step);
            }

            function loadStepContent(step) {
                const content = document.getElementById('content');

                switch(step) {
                    case 1:
                        content.innerHTML = `
                            <div class="panel">
                                <h2>Step 1: Legacy System Discovery</h2>
                                <p>Discover and inventory legacy mainframe systems in your environment.</p>
                                <button class="btn" onclick="discoverSystems()">Discover Systems</button>
                                <div id="systems-output" class="output" style="display: none;"></div>
                            </div>
                        `;
                        break;
                    case 2:
                        content.innerHTML = `
                            <div class="panel">
                                <h2>Step 2: COBOL Code Analysis</h2>
                                <p>AI-powered analysis of COBOL source code to understand structure, complexity, and modernization opportunities.</p>
                                <button class="btn" onclick="analyzeCOBOL()">Analyze COBOL Files</button>
                                <div id="cobol-output" class="output" style="display: none;"></div>
                            </div>
                        `;
                        break;
                    case 3:
                        content.innerHTML = `
                            <div class="panel">
                                <h2>Step 3: Protocol Analysis</h2>
                                <p>Reverse engineer mainframe binary protocols and data formats.</p>
                                <p style="color: #8892b0; font-size: 0.9em;">Sample data: EBCDIC encoded customer record</p>
                                <button class="btn" onclick="analyzeProtocol()">Analyze Protocol</button>
                                <div id="protocol-output" class="output" style="display: none;"></div>
                            </div>
                        `;
                        break;
                    case 4:
                        content.innerHTML = `
                            <div class="panel">
                                <h2>Step 4: Modern Code Generation</h2>
                                <p>Automatically generate modern Python/FastAPI code from COBOL analysis.</p>
                                <button class="btn" onclick="generateCode()">Generate Code</button>
                                <div id="code-output" style="display: none; margin-top: 15px;">
                                    <div class="code-tabs">
                                        <div class="code-tab active" onclick="showCodeTab('models')">Python Models</div>
                                        <div class="code-tab" onclick="showCodeTab('api')">FastAPI Endpoints</div>
                                        <div class="code-tab" onclick="showCodeTab('sql')">SQL Schema</div>
                                    </div>
                                    <pre><code id="code-content"></code></pre>
                                </div>
                            </div>
                        `;
                        break;
                    case 5:
                        content.innerHTML = `
                            <div class="panel">
                                <h2>Step 5: Modernization Plan</h2>
                                <p>Generate a comprehensive modernization plan with phases, estimates, and documentation.</p>
                                <select id="approach-select" style="padding: 10px; margin-right: 10px; background: #222; color: #fff; border: 1px solid #444; border-radius: 8px;">
                                    <option value="refactor">Refactor (Transform Code)</option>
                                    <option value="replatform">Replatform (Move to Cloud)</option>
                                    <option value="rearchitect">Rearchitect (Redesign)</option>
                                    <option value="rebuild">Rebuild (From Scratch)</option>
                                </select>
                                <button class="btn" onclick="createPlan()">Create Plan</button>
                                <div id="plan-output" class="output" style="display: none;"></div>
                            </div>
                        `;
                        break;
                }
            }

            async function discoverSystems() {
                const output = document.getElementById('systems-output');
                output.style.display = 'block';
                output.innerHTML = '<span class="loading"></span> Discovering legacy systems...';

                try {
                    const response = await fetch('/api/systems');
                    const data = await response.json();

                    let html = '<h3 style="color: #00ff88;">Discovered ' + data.total + ' Legacy Systems</h3>';

                    data.systems.forEach(sys => {
                        html += `
                            <div style="background: rgba(255,255,255,0.05); padding: 15px; margin: 10px 0; border-radius: 8px;">
                                <h4 style="color: #00d9ff; margin: 0 0 10px 0;">${sys.name} (${sys.system_id})</h4>
                                <div class="metric"><span class="metric-label">Platform:</span> <span class="metric-value">${sys.platform}</span></div>
                                <div class="metric"><span class="metric-label">Language:</span> <span class="metric-value">${sys.language}</span></div>
                                <div class="metric"><span class="metric-label">Lines:</span> <span class="metric-value">${(sys.lines_of_code/1000000).toFixed(1)}M</span></div>
                                <div class="metric"><span class="metric-label">Age:</span> <span class="metric-value">${sys.age_years} years</span></div>
                                <div class="metric"><span class="metric-label">Status:</span> <span class="metric-value">${sys.status}</span></div>
                            </div>
                        `;
                    });

                    output.innerHTML = html;
                } catch (error) {
                    output.innerHTML = '<span style="color: #ff4444;">Error: ' + error.message + '</span>';
                }
            }

            async function analyzeCOBOL() {
                const output = document.getElementById('cobol-output');
                output.style.display = 'block';
                output.innerHTML = '<span class="loading"></span> Analyzing COBOL files...';

                try {
                    const listResponse = await fetch('/api/analyze/cobol/list');
                    const files = await listResponse.json();

                    if (files.files.length === 0) {
                        output.innerHTML = '<span style="color: #ffaa00;">No COBOL files found in demo directory.</span>';
                        return;
                    }

                    const analysisResponse = await fetch('/api/analyze/cobol/' + files.files[0].name);
                    const analysis = await analysisResponse.json();

                    let html = `
                        <h3 style="color: #00ff88;">Analysis: ${analysis.name}</h3>
                        <div class="metric"><span class="metric-label">Lines of Code:</span> <span class="metric-value">${analysis.lines_of_code}</span></div>
                        <div class="metric"><span class="metric-label">Complexity:</span> <span class="metric-value">${analysis.complexity_score}</span></div>
                        <div class="metric"><span class="metric-label">Data Divisions:</span> <span class="metric-value">${analysis.data_divisions}</span></div>
                        <div class="metric"><span class="metric-label">Procedure Divisions:</span> <span class="metric-value">${analysis.procedure_divisions}</span></div>

                        <h4 style="color: #00d9ff; margin-top: 20px;">Working Storage Variables</h4>
                        <p style="color: #8892b0;">${analysis.analysis.working_storage.variable_count} variables detected</p>

                        <h4 style="color: #00d9ff; margin-top: 20px;">Legacy Patterns Detected</h4>
                    `;

                    analysis.analysis.legacy_patterns.forEach(pattern => {
                        html += `<div style="padding: 10px; margin: 5px 0; background: rgba(255,${pattern.severity === 'critical' ? '68,68' : pattern.severity === 'high' ? '170,0' : '255,136'},0.1); border-radius: 6px;">
                            <strong>${pattern.pattern}</strong> (${pattern.severity})<br>
                            <span style="color: #8892b0;">${pattern.description}</span>
                        </div>`;
                    });

                    html += '<h4 style="color: #00d9ff; margin-top: 20px;">Modernization Opportunities</h4>';

                    analysis.analysis.modernization_opportunities.forEach(opp => {
                        html += `<div style="padding: 10px; margin: 5px 0; background: rgba(0,255,136,0.1); border-radius: 6px;">
                            <strong>${opp.area}</strong><br>
                            <span style="color: #8892b0;">Current: ${opp.current}<br>Modern: ${opp.modern}</span>
                        </div>`;
                    });

                    output.innerHTML = html;
                } catch (error) {
                    output.innerHTML = '<span style="color: #ff4444;">Error: ' + error.message + '</span>';
                }
            }

            async function analyzeProtocol() {
                const output = document.getElementById('protocol-output');
                output.style.display = 'block';
                output.innerHTML = '<span class="loading"></span> Analyzing protocol data...';

                // Sample EBCDIC data (simulated customer record)
                const sampleData = 'd1d6c8d540e2d4c9e3c840404040404040404040f1f2f3f4f5f6f7f8f9f0c1c3c3d6e4d5e340';

                try {
                    const response = await fetch('/api/analyze/protocol', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            protocol_data: sampleData,
                            system_context: 'IBM z/OS Customer Master File'
                        })
                    });
                    const analysis = await response.json();

                    let html = `
                        <h3 style="color: #00ff88;">Protocol Analysis Results</h3>
                        <div class="metric"><span class="metric-label">Encoding:</span> <span class="metric-value">${analysis.encoding}</span></div>
                        <div class="metric"><span class="metric-label">Structure:</span> <span class="metric-value">${analysis.structure.type}</span></div>
                        <div class="metric"><span class="metric-label">Length:</span> <span class="metric-value">${analysis.raw_length} bytes</span></div>

                        <h4 style="color: #00d9ff; margin-top: 20px;">Detected Fields</h4>
                    `;

                    analysis.fields.forEach(field => {
                        html += `<div style="padding: 8px; margin: 4px 0; background: rgba(255,255,255,0.05); border-radius: 4px; font-family: monospace;">
                            <span style="color: #00d9ff;">${field.name}</span> @ offset ${field.offset}, ${field.length} bytes (${field.type})
                        </div>`;
                    });

                    html += '<h4 style="color: #00d9ff; margin-top: 20px;">Recommendations</h4>';

                    analysis.recommendations.forEach(rec => {
                        html += `<div style="padding: 10px; margin: 5px 0; background: rgba(0,217,255,0.1); border-radius: 6px;">
                            <strong>${rec.issue}</strong><br>
                            <span style="color: #00ff88;">${rec.solution}</span>
                        </div>`;
                    });

                    output.innerHTML = html;
                } catch (error) {
                    output.innerHTML = '<span style="color: #ff4444;">Error: ' + error.message + '</span>';
                }
            }

            let generatedCode = {};

            async function generateCode() {
                const output = document.getElementById('code-output');
                output.style.display = 'block';
                document.getElementById('code-content').textContent = 'Generating modern code...';

                try {
                    const listResponse = await fetch('/api/analyze/cobol/list');
                    const files = await listResponse.json();

                    if (files.files.length === 0) {
                        document.getElementById('code-content').textContent = 'No COBOL files found.';
                        return;
                    }

                    const response = await fetch('/api/generate', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({cobol_file: files.files[0].name})
                    });
                    const data = await response.json();

                    generatedCode = data.generated_code;
                    showCodeTab('models');
                } catch (error) {
                    document.getElementById('code-content').textContent = 'Error: ' + error.message;
                }
            }

            function showCodeTab(tab) {
                document.querySelectorAll('.code-tab').forEach(el => el.classList.remove('active'));
                event.target.classList.add('active');

                const content = document.getElementById('code-content');
                switch(tab) {
                    case 'models':
                        content.textContent = generatedCode.python_models || 'No code generated';
                        break;
                    case 'api':
                        content.textContent = generatedCode.fastapi_endpoints || 'No code generated';
                        break;
                    case 'sql':
                        content.textContent = generatedCode.sql_schema || 'No code generated';
                        break;
                }
            }

            async function createPlan() {
                const output = document.getElementById('plan-output');
                const approach = document.getElementById('approach-select').value;
                output.style.display = 'block';
                output.innerHTML = '<span class="loading"></span> Creating modernization plan...';

                try {
                    const response = await fetch('/api/modernize', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            system_id: 'SYS001',
                            approach: approach,
                            target_language: 'python',
                            target_framework: 'fastapi'
                        })
                    });
                    const plan = await response.json();

                    let html = `
                        <h3 style="color: #00ff88;">Modernization Plan Generated</h3>
                        <div class="metric"><span class="metric-label">Plan ID:</span> <span class="metric-value">${plan.plan_id}</span></div>
                        <div class="metric"><span class="metric-label">Approach:</span> <span class="metric-value">${plan.approach.toUpperCase()}</span></div>
                        <div class="metric"><span class="metric-label">Risk Level:</span> <span class="metric-value">${plan.risk_level.toUpperCase()}</span></div>
                        <div class="metric"><span class="metric-label">Estimated Effort:</span> <span class="metric-value">${plan.estimated_effort_days} days</span></div>

                        <h4 style="color: #00d9ff; margin-top: 20px;">Project Phases</h4>
                    `;

                    plan.phases.forEach(phase => {
                        html += `<div style="padding: 15px; margin: 10px 0; background: rgba(255,255,255,0.05); border-radius: 8px; border-left: 3px solid #00d9ff;">
                            <strong>Phase ${phase.phase}: ${phase.name}</strong> (${phase.duration_weeks} weeks)<br>
                            <span style="color: #8892b0;">${phase.description}</span>
                        </div>`;
                    });

                    html += `<h4 style="color: #00d9ff; margin-top: 20px;">Documentation Preview</h4>
                        <pre style="max-height: 300px; overflow-y: auto; font-size: 0.8em;">${plan.documentation.substring(0, 2000)}...</pre>`;

                    output.innerHTML = html;
                } catch (error) {
                    output.innerHTML = '<span style="color: #ff4444;">Error: ' + error.message + '</span>';
                }
            }

            // Initialize first step
            showStep(1);
        </script>
    </body>
    </html>
    """)

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "UC1_Legacy_Mainframe_Modernization",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
