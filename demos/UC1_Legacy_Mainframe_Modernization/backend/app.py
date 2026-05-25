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

@app.get("/presentation")
async def presentation_mode():
    """Full-screen presentation mode for 15-minute demo."""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>QBITEL Bridge - Mainframe Modernization Demo</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

            * { box-sizing: border-box; margin: 0; padding: 0; }
            html, body { width: 100%; height: 100%; overflow: hidden; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; background: #08090d; color: #e2e8f0; }

            /* Slide container */
            .slides { width: 100%; height: 100%; position: relative; }
            .slide { position: absolute; top: 0; left: 0; width: 100%; height: 100%; opacity: 0; pointer-events: none; transition: opacity 0.6s ease; display: flex; flex-direction: column; justify-content: center; align-items: center; padding: 60px 80px; }
            .slide.active { opacity: 1; pointer-events: all; }

            /* Backgrounds */
            .bg-hero { background: radial-gradient(ellipse at 20% 50%, rgba(0,217,255,0.15) 0%, transparent 60%), radial-gradient(ellipse at 80% 20%, rgba(0,255,136,0.1) 0%, transparent 50%), linear-gradient(135deg, #0a0b10 0%, #111827 50%, #0a0b10 100%); }
            .bg-dark { background: linear-gradient(180deg, #0a0b10 0%, #111827 100%); }
            .bg-gradient { background: radial-gradient(ellipse at 50% 0%, rgba(0,217,255,0.08) 0%, transparent 60%), linear-gradient(180deg, #0d0e14 0%, #111827 100%); }
            .bg-demo { background: #0a0b10; }

            /* Typography */
            h1 { font-size: 4em; font-weight: 900; line-height: 1.1; letter-spacing: -0.03em; }
            h2 { font-size: 2.8em; font-weight: 800; line-height: 1.15; letter-spacing: -0.02em; margin-bottom: 24px; }
            h3 { font-size: 1.6em; font-weight: 700; margin-bottom: 16px; }
            .gradient-text { background: linear-gradient(135deg, #00d9ff 0%, #00ff88 50%, #00d9ff 100%); background-size: 200% auto; -webkit-background-clip: text; -webkit-text-fill-color: transparent; animation: shimmer 3s linear infinite; }
            @keyframes shimmer { to { background-position: 200% center; } }
            .subtitle { font-size: 1.5em; color: #94a3b8; font-weight: 400; max-width: 800px; line-height: 1.5; }
            .label { font-size: 0.85em; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase; color: #00d9ff; margin-bottom: 16px; }

            /* Navigation */
            .nav-bar { position: fixed; bottom: 0; left: 0; right: 0; height: 56px; background: rgba(10,11,16,0.95); backdrop-filter: blur(12px); border-top: 1px solid rgba(255,255,255,0.06); display: flex; align-items: center; justify-content: space-between; padding: 0 32px; z-index: 100; }
            .nav-btn { padding: 8px 20px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.15); background: transparent; color: #e2e8f0; font-size: 0.9em; font-weight: 500; cursor: pointer; transition: all 0.2s; font-family: inherit; }
            .nav-btn:hover { background: rgba(255,255,255,0.08); border-color: #00d9ff; }
            .nav-btn.primary { background: linear-gradient(135deg, #00d9ff, #00ff88); color: #000; border: none; font-weight: 700; }
            .nav-btn.primary:hover { transform: translateY(-1px); box-shadow: 0 4px 20px rgba(0,217,255,0.3); }
            .progress-bar { flex: 1; margin: 0 24px; height: 3px; background: rgba(255,255,255,0.08); border-radius: 2px; overflow: hidden; }
            .progress-fill { height: 100%; background: linear-gradient(90deg, #00d9ff, #00ff88); transition: width 0.4s ease; border-radius: 2px; }
            .slide-counter { font-size: 0.85em; color: #64748b; font-weight: 500; min-width: 60px; text-align: right; }
            .timer { font-size: 0.85em; color: #64748b; font-family: 'JetBrains Mono', monospace; min-width: 55px; }

            /* Crisis cards */
            .crisis-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 28px; margin-top: 40px; max-width: 1100px; }
            .crisis-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 32px; transition: all 0.3s; }
            .crisis-card:hover { border-color: rgba(0,217,255,0.3); transform: translateY(-4px); }
            .crisis-icon { font-size: 2.5em; margin-bottom: 16px; }
            .crisis-card h3 { color: #f8fafc; font-size: 1.25em; }
            .crisis-card p { color: #94a3b8; font-size: 0.95em; line-height: 1.6; }
            .crisis-stat { font-size: 2em; font-weight: 800; color: #ff4444; margin: 8px 0; }

            /* Architecture diagram */
            .arch-container { display: flex; flex-direction: column; gap: 12px; max-width: 900px; width: 100%; margin-top: 32px; }
            .arch-layer { display: flex; align-items: center; padding: 20px 28px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08); transition: all 0.3s; }
            .arch-layer:hover { border-color: rgba(0,217,255,0.3); }
            .arch-layer-name { font-weight: 700; font-size: 1.1em; min-width: 200px; }
            .arch-layer-tech { color: #94a3b8; font-size: 0.9em; }
            .arch-layer-badge { padding: 4px 12px; border-radius: 20px; font-size: 0.75em; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-right: 16px; }
            .badge-react { background: rgba(97,218,251,0.15); color: #61dafb; }
            .badge-go { background: rgba(0,173,216,0.15); color: #00add8; }
            .badge-python { background: rgba(255,212,59,0.15); color: #ffd43b; }
            .badge-rust { background: rgba(222,165,132,0.15); color: #dea584; }

            /* Journey steps */
            .journey-grid { display: flex; gap: 16px; margin-top: 36px; max-width: 1100px; }
            .journey-step { flex: 1; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 24px 20px; text-align: center; position: relative; }
            .journey-step::after { content: ''; position: absolute; right: -12px; top: 50%; transform: translateY(-50%); width: 0; height: 0; border-top: 8px solid transparent; border-bottom: 8px solid transparent; border-left: 8px solid #00d9ff; }
            .journey-step:last-child::after { display: none; }
            .journey-num { width: 36px; height: 36px; border-radius: 50%; background: linear-gradient(135deg, #00d9ff, #00ff88); color: #000; font-weight: 800; font-size: 0.9em; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 12px; }
            .journey-step h4 { color: #f8fafc; font-size: 1em; margin-bottom: 6px; }
            .journey-step p { color: #94a3b8; font-size: 0.8em; line-height: 1.4; }

            /* Demo panel */
            .demo-layout { display: flex; width: 100%; height: calc(100% - 56px); }
            .demo-sidebar { width: 260px; background: rgba(255,255,255,0.02); border-right: 1px solid rgba(255,255,255,0.06); padding: 24px 16px; overflow-y: auto; }
            .demo-main { flex: 1; padding: 32px 40px; overflow-y: auto; }
            .demo-step { padding: 14px 16px; border-radius: 10px; margin-bottom: 8px; cursor: pointer; transition: all 0.2s; border: 1px solid transparent; }
            .demo-step:hover { background: rgba(255,255,255,0.04); }
            .demo-step.active { background: rgba(0,217,255,0.08); border-color: rgba(0,217,255,0.3); }
            .demo-step.done { opacity: 0.6; }
            .demo-step.done::before { content: ''; display: inline-block; width: 8px; height: 8px; background: #00ff88; border-radius: 50%; margin-right: 8px; }
            .demo-step-num { display: inline-flex; align-items: center; justify-content: center; width: 24px; height: 24px; border-radius: 50%; background: rgba(0,217,255,0.2); color: #00d9ff; font-size: 0.75em; font-weight: 700; margin-right: 10px; }
            .demo-step h4 { font-size: 0.95em; color: #f8fafc; display: inline; }
            .demo-step p { font-size: 0.8em; color: #64748b; margin-top: 4px; margin-left: 34px; }

            /* Demo content panels */
            .demo-panel { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; padding: 28px; margin-bottom: 20px; }
            .demo-panel h2 { font-size: 1.5em; margin-bottom: 8px; color: #f8fafc; }
            .demo-panel .desc { color: #94a3b8; font-size: 0.95em; margin-bottom: 20px; }
            .demo-btn { display: inline-flex; align-items: center; gap: 8px; padding: 12px 28px; background: linear-gradient(135deg, #00d9ff, #00ff88); color: #000; border: none; border-radius: 10px; font-weight: 700; font-size: 0.95em; cursor: pointer; transition: all 0.2s; font-family: inherit; }
            .demo-btn:hover { transform: translateY(-2px); box-shadow: 0 6px 24px rgba(0,217,255,0.3); }
            .demo-btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; box-shadow: none; }
            .demo-btn svg { width: 18px; height: 18px; }
            .demo-output { background: rgba(0,0,0,0.4); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 20px; margin-top: 16px; max-height: 420px; overflow-y: auto; display: none; }
            .demo-output.visible { display: block; animation: fadeUp 0.4s ease; }
            @keyframes fadeUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

            /* System card */
            .sys-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 20px; margin: 10px 0; }
            .sys-card h4 { color: #00d9ff; margin-bottom: 10px; }
            .sys-metrics { display: flex; flex-wrap: wrap; gap: 8px; }
            .sys-metric { padding: 6px 14px; border-radius: 20px; background: rgba(0,217,255,0.08); font-size: 0.85em; }
            .sys-metric .label { color: #94a3b8; font-size: 0.85em; letter-spacing: 0; text-transform: none; margin: 0; }
            .sys-metric .value { color: #00ff88; font-weight: 600; }

            /* Pattern badges */
            .pattern { padding: 10px 14px; margin: 5px 0; border-radius: 8px; font-size: 0.9em; }
            .pattern.critical { background: rgba(255,68,68,0.1); border-left: 3px solid #ff4444; }
            .pattern.high { background: rgba(255,170,0,0.1); border-left: 3px solid #ffaa00; }
            .pattern.warning { background: rgba(255,255,0,0.08); border-left: 3px solid #ffd700; }
            .pattern.opportunity { background: rgba(0,255,136,0.08); border-left: 3px solid #00ff88; }

            /* Code display */
            .code-tabs { display: flex; gap: 4px; margin-bottom: -1px; position: relative; z-index: 1; }
            .code-tab { padding: 10px 20px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-bottom: none; border-radius: 8px 8px 0 0; cursor: pointer; font-size: 0.85em; font-weight: 600; color: #64748b; transition: all 0.2s; }
            .code-tab.active { background: rgba(0,0,0,0.5); color: #00d9ff; border-color: rgba(0,217,255,0.2); }
            pre.code-block { background: rgba(0,0,0,0.5); border: 1px solid rgba(255,255,255,0.06); border-radius: 0 10px 10px 10px; padding: 20px; overflow-x: auto; font-family: 'JetBrains Mono', monospace; font-size: 0.82em; line-height: 1.6; }

            /* Phase timeline */
            .phase { padding: 16px 20px; margin: 8px 0; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; border-left: 3px solid #00d9ff; display: flex; align-items: flex-start; gap: 16px; }
            .phase-num { min-width: 32px; height: 32px; border-radius: 50%; background: rgba(0,217,255,0.15); color: #00d9ff; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.85em; }
            .phase-info h4 { color: #f8fafc; font-size: 0.95em; }
            .phase-info p { color: #94a3b8; font-size: 0.85em; margin-top: 2px; }
            .phase-info .duration { color: #00ff88; font-weight: 600; }

            /* Impact metrics */
            .impact-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; max-width: 1000px; margin-top: 36px; }
            .impact-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 28px; text-align: center; }
            .impact-card .before { font-size: 1.1em; color: #ff4444; text-decoration: line-through; opacity: 0.7; }
            .impact-card .after { font-size: 2.2em; font-weight: 800; background: linear-gradient(135deg, #00d9ff, #00ff88); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            .impact-card .metric-name { color: #94a3b8; font-size: 0.9em; margin-top: 8px; }

            /* CTA */
            .cta-box { max-width: 700px; text-align: center; }
            .cta-box h2 { font-size: 2.5em; margin-bottom: 16px; }
            .cta-features { display: flex; gap: 16px; flex-wrap: wrap; justify-content: center; margin: 28px 0; }
            .cta-feature { padding: 8px 18px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 24px; font-size: 0.9em; color: #94a3b8; }
            .cta-btn { display: inline-block; padding: 16px 48px; background: linear-gradient(135deg, #00d9ff, #00ff88); color: #000; border-radius: 12px; font-weight: 800; font-size: 1.2em; text-decoration: none; transition: all 0.3s; cursor: pointer; border: none; font-family: inherit; }
            .cta-btn:hover { transform: translateY(-3px); box-shadow: 0 8px 32px rgba(0,217,255,0.4); }

            /* Loader */
            .loader { display: inline-block; width: 18px; height: 18px; border: 2px solid rgba(0,217,255,0.3); border-top-color: #00d9ff; border-radius: 50%; animation: spin 0.8s linear infinite; margin-right: 8px; vertical-align: middle; }
            @keyframes spin { to { transform: rotate(360deg); } }

            /* Keyboard hints */
            .key-hints { position: fixed; top: 16px; right: 24px; z-index: 101; display: flex; gap: 8px; opacity: 0.4; transition: opacity 0.3s; }
            .key-hints:hover { opacity: 1; }
            .key-hint { padding: 4px 10px; background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1); border-radius: 6px; font-size: 0.75em; color: #64748b; font-family: 'JetBrains Mono', monospace; }
        </style>
    </head>
    <body>
        <div class="slides" id="slides">

            <!-- SLIDE 0: Title -->
            <div class="slide bg-hero active" data-time="0:00">
                <div class="label">UC1 &mdash; Legacy Mainframe Modernization</div>
                <h1><span class="gradient-text">QBITEL Bridge</span></h1>
                <p class="subtitle" style="margin-top:20px;">AI-Powered Quantum-Safe Security for Legacy Systems</p>
                <div style="margin-top:48px; display:flex; gap:48px; color:#64748b; font-size:0.95em;">
                    <div><strong style="color:#00ff88; font-size:1.8em; display:block;">$3T</strong>Daily COBOL Transactions</div>
                    <div><strong style="color:#00ff88; font-size:1.8em; display:block;">60%</strong>Fortune 500 on Legacy</div>
                    <div><strong style="color:#00ff88; font-size:1.8em; display:block;">38yr</strong>Average System Age</div>
                </div>
            </div>

            <!-- SLIDE 1: The Problem -->
            <div class="slide bg-gradient" data-time="0:30">
                <div class="label">The Problem</div>
                <h2>Three Converging Crises</h2>
                <div class="crisis-grid">
                    <div class="crisis-card">
                        <div class="crisis-icon">&#x1f4dc;</div>
                        <h3>Legacy Crisis</h3>
                        <div class="crisis-stat">$2-10M</div>
                        <p>Cost to reverse-engineer ONE system. Original developers retired. No documentation. 6-12 months timeline.</p>
                    </div>
                    <div class="crisis-card">
                        <div class="crisis-icon">&#x269b;&#xfe0f;</div>
                        <h3>Quantum Threat</h3>
                        <div class="crisis-stat">5-10 yrs</div>
                        <p>Until RSA/ECC breaks. Nation-states harvesting encrypted data TODAY for future decryption.</p>
                    </div>
                    <div class="crisis-card">
                        <div class="crisis-icon">&#x23f1;&#xfe0f;</div>
                        <h3>Speed Gap</h3>
                        <div class="crisis-stat">65 min</div>
                        <p>Average SOC response time. Machine-speed attacks happen in seconds. Humans can't keep up.</p>
                    </div>
                </div>
            </div>

            <!-- SLIDE 2: Solution Overview -->
            <div class="slide bg-gradient" data-time="2:30">
                <div class="label">The Solution</div>
                <h2>Five-Stage <span class="gradient-text">Modernization Journey</span></h2>
                <div class="journey-grid">
                    <div class="journey-step">
                        <div class="journey-num">1</div>
                        <h4>Discover</h4>
                        <p>AI learns unknown protocols from raw traffic in 2-4 hours</p>
                    </div>
                    <div class="journey-step">
                        <div class="journey-num">2</div>
                        <h4>Protect</h4>
                        <p>NIST Level 5 PQC wrapping &mdash; zero code changes</p>
                    </div>
                    <div class="journey-step">
                        <div class="journey-num">3</div>
                        <h4>Translate</h4>
                        <p>Auto-generate REST APIs + SDKs in 6 languages</p>
                    </div>
                    <div class="journey-step">
                        <div class="journey-num">4</div>
                        <h4>Comply</h4>
                        <p>9 frameworks automated in under 10 minutes</p>
                    </div>
                    <div class="journey-step">
                        <div class="journey-num">5</div>
                        <h4>Operate</h4>
                        <p>78% autonomous response, &lt;1s decision time</p>
                    </div>
                </div>
            </div>

            <!-- SLIDE 3: Architecture -->
            <div class="slide bg-gradient" data-time="3:30">
                <div class="label">Platform Architecture</div>
                <h2>Four-Layer <span class="gradient-text">Polyglot Design</span></h2>
                <div class="arch-container">
                    <div class="arch-layer" style="background: rgba(97,218,251,0.04);">
                        <span class="arch-layer-badge badge-react">React/TS</span>
                        <span class="arch-layer-name">UI Console</span>
                        <span class="arch-layer-tech">Admin Dashboard &bull; Protocol Copilot &bull; Marketplace</span>
                    </div>
                    <div class="arch-layer" style="background: rgba(0,173,216,0.04);">
                        <span class="arch-layer-badge badge-go">Go</span>
                        <span class="arch-layer-name">Control Plane</span>
                        <span class="arch-layer-tech">Service Orchestration &bull; OPA Policies &bull; Vault Secrets &bull; gRPC</span>
                    </div>
                    <div class="arch-layer" style="background: rgba(255,212,59,0.04);">
                        <span class="arch-layer-badge badge-python">Python</span>
                        <span class="arch-layer-name">AI Engine</span>
                        <span class="arch-layer-tech">Protocol Discovery &bull; Multi-Agent System &bull; LLM &bull; RAG &bull; Compliance</span>
                    </div>
                    <div class="arch-layer" style="background: rgba(222,165,132,0.04);">
                        <span class="arch-layer-badge badge-rust">Rust</span>
                        <span class="arch-layer-name">Data Plane</span>
                        <span class="arch-layer-tech">PQC-TLS &bull; DPDK Packet Processing &bull; DPI &bull; Protocol Adapters &bull; &lt;1ms</span>
                    </div>
                </div>
                <p style="margin-top:24px; color:#64748b; font-size:0.9em;">100% Open Source &bull; Apache 2.0 License &bull; Air-Gapped Capable</p>
            </div>

            <!-- SLIDE 4: Live Demo -->
            <div class="slide bg-demo" data-time="4:30" style="padding:0; justify-content:flex-start; align-items:stretch;">
                <div class="demo-layout">
                    <div class="demo-sidebar">
                        <div style="padding:4px 0 16px 0;">
                            <div class="label" style="margin-bottom:4px;">Live Demo</div>
                            <h3 style="font-size:1.1em; color:#f8fafc;">Mainframe Modernization</h3>
                        </div>
                        <div class="demo-step active" onclick="showDemoStep(1)" id="ds1">
                            <span class="demo-step-num">1</span><h4>System Discovery</h4>
                            <p>Discover legacy systems</p>
                        </div>
                        <div class="demo-step" onclick="showDemoStep(2)" id="ds2">
                            <span class="demo-step-num">2</span><h4>COBOL Analysis</h4>
                            <p>AI code analysis</p>
                        </div>
                        <div class="demo-step" onclick="showDemoStep(3)" id="ds3">
                            <span class="demo-step-num">3</span><h4>Protocol Analysis</h4>
                            <p>Reverse engineering</p>
                        </div>
                        <div class="demo-step" onclick="showDemoStep(4)" id="ds4">
                            <span class="demo-step-num">4</span><h4>Code Generation</h4>
                            <p>Modern code output</p>
                        </div>
                        <div class="demo-step" onclick="showDemoStep(5)" id="ds5">
                            <span class="demo-step-num">5</span><h4>Modernization Plan</h4>
                            <p>Roadmap generation</p>
                        </div>
                    </div>
                    <div class="demo-main" id="demo-content">
                        <!-- Dynamic demo content -->
                    </div>
                </div>
            </div>

            <!-- SLIDE 5: Impact -->
            <div class="slide bg-gradient" data-time="12:30">
                <div class="label">Business Impact</div>
                <h2>Measurable <span class="gradient-text">Results</span></h2>
                <div class="impact-grid">
                    <div class="impact-card">
                        <div class="before">6-12 months</div>
                        <div class="after">2-4 hours</div>
                        <div class="metric-name">Protocol Discovery</div>
                    </div>
                    <div class="impact-card">
                        <div class="before">None</div>
                        <div class="after">NIST Level 5</div>
                        <div class="metric-name">Quantum Readiness</div>
                    </div>
                    <div class="impact-card">
                        <div class="before">65 minutes</div>
                        <div class="after">&lt;1 second</div>
                        <div class="metric-name">Security Response</div>
                    </div>
                    <div class="impact-card">
                        <div class="before">2-4 weeks</div>
                        <div class="after">&lt;10 minutes</div>
                        <div class="metric-name">Compliance Reports</div>
                    </div>
                    <div class="impact-card">
                        <div class="before">$5-50M / system</div>
                        <div class="after">$200K-500K</div>
                        <div class="metric-name">Integration Cost</div>
                    </div>
                    <div class="impact-card">
                        <div class="before">$10-50 / event</div>
                        <div class="after">&lt;$0.01</div>
                        <div class="metric-name">Security Cost / Event</div>
                    </div>
                </div>
            </div>

            <!-- SLIDE 6: CTA -->
            <div class="slide bg-hero" data-time="14:00">
                <div class="cta-box">
                    <div class="label">Next Steps</div>
                    <h2><span class="gradient-text">Start Your PoC</span></h2>
                    <p class="subtitle" style="font-size:1.15em; margin: 0 auto;">2-week proof of concept. Connect to your test environment. Full protocol discovery and modernization assessment.</p>
                    <div class="cta-features">
                        <span class="cta-feature">100% Open Source</span>
                        <span class="cta-feature">Air-Gapped Ready</span>
                        <span class="cta-feature">Zero Code Changes</span>
                        <span class="cta-feature">9 Compliance Frameworks</span>
                        <span class="cta-feature">Apache 2.0 License</span>
                    </div>
                    <button class="cta-btn" style="margin-top:12px;">Contact Us &rarr;</button>
                    <p style="margin-top:20px; color:#64748b; font-size:0.9em;">enterprise@qbitel.com</p>
                </div>
            </div>
        </div>

        <!-- Navigation bar -->
        <div class="nav-bar">
            <button class="nav-btn" onclick="prevSlide()" id="btn-prev">&larr; Back</button>
            <span class="timer" id="timer">0:00</span>
            <div class="progress-bar"><div class="progress-fill" id="progress"></div></div>
            <span class="slide-counter" id="counter">1 / 7</span>
            <button class="nav-btn primary" onclick="nextSlide()" id="btn-next">Next &rarr;</button>
        </div>

        <!-- Keyboard hints -->
        <div class="key-hints">
            <span class="key-hint">&larr; &rarr; Navigate</span>
            <span class="key-hint">F Full Screen</span>
            <span class="key-hint">T Timer</span>
        </div>

        <script>
            // ===== Slide Navigation =====
            let currentSlide = 0;
            const slides = document.querySelectorAll('.slide');
            const totalSlides = slides.length;
            let timerRunning = false;
            let timerStart = 0;
            let timerInterval = null;

            function goToSlide(n) {
                if (n < 0 || n >= totalSlides) return;
                slides[currentSlide].classList.remove('active');
                currentSlide = n;
                slides[currentSlide].classList.add('active');
                document.getElementById('counter').textContent = (currentSlide + 1) + ' / ' + totalSlides;
                document.getElementById('progress').style.width = ((currentSlide + 1) / totalSlides * 100) + '%';
                if (currentSlide === 4) initDemoSlide();
            }

            function nextSlide() { goToSlide(currentSlide + 1); }
            function prevSlide() { goToSlide(currentSlide - 1); }

            // Keyboard navigation
            document.addEventListener('keydown', (e) => {
                if (e.key === 'ArrowRight' || e.key === ' ') { e.preventDefault(); nextSlide(); }
                else if (e.key === 'ArrowLeft') { e.preventDefault(); prevSlide(); }
                else if (e.key === 'f' || e.key === 'F') {
                    if (!document.fullscreenElement) document.documentElement.requestFullscreen();
                    else document.exitFullscreen();
                }
                else if (e.key === 't' || e.key === 'T') { toggleTimer(); }
                else if (e.key === 'Home') { goToSlide(0); }
                else if (e.key === 'End') { goToSlide(totalSlides - 1); }
            });

            // Timer
            function toggleTimer() {
                if (timerRunning) {
                    clearInterval(timerInterval);
                    timerRunning = false;
                } else {
                    if (timerStart === 0) timerStart = Date.now();
                    timerInterval = setInterval(updateTimer, 1000);
                    timerRunning = true;
                }
            }
            function updateTimer() {
                const elapsed = Math.floor((Date.now() - timerStart) / 1000);
                const min = Math.floor(elapsed / 60);
                const sec = elapsed % 60;
                document.getElementById('timer').textContent = min + ':' + String(sec).padStart(2, '0');
            }

            // ===== Live Demo Logic =====
            let currentDemoStep = 1;
            let demoData = {};

            function initDemoSlide() {
                if (!demoData.initialized) {
                    demoData.initialized = true;
                    showDemoStep(1);
                }
            }

            function showDemoStep(step) {
                currentDemoStep = step;
                for (let i = 1; i <= 5; i++) {
                    const el = document.getElementById('ds' + i);
                    el.classList.remove('active');
                    if (i < step) el.classList.add('done');
                }
                document.getElementById('ds' + step).classList.add('active');
                renderDemoContent(step);
            }

            function renderDemoContent(step) {
                const c = document.getElementById('demo-content');
                switch(step) {
                    case 1: c.innerHTML = `
                        <div class="demo-panel">
                            <h2>Step 1: Legacy System Discovery</h2>
                            <p class="desc">Our AI agents passively tap your network and discover every legacy system &mdash; in 2-4 hours versus 6-12 months of manual reverse engineering.</p>
                            <button class="demo-btn" onclick="runDiscover()">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
                                Discover Systems
                            </button>
                            <div class="demo-output" id="out1"></div>
                        </div>`; break;
                    case 2: c.innerHTML = `
                        <div class="demo-panel">
                            <h2>Step 2: AI-Powered COBOL Analysis</h2>
                            <p class="desc">Deep analysis of COBOL source code &mdash; complexity scoring, legacy pattern detection, and modernization opportunities identified in seconds.</p>
                            <button class="demo-btn" onclick="runCobolAnalysis()">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m10 20-6-6 6-6"/><path d="m14 4 6 6-6 6"/></svg>
                                Analyze COBOL
                            </button>
                            <div class="demo-output" id="out2"></div>
                        </div>`; break;
                    case 3: c.innerHTML = `
                        <div class="demo-panel">
                            <h2>Step 3: Protocol Reverse Engineering</h2>
                            <p class="desc">Decode EBCDIC and proprietary mainframe binary formats. Detect field boundaries, data types, and encoding &mdash; fully automated.</p>
                            <button class="demo-btn" onclick="runProtocol()">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8"/><path d="M12 17v4"/></svg>
                                Analyze Protocol Data
                            </button>
                            <div class="demo-output" id="out3"></div>
                        </div>`; break;
                    case 4: c.innerHTML = `
                        <div class="demo-panel">
                            <h2>Step 4: Modern Code Generation</h2>
                            <p class="desc">Auto-generate production-ready Python, FastAPI endpoints, and SQL schemas from COBOL analysis. 100% data fidelity.</p>
                            <button class="demo-btn" onclick="runCodeGen()">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>
                                Generate Modern Code
                            </button>
                            <div id="code-area" style="display:none; margin-top:16px;">
                                <div class="code-tabs">
                                    <div class="code-tab active" onclick="switchTab(this,'models')">Python Models</div>
                                    <div class="code-tab" onclick="switchTab(this,'api')">FastAPI Endpoints</div>
                                    <div class="code-tab" onclick="switchTab(this,'sql')">SQL Schema</div>
                                </div>
                                <pre class="code-block"><code id="code-display"></code></pre>
                            </div>
                        </div>`; break;
                    case 5: c.innerHTML = `
                        <div class="demo-panel">
                            <h2>Step 5: Modernization Roadmap</h2>
                            <p class="desc">Generate a complete, auditable modernization plan with phases, risk assessment, effort estimation, and deliverables.</p>
                            <div style="margin-bottom:16px;">
                                <select id="approach" style="padding:10px 16px; background:#1a1b26; color:#e2e8f0; border:1px solid rgba(255,255,255,0.1); border-radius:8px; font-family:inherit; font-size:0.9em; margin-right:8px;">
                                    <option value="refactor">Refactor (Transform Code)</option>
                                    <option value="replatform">Replatform (Cloud Migration)</option>
                                    <option value="rearchitect">Rearchitect (Redesign)</option>
                                </select>
                                <button class="demo-btn" onclick="runPlan()" style="vertical-align:middle;">
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                                    Generate Plan
                                </button>
                            </div>
                            <div class="demo-output" id="out5"></div>
                        </div>`; break;
                }
            }

            // ===== API Calls =====
            async function runDiscover() {
                const out = document.getElementById('out1');
                out.classList.add('visible');
                out.innerHTML = '<span class="loader"></span> Scanning network for legacy systems...';
                try {
                    const r = await fetch('/api/systems');
                    const d = await r.json();
                    let h = '<h3 style="color:#00ff88; margin-bottom:16px;">Discovered ' + d.total + ' Legacy Mainframe Systems</h3>';
                    d.systems.forEach(s => {
                        const statusColor = s.status === 'active' ? '#00ff88' : '#ffaa00';
                        h += '<div class="sys-card"><h4>' + s.name + ' <span style="color:#64748b; font-weight:400;">(' + s.system_id + ')</span></h4><div class="sys-metrics">' +
                            '<span class="sys-metric"><span class="label">Platform </span><span class="value">' + s.platform + '</span></span>' +
                            '<span class="sys-metric"><span class="label">Language </span><span class="value">' + s.language + '</span></span>' +
                            '<span class="sys-metric"><span class="label">LOC </span><span class="value">' + (s.lines_of_code/1e6).toFixed(1) + 'M</span></span>' +
                            '<span class="sys-metric"><span class="label">Age </span><span class="value">' + s.age_years + ' years</span></span>' +
                            '<span class="sys-metric"><span class="label">Status </span><span class="value" style="color:' + statusColor + '">' + s.status + '</span></span>' +
                            '<span class="sys-metric"><span class="label">Deps </span><span class="value">' + s.dependencies.join(', ') + '</span></span>' +
                            '</div></div>';
                    });
                    out.innerHTML = h;
                } catch(e) { out.innerHTML = '<span style="color:#ff4444;">Error: ' + e.message + '</span>'; }
            }

            async function runCobolAnalysis() {
                const out = document.getElementById('out2');
                out.classList.add('visible');
                out.innerHTML = '<span class="loader"></span> Analyzing COBOL source code...';
                try {
                    const lr = await fetch('/api/analyze/cobol/list');
                    const files = await lr.json();
                    if (!files.files.length) { out.innerHTML = 'No COBOL files found.'; return; }
                    const ar = await fetch('/api/analyze/cobol/' + files.files[0].name);
                    const a = await ar.json();
                    let h = '<h3 style="color:#00ff88; margin-bottom:16px;">Analysis: ' + a.name + '</h3>';
                    h += '<div class="sys-metrics" style="margin-bottom:16px;">' +
                        '<span class="sys-metric"><span class="label">Lines </span><span class="value">' + a.lines_of_code + '</span></span>' +
                        '<span class="sys-metric"><span class="label">Complexity </span><span class="value">' + a.complexity_score + '</span></span>' +
                        '<span class="sys-metric"><span class="label">Data Divisions </span><span class="value">' + a.data_divisions + '</span></span>' +
                        '<span class="sys-metric"><span class="label">Variables </span><span class="value">' + (a.analysis.working_storage?.variable_count || 0) + '</span></span>' +
                        '</div>';
                    h += '<h4 style="color:#94a3b8; margin:16px 0 8px;">Legacy Patterns Detected</h4>';
                    (a.analysis.legacy_patterns || []).forEach(p => {
                        const cls = p.severity === 'critical' ? 'critical' : p.severity === 'high' ? 'high' : 'warning';
                        h += '<div class="pattern ' + cls + '"><strong>' + p.pattern + '</strong> <span style="opacity:0.6">(' + p.severity + ')</span><br><span style="color:#94a3b8; font-size:0.9em;">' + p.description + '</span></div>';
                    });
                    h += '<h4 style="color:#94a3b8; margin:16px 0 8px;">Modernization Opportunities</h4>';
                    (a.analysis.modernization_opportunities || []).forEach(o => {
                        h += '<div class="pattern opportunity"><strong>' + o.area + '</strong><br><span style="color:#94a3b8; font-size:0.9em;">' + o.current + ' &rarr; <span style="color:#00ff88;">' + o.modern + '</span></span></div>';
                    });
                    out.innerHTML = h;
                } catch(e) { out.innerHTML = '<span style="color:#ff4444;">Error: ' + e.message + '</span>'; }
            }

            async function runProtocol() {
                const out = document.getElementById('out3');
                out.classList.add('visible');
                out.innerHTML = '<span class="loader"></span> Decoding EBCDIC mainframe data...';
                try {
                    const r = await fetch('/api/analyze/protocol', {
                        method: 'POST', headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ protocol_data: 'd1d6c8d540e2d4c9e3c840404040404040404040f1f2f3f4f5f6f7f8f9f0c1c3c3d6e4d5e340', system_context: 'IBM z/OS Customer Master File' })
                    });
                    const a = await r.json();
                    let h = '<h3 style="color:#00ff88; margin-bottom:16px;">Protocol Analysis Results</h3>';
                    h += '<div class="sys-metrics" style="margin-bottom:16px;">' +
                        '<span class="sys-metric"><span class="label">Encoding </span><span class="value">' + a.encoding + '</span></span>' +
                        '<span class="sys-metric"><span class="label">Structure </span><span class="value">' + a.structure.type + '</span></span>' +
                        '<span class="sys-metric"><span class="label">Length </span><span class="value">' + a.raw_length + ' bytes</span></span>' +
                        '</div>';
                    h += '<h4 style="color:#94a3b8; margin:12px 0 8px;">Detected Fields</h4>';
                    (a.fields || []).forEach(f => {
                        h += '<div style="padding:8px 12px; margin:4px 0; background:rgba(0,217,255,0.05); border-radius:6px; font-family: JetBrains Mono, monospace; font-size:0.85em;">' +
                            '<span style="color:#00d9ff; font-weight:600;">' + f.name + '</span> <span style="color:#64748b;">@ offset ' + f.offset + ', ' + f.length + ' bytes (' + f.type + ')</span></div>';
                    });
                    h += '<h4 style="color:#94a3b8; margin:16px 0 8px;">Recommendations</h4>';
                    (a.recommendations || []).forEach(r => {
                        h += '<div class="pattern opportunity"><strong>' + r.issue + '</strong><br><span style="color:#00ff88; font-size:0.9em;">' + r.solution + '</span></div>';
                    });
                    out.innerHTML = h;
                } catch(e) { out.innerHTML = '<span style="color:#ff4444;">Error: ' + e.message + '</span>'; }
            }

            let genCode = {};
            async function runCodeGen() {
                const area = document.getElementById('code-area');
                const display = document.getElementById('code-display');
                area.style.display = 'block';
                display.textContent = 'Generating modern code...';
                try {
                    const lr = await fetch('/api/analyze/cobol/list');
                    const files = await lr.json();
                    if (!files.files.length) { display.textContent = 'No COBOL files found.'; return; }
                    const r = await fetch('/api/generate', {
                        method: 'POST', headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ cobol_file: files.files[0].name })
                    });
                    const d = await r.json();
                    genCode = d.generated_code || {};
                    display.textContent = genCode.python_models || 'No models generated';
                } catch(e) { display.textContent = 'Error: ' + e.message; }
            }

            function switchTab(el, tab) {
                document.querySelectorAll('.code-tab').forEach(t => t.classList.remove('active'));
                el.classList.add('active');
                const d = document.getElementById('code-display');
                if (tab === 'models') d.textContent = genCode.python_models || '';
                else if (tab === 'api') d.textContent = genCode.fastapi_endpoints || '';
                else if (tab === 'sql') d.textContent = genCode.sql_schema || '';
            }

            async function runPlan() {
                const out = document.getElementById('out5');
                out.classList.add('visible');
                out.innerHTML = '<span class="loader"></span> Generating modernization plan...';
                const approach = document.getElementById('approach').value;
                try {
                    const r = await fetch('/api/modernize', {
                        method: 'POST', headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ system_id: 'SYS001', approach: approach, target_language: 'python', target_framework: 'fastapi' })
                    });
                    const p = await r.json();
                    let h = '<h3 style="color:#00ff88; margin-bottom:16px;">Modernization Plan Generated</h3>';
                    h += '<div class="sys-metrics" style="margin-bottom:16px;">' +
                        '<span class="sys-metric"><span class="label">Plan </span><span class="value">' + p.plan_id + '</span></span>' +
                        '<span class="sys-metric"><span class="label">Approach </span><span class="value">' + p.approach.toUpperCase() + '</span></span>' +
                        '<span class="sys-metric"><span class="label">Risk </span><span class="value" style="color:#ffaa00;">' + p.risk_level.toUpperCase() + '</span></span>' +
                        '<span class="sys-metric"><span class="label">Effort </span><span class="value">' + p.estimated_effort_days + ' days</span></span>' +
                        '</div>';
                    h += '<h4 style="color:#94a3b8; margin:12px 0 8px;">Project Phases</h4>';
                    let totalWeeks = 0;
                    (p.phases || []).forEach(ph => {
                        totalWeeks += ph.duration_weeks;
                        h += '<div class="phase"><div class="phase-num">' + ph.phase + '</div><div class="phase-info"><h4>' + ph.name + ' <span class="duration">(' + ph.duration_weeks + ' weeks)</span></h4><p>' + ph.description + '</p></div></div>';
                    });
                    h += '<div style="margin-top:16px; padding:12px 16px; background:rgba(0,255,136,0.08); border-radius:8px; text-align:center;"><strong style="color:#00ff88;">Total Duration: ' + totalWeeks + ' weeks</strong></div>';
                    out.innerHTML = h;
                } catch(e) { out.innerHTML = '<span style="color:#ff4444;">Error: ' + e.message + '</span>'; }
            }

            // Initialize first slide
            goToSlide(0);
        </script>
    </body>
    </html>
    """)

# ============================================================================
# E2E Demo Backend — Simulators
# ============================================================================

import random

class NetworkTrafficSimulator:
    """Simulates network traffic capture from mainframe environment."""

    def __init__(self):
        self._scan_targets = [
            {"ip": "10.1.50.10", "port": 23, "protocol": "TN3270e", "system": "SYS001", "name": "Core Banking System", "latency_ms": 2.1},
            {"ip": "10.1.50.11", "port": 1414, "protocol": "IBM MQ", "system": "SYS001", "name": "Core Banking System", "latency_ms": 0.8},
            {"ip": "10.1.50.20", "port": 23, "protocol": "TN3270e", "system": "SYS002", "name": "Customer Master System", "latency_ms": 1.9},
            {"ip": "10.1.50.20", "port": 446, "protocol": "DRDA/DB2", "system": "SYS002", "name": "Customer Master System", "latency_ms": 1.2},
            {"ip": "10.1.50.30", "port": 23, "protocol": "TN3270e", "system": "SYS003", "name": "Account Processing Batch", "latency_ms": 3.4},
            {"ip": "10.1.50.30", "port": 1414, "protocol": "IBM MQ", "system": "SYS003", "name": "Account Processing Batch", "latency_ms": 1.1},
            {"ip": "10.1.50.10", "port": 8090, "protocol": "CICS/TS", "system": "SYS001", "name": "Core Banking System", "latency_ms": 0.6},
        ]
        self._ebcdic_samples = [
            {"label": "Customer Record (EBCDIC)", "hex": "d1d6c8d540e2d4c9e3c840404040404040404040f1f2f3f4f5f6f7f8f9f0c1c3c3d6e4d5e340", "decoded": "JOHN SMITH          1234567890ACCOUNT "},
            {"label": "SWIFT MT103 Header",      "hex": "f1f5f3c6c9d540d4e3f1f0f340e2e6c9c6e340e3d9c1d5e2c6c5d940f2f0f2f6f0f3f0f4",     "decoded": "153FIN MT103 SWIFT TRANSFER 20260304"},
            {"label": "ISO-8583 Auth Request",    "hex": "f0f1f0f0f2f0f0f0f0f0f0f0f0f1f5f0f0f0f0f0f0f0f5f2f4f3f6f1f2f3f4f5f6f7f8",   "decoded": "0100 200000000150000000524361234567 8"},
            {"label": "CICS Transaction (ACCT)",  "hex": "c1c3c3e3f0f0f0f1d7d9d6c3c5e2e2c9d5c740c2c1d3c1d5c3c540c9d5d8e4c9d9e8",       "decoded": "ACCT0001PROCESSING BALANCE INQUIRY"},
            {"label": "MQ Message Header",        "hex": "d4d840404040f0f0f0f1f2f3d8e4c5e4c540d4c1d5c1c7c5d940d7e4e340d6d7c5d9",       "decoded": "MQ    000123QUEUE MANAGER PUT OPER"},
        ]

    def scan_network(self) -> dict:
        results = []
        for t in self._scan_targets:
            results.append({**t, "status": "discovered", "encryption": "NONE", "risk": "CRITICAL" if t["protocol"] == "TN3270e" else "HIGH"})
        unique_systems = {t["system"] for t in self._scan_targets}
        unique_protos = {t["protocol"] for t in self._scan_targets}
        return {
            "scan_id": f"SCAN-{uuid.uuid4().hex[:8].upper()}",
            "timestamp": datetime.now().isoformat(),
            "network": "10.1.50.0/24",
            "duration_seconds": round(random.uniform(1.8, 3.2), 1),
            "targets_found": len(self._scan_targets),
            "unique_systems": len(unique_systems),
            "unique_protocols": len(unique_protos),
            "unencrypted_channels": sum(1 for t in self._scan_targets if True),
            "results": results,
        }

    def capture_traffic(self) -> dict:
        packets = []
        for i, sample in enumerate(self._ebcdic_samples):
            packets.append({
                "packet_id": i + 1,
                "timestamp": (datetime.now() - timedelta(seconds=random.randint(0, 60))).isoformat(),
                "src": f"10.1.50.{random.choice([10,20,30])}:{random.randint(1024,65535)}",
                "dst": f"10.2.1.{random.randint(1,50)}:{random.choice([23,443,1414,8090])}",
                "protocol": random.choice(["TN3270e", "CICS/TS", "IBM MQ", "DRDA/DB2"]),
                "length": len(sample["hex"]) // 2,
                "label": sample["label"],
                "hex_dump": sample["hex"],
                "decoded_ascii": sample["decoded"],
                "encryption": "NONE",
                "pii_detected": "SSN" in sample["decoded"].upper() or "ACCOUNT" in sample["decoded"].upper(),
            })
        return {
            "capture_id": f"CAP-{uuid.uuid4().hex[:8].upper()}",
            "timestamp": datetime.now().isoformat(),
            "duration_ms": round(random.uniform(45, 120), 1),
            "packets_captured": len(packets),
            "unencrypted": len(packets),
            "pii_exposure": sum(1 for p in packets if p["pii_detected"]),
            "packets": packets,
        }


class PQCDemoSimulator:
    """Simulates PQC crypto operations with real NIST FIPS parameter sizes."""

    # Real sizes from ai_engine/crypto/mlkem.py and dilithium.py
    MLKEM768 = {"name": "ML-KEM-768", "fips": "FIPS 203", "nist_level": 3, "public_key": 1184, "private_key": 2400, "ciphertext": 1088, "shared_secret": 32}
    DILITHIUM3 = {"name": "ML-DSA-65 (Dilithium-3)", "fips": "FIPS 204", "nist_level": 3, "public_key": 1952, "private_key": 4000, "signature": 3293}

    def generate_keypair(self) -> dict:
        t0 = time.time()
        pub_hex = os.urandom(self.MLKEM768["public_key"]).hex()
        priv_hex = os.urandom(self.MLKEM768["private_key"]).hex()
        sig_pub_hex = os.urandom(self.DILITHIUM3["public_key"]).hex()
        latency = round((time.time() - t0) * 1000, 2)
        return {
            "algorithm": "ML-KEM-768 + ML-DSA-65",
            "kem": {**self.MLKEM768, "public_key_hex": pub_hex[:64] + "...", "public_key_bytes": self.MLKEM768["public_key"]},
            "signer": {**self.DILITHIUM3, "public_key_hex": sig_pub_hex[:64] + "...", "public_key_bytes": self.DILITHIUM3["public_key"]},
            "total_key_material_bytes": self.MLKEM768["public_key"] + self.MLKEM768["private_key"] + self.DILITHIUM3["public_key"] + self.DILITHIUM3["private_key"],
            "generation_latency_ms": max(latency, round(random.uniform(0.3, 0.9), 2)),
            "nist_security_level": 3,
            "quantum_safe": True,
        }

    def encrypt_transaction(self, plaintext: str) -> dict:
        t0 = time.time()
        plain_bytes = plaintext.encode("utf-8")
        ciphertext = os.urandom(self.MLKEM768["ciphertext"] + len(plain_bytes))
        shared_secret = os.urandom(32)
        latency = round((time.time() - t0) * 1000, 2)
        return {
            "operation": "PQC-KEM Encapsulation + AES-256-GCM",
            "algorithm": "ML-KEM-768",
            "plaintext": plaintext,
            "plaintext_bytes": len(plain_bytes),
            "ciphertext_hex": ciphertext.hex()[:120] + "...",
            "ciphertext_bytes": len(ciphertext),
            "shared_secret_hex": shared_secret.hex(),
            "overhead_bytes": len(ciphertext) - len(plain_bytes),
            "overhead_percent": round((len(ciphertext) - len(plain_bytes)) / len(plain_bytes) * 100, 1),
            "encryption_latency_ms": max(latency, round(random.uniform(0.2, 0.7), 2)),
            "nist_level": 3,
        }

    def sign_transaction(self, message: str) -> dict:
        t0 = time.time()
        signature = os.urandom(self.DILITHIUM3["signature"])
        latency = round((time.time() - t0) * 1000, 2)
        return {
            "operation": "ML-DSA-65 Digital Signature",
            "algorithm": self.DILITHIUM3["name"],
            "message_hash": hashlib.sha256(message.encode()).hexdigest(),
            "signature_hex": signature.hex()[:120] + "...",
            "signature_bytes": self.DILITHIUM3["signature"],
            "signing_latency_ms": max(latency, round(random.uniform(0.4, 1.1), 2)),
            "verification_latency_ms": round(random.uniform(0.2, 0.5), 2),
            "nist_level": 3,
            "verified": True,
        }


class SecurityMonitorSim:
    """Simulates live security monitoring dashboard."""

    def __init__(self):
        self._base_time = datetime.now()
        self._threat_templates = [
            {"type": "BRUTE_FORCE", "severity": "HIGH", "source": "203.0.113.42", "target": "10.1.50.10:23", "action": "BLOCKED", "detail": "TN3270e login brute-force: 847 attempts in 30s", "response_ms": 12},
            {"type": "ANOMALY", "severity": "MEDIUM", "source": "10.2.1.15", "target": "10.1.50.20:446", "action": "FLAGGED", "detail": "Unusual DB2 query pattern: 3x baseline read volume", "response_ms": 340},
            {"type": "DATA_EXFIL", "severity": "CRITICAL", "source": "10.2.1.33", "target": "198.51.100.7:443", "action": "BLOCKED", "detail": "Bulk customer PII transfer detected (45K records)", "response_ms": 8},
            {"type": "HARVEST_ATTACK", "severity": "CRITICAL", "source": "192.0.2.99", "target": "10.1.50.10:1414", "action": "MITIGATED", "detail": "Harvest-now-decrypt-later: MQ messages captured — PQC re-encryption applied", "response_ms": 3},
            {"type": "CICS_INJECTION", "severity": "HIGH", "source": "10.2.1.22", "target": "10.1.50.10:8090", "action": "BLOCKED", "detail": "CICS transaction injection attempt via modified 3270 data stream", "response_ms": 6},
            {"type": "COMPLIANCE_DRIFT", "severity": "LOW", "source": "INTERNAL", "target": "SYS003", "action": "ALERTED", "detail": "TLS certificate approaching expiry (14 days remaining)", "response_ms": 0},
        ]

    def get_metrics(self) -> dict:
        uptime = (datetime.now() - self._base_time).total_seconds()
        return {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": round(uptime),
            "transactions_today": 15_234_891 + int(uptime * 176),
            "pqc_encryptions": 14_987_320 + int(uptime * 173),
            "pqc_signatures": 14_987_320 + int(uptime * 173),
            "threats_blocked_today": 1247 + random.randint(0, 3),
            "avg_encryption_latency_ms": round(random.uniform(0.4, 0.8), 2),
            "avg_response_time_ms": round(random.uniform(6, 18), 1),
            "autonomous_response_rate": 0.78,
            "systems_protected": 3,
            "channels_encrypted": 7,
            "kafka_throughput_msg_sec": random.randint(98000, 103000),
        }

    def get_threat_events(self) -> list:
        events = []
        for i, t in enumerate(self._threat_templates):
            events.append({
                **t,
                "event_id": f"EVT-{uuid.uuid4().hex[:8].upper()}",
                "timestamp": (datetime.now() - timedelta(minutes=random.randint(1, 120))).isoformat(),
            })
        return sorted(events, key=lambda e: e["timestamp"], reverse=True)

    def get_agent_status(self) -> list:
        agents = [
            {"id": "AGT-001", "name": "Protocol Sentinel", "capability": "PROTOCOL_ANALYSIS", "status": "MONITORING", "tasks_completed": random.randint(12400, 12500), "uptime_hours": round(random.uniform(168, 720), 1)},
            {"id": "AGT-002", "name": "Threat Hunter", "capability": "THREAT_ANALYSIS", "status": "PROCESSING", "tasks_completed": random.randint(8700, 8900), "uptime_hours": round(random.uniform(168, 720), 1)},
            {"id": "AGT-003", "name": "Compliance Auditor", "capability": "COMPLIANCE_CHECK", "status": "IDLE", "tasks_completed": random.randint(3200, 3400), "uptime_hours": round(random.uniform(168, 720), 1)},
            {"id": "AGT-004", "name": "Anomaly Detector", "capability": "ANOMALY_DETECTION", "status": "MONITORING", "tasks_completed": random.randint(45000, 46000), "uptime_hours": round(random.uniform(168, 720), 1)},
            {"id": "AGT-005", "name": "Incident Responder", "capability": "INCIDENT_RESPONSE", "status": "STANDBY", "tasks_completed": random.randint(1200, 1300), "uptime_hours": round(random.uniform(168, 720), 1)},
        ]
        return agents


class ComplianceReportGenerator:
    """Generates compliance assessment reports."""

    def generate_report(self) -> dict:
        frameworks = [
            {"name": "PCI-DSS 4.0", "score": 94, "status": "COMPLIANT", "controls_total": 64, "controls_passed": 60, "controls_failed": 2, "controls_na": 2,
             "findings": ["Req 3.5.1: PQC key rotation schedule defined", "Req 4.1: TLS 1.3 with ML-KEM-768 enforced", "Req 6.2.4: COBOL input validation gap identified (remediation in progress)"]},
            {"name": "DORA (EU)", "score": 91, "status": "COMPLIANT", "controls_total": 41, "controls_passed": 37, "controls_failed": 1, "controls_na": 3,
             "findings": ["Art 5: ICT risk management framework active", "Art 11: Incident classification automated", "Art 15: Third-party risk register needs update"]},
            {"name": "SOX", "score": 97, "status": "COMPLIANT", "controls_total": 38, "controls_passed": 37, "controls_failed": 0, "controls_na": 1,
             "findings": ["Sec 302: Financial data integrity verified via ML-DSA signatures", "Sec 404: Internal controls monitored by agent AGT-003"]},
            {"name": "NIST 800-53", "score": 89, "status": "CONDITIONAL", "controls_total": 122, "controls_passed": 108, "controls_failed": 5, "controls_na": 9,
             "findings": ["SC-13: PQC algorithms FIPS 203/204/205 implemented", "AU-6: Audit log analysis automated", "IA-7: Cryptographic module validation pending CMVP"]},
            {"name": "HIPAA", "score": 92, "status": "COMPLIANT", "controls_total": 54, "controls_passed": 50, "controls_failed": 1, "controls_na": 3,
             "findings": ["164.312(a): Access controls enforced via zero-trust", "164.312(e): PHI encrypted with ML-KEM-768 in transit"]},
        ]
        overall = round(sum(f["score"] for f in frameworks) / len(frameworks), 1)
        return {
            "report_id": f"COMP-{uuid.uuid4().hex[:8].upper()}",
            "generated_at": datetime.now().isoformat(),
            "assessment_period": f"{(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}",
            "overall_score": overall,
            "overall_status": "COMPLIANT" if overall >= 85 else "NON-COMPLIANT",
            "frameworks": frameworks,
            "pqc_readiness": {"algorithms": ["ML-KEM-768 (FIPS 203)", "ML-DSA-65 (FIPS 204)", "SLH-DSA (FIPS 205)"], "nist_level": 3, "quantum_safe": True, "cmvp_status": "Pending"},
        }

    def get_audit_trail(self) -> list:
        entries = [
            {"action": "PQC_KEY_ROTATION", "actor": "AGT-001", "target": "SYS001/TN3270e", "result": "SUCCESS", "detail": "ML-KEM-768 session keys rotated (15-min interval)"},
            {"action": "THREAT_BLOCKED", "actor": "AGT-002", "target": "10.1.50.10:23", "result": "SUCCESS", "detail": "Brute-force attack blocked: 847 attempts from 203.0.113.42"},
            {"action": "COMPLIANCE_SCAN", "actor": "AGT-003", "target": "ALL_SYSTEMS", "result": "SUCCESS", "detail": "Automated PCI-DSS 4.0 control assessment completed: 94/100"},
            {"action": "DATA_ENCRYPTED", "actor": "SYSTEM", "target": "SYS001/MQ", "result": "SUCCESS", "detail": "Retroactive PQC encryption applied to 1.2M queued MQ messages"},
            {"action": "ANOMALY_FLAGGED", "actor": "AGT-004", "target": "SYS002/DB2", "result": "REVIEW", "detail": "DB2 query volume 3x baseline — flagged for SOC review"},
            {"action": "CERT_RENEWED", "actor": "SYSTEM", "target": "SYS003/TLS", "result": "SUCCESS", "detail": "TLS 1.3 certificate renewed with hybrid X25519-ML-KEM-768"},
            {"action": "COBOL_SCAN", "actor": "AGT-001", "target": "CUSTMAST.cbl", "result": "SUCCESS", "detail": "Legacy pattern analysis: 3 critical findings, 4 modernization opportunities"},
            {"action": "MODERNIZATION", "actor": "AGT-003", "target": "SYS002", "result": "SUCCESS", "detail": "Python/FastAPI code generation completed for Customer Master module"},
        ]
        for e in entries:
            e["timestamp"] = (datetime.now() - timedelta(minutes=random.randint(1, 240))).isoformat()
            e["event_id"] = f"AUD-{uuid.uuid4().hex[:8].upper()}"
        return sorted(entries, key=lambda e: e["timestamp"], reverse=True)


# Initialize E2E simulators
network_sim = NetworkTrafficSimulator()
pqc_sim = PQCDemoSimulator()
security_mon = SecurityMonitorSim()
compliance_gen = ComplianceReportGenerator()

# ============================================================================
# E2E Demo API Endpoints
# ============================================================================

@app.get("/api/e2e/network/scan")
async def e2e_network_scan():
    return network_sim.scan_network()

@app.get("/api/e2e/network/capture")
async def e2e_network_capture():
    return network_sim.capture_traffic()

@app.get("/api/e2e/pqc/keygen")
async def e2e_pqc_keygen():
    return pqc_sim.generate_keypair()

class E2EEncryptRequest(BaseModel):
    plaintext: str = "{1:F01BANKUS33AXXX0000000000}{2:O1030900260304BANKGB2LAXXX00000000002603040900N}{4:\n:20:TXN-2026-00847\n:23B:CRED\n:32A:260304USD1500000,00\n:50K:/US33XXX0123456789\nACME CORPORATION\n:59:/GB2LXXX9876543210\nGLOBAL TRADING LTD\n:71A:SHA\n-}"

@app.post("/api/e2e/pqc/encrypt")
async def e2e_pqc_encrypt(req: E2EEncryptRequest):
    return pqc_sim.encrypt_transaction(req.plaintext)

class E2ESignRequest(BaseModel):
    message: str = "SWIFT MT103 Wire Transfer TXN-2026-00847 USD 1,500,000.00"

@app.post("/api/e2e/pqc/sign")
async def e2e_pqc_sign(req: E2ESignRequest):
    return pqc_sim.sign_transaction(req.message)

@app.get("/api/e2e/security/metrics")
async def e2e_security_metrics():
    return security_mon.get_metrics()

@app.get("/api/e2e/security/threats")
async def e2e_security_threats():
    return security_mon.get_threat_events()

@app.get("/api/e2e/security/agents")
async def e2e_security_agents():
    return security_mon.get_agent_status()

@app.get("/api/e2e/compliance/report")
async def e2e_compliance_report():
    return compliance_gen.generate_report()

@app.get("/api/e2e/compliance/audit")
async def e2e_compliance_audit():
    return compliance_gen.get_audit_trail()


# ============================================================================
# E2E Demo Frontend
# ============================================================================

@app.get("/e2e-demo")
async def e2e_demo_page():
    """Practical end-to-end product demo."""
    return HTMLResponse(content=E2E_DEMO_HTML)


E2E_DEMO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QBITEL Bridge — End-to-End Demo</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#08090d;--bg2:#0f1117;--bg3:#161922;--border:rgba(255,255,255,.07);--text:#e2e8f0;--dim:#64748b;--accent:#00d9ff;--green:#00ff88;--red:#ff4757;--orange:#ff9f43;--yellow:#ffd43b}
html,body{height:100%;font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--text);overflow:hidden}
.app{display:flex;height:100%}

/* Sidebar */
.sidebar{width:270px;background:var(--bg2);border-right:1px solid var(--border);display:flex;flex-direction:column;flex-shrink:0}
.sidebar-header{padding:20px 18px 16px;border-bottom:1px solid var(--border)}
.sidebar-header h1{font-size:1.15em;font-weight:800;background:linear-gradient(135deg,var(--accent),var(--green));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.sidebar-header p{font-size:.75em;color:var(--dim);margin-top:2px}
.steps{flex:1;overflow-y:auto;padding:12px 10px}
.step{display:flex;align-items:flex-start;gap:10px;padding:10px 12px;border-radius:8px;cursor:pointer;transition:.15s;margin-bottom:4px;border:1px solid transparent}
.step:hover{background:rgba(255,255,255,.03)}
.step.active{background:rgba(0,217,255,.06);border-color:rgba(0,217,255,.2)}
.step.done .step-num{background:var(--green);color:#000}
.step-num{width:22px;height:22px;border-radius:50%;background:var(--bg3);color:var(--dim);font-size:.7em;font-weight:700;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px;transition:.2s}
.step.active .step-num{background:var(--accent);color:#000}
.step-text h4{font-size:.85em;font-weight:600;color:var(--text)}
.step-text p{font-size:.7em;color:var(--dim);margin-top:1px}
.step.done .step-text h4{color:var(--dim)}
.sidebar-footer{padding:12px 18px;border-top:1px solid var(--border);font-size:.72em;color:var(--dim)}
.sidebar-footer span{color:var(--green);font-weight:600}

/* Main content */
.main{flex:1;display:flex;flex-direction:column;overflow:hidden}
.topbar{height:44px;background:var(--bg2);border-bottom:1px solid var(--border);display:flex;align-items:center;padding:0 24px;gap:12px;flex-shrink:0}
.topbar .tag{padding:3px 10px;border-radius:12px;font-size:.7em;font-weight:600;background:rgba(0,217,255,.1);color:var(--accent)}
.topbar .timer{font-family:'JetBrains Mono',monospace;font-size:.8em;color:var(--dim);margin-left:auto}
.content{flex:1;overflow-y:auto;padding:28px 32px}

/* Panels */
.panel{background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:24px;margin-bottom:20px}
.panel h2{font-size:1.3em;font-weight:700;margin-bottom:4px}
.panel .desc{color:var(--dim);font-size:.9em;margin-bottom:18px;line-height:1.5}
.btn{display:inline-flex;align-items:center;gap:6px;padding:10px 24px;border:none;border-radius:8px;font-weight:700;font-size:.88em;cursor:pointer;transition:.2s;font-family:inherit}
.btn-primary{background:linear-gradient(135deg,var(--accent),var(--green));color:#000}
.btn-primary:hover{transform:translateY(-1px);box-shadow:0 4px 16px rgba(0,217,255,.25)}
.btn-primary:disabled{opacity:.35;transform:none;cursor:not-allowed;box-shadow:none}
.btn-secondary{background:rgba(255,255,255,.06);color:var(--text);border:1px solid var(--border)}
.btn-secondary:hover{background:rgba(255,255,255,.1)}

/* Output area */
.out{background:rgba(0,0,0,.35);border:1px solid var(--border);border-radius:8px;padding:16px;margin-top:14px;display:none;animation:fadeIn .3s}
.out.show{display:block}
@keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}

/* Terminal */
.term{background:#000;border-radius:8px;padding:14px;font-family:'JetBrains Mono',monospace;font-size:.78em;line-height:1.7;max-height:300px;overflow-y:auto;white-space:pre-wrap}
.term .g{color:var(--green)}.term .c{color:var(--accent)}.term .r{color:var(--red)}.term .y{color:var(--yellow)}.term .d{color:var(--dim)}.term .o{color:var(--orange)}

/* Metric badges */
.metrics{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}
.metric{padding:6px 14px;border-radius:20px;background:rgba(0,217,255,.06);font-size:.82em}
.metric .l{color:var(--dim)}.metric .v{color:var(--green);font-weight:600}

/* Cards */
.card{background:rgba(255,255,255,.02);border:1px solid var(--border);border-radius:10px;padding:16px;margin:8px 0}
.card h4{color:var(--accent);font-size:.95em;margin-bottom:8px}
.card-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}

/* Severity badges */
.sev{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.72em;font-weight:700;text-transform:uppercase;letter-spacing:.04em}
.sev-critical{background:rgba(255,71,87,.15);color:var(--red)}
.sev-high{background:rgba(255,159,67,.15);color:var(--orange)}
.sev-medium{background:rgba(255,212,59,.12);color:var(--yellow)}
.sev-low{background:rgba(0,255,136,.1);color:var(--green)}

/* Patterns */
.pat{padding:10px 14px;margin:5px 0;border-radius:6px;font-size:.88em;border-left:3px solid}
.pat.critical{background:rgba(255,71,87,.06);border-color:var(--red)}
.pat.high{background:rgba(255,159,67,.06);border-color:var(--orange)}
.pat.warning{background:rgba(255,212,59,.05);border-color:var(--yellow)}
.pat.info{background:rgba(0,217,255,.05);border-color:var(--accent)}
.pat.good{background:rgba(0,255,136,.05);border-color:var(--green)}

/* Code */
.code-tabs{display:flex;gap:2px;margin-bottom:-1px;position:relative;z-index:1}
.ctab{padding:8px 16px;background:rgba(255,255,255,.03);border:1px solid var(--border);border-bottom:none;border-radius:6px 6px 0 0;cursor:pointer;font-size:.8em;font-weight:600;color:var(--dim);transition:.15s}
.ctab.active{background:#000;color:var(--accent);border-color:rgba(0,217,255,.15)}
pre.cblock{background:#000;border:1px solid var(--border);border-radius:0 8px 8px 8px;padding:16px;overflow-x:auto;font-family:'JetBrains Mono',monospace;font-size:.78em;line-height:1.6;max-height:350px;overflow-y:auto}

/* Dashboard grid */
.dash-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.dash-card{background:var(--bg3);border:1px solid var(--border);border-radius:10px;padding:18px;text-align:center}
.dash-card .num{font-size:2em;font-weight:800;font-family:'JetBrains Mono',monospace;background:linear-gradient(135deg,var(--accent),var(--green));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.dash-card .lbl{font-size:.78em;color:var(--dim);margin-top:4px}

/* Threat timeline */
.threat{display:flex;align-items:flex-start;gap:12px;padding:12px;margin:6px 0;background:rgba(255,255,255,.015);border-radius:8px;border-left:3px solid}
.threat.BLOCKED{border-color:var(--green)}.threat.FLAGGED{border-color:var(--orange)}.threat.MITIGATED{border-color:var(--accent)}.threat.ALERTED{border-color:var(--yellow)}
.threat-time{font-family:'JetBrains Mono',monospace;font-size:.72em;color:var(--dim);min-width:55px}
.threat-info{flex:1}.threat-info strong{font-size:.88em}.threat-info p{font-size:.8em;color:var(--dim);margin-top:2px}

/* Agent panel */
.agent{display:flex;align-items:center;gap:12px;padding:10px 14px;margin:4px 0;background:rgba(255,255,255,.02);border-radius:8px}
.agent-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}
.agent-dot.MONITORING{background:var(--green);box-shadow:0 0 6px var(--green)}
.agent-dot.PROCESSING{background:var(--accent);box-shadow:0 0 6px var(--accent);animation:pulse 1.5s infinite}
.agent-dot.IDLE{background:var(--dim)}
.agent-dot.STANDBY{background:var(--yellow);box-shadow:0 0 6px var(--yellow)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.agent-name{font-weight:600;font-size:.88em;flex:1}
.agent-cap{font-size:.72em;color:var(--dim)}
.agent-tasks{font-family:'JetBrains Mono',monospace;font-size:.75em;color:var(--accent)}

/* Compliance bar */
.comp-fw{margin:10px 0}
.comp-fw-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}
.comp-fw-name{font-weight:600;font-size:.9em}
.comp-fw-score{font-family:'JetBrains Mono',monospace;font-weight:700;font-size:.95em}
.comp-bar{height:8px;background:rgba(255,255,255,.06);border-radius:4px;overflow:hidden}
.comp-bar-fill{height:100%;border-radius:4px;transition:width .8s ease}
.comp-findings{margin-top:6px;padding-left:12px}
.comp-findings li{font-size:.8em;color:var(--dim);margin:3px 0;list-style:disc}

/* Phase timeline */
.phase{display:flex;align-items:flex-start;gap:14px;padding:14px 16px;margin:8px 0;background:rgba(255,255,255,.02);border:1px solid var(--border);border-radius:10px;border-left:3px solid var(--accent)}
.phase-num{min-width:30px;height:30px;border-radius:50%;background:rgba(0,217,255,.12);color:var(--accent);display:flex;align-items:center;justify-content:center;font-weight:700;font-size:.82em}
.phase-info h4{font-size:.95em}.phase-info p{font-size:.82em;color:var(--dim);margin-top:2px}
.phase-dur{color:var(--green);font-weight:600}

/* Loader */
.loader{display:inline-block;width:16px;height:16px;border:2px solid rgba(0,217,255,.2);border-top-color:var(--accent);border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle;margin-right:6px}
@keyframes spin{to{transform:rotate(360deg)}}

/* Before/After */
.ba{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:12px 0}
.ba-col{background:#000;border-radius:8px;padding:14px}
.ba-col h5{font-size:.8em;font-weight:700;margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em}
.ba-col.before h5{color:var(--red)}.ba-col.after h5{color:var(--green)}
.ba-col pre{font-family:'JetBrains Mono',monospace;font-size:.72em;line-height:1.6;white-space:pre-wrap;word-break:break-all}

/* Scrollbar */
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:rgba(255,255,255,.1);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:rgba(255,255,255,.2)}
</style>
</head>
<body>
<div class="app">
  <div class="sidebar">
    <div class="sidebar-header">
      <h1>QBITEL Bridge</h1>
      <p>End-to-End Mainframe Modernization</p>
    </div>
    <div class="steps" id="steps"></div>
    <div class="sidebar-footer">Steps completed: <span id="done-count">0</span>/8</div>
  </div>
  <div class="main">
    <div class="topbar">
      <span class="tag" id="step-tag">Step 1 of 8</span>
      <span id="step-title" style="font-weight:600;font-size:.9em;"></span>
      <span class="timer" id="timer">0:00</span>
    </div>
    <div class="content" id="content"></div>
  </div>
</div>

<script>
const STEPS=[
  {n:1,title:"Network Discovery",desc:"Scan and discover legacy systems",icon:"1"},
  {n:2,title:"Traffic Capture & Protocol Analysis",desc:"Capture and decode mainframe traffic",icon:"2"},
  {n:3,title:"COBOL Deep Analysis",desc:"AI-powered code analysis",icon:"3"},
  {n:4,title:"PQC Security Layer",desc:"Quantum-safe encryption",icon:"4"},
  {n:5,title:"Code Generation",desc:"COBOL to modern Python/FastAPI",icon:"5"},
  {n:6,title:"Security Monitoring",desc:"Live threat detection dashboard",icon:"6"},
  {n:7,title:"Compliance Report",desc:"Automated regulatory assessment",icon:"7"},
  {n:8,title:"Modernization Roadmap",desc:"Phased migration plan",icon:"8"}
];
let cur=1,done=new Set(),timerStart=Date.now(),timerInt=null,monitorInt=null;

// Build sidebar
const stepsEl=document.getElementById('steps');
STEPS.forEach(s=>{
  const d=document.createElement('div');
  d.className='step'+(s.n===1?' active':'');
  d.id='s'+s.n;
  d.innerHTML=`<div class="step-num">${s.n}</div><div class="step-text"><h4>${s.title}</h4><p>${s.desc}</p></div>`;
  d.onclick=()=>goStep(s.n);
  stepsEl.appendChild(d);
});

// Timer
timerInt=setInterval(()=>{
  const s=Math.floor((Date.now()-timerStart)/1000);
  document.getElementById('timer').textContent=Math.floor(s/60)+':'+String(s%60).padStart(2,'0');
},1000);

function goStep(n){
  if(monitorInt){clearInterval(monitorInt);monitorInt=null;}
  cur=n;
  document.querySelectorAll('.step').forEach((el,i)=>{
    el.classList.remove('active');
    if(i+1===n)el.classList.add('active');
    if(done.has(i+1))el.classList.add('done');
  });
  document.getElementById('step-tag').textContent=`Step ${n} of 8`;
  document.getElementById('step-title').textContent=STEPS[n-1].title;
  render(n);
}

function markDone(n){done.add(n);document.getElementById('s'+n).classList.add('done');document.getElementById('done-count').textContent=done.size;}
function showOut(id){document.getElementById(id).classList.add('show');}
function h(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
async function api(path,opts){const r=await fetch(path,opts);return r.json();}

// ========== STEP RENDERERS ==========

function render(n){
  const c=document.getElementById('content');
  switch(n){
    case 1: renderDiscovery(c);break;
    case 2: renderCapture(c);break;
    case 3: renderCobol(c);break;
    case 4: renderPQC(c);break;
    case 5: renderCodeGen(c);break;
    case 6: renderMonitor(c);break;
    case 7: renderCompliance(c);break;
    case 8: renderRoadmap(c);break;
  }
}

// --- Step 1: Network Discovery ---
function renderDiscovery(c){
  c.innerHTML=`<div class="panel"><h2>Network Discovery</h2><p class="desc">QBITEL passively scans the network to discover legacy mainframe systems, protocols, and unencrypted channels — in minutes, not months.</p><button class="btn btn-primary" onclick="runScan()">Scan Network</button><div class="out" id="out1"></div></div>`;
}
async function runScan(){
  const o=document.getElementById('out1');showOut('out1');
  o.innerHTML=`<div class="term" id="term1"></div>`;
  const t=document.getElementById('term1');
  const lines=['<span class="d">[QBITEL] Initializing passive network scan...</span>','<span class="d">[QBITEL] Target: 10.1.50.0/24</span>','<span class="c">[SCAN] Sending ARP probes...</span>',''];
  for(let l of lines){t.innerHTML+=l+'\\n';await sleep(300);}
  const d=await api('/api/e2e/network/scan');
  for(let r of d.results){
    t.innerHTML+=`<span class="g">[FOUND]</span> ${r.ip}:<span class="c">${r.port}</span> — <span class="y">${r.protocol}</span> → ${r.name} <span class="r">[${r.encryption}]</span>\\n`;
    await sleep(250);
  }
  t.innerHTML+='\\n<span class="g">[COMPLETE]</span> Scan finished in <span class="c">'+d.duration_seconds+'s</span>\\n';
  t.innerHTML+=`<span class="o">[WARNING]</span> <span class="r">${d.unencrypted_channels} unencrypted channels detected!</span>\\n`;
  // Summary cards
  o.innerHTML+=`<div class="dash-grid" style="margin-top:14px"><div class="dash-card"><div class="num">${d.unique_systems}</div><div class="lbl">Legacy Systems</div></div><div class="dash-card"><div class="num">${d.unique_protocols}</div><div class="lbl">Protocols</div></div><div class="dash-card"><div class="num">${d.targets_found}</div><div class="lbl">Channels Found</div></div><div class="dash-card"><div class="num" style="-webkit-text-fill-color:var(--red)">${d.unencrypted_channels}</div><div class="lbl">Unencrypted</div></div></div>`;
  o.innerHTML+=`<div style="margin-top:12px;text-align:right"><button class="btn btn-primary" onclick="markDone(1);goStep(2)">Next: Capture Traffic &rarr;</button></div>`;
}

// --- Step 2: Traffic Capture & Protocol ---
function renderCapture(c){
  c.innerHTML=`<div class="panel"><h2>Traffic Capture & Protocol Analysis</h2><p class="desc">Capture live mainframe traffic and reverse-engineer binary protocols — EBCDIC encoding, field boundaries, data types — fully automated.</p><button class="btn btn-primary" onclick="runCapture()">Capture & Analyze</button><div class="out" id="out2"></div></div>`;
}
async function runCapture(){
  const o=document.getElementById('out2');showOut('out2');
  o.innerHTML='<span class="loader"></span> Capturing mainframe traffic...';
  const d=await api('/api/e2e/network/capture');
  let html=`<div class="term" style="margin-bottom:14px">`;
  html+=`<span class="g">[CAPTURE]</span> ${d.packets_captured} packets captured in <span class="c">${d.duration_ms}ms</span>\\n`;
  html+=`<span class="o">[ALERT]</span> <span class="r">${d.pii_exposure} packets contain exposed PII</span>\\n\\n`;
  for(let p of d.packets){
    html+=`<span class="d">--- Packet #${p.packet_id}: ${p.label} (${p.protocol}, ${p.length}B) ---</span>\\n`;
    html+=`<span class="d">  ${p.src} → ${p.dst}</span>\\n`;
    html+=`  <span class="y">HEX:</span>  <span class="c">${p.hex_dump}</span>\\n`;
    html+=`  <span class="y">ASCII:</span> <span class="g">${h(p.decoded_ascii)}</span>\\n`;
    if(p.pii_detected) html+=`  <span class="r">[PII DETECTED] Sensitive data in plaintext!</span>\\n`;
    html+='\\n';
  }
  html+=`</div>`;
  // Now run protocol analysis
  const pa=await api('/api/analyze/protocol',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({protocol_data:d.packets[0].hex_dump,system_context:'IBM z/OS Mainframe'})});
  html+=`<div class="card"><h4>Protocol Analysis Results</h4><div class="metrics"><span class="metric"><span class="l">Encoding </span><span class="v">${pa.encoding}</span></span><span class="metric"><span class="l">Structure </span><span class="v">${pa.structure.type}</span></span><span class="metric"><span class="l">Length </span><span class="v">${pa.raw_length}B</span></span></div>`;
  if(pa.fields&&pa.fields.length){
    html+=`<table style="width:100%;font-size:.82em;margin-top:8px;border-collapse:collapse"><tr style="color:var(--dim);text-align:left"><th style="padding:6px 8px">Field</th><th>Offset</th><th>Length</th><th>Type</th></tr>`;
    pa.fields.forEach(f=>{html+=`<tr style="border-top:1px solid var(--border)"><td style="padding:6px 8px;color:var(--accent)">${f.name}</td><td>${f.offset}</td><td>${f.length}B</td><td>${f.type}</td></tr>`;});
    html+=`</table>`;
  }
  if(pa.recommendations&&pa.recommendations.length){
    html+=`<div style="margin-top:12px">`;
    pa.recommendations.forEach(r=>{html+=`<div class="pat good"><strong>${r.issue}</strong><br><span style="color:var(--green)">${r.solution}</span></div>`;});
    html+=`</div>`;
  }
  html+=`</div>`;
  html+=`<div style="margin-top:12px;text-align:right"><button class="btn btn-primary" onclick="markDone(2);goStep(3)">Next: COBOL Analysis &rarr;</button></div>`;
  o.innerHTML=html;
}

// --- Step 3: COBOL Deep Analysis ---
function renderCobol(c){
  c.innerHTML=`<div class="panel"><h2>COBOL Deep Analysis</h2><p class="desc">AI-powered analysis of legacy COBOL source code — complexity scoring, legacy pattern detection, and modernization opportunity mapping.</p><button class="btn btn-primary" onclick="runCobol()">Analyze COBOL Programs</button><div class="out" id="out3"></div></div>`;
}
async function runCobol(){
  const o=document.getElementById('out3');showOut('out3');
  o.innerHTML='<span class="loader"></span> Analyzing COBOL source files...';
  const list=await api('/api/analyze/cobol/list');
  let html='';
  for(let f of list.files){
    const a=await api('/api/analyze/cobol/'+f.name);
    html+=`<div class="card"><h4>${a.name} <span style="color:var(--dim);font-weight:400">(${a.lines_of_code} LOC)</span></h4>`;
    html+=`<div class="metrics"><span class="metric"><span class="l">Complexity </span><span class="v">${a.complexity_score}</span></span><span class="metric"><span class="l">Data Divisions </span><span class="v">${a.data_divisions}</span></span><span class="metric"><span class="l">Procedures </span><span class="v">${a.procedure_divisions}</span></span><span class="metric"><span class="l">Variables </span><span class="v">${a.analysis.working_storage?.variable_count||0}</span></span></div>`;
    if(a.analysis.legacy_patterns?.length){
      html+=`<div style="margin-top:10px"><strong style="font-size:.85em;color:var(--dim)">Legacy Patterns Detected</strong>`;
      a.analysis.legacy_patterns.forEach(p=>{
        const cls=p.severity==='critical'?'critical':p.severity==='high'?'high':'warning';
        html+=`<div class="pat ${cls}"><span class="sev sev-${p.severity}">${p.severity}</span> <strong style="margin-left:6px">${p.pattern}</strong><br><span style="color:var(--dim);font-size:.85em">${p.description}</span></div>`;
      });
      html+=`</div>`;
    }
    if(a.analysis.modernization_opportunities?.length){
      html+=`<div style="margin-top:10px"><strong style="font-size:.85em;color:var(--dim)">Modernization Opportunities</strong>`;
      a.analysis.modernization_opportunities.forEach(op=>{
        html+=`<div class="pat good"><strong>${op.area}</strong><br><span style="color:var(--dim);font-size:.85em">${op.current}</span> &rarr; <span style="color:var(--green);font-size:.85em">${op.modern}</span></div>`;
      });
      html+=`</div>`;
    }
    html+=`</div>`;
  }
  html+=`<div style="margin-top:12px;text-align:right"><button class="btn btn-primary" onclick="markDone(3);goStep(4)">Next: PQC Security &rarr;</button></div>`;
  o.innerHTML=html;
}

// --- Step 4: PQC Security Layer ---
function renderPQC(c){
  c.innerHTML=`<div class="panel"><h2>Post-Quantum Cryptographic Protection</h2><p class="desc">Wrap all mainframe communications in NIST Level 3 quantum-safe encryption — zero code changes, &lt;1ms overhead. Protect against harvest-now-decrypt-later attacks.</p><button class="btn btn-primary" onclick="runPQC()">Apply PQC Protection</button><div class="out" id="out4"></div></div>`;
}
async function runPQC(){
  const o=document.getElementById('out4');showOut('out4');
  o.innerHTML='<span class="loader"></span> Generating quantum-safe keypair...';
  // 1. Keygen
  const kg=await api('/api/e2e/pqc/keygen');
  let html=`<div class="card"><h4>Key Generation — ${kg.algorithm}</h4><div class="metrics"><span class="metric"><span class="l">KEM Public Key </span><span class="v">${kg.kem.public_key_bytes}B</span></span><span class="metric"><span class="l">KEM Ciphertext </span><span class="v">${kg.kem.ciphertext}B</span></span><span class="metric"><span class="l">Signer Public Key </span><span class="v">${kg.signer.public_key_bytes}B</span></span><span class="metric"><span class="l">Signature </span><span class="v">${kg.signer.signature}B</span></span><span class="metric"><span class="l">Latency </span><span class="v">${kg.generation_latency_ms}ms</span></span><span class="metric"><span class="l">NIST Level </span><span class="v">${kg.nist_security_level}</span></span></div></div>`;
  // 2. Encrypt
  const enc=await api('/api/e2e/pqc/encrypt',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({})});
  html+=`<div class="card"><h4>Transaction Encryption — SWIFT MT103 Wire Transfer</h4><div class="ba"><div class="ba-col before"><h5>Before (Plaintext)</h5><pre>${h(enc.plaintext)}</pre></div><div class="ba-col after"><h5>After (PQC Encrypted)</h5><pre>${enc.ciphertext_hex}</pre></div></div><div class="metrics"><span class="metric"><span class="l">Plain </span><span class="v">${enc.plaintext_bytes}B</span></span><span class="metric"><span class="l">Cipher </span><span class="v">${enc.ciphertext_bytes}B</span></span><span class="metric"><span class="l">Overhead </span><span class="v">${enc.overhead_percent}%</span></span><span class="metric"><span class="l">Latency </span><span class="v">${enc.encryption_latency_ms}ms</span></span></div></div>`;
  // 3. Sign
  const sig=await api('/api/e2e/pqc/sign',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({})});
  html+=`<div class="card"><h4>Digital Signature — ${sig.algorithm}</h4><div class="metrics"><span class="metric"><span class="l">Message Hash </span><span class="v" style="font-family:'JetBrains Mono',monospace;font-size:.8em">${sig.message_hash.substring(0,24)}...</span></span><span class="metric"><span class="l">Signature </span><span class="v">${sig.signature_bytes}B</span></span><span class="metric"><span class="l">Sign Latency </span><span class="v">${sig.signing_latency_ms}ms</span></span><span class="metric"><span class="l">Verify Latency </span><span class="v">${sig.verification_latency_ms}ms</span></span><span class="metric"><span class="l">Verified </span><span class="v" style="color:var(--green)">TRUE</span></span></div></div>`;
  html+=`<div style="margin-top:12px;text-align:right"><button class="btn btn-primary" onclick="markDone(4);goStep(5)">Next: Code Generation &rarr;</button></div>`;
  o.innerHTML=html;
}

// --- Step 5: Code Generation ---
let genCode={};
function renderCodeGen(c){
  c.innerHTML=`<div class="panel"><h2>Modern Code Generation</h2><p class="desc">Auto-generate production-ready Python dataclasses, FastAPI endpoints, and SQL schemas from COBOL analysis. 100% data fidelity with original structures.</p><button class="btn btn-primary" onclick="runGen()">Generate Modern Code</button><div class="out" id="out5"></div></div>`;
}
async function runGen(){
  const o=document.getElementById('out5');showOut('out5');
  o.innerHTML='<span class="loader"></span> Transforming COBOL to Python/FastAPI...';
  const list=await api('/api/analyze/cobol/list');
  if(!list.files.length){o.innerHTML='No COBOL files found.';return;}
  const d=await api('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cobol_file:list.files[0].name})});
  genCode=d.generated_code||{};
  let html=`<div class="code-tabs"><div class="ctab active" onclick="switchGenTab(this,'models')">Python Models</div><div class="ctab" onclick="switchGenTab(this,'api')">FastAPI Endpoints</div><div class="ctab" onclick="switchGenTab(this,'sql')">SQL Schema</div></div><pre class="cblock"><code id="gen-code">${h(genCode.python_models||'')}</code></pre>`;
  html+=`<div style="margin-top:12px;text-align:right"><button class="btn btn-primary" onclick="markDone(5);goStep(6)">Next: Security Monitor &rarr;</button></div>`;
  o.innerHTML=html;
}
function switchGenTab(el,tab){
  document.querySelectorAll('.ctab').forEach(t=>t.classList.remove('active'));
  el.classList.add('active');
  const d=document.getElementById('gen-code');
  if(tab==='models')d.textContent=genCode.python_models||'';
  else if(tab==='api')d.textContent=genCode.fastapi_endpoints||'';
  else d.textContent=genCode.sql_schema||'';
}

// --- Step 6: Security Monitoring ---
function renderMonitor(c){
  c.innerHTML=`<div class="panel"><h2>Live Security Monitoring</h2><p class="desc">Real-time threat detection, PQC operations monitoring, and autonomous agent orchestration — 78% autonomous response rate, &lt;1 second decision time.</p><button class="btn btn-primary" onclick="startMonitor()">Start Monitoring</button><div class="out" id="out6"></div></div>`;
}
async function startMonitor(){
  const o=document.getElementById('out6');showOut('out6');
  o.innerHTML='<span class="loader"></span> Connecting to security fabric...';
  await renderMonitorData(o);
  monitorInt=setInterval(()=>refreshMetrics(),3000);
}
async function renderMonitorData(o){
  const[m,t,a]=await Promise.all([api('/api/e2e/security/metrics'),api('/api/e2e/security/threats'),api('/api/e2e/security/agents')]);
  let html=`<div class="dash-grid" id="dash-metrics"><div class="dash-card"><div class="num" id="m-tx">${(m.transactions_today).toLocaleString()}</div><div class="lbl">Transactions Today</div></div><div class="dash-card"><div class="num" id="m-pqc">${(m.pqc_encryptions).toLocaleString()}</div><div class="lbl">PQC Encryptions</div></div><div class="dash-card"><div class="num" id="m-threats">${m.threats_blocked_today}</div><div class="lbl">Threats Blocked</div></div><div class="dash-card"><div class="num" id="m-lat">${m.avg_encryption_latency_ms}ms</div><div class="lbl">Avg PQC Latency</div></div><div class="dash-card"><div class="num" id="m-auto">78%</div><div class="lbl">Autonomous Response</div></div><div class="dash-card"><div class="num" id="m-kafka">${(m.kafka_throughput_msg_sec).toLocaleString()}</div><div class="lbl">Kafka msg/sec</div></div></div>`;
  // Agents
  html+=`<div class="card" style="margin-top:14px"><h4>Agent Orchestration (5 Active)</h4><div id="agents-panel">`;
  a.forEach(ag=>{html+=`<div class="agent"><div class="agent-dot ${ag.status}"></div><div class="agent-name">${ag.name}</div><div class="agent-cap">${ag.capability}</div><div class="agent-tasks">${ag.tasks_completed.toLocaleString()} tasks</div></div>`;});
  html+=`</div></div>`;
  // Threats
  html+=`<div class="card" style="margin-top:14px"><h4>Threat Events (Last 2 Hours)</h4><div id="threats-panel">`;
  t.forEach(ev=>{
    const ts=new Date(ev.timestamp);
    const tstr=ts.getHours()+':'+String(ts.getMinutes()).padStart(2,'0');
    html+=`<div class="threat ${ev.action}"><div class="threat-time">${tstr}</div><div class="threat-info"><strong><span class="sev sev-${ev.severity.toLowerCase()}">${ev.severity}</span> ${ev.type}</strong><p>${ev.detail}</p><p style="color:var(--accent)">Action: ${ev.action} | Response: ${ev.response_ms}ms | ${ev.source} &rarr; ${ev.target}</p></div></div>`;
  });
  html+=`</div></div>`;
  html+=`<div style="margin-top:12px;text-align:right"><button class="btn btn-primary" onclick="markDone(6);goStep(7)">Next: Compliance &rarr;</button></div>`;
  o.innerHTML=html;
}
async function refreshMetrics(){
  try{
    const m=await api('/api/e2e/security/metrics');
    const el=id=>document.getElementById(id);
    if(el('m-tx'))el('m-tx').textContent=m.transactions_today.toLocaleString();
    if(el('m-pqc'))el('m-pqc').textContent=m.pqc_encryptions.toLocaleString();
    if(el('m-threats'))el('m-threats').textContent=m.threats_blocked_today;
    if(el('m-lat'))el('m-lat').textContent=m.avg_encryption_latency_ms+'ms';
    if(el('m-kafka'))el('m-kafka').textContent=m.kafka_throughput_msg_sec.toLocaleString();
  }catch(e){}
}

// --- Step 7: Compliance Report ---
function renderCompliance(c){
  c.innerHTML=`<div class="panel"><h2>Compliance & Audit Report</h2><p class="desc">Automated compliance assessment across 5 regulatory frameworks. Generate audit-ready evidence in under 10 minutes — versus 2-4 weeks manually.</p><button class="btn btn-primary" onclick="runCompliance()">Generate Report</button><div class="out" id="out7"></div></div>`;
}
async function runCompliance(){
  const o=document.getElementById('out7');showOut('out7');
  o.innerHTML='<span class="loader"></span> Running compliance assessment...';
  const[rpt,aud]=await Promise.all([api('/api/e2e/compliance/report'),api('/api/e2e/compliance/audit')]);
  let html=`<div class="dash-grid" style="margin-bottom:14px"><div class="dash-card"><div class="num">${rpt.overall_score}%</div><div class="lbl">Overall Score</div></div><div class="dash-card"><div class="num" style="-webkit-text-fill-color:var(--green)">${rpt.overall_status}</div><div class="lbl">Status</div></div><div class="dash-card"><div class="num">${rpt.frameworks.length}</div><div class="lbl">Frameworks</div></div><div class="dash-card"><div class="num">NIST ${rpt.pqc_readiness.nist_level}</div><div class="lbl">PQC Level</div></div></div>`;
  // Frameworks
  rpt.frameworks.forEach(fw=>{
    const color=fw.score>=95?'var(--green)':fw.score>=90?'var(--accent)':fw.score>=80?'var(--yellow)':'var(--red)';
    html+=`<div class="comp-fw"><div class="comp-fw-header"><span class="comp-fw-name">${fw.name} <span class="sev sev-${fw.status==='COMPLIANT'?'low':'medium'}">${fw.status}</span></span><span class="comp-fw-score" style="color:${color}">${fw.score}/100</span></div><div class="comp-bar"><div class="comp-bar-fill" style="width:${fw.score}%;background:${color}"></div></div><div style="font-size:.78em;color:var(--dim);margin-top:3px">${fw.controls_passed}/${fw.controls_total} controls passed, ${fw.controls_failed} failed, ${fw.controls_na} N/A</div><ul class="comp-findings">${fw.findings.map(f=>'<li>'+f+'</li>').join('')}</ul></div>`;
  });
  // Audit Trail
  html+=`<div class="card" style="margin-top:14px"><h4>Recent Audit Trail</h4>`;
  aud.slice(0,6).forEach(e=>{
    const ts=new Date(e.timestamp);
    const tstr=ts.getHours()+':'+String(ts.getMinutes()).padStart(2,'0');
    const col=e.result==='SUCCESS'?'var(--green)':e.result==='REVIEW'?'var(--orange)':'var(--red)';
    html+=`<div style="display:flex;gap:10px;padding:8px 0;border-bottom:1px solid var(--border);font-size:.82em"><span style="color:var(--dim);min-width:45px;font-family:'JetBrains Mono',monospace">${tstr}</span><span style="color:var(--accent);min-width:130px">${e.action}</span><span style="flex:1;color:var(--dim)">${e.detail}</span><span style="color:${col};font-weight:600">${e.result}</span></div>`;
  });
  html+=`</div>`;
  html+=`<div style="margin-top:12px;text-align:right"><button class="btn btn-primary" onclick="markDone(7);goStep(8)">Next: Roadmap &rarr;</button></div>`;
  o.innerHTML=html;
}

// --- Step 8: Modernization Roadmap ---
function renderRoadmap(c){
  c.innerHTML=`<div class="panel"><h2>Modernization Roadmap</h2><p class="desc">Generate a comprehensive, phased modernization plan with risk assessment, effort estimation, and deliverables — board-ready.</p><div style="margin-bottom:14px"><select id="approach" style="padding:8px 14px;background:var(--bg3);color:var(--text);border:1px solid var(--border);border-radius:6px;font-family:inherit;font-size:.88em;margin-right:8px"><option value="refactor">Refactor (Transform Code)</option><option value="replatform">Replatform (Cloud Migration)</option><option value="rearchitect">Rearchitect (Redesign)</option></select><button class="btn btn-primary" onclick="runRoadmap()">Generate Roadmap</button></div><div class="out" id="out8"></div></div>`;
}
async function runRoadmap(){
  const o=document.getElementById('out8');showOut('out8');
  const approach=document.getElementById('approach').value;
  o.innerHTML='<span class="loader"></span> Generating modernization roadmap...';
  const p=await api('/api/modernize',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({system_id:'SYS001',approach:approach,target_language:'python',target_framework:'fastapi'})});
  let html=`<div class="dash-grid" style="margin-bottom:14px"><div class="dash-card"><div class="num">${p.approach.toUpperCase()}</div><div class="lbl">Approach</div></div><div class="dash-card"><div class="num" style="-webkit-text-fill-color:var(--orange)">${p.risk_level.toUpperCase()}</div><div class="lbl">Risk Level</div></div><div class="dash-card"><div class="num">${p.estimated_effort_days}</div><div class="lbl">Person-Days</div></div><div class="dash-card"><div class="num">${p.phases.length}</div><div class="lbl">Phases</div></div></div>`;
  let totalWeeks=0;
  p.phases.forEach(ph=>{
    totalWeeks+=ph.duration_weeks;
    html+=`<div class="phase"><div class="phase-num">${ph.phase}</div><div class="phase-info"><h4>${ph.name} <span class="phase-dur">(${ph.duration_weeks} weeks)</span></h4><p>${ph.description}</p></div></div>`;
  });
  html+=`<div style="margin-top:14px;padding:14px;background:rgba(0,255,136,.06);border-radius:8px;text-align:center;font-weight:700;color:var(--green)">Total Duration: ${totalWeeks} weeks &bull; Estimated Go-Live: ${getGoLiveDate(totalWeeks)}</div>`;
  html+=`<div style="margin-top:14px;text-align:center"><button class="btn btn-primary" onclick="markDone(8);showSummary()" style="padding:14px 40px;font-size:1em">Complete Demo</button></div>`;
  o.innerHTML=html;
}
function getGoLiveDate(weeks){const d=new Date();d.setDate(d.getDate()+weeks*7);return d.toLocaleDateString('en-US',{year:'numeric',month:'long',day:'numeric'});}

function showSummary(){
  const c=document.getElementById('content');
  c.innerHTML=`<div style="text-align:center;padding:60px 40px"><h2 style="font-size:2em;margin-bottom:12px"><span style="background:linear-gradient(135deg,var(--accent),var(--green));-webkit-background-clip:text;-webkit-text-fill-color:transparent">Demo Complete</span></h2><p style="color:var(--dim);font-size:1.05em;margin-bottom:32px;max-width:600px;margin-left:auto;margin-right:auto">You've seen the full QBITEL Bridge mainframe modernization journey — from network discovery to quantum-safe protection to automated compliance.</p><div class="dash-grid" style="max-width:800px;margin:0 auto 32px"><div class="dash-card"><div class="num">8/8</div><div class="lbl">Steps Completed</div></div><div class="dash-card"><div class="num">${document.getElementById('timer').textContent}</div><div class="lbl">Demo Duration</div></div><div class="dash-card"><div class="num">3</div><div class="lbl">Systems Protected</div></div><div class="dash-card"><div class="num">NIST 3</div><div class="lbl">PQC Security Level</div></div></div><div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-bottom:24px"><span style="padding:8px 18px;background:rgba(255,255,255,.04);border:1px solid var(--border);border-radius:20px;font-size:.88em;color:var(--dim)">100% Open Source</span><span style="padding:8px 18px;background:rgba(255,255,255,.04);border:1px solid var(--border);border-radius:20px;font-size:.88em;color:var(--dim)">Air-Gapped Ready</span><span style="padding:8px 18px;background:rgba(255,255,255,.04);border:1px solid var(--border);border-radius:20px;font-size:.88em;color:var(--dim)">Zero Code Changes</span><span style="padding:8px 18px;background:rgba(255,255,255,.04);border:1px solid var(--border);border-radius:20px;font-size:.88em;color:var(--dim)">Apache 2.0 License</span></div><p style="color:var(--dim);font-size:.95em">Contact: <strong style="color:var(--accent)">enterprise@qbitel.com</strong></p></div>`;
}

function sleep(ms){return new Promise(r=>setTimeout(r,ms));}

// Init
goStep(1);
</script>
</body>
</html>"""


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
