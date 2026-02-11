# UC1: Legacy Mainframe Modernization Demo

A complete working demonstration of QBITEL's Legacy Mainframe Modernization capabilities.

## Overview

This demo showcases AI-powered modernization of legacy IBM mainframe COBOL systems, including:

- **Legacy System Discovery** - Inventory and assess legacy mainframe systems
- **COBOL Code Analysis** - Deep analysis of COBOL source code with complexity scoring
- **Protocol Reverse Engineering** - Decode EBCDIC and mainframe binary formats
- **Modern Code Generation** - Auto-generate Python/FastAPI from COBOL
- **Modernization Planning** - Risk assessment and project planning

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
```

### Run Options

**1. Web UI (Recommended)**
```bash
python run_demo.py --server
# Open: http://localhost:8001
```

**2. Automated Demo**
```bash
python run_demo.py --auto
```

**3. Interactive CLI**
```bash
python run_demo.py
```

## Demo Contents

### Sample COBOL Programs

- `cobol_samples/CUSTMAST.cbl` - Customer Master File Maintenance (196 LOC)
- `cobol_samples/ACCTPROC.cbl` - Account Processing Batch (390 LOC)

### Sample Data

- `data/sample_customer_records.dat` - Fixed-width customer records
- `data/sample_transactions.dat` - Transaction records

### Simulated Legacy Systems

| System | Description | LOC | Age |
|--------|-------------|-----|-----|
| SYS001 | Core Banking System | 2.5M | 38 years |
| SYS002 | Customer Master System | 850K | 32 years |
| SYS003 | Account Processing Batch | 1.2M | 29 years |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/systems` | GET | List legacy systems |
| `/api/systems/{id}` | GET | Get system details |
| `/api/analyze/cobol/list` | GET | List COBOL files |
| `/api/analyze/cobol/{file}` | GET | Analyze COBOL file |
| `/api/analyze/protocol` | POST | Analyze protocol data |
| `/api/modernize` | POST | Create modernization plan |
| `/api/generate` | POST | Generate modern code |
| `/demo` | GET | Interactive demo UI |

## Features Demonstrated

### 1. COBOL Analysis

- Working storage analysis
- Procedure division parsing
- Control flow detection
- Legacy pattern identification
- Complexity scoring

### 2. Protocol Analysis

- EBCDIC encoding detection
- Fixed-length record parsing
- Field boundary detection
- Conversion code generation

### 3. Code Generation

- Python dataclass models
- FastAPI endpoints
- SQL schema generation
- Documentation

### 4. Modernization Planning

- Multi-phase project plans
- Risk assessment
- Effort estimation
- Generated documentation

## Architecture

```
UC1_Legacy_Mainframe_Modernization/
├── backend/
│   └── app.py              # FastAPI application
├── cobol_samples/
│   ├── CUSTMAST.cbl        # Customer master program
│   └── ACCTPROC.cbl        # Account processing program
├── data/
│   ├── sample_customer_records.dat
│   └── sample_transactions.dat
├── output/                  # Generated artifacts
├── run_demo.py             # Demo runner
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Sample Output

### COBOL Analysis
```json
{
  "program_id": "CUSTMAST",
  "lines_of_code": 196,
  "complexity_score": 17.0,
  "modernization_opportunities": [
    {
      "area": "file_processing",
      "current": "Sequential/Indexed file I/O",
      "modern": "Modern file formats (JSON, Parquet) or databases"
    }
  ]
}
```

### Generated Python Model
```python
@dataclass
class CustomerRecord:
    """Generated from COBOL record: CUSTOMER-RECORD"""
    cust_id: Optional[int] = None
    cust_first_name: Optional[str] = None
    cust_last_name: Optional[str] = None
    cust_balance: Optional[Decimal] = None
    cust_status: Optional[str] = None
```

## Integration with QBITEL

This demo integrates with the full QBITEL platform:

- `ai_engine/legacy/` - Legacy System Whisperer service
- `ai_engine/llm/legacy_whisperer.py` - LLM-powered analysis
- `ai_engine/llm/unified_llm_service.py` - LLM orchestration

## License

Part of QBITEL Enterprise Platform. All rights reserved.
