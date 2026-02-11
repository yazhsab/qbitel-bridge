# QBITEL - Agent Instructions

This document provides context for AI coding agents working on the QBITEL codebase.

## Quick Start

```bash
# View all beads (memory items)
cat .beads/issues.jsonl | jq -s '.' | head -100

# Find specific topics
grep -i "legacy" .beads/issues.jsonl | jq '.'
grep -i "llm" .beads/issues.jsonl | jq '.'
```

## Project Overview

QBITEL is an enterprise-grade quantum-safe security platform with:
- **AI-powered protocol discovery** - Automatic protocol reverse engineering
- **Legacy system modernization** - COBOL/mainframe transformation
- **Post-quantum cryptography** - NIST Level 5 compliance
- **Cloud-native security** - Service mesh, container security

## Key Memory Beads

| Bead ID | Topic | Priority |
|---------|-------|----------|
| qbitel-0001 | Architecture Overview | P0 |
| qbitel-0017 | UC1: Legacy Mainframe Modernization | P0 |
| qbitel-0019 | Integration Architecture | P1 |
| qbitel-0026 | TODO: Integrate Demo with Production | P1 |

## Module Map

### Core AI Engine (`ai_engine/`)

```
ai_engine/
├── legacy/           # Legacy System Whisperer (~17K LOC)
│   ├── service.py              # Main orchestrator
│   ├── enhanced_detector.py    # VAE + LLM anomaly detection
│   ├── predictive_analytics.py # ML failure prediction
│   ├── decision_support.py     # LLM recommendations
│   └── knowledge_capture.py    # Tribal knowledge
│
├── llm/              # LLM Integration (~15K LOC)
│   ├── unified_llm_service.py  # Multi-provider LLM
│   ├── legacy_whisperer.py     # Protocol analysis
│   ├── rag_engine.py           # Hybrid search
│   └── translation_studio.py   # Protocol translation
│
├── discovery/        # Protocol Discovery (~62K LOC)
│   └── protocol_discovery_orchestrator.py  # PCFG learning
│
├── detection/        # Field Detection (~45K LOC)
│   └── field_detector.py       # BiLSTM-CRF model
│
├── cloud_native/     # Cloud Security
├── domains/          # Domain-specific (auto, aviation, healthcare)
└── agents/           # Autonomous agents
```

### Use Case Demos (`demos/`)

```
demos/
└── UC1_Legacy_Mainframe_Modernization/
    ├── backend/app.py          # FastAPI demo
    ├── cobol_samples/          # COBOL examples
    ├── run_demo.py             # Demo runner
    └── README.md
```

## Production vs Demo Code

**IMPORTANT**: The UC1 demo is an MVP that should be enhanced to use production modules.

| Feature | Demo (MVP) | Production Module |
|---------|------------|-------------------|
| COBOL Analysis | Basic regex | `ai_engine/legacy/service.py` |
| Protocol Analysis | Heuristics | `ai_engine/llm/legacy_whisperer.py` |
| Code Generation | Templates | LLM-powered with validation |
| Knowledge Capture | None | `ai_engine/legacy/knowledge_capture.py` |
| Failure Prediction | None | `ai_engine/legacy/predictive_analytics.py` |

## Key Classes

### LegacySystemWhispererService
```python
from ai_engine.legacy.service import LegacySystemWhispererService

service = LegacySystemWhispererService()
await service.initialize()
await service.register_legacy_system(system_context)
health = await service.analyze_system_health(system_id)
```

### UnifiedLLMService
```python
from ai_engine.llm.unified_llm_service import UnifiedLLMService

llm = UnifiedLLMService()
response = await llm.process_request(prompt, model="claude-sonnet")
```

### LegacyWhisperer
```python
from ai_engine.llm.legacy_whisperer import LegacyWhisperer

whisperer = LegacyWhisperer(llm_service)
spec = await whisperer.reverse_engineer_protocol(samples, context)
code = await whisperer.generate_adapter_code(spec, "python")
```

## Configuration

- **Development**: `config/qbitel.yaml`
- **Production**: `config/qbitel.production.yaml`
- **Compliance**: `config/compliance.yaml`
- **Environment**: `QBITEL_AI_*` prefix

## Testing

```bash
# Run all tests
pytest ai_engine/tests/ -v

# Run legacy module tests
pytest ai_engine/legacy/tests/ -v

# Run with coverage
pytest --cov=ai_engine --cov-report=html
```

## Common Tasks

### 1. Enhance UC1 Demo
See bead `qbitel-0026` for details on integrating with production code.

### 2. Add New Protocol Support
1. Add samples to `datasets/protocols/`
2. Update `ai_engine/llm/translation_studio.py`
3. Add tests

### 3. Add New LLM Provider
1. Update `ai_engine/llm/unified_llm_service.py`
2. Add provider configuration
3. Update fallback chain

## Beads Memory System

This project uses [Beads](https://github.com/steveyegge/beads) for AI agent memory.

```bash
# Memory is stored in .beads/issues.jsonl
# Each line is a JSON object with:
# - id: Unique identifier (qbitel-XXXX)
# - title: Brief description
# - description: Detailed documentation
# - status: open|closed|pinned
# - labels: Categories
# - priority: 0-4 (0 = highest)
```

## Contact

For questions about this codebase, refer to the beads memory system or the documentation in `docs/`.
