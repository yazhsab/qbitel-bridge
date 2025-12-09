# CRONOS AI Datasets

This directory contains training, validation, and test datasets for CRONOS AI's ML components.

## Directory Structure

```
datasets/
├── protocols/                    # Protocol message samples
│   ├── iso8583/                 # Banking transaction messages
│   ├── swift/                   # SWIFT MT/MX messages
│   ├── modbus/                  # Industrial SCADA protocol
│   ├── hl7/                     # Healthcare messages
│   ├── dnp3/                    # Utility SCADA protocol
│   └── fhir/                    # Modern healthcare API
│
├── field_detection/             # Labeled data for BiLSTM-CRF
│   ├── training/                # 80% of labeled data
│   ├── validation/              # 10% for hyperparameter tuning
│   └── test/                    # 10% for final evaluation
│
├── threat_intelligence/         # Security threat data
│   ├── mitre_attack/            # MITRE ATT&CK techniques
│   ├── cve/                     # CVE vulnerability data
│   ├── iocs/                    # Indicators of Compromise
│   └── malware_samples/         # Malware hashes and signatures
│
├── security_events/             # Security log data
│   ├── authentication/          # Auth success/failure logs
│   ├── network/                 # Network traffic logs
│   └── anomalies/               # Labeled anomaly events
│
└── anomaly_detection/           # Time-series anomaly data
    ├── normal/                  # Baseline normal behavior
    └── anomalous/               # Labeled anomalies
```

## Dataset Sources

### Public Datasets Used
- MITRE ATT&CK: https://attack.mitre.org/
- NVD CVE Database: https://nvd.nist.gov/
- Abuse.ch: https://abuse.ch/
- HL7 Examples: https://hl7.org/
- FHIR Examples: https://www.hl7.org/fhir/

### Synthetic Datasets
- ISO 8583: Generated based on specification
- SWIFT MT: Generated based on ISO 15022
- Modbus: Generated based on protocol specification
- Security Events: Synthetic but realistic patterns

## Usage

```python
from cronos_ai.datasets import DatasetLoader

# Load protocol samples
loader = DatasetLoader()
iso8583_samples = loader.load_protocol('iso8583')

# Load labeled field detection data
train_data = loader.load_field_detection('training')
val_data = loader.load_field_detection('validation')
test_data = loader.load_field_detection('test')

# Load threat intelligence
mitre_data = loader.load_threat_intel('mitre_attack')
```

## Data Formats

### Protocol Samples
- Format: Binary files with JSON metadata
- Naming: `{protocol}_{message_type}_{index}.bin`
- Metadata: `{protocol}_{message_type}_{index}.json`

### Field Detection Labels
- Format: JSON with IOB tags
- Schema: See `field_detection/schema.json`

### Threat Intelligence
- Format: STIX 2.1 JSON
- Schema: OASIS STIX standard

### Security Events
- Format: JSON Lines (.jsonl)
- Schema: Elastic Common Schema (ECS)
