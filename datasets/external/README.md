# External Dataset Sources

This directory contains scripts and documentation for integrating real-world public datasets into QBITEL training pipelines.

## Quick Start

```bash
# Fetch all available threat intelligence (MITRE ATT&CK, URLhaus, MalwareBazaar)
python3 fetch_threat_intel.py

# Fetch network/anomaly datasets (NAB)
python3 fetch_network_datasets.py

# Fetch from ML platforms (Hugging Face, Kaggle, UCI)
python3 fetch_ml_platforms.py
```

## Already Integrated Datasets

These datasets have been automatically fetched and are ready for use:

### Threat Intelligence (Auto-fetched)
| Dataset | Records | Size | License | Status |
|---------|---------|------|---------|--------|
| **MITRE ATT&CK** | 703 techniques | 569 KB | Apache 2.0 | ✓ Integrated |
| **URLhaus** | ~5,000 malicious URLs | 1.7 MB | CC0 | ✓ Integrated |
| **MalwareBazaar** | ~700 hashes | 322 KB | CC0 | ✓ Integrated |

### Anomaly Detection (Auto-fetched)
| Dataset | Records | Size | License | Status |
|---------|---------|------|---------|--------|
| **NAB (Numenta)** | 6 time series, 43K points | 1.9 MB | Apache 2.0 | ✓ Integrated |

### Hugging Face Datasets (Auto-fetched)
| Dataset | Records | Use Case | Status |
|---------|---------|----------|--------|
| **pirocheto/phishing-url** | 11,430 | Phishing URL detection | ✓ Integrated |
| **tumeteor/Security-TTP-Mapping** | 20,736 | MITRE TTP mapping | ✓ Integrated |

## Available Public Datasets

### 1. Threat Intelligence Feeds

| Dataset | Source | Size | License |
|---------|--------|------|---------|
| **MITRE ATT&CK** | https://attack.mitre.org | 700+ techniques | Apache 2.0 |
| **Abuse.ch MalwareBazaar** | https://bazaar.abuse.ch | 1M+ samples | CC0 |
| **Abuse.ch URLhaus** | https://urlhaus.abuse.ch | 2M+ URLs | CC0 |
| **Abuse.ch ThreatFox** | https://threatfox.abuse.ch | IOCs | CC0 |

### 2. Hugging Face Datasets (pip install datasets)

| Dataset ID | Description | Size |
|------------|-------------|------|
| `pirocheto/phishing-url` | Phishing URL classification | ~11K |
| `tumeteor/Security-TTP-Mapping` | Security TTP mapping | ~21K |
| `cw1521/ember2018-malware` | EMBER malware features | Large |
| `zefang-liu/phishing-email-dataset` | Phishing emails | ~9K |
| `ethanolivertroy/nist-cybersecurity-training` | NIST security Q&A | ~800 |

### 3. Kaggle Datasets (pip install kaggle)

| Dataset ID | Description | Size |
|------------|-------------|------|
| `dhoogla/cicidss` | CICIDS2017 Network Intrusion | ~600MB |
| `sampadab17/network-intrusion-detection` | NSL-KDD | ~20MB |
| `mrwellsdavid/unsw-nb15` | UNSW-NB15 | ~200MB |
| `mlg-ulb/creditcardfraud` | Credit Card Fraud | ~150MB |
| `claudelemante/malware-data` | PE Malware Features | ~50MB |

### 4. UCI ML Repository (No auth required)

| Dataset | Description | URL |
|---------|-------------|-----|
| `kdd99` | KDD Cup 1999 IDS | Auto-fetch |
| `spambase` | Spam Classification | Auto-fetch |
| `phishing` | Phishing Websites | Auto-fetch |

### 5. Network Security (Manual Download)

| Dataset | Source | Size | Use Case |
|---------|--------|------|----------|
| **CICIDS2017** | unb.ca/cic/datasets | 6GB | IDS Training |
| **UNSW-NB15** | research.unsw.edu.au | 2GB | Modern Attacks |
| **Wireshark PCAP** | wiki.wireshark.org | Various | Protocol Analysis |

## Setup Instructions

### Hugging Face Datasets
```bash
pip install datasets huggingface_hub
# No API key required for public datasets
```

### Kaggle Datasets
```bash
pip install kaggle
# Setup API key:
# 1. Go to kaggle.com/account
# 2. Click "Create New API Token"
# 3. Save kaggle.json to ~/.kaggle/
# 4. chmod 600 ~/.kaggle/kaggle.json
```

### Fetching Specific Datasets
```python
from fetch_ml_platforms import MLPlatformFetcher

fetcher = MLPlatformFetcher()

# Hugging Face
fetcher.fetch_huggingface_dataset('pirocheto/phishing-url')

# Kaggle (requires API key)
fetcher.fetch_kaggle_dataset('mlg-ulb/creditcardfraud')

# UCI (no auth)
fetcher.fetch_uci_dataset('spambase')
```

## Data Licensing

| License | Usage |
|---------|-------|
| **CC0/Public Domain** | Free for any use |
| **Apache 2.0** | Free with attribution |
| **Research Only** | Academic use only |

## Directory Structure

```
external/
├── fetch_threat_intel.py      # MITRE, URLhaus, MalwareBazaar
├── fetch_network_datasets.py  # NAB, PCAP documentation
├── fetch_ml_platforms.py      # HuggingFace, Kaggle, UCI
├── ml_platforms/
│   ├── huggingface/           # Downloaded HF datasets
│   ├── kaggle/                # Downloaded Kaggle datasets
│   ├── uci/                   # Downloaded UCI datasets
│   └── dataset_catalog.json   # Full catalog
└── README.md
```
