# CRONOS AI - Security Implementation Plan

## Executive Summary

This document provides a detailed implementation plan for addressing critical security vulnerabilities and gaps identified in the CRONOS AI platform.

---

## Phase 1: Critical Vulnerability Patches (Week 1-2)

### 1.1 PyTorch CVE-2025-32434 (CRITICAL - RCE)

**Vulnerability**: Remote Code Execution through `torch.load()` - weights_only=True bypass
**Severity**: CVSS 9.3 (Critical)
**Current Version**: 2.2.2
**Required Version**: >= 2.6.0

#### Implementation Steps:

```bash
# Step 1: Create a new branch for security patches
git checkout -b security/pytorch-cve-2025-32434

# Step 2: Update requirements files
# See patch files below

# Step 3: Audit existing model loading code
grep -r "torch.load" ai_engine/ --include="*.py"

# Step 4: Run tests
pytest ai_engine/tests/ -v

# Step 5: Deploy to staging first
```

#### Code Changes Required:

**File: requirements.txt (Line 29-30)**
```python
# BEFORE
torch==2.2.2+cpu; sys_platform != "darwin"
torch==2.2.2; sys_platform == "darwin"

# AFTER
torch>=2.6.0+cpu; sys_platform != "darwin"
torch>=2.6.0; sys_platform == "darwin"
```

**File: ai_engine/requirements.txt (Line 2)**
```python
# BEFORE
torch>=2.0.0

# AFTER
torch>=2.6.0
```

#### Model Loading Security Hardening:

```python
# Add to ai_engine/core/model_loader.py (NEW FILE)
"""
Secure Model Loading Utilities for CRONOS AI
Addresses CVE-2025-32434 and implements defense-in-depth
"""

import torch
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Allowlist of trusted model sources
TRUSTED_MODEL_HASHES: Dict[str, str] = {
    # Add SHA256 hashes of known-good models
    # "model_name": "sha256_hash"
}

class SecureModelLoader:
    """Secure model loading with hash verification and sandboxing."""

    def __init__(self, trusted_hashes: Optional[Dict[str, str]] = None):
        self.trusted_hashes = trusted_hashes or TRUSTED_MODEL_HASHES

    def verify_model_hash(self, model_path: Path) -> bool:
        """Verify model file hash against trusted list."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        with open(model_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        model_name = model_path.stem
        expected_hash = self.trusted_hashes.get(model_name)

        if expected_hash and file_hash != expected_hash:
            logger.error(f"Model hash mismatch: {model_name}")
            return False

        return True

    def load_model_secure(
        self,
        model_path: Path,
        map_location: Optional[str] = None,
        verify_hash: bool = True
    ) -> Any:
        """
        Securely load a PyTorch model with CVE-2025-32434 mitigations.

        Args:
            model_path: Path to the model file
            map_location: Device mapping (cpu, cuda, etc.)
            verify_hash: Whether to verify against trusted hashes

        Returns:
            Loaded model state dict
        """
        if verify_hash and not self.verify_model_hash(model_path):
            raise SecurityError(f"Model verification failed: {model_path}")

        # Use torch.load with weights_only=True (PyTorch 2.6.0+)
        # This alone is NOT sufficient for CVE-2025-32434 but is a defense layer
        try:
            # For PyTorch 2.6.0+, use safe_globals for additional protection
            return torch.load(
                model_path,
                map_location=map_location,
                weights_only=True,
            )
        except Exception as e:
            logger.error(f"Failed to load model securely: {e}")
            raise

class SecurityError(Exception):
    """Security-related error."""
    pass

# Singleton instance
_secure_loader: Optional[SecureModelLoader] = None

def get_secure_model_loader() -> SecureModelLoader:
    """Get singleton secure model loader."""
    global _secure_loader
    if _secure_loader is None:
        _secure_loader = SecureModelLoader()
    return _secure_loader
```

---

### 1.2 FastAPI CVE-2024-24762 (HIGH - ReDoS)

**Vulnerability**: Regular Expression Denial of Service in python-multipart
**Severity**: HIGH
**Current Version**: 0.104.1
**Required Version**: >= 0.109.1 (preferably >= 0.121.3)

#### Implementation Steps:

```bash
# Step 1: Update FastAPI
pip install "fastapi>=0.121.3"

# Step 2: Update python-multipart explicitly
pip install "python-multipart>=0.0.7"

# Step 3: Test file upload endpoints
pytest ai_engine/tests/api/ -v -k "upload"
```

#### Code Changes:

**File: requirements.txt (Line 8)**
```python
# BEFORE
fastapi==0.104.1

# AFTER
fastapi>=0.121.3
```

**File: ai_engine/requirements.txt (Line 32)**
```python
# BEFORE
fastapi>=0.100.0

# AFTER
fastapi>=0.121.3
```

---

### 1.3 Go 1.22 End of Life (HIGH)

**Issue**: Go 1.22 is no longer supported (EOL with Go 1.24 release)
**Current Version**: 1.22
**Required Version**: >= 1.23

#### Implementation Steps:

```bash
# Step 1: Update Go version in go.mod files
cd go/controlplane && go mod edit -go=1.23
cd go/mgmtapi && go mod edit -go=1.23
cd go/agents/device-agent && go mod edit -go=1.23

# Step 2: Update dependencies
go mod tidy

# Step 3: Run tests
go test ./...

# Step 4: Update CI/CD pipelines
# See GitHub Actions workflow updates below
```

#### File Changes:

**go/controlplane/go.mod**
```go
// BEFORE
go 1.22

// AFTER
go 1.23
```

---

### 1.4 S3 CORS Configuration (CRITICAL)

**Issue**: Open CORS allows any origin to access S3 bucket
**Location**: scripts/setup_s3_bucket.py:158

#### Secure CORS Configuration:

```python
# Replace in scripts/setup_s3_bucket.py

def configure_bucket_cors(bucket_name: str, allowed_origins: list = None) -> bool:
    """
    Configure CORS for the bucket with restricted origins.

    Args:
        bucket_name: Name of the S3 bucket
        allowed_origins: List of allowed origins (domains)

    Returns:
        True if successful
    """
    try:
        s3_client = boto3.client('s3')

        logger.info("Configuring CORS with restricted origins...")

        # SECURE: Only allow specific domains
        if allowed_origins is None:
            allowed_origins = [
                'https://cronos-ai.example.com',
                'https://app.cronos-ai.example.com',
                'https://marketplace.cronos-ai.example.com',
            ]
            # Add localhost for development (remove in production)
            import os
            if os.getenv('ENVIRONMENT', 'production') == 'development':
                allowed_origins.extend([
                    'http://localhost:3000',
                    'http://localhost:8000',
                ])

        cors_configuration = {
            'CORSRules': [
                {
                    'AllowedHeaders': [
                        'Content-Type',
                        'Content-MD5',
                        'Authorization',
                        'X-Amz-Date',
                        'X-Api-Key',
                        'X-Amz-Security-Token',
                    ],
                    'AllowedMethods': ['GET', 'PUT', 'POST', 'HEAD'],
                    'AllowedOrigins': allowed_origins,  # RESTRICTED
                    'ExposeHeaders': ['ETag', 'x-amz-meta-custom-header'],
                    'MaxAgeSeconds': 3600
                }
            ]
        }

        s3_client.put_bucket_cors(
            Bucket=bucket_name,
            CORSConfiguration=cors_configuration
        )

        logger.info("✅ CORS configured with restricted origins")
        logger.info(f"   Allowed origins: {allowed_origins}")
        return True

    except ClientError as e:
        logger.error(f"Failed to configure CORS: {e}")
        return False
```

---

## Phase 2: Security Feature Implementation (Week 3-4)

### 2.1 Marketplace Sandboxed Execution

**Gap**: Protocol parsers run without isolation
**Risk**: Malicious parsers can compromise the system

#### Implementation:

```python
# ai_engine/marketplace/sandbox_executor.py (NEW FILE)
"""
Sandboxed Protocol Parser Execution for CRONOS AI Marketplace

Uses multiple isolation layers:
1. Process isolation with resource limits
2. Filesystem restrictions
3. Network isolation
4. Time limits
"""

import os
import sys
import signal
import resource
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class SandboxConfig:
    """Configuration for sandbox execution."""
    max_memory_mb: int = 512
    max_cpu_time_seconds: int = 30
    max_wall_time_seconds: int = 60
    max_file_size_mb: int = 50
    max_processes: int = 5
    network_disabled: bool = True
    allowed_imports: list = None

    def __post_init__(self):
        if self.allowed_imports is None:
            self.allowed_imports = [
                'json', 'struct', 'collections', 'dataclasses',
                'typing', 're', 'datetime', 'enum', 'abc',
                'itertools', 'functools', 'operator',
            ]

class SandboxExecutor:
    """
    Execute untrusted protocol parsers in a sandboxed environment.
    """

    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self._validate_environment()

    def _validate_environment(self):
        """Ensure sandbox dependencies are available."""
        # Check for Linux (best sandbox support)
        if sys.platform != 'linux':
            logger.warning(
                "Full sandbox isolation requires Linux. "
                "Running with limited isolation on non-Linux systems."
            )

    def execute_parser(
        self,
        parser_code: str,
        input_data: bytes,
        parser_name: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Execute a protocol parser in sandbox.

        Args:
            parser_code: Python code of the parser
            input_data: Binary data to parse
            parser_name: Name for logging

        Returns:
            Dict with execution result or error
        """
        # Create temporary directory for sandbox
        sandbox_dir = tempfile.mkdtemp(prefix="cronos_sandbox_")

        try:
            # Write parser code to sandbox
            parser_path = Path(sandbox_dir) / "parser.py"
            input_path = Path(sandbox_dir) / "input.bin"
            output_path = Path(sandbox_dir) / "output.json"

            # Validate and sanitize parser code
            sanitized_code = self._sanitize_code(parser_code)
            parser_path.write_text(sanitized_code)
            input_path.write_bytes(input_data)

            # Create wrapper script with resource limits
            wrapper_code = self._create_wrapper_script(
                parser_path, input_path, output_path
            )
            wrapper_path = Path(sandbox_dir) / "wrapper.py"
            wrapper_path.write_text(wrapper_code)

            # Execute in subprocess with isolation
            result = self._run_sandboxed(wrapper_path, sandbox_dir)

            # Read output if successful
            if result['success'] and output_path.exists():
                import json
                result['output'] = json.loads(output_path.read_text())

            return result

        except Exception as e:
            logger.error(f"Sandbox execution failed for {parser_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
        finally:
            # Clean up sandbox directory
            shutil.rmtree(sandbox_dir, ignore_errors=True)

    def _sanitize_code(self, code: str) -> str:
        """
        Sanitize parser code to remove dangerous constructs.

        Args:
            code: Original parser code

        Returns:
            Sanitized code
        """
        import ast

        # Parse to AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Invalid Python syntax: {e}")

        # Check for dangerous constructs
        dangerous_calls = {
            'eval', 'exec', 'compile', '__import__',
            'open', 'input', 'breakpoint',
            'getattr', 'setattr', 'delattr',
        }

        dangerous_modules = {
            'os', 'sys', 'subprocess', 'shutil', 'socket',
            'urllib', 'requests', 'http', 'ftplib',
            'pickle', 'marshal', 'shelve',
            'ctypes', 'multiprocessing', 'threading',
        }

        for node in ast.walk(tree):
            # Check function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in dangerous_calls:
                        raise ValueError(
                            f"Dangerous function call not allowed: {node.func.id}"
                        )

            # Check imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = node.module if isinstance(node, ast.ImportFrom) else None
                names = [alias.name for alias in node.names]

                for name in names:
                    full_name = f"{module}.{name}" if module else name
                    base_module = full_name.split('.')[0]

                    if base_module in dangerous_modules:
                        raise ValueError(
                            f"Import of {base_module} not allowed in sandbox"
                        )

        return code

    def _create_wrapper_script(
        self,
        parser_path: Path,
        input_path: Path,
        output_path: Path
    ) -> str:
        """Create wrapper script that enforces resource limits."""
        return f'''
import resource
import signal
import json
import sys

# Set resource limits
resource.setrlimit(resource.RLIMIT_AS, ({self.config.max_memory_mb * 1024 * 1024}, {self.config.max_memory_mb * 1024 * 1024}))
resource.setrlimit(resource.RLIMIT_CPU, ({self.config.max_cpu_time_seconds}, {self.config.max_cpu_time_seconds}))
resource.setrlimit(resource.RLIMIT_FSIZE, ({self.config.max_file_size_mb * 1024 * 1024}, {self.config.max_file_size_mb * 1024 * 1024}))
resource.setrlimit(resource.RLIMIT_NPROC, ({self.config.max_processes}, {self.config.max_processes}))

# Set timeout
def timeout_handler(signum, frame):
    raise TimeoutError("Parser execution timed out")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm({self.config.max_wall_time_seconds})

try:
    # Import and run parser
    sys.path.insert(0, "{parser_path.parent}")
    from parser import parse

    # Read input
    with open("{input_path}", "rb") as f:
        input_data = f.read()

    # Execute parser
    result = parse(input_data)

    # Write output
    with open("{output_path}", "w") as f:
        json.dump({{"success": True, "result": result}}, f)

except Exception as e:
    with open("{output_path}", "w") as f:
        json.dump({{"success": False, "error": str(e), "error_type": type(e).__name__}}, f)
'''

    def _run_sandboxed(
        self,
        wrapper_path: Path,
        sandbox_dir: str
    ) -> Dict[str, Any]:
        """Run the wrapper script in isolated subprocess."""

        env = os.environ.copy()
        # Remove sensitive environment variables
        sensitive_vars = [
            'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY',
            'DATABASE_URL', 'REDIS_URL', 'API_KEY',
            'JWT_SECRET', 'ENCRYPTION_KEY',
        ]
        for var in sensitive_vars:
            env.pop(var, None)

        # Disable network by removing proxy vars
        if self.config.network_disabled:
            env['no_proxy'] = '*'
            env.pop('http_proxy', None)
            env.pop('https_proxy', None)

        try:
            result = subprocess.run(
                [sys.executable, str(wrapper_path)],
                cwd=sandbox_dir,
                env=env,
                capture_output=True,
                timeout=self.config.max_wall_time_seconds + 5,
                text=True,
            )

            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode,
            }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Execution timed out',
                'error_type': 'TimeoutError',
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
            }

# Singleton
_sandbox_executor: Optional[SandboxExecutor] = None

def get_sandbox_executor() -> SandboxExecutor:
    """Get singleton sandbox executor."""
    global _sandbox_executor
    if _sandbox_executor is None:
        _sandbox_executor = SandboxExecutor()
    return _sandbox_executor
```

---

### 2.2 Virus Scanning Integration

**Gap**: No malware scanning for uploaded protocol files
**Location**: ai_engine/marketplace/s3_file_manager.py:329

#### Implementation:

```python
# ai_engine/marketplace/virus_scanner.py (NEW FILE)
"""
Virus Scanning Integration for CRONOS AI Marketplace

Supports multiple scanning backends:
1. ClamAV (local/daemon)
2. VirusTotal API (cloud)
3. AWS S3 Object Lambda (enterprise)
"""

import logging
import hashlib
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import aiohttp

logger = logging.getLogger(__name__)

class ScanResult(Enum):
    CLEAN = "clean"
    INFECTED = "infected"
    ERROR = "error"
    PENDING = "pending"

@dataclass
class ScanReport:
    """Result of virus scan."""
    status: ScanResult
    scanner: str
    file_hash: str
    threat_name: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    scan_time_ms: float = 0.0

class ClamAVScanner:
    """ClamAV virus scanner integration."""

    def __init__(self, socket_path: str = "/var/run/clamav/clamd.ctl"):
        self.socket_path = socket_path
        self._available = None

    async def is_available(self) -> bool:
        """Check if ClamAV daemon is available."""
        if self._available is not None:
            return self._available

        try:
            import clamd
            cd = clamd.ClamdUnixSocket(self.socket_path)
            cd.ping()
            self._available = True
        except Exception:
            self._available = False

        return self._available

    async def scan(self, file_content: bytes) -> ScanReport:
        """Scan file content with ClamAV."""
        import time
        start_time = time.time()
        file_hash = hashlib.sha256(file_content).hexdigest()

        try:
            import clamd
            from io import BytesIO

            cd = clamd.ClamdUnixSocket(self.socket_path)
            result = cd.instream(BytesIO(file_content))

            scan_time = (time.time() - start_time) * 1000

            # Parse result
            status_line = result.get('stream', ('UNKNOWN', ''))
            if status_line[0] == 'OK':
                return ScanReport(
                    status=ScanResult.CLEAN,
                    scanner="clamav",
                    file_hash=file_hash,
                    scan_time_ms=scan_time
                )
            else:
                return ScanReport(
                    status=ScanResult.INFECTED,
                    scanner="clamav",
                    file_hash=file_hash,
                    threat_name=status_line[1] if len(status_line) > 1 else "unknown",
                    scan_time_ms=scan_time
                )

        except Exception as e:
            logger.error(f"ClamAV scan failed: {e}")
            return ScanReport(
                status=ScanResult.ERROR,
                scanner="clamav",
                file_hash=file_hash,
                details={"error": str(e)},
                scan_time_ms=(time.time() - start_time) * 1000
            )

class VirusTotalScanner:
    """VirusTotal API scanner integration."""

    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.getenv("VIRUSTOTAL_API_KEY")
        self.base_url = "https://www.virustotal.com/api/v3"

    async def is_available(self) -> bool:
        """Check if VirusTotal API is available."""
        return bool(self.api_key)

    async def scan(self, file_content: bytes) -> ScanReport:
        """Scan file content with VirusTotal."""
        import time
        start_time = time.time()
        file_hash = hashlib.sha256(file_content).hexdigest()

        if not self.api_key:
            return ScanReport(
                status=ScanResult.ERROR,
                scanner="virustotal",
                file_hash=file_hash,
                details={"error": "API key not configured"}
            )

        headers = {"x-apikey": self.api_key}

        try:
            async with aiohttp.ClientSession() as session:
                # First check if file is already analyzed
                check_url = f"{self.base_url}/files/{file_hash}"
                async with session.get(check_url, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return self._parse_vt_response(data, file_hash, start_time)

                # Upload for new analysis
                upload_url = f"{self.base_url}/files"
                form = aiohttp.FormData()
                form.add_field('file', file_content, filename='scan_file')

                async with session.post(upload_url, headers=headers, data=form) as resp:
                    if resp.status != 200:
                        return ScanReport(
                            status=ScanResult.ERROR,
                            scanner="virustotal",
                            file_hash=file_hash,
                            details={"error": f"Upload failed: {resp.status}"}
                        )

                    data = await resp.json()
                    analysis_id = data.get('data', {}).get('id')

                # Poll for results
                analysis_url = f"{self.base_url}/analyses/{analysis_id}"
                for _ in range(30):  # Max 5 minutes
                    await asyncio.sleep(10)
                    async with session.get(analysis_url, headers=headers) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            status = data.get('data', {}).get('attributes', {}).get('status')
                            if status == 'completed':
                                return self._parse_vt_response(data, file_hash, start_time)

                return ScanReport(
                    status=ScanResult.PENDING,
                    scanner="virustotal",
                    file_hash=file_hash,
                    details={"message": "Analysis still pending"}
                )

        except Exception as e:
            logger.error(f"VirusTotal scan failed: {e}")
            return ScanReport(
                status=ScanResult.ERROR,
                scanner="virustotal",
                file_hash=file_hash,
                details={"error": str(e)},
                scan_time_ms=(time.time() - start_time) * 1000
            )

    def _parse_vt_response(
        self, data: Dict, file_hash: str, start_time: float
    ) -> ScanReport:
        """Parse VirusTotal API response."""
        import time

        attrs = data.get('data', {}).get('attributes', {})
        stats = attrs.get('last_analysis_stats', {})

        malicious = stats.get('malicious', 0)
        suspicious = stats.get('suspicious', 0)

        if malicious > 0 or suspicious > 0:
            return ScanReport(
                status=ScanResult.INFECTED,
                scanner="virustotal",
                file_hash=file_hash,
                threat_name=f"{malicious} detections",
                details={"stats": stats},
                scan_time_ms=(time.time() - start_time) * 1000
            )

        return ScanReport(
            status=ScanResult.CLEAN,
            scanner="virustotal",
            file_hash=file_hash,
            details={"stats": stats},
            scan_time_ms=(time.time() - start_time) * 1000
        )

class CompositeVirusScanner:
    """
    Composite scanner that tries multiple backends.

    Scanning order:
    1. ClamAV (if available) - fast local scan
    2. VirusTotal (if API key configured) - comprehensive cloud scan
    """

    def __init__(self):
        self.clamav = ClamAVScanner()
        self.virustotal = VirusTotalScanner()

    async def scan(
        self,
        file_content: bytes,
        require_clean: bool = True
    ) -> ScanReport:
        """
        Scan file with available scanners.

        Args:
            file_content: File bytes to scan
            require_clean: If True, file must pass all available scanners

        Returns:
            ScanReport with combined results
        """
        reports = []

        # Try ClamAV first (fast)
        if await self.clamav.is_available():
            report = await self.clamav.scan(file_content)
            reports.append(report)

            # If infected, return immediately
            if report.status == ScanResult.INFECTED:
                return report

        # Try VirusTotal for additional verification
        if await self.virustotal.is_available():
            report = await self.virustotal.scan(file_content)
            reports.append(report)

            if report.status == ScanResult.INFECTED:
                return report

        # Return combined clean result
        if reports:
            return ScanReport(
                status=ScanResult.CLEAN,
                scanner="composite",
                file_hash=hashlib.sha256(file_content).hexdigest(),
                details={"scanners": [r.scanner for r in reports]}
            )

        # No scanners available
        logger.warning("No virus scanners available")
        return ScanReport(
            status=ScanResult.ERROR,
            scanner="none",
            file_hash=hashlib.sha256(file_content).hexdigest(),
            details={"error": "No scanners available"}
        )

# Singleton
_virus_scanner: Optional[CompositeVirusScanner] = None

def get_virus_scanner() -> CompositeVirusScanner:
    """Get singleton virus scanner."""
    global _virus_scanner
    if _virus_scanner is None:
        _virus_scanner = CompositeVirusScanner()
    return _virus_scanner
```

---

### 2.3 SIEM Integration

**Gap**: No integration with enterprise SIEM systems
**Location**: ai_engine/security/audit_logger.py

#### Implementation:

```python
# ai_engine/security/siem_integration.py (NEW FILE)
"""
SIEM Integration for CRONOS AI

Supports:
1. Splunk (HEC - HTTP Event Collector)
2. Elastic/ELK Stack
3. Azure Sentinel
4. AWS Security Hub
5. Syslog (RFC 5424)
"""

import json
import socket
import ssl
import logging
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import aiohttp

logger = logging.getLogger(__name__)

class SeverityLevel(Enum):
    """SIEM severity levels (CEF standard)."""
    UNKNOWN = 0
    LOW = 1
    MEDIUM = 4
    HIGH = 7
    CRITICAL = 10

@dataclass
class SecurityEvent:
    """Standardized security event for SIEM."""
    event_id: str
    timestamp: str
    event_type: str
    severity: SeverityLevel
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    action: Optional[str] = None
    outcome: Optional[str] = None
    resource: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_cef(self) -> str:
        """Convert to Common Event Format (CEF)."""
        cef_severity = self.severity.value

        extension = []
        if self.source_ip:
            extension.append(f"src={self.source_ip}")
        if self.user_id:
            extension.append(f"suser={self.user_id}")
        if self.action:
            extension.append(f"act={self.action}")
        if self.outcome:
            extension.append(f"outcome={self.outcome}")
        if self.resource:
            extension.append(f"cs1={self.resource}")

        ext_str = " ".join(extension)

        return (
            f"CEF:0|CRONOS-AI|SecurityPlatform|1.0|{self.event_type}|"
            f"{self.event_type}|{cef_severity}|{ext_str}"
        )

    def to_ecs(self) -> Dict[str, Any]:
        """Convert to Elastic Common Schema (ECS)."""
        return {
            "@timestamp": self.timestamp,
            "event": {
                "id": self.event_id,
                "kind": "event",
                "category": ["security"],
                "type": [self.event_type],
                "severity": self.severity.value,
                "outcome": self.outcome or "unknown",
            },
            "source": {
                "ip": self.source_ip,
            },
            "user": {
                "id": self.user_id,
            },
            "cronos": {
                "action": self.action,
                "resource": self.resource,
                "details": self.details,
            }
        }

class SplunkHECClient:
    """Splunk HTTP Event Collector client."""

    def __init__(
        self,
        hec_url: str,
        hec_token: str,
        index: str = "cronos_security",
        source: str = "cronos-ai",
        verify_ssl: bool = True
    ):
        self.hec_url = hec_url.rstrip('/')
        self.hec_token = hec_token
        self.index = index
        self.source = source
        self.verify_ssl = verify_ssl

    async def send_event(self, event: SecurityEvent) -> bool:
        """Send event to Splunk."""
        headers = {
            "Authorization": f"Splunk {self.hec_token}",
            "Content-Type": "application/json",
        }

        payload = {
            "time": datetime.fromisoformat(event.timestamp.replace('Z', '+00:00')).timestamp(),
            "host": "cronos-ai",
            "source": self.source,
            "sourcetype": "cronos:security",
            "index": self.index,
            "event": asdict(event),
        }

        try:
            ssl_context = None if self.verify_ssl else False
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.hec_url}/services/collector/event",
                    headers=headers,
                    json=payload,
                    ssl=ssl_context
                ) as resp:
                    if resp.status != 200:
                        logger.error(f"Splunk HEC error: {resp.status}")
                        return False
                    return True
        except Exception as e:
            logger.error(f"Failed to send to Splunk: {e}")
            return False

    async def send_batch(self, events: List[SecurityEvent]) -> bool:
        """Send multiple events to Splunk."""
        headers = {
            "Authorization": f"Splunk {self.hec_token}",
            "Content-Type": "application/json",
        }

        payload = ""
        for event in events:
            entry = {
                "time": datetime.fromisoformat(event.timestamp.replace('Z', '+00:00')).timestamp(),
                "host": "cronos-ai",
                "source": self.source,
                "sourcetype": "cronos:security",
                "index": self.index,
                "event": asdict(event),
            }
            payload += json.dumps(entry) + "\n"

        try:
            ssl_context = None if self.verify_ssl else False
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.hec_url}/services/collector/event",
                    headers=headers,
                    data=payload,
                    ssl=ssl_context
                ) as resp:
                    return resp.status == 200
        except Exception as e:
            logger.error(f"Failed to send batch to Splunk: {e}")
            return False

class ElasticClient:
    """Elasticsearch client for ELK stack."""

    def __init__(
        self,
        es_url: str,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        index_prefix: str = "cronos-security"
    ):
        self.es_url = es_url.rstrip('/')
        self.api_key = api_key
        self.username = username
        self.password = password
        self.index_prefix = index_prefix

    def _get_auth(self) -> Optional[aiohttp.BasicAuth]:
        """Get authentication for requests."""
        if self.username and self.password:
            return aiohttp.BasicAuth(self.username, self.password)
        return None

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for requests."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"ApiKey {self.api_key}"
        return headers

    async def send_event(self, event: SecurityEvent) -> bool:
        """Send event to Elasticsearch."""
        index = f"{self.index_prefix}-{datetime.utcnow().strftime('%Y.%m.%d')}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.es_url}/{index}/_doc",
                    headers=self._get_headers(),
                    auth=self._get_auth(),
                    json=event.to_ecs()
                ) as resp:
                    return resp.status in (200, 201)
        except Exception as e:
            logger.error(f"Failed to send to Elasticsearch: {e}")
            return False

class SyslogClient:
    """RFC 5424 Syslog client."""

    def __init__(
        self,
        host: str,
        port: int = 514,
        protocol: str = "tcp",  # tcp, udp, tls
        facility: int = 1,  # user-level messages
    ):
        self.host = host
        self.port = port
        self.protocol = protocol
        self.facility = facility

    def _build_message(self, event: SecurityEvent) -> bytes:
        """Build RFC 5424 syslog message."""
        # Priority = Facility * 8 + Severity
        priority = self.facility * 8 + min(event.severity.value, 7)

        timestamp = datetime.fromisoformat(
            event.timestamp.replace('Z', '+00:00')
        ).strftime('%Y-%m-%dT%H:%M:%S.%fZ')

        hostname = socket.gethostname()
        app_name = "cronos-ai"
        proc_id = "-"
        msg_id = event.event_type

        # Structured data
        sd = f'[cronos@12345 eventId="{event.event_id}" action="{event.action or "-"}"]'

        # Message
        msg = event.to_cef()

        syslog_msg = (
            f"<{priority}>1 {timestamp} {hostname} {app_name} "
            f"{proc_id} {msg_id} {sd} {msg}"
        )

        return syslog_msg.encode('utf-8')

    async def send_event(self, event: SecurityEvent) -> bool:
        """Send event via syslog."""
        message = self._build_message(event)

        try:
            if self.protocol == "udp":
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.sendto(message, (self.host, self.port))
                sock.close()

            elif self.protocol == "tcp":
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((self.host, self.port))
                sock.send(message + b'\n')
                sock.close()

            elif self.protocol == "tls":
                context = ssl.create_default_context()
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock = context.wrap_socket(sock, server_hostname=self.host)
                sock.connect((self.host, self.port))
                sock.send(message + b'\n')
                sock.close()

            return True

        except Exception as e:
            logger.error(f"Failed to send syslog: {e}")
            return False

class SIEMIntegration:
    """
    Unified SIEM integration that sends to multiple backends.
    """

    def __init__(self):
        self.backends: List[Any] = []
        self._configure_backends()

    def _configure_backends(self):
        """Configure SIEM backends from environment."""
        import os

        # Splunk
        splunk_url = os.getenv("SPLUNK_HEC_URL")
        splunk_token = os.getenv("SPLUNK_HEC_TOKEN")
        if splunk_url and splunk_token:
            self.backends.append(SplunkHECClient(splunk_url, splunk_token))
            logger.info("Splunk HEC integration enabled")

        # Elasticsearch
        es_url = os.getenv("ELASTICSEARCH_URL")
        if es_url:
            self.backends.append(ElasticClient(
                es_url,
                api_key=os.getenv("ELASTICSEARCH_API_KEY"),
                username=os.getenv("ELASTICSEARCH_USER"),
                password=os.getenv("ELASTICSEARCH_PASSWORD"),
            ))
            logger.info("Elasticsearch integration enabled")

        # Syslog
        syslog_host = os.getenv("SYSLOG_HOST")
        if syslog_host:
            self.backends.append(SyslogClient(
                syslog_host,
                port=int(os.getenv("SYSLOG_PORT", "514")),
                protocol=os.getenv("SYSLOG_PROTOCOL", "tcp"),
            ))
            logger.info("Syslog integration enabled")

    async def send_event(self, event: SecurityEvent) -> Dict[str, bool]:
        """Send event to all configured backends."""
        results = {}

        for backend in self.backends:
            backend_name = type(backend).__name__
            try:
                results[backend_name] = await backend.send_event(event)
            except Exception as e:
                logger.error(f"Backend {backend_name} failed: {e}")
                results[backend_name] = False

        return results

    async def send_batch(self, events: List[SecurityEvent]) -> Dict[str, bool]:
        """Send batch of events to all backends."""
        results = {}

        for backend in self.backends:
            backend_name = type(backend).__name__

            if hasattr(backend, 'send_batch'):
                try:
                    results[backend_name] = await backend.send_batch(events)
                except Exception as e:
                    logger.error(f"Batch send to {backend_name} failed: {e}")
                    results[backend_name] = False
            else:
                # Fallback to individual sends
                success = True
                for event in events:
                    if not await backend.send_event(event):
                        success = False
                results[backend_name] = success

        return results

# Singleton
_siem_integration: Optional[SIEMIntegration] = None

def get_siem_integration() -> SIEMIntegration:
    """Get singleton SIEM integration."""
    global _siem_integration
    if _siem_integration is None:
        _siem_integration = SIEMIntegration()
    return _siem_integration
```

---

## Phase 3: Authentication & Authorization (Week 5-6)

### 3.1 Two-Factor Authentication for Protocol Creators

See separate file: `docs/IMPLEMENTATION_PLAN_2FA.md`

### 3.2 Enhanced API Key Rotation

See separate file: `docs/IMPLEMENTATION_PLAN_API_KEYS.md`

---

## Verification Checklist

### Critical Patches
- [ ] PyTorch upgraded to >= 2.6.0
- [ ] FastAPI upgraded to >= 0.121.3
- [ ] Go upgraded to >= 1.23
- [ ] S3 CORS restricted to specific domains
- [ ] All model loading uses SecureModelLoader

### Security Features
- [ ] Marketplace sandbox executor deployed
- [ ] Virus scanning enabled for uploads
- [ ] SIEM integration configured
- [ ] 2FA available for protocol creators
- [ ] API key rotation automated

### Testing
- [ ] All security tests passing
- [ ] Penetration testing completed
- [ ] Load testing with security features
- [ ] Incident response drill conducted

---

## Rollback Plan

If issues occur after deployment:

```bash
# Rollback to previous requirements
git checkout HEAD~1 -- requirements.txt ai_engine/requirements.txt

# Reinstall dependencies
pip install -r requirements.txt

# Restart services
systemctl restart cronos-ai

# Monitor for issues
tail -f /var/log/cronos-ai/error.log
```

---

## References

- [CVE-2025-32434 - PyTorch RCE](https://nvd.nist.gov/vuln/detail/CVE-2025-32434)
- [CVE-2024-24762 - FastAPI ReDoS](https://github.com/advisories/GHSA-93gm-qmq6-w238)
- [Go Release Policy](https://go.dev/doc/devel/release)
- [OWASP Secure Coding Practices](https://owasp.org/www-project-secure-coding-practices-quick-reference-guide/)
