"""
QBITEL - Modernization Recommender Service

Adapter code generation, testing, and deployment guide creation for legacy modernization.

Responsibilities:
- Generate production-ready adapter code
- Generate comprehensive test suites
- Extract dependencies from generated code
- Generate configuration templates
- Create deployment guides
- Assess code quality scores
- Create integration documentation
- Generate implementation roadmaps
"""

import json
import logging
from typing import Dict, Any, List

from .models import (
    ProtocolSpecification,
    AdapterCode,
    AdapterLanguage,
    LegacyWhispererException,
)
from ...llm.unified_llm_service import get_llm_service, LLMRequest

logger = logging.getLogger(__name__)


class ModernizationRecommender:
    """
    Service for generating modernization artifacts (code, tests, documentation).

    Features:
    - Multi-language adapter code generation (Python, Java, Go, Rust, TypeScript, C#)
    - Comprehensive test suite generation
    - Configuration template generation
    - Deployment guide generation
    - Code quality assessment
    """

    def __init__(self, llm_service=None):
        """
        Initialize the Modernization Recommender.

        Args:
            llm_service: Optional LLM service instance
        """
        self.llm_service = llm_service or get_llm_service()
        self.logger = logging.getLogger(__name__)

    async def generate_transformation_logic(
        self,
        legacy_protocol: ProtocolSpecification,
        target_protocol: str,
        language: AdapterLanguage,
        differences: Dict[str, Any],
    ) -> str:
        """
        Generate transformation logic code using LLM.

        Args:
            legacy_protocol: Legacy protocol specification
            target_protocol: Target protocol name
            language: Programming language for adapter
            differences: Protocol differences analysis

        Returns:
            Generated adapter code
        """
        llm_request = LLMRequest(
            prompt=f"""
            Generate production-ready {language.value} code for a protocol adapter:

            Source: {legacy_protocol.protocol_name}
            Target: {target_protocol}

            Legacy Protocol Fields:
            {chr(10).join(f"- {f.name}: {f.field_type} at offset {f.offset}, length {f.length}" for f in legacy_protocol.fields)}

            Transformation Requirements:
            {json.dumps(differences.get('transformation_requirements', []), indent=2)}

            Please generate:
            1. Complete adapter class/module
            2. Field transformation functions
            3. Error handling
            4. Logging
            5. Configuration management
            6. Connection pooling (if applicable)
            7. Retry logic
            8. Performance optimizations

            Requirements:
            - Production-ready code quality
            - Comprehensive error handling
            - Type hints/annotations
            - Docstrings
            - Thread-safe if applicable
            - Async support if beneficial

            Generate only the code, no explanations.
            """,
            feature_domain="legacy_whisperer",
            context={"generation_type": "adapter_code"},
            max_tokens=4000,
            temperature=0.2,
        )

        response = await self.llm_service.process_request(llm_request)

        # Extract code from response
        content = response.content
        if f"```{language.value}" in content:
            content = content.split(f"```{language.value}")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        return content.strip()

    async def generate_test_cases(
        self,
        legacy_protocol: ProtocolSpecification,
        target_protocol: str,
        language: AdapterLanguage,
        adapter_code: str,
    ) -> str:
        """
        Generate comprehensive test cases.

        Args:
            legacy_protocol: Legacy protocol specification
            target_protocol: Target protocol name
            language: Programming language
            adapter_code: Generated adapter code

        Returns:
            Generated test code
        """
        llm_request = LLMRequest(
            prompt=f"""
            Generate comprehensive test cases in {language.value} for this protocol adapter:

            Adapter Code Summary:
            - Source: {legacy_protocol.protocol_name}
            - Target: {target_protocol}
            - Language: {language.value}

            Generate tests for:
            1. Basic transformation functionality
            2. Edge cases (empty data, malformed data)
            3. Error handling
            4. Performance tests
            5. Integration tests
            6. Concurrent access tests (if applicable)

            Use appropriate testing framework for {language.value}.
            Include setup, teardown, and fixtures.
            Aim for >85% code coverage.

            Generate only the test code.
            """,
            feature_domain="legacy_whisperer",
            context={"generation_type": "test_code"},
            max_tokens=3000,
            temperature=0.2,
        )

        response = await self.llm_service.process_request(llm_request)

        content = response.content
        if f"```{language.value}" in content:
            content = content.split(f"```{language.value}")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        return content.strip()

    async def generate_integration_guide(
        self,
        legacy_protocol: ProtocolSpecification,
        target_protocol: str,
        language: AdapterLanguage,
        adapter_code: str,
    ) -> str:
        """
        Generate integration documentation.

        Args:
            legacy_protocol: Legacy protocol specification
            target_protocol: Target protocol name
            language: Programming language
            adapter_code: Generated adapter code

        Returns:
            Integration documentation (Markdown)
        """
        llm_request = LLMRequest(
            prompt=f"""
            Generate comprehensive integration documentation for this protocol adapter:

            Adapter: {legacy_protocol.protocol_name} → {target_protocol}
            Language: {language.value}

            Include:
            1. Overview and architecture
            2. Installation instructions
            3. Configuration guide
            4. Usage examples
            5. API reference
            6. Troubleshooting guide
            7. Performance tuning
            8. Security considerations
            9. Migration checklist
            10. FAQ

            Format in Markdown.
            """,
            feature_domain="legacy_whisperer",
            context={"generation_type": "documentation"},
            max_tokens=3000,
            temperature=0.3,
        )

        response = await self.llm_service.process_request(llm_request)
        return response.content

    async def generate_deployment_guide(
        self,
        legacy_protocol: ProtocolSpecification,
        target_protocol: str,
        language: AdapterLanguage,
        adapter_code: str,
    ) -> str:
        """
        Generate deployment guide.

        Args:
            legacy_protocol: Legacy protocol specification
            target_protocol: Target protocol name
            language: Programming language
            adapter_code: Generated adapter code

        Returns:
            Deployment guide (Markdown)
        """
        llm_request = LLMRequest(
            prompt=f"""
            Generate a deployment guide for this protocol adapter:

            Adapter: {legacy_protocol.protocol_name} → {target_protocol}
            Language: {language.value}

            Include:
            1. System requirements
            2. Pre-deployment checklist
            3. Deployment steps (development, staging, production)
            4. Docker/container deployment
            5. Kubernetes deployment (if applicable)
            6. Monitoring setup
            7. Backup and recovery procedures
            8. Rollback procedures
            9. Health checks
            10. Post-deployment validation

            Format in Markdown.
            """,
            feature_domain="legacy_whisperer",
            context={"generation_type": "deployment_guide"},
            max_tokens=2500,
            temperature=0.2,
        )

        response = await self.llm_service.process_request(llm_request)
        return response.content

    async def generate_implementation_guidance(
        self,
        behavior: str,
        approaches: List[Dict[str, Any]],
        risks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate implementation guidance for modernization.

        Args:
            behavior: Description of the legacy behavior
            approaches: Modernization approaches
            risks: Risk assessments

        Returns:
            Implementation guidance dictionary
        """
        llm_request = LLMRequest(
            prompt=f"""
            Generate implementation guidance for modernizing this legacy behavior:

            Behavior: {behavior}
            Approaches: {json.dumps(approaches, indent=2)}
            Risks: {json.dumps(risks, indent=2)}

            Provide:
            1. Step-by-step implementation steps
            2. Estimated effort (person-hours/days/weeks)
            3. Required expertise (roles and skills)
            4. Success criteria
            5. Testing strategy

            Format as JSON.
            """,
            feature_domain="legacy_whisperer",
            context={"analysis_type": "implementation_guidance"},
            max_tokens=2000,
            temperature=0.2,
        )

        response = await self.llm_service.process_request(llm_request)

        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return json.loads(content.strip())
        except:
            return {
                "steps": [
                    "Analyze current implementation",
                    "Design new solution",
                    "Implement",
                    "Test",
                    "Deploy",
                ],
                "effort": "4-6 weeks",
                "expertise": ["Senior Developer", "System Architect"],
            }

    def extract_dependencies(self, code: str, language: AdapterLanguage) -> List[str]:
        """
        Extract dependencies from generated code.

        Args:
            code: Generated code
            language: Programming language

        Returns:
            List of dependency names
        """
        dependencies = []

        if language == AdapterLanguage.PYTHON:
            # Extract import statements
            for line in code.split("\n"):
                line = line.strip()
                if line.startswith("import ") or line.startswith("from "):
                    # Extract package name
                    if "import" in line:
                        pkg = line.split("import")[1].split()[0].split(".")[0]
                        if pkg not in ["os", "sys", "time", "json", "typing"]:
                            dependencies.append(pkg)

        elif language == AdapterLanguage.JAVA:
            # Extract Maven dependencies
            for line in code.split("\n"):
                if "import" in line and not line.strip().startswith("//"):
                    pkg = line.split("import")[1].split(";")[0].strip()
                    if not pkg.startswith("java."):
                        dependencies.append(pkg.split(".")[0])

        elif language == AdapterLanguage.GO:
            # Extract Go imports
            in_import_block = False
            for line in code.split("\n"):
                line = line.strip()
                if line == "import (":
                    in_import_block = True
                elif line == ")" and in_import_block:
                    in_import_block = False
                elif in_import_block and line.startswith('"'):
                    pkg = line.strip('"')
                    if not pkg.startswith("fmt") and "/" in pkg:
                        dependencies.append(pkg)

        elif language == AdapterLanguage.TYPESCRIPT:
            # Extract npm dependencies
            for line in code.split("\n"):
                if line.strip().startswith("import ") and "from" in line:
                    pkg = line.split("from")[1].strip().strip("';\"")
                    if not pkg.startswith("."):
                        dependencies.append(pkg)

        return list(set(dependencies))

    def generate_config_template(
        self,
        legacy_protocol: ProtocolSpecification,
        target_protocol: str,
        language: AdapterLanguage,
    ) -> str:
        """
        Generate configuration template.

        Args:
            legacy_protocol: Legacy protocol specification
            target_protocol: Target protocol name
            language: Programming language

        Returns:
            Configuration template string
        """
        if language == AdapterLanguage.PYTHON:
            return f"""
# Configuration for {legacy_protocol.protocol_name} → {target_protocol} Adapter

[adapter]
source_protocol = "{legacy_protocol.protocol_name}"
target_protocol = "{target_protocol}"
log_level = "INFO"

[source]
host = "localhost"
port = 5000
timeout = 30
max_retries = 3

[target]
host = "localhost"
port = 8000
timeout = 30
max_connections = 100

[performance]
batch_size = 100
worker_threads = 4
enable_caching = true
cache_ttl = 300

[security]
enable_tls = true
verify_certificates = true
api_key = "${{API_KEY}}"
"""
        elif language == AdapterLanguage.JAVA:
            return f"""
# Configuration for {legacy_protocol.protocol_name} → {target_protocol} Adapter

adapter.source.protocol={legacy_protocol.protocol_name}
adapter.target.protocol={target_protocol}
adapter.log.level=INFO

source.host=localhost
source.port=5000
source.timeout=30000
source.max.retries=3

target.host=localhost
target.port=8000
target.timeout=30000
target.max.connections=100

performance.batch.size=100
performance.worker.threads=4
performance.cache.enabled=true
performance.cache.ttl=300

security.tls.enabled=true
security.certificates.verify=true
security.api.key=${{API_KEY}}
"""
        elif language == AdapterLanguage.GO:
            return f"""
# Configuration for {legacy_protocol.protocol_name} → {target_protocol} Adapter

source_protocol: "{legacy_protocol.protocol_name}"
target_protocol: "{target_protocol}"
log_level: "info"

source:
  host: "localhost"
  port: 5000
  timeout: 30s
  max_retries: 3

target:
  host: "localhost"
  port: 8000
  timeout: 30s
  max_connections: 100

performance:
  batch_size: 100
  worker_threads: 4
  cache_enabled: true
  cache_ttl: 300s

security:
  tls_enabled: true
  verify_certs: true
  api_key: "${{API_KEY}}"
"""
        else:
            return f"# Configuration template for {language.value}"

    async def assess_code_quality(self, adapter_code: str, test_code: str) -> float:
        """
        Assess generated code quality.

        Args:
            adapter_code: Generated adapter code
            test_code: Generated test code

        Returns:
            Code quality score (0.0 to 1.0)
        """
        quality_score = 0.0

        # Check for error handling
        if "try" in adapter_code and "except" in adapter_code:
            quality_score += 0.2

        # Check for logging
        if "log" in adapter_code.lower():
            quality_score += 0.15

        # Check for docstrings/comments
        if '"""' in adapter_code or "///" in adapter_code:
            quality_score += 0.15

        # Check for type hints (Python)
        if "->" in adapter_code or ": " in adapter_code:
            quality_score += 0.1

        # Check for tests
        if len(test_code) > 100:
            quality_score += 0.2

        # Check for configuration management
        if "config" in adapter_code.lower():
            quality_score += 0.1

        # Check for async support
        if "async" in adapter_code:
            quality_score += 0.1

        return min(1.0, quality_score)
