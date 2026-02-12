"""
Protocol Adapter Generator

Zero-touch generation of protocol adapters for data transformation.
Automatically creates adapters based on protocol analysis.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid
import re

logger = logging.getLogger(__name__)


class AdapterType(Enum):
    """Types of protocol adapters."""

    INBOUND = "inbound"  # External to internal format
    OUTBOUND = "outbound"  # Internal to external format
    BIDIRECTIONAL = "bidirectional"  # Both directions
    VALIDATOR = "validator"  # Validation only


class TransformationType(Enum):
    """Types of field transformations."""

    DIRECT_MAP = "direct_map"  # 1:1 mapping
    FORMAT_CONVERSION = "format_conversion"  # Format change (date, number)
    ENCODING_CONVERSION = "encoding_conversion"  # Character encoding
    AGGREGATION = "aggregation"  # Combine multiple fields
    SPLIT = "split"  # Split one field into multiple
    LOOKUP = "lookup"  # Value lookup/translation
    COMPUTED = "computed"  # Computed from other fields
    DEFAULT = "default"  # Use default value
    CONDITIONAL = "conditional"  # Conditional transformation


@dataclass
class FieldMapping:
    """Mapping between source and target fields."""

    source_field: str
    target_field: str
    transformation: TransformationType
    transform_config: Dict[str, Any] = field(default_factory=dict)
    required: bool = False
    default_value: Optional[Any] = None
    validation_rules: List[str] = field(default_factory=list)


@dataclass
class AdapterConfig:
    """Configuration for adapter generation."""

    name: str
    adapter_type: AdapterType
    source_protocol: str
    target_protocol: str
    mappings: List[FieldMapping] = field(default_factory=list)
    pre_processors: List[str] = field(default_factory=list)
    post_processors: List[str] = field(default_factory=list)
    error_handling: str = "fail_fast"  # fail_fast, skip_invalid, best_effort
    enable_validation: bool = True
    enable_logging: bool = True
    enable_metrics: bool = True


@dataclass
class GeneratedAdapter:
    """A generated protocol adapter."""

    adapter_id: str
    name: str
    config: AdapterConfig
    generated_code: str
    validation_code: str
    test_code: str
    documentation: str
    generated_at: datetime = field(default_factory=datetime.utcnow)

    # Stats
    field_count: int = 0
    transformation_count: int = 0
    complexity_score: float = 0.0


class ProtocolAdapterGenerator:
    """
    Automatic protocol adapter generator.

    Capabilities:
    - Auto-generate field mappings from protocol analysis
    - Generate transformation code
    - Generate validation logic
    - Generate test cases
    - Support PQC-aware data handling
    """

    def __init__(self):
        self._transform_templates = self._load_transform_templates()

    def generate_adapter(
        self,
        source_analysis: Dict[str, Any],
        target_analysis: Dict[str, Any],
        config: Optional[AdapterConfig] = None,
    ) -> GeneratedAdapter:
        """
        Generate a protocol adapter.

        Args:
            source_analysis: Source protocol analysis
            target_analysis: Target protocol analysis
            config: Optional adapter configuration

        Returns:
            GeneratedAdapter with code and documentation
        """
        # Create default config if not provided
        if config is None:
            config = AdapterConfig(
                name=f"{source_analysis.get('protocol_type', 'source')}_to_{target_analysis.get('protocol_type', 'target')}",
                adapter_type=AdapterType.BIDIRECTIONAL,
                source_protocol=source_analysis.get("protocol_type", "unknown"),
                target_protocol=target_analysis.get("protocol_type", "unknown"),
            )

        # Generate field mappings
        if not config.mappings:
            config.mappings = self._generate_field_mappings(source_analysis, target_analysis)

        # Generate adapter code
        adapter_code = self._generate_adapter_code(config)

        # Generate validation code
        validation_code = self._generate_validation_code(config)

        # Generate test code
        test_code = self._generate_test_code(config)

        # Generate documentation
        documentation = self._generate_documentation(config)

        # Calculate complexity
        complexity = self._calculate_complexity(config)

        return GeneratedAdapter(
            adapter_id=str(uuid.uuid4()),
            name=config.name,
            config=config,
            generated_code=adapter_code,
            validation_code=validation_code,
            test_code=test_code,
            documentation=documentation,
            field_count=len(config.mappings),
            transformation_count=sum(1 for m in config.mappings if m.transformation != TransformationType.DIRECT_MAP),
            complexity_score=complexity,
        )

    def generate_from_comparison(
        self,
        comparison: Dict[str, Any],
        adapter_name: str,
    ) -> GeneratedAdapter:
        """
        Generate adapter from protocol comparison result.

        Args:
            comparison: Output from ProtocolAnalyzer.compare_protocols()
            adapter_name: Name for the adapter

        Returns:
            GeneratedAdapter
        """
        mappings = []

        # Create mappings from comparison
        for mapping in comparison.get("field_mappings", []):
            transform_type = (
                TransformationType.DIRECT_MAP if mapping.get("type_compatible") else TransformationType.FORMAT_CONVERSION
            )

            mappings.append(
                FieldMapping(
                    source_field=mapping["source"],
                    target_field=mapping["target"],
                    transformation=transform_type,
                    transform_config={"confidence": mapping.get("confidence", 1.0)},
                )
            )

        config = AdapterConfig(
            name=adapter_name,
            adapter_type=AdapterType.BIDIRECTIONAL,
            source_protocol=comparison.get("source_protocol", "unknown"),
            target_protocol=comparison.get("target_protocol", "unknown"),
            mappings=mappings,
        )

        return self.generate_adapter({}, {}, config)

    def _generate_field_mappings(
        self,
        source_analysis: Dict[str, Any],
        target_analysis: Dict[str, Any],
    ) -> List[FieldMapping]:
        """Generate field mappings from protocol analysis."""
        mappings = []

        source_fields = {f.get("name", ""): f for f in source_analysis.get("fields", [])}
        target_fields = {f.get("name", ""): f for f in target_analysis.get("fields", [])}

        # Match fields by name similarity
        for source_name, source_field in source_fields.items():
            best_match = self._find_best_match(source_name, target_fields)

            if best_match:
                target_name, target_field, confidence = best_match

                # Determine transformation type
                transform_type = self._determine_transformation(source_field, target_field)

                mappings.append(
                    FieldMapping(
                        source_field=source_name,
                        target_field=target_name,
                        transformation=transform_type,
                        transform_config={
                            "source_type": source_field.get("data_type"),
                            "target_type": target_field.get("data_type"),
                            "confidence": confidence,
                        },
                        required=source_field.get("required", False),
                    )
                )

        return mappings

    def _find_best_match(
        self,
        source_name: str,
        target_fields: Dict[str, Dict],
    ) -> Optional[Tuple[str, Dict, float]]:
        """Find best matching target field."""
        best_match = None
        best_score = 0.0

        source_tokens = set(re.split(r"[_\-\s]", source_name.lower()))

        for target_name, target_field in target_fields.items():
            target_tokens = set(re.split(r"[_\-\s]", target_name.lower()))

            # Calculate similarity
            if source_tokens and target_tokens:
                intersection = source_tokens & target_tokens
                union = source_tokens | target_tokens
                score = len(intersection) / len(union)

                # Exact match bonus
                if source_name.lower() == target_name.lower():
                    score = 1.0

                if score > best_score:
                    best_score = score
                    best_match = (target_name, target_field, score)

        if best_score >= 0.3:  # Minimum threshold
            return best_match
        return None

    def _determine_transformation(
        self,
        source_field: Dict,
        target_field: Dict,
    ) -> TransformationType:
        """Determine appropriate transformation type."""
        source_type = source_field.get("data_type", "").lower()
        target_type = target_field.get("data_type", "").lower()

        if source_type == target_type:
            return TransformationType.DIRECT_MAP

        # Date conversions
        if "date" in source_type or "date" in target_type:
            return TransformationType.FORMAT_CONVERSION

        # Numeric conversions
        if source_type in ("int", "integer", "decimal", "float") and target_type in (
            "int",
            "integer",
            "decimal",
            "float",
            "string",
        ):
            return TransformationType.FORMAT_CONVERSION

        # Encoding
        if source_type == "ebcdic" or target_type == "ebcdic":
            return TransformationType.ENCODING_CONVERSION

        return TransformationType.FORMAT_CONVERSION

    def _generate_adapter_code(self, config: AdapterConfig) -> str:
        """Generate Python adapter code."""
        code = f'''"""
Auto-generated Protocol Adapter: {config.name}
Source: {config.source_protocol}
Target: {config.target_protocol}
Generated by Qbitel AI Zero-Touch Automation
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class {self._to_class_name(config.name)}Adapter:
    """
    Protocol adapter for {config.source_protocol} to {config.target_protocol} conversion.

    Adapter Type: {config.adapter_type.value}
    Field Mappings: {len(config.mappings)}
    """

    def __init__(self):
        self._field_mappings = {{
'''
        # Add field mappings
        for mapping in config.mappings:
            code += f'            "{mapping.source_field}": "{mapping.target_field}",\n'

        code += '''        }
        self._transformers = self._init_transformers()

    def transform(self, source_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform source data to target format.

        Args:
            source_data: Source protocol data

        Returns:
            Transformed data in target format
        """
        target_data = {}
        errors = []

        for source_field, target_field in self._field_mappings.items():
            try:
                if source_field in source_data:
                    value = source_data[source_field]
                    transformer = self._transformers.get(source_field)

                    if transformer:
                        value = transformer(value)

                    target_data[target_field] = value
            except Exception as e:
                errors.append(f"Transform error for {source_field}: {e}")
                logger.warning(f"Transform error: {e}")

        if errors and "{config.error_handling}" == "fail_fast":
            raise ValueError(f"Transform failed: {errors}")

        return target_data

    def reverse_transform(self, target_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform from target back to source format."""
        reverse_mappings = {{v: k for k, v in self._field_mappings.items()}}
        source_data = {{}}

        for target_field, source_field in reverse_mappings.items():
            if target_field in target_data:
                source_data[source_field] = target_data[target_field]

        return source_data

    def _init_transformers(self) -> Dict[str, callable]:
        """Initialize field transformers."""
        return {{
'''
        # Add transformers for non-direct mappings
        for mapping in config.mappings:
            if mapping.transformation != TransformationType.DIRECT_MAP:
                code += (
                    f'            "{mapping.source_field}": self._transform_{self._to_method_name(mapping.source_field)},\n'
                )

        code += """        }
"""
        # Add transformer methods
        for mapping in config.mappings:
            if mapping.transformation != TransformationType.DIRECT_MAP:
                code += self._generate_transformer_method(mapping)

        code += '''
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate transformed data."""
        # Validation logic here
        return True
'''
        return code

    def _generate_transformer_method(self, mapping: FieldMapping) -> str:
        """Generate transformer method for a field."""
        method_name = self._to_method_name(mapping.source_field)

        if mapping.transformation == TransformationType.FORMAT_CONVERSION:
            return f'''
    def _transform_{method_name}(self, value: Any) -> Any:
        """Transform {mapping.source_field}."""
        # Format conversion: {mapping.transform_config}
        if value is None:
            return {repr(mapping.default_value)}
        return str(value)
'''
        elif mapping.transformation == TransformationType.ENCODING_CONVERSION:
            return f'''
    def _transform_{method_name}(self, value: Any) -> Any:
        """Transform {mapping.source_field} encoding."""
        if isinstance(value, bytes):
            return value.decode('utf-8', errors='replace')
        return value
'''
        else:
            return f'''
    def _transform_{method_name}(self, value: Any) -> Any:
        """Transform {mapping.source_field}."""
        return value
'''

    def _generate_validation_code(self, config: AdapterConfig) -> str:
        """Generate validation code."""
        code = f'''"""
Validation module for {config.name}
"""

from typing import Any, Dict, List
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of validation."""
    valid: bool
    errors: List[str]
    warnings: List[str]


class {self._to_class_name(config.name)}Validator:
    """Validator for {config.name} adapter."""

    def validate(self, data: Dict[str, Any], direction: str = "inbound") -> ValidationResult:
        """
        Validate data.

        Args:
            data: Data to validate
            direction: 'inbound' or 'outbound'

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

'''
        # Add required field checks
        for mapping in config.mappings:
            if mapping.required:
                field = mapping.source_field if config.adapter_type == AdapterType.INBOUND else mapping.target_field
                code += f"""        if "{field}" not in data or data["{field}"] is None:
            errors.append("Required field missing: {field}")

"""
        code += """        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
"""
        return code

    def _generate_test_code(self, config: AdapterConfig) -> str:
        """Generate test code."""
        code = f'''"""
Tests for {config.name}
"""

import pytest
from {self._to_module_name(config.name)} import {self._to_class_name(config.name)}Adapter


class Test{self._to_class_name(config.name)}Adapter:
    """Tests for the adapter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = {self._to_class_name(config.name)}Adapter()

    def test_transform_basic(self):
        """Test basic transformation."""
        source_data = {{
'''
        # Add sample data
        for mapping in config.mappings[:5]:  # First 5 fields
            code += f'            "{mapping.source_field}": "test_value",\n'

        code += f'''        }}

        result = self.adapter.transform(source_data)

        assert result is not None
        assert isinstance(result, dict)

    def test_reverse_transform(self):
        """Test reverse transformation."""
        target_data = {{
'''
        for mapping in config.mappings[:5]:
            code += f'            "{mapping.target_field}": "test_value",\n'

        code += '''        }

        result = self.adapter.reverse_transform(target_data)

        assert result is not None
        assert isinstance(result, dict)

    def test_transform_empty(self):
        """Test transformation of empty data."""
        result = self.adapter.transform({})
        assert result == {}
'''
        return code

    def _generate_documentation(self, config: AdapterConfig) -> str:
        """Generate documentation."""
        doc = f"""# {config.name} Adapter

## Overview

Auto-generated protocol adapter for converting between {config.source_protocol} and {config.target_protocol}.

- **Adapter Type**: {config.adapter_type.value}
- **Field Mappings**: {len(config.mappings)}
- **Error Handling**: {config.error_handling}

## Field Mappings

| Source Field | Target Field | Transformation | Required |
|--------------|--------------|----------------|----------|
"""
        for mapping in config.mappings:
            doc += f'| {mapping.source_field} | {mapping.target_field} | {mapping.transformation.value} | {"Yes" if mapping.required else "No"} |\n'

        doc += f"""
## Usage

```python
from {self._to_module_name(config.name)} import {self._to_class_name(config.name)}Adapter

adapter = {self._to_class_name(config.name)}Adapter()

# Transform source to target
target_data = adapter.transform(source_data)

# Transform target back to source
source_data = adapter.reverse_transform(target_data)
```

## PQC Considerations

When handling sensitive data, ensure PQC-ready encryption is applied:

- Use ML-KEM-768 for key encapsulation
- Use ML-DSA-65 for signatures
- Enable hybrid mode during transition

## Generated by Qbitel AI

This adapter was auto-generated by Qbitel AI Zero-Touch Automation.
"""
        return doc

    def _calculate_complexity(self, config: AdapterConfig) -> float:
        """Calculate adapter complexity score."""
        score = 0.0

        # Field count factor
        score += min(len(config.mappings) / 100, 0.3)

        # Transformation complexity
        for mapping in config.mappings:
            if mapping.transformation == TransformationType.DIRECT_MAP:
                score += 0.001
            elif mapping.transformation == TransformationType.FORMAT_CONVERSION:
                score += 0.005
            elif mapping.transformation == TransformationType.ENCODING_CONVERSION:
                score += 0.01
            elif mapping.transformation in (TransformationType.AGGREGATION, TransformationType.COMPUTED):
                score += 0.02

        return min(score, 1.0)

    def _to_class_name(self, name: str) -> str:
        """Convert name to PascalCase class name."""
        parts = re.split(r"[_\-\s]+", name)
        return "".join(p.capitalize() for p in parts)

    def _to_method_name(self, name: str) -> str:
        """Convert name to snake_case method name."""
        name = re.sub(r"[^\w]", "_", name)
        return name.lower()

    def _to_module_name(self, name: str) -> str:
        """Convert name to module name."""
        return name.lower().replace("-", "_").replace(" ", "_")

    def _load_transform_templates(self) -> Dict[str, str]:
        """Load transformation templates."""
        return {
            "date_iso_to_yymmdd": "datetime.strptime(value, '%Y-%m-%d').strftime('%y%m%d')",
            "yymmdd_to_date_iso": "datetime.strptime(value, '%y%m%d').strftime('%Y-%m-%d')",
            "amount_to_cents": "int(float(value) * 100)",
            "cents_to_amount": "float(value) / 100",
        }
