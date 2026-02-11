"""
QBITEL - Translation Studio Models Tests
Enterprise-grade unit tests for translation studio data models and structures.
"""

import pytest
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any

from ai_engine.translation.models import (
    ProtocolSchema,
    APISpecification,
    GeneratedSDK,
    TranslationRequest,
    CodeLanguage,
    APIStyle,
    SecurityLevel,
    GenerationStatus,
    TranslationMode,
    QualityLevel,
    FieldType,
    create_protocol_schema,
    create_api_specification,
    validate_field_definition,
    calculate_schema_complexity,
)
from ai_engine.translation.exceptions import (
    ValidationException,
    SchemaValidationException,
)
from ai_engine.translation.models import TranslationResult


class TestProtocolSchema:
    """Test cases for ProtocolSchema model."""

    def test_protocol_schema_creation(self, sample_protocol_schema):
        """Test creating a protocol schema."""
        schema = sample_protocol_schema

        assert schema.name == "TestProtocol"
        assert schema.version == "1.0"
        assert len(schema.fields) == 4
        assert schema.semantic_info["domain"] == "messaging"

    def test_protocol_schema_validation(self):
        """Test protocol schema validation."""
        # Valid schema
        valid_fields = [
            {
                "name": "header",
                "type": FieldType.INTEGER.value,
                "size": 4,
                "description": "Message header",
            }
        ]

        schema = ProtocolSchema(
            name="ValidProtocol", version="1.0", fields=valid_fields
        )

        validation_errors = schema.validate()
        assert len(validation_errors) == 0

    def test_protocol_schema_invalid_fields(self):
        """Test protocol schema with invalid fields."""
        invalid_fields = [
            {
                "name": "",  # Invalid: empty name
                "type": "invalid_type",  # Invalid: unknown type
                "size": -1,  # Invalid: negative size
                "description": "Invalid field",
            }
        ]

        schema = ProtocolSchema(
            name="InvalidProtocol", version="1.0", fields=invalid_fields
        )

        validation_errors = schema.validate()
        assert len(validation_errors) > 0
        assert any("name" in error for error in validation_errors)
        assert any("type" in error for error in validation_errors)
        assert any("size" in error for error in validation_errors)

    def test_protocol_schema_to_dict(self, sample_protocol_schema):
        """Test converting protocol schema to dictionary."""
        schema_dict = sample_protocol_schema.to_dict()

        assert isinstance(schema_dict, dict)
        assert schema_dict["name"] == "TestProtocol"
        assert schema_dict["version"] == "1.0"
        assert "fields" in schema_dict
        assert "semantic_info" in schema_dict

    def test_protocol_schema_from_dict(self, sample_protocol_schema):
        """Test creating protocol schema from dictionary."""
        schema_dict = sample_protocol_schema.to_dict()
        reconstructed_schema = ProtocolSchema.from_dict(schema_dict)

        assert reconstructed_schema.name == sample_protocol_schema.name
        assert reconstructed_schema.version == sample_protocol_schema.version
        assert len(reconstructed_schema.fields) == len(sample_protocol_schema.fields)

    def test_calculate_complexity(self, sample_protocol_schema):
        """Test schema complexity calculation."""
        complexity = calculate_schema_complexity(sample_protocol_schema)

        assert isinstance(complexity, float)
        assert 0.0 <= complexity <= 1.0

        # Schema with variable-length fields should have higher complexity
        assert complexity > 0.5  # Has variable-length field

    @pytest.mark.parametrize(
        "field_type,expected_valid",
        [
            (FieldType.INTEGER, True),
            (FieldType.STRING, True),
            (FieldType.BINARY, True),
            (FieldType.BOOLEAN, True),
            (FieldType.FLOAT, True),
            ("invalid_type", False),
        ],
    )
    def test_field_type_validation(self, field_type, expected_valid):
        """Test field type validation."""
        field_def = {
            "name": "test_field",
            "type": field_type.value if hasattr(field_type, "value") else field_type,
            "size": 4,
            "description": "Test field",
        }

        is_valid = validate_field_definition(field_def)
        assert is_valid == expected_valid


class TestAPISpecification:
    """Test cases for APISpecification model."""

    def test_api_specification_creation(self, sample_api_specification):
        """Test creating an API specification."""
        spec = sample_api_specification

        assert spec.title == "Test Protocol API"
        assert spec.version == "1.0.0"
        assert spec.api_style == APIStyle.REST
        assert spec.security_level == SecurityLevel.AUTHENTICATED
        assert len(spec.endpoints) == 2

    def test_api_specification_validation(self, sample_api_specification):
        """Test API specification validation."""
        validation_errors = sample_api_specification.validate()
        assert len(validation_errors) == 0

    def test_api_specification_invalid_endpoints(self):
        """Test API specification with invalid endpoints."""
        invalid_spec = APISpecification(
            spec_id=str(uuid.uuid4()),
            title="Invalid API",
            version="1.0.0",
            endpoints=[
                {
                    "path": "",  # Invalid: empty path
                    "method": "INVALID",  # Invalid: unknown method
                    "operation_id": "",  # Invalid: empty operation ID
                }
            ],
        )

        validation_errors = invalid_spec.validate()
        assert len(validation_errors) > 0

    def test_api_specification_to_openapi_dict(self, sample_api_specification):
        """Test converting API specification to OpenAPI dictionary."""
        openapi_dict = sample_api_specification.to_openapi_dict()

        assert isinstance(openapi_dict, dict)
        assert openapi_dict["openapi"] == "3.0.0"
        assert openapi_dict["info"]["title"] == "Test Protocol API"
        assert "paths" in openapi_dict
        assert "components" in openapi_dict

    def test_api_specification_security_schemes(self, sample_api_specification):
        """Test API specification security schemes generation."""
        openapi_dict = sample_api_specification.to_openapi_dict()

        # Should include security schemes for authenticated API
        assert "components" in openapi_dict
        assert "securitySchemes" in openapi_dict["components"]
        assert "bearerAuth" in openapi_dict["components"]["securitySchemes"]

    def test_api_specification_extensions(self, sample_api_specification):
        """Ensure custom extensions are surfaced in OpenAPI output."""
        sample_api_specification.extensions = {"graphql": {"endpoint": "/graphql"}}
        openapi_dict = sample_api_specification.to_openapi_dict()

        assert "x-qbitel-extensions" in openapi_dict
        assert "graphql" in openapi_dict["x-qbitel-extensions"]

    def test_create_api_specification_factory(self, sample_protocol_schema):
        """Test API specification factory function."""
        spec = create_api_specification(
            protocol_schema=sample_protocol_schema,
            api_style=APIStyle.REST,
            security_level=SecurityLevel.PUBLIC,
        )

        assert isinstance(spec, APISpecification)
        assert spec.title == f"{sample_protocol_schema.name} API"
        assert spec.api_style == APIStyle.REST
        assert spec.security_level == SecurityLevel.PUBLIC

    @pytest.mark.parametrize(
        "api_style,expected_paths",
        [
            (APIStyle.REST, ["/messages", "/messages/{id}"]),
            (APIStyle.GRAPHQL, ["/graphql"]),
            (APIStyle.GRPC, []),  # gRPC doesn't use HTTP paths
        ],
    )
    def test_api_style_specific_generation(
        self, sample_protocol_schema, api_style, expected_paths
    ):
        """Test API generation for different styles."""
        spec = create_api_specification(
            protocol_schema=sample_protocol_schema,
            api_style=api_style,
            security_level=SecurityLevel.PUBLIC,
        )

        if api_style == APIStyle.GRPC:
            # gRPC specs don't have HTTP endpoints
            assert len(spec.endpoints) == 0
        else:
            endpoint_paths = [endpoint.path for endpoint in spec.endpoints]
            for expected_path in expected_paths:
                assert any(expected_path in path for path in endpoint_paths)


class TestGeneratedSDK:
    """Test cases for GeneratedSDK model."""

    def test_generated_sdk_creation(self, sample_generated_sdk):
        """Test creating a generated SDK."""
        sdk = sample_generated_sdk

        assert sdk.language == CodeLanguage.PYTHON
        assert sdk.name == "test-protocol-python-sdk"
        assert sdk.version == "1.0.0"
        assert len(sdk.source_files) == 3
        assert len(sdk.test_files) == 1
        assert len(sdk.config_files) == 2

    def test_generated_sdk_validation(self, sample_generated_sdk):
        """Test generated SDK validation."""
        validation_errors = sample_generated_sdk.validate()
        assert len(validation_errors) == 0

    def test_generated_sdk_invalid_files(self):
        """Test generated SDK with invalid files."""
        invalid_sdk = GeneratedSDK(
            sdk_id=str(uuid.uuid4()),
            name="",  # Invalid: empty name
            language=CodeLanguage.PYTHON,
            version="invalid-version",  # Invalid: not semver
            source_files={},  # Invalid: no source files
            test_files={},
            config_files={},
        )

        validation_errors = invalid_sdk.validate()
        assert len(validation_errors) > 0

    def test_generated_sdk_to_dict(self, sample_generated_sdk):
        """Test converting generated SDK to dictionary."""
        sdk_dict = sample_generated_sdk.to_dict()

        assert isinstance(sdk_dict, dict)
        assert sdk_dict["name"] == "test-protocol-python-sdk"
        assert sdk_dict["language"] == CodeLanguage.PYTHON.value
        assert "source_files" in sdk_dict
        assert "test_files" in sdk_dict

    def test_generated_sdk_file_count(self, sample_generated_sdk):
        """Test generated SDK file count calculation."""
        total_files = sample_generated_sdk.total_files

        expected_count = (
            len(sample_generated_sdk.source_files)
            + len(sample_generated_sdk.test_files)
            + len(sample_generated_sdk.config_files)
            + len(sample_generated_sdk.documentation_files)
        )

        assert total_files == expected_count

    @pytest.mark.parametrize(
        "language,expected_extensions",
        [
            (CodeLanguage.PYTHON, [".py"]),
            (CodeLanguage.TYPESCRIPT, [".ts"]),
            (CodeLanguage.JAVASCRIPT, [".js"]),
            (CodeLanguage.GO, [".go"]),
            (CodeLanguage.JAVA, [".java"]),
        ],
    )
    def test_generated_sdk_language_extensions(self, language, expected_extensions):
        """Test generated SDK file extensions for different languages."""
        sdk = GeneratedSDK(
            sdk_id=str(uuid.uuid4()),
            name=f"test-{language.value}-sdk",
            language=language,
            version="1.0.0",
            source_files={f"client{ext}": "code" for ext in expected_extensions},
            test_files={
                f"test_client{ext}": "test code" for ext in expected_extensions
            },
            config_files={"package.json": "config"},
        )

        # Check that source files have correct extensions
        for filename in sdk.source_files.keys():
            assert any(filename.endswith(ext) for ext in expected_extensions)


class TestTranslationRequest:
    """Test cases for TranslationRequest model."""

    def test_translation_request_creation(self, sample_translation_request):
        """Test creating a translation request."""
        request = sample_translation_request

        assert request.source_protocol == "HTTP"
        assert request.target_protocol == "WebSocket"
        assert isinstance(request.data, bytes)
        assert "user_id" in request.metadata

    def test_translation_request_validation(self, sample_translation_request):
        """Test translation request validation."""
        validation_errors = sample_translation_request.validate()
        assert len(validation_errors) == 0

    def test_translation_request_invalid_protocols(self):
        """Test translation request with invalid protocols."""
        invalid_request = TranslationRequest(
            request_id=str(uuid.uuid4()),
            source_protocol="",  # Invalid: empty protocol
            target_protocol="",  # Invalid: empty protocol
            data=b"test data",
        )

        validation_errors = invalid_request.validate()
        assert len(validation_errors) > 0

    def test_translation_request_to_dict(self, sample_translation_request):
        """Test converting translation request to dictionary."""
        request_dict = sample_translation_request.to_dict()

        assert isinstance(request_dict, dict)
        assert request_dict["source_protocol"] == "HTTP"
        assert request_dict["target_protocol"] == "WebSocket"
        assert "request_id" in request_dict
        assert "metadata" in request_dict


class TestTranslationResult:
    """Test cases for TranslationResult model."""

    def test_translation_result_creation(self, sample_translation_result):
        """Test creating a translation result."""
        result = sample_translation_result

        assert result.source_protocol == "HTTP"
        assert result.target_protocol == "WebSocket"
        assert isinstance(result.translated_data, bytes)
        assert 0.0 <= result.confidence <= 1.0
        assert result.processing_time > 0

    def test_translation_result_validation(self, sample_translation_result):
        """Test translation result validation."""
        validation_errors = sample_translation_result.validate()
        assert len(validation_errors) == 0

    def test_translation_result_invalid_confidence(self):
        """Test translation result with invalid confidence score."""
        invalid_result = TranslationResult(
            translation_id=str(uuid.uuid4()),
            source_protocol="HTTP",
            target_protocol="WebSocket",
            translated_data=b"translated data",
            confidence=1.5,  # Invalid: > 1.0
            processing_time=0.1,
        )

        validation_errors = invalid_result.validate()
        assert len(validation_errors) > 0
        assert any("confidence" in error for error in validation_errors)

    def test_translation_result_quality_assessment(self, sample_translation_result):
        """Test translation result quality assessment."""
        # High confidence should be good quality
        assert sample_translation_result.confidence > 0.8
        assert len(sample_translation_result.validation_errors) == 0

        # Result should be considered successful
        is_successful = (
            sample_translation_result.confidence > 0.7
            and len(sample_translation_result.validation_errors) == 0
        )
        assert is_successful


class TestEnumValues:
    """Test cases for enum values and validation."""

    def test_code_language_enum(self):
        """Test CodeLanguage enum values."""
        assert CodeLanguage.PYTHON.value == "python"
        assert CodeLanguage.TYPESCRIPT.value == "typescript"
        assert CodeLanguage.JAVASCRIPT.value == "javascript"
        assert CodeLanguage.GO.value == "go"
        assert CodeLanguage.JAVA.value == "java"

    def test_api_style_enum(self):
        """Test APIStyle enum values."""
        assert APIStyle.REST.value == "rest"
        assert APIStyle.GRAPHQL.value == "graphql"
        assert APIStyle.GRPC.value == "grpc"
        assert APIStyle.WEBSOCKET.value == "websocket"

    def test_security_level_enum(self):
        """Test SecurityLevel enum values."""
        assert SecurityLevel.PUBLIC.value == "public"
        assert SecurityLevel.AUTHENTICATED.value == "authenticated"
        assert SecurityLevel.AUTHORIZED.value == "authorized"

    def test_field_type_enum(self):
        """Test FieldType enum values."""
        assert FieldType.INTEGER.value == "integer"
        assert FieldType.STRING.value == "string"
        assert FieldType.BINARY.value == "binary"
        assert FieldType.BOOLEAN.value == "boolean"
        assert FieldType.FLOAT.value == "float"

    def test_generation_status_enum(self):
        """Test GenerationStatus enum values."""
        assert GenerationStatus.PENDING.value == "pending"
        assert GenerationStatus.IN_PROGRESS.value == "in_progress"
        assert GenerationStatus.COMPLETED.value == "completed"
        assert GenerationStatus.FAILED.value == "failed"

    @pytest.mark.parametrize(
        "enum_class",
        [
            CodeLanguage,
            APIStyle,
            SecurityLevel,
            FieldType,
            GenerationStatus,
            TranslationMode,
            QualityLevel,
        ],
    )
    def test_enum_completeness(self, enum_class):
        """Test that enums have at least one value."""
        assert len(list(enum_class)) > 0

        # Test that all enum values are strings
        for enum_value in enum_class:
            assert isinstance(enum_value.value, str)
            assert len(enum_value.value) > 0


class TestFactoryFunctions:
    """Test cases for factory functions."""

    def test_create_protocol_schema(self):
        """Test protocol schema factory function."""
        name = "TestProtocol"
        fields = [
            {
                "name": "header",
                "type": FieldType.INTEGER.value,
                "size": 4,
                "description": "Message header",
            }
        ]

        schema = create_protocol_schema(name, fields)

        assert isinstance(schema, ProtocolSchema)
        assert schema.name == name
        assert len(schema.fields) == 1
        assert schema.schema_id is not None

    def test_create_protocol_schema_with_metadata(self):
        """Test protocol schema factory with metadata."""
        schema = create_protocol_schema(
            name="TestProtocol",
            fields=[],
            version="2.0",
            description="Test protocol with metadata",
            semantic_info={"domain": "testing"},
        )

        assert schema.version == "2.0"
        assert schema.description == "Test protocol with metadata"
        assert schema.semantic_info["domain"] == "testing"

    def test_validate_field_definition_valid(self):
        """Test field definition validation with valid field."""
        valid_field = {
            "name": "test_field",
            "type": FieldType.STRING.value,
            "description": "A test field",
            "variable_length": True,
        }

        is_valid = validate_field_definition(valid_field)
        assert is_valid is True

    def test_validate_field_definition_invalid(self):
        """Test field definition validation with invalid field."""
        invalid_field = {
            "name": "",  # Invalid: empty name
            "type": "invalid_type",  # Invalid: unknown type
            "size": -1,  # Invalid: negative size
        }

        is_valid = validate_field_definition(invalid_field)
        assert is_valid is False


class TestModelSerialization:
    """Test cases for model serialization and deserialization."""

    def test_protocol_schema_json_serialization(self, sample_protocol_schema):
        """Test JSON serialization of protocol schema."""
        import json

        schema_dict = sample_protocol_schema.to_dict()
        json_str = json.dumps(schema_dict)

        # Should be valid JSON
        parsed_dict = json.loads(json_str)
        assert parsed_dict["name"] == sample_protocol_schema.name

    def test_api_specification_openapi_serialization(self, sample_api_specification):
        """Test OpenAPI serialization of API specification."""
        import json

        openapi_dict = sample_api_specification.to_openapi_dict()
        json_str = json.dumps(openapi_dict)

        # Should be valid JSON and valid OpenAPI
        parsed_dict = json.loads(json_str)
        assert parsed_dict["openapi"] == "3.0.0"
        assert "info" in parsed_dict
        assert "paths" in parsed_dict

    def test_generated_sdk_serialization(self, sample_generated_sdk):
        """Test serialization of generated SDK."""
        import json

        sdk_dict = sample_generated_sdk.to_dict()
        json_str = json.dumps(sdk_dict)

        # Should be valid JSON
        parsed_dict = json.loads(json_str)
        assert parsed_dict["name"] == sample_generated_sdk.name
        assert parsed_dict["language"] == sample_generated_sdk.language.value


class TestModelEdgeCases:
    """Test cases for model edge cases and error conditions."""

    def test_protocol_schema_empty_fields(self):
        """Test protocol schema with no fields."""
        schema = ProtocolSchema(name="EmptyProtocol", version="1.0", fields=[])

        validation_errors = schema.validate()
        assert len(validation_errors) > 0
        assert any("fields" in error for error in validation_errors)

    def test_api_specification_duplicate_endpoints(self):
        """Test API specification with duplicate endpoints."""
        duplicate_endpoints = [
            {"path": "/messages", "method": "GET", "operation_id": "get_messages_1"},
            {"path": "/messages", "method": "GET", "operation_id": "get_messages_2"},
        ]

        spec = APISpecification(
            spec_id=str(uuid.uuid4()),
            title="Duplicate API",
            version="1.0.0",
            endpoints=duplicate_endpoints,
        )

        validation_errors = spec.validate()
        assert len(validation_errors) > 0

    def test_generated_sdk_empty_source_files(self):
        """Test generated SDK with no source files."""
        sdk = GeneratedSDK(
            sdk_id=str(uuid.uuid4()),
            name="empty-sdk",
            language=CodeLanguage.PYTHON,
            version="1.0.0",
            source_files={},  # Empty source files
            test_files={},
            config_files={},
        )

        validation_errors = sdk.validate()
        assert len(validation_errors) > 0

    def test_translation_request_large_data(self):
        """Test translation request with large data payload."""
        large_data = b"x" * (10 * 1024 * 1024)  # 10MB

        request = TranslationRequest(
            request_id=str(uuid.uuid4()),
            source_protocol="HTTP",
            target_protocol="WebSocket",
            data=large_data,
        )

        # Should handle large data gracefully
        assert len(request.data) == 10 * 1024 * 1024

        # Validation should pass for large but reasonable data
        validation_errors = request.validate()
        # May have warnings but should not fail completely
        assert isinstance(validation_errors, list)


if __name__ == "__main__":
    pytest.main([__file__])
