"""Unit tests for GraphQL/gRPC/example helpers."""

import base64
from datetime import datetime

from ai_engine.translation.models import ProtocolField, ProtocolSchema, ProtocolFormat
from ai_engine.translation.api_generation.schema_utils import (
    generate_examples,
    generate_graphql_assets,
    generate_grpc_assets,
)


def _sample_schema() -> ProtocolSchema:
    return ProtocolSchema(
        name="test_protocol",
        version="1.0",
        description="Sample schema",
        format=ProtocolFormat.BINARY,
        fields=[
            ProtocolField(name="message_type", field_type="integer", offset=0, length=1, description="Type"),
            ProtocolField(name="timestamp", field_type="timestamp", offset=1, length=8, description="Epoch"),
            ProtocolField(name="payload", field_type="string", offset=9, length=0, description="Payload", optional=True),
        ]
    )


def test_generate_graphql_assets_contains_core_sections():
    schema = _sample_schema()
    assets = generate_graphql_assets(schema, base_path="/api/test")

    assert "type TestProtocol" in assets["sdl"]
    assert "schema {" in assets["sdl"]
    assert assets["endpoint"].endswith("/graphql")
    assert "query GetTestProtocol" in assets["operations"]["query"]


def test_generate_grpc_assets_structure():
    schema = _sample_schema()
    assets = generate_grpc_assets(schema)

    assert "service TestProtocolService" in assets["proto"]
    assert "message TestProtocol" in assets["proto"]
    assert assets["service"] == "TestProtocolService"


def test_generate_examples_from_samples():
    schema = _sample_schema()
    timestamp = int(datetime(2023, 1, 1).timestamp())
    raw = bytes([1]) + timestamp.to_bytes(8, "big") + b"hello"
    examples = generate_examples(schema, [raw])

    assert len(examples) == 1
    example = examples[0]
    assert example["structured"]["message_type"] == 1
    assert example["structured"]["payload"] == "hello"
    assert example["raw"]["base64"] == base64.b64encode(raw).decode("ascii")


def test_generate_examples_synthetic_when_missing_samples():
    schema = _sample_schema()
    examples = generate_examples(schema, [])

    assert len(examples) == 1
    assert "structured" in examples[0]
    assert examples[0]["structured"]["message_type"] is not None
