"""Utility builders for GraphQL, gRPC, and example generation.

These helpers are shared between the API generator and the discovery orchestrator
so that protocol-derived APIs expose consistent secondary artefacts.
"""

from __future__ import annotations

import base64
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..models import ProtocolField, ProtocolSchema, SecurityLevel

GRAPHQL_SCALAR_OVERRIDES = {
    "timestamp": "DateTime",
    "boolean": "Boolean",
    "float": "Float",
    "double": "Float",
    "binary": "Base64String",
}

PROTOBUF_TYPE_OVERRIDES = {
    "boolean": "bool",
    "float": "float",
    "double": "double",
    "binary": "bytes",
    "string": "string",
    "timestamp": "google.protobuf.Timestamp",
}

_IDENTIFIER_RE = re.compile(r"[^0-9A-Za-z_]")


def _to_pascal_case(value: str) -> str:
    parts = re.split(r"[^0-9A-Za-z]+", value)
    capitalised = [p.capitalize() for p in parts if p]
    result = "".join(capitalised) or "Protocol"
    if result[0].isdigit():
        result = f"P{result}"
    return result


def _to_camel_case(value: str) -> str:
    parts = re.split(r"[^0-9A-Za-z]+", value)
    if not parts:
        return "field"
    head, *tail = parts
    if not head:
        head = "field"
    identifier = head.lower() + "".join(p.capitalize() for p in tail if p)
    identifier = _IDENTIFIER_RE.sub("", identifier)
    if not identifier:
        identifier = "field"
    if identifier[0].isdigit():
        identifier = f"f{identifier}"
    return identifier


def _sanitize_proto_identifier(value: str) -> str:
    cleaned = _IDENTIFIER_RE.sub("", value)
    if not cleaned:
        cleaned = "Field"
    if cleaned[0].isdigit():
        cleaned = f"F{cleaned}"
    return cleaned


def _map_graphql_type(field: ProtocolField) -> Tuple[str, Optional[str]]:
    field_type = (field.field_type or "string").lower()
    gql_type = GRAPHQL_SCALAR_OVERRIDES.get(field_type)
    scalar_required = gql_type in {"DateTime", "Base64String"}
    if not gql_type:
        if field_type in {"integer", "length", "checksum"}:
            gql_type = "Int"
        elif field_type == "float" or field_type == "double":
            gql_type = "Float"
        elif field_type == "boolean":
            gql_type = "Boolean"
        else:
            gql_type = "String"
    scalar_name = gql_type if scalar_required else None
    return gql_type, scalar_name


def _default_value_for_field(field: ProtocolField) -> Any:
    field_type = (field.field_type or "string").lower()
    if field_type in {"integer", "length", "checksum"}:
        return field.length or 0
    if field_type == "boolean":
        return True
    if field_type == "timestamp":
        return datetime.now(timezone.utc).isoformat()
    if field_type in {"float", "double"}:
        return round(math.pi, 3)
    if field_type == "binary":
        return base64.b64encode(b"sample").decode("ascii")
    if field_type == "address":
        return "127.0.0.1"
    return f"example_{field.name}"


def generate_graphql_assets(
    protocol_schema: ProtocolSchema,
    base_path: str = "/api/v1",
    security_level: Optional[SecurityLevel] = None
) -> Dict[str, Any]:
    """Create GraphQL SDL and example operations for a protocol schema."""
    type_name = _to_pascal_case(protocol_schema.name)
    input_name = f"{type_name}Input"
    filter_name = f"{type_name}Filter"
    delete_payload = "DeleteResult"

    scalar_declarations: Dict[str, str] = {}
    field_lines: List[str] = []
    input_lines: List[str] = []
    filter_lines: List[str] = []

    for field in protocol_schema.fields:
        gql_type, scalar = _map_graphql_type(field)
        if scalar:
            if scalar == "DateTime":
                scalar_declarations[scalar] = "scalar DateTime"
            elif scalar == "Base64String":
                scalar_declarations[scalar] = "scalar Base64String"
        field_name = _to_camel_case(field.name)
        required = "!" if not getattr(field, "optional", False) else ""
        description_comment = f" # {field.description}" if field.description else ""
        field_lines.append(f"  {field_name}: {gql_type}{required}{description_comment}")
        input_lines.append(f"  {field_name}: {gql_type}{description_comment}")
        filter_lines.append(f"  {field_name}: {gql_type}")

    type_definition = "\n".join([f"type {type_name} {{"] + field_lines + ["}"])
    input_definition = "\n".join([f"input {input_name} {{"] + input_lines + ["}"])
    filter_definition = "\n".join([f"input {filter_name} {{"] + filter_lines + ["}"])

    query_definition = (
        "type Query {\n"
        f"  get{type_name}(id: ID!): {type_name}\n"
        f"  list{type_name}(filter: {filter_name}, limit: Int = 50, offset: Int = 0): [{type_name}!]!\n"
        "}"
    )

    mutation_definition = (
        "type Mutation {\n"
        f"  create{type_name}(input: {input_name}!): {type_name}!\n"
        f"  update{type_name}(id: ID!, input: {input_name}!): {type_name}!\n"
        f"  delete{type_name}(id: ID!): {delete_payload}!\n"
        "}"
    )

    subscription_definition = (
        "type Subscription {\n"
        f"  {protocol_schema.name.lower()}Created: {type_name}!\n"
        f"  {protocol_schema.name.lower()}Updated: {type_name}!\n"
        f"  {protocol_schema.name.lower()}Deleted: {delete_payload}!\n"
        "}"
    )

    delete_payload_definition = (
        f"type {delete_payload} {{\n"
        "  id: ID!\n"
        "  deleted: Boolean!\n"
        "}"
    )

    schema_root = "schema {\n  query: Query\n  mutation: Mutation\n  subscription: Subscription\n}"

    sdl_parts: List[str] = []
    if scalar_declarations:
        sdl_parts.extend(scalar_declarations.values())
    sdl_parts.extend([
        type_definition,
        input_definition,
        filter_definition,
        delete_payload_definition,
        query_definition,
        mutation_definition,
        subscription_definition,
        schema_root,
    ])
    sdl = "\n\n".join(sdl_parts)

    field_selection = "\n    ".join(_to_camel_case(field.name) for field in protocol_schema.fields)
    sample_query = (
        "query Get{type_name}($id: ID!) {{\n"
        "  get{type_name}(id: $id) {{\n    {fields}\n  }}\n"
        "}}"
    ).format(type_name=type_name, fields=field_selection)
    sample_mutation = (
        "mutation Create{type_name}($input: {input_name}!) {{\n"
        "  create{type_name}(input: $input) {{\n    {fields}\n  }}\n"
        "}}"
    ).format(type_name=type_name, input_name=input_name, fields=field_selection)

    headers: Dict[str, str] = {}
    if security_level and security_level != SecurityLevel.PUBLIC:
        headers["Authorization"] = "Bearer <token>"

    return {
        "endpoint": f"{base_path.rstrip('/')}/graphql",
        "sdl": sdl,
        "operations": {
            "query": sample_query,
            "mutation": sample_mutation,
            "subscription": f"subscription {{ {protocol_schema.name.lower()}Created {{ id }} }}",
        },
        "headers": headers,
    }


def generate_grpc_assets(protocol_schema: ProtocolSchema) -> Dict[str, Any]:
    """Create protobuf service definition for a protocol schema."""
    type_name = _to_pascal_case(protocol_schema.name)
    package_name = f"cronos.{_sanitize_proto_identifier(protocol_schema.name.lower())}"
    service_name = f"{type_name}Service"

    imports: List[str] = []
    message_fields: List[str] = []
    field_number = 1

    for field in protocol_schema.fields:
        field_type = (field.field_type or "string").lower()
        proto_type = PROTOBUF_TYPE_OVERRIDES.get(field_type)
        if not proto_type:
            if field_type in {"integer", "length", "checksum"}:
                proto_type = "uint64" if (field.length or 4) > 4 else "uint32"
            elif field_type == "float":
                proto_type = "float"
            else:
                proto_type = "string"
        if proto_type == "google.protobuf.Timestamp":
            import_stmt = 'import "google/protobuf/timestamp.proto";'
            if import_stmt not in imports:
                imports.append(import_stmt)
        field_name = _sanitize_proto_identifier(field.name.lower())
        message_fields.append(f"  {proto_type} {field_name} = {field_number};")
        field_number += 1

    message_definition = "\n".join([f"message {type_name} {{"] + message_fields + ["}"])

    proto_template = [
        'syntax = "proto3";',
        f'package {package_name};'
    ]
    proto_template.extend(imports)
    proto_template.append("")
    proto_template.append(message_definition)
    proto_template.append("")
    proto_template.extend(
        [
            f"message Get{type_name}Request {{\n  string id = 1;\n}}",
            f"message List{type_name}Request {{\n  uint32 page_size = 1;\n  string page_token = 2;\n}}",
            f"message List{type_name}Response {{\n  repeated {type_name} items = 1;\n  string next_page_token = 2;\n}}",
            f"message Mutate{type_name}Request {{\n  {type_name} payload = 1;\n}}",
            f"message Delete{type_name}Request {{\n  string id = 1;\n}}",
            f"message Delete{type_name}Response {{\n  string id = 1;\n  bool deleted = 2;\n}}",
            f"message Stream{type_name}Request {{\n  string filter = 1;\n}}",
        ]
    )
    proto_template.append("")
    proto_template.append(
        f"service {service_name} {{\n"
        f"  rpc Get(Get{type_name}Request) returns ({type_name});\n"
        f"  rpc List(List{type_name}Request) returns (List{type_name}Response);\n"
        f"  rpc Create(Mutate{type_name}Request) returns ({type_name});\n"
        f"  rpc Update(Mutate{type_name}Request) returns ({type_name});\n"
        f"  rpc Delete(Delete{type_name}Request) returns (Delete{type_name}Response);\n"
        f"  rpc Stream(Stream{type_name}Request) returns (stream {type_name});\n"
        "}"
    )

    proto = "\n".join(proto_template)
    return {
        "package": package_name,
        "service": service_name,
        "proto": proto,
    }


def _slice_field(field: ProtocolField, message: bytes) -> bytes:
    start = max(field.offset, 0)
    if start >= len(message):
        return b""
    length = field.length if field.length and field.length > 0 else len(message) - start
    end = min(len(message), start + length)
    return message[start:end]


def _interpret_field(field: ProtocolField, raw: bytes) -> Any:
    field_type = (field.field_type or "string").lower()
    if not raw:
        return None
    if field_type in {"integer", "length", "checksum"}:
        return int.from_bytes(raw, byteorder="big", signed=False)
    if field_type == "boolean":
        return bool(raw[-1] & 0x01)
    if field_type == "timestamp":
        try:
            epoch = int.from_bytes(raw, byteorder="big", signed=False)
            return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()
        except Exception:
            return int.from_bytes(raw, byteorder="big", signed=False)
    if field_type in {"float", "double"} and len(raw) in {4, 8}:
        import struct
        fmt = "!f" if len(raw) == 4 else "!d"
        try:
            return struct.unpack(fmt, raw)[0]
        except Exception:
            return None
    try:
        decoded = raw.decode("utf-8", errors="replace").strip("\x00")
        if decoded:
            return decoded
    except Exception:
        pass
    return raw.hex()


def generate_examples(
    protocol_schema: ProtocolSchema,
    sample_messages: Optional[Sequence[bytes]] = None,
    *,
    limit: int = 3
) -> List[Dict[str, Any]]:
    """Generate structured example payloads for a protocol schema."""
    examples: List[Dict[str, Any]] = []
    messages = list(sample_messages or [])[:limit] if sample_messages else []

    if not messages:
        synthetic = {
            field.name: _default_value_for_field(field)
            for field in protocol_schema.fields
        }
        examples.append(
            {
                "id": "synthetic",
                "summary": "Synthetic example derived from schema",
                "structured": synthetic,
                "raw": None,
                "fields": {},
            }
        )
        return examples

    for idx, message in enumerate(messages[:limit]):
        field_details: Dict[str, Any] = {}
        structured: Dict[str, Any] = {}
        for field in protocol_schema.fields:
            raw_slice = _slice_field(field, message)
            interpreted = _interpret_field(field, raw_slice)
            structured[field.name] = interpreted
            field_details[field.name] = {
                "offset": field.offset,
                "length": field.length,
                "raw_hex": raw_slice.hex(),
                "raw_base64": base64.b64encode(raw_slice).decode("ascii") if raw_slice else None,
                "interpreted": interpreted,
            }
        examples.append(
            {
                "id": f"example_{idx+1}",
                "summary": f"Observed sample #{idx+1}",
                "raw": {
                    "base64": base64.b64encode(message).decode("ascii"),
                    "hex": message.hex(),
                    "length": len(message),
                },
                "fields": field_details,
                "structured": structured,
            }
        )

    return examples


__all__ = [
    "generate_graphql_assets",
    "generate_grpc_assets",
    "generate_examples",
]
