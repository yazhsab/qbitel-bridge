"""
SDK Generator Core

Main SDK generator with support for multiple languages.
"""

import logging
import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
import yaml
import json

logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported SDK languages."""

    PYTHON = "python"
    TYPESCRIPT = "typescript"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    CSHARP = "csharp"


@dataclass
class GeneratorConfig:
    """Configuration for SDK generation."""

    # Package info
    package_name: str = "qbitel"
    package_version: str = "1.0.0"
    description: str = "QBITEL SDK"
    author: str = "QBITEL Team"
    license: str = "MIT"

    # API info
    api_base_url: str = "http://localhost:8000"
    api_version: str = "v1"

    # Generation options
    generate_tests: bool = True
    generate_docs: bool = True
    generate_examples: bool = True

    # Language-specific options
    language_options: Dict[Language, Dict[str, Any]] = field(default_factory=dict)

    # Custom templates
    template_dir: Optional[str] = None


@dataclass
class OpenAPISchema:
    """Parsed OpenAPI schema."""

    title: str
    version: str
    description: str
    servers: List[Dict[str, str]]
    paths: Dict[str, Dict[str, Any]]
    components: Dict[str, Any]
    tags: List[Dict[str, str]]

    @classmethod
    def from_file(cls, path: str) -> "OpenAPISchema":
        """Load from file."""
        path = Path(path)

        if path.suffix in [".yaml", ".yml"]:
            with open(path) as f:
                data = yaml.safe_load(f)
        else:
            with open(path) as f:
                data = json.load(f)

        info = data.get("info", {})
        return cls(
            title=info.get("title", "API"),
            version=info.get("version", "1.0.0"),
            description=info.get("description", ""),
            servers=data.get("servers", []),
            paths=data.get("paths", {}),
            components=data.get("components", {}),
            tags=data.get("tags", []),
        )


class BaseLanguageGenerator:
    """Base class for language-specific generators."""

    language: Language = None
    file_extension: str = ""

    def __init__(
        self,
        schema: OpenAPISchema,
        config: GeneratorConfig,
        output_dir: Path,
    ):
        self.schema = schema
        self.config = config
        self.output_dir = output_dir
        self.language_config = config.language_options.get(self.language, {})

    def generate(self) -> None:
        """Generate the SDK."""
        raise NotImplementedError

    def _ensure_dir(self, path: Path) -> None:
        """Ensure directory exists."""
        path.mkdir(parents=True, exist_ok=True)

    def _write_file(self, path: Path, content: str) -> None:
        """Write content to file."""
        self._ensure_dir(path.parent)
        with open(path, "w") as f:
            f.write(content)
        logger.debug(f"Generated: {path}")


class SDKGenerator:
    """
    Multi-language SDK generator.

    Parses OpenAPI specifications and generates type-safe SDKs
    for multiple programming languages.

    Example:
        generator = SDKGenerator("openapi.yaml")
        generator.generate(Language.GO, "./sdks/go")
        generator.generate(Language.RUST, "./sdks/rust")
    """

    def __init__(
        self,
        openapi_spec_path: str,
        config: Optional[GeneratorConfig] = None,
    ):
        """
        Initialize SDK generator.

        Args:
            openapi_spec_path: Path to OpenAPI specification
            config: Generator configuration
        """
        self.schema = OpenAPISchema.from_file(openapi_spec_path)
        self.config = config or GeneratorConfig()
        self._generators: Dict[Language, Type[BaseLanguageGenerator]] = {}

        # Register built-in generators
        self._register_generators()

        logger.info(f"SDK Generator initialized for {self.schema.title}")

    def _register_generators(self) -> None:
        """Register language generators."""
        from sdk_generator.languages.go import GoGenerator
        from sdk_generator.languages.rust import RustGenerator
        from sdk_generator.languages.java import JavaGenerator
        from sdk_generator.languages.csharp import CSharpGenerator

        self._generators[Language.GO] = GoGenerator
        self._generators[Language.RUST] = RustGenerator
        self._generators[Language.JAVA] = JavaGenerator
        self._generators[Language.CSHARP] = CSharpGenerator

    def generate(
        self,
        language: Language,
        output_dir: str,
        clean: bool = True,
    ) -> None:
        """
        Generate SDK for a specific language.

        Args:
            language: Target language
            output_dir: Output directory
            clean: Clean output directory first
        """
        output_path = Path(output_dir)

        if clean and output_path.exists():
            shutil.rmtree(output_path)

        generator_class = self._generators.get(language)
        if not generator_class:
            raise ValueError(f"No generator for language: {language}")

        generator = generator_class(self.schema, self.config, output_path)
        generator.generate()

        logger.info(f"Generated {language.value} SDK in {output_dir}")

    def generate_all(
        self,
        output_dir: str,
        languages: Optional[List[Language]] = None,
    ) -> None:
        """
        Generate SDKs for all languages.

        Args:
            output_dir: Base output directory
            languages: Languages to generate (all if None)
        """
        languages = languages or list(Language)

        for language in languages:
            try:
                lang_dir = os.path.join(output_dir, language.value)
                self.generate(language, lang_dir)
            except Exception as e:
                logger.error(f"Failed to generate {language.value} SDK: {e}")

    def register_generator(
        self,
        language: Language,
        generator_class: Type[BaseLanguageGenerator],
    ) -> None:
        """Register a custom generator."""
        self._generators[language] = generator_class


# Type mapping utilities
class TypeMapper:
    """Maps OpenAPI types to language-specific types."""

    # OpenAPI to language type mappings
    TYPE_MAPS = {
        Language.GO: {
            "string": "string",
            "integer": "int64",
            "number": "float64",
            "boolean": "bool",
            "array": "[]",
            "object": "map[string]interface{}",
            "null": "nil",
            # Formats
            "date": "time.Time",
            "date-time": "time.Time",
            "uuid": "string",
            "uri": "string",
            "email": "string",
            "binary": "[]byte",
        },
        Language.RUST: {
            "string": "String",
            "integer": "i64",
            "number": "f64",
            "boolean": "bool",
            "array": "Vec",
            "object": "HashMap<String, serde_json::Value>",
            "null": "Option",
            # Formats
            "date": "chrono::NaiveDate",
            "date-time": "chrono::DateTime<chrono::Utc>",
            "uuid": "uuid::Uuid",
            "uri": "String",
            "email": "String",
            "binary": "Vec<u8>",
        },
        Language.JAVA: {
            "string": "String",
            "integer": "Long",
            "number": "Double",
            "boolean": "Boolean",
            "array": "List",
            "object": "Map<String, Object>",
            "null": "null",
            # Formats
            "date": "LocalDate",
            "date-time": "OffsetDateTime",
            "uuid": "UUID",
            "uri": "URI",
            "email": "String",
            "binary": "byte[]",
        },
        Language.CSHARP: {
            "string": "string",
            "integer": "long",
            "number": "double",
            "boolean": "bool",
            "array": "List",
            "object": "Dictionary<string, object>",
            "null": "null",
            # Formats
            "date": "DateOnly",
            "date-time": "DateTimeOffset",
            "uuid": "Guid",
            "uri": "Uri",
            "email": "string",
            "binary": "byte[]",
        },
    }

    @classmethod
    def map_type(
        cls,
        language: Language,
        openapi_type: str,
        format: Optional[str] = None,
        items: Optional[Dict] = None,
    ) -> str:
        """Map OpenAPI type to language type."""
        type_map = cls.TYPE_MAPS.get(language, {})

        # Check format first
        if format and format in type_map:
            return type_map[format]

        base_type = type_map.get(openapi_type, "any")

        # Handle arrays
        if openapi_type == "array" and items:
            item_type = cls.map_type(
                language,
                items.get("type", "string"),
                items.get("format"),
            )
            if language == Language.GO:
                return f"[]{item_type}"
            elif language == Language.RUST:
                return f"Vec<{item_type}>"
            elif language == Language.JAVA:
                return f"List<{item_type}>"
            elif language == Language.CSHARP:
                return f"List<{item_type}>"

        return base_type

    @classmethod
    def map_schema_type(
        cls,
        language: Language,
        schema: Dict[str, Any],
    ) -> str:
        """Map OpenAPI schema to language type."""
        if "$ref" in schema:
            # Reference to component
            ref = schema["$ref"]
            type_name = ref.split("/")[-1]
            return cls._format_type_name(language, type_name)

        return cls.map_type(
            language,
            schema.get("type", "string"),
            schema.get("format"),
            schema.get("items"),
        )

    @classmethod
    def _format_type_name(cls, language: Language, name: str) -> str:
        """Format type name for language."""
        if language == Language.GO:
            # Go uses PascalCase
            return "".join(word.capitalize() for word in name.split("_"))
        elif language == Language.RUST:
            # Rust uses PascalCase for types
            return "".join(word.capitalize() for word in name.split("_"))
        elif language == Language.JAVA:
            # Java uses PascalCase
            return "".join(word.capitalize() for word in name.split("_"))
        elif language == Language.CSHARP:
            # C# uses PascalCase
            return "".join(word.capitalize() for word in name.split("_"))
        return name
