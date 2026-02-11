"""
Multi-Language SDK Generator

Generates type-safe, idiomatic SDKs for QBITEL API in multiple languages:
- Python (existing)
- TypeScript/JavaScript (existing)
- Go
- Rust
- Java
- C#

Features:
- OpenAPI specification parsing
- Language-specific code generation
- Type mapping and validation
- Async/await support where available
- Comprehensive error handling
- Unit test generation
- Documentation generation

Usage:
    from sdk_generator import SDKGenerator, Language

    generator = SDKGenerator(openapi_spec_path="openapi.yaml")

    # Generate Go SDK
    generator.generate(Language.GO, output_dir="./sdks/go")

    # Generate all SDKs
    generator.generate_all(output_dir="./sdks")
"""

from sdk_generator.generator import (
    SDKGenerator,
    Language,
    GeneratorConfig,
)
from sdk_generator.languages.go import GoGenerator
from sdk_generator.languages.rust import RustGenerator
from sdk_generator.languages.java import JavaGenerator
from sdk_generator.languages.csharp import CSharpGenerator

__all__ = [
    "SDKGenerator",
    "Language",
    "GeneratorConfig",
    "GoGenerator",
    "RustGenerator",
    "JavaGenerator",
    "CSharpGenerator",
]
