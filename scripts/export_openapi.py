#!/usr/bin/env python3
"""
Export OpenAPI specification from FastAPI application.

This script generates the OpenAPI (Swagger) specification file
for the CRONOS AI API.

Usage:
    python scripts/export_openapi.py
    python scripts/export_openapi.py --output docs/openapi.json
    python scripts/export_openapi.py --format yaml
"""

import argparse
import json
import sys
from pathlib import Path

# Add ai_engine to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_engine.api.rest import create_app
from ai_engine.core.config import Config


def export_openapi(output_file: str, format: str = "json"):
    """
    Export OpenAPI specification to file.

    Args:
        output_file: Path to output file
        format: Output format ('json' or 'yaml')
    """
    print(f"Generating OpenAPI specification...")

    # Create minimal config for app initialization
    config = Config()
    config.debug = True  # Enable docs endpoints

    # Create FastAPI app
    app = create_app(config)

    # Get OpenAPI schema
    openapi_schema = app.openapi()

    # Write to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(output_path, "w") as f:
            json.dump(openapi_schema, f, indent=2, sort_keys=False)
        print(f"✓ OpenAPI JSON specification exported to: {output_path}")
    elif format == "yaml":
        try:
            import yaml

            with open(output_path, "w") as f:
                yaml.dump(openapi_schema, f, default_flow_style=False, sort_keys=False)
            print(f"✓ OpenAPI YAML specification exported to: {output_path}")
        except ImportError:
            print(
                "ERROR: PyYAML is required for YAML output. Install with: pip install pyyaml"
            )
            sys.exit(1)
    else:
        print(f"ERROR: Unsupported format: {format}")
        sys.exit(1)

    # Print summary
    print(f"\nOpenAPI Specification Summary:")
    print(f"  Title: {openapi_schema.get('info', {}).get('title')}")
    print(f"  Version: {openapi_schema.get('info', {}).get('version')}")
    print(f"  Endpoints: {len(openapi_schema.get('paths', {}))}")
    print(f"  Tags: {len(openapi_schema.get('tags', []))}")
    print(f"  Schemas: {len(openapi_schema.get('components', {}).get('schemas', {}))}")

    # Print access URLs
    print(f"\nDocumentation URLs (when server is running):")
    print(f"  Swagger UI: http://localhost:8000/docs")
    print(f"  ReDoc: http://localhost:8000/redoc")
    print(f"  OpenAPI JSON: http://localhost:8000/openapi.json")

    return openapi_schema


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export OpenAPI specification for CRONOS AI API"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="docs/openapi.json",
        help="Output file path (default: docs/openapi.json)",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["json", "yaml"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate the OpenAPI specification"
    )

    args = parser.parse_args()

    try:
        schema = export_openapi(args.output, args.format)

        if args.validate:
            print("\nValidating OpenAPI specification...")
            try:
                from openapi_spec_validator import validate_spec

                validate_spec(schema)
                print("✓ OpenAPI specification is valid!")
            except ImportError:
                print(
                    "WARNING: openapi-spec-validator not installed. Skipping validation."
                )
                print("Install with: pip install openapi-spec-validator")
            except Exception as e:
                print(f"✗ OpenAPI specification validation failed: {e}")
                sys.exit(1)

        print("\n✓ Export completed successfully!")

    except Exception as e:
        print(f"ERROR: Failed to export OpenAPI specification: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
