#!/usr/bin/env python3
"""
QBITEL - Production Secrets Generator

This script generates all required secrets for production deployment.
Run this script once to generate secrets, then securely store them.

Usage:
    python generate_secrets.py [--output-file .env.production]
"""

import secrets
import argparse
import sys
from pathlib import Path
from datetime import datetime

try:
    from cryptography.fernet import Fernet
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False
    print("Warning: cryptography package not installed. Install with: pip install cryptography")


def generate_password(length: int = 32) -> str:
    """Generate a URL-safe password."""
    return secrets.token_urlsafe(length)


def generate_api_key(prefix: str = "qbitel") -> str:
    """Generate an API key with prefix."""
    return f"{prefix}_{secrets.token_urlsafe(32)}"


def generate_fernet_key() -> str:
    """Generate a Fernet encryption key."""
    if HAS_CRYPTOGRAPHY:
        return Fernet.generate_key().decode()
    else:
        # Fallback: generate a base64-encoded 32-byte key
        import base64
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()


def generate_hex_key(length: int = 32) -> str:
    """Generate a hex-encoded key."""
    return secrets.token_hex(length)


def main():
    parser = argparse.ArgumentParser(description="Generate production secrets for QBITEL")
    parser.add_argument(
        "--output-file",
        type=str,
        default=".env.production",
        help="Output file for generated secrets"
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Only print secrets, don't write to file"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("QBITEL - Production Secrets Generator")
    print("=" * 70)
    print(f"Generated at: {datetime.now().isoformat()}")
    print()

    # Generate all secrets
    secrets_dict = {
        # Database
        "QBITEL_AI_DB_PASSWORD": generate_password(32),

        # Redis
        "QBITEL_AI_REDIS_PASSWORD": generate_password(32),

        # Security
        "QBITEL_AI_JWT_SECRET": generate_password(48),
        "QBITEL_AI_ENCRYPTION_KEY": generate_fernet_key(),
        "QBITEL_AI_API_KEY": generate_api_key("qbitel"),

        # Grafana
        "GF_SECURITY_ADMIN_PASSWORD": generate_password(24),
    }

    # Print secrets
    print("Generated Secrets:")
    print("-" * 70)
    for key, value in secrets_dict.items():
        print(f"{key}={value}")
    print("-" * 70)
    print()

    if args.print_only:
        print("Secrets printed only (not saved to file).")
        return

    # Read template and generate production file
    template_path = Path(__file__).parent / ".env.production.template"
    output_path = Path(args.output_file)

    if template_path.exists():
        with open(template_path, 'r') as f:
            content = f.read()

        # Replace placeholders with generated secrets
        replacements = {
            "REPLACE_WITH_GENERATED_PASSWORD_MIN_16_CHARS": secrets_dict["QBITEL_AI_DB_PASSWORD"],
            "REPLACE_WITH_GENERATED_SECRET_MIN_32_CHARS": secrets_dict["QBITEL_AI_JWT_SECRET"],
            "REPLACE_WITH_FERNET_KEY": secrets_dict["QBITEL_AI_ENCRYPTION_KEY"],
            "REPLACE_WITH_GENERATED_API_KEY": secrets_dict["QBITEL_AI_API_KEY"],
        }

        for placeholder, value in replacements.items():
            content = content.replace(placeholder, value, 1)  # Replace first occurrence
            # Handle Redis password separately (second occurrence)
            if placeholder == "REPLACE_WITH_GENERATED_PASSWORD_MIN_16_CHARS":
                content = content.replace(placeholder, secrets_dict["QBITEL_AI_REDIS_PASSWORD"], 1)

        with open(output_path, 'w') as f:
            f.write(content)

        print(f"Production environment file created: {output_path}")
    else:
        # Create minimal env file
        with open(output_path, 'w') as f:
            f.write(f"# Generated at: {datetime.now().isoformat()}\n")
            f.write("# QBITEL Production Secrets\n\n")
            for key, value in secrets_dict.items():
                f.write(f"{key}={value}\n")

        print(f"Minimal secrets file created: {output_path}")

    print()
    print("IMPORTANT SECURITY REMINDERS:")
    print("1. Store these secrets in a secure secrets manager (Vault, AWS Secrets Manager, etc.)")
    print("2. Never commit .env.production to version control")
    print("3. Rotate secrets regularly (recommended: every 90 days)")
    print("4. Use different secrets for each environment")
    print()
    print("Add to .gitignore:")
    print("  .env.production")
    print("  *.env")
    print("  secrets/")


if __name__ == "__main__":
    main()
