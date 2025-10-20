"""
CRONOS AI - Marketplace Data Seeding Script

Seeds the marketplace with sample protocols, users, and data for testing and development.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import uuid
from decimal import Decimal
from datetime import datetime

from sqlalchemy.orm import Session

# Import after path is set
from ai_engine.core.database_manager import get_database_manager, initialize_database_manager
from ai_engine.core.config import get_config
from ai_engine.models.marketplace import (
    MarketplaceUser,
    MarketplaceProtocol,
    ProtocolCategory,
    ProtocolType,
    SpecFormat,
    AuthorType,
    LicenseType,
    PriceModel,
    CertificationStatus,
    ProtocolStatus,
    UserType,
)


def seed_marketplace():
    """Seed marketplace with sample data."""
    print("🌱 Seeding CRONOS AI Marketplace...")

    # Initialize database manager
    config = get_config()
    initialize_database_manager(config)
    db_manager = get_database_manager()
    session = db_manager.get_session()

    try:
        # Check if data already exists
        existing_users = session.query(MarketplaceUser).count()
        if existing_users > 0:
            print(f"⚠️  Marketplace already has {existing_users} users. Skipping seed.")
            response = input("Do you want to continue and add more data? (y/N): ")
            if response.lower() != 'y':
                print("❌ Seeding cancelled.")
                return

        print("\n📋 Creating sample users...")

        # Create sample users
        users = [
            MarketplaceUser(
                user_id=uuid.uuid4(),
                email="fintech@example.com",
                username="fintech_labs",
                full_name="FinTech Labs Inc.",
                user_type=UserType.VENDOR,
                organization="FinTech Labs",
                is_verified=True,
                reputation_score=500,
                total_contributions=15,
                total_downloads=1250,
                bio="Leading provider of financial protocol solutions",
                website_url="https://fintechlabs.example.com",
            ),
            MarketplaceUser(
                user_id=uuid.uuid4(),
                email="healthcare@example.com",
                username="health_systems",
                full_name="Healthcare Systems Corp",
                user_type=UserType.VENDOR,
                organization="Healthcare Systems Corp",
                is_verified=True,
                reputation_score=350,
                total_contributions=8,
                total_downloads=680,
                bio="Healthcare interoperability specialists",
                website_url="https://healthsystems.example.com",
            ),
            MarketplaceUser(
                user_id=uuid.uuid4(),
                email="community@example.com",
                username="protocol_creator",
                full_name="Community Contributor",
                user_type=UserType.INDIVIDUAL,
                is_verified=True,
                reputation_score=120,
                total_contributions=3,
                total_downloads=450,
                bio="Open source protocol enthusiast",
            ),
            MarketplaceUser(
                user_id=uuid.uuid4(),
                email="cronos@example.com",
                username="cronos_official",
                full_name="CRONOS AI",
                user_type=UserType.VENDOR,
                organization="CRONOS AI",
                is_verified=True,
                reputation_score=1000,
                total_contributions=25,
                total_downloads=5000,
                bio="Official CRONOS AI protocol repository",
                website_url="https://cronos-ai.com",
            ),
        ]

        for user in users:
            session.add(user)
            print(f"  ✓ Created user: {user.username}")

        session.flush()

        print("\n📦 Creating sample protocols...")

        # Create sample protocols
        protocols = [
            MarketplaceProtocol(
                protocol_id=uuid.uuid4(),
                protocol_name="iso8583-v1987",
                display_name="ISO 8583:1987 Financial Transaction Protocol",
                short_description="Legacy ATM and POS transaction protocol",
                long_description="""# ISO 8583:1987

Comprehensive ISO 8583:1987 implementation for financial transaction processing.

## Features
- Complete message parsing and generation
- Support for all standard message types
- Bitmap field handling
- BCD and ASCII encoding
- Extensive validation

## Use Cases
- ATM transactions
- POS terminal communication
- Credit/debit card processing
- Financial institution messaging""",
                category=ProtocolCategory.FINANCE,
                subcategory="payment_processing",
                tags=["banking", "atm", "pos", "legacy", "iso8583"],
                version="2.1.0",
                protocol_type=ProtocolType.BINARY,
                industry="finance",
                spec_format=SpecFormat.YAML,
                spec_file_url="s3://cronos-marketplace/protocols/iso8583/v2.1.0/spec.yaml",
                parser_code_url="s3://cronos-marketplace/protocols/iso8583/v2.1.0/parser.py",
                test_data_url="s3://cronos-marketplace/protocols/iso8583/v2.1.0/test_samples.bin",
                documentation_url="https://marketplace.cronos-ai.com/protocols/iso8583/docs",
                author_id=users[0].user_id,
                author_type=AuthorType.VENDOR,
                organization="FinTech Labs",
                license_type=LicenseType.PAID,
                price_model=PriceModel.ONE_TIME,
                base_price=Decimal("499.00"),
                certification_status=CertificationStatus.CERTIFIED,
                certification_date=datetime.utcnow(),
                status=ProtocolStatus.PUBLISHED,
                min_cronos_version="1.2.0",
                supported_cronos_versions=["1.2.x", "1.3.x", "1.4.x"],
                is_featured=True,
                is_official=False,
                average_rating=Decimal("4.7"),
                total_ratings=23,
                download_count=145,
                active_installations=42,
                published_at=datetime.utcnow(),
            ),
            MarketplaceProtocol(
                protocol_id=uuid.uuid4(),
                protocol_name="hl7v3-cda",
                display_name="HL7 v3 Clinical Document Architecture",
                short_description="Healthcare interoperability protocol for clinical documents",
                long_description="""# HL7 v3 CDA

Full implementation of HL7 v3 Clinical Document Architecture for healthcare data exchange.

## Features
- Complete CDA document parsing
- FHIR resource mapping
- Privacy and security compliance
- Validation against CDA schemas

## Compliance
- HIPAA compliant
- HITECH Act compatible
- ONC certification ready""",
                category=ProtocolCategory.HEALTHCARE,
                subcategory="clinical_messaging",
                tags=["hl7", "healthcare", "ehr", "fhir", "cda"],
                version="1.0.0",
                protocol_type=ProtocolType.XML,
                industry="healthcare",
                spec_format=SpecFormat.YAML,
                spec_file_url="s3://cronos-marketplace/protocols/hl7v3/v1.0.0/spec.yaml",
                parser_code_url="s3://cronos-marketplace/protocols/hl7v3/v1.0.0/parser.py",
                test_data_url="s3://cronos-marketplace/protocols/hl7v3/v1.0.0/test_samples.xml",
                documentation_url="https://marketplace.cronos-ai.com/protocols/hl7v3/docs",
                author_id=users[1].user_id,
                author_type=AuthorType.VENDOR,
                organization="Healthcare Systems Corp",
                license_type=LicenseType.PAID,
                price_model=PriceModel.SUBSCRIPTION,
                base_price=Decimal("99.00"),
                certification_status=CertificationStatus.CERTIFIED,
                certification_date=datetime.utcnow(),
                status=ProtocolStatus.PUBLISHED,
                min_cronos_version="1.3.0",
                supported_cronos_versions=["1.3.x", "1.4.x"],
                is_featured=True,
                is_official=False,
                average_rating=Decimal("4.5"),
                total_ratings=12,
                download_count=78,
                active_installations=28,
                published_at=datetime.utcnow(),
            ),
            MarketplaceProtocol(
                protocol_id=uuid.uuid4(),
                protocol_name="mqtt-v5",
                display_name="MQTT v5.0 Protocol",
                short_description="IoT messaging protocol for lightweight pub/sub communication",
                long_description="""# MQTT v5.0

MQTT v5 protocol implementation with enhanced features for IoT applications.

## Features
- Full MQTT v5.0 specification
- Quality of Service (QoS) levels 0, 1, 2
- Retained messages
- Last Will and Testament
- User properties
- Request/Response pattern

## Performance
- Lightweight and efficient
- Low bandwidth usage
- Suitable for constrained devices""",
                category=ProtocolCategory.IOT,
                subcategory="messaging",
                tags=["iot", "mqtt", "pubsub", "messaging"],
                version="5.0.0",
                protocol_type=ProtocolType.BINARY,
                industry="iot",
                spec_format=SpecFormat.YAML,
                spec_file_url="s3://cronos-marketplace/protocols/mqtt/v5.0.0/spec.yaml",
                parser_code_url="s3://cronos-marketplace/protocols/mqtt/v5.0.0/parser.py",
                test_data_url="s3://cronos-marketplace/protocols/mqtt/v5.0.0/test_samples.bin",
                documentation_url="https://marketplace.cronos-ai.com/protocols/mqtt/docs",
                author_id=users[2].user_id,
                author_type=AuthorType.COMMUNITY,
                license_type=LicenseType.FREE,
                certification_status=CertificationStatus.CERTIFIED,
                certification_date=datetime.utcnow(),
                status=ProtocolStatus.PUBLISHED,
                min_cronos_version="1.0.0",
                supported_cronos_versions=["1.x.x"],
                is_featured=False,
                is_official=False,
                average_rating=Decimal("4.8"),
                total_ratings=35,
                download_count=256,
                active_installations=89,
                published_at=datetime.utcnow(),
            ),
            MarketplaceProtocol(
                protocol_id=uuid.uuid4(),
                protocol_name="modbus-tcp",
                display_name="Modbus TCP/IP Protocol",
                short_description="Industrial automation protocol for PLCs and SCADA systems",
                long_description="""# Modbus TCP/IP

Official CRONOS AI implementation of Modbus TCP/IP for industrial automation.

## Features
- Complete Modbus TCP specification
- Support for all function codes
- Master and slave implementations
- Error handling and validation

## Industrial Use Cases
- PLC communication
- SCADA systems
- Industrial IoT
- Building automation""",
                category=ProtocolCategory.MANUFACTURING,
                subcategory="industrial_automation",
                tags=["modbus", "scada", "plc", "industrial", "automation"],
                version="1.5.0",
                protocol_type=ProtocolType.BINARY,
                industry="manufacturing",
                spec_format=SpecFormat.YAML,
                spec_file_url="s3://cronos-marketplace/protocols/modbus/v1.5.0/spec.yaml",
                parser_code_url="s3://cronos-marketplace/protocols/modbus/v1.5.0/parser.py",
                test_data_url="s3://cronos-marketplace/protocols/modbus/v1.5.0/test_samples.bin",
                documentation_url="https://marketplace.cronos-ai.com/protocols/modbus/docs",
                author_id=users[3].user_id,
                author_type=AuthorType.CRONOS,
                organization="CRONOS AI",
                license_type=LicenseType.FREE,
                certification_status=CertificationStatus.CERTIFIED,
                certification_date=datetime.utcnow(),
                status=ProtocolStatus.PUBLISHED,
                min_cronos_version="1.0.0",
                supported_cronos_versions=["1.x.x"],
                is_featured=True,
                is_official=True,
                average_rating=Decimal("4.9"),
                total_ratings=67,
                download_count=892,
                active_installations=234,
                published_at=datetime.utcnow(),
            ),
        ]

        for protocol in protocols:
            session.add(protocol)
            print(f"  ✓ Created protocol: {protocol.display_name}")

        session.commit()

        print("\n✅ Successfully seeded marketplace with sample data!")
        print(f"\n📊 Summary:")
        print(f"  - Users created: {len(users)}")
        print(f"  - Protocols created: {len(protocols)}")
        print(f"\n🔍 Sample data breakdown:")
        print(f"  - Free protocols: {sum(1 for p in protocols if p.license_type == LicenseType.FREE)}")
        print(f"  - Paid protocols: {sum(1 for p in protocols if p.license_type == LicenseType.PAID)}")
        print(f"  - Featured protocols: {sum(1 for p in protocols if p.is_featured)}")
        print(f"  - Official protocols: {sum(1 for p in protocols if p.is_official)}")

        print(f"\n🚀 Next steps:")
        print(f"  1. Start the API server: python3 -m ai_engine")
        print(f"  2. Open API docs: http://localhost:8000/docs")
        print(f"  3. Try search: http://localhost:8000/api/v1/marketplace/protocols/search")

    except Exception as e:
        session.rollback()
        print(f"\n❌ Error seeding data: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    seed_marketplace()
