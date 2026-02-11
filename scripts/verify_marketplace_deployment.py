"""
QBITEL - Marketplace Deployment Verification Script

Verifies that all marketplace components are properly installed and configured.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def verify_imports():
    """Verify all marketplace modules can be imported."""
    print("🔍 Verifying module imports...")

    try:
        from ai_engine.models.marketplace import (
            MarketplaceProtocol,
            MarketplaceUser,
            MarketplaceInstallation,
            MarketplaceReview,
            MarketplaceTransaction,
            MarketplaceValidation,
        )
        print("  ✓ Database models imported successfully")
    except Exception as e:
        print(f"  ✗ Failed to import database models: {e}")
        return False

    try:
        from ai_engine.api.marketplace_endpoints import router
        print("  ✓ API endpoints imported successfully")
    except Exception as e:
        print(f"  ✗ Failed to import API endpoints: {e}")
        return False

    try:
        from ai_engine.api.marketplace_schemas import (
            ProtocolSearchRequest,
            ProtocolSubmitRequest,
        )
        print("  ✓ API schemas imported successfully")
    except Exception as e:
        print(f"  ✗ Failed to import API schemas: {e}")
        return False

    try:
        from ai_engine.marketplace import (
            ProtocolValidator,
            MarketplaceKnowledgeBaseIntegration,
        )
        print("  ✓ Marketplace core modules imported successfully")
    except Exception as e:
        print(f"  ✗ Failed to import marketplace modules: {e}")
        return False

    return True


def verify_configuration():
    """Verify marketplace configuration."""
    print("\n🔍 Verifying configuration...")

    try:
        from ai_engine.core.config import get_config

        config = get_config()
        print(f"  ✓ Configuration loaded successfully")
        print(f"    - Environment: {config.environment}")
        print(f"    - Marketplace enabled: {config.marketplace.enabled}")
        print(f"    - S3 Bucket: {config.marketplace.s3_bucket}")
        print(f"    - Platform fee: {config.marketplace.platform_fee * 100}%")

        return True
    except Exception as e:
        print(f"  ✗ Failed to load configuration: {e}")
        return False


def verify_database_models():
    """Verify database models are properly defined."""
    print("\n🔍 Verifying database models...")

    try:
        from ai_engine.models.marketplace import (
            MarketplaceProtocol,
            MarketplaceUser,
        )
        from sqlalchemy import inspect

        # Check MarketplaceProtocol
        protocol_cols = [c.name for c in inspect(MarketplaceProtocol).columns]
        required_cols = ['protocol_id', 'protocol_name', 'display_name', 'category']

        for col in required_cols:
            if col in protocol_cols:
                print(f"  ✓ MarketplaceProtocol has column: {col}")
            else:
                print(f"  ✗ MarketplaceProtocol missing column: {col}")
                return False

        # Check MarketplaceUser
        user_cols = [c.name for c in inspect(MarketplaceUser).columns]
        required_cols = ['user_id', 'email', 'username', 'user_type']

        for col in required_cols:
            if col in user_cols:
                print(f"  ✓ MarketplaceUser has column: {col}")
            else:
                print(f"  ✗ MarketplaceUser missing column: {col}")
                return False

        print(f"\n  ℹ️  Total MarketplaceProtocol columns: {len(protocol_cols)}")
        print(f"  ℹ️  Total MarketplaceUser columns: {len(user_cols)}")

        return True
    except Exception as e:
        print(f"  ✗ Failed to verify database models: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_api_endpoints():
    """Verify API endpoints are registered."""
    print("\n🔍 Verifying API endpoints...")

    try:
        from ai_engine.api.marketplace_endpoints import router

        # Count routes
        route_count = len(router.routes)
        print(f"  ✓ Marketplace router has {route_count} routes")

        # List endpoints
        print(f"\n  📋 Available endpoints:")
        for route in router.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                methods = ', '.join(route.methods) if route.methods else 'N/A'
                print(f"    - {methods:6} {route.path}")

        return True
    except Exception as e:
        print(f"  ✗ Failed to verify API endpoints: {e}")
        return False


def verify_files():
    """Verify all required files exist."""
    print("\n🔍 Verifying file structure...")

    base_dir = Path(__file__).parent.parent / "ai_engine"

    required_files = [
        "models/marketplace.py",
        "api/marketplace_endpoints.py",
        "api/marketplace_schemas.py",
        "marketplace/__init__.py",
        "marketplace/protocol_validator.py",
        "marketplace/knowledge_base_integration.py",
        "alembic/versions/004_add_marketplace_tables.py",
        "tests/test_marketplace_basic.py",
    ]

    all_exist = True
    for file_path in required_files:
        full_path = base_dir / file_path
        if full_path.exists():
            size_kb = full_path.stat().st_size / 1024
            print(f"  ✓ {file_path} ({size_kb:.1f} KB)")
        else:
            print(f"  ✗ {file_path} (MISSING)")
            all_exist = False

    return all_exist


def verify_documentation():
    """Verify documentation files exist."""
    print("\n🔍 Verifying documentation...")

    docs_dir = Path(__file__).parent.parent / "docs"

    required_docs = [
        "PROTOCOL_MARKETPLACE_ARCHITECTURE.md",
        "PROTOCOL_MARKETPLACE_IMPLEMENTATION_COMPLETE.md",
        "MARKETPLACE_QUICK_START.md",
        "MARKETPLACE_IMPLEMENTATION_SUMMARY.md",
    ]

    all_exist = True
    for doc in required_docs:
        doc_path = docs_dir / doc
        if doc_path.exists():
            size_kb = doc_path.stat().st_size / 1024
            print(f"  ✓ {doc} ({size_kb:.1f} KB)")
        else:
            print(f"  ✗ {doc} (MISSING)")
            all_exist = False

    return all_exist


def main():
    """Run all verification checks."""
    print("="*60)
    print("  QBITEL - Marketplace Deployment Verification")
    print("="*60)

    results = {}

    results['imports'] = verify_imports()
    results['configuration'] = verify_configuration()
    results['database_models'] = verify_database_models()
    results['api_endpoints'] = verify_api_endpoints()
    results['files'] = verify_files()
    results['documentation'] = verify_documentation()

    # Summary
    print("\n" + "="*60)
    print("  VERIFICATION SUMMARY")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for check, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {check.replace('_', ' ').title():25} {status}")

    print(f"\n  Overall: {passed}/{total} checks passed ({passed/total*100:.0f}%)")

    if passed == total:
        print("\n  🎉 All verification checks passed!")
        print("  ✅ Marketplace is ready for deployment")
        print("\n  Next steps:")
        print("    1. Set up database connection in .env")
        print("    2. Run: alembic upgrade head")
        print("    3. Run: python3 scripts/seed_marketplace.py")
        print("    4. Start: python3 -m ai_engine")
        return 0
    else:
        print("\n  ⚠️  Some verification checks failed")
        print("  Please review the errors above and fix before deployment")
        return 1


if __name__ == "__main__":
    sys.exit(main())
