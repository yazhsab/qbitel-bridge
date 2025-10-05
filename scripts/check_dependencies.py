#!/usr/bin/env python3
"""
CRONOS AI - Dependency Verification Script
Checks all runtime dependencies and provides installation guidance.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_engine.core.dependency_manager import check_dependencies


def main():
    """Main entry point for dependency checking."""
    print("Checking CRONOS AI dependencies...\n")

    is_valid = check_dependencies()

    if is_valid:
        print("\n✓ All required dependencies are installed and ready!")
        sys.exit(0)
    else:
        print("\n✗ Some required dependencies are missing or have issues.")
        print("Please install missing dependencies before running CRONOS AI.")
        sys.exit(1)


if __name__ == "__main__":
    main()
