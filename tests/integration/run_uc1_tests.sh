#!/bin/bash
# QBITEL - UC1 Legacy Mainframe Modernization Integration Test Runner
#
# This script runs the complete UC1 E2E integration test suite.
#
# Usage:
#   ./run_uc1_tests.sh              # Run all UC1 tests
#   ./run_uc1_tests.sh --quick      # Run quick tests only (skip slow)
#   ./run_uc1_tests.sh --verbose    # Run with verbose output
#   ./run_uc1_tests.sh --coverage   # Run with coverage report

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default options
VERBOSE=""
COVERAGE=""
SKIP_SLOW=""
MARKERS="integration"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            SKIP_SLOW="--ignore-glob='*slow*' -m 'not slow'"
            echo -e "${YELLOW}Running quick tests only (skipping slow tests)${NC}"
            shift
            ;;
        --verbose|-v)
            VERBOSE="-v"
            shift
            ;;
        --coverage)
            COVERAGE="--cov=ai_engine --cov-report=html --cov-report=term-missing"
            shift
            ;;
        --uc1-only)
            MARKERS="integration and uc1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--quick] [--verbose] [--coverage] [--uc1-only]"
            exit 1
            ;;
    esac
done

# Header
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     QBITEL - UC1 Legacy Mainframe Modernization Tests       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Check Python environment
echo -e "${YELLOW}Checking Python environment...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 not found${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1)
echo -e "  Python: ${GREEN}$PYTHON_VERSION${NC}"

# Check pytest
if ! python3 -c "import pytest" &> /dev/null; then
    echo -e "${RED}Error: pytest not installed${NC}"
    echo "Install with: pip install pytest pytest-asyncio"
    exit 1
fi

# Check for optional dependencies
echo -e "${YELLOW}Checking optional dependencies...${NC}"

check_module() {
    if python3 -c "import $1" &> /dev/null; then
        echo -e "  $1: ${GREEN}available${NC}"
        return 0
    else
        echo -e "  $1: ${YELLOW}not installed (some tests may be skipped)${NC}"
        return 1
    fi
}

check_module "ai_engine.discovery.protocol_signatures" || true
check_module "ai_engine.translation.protocol_bridge" || true
check_module "ai_engine.llm.legacy_whisperer" || true
check_module "datasets.legacy_mainframe.protocol_samples.sample_protocol_data" || true

echo ""

# Run the tests
echo -e "${BLUE}Running UC1 Integration Tests...${NC}"
echo ""

# Build the pytest command
PYTEST_CMD="python3 -m pytest"
PYTEST_CMD="$PYTEST_CMD $SCRIPT_DIR/test_uc1_e2e_flow.py"
PYTEST_CMD="$PYTEST_CMD $VERBOSE"
PYTEST_CMD="$PYTEST_CMD $COVERAGE"
PYTEST_CMD="$PYTEST_CMD --tb=short"
PYTEST_CMD="$PYTEST_CMD --asyncio-mode=auto"

if [ -n "$SKIP_SLOW" ]; then
    PYTEST_CMD="$PYTEST_CMD -m 'not slow'"
fi

echo -e "${YELLOW}Command: $PYTEST_CMD${NC}"
echo ""

# Execute tests
if $PYTEST_CMD; then
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    ALL TESTS PASSED                            ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
    exit 0
else
    EXIT_CODE=$?
    echo ""
    echo -e "${RED}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                    SOME TESTS FAILED                           ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════════╝${NC}"
    exit $EXIT_CODE
fi
