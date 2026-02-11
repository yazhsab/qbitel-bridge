#!/bin/bash
# ============================================================================
# QBITEL - Integration Test Runner
# ============================================================================
#
# This script runs all integration tests for the Legacy System Whisperer.
#
# USAGE:
#   ./run_integration_tests.sh [OPTIONS]
#
# OPTIONS:
#   --all           Run all tests including slow tests
#   --db-only       Run only database tests
#   --llm-only      Run only LLM tests
#   --e2e-only      Run only E2E tests
#   --with-coverage Generate coverage report
#   --verbose       Verbose output
#   --skip-slow     Skip slow tests (default)
#
# PREREQUISITES:
#   - Python 3.9+
#   - pytest, pytest-asyncio, pytest-cov
#   - Docker (for database tests)
#   - API keys configured (for LLM tests)
#
# ============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default options
RUN_ALL=false
DB_ONLY=false
LLM_ONLY=false
E2E_ONLY=false
WITH_COVERAGE=false
VERBOSE=false
SKIP_SLOW=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            RUN_ALL=true
            SKIP_SLOW=false
            shift
            ;;
        --db-only)
            DB_ONLY=true
            shift
            ;;
        --llm-only)
            LLM_ONLY=true
            shift
            ;;
        --e2e-only)
            E2E_ONLY=true
            shift
            ;;
        --with-coverage)
            WITH_COVERAGE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --skip-slow)
            SKIP_SLOW=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}QBITEL - Integration Test Runner${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"

    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: Python 3 not found${NC}"
        exit 1
    fi

    # Check pytest
    if ! python3 -c "import pytest" &> /dev/null; then
        echo -e "${RED}Error: pytest not installed. Run: pip install pytest pytest-asyncio${NC}"
        exit 1
    fi

    # Check pytest-asyncio
    if ! python3 -c "import pytest_asyncio" &> /dev/null; then
        echo -e "${YELLOW}Warning: pytest-asyncio not installed. Installing...${NC}"
        pip install pytest-asyncio
    fi

    echo -e "${GREEN}Prerequisites OK${NC}"
    echo ""
}

# Build pytest command
build_pytest_cmd() {
    local cmd="python3 -m pytest"

    # Add verbose flag
    if [ "$VERBOSE" = true ]; then
        cmd="$cmd -v"
    fi

    # Add coverage
    if [ "$WITH_COVERAGE" = true ]; then
        cmd="$cmd --cov=ai_engine --cov-report=html --cov-report=term-missing"
    fi

    # Add markers
    cmd="$cmd -m integration"

    # Skip slow tests
    if [ "$SKIP_SLOW" = true ]; then
        cmd="$cmd --ignore-glob='*slow*' -m 'integration and not slow'"
    fi

    # Add test files
    if [ "$DB_ONLY" = true ]; then
        cmd="$cmd $SCRIPT_DIR/test_database_integration.py"
    elif [ "$LLM_ONLY" = true ]; then
        cmd="$cmd $SCRIPT_DIR/test_llm_integration.py"
    elif [ "$E2E_ONLY" = true ]; then
        cmd="$cmd $SCRIPT_DIR/test_legacy_whisperer_e2e.py"
    else
        cmd="$cmd $SCRIPT_DIR/"
    fi

    echo "$cmd"
}

# Run tests
run_tests() {
    local cmd=$(build_pytest_cmd)

    echo -e "${YELLOW}Running tests with command:${NC}"
    echo "$cmd"
    echo ""

    # Set environment variables
    export QBITEL_AI_ENVIRONMENT=testing
    export QBITEL_AI_DEBUG=true
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

    # Run tests
    cd "$PROJECT_ROOT"
    eval "$cmd"
}

# Main
check_prerequisites
run_tests

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Tests completed!${NC}"
echo -e "${GREEN}============================================${NC}"

if [ "$WITH_COVERAGE" = true ]; then
    echo ""
    echo -e "${YELLOW}Coverage report available at: $PROJECT_ROOT/htmlcov/index.html${NC}"
fi
