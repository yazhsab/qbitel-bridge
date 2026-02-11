#!/bin/bash
#
# QBITEL - Smoke Test Runner
#
# Quick validation tests for CI/CD pipelines.
# Usage: ./run_smoke_tests.sh [options]
#
# Options:
#   --verbose    Enable verbose output
#   --fail-fast  Stop on first failure
#   --coverage   Enable coverage reporting
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default options
VERBOSE=""
FAIL_FAST=""
COVERAGE=""
TIMEOUT=120  # 2 minutes max

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v)
            VERBOSE="-v"
            shift
            ;;
        --fail-fast|-x)
            FAIL_FAST="-x"
            shift
            ;;
        --coverage)
            COVERAGE="--cov=ai_engine --cov-report=term-missing"
            shift
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}QBITEL - Smoke Tests${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Check Python environment
echo -e "${YELLOW}Checking Python environment...${NC}"
python --version || { echo -e "${RED}Python not found${NC}"; exit 1; }

# Check pytest
python -c "import pytest" 2>/dev/null || {
    echo -e "${RED}pytest not found. Installing...${NC}"
    pip install pytest pytest-asyncio pytest-cov
}

# Set environment
export QBITEL_ENV=test
export QBITEL_LOG_LEVEL=WARNING
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Run smoke tests with timeout
echo ""
echo -e "${YELLOW}Running smoke tests...${NC}"
echo ""

START_TIME=$(date +%s)

# Run pytest
timeout "$TIMEOUT" python -m pytest \
    tests/smoke/ \
    $VERBOSE \
    $FAIL_FAST \
    $COVERAGE \
    --tb=short \
    --durations=10 \
    -q \
    || TEST_EXIT_CODE=$?

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo -e "${GREEN}========================================${NC}"

if [ "${TEST_EXIT_CODE:-0}" -eq 0 ]; then
    echo -e "${GREEN}✓ Smoke tests PASSED${NC}"
    echo -e "Duration: ${DURATION}s"
    exit 0
elif [ "${TEST_EXIT_CODE:-0}" -eq 124 ]; then
    echo -e "${RED}✗ Smoke tests TIMED OUT${NC}"
    echo -e "Tests exceeded ${TIMEOUT}s timeout"
    exit 1
else
    echo -e "${RED}✗ Smoke tests FAILED${NC}"
    echo -e "Duration: ${DURATION}s"
    exit "${TEST_EXIT_CODE}"
fi
