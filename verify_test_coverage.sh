#!/bin/bash
# Test Coverage Verification Script
# Run this in CI to verify all new tests work

echo "=========================================="
echo "CRONOS AI - Test Coverage Verification"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0.32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Step 1: Verifying new compliance test files exist...${NC}"
test_files=(
    "ai_engine/tests/test_data_retention.py"
    "ai_engine/tests/test_gdpr_compliance.py"
    "ai_engine/tests/test_soc2_controls.py"
    "ai_engine/tests/conftest_mocks.py"
)

for file in "${test_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file exists"
    else
        echo -e "${RED}✗${NC} $file MISSING"
        exit 1
    fi
done

echo ""
echo -e "${YELLOW}Step 2: Running new compliance tests...${NC}"

# Test data retention
echo "Testing data_retention.py..."
python3 -m pytest ai_engine/tests/test_data_retention.py -v --tb=short || {
    echo -e "${RED}Data retention tests failed${NC}"
    exit 1
}

# Test GDPR compliance
echo "Testing gdpr_compliance.py..."
python3 -m pytest ai_engine/tests/test_gdpr_compliance.py -v --tb=short || {
    echo -e "${RED}GDPR compliance tests failed${NC}"
    exit 1
}

# Test SOC2 controls
echo "Testing soc2_controls.py..."
python3 -m pytest ai_engine/tests/test_soc2_controls.py -v --tb=short || {
    echo -e "${RED}SOC2 controls tests failed${NC}"
    exit 1
}

echo ""
echo -e "${YELLOW}Step 3: Checking coverage for compliance modules...${NC}"

# Coverage for data retention
echo "Coverage for data_retention.py:"
python3 -m pytest ai_engine/tests/test_data_retention.py \
    --cov=ai_engine.compliance.data_retention \
    --cov-report=term-missing \
    --cov-report=json:coverage_retention.json \
    -q

# Coverage for GDPR
echo "Coverage for gdpr_compliance.py:"
python3 -m pytest ai_engine/tests/test_gdpr_compliance.py \
    --cov=ai_engine.compliance.gdpr_compliance \
    --cov-report=term-missing \
    --cov-report=json:coverage_gdpr.json \
    -q

# Coverage for SOC2
echo "Coverage for soc2_controls.py:"
python3 -m pytest ai_engine/tests/test_soc2_controls.py \
    --cov=ai_engine.compliance.soc2_controls \
    --cov-report=term-missing \
    --cov-report=json:coverage_soc2.json \
    -q

echo ""
echo -e "${YELLOW}Step 4: Verifying previously broken tests now run...${NC}"

# Test trainer comprehensive (should collect, may have failures)
echo "Collecting trainer_comprehensive tests:"
python3 -m pytest ai_engine/tests/test_trainer_comprehensive.py --collect-only -q || {
    echo -e "${RED}Trainer tests still have import errors${NC}"
    exit 1
}

# Test auth enterprise comprehensive
echo "Collecting auth_enterprise_comprehensive tests:"
python3 -m pytest ai_engine/tests/test_auth_enterprise_comprehensive.py --collect-only -q || {
    echo -e "${RED}Auth enterprise tests still have import errors${NC}"
    exit 1
}

echo ""
echo -e "${GREEN}=========================================="
echo "✓ ALL VERIFICATION CHECKS PASSED"
echo "==========================================${NC}"
echo ""
echo "Summary:"
echo "- 3 new test files created and passing"
echo "- Compliance modules have test coverage"
echo "- Previously broken tests can now import"
echo ""
echo "Run full test suite with:"
echo "  pytest ai_engine/tests/ -v --cov=ai_engine --cov-report=html"
