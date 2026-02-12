#!/bin/bash
# Script to build the complete QBITEL technical reference document

OUTPUT="/Users/prabakarankannan/qbitel/QBITEL_COMPLETE_TECHNICAL_REFERENCE.md"
cd /Users/prabakarankannan/qbitel

{
echo "# QBITEL Bridge - Complete Technical Reference Document"
echo "## Compiled for NotebookLM"
echo ""
echo "This document contains the COMPLETE contents of 15 technical documentation files from the QBITEL Bridge repository. It is intended as a comprehensive reference for NotebookLM ingestion."
echo ""
echo "---"
echo ""

echo "# ============================================================"
echo "# FILE 1: README.md (Root README - Master Overview)"
echo "# ============================================================"
echo ""
cat README.md
echo ""
echo ""
echo "---"
echo ""

echo "# ============================================================"
echo "# FILE 2: architecture.md (Full Architecture Document)"
echo "# ============================================================"
echo ""
cat architecture.md
echo ""
echo ""
echo "---"
echo ""

echo "# ============================================================"
echo "# FILE 3: docs/API.md (API Documentation)"
echo "# ============================================================"
echo ""
cat docs/API.md
echo ""
echo ""
echo "---"
echo ""

echo "# ============================================================"
echo "# FILE 4: docs/KNOWLEDGE_BASE.md (Comprehensive Knowledge Base)"
echo "# ============================================================"
echo ""
cat docs/KNOWLEDGE_BASE.md
echo ""
echo ""
echo "---"
echo ""

echo "# ============================================================"
echo "# FILE 5: DEVELOPMENT.md (Development Guide)"
echo "# ============================================================"
echo ""
cat DEVELOPMENT.md
echo ""
echo ""
echo "---"
echo ""

echo "# ============================================================"
echo "# FILE 6: DEPLOYMENT.md (Deployment Guide)"
echo "# ============================================================"
echo ""
cat DEPLOYMENT.md
echo ""
echo ""
echo "---"
echo ""

echo "# ============================================================"
echo "# FILE 7: QUICKSTART.md (Quick Start Guide)"
echo "# ============================================================"
echo ""
cat QUICKSTART.md
echo ""
echo ""
echo "---"
echo ""

echo "# ============================================================"
echo "# FILE 8: docs/IMPLEMENTATION_PLAN.md (Implementation Plan)"
echo "# ============================================================"
echo ""
cat docs/IMPLEMENTATION_PLAN.md
echo ""
echo ""
echo "---"
echo ""

echo "# ============================================================"
echo "# FILE 9: docs/ENVIRONMENT_VARIABLE_CONFIGURATION.md"
echo "# ============================================================"
echo ""
cat docs/ENVIRONMENT_VARIABLE_CONFIGURATION.md
echo ""
echo ""
echo "---"
echo ""

echo "# ============================================================"
echo "# FILE 10: docs/LOCAL_DEPLOYMENT_GUIDE.md"
echo "# ============================================================"
echo ""
cat docs/LOCAL_DEPLOYMENT_GUIDE.md
echo ""
echo ""
echo "---"
echo ""

echo "# ============================================================"
echo "# FILE 11: docs/DATABASE_MIGRATIONS.md"
echo "# ============================================================"
echo ""
cat docs/DATABASE_MIGRATIONS.md
echo ""
echo ""
echo "---"
echo ""

echo "# ============================================================"
echo "# FILE 12: docs/PRODUCTION_DEPLOYMENT_CHECKLIST.md"
echo "# ============================================================"
echo ""
cat docs/PRODUCTION_DEPLOYMENT_CHECKLIST.md
echo ""
echo ""
echo "---"
echo ""

echo "# ============================================================"
echo "# FILE 13: SECURITY.md (Security Policy)"
echo "# ============================================================"
echo ""
cat SECURITY.md
echo ""
echo ""
echo "---"
echo ""

echo "# ============================================================"
echo "# FILE 14: AGENTS.md (Agent Instructions)"
echo "# ============================================================"
echo ""
cat AGENTS.md
echo ""
echo ""
echo "---"
echo ""

echo "# ============================================================"
echo "# FILE 15: PQC_IMPLEMENTATION_PLAN.md"
echo "# ============================================================"
echo ""
cat PQC_IMPLEMENTATION_PLAN.md
echo ""
echo ""
echo "---"
echo ""
echo "# END OF COMPLETE TECHNICAL REFERENCE DOCUMENT"
} > "$OUTPUT"

echo "Done. Output: $OUTPUT"
echo "Total lines: $(wc -l < "$OUTPUT")"
