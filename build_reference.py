#!/usr/bin/env python3
"""Build the complete QBITEL technical reference document."""
import os

OUTPUT = "/Users/prabakarankannan/qbitel/QBITEL_COMPLETE_TECHNICAL_REFERENCE.md"
BASE = "/Users/prabakarankannan/qbitel"

files = [
    ("README.md", "FILE 1: README.md (Root README - Master Overview)"),
    ("architecture.md", "FILE 2: architecture.md (Full Architecture Document)"),
    ("docs/API.md", "FILE 3: docs/API.md (API Documentation)"),
    ("docs/KNOWLEDGE_BASE.md", "FILE 4: docs/KNOWLEDGE_BASE.md (Comprehensive Knowledge Base)"),
    ("DEVELOPMENT.md", "FILE 5: DEVELOPMENT.md (Development Guide)"),
    ("DEPLOYMENT.md", "FILE 6: DEPLOYMENT.md (Deployment Guide)"),
    ("QUICKSTART.md", "FILE 7: QUICKSTART.md (Quick Start Guide)"),
    ("docs/IMPLEMENTATION_PLAN.md", "FILE 8: docs/IMPLEMENTATION_PLAN.md (Implementation Plan)"),
    ("docs/ENVIRONMENT_VARIABLE_CONFIGURATION.md", "FILE 9: docs/ENVIRONMENT_VARIABLE_CONFIGURATION.md"),
    ("docs/LOCAL_DEPLOYMENT_GUIDE.md", "FILE 10: docs/LOCAL_DEPLOYMENT_GUIDE.md"),
    ("docs/DATABASE_MIGRATIONS.md", "FILE 11: docs/DATABASE_MIGRATIONS.md"),
    ("docs/PRODUCTION_DEPLOYMENT_CHECKLIST.md", "FILE 12: docs/PRODUCTION_DEPLOYMENT_CHECKLIST.md"),
    ("SECURITY.md", "FILE 13: SECURITY.md (Security Policy)"),
    ("AGENTS.md", "FILE 14: AGENTS.md (Agent Instructions)"),
    ("PQC_IMPLEMENTATION_PLAN.md", "FILE 15: PQC_IMPLEMENTATION_PLAN.md"),
]

with open(OUTPUT, "w") as out:
    out.write("# QBITEL Bridge - Complete Technical Reference Document\n")
    out.write("## Compiled for NotebookLM\n\n")
    out.write("This document contains the COMPLETE contents of 15 technical documentation files from the QBITEL Bridge repository. It is intended as a comprehensive reference for NotebookLM ingestion.\n\n")
    out.write("---\n\n")

    for filepath, label in files:
        full_path = os.path.join(BASE, filepath)
        out.write("# ============================================================\n")
        out.write(f"# {label}\n")
        out.write("# ============================================================\n\n")

        try:
            with open(full_path, "r") as f:
                content = f.read()
            out.write(content)
            out.write("\n\n---\n\n")
            print(f"OK: {filepath} ({len(content)} chars)")
        except Exception as e:
            out.write(f"ERROR: Could not read {filepath}: {e}\n\n---\n\n")
            print(f"ERROR: {filepath}: {e}")

    out.write("# END OF COMPLETE TECHNICAL REFERENCE DOCUMENT\n")

print(f"\nDone. Output: {OUTPUT}")
