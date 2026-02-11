#!/usr/bin/env python3
"""
QBITEL - Dependency Upgrade Migration Script

This script safely upgrades dependencies to address security vulnerabilities
and modernize the tech stack.

Usage:
    python scripts/migrations/upgrade_dependencies.py --check    # Check only
    python scripts/migrations/upgrade_dependencies.py --apply    # Apply upgrades
    python scripts/migrations/upgrade_dependencies.py --rollback # Rollback
"""

import os
import sys
import subprocess
import shutil
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Colors for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

class Severity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

@dataclass
class DependencyUpgrade:
    """Represents a dependency upgrade."""
    name: str
    current_version: str
    target_version: str
    severity: Severity
    cve: Optional[str]
    description: str
    breaking_changes: List[str]
    files_to_update: List[str]

# Define all required upgrades
UPGRADES: List[DependencyUpgrade] = [
    DependencyUpgrade(
        name="torch",
        current_version="2.2.2",
        target_version=">=2.6.0",
        severity=Severity.CRITICAL,
        cve="CVE-2025-32434",
        description="Remote Code Execution via torch.load() weights_only bypass",
        breaking_changes=[
            "torch.load() now requires explicit weights_only parameter",
            "Some deprecated functions removed",
            "CUDA 11.x support dropped (use CUDA 12.x)",
        ],
        files_to_update=[
            "requirements.txt",
            "ai_engine/requirements.txt",
        ],
    ),
    DependencyUpgrade(
        name="fastapi",
        current_version="0.104.1",
        target_version=">=0.121.3",
        severity=Severity.HIGH,
        cve="CVE-2024-24762",
        description="ReDoS vulnerability in python-multipart dependency",
        breaking_changes=[
            "Response model validation stricter",
            "OpenAPI schema generation changes",
            "Some deprecated parameters removed",
        ],
        files_to_update=[
            "requirements.txt",
            "ai_engine/requirements.txt",
        ],
    ),
    DependencyUpgrade(
        name="anthropic",
        current_version="0.8.1",
        target_version=">=0.75.0",
        severity=Severity.MEDIUM,
        cve=None,
        description="SDK severely outdated (60+ versions behind)",
        breaking_changes=[
            "Client initialization changed",
            "Message format updated",
            "Streaming API changed",
            "New features like prompt caching available",
        ],
        files_to_update=[
            "requirements.txt",
            "ai_engine/requirements.txt",
            "requirements-copilot.txt",
        ],
    ),
    DependencyUpgrade(
        name="openai",
        current_version="1.10.0",
        target_version=">=1.50.0",
        severity=Severity.MEDIUM,
        cve=None,
        description="SDK outdated (40+ versions behind)",
        breaking_changes=[
            "Async client changes",
            "Response object structure updated",
            "New features available",
        ],
        files_to_update=[
            "requirements.txt",
            "ai_engine/requirements.txt",
        ],
    ),
    DependencyUpgrade(
        name="uvicorn",
        current_version="0.24.0",
        target_version=">=0.30.0",
        severity=Severity.LOW,
        cve=None,
        description="Update for bug fixes and performance",
        breaking_changes=[
            "Some logging changes",
            "SSL configuration updates",
        ],
        files_to_update=[
            "requirements.txt",
            "ai_engine/requirements.txt",
        ],
    ),
    DependencyUpgrade(
        name="cryptography",
        current_version="41.0.7",
        target_version=">=43.0.0",
        severity=Severity.MEDIUM,
        cve=None,
        description="Update for security fixes and new algorithms",
        breaking_changes=[
            "Some deprecated APIs removed",
            "OpenSSL version requirements changed",
        ],
        files_to_update=[
            "requirements.txt",
            "ai_engine/requirements.txt",
        ],
    ),
]

# New dependencies to add
NEW_DEPENDENCIES = {
    "requirements.txt": [
        "# AI/ML Modernization",
        "litellm>=1.5.0",
        "langgraph>=0.0.5",
        "",
        "# MLOps",
        "evidently>=0.4.0",
        "",
        "# Vector Database",
        "# qdrant-client already present in ai_engine/requirements.txt",
    ],
    "ai_engine/requirements.txt": [
        "",
        "# Agentic AI Framework",
        "langgraph>=0.0.5",
        "crewai>=0.1.5",
        "",
        "# Unified LLM Interface",
        "litellm>=1.5.0",
        "",
        "# MLOps Enhancements",
        "evidently>=0.4.0",
        "feast>=0.35.0",
    ],
}


class DependencyMigrator:
    """Handles dependency migration and rollback."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.backup_dir = project_root / ".dependency_backups"
        self.backup_dir.mkdir(exist_ok=True)

    def check_upgrades(self) -> Dict[str, List[str]]:
        """Check which upgrades are needed."""
        print(f"\n{Colors.BOLD}Checking dependency upgrades...{Colors.RESET}\n")

        results = {
            "critical": [],
            "high": [],
            "medium": [],
            "low": [],
            "up_to_date": [],
        }

        for upgrade in UPGRADES:
            status = self._check_upgrade_status(upgrade)
            if status == "needed":
                severity_key = upgrade.severity.value.lower()
                results[severity_key].append(upgrade)
            else:
                results["up_to_date"].append(upgrade.name)

        return results

    def _check_upgrade_status(self, upgrade: DependencyUpgrade) -> str:
        """Check if an upgrade is needed."""
        try:
            import importlib.metadata
            current = importlib.metadata.version(upgrade.name)

            # Parse version strings
            from packaging import version
            if version.parse(current) >= version.parse(upgrade.target_version.replace(">=", "")):
                return "up_to_date"
            return "needed"
        except Exception:
            return "needed"

    def print_report(self, results: Dict[str, List]):
        """Print upgrade report."""
        print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}DEPENDENCY UPGRADE REPORT{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*70}{Colors.RESET}\n")

        if results["critical"]:
            print(f"{Colors.RED}{Colors.BOLD}CRITICAL UPGRADES REQUIRED:{Colors.RESET}")
            for upgrade in results["critical"]:
                self._print_upgrade(upgrade)

        if results["high"]:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}HIGH PRIORITY UPGRADES:{Colors.RESET}")
            for upgrade in results["high"]:
                self._print_upgrade(upgrade)

        if results["medium"]:
            print(f"\n{Colors.BLUE}{Colors.BOLD}MEDIUM PRIORITY UPGRADES:{Colors.RESET}")
            for upgrade in results["medium"]:
                self._print_upgrade(upgrade)

        if results["low"]:
            print(f"\n{Colors.GREEN}LOW PRIORITY UPGRADES:{Colors.RESET}")
            for upgrade in results["low"]:
                self._print_upgrade(upgrade)

        if results["up_to_date"]:
            print(f"\n{Colors.GREEN}Already up to date: {', '.join(results['up_to_date'])}{Colors.RESET}")

        print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}\n")

    def _print_upgrade(self, upgrade: DependencyUpgrade):
        """Print details of a single upgrade."""
        cve_str = f" ({upgrade.cve})" if upgrade.cve else ""
        print(f"\n  {Colors.BOLD}{upgrade.name}{Colors.RESET}{cve_str}")
        print(f"    Current: {upgrade.current_version} -> Target: {upgrade.target_version}")
        print(f"    {upgrade.description}")
        if upgrade.breaking_changes:
            print(f"    {Colors.YELLOW}Breaking changes:{Colors.RESET}")
            for change in upgrade.breaking_changes[:3]:
                print(f"      - {change}")

    def create_backup(self) -> str:
        """Create backup of requirements files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / timestamp

        backup_path.mkdir(exist_ok=True)

        files_to_backup = [
            "requirements.txt",
            "ai_engine/requirements.txt",
            "requirements-copilot.txt",
            "go/controlplane/go.mod",
            "go/mgmtapi/go.mod",
            "go/agents/device-agent/go.mod",
        ]

        for file in files_to_backup:
            src = self.project_root / file
            if src.exists():
                dst = backup_path / file
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                print(f"  Backed up: {file}")

        # Save backup manifest
        manifest = {
            "timestamp": timestamp,
            "files": files_to_backup,
            "upgrades": [u.name for u in UPGRADES],
        }
        with open(backup_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"\n{Colors.GREEN}Backup created: {backup_path}{Colors.RESET}")
        return str(backup_path)

    def apply_upgrades(self, dry_run: bool = False) -> bool:
        """Apply all upgrades."""
        print(f"\n{Colors.BOLD}Applying dependency upgrades...{Colors.RESET}\n")

        if not dry_run:
            backup_path = self.create_backup()
            print(f"Backup created at: {backup_path}\n")

        success = True

        # Update Python requirements files
        for upgrade in UPGRADES:
            for file_path in upgrade.files_to_update:
                full_path = self.project_root / file_path
                if full_path.exists():
                    if dry_run:
                        print(f"[DRY RUN] Would update {upgrade.name} in {file_path}")
                    else:
                        if not self._update_requirements_file(full_path, upgrade):
                            success = False

        # Add new dependencies
        for file_path, new_deps in NEW_DEPENDENCIES.items():
            full_path = self.project_root / file_path
            if full_path.exists():
                if dry_run:
                    print(f"[DRY RUN] Would add new dependencies to {file_path}")
                else:
                    self._add_new_dependencies(full_path, new_deps)

        # Update Go modules
        if not dry_run:
            self._update_go_modules()

        return success

    def _update_requirements_file(
        self,
        file_path: Path,
        upgrade: DependencyUpgrade
    ) -> bool:
        """Update a single requirements file."""
        try:
            content = file_path.read_text()
            lines = content.split('\n')
            updated_lines = []

            for line in lines:
                # Check if this line contains the package
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    # Parse package name
                    for sep in ['==', '>=', '<=', '~=', '!=', '<', '>']:
                        if sep in stripped:
                            pkg_name = stripped.split(sep)[0].strip()
                            # Handle extras like package[extra]
                            pkg_name = pkg_name.split('[')[0]

                            if pkg_name.lower() == upgrade.name.lower():
                                # Replace the line
                                new_line = f"{pkg_name}{upgrade.target_version}"
                                updated_lines.append(new_line)
                                print(f"  {Colors.GREEN}Updated {upgrade.name} in {file_path.name}{Colors.RESET}")
                                break
                    else:
                        updated_lines.append(line)
                else:
                    updated_lines.append(line)

            file_path.write_text('\n'.join(updated_lines))
            return True

        except Exception as e:
            print(f"  {Colors.RED}Error updating {file_path}: {e}{Colors.RESET}")
            return False

    def _add_new_dependencies(self, file_path: Path, new_deps: List[str]):
        """Add new dependencies to a requirements file."""
        try:
            content = file_path.read_text()

            # Check if dependencies already exist
            deps_to_add = []
            for dep in new_deps:
                if dep.startswith('#') or not dep.strip():
                    deps_to_add.append(dep)
                else:
                    pkg_name = dep.split('>=')[0].split('==')[0].strip()
                    if pkg_name.lower() not in content.lower():
                        deps_to_add.append(dep)

            if deps_to_add:
                content += '\n' + '\n'.join(deps_to_add)
                file_path.write_text(content)
                print(f"  {Colors.GREEN}Added new dependencies to {file_path.name}{Colors.RESET}")

        except Exception as e:
            print(f"  {Colors.RED}Error adding dependencies to {file_path}: {e}{Colors.RESET}")

    def _update_go_modules(self):
        """Update Go module versions."""
        go_dirs = [
            "go/controlplane",
            "go/mgmtapi",
            "go/agents/device-agent",
        ]

        for go_dir in go_dirs:
            go_mod = self.project_root / go_dir / "go.mod"
            if go_mod.exists():
                try:
                    content = go_mod.read_text()
                    # Update Go version
                    content = content.replace("go 1.22", "go 1.23")
                    go_mod.write_text(content)
                    print(f"  {Colors.GREEN}Updated Go version in {go_dir}/go.mod{Colors.RESET}")

                    # Run go mod tidy
                    subprocess.run(
                        ["go", "mod", "tidy"],
                        cwd=self.project_root / go_dir,
                        capture_output=True,
                    )
                except Exception as e:
                    print(f"  {Colors.YELLOW}Warning: Could not update {go_dir}: {e}{Colors.RESET}")

    def rollback(self, backup_timestamp: Optional[str] = None) -> bool:
        """Rollback to a previous backup."""
        if backup_timestamp:
            backup_path = self.backup_dir / backup_timestamp
        else:
            # Find most recent backup
            backups = sorted(self.backup_dir.iterdir(), reverse=True)
            if not backups:
                print(f"{Colors.RED}No backups found{Colors.RESET}")
                return False
            backup_path = backups[0]

        if not backup_path.exists():
            print(f"{Colors.RED}Backup not found: {backup_path}{Colors.RESET}")
            return False

        print(f"\n{Colors.BOLD}Rolling back to: {backup_path}{Colors.RESET}\n")

        manifest_path = backup_path / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)

            for file in manifest["files"]:
                src = backup_path / file
                dst = self.project_root / file
                if src.exists():
                    shutil.copy2(src, dst)
                    print(f"  Restored: {file}")

        print(f"\n{Colors.GREEN}Rollback complete{Colors.RESET}")
        return True

    def verify_installation(self) -> bool:
        """Verify that upgrades work correctly."""
        print(f"\n{Colors.BOLD}Verifying installation...{Colors.RESET}\n")

        # Install updated dependencies
        print("Installing dependencies...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            cwd=self.project_root,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"{Colors.RED}Installation failed:{Colors.RESET}")
            print(result.stderr)
            return False

        # Run tests
        print("\nRunning tests...")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "ai_engine/tests/", "-v", "--tb=short", "-x"],
            cwd=self.project_root,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"{Colors.YELLOW}Some tests failed:{Colors.RESET}")
            print(result.stdout[-2000:])  # Last 2000 chars
            return False

        print(f"\n{Colors.GREEN}Verification successful!{Colors.RESET}")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="QBITEL Dependency Upgrade Migration Script"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check which upgrades are needed"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply all upgrades"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without applying"
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback to previous backup"
    )
    parser.add_argument(
        "--backup-timestamp",
        type=str,
        help="Specific backup timestamp to rollback to"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify installation after upgrade"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root directory"
    )

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    migrator = DependencyMigrator(project_root)

    print(f"\n{Colors.BOLD}QBITEL Dependency Migration Tool{Colors.RESET}")
    print(f"Project root: {project_root}\n")

    if args.check or not any([args.apply, args.rollback, args.verify]):
        results = migrator.check_upgrades()
        migrator.print_report(results)

        total_critical = len(results["critical"])
        total_high = len(results["high"])

        if total_critical > 0:
            print(f"{Colors.RED}{Colors.BOLD}ACTION REQUIRED: {total_critical} critical upgrades needed!{Colors.RESET}")
            print(f"Run: python {sys.argv[0]} --apply\n")
            return 1

        if total_high > 0:
            print(f"{Colors.YELLOW}Recommended: {total_high} high priority upgrades available{Colors.RESET}")

    if args.apply:
        success = migrator.apply_upgrades(dry_run=args.dry_run)

        if not args.dry_run and args.verify:
            migrator.verify_installation()

        return 0 if success else 1

    if args.rollback:
        success = migrator.rollback(args.backup_timestamp)
        return 0 if success else 1

    if args.verify:
        success = migrator.verify_installation()
        return 0 if success else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
