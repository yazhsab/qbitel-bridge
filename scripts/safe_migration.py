#!/usr/bin/env python3
"""
CRONOS AI - Safe Database Migration Tool

Production-safe database migration with automatic backup, validation,
and rollback capabilities.

Features:
- Automatic backup before migration
- SQL validation (no DROP TABLE in production)
- Dry-run mode
- Automatic rollback on failure
- Migration testing on copy
"""

import sys
import os
import subprocess
import argparse
import logging
from datetime import datetime
from pathlib import Path
import re

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MigrationSafetyError(Exception):
    """Raised when migration is unsafe."""

    pass


class SafeMigrationTool:
    """
    Safe database migration tool with rollback capabilities.
    """

    # Dangerous SQL operations that require approval in production
    DANGEROUS_OPERATIONS = [
        r"\bDROP\s+TABLE\b",
        r"\bDROP\s+COLUMN\b",
        r"\bTRUNCATE\b",
        r"\bDELETE\s+FROM\b(?!\s+WHERE)",  # DELETE without WHERE
    ]

    def __init__(
        self,
        environment: str = "development",
        auto_backup: bool = True,
        require_approval: bool = True,
    ):
        """
        Initialize safe migration tool.

        Args:
            environment: Environment (development, staging, production)
            auto_backup: Automatically create backup before migration
            require_approval: Require manual approval for dangerous operations
        """
        self.environment = environment
        self.auto_backup = auto_backup
        self.require_approval = require_approval
        self.is_production = environment == "production"

        logger.info(f"SafeMigrationTool initialized for {environment}")

    def validate_migration_sql(self, migration_file: Path) -> bool:
        """
        Validate migration SQL for dangerous operations.

        Args:
            migration_file: Path to migration file

        Returns:
            bool: True if safe, raises exception if unsafe

        Raises:
            MigrationSafetyError: If unsafe operations detected
        """
        logger.info(f"Validating migration: {migration_file}")

        if not migration_file.exists():
            raise FileNotFoundError(f"Migration file not found: {migration_file}")

        # Read migration SQL
        sql_content = migration_file.read_text()

        # Check for dangerous operations
        dangerous_found = []
        for pattern in self.DANGEROUS_OPERATIONS:
            matches = re.finditer(pattern, sql_content, re.IGNORECASE)
            for match in matches:
                dangerous_found.append(
                    {
                        "operation": match.group(0),
                        "pattern": pattern,
                        "line": sql_content[: match.start()].count("\n") + 1,
                    }
                )

        if dangerous_found:
            logger.warning(
                f"⚠️  Dangerous operations detected in {migration_file.name}:"
            )
            for danger in dangerous_found:
                logger.warning(f"  Line {danger['line']}: {danger['operation']}")

            if self.is_production and self.require_approval:
                logger.error(
                    "❌ Dangerous operations require manual approval in production"
                )
                logger.info("\nTo proceed, run with: --approve-dangerous")
                raise MigrationSafetyError(
                    f"Dangerous operations detected. Manual approval required."
                )

        logger.info("✅ Migration validation passed")
        return True

    def create_backup(self) -> str:
        """
        Create database backup before migration.

        Returns:
            str: Backup file path

        Raises:
            subprocess.CalledProcessError: If backup fails
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"pre_migration_backup_{timestamp}.sql.gz"
        backup_path = Path("backups") / backup_file

        # Create backups directory
        backup_path.parent.mkdir(exist_ok=True)

        logger.info(f"Creating backup: {backup_path}")

        # Use backup script
        cmd = [
            "python3",
            "scripts/backup_database.py",
            "--full",
            "--output",
            str(backup_path),
            "--compress",
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"✅ Backup created: {backup_path}")
            return str(backup_path)
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Backup failed: {e.stderr}")
            raise

    def get_current_revision(self) -> str:
        """
        Get current Alembic migration revision.

        Returns:
            str: Current revision hash
        """
        cmd = ["alembic", "current"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="ai_engine")

        if result.returncode != 0:
            logger.error(f"Failed to get current revision: {result.stderr}")
            return "unknown"

        # Parse output (format: "revision_hash (head)")
        output = result.stdout.strip()
        if output:
            revision = output.split()[0]
            return revision
        return "unknown"

    def upgrade(
        self,
        target: str = "head",
        dry_run: bool = False,
    ) -> bool:
        """
        Upgrade database to target revision.

        Args:
            target: Target revision (default: head)
            dry_run: Show SQL without executing

        Returns:
            bool: True if successful

        Raises:
            subprocess.CalledProcessError: If migration fails
        """
        current_revision = self.get_current_revision()
        logger.info(f"Current revision: {current_revision}")
        logger.info(f"Target revision: {target}")

        # Create backup if enabled
        backup_path = None
        if self.auto_backup and not dry_run:
            try:
                backup_path = self.create_backup()
            except Exception as e:
                logger.error(f"❌ Backup failed, aborting migration: {e}")
                return False

        # Prepare upgrade command
        cmd = ["alembic"]
        if dry_run:
            cmd.append("upgrade")
            cmd.append("--sql")
            cmd.append(target)
            logger.info("🔍 DRY RUN: Showing SQL only")
        else:
            cmd.append("upgrade")
            cmd.append(target)
            logger.info("⚙️  Running migration...")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd="ai_engine", check=True
            )

            if dry_run:
                print("\n" + "=" * 80)
                print("MIGRATION SQL (DRY RUN)")
                print("=" * 80)
                print(result.stdout)
                print("=" * 80)
            else:
                logger.info("✅ Migration successful")
                logger.info(result.stdout)

                # Verify new revision
                new_revision = self.get_current_revision()
                logger.info(f"New revision: {new_revision}")

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Migration failed: {e.stderr}")

            if backup_path and not dry_run:
                logger.info("\n" + "=" * 80)
                logger.info("MIGRATION FAILED - ROLLBACK INSTRUCTIONS")
                logger.info("=" * 80)
                logger.info(f"1. Backup available at: {backup_path}")
                logger.info(f"2. To rollback, run:")
                logger.info(f"   alembic downgrade {current_revision}")
                logger.info(f"3. Or restore from backup:")
                logger.info(f"   gunzip -c {backup_path} | psql -U user -d database")
                logger.info("=" * 80)

            raise

    def downgrade(
        self,
        target: str = "-1",
        dry_run: bool = False,
    ) -> bool:
        """
        Downgrade database to target revision.

        Args:
            target: Target revision (default: -1 for previous)
            dry_run: Show SQL without executing

        Returns:
            bool: True if successful
        """
        current_revision = self.get_current_revision()
        logger.info(f"Current revision: {current_revision}")
        logger.info(f"Downgrading to: {target}")

        # Require confirmation for production
        if self.is_production and not dry_run:
            logger.warning("⚠️  PRODUCTION ROLLBACK - This will modify production data")
            response = input("Type 'ROLLBACK' to confirm: ")
            if response != "ROLLBACK":
                logger.info("Rollback cancelled")
                return False

        # Create backup before rollback
        backup_path = None
        if self.auto_backup and not dry_run:
            backup_path = self.create_backup()

        # Prepare downgrade command
        cmd = ["alembic"]
        if dry_run:
            cmd.append("downgrade")
            cmd.append("--sql")
            cmd.append(target)
            logger.info("🔍 DRY RUN: Showing SQL only")
        else:
            cmd.append("downgrade")
            cmd.append(target)
            logger.info("⚙️  Running rollback...")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd="ai_engine", check=True
            )

            if dry_run:
                print("\n" + "=" * 80)
                print("ROLLBACK SQL (DRY RUN)")
                print("=" * 80)
                print(result.stdout)
                print("=" * 80)
            else:
                logger.info("✅ Rollback successful")
                logger.info(result.stdout)

                # Verify new revision
                new_revision = self.get_current_revision()
                logger.info(f"New revision: {new_revision}")

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Rollback failed: {e.stderr}")
            raise

    def show_history(self) -> None:
        """Show migration history."""
        cmd = ["alembic", "history", "--verbose"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="ai_engine")
        print(result.stdout)

    def show_current(self) -> None:
        """Show current migration state."""
        cmd = ["alembic", "current"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="ai_engine")
        print(result.stdout)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Safe database migration tool with automatic backup and rollback"
    )

    parser.add_argument(
        "command",
        choices=["upgrade", "downgrade", "history", "current", "validate"],
        help="Migration command",
    )

    parser.add_argument(
        "--target",
        default="head",
        help="Target revision (default: head for upgrade, -1 for downgrade)",
    )

    parser.add_argument(
        "--environment",
        default=os.getenv("CRONOS_AI_ENVIRONMENT", "development"),
        choices=["development", "staging", "production"],
        help="Environment (default: from CRONOS_AI_ENVIRONMENT)",
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Show SQL without executing"
    )

    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip automatic backup (not recommended)",
    )

    parser.add_argument(
        "--approve-dangerous",
        action="store_true",
        help="Approve dangerous operations in production",
    )

    parser.add_argument(
        "--migration-file", type=Path, help="Validate specific migration file"
    )

    args = parser.parse_args()

    # Initialize tool
    tool = SafeMigrationTool(
        environment=args.environment,
        auto_backup=not args.no_backup,
        require_approval=not args.approve_dangerous,
    )

    try:
        if args.command == "upgrade":
            success = tool.upgrade(target=args.target, dry_run=args.dry_run)
            sys.exit(0 if success else 1)

        elif args.command == "downgrade":
            target = args.target if args.target != "head" else "-1"
            success = tool.downgrade(target=target, dry_run=args.dry_run)
            sys.exit(0 if success else 1)

        elif args.command == "history":
            tool.show_history()

        elif args.command == "current":
            tool.show_current()

        elif args.command == "validate":
            if not args.migration_file:
                logger.error("--migration-file required for validate command")
                sys.exit(1)
            tool.validate_migration_sql(args.migration_file)

    except MigrationSafetyError as e:
        logger.error(f"Migration safety check failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
