"""
QBITEL - Backup Automation

Automated backup scheduling, monitoring, and alerting system.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

from .backup_manager import BackupManager, BackupType, BackupStatus


@dataclass
class BackupSchedule:
    """Backup schedule configuration."""

    name: str
    source_paths: List[str]
    backup_type: BackupType
    schedule_cron: str  # Cron expression
    retention_days: int
    enabled: bool = True
    tags: Dict[str, str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


class BackupMonitor:
    """Monitor backup health and send alerts."""

    def __init__(self, backup_manager: BackupManager):
        """Initialize backup monitor."""
        self.backup_manager = backup_manager
        self.logger = logging.getLogger(__name__)
        self.alert_thresholds = {
            "max_backup_age_hours": 24,
            "min_verified_backups": 3,
            "max_failed_backups": 2,
            "max_backup_size_gb": 100,
        }

    async def check_backup_health(self) -> Dict[str, Any]:
        """
        Check overall backup health.

        Returns:
            Health check results with alerts
        """
        self.logger.info("Checking backup health")

        health_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "healthy",
            "alerts": [],
            "warnings": [],
            "statistics": {},
        }

        try:
            # Get backup statistics
            stats = self.backup_manager.get_backup_statistics()
            health_report["statistics"] = stats

            # Check backup age
            if stats.get("newest_backup"):
                newest_backup = stats["newest_backup"]
                age_hours = (datetime.utcnow() - newest_backup).total_seconds() / 3600

                if age_hours > self.alert_thresholds["max_backup_age_hours"]:
                    health_report["alerts"].append(
                        {
                            "severity": "critical",
                            "message": f"No recent backups. Last backup was {age_hours:.1f} hours ago",
                            "threshold": self.alert_thresholds["max_backup_age_hours"],
                        }
                    )
                    health_report["status"] = "unhealthy"
            else:
                health_report["alerts"].append(
                    {
                        "severity": "critical",
                        "message": "No backups found",
                        "threshold": None,
                    }
                )
                health_report["status"] = "unhealthy"

            # Check verified backups
            verified_count = stats.get("verified_backups", 0)
            if verified_count < self.alert_thresholds["min_verified_backups"]:
                health_report["warnings"].append(
                    {
                        "severity": "warning",
                        "message": f"Only {verified_count} verified backups (minimum: {self.alert_thresholds['min_verified_backups']})",
                        "threshold": self.alert_thresholds["min_verified_backups"],
                    }
                )
                if health_report["status"] == "healthy":
                    health_report["status"] = "degraded"

            # Check failed backups
            failed_count = stats.get("failed_backups", 0)
            if failed_count > self.alert_thresholds["max_failed_backups"]:
                health_report["alerts"].append(
                    {
                        "severity": "high",
                        "message": f"{failed_count} failed backups detected",
                        "threshold": self.alert_thresholds["max_failed_backups"],
                    }
                )
                health_report["status"] = "unhealthy"

            # Check total backup size
            total_size_gb = stats.get("total_size_gb", 0)
            if total_size_gb > self.alert_thresholds["max_backup_size_gb"]:
                health_report["warnings"].append(
                    {
                        "severity": "warning",
                        "message": f"Total backup size ({total_size_gb:.2f} GB) exceeds threshold",
                        "threshold": self.alert_thresholds["max_backup_size_gb"],
                    }
                )

            self.logger.info(
                f"Backup health check completed: {health_report['status']}"
            )

        except Exception as e:
            self.logger.error(f"Backup health check failed: {e}")
            health_report["status"] = "error"
            health_report["alerts"].append(
                {
                    "severity": "critical",
                    "message": f"Health check error: {str(e)}",
                    "threshold": None,
                }
            )

        return health_report

    async def send_alert(self, alert: Dict[str, Any]) -> None:
        """
        Send backup alert.

        Args:
            alert: Alert information
        """
        self.logger.warning(f"Backup alert: {alert}")

        # In production, integrate with alerting systems:
        # - PagerDuty
        # - Slack
        # - Email
        # - SMS
        # - Webhook

        # Example: Send to monitoring system
        # await self.monitoring_client.send_alert(alert)


class BackupAutomation:
    """Automated backup scheduling and execution."""

    def __init__(self, backup_manager: BackupManager, schedules: List[BackupSchedule]):
        """Initialize backup automation."""
        self.backup_manager = backup_manager
        self.schedules = schedules
        self.monitor = BackupMonitor(backup_manager)
        self.logger = logging.getLogger(__name__)

        self._running = False
        self._tasks: List[asyncio.Task] = []

    async def start(self) -> None:
        """Start automated backup system."""
        if self._running:
            self.logger.warning("Backup automation already running")
            return

        self._running = True
        self.logger.info("Starting backup automation")

        # Start scheduled backup tasks
        for schedule in self.schedules:
            if schedule.enabled:
                task = asyncio.create_task(self._run_scheduled_backup(schedule))
                self._tasks.append(task)

        # Start monitoring task
        monitor_task = asyncio.create_task(self._run_monitoring())
        self._tasks.append(monitor_task)

        # Start cleanup task
        cleanup_task = asyncio.create_task(self._run_cleanup())
        self._tasks.append(cleanup_task)

    async def stop(self) -> None:
        """Stop automated backup system."""
        if not self._running:
            return

        self._running = False
        self.logger.info("Stopping backup automation")

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)

        self._tasks.clear()

    async def _run_scheduled_backup(self, schedule: BackupSchedule) -> None:
        """Run scheduled backup."""
        self.logger.info(f"Starting scheduled backup: {schedule.name}")

        while self._running:
            try:
                # Create backup
                metadata = await self.backup_manager.create_backup(
                    source_paths=schedule.source_paths,
                    backup_type=schedule.backup_type,
                    tags={
                        **schedule.tags,
                        "schedule": schedule.name,
                        "automated": "true",
                    },
                )

                self.logger.info(f"Scheduled backup completed: {metadata.backup_id}")

                # Wait for next scheduled time
                # In production, use proper cron scheduling
                await asyncio.sleep(3600)  # 1 hour for testing

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Scheduled backup failed: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry

    async def _run_monitoring(self) -> None:
        """Run backup monitoring."""
        self.logger.info("Starting backup monitoring")

        while self._running:
            try:
                # Check backup health
                health_report = await self.monitor.check_backup_health()

                # Send alerts if needed
                for alert in health_report.get("alerts", []):
                    await self.monitor.send_alert(alert)

                # Wait before next check
                await asyncio.sleep(300)  # Check every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Backup monitoring error: {e}")
                await asyncio.sleep(60)

    async def _run_cleanup(self) -> None:
        """Run backup cleanup."""
        self.logger.info("Starting backup cleanup")

        while self._running:
            try:
                # Clean up old backups
                deleted = await self.backup_manager.cleanup_old_backups()

                if deleted:
                    self.logger.info(f"Cleaned up {len(deleted)} old backups")

                # Wait before next cleanup (daily)
                await asyncio.sleep(86400)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Backup cleanup error: {e}")
                await asyncio.sleep(3600)

    async def test_restore(self, backup_id: str) -> bool:
        """
        Test backup restore procedure.

        Args:
            backup_id: Backup ID to test

        Returns:
            True if test successful, False otherwise
        """
        self.logger.info(f"Testing restore for backup: {backup_id}")

        import tempfile

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Restore to temporary directory
                success = await self.backup_manager.restore_backup(
                    backup_id=backup_id,
                    restore_path=temp_dir,
                    verify_before_restore=True,
                )

                if success:
                    self.logger.info(f"Restore test successful: {backup_id}")
                else:
                    self.logger.error(f"Restore test failed: {backup_id}")

                return success

        except Exception as e:
            self.logger.error(f"Restore test error: {e}")
            return False

    async def run_disaster_recovery_drill(self) -> Dict[str, Any]:
        """
        Run disaster recovery drill.

        Returns:
            Drill results
        """
        self.logger.info("Starting disaster recovery drill")

        drill_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success",
            "tests": [],
            "duration_seconds": 0,
        }

        start_time = datetime.utcnow()

        try:
            # Get latest verified backup
            backups = self.backup_manager.list_backups(status=BackupStatus.VERIFIED)

            if not backups:
                drill_results["status"] = "failed"
                drill_results["tests"].append(
                    {
                        "name": "find_backup",
                        "status": "failed",
                        "message": "No verified backups found",
                    }
                )
                return drill_results

            latest_backup = backups[0]

            # Test 1: Verify backup integrity
            verify_success = await self.backup_manager.verify_backup(
                latest_backup.backup_id
            )
            drill_results["tests"].append(
                {
                    "name": "verify_backup",
                    "status": "passed" if verify_success else "failed",
                    "backup_id": latest_backup.backup_id,
                }
            )

            # Test 2: Test restore
            restore_success = await self.test_restore(latest_backup.backup_id)
            drill_results["tests"].append(
                {
                    "name": "test_restore",
                    "status": "passed" if restore_success else "failed",
                    "backup_id": latest_backup.backup_id,
                }
            )

            # Test 3: Check backup health
            health_report = await self.monitor.check_backup_health()
            drill_results["tests"].append(
                {
                    "name": "health_check",
                    "status": (
                        "passed"
                        if health_report["status"] in ["healthy", "degraded"]
                        else "failed"
                    ),
                    "health_status": health_report["status"],
                }
            )

            # Determine overall status
            if all(test["status"] == "passed" for test in drill_results["tests"]):
                drill_results["status"] = "success"
            else:
                drill_results["status"] = "failed"

        except Exception as e:
            self.logger.error(f"Disaster recovery drill error: {e}")
            drill_results["status"] = "error"
            drill_results["error"] = str(e)

        finally:
            end_time = datetime.utcnow()
            drill_results["duration_seconds"] = (end_time - start_time).total_seconds()

        self.logger.info(
            f"Disaster recovery drill completed: {drill_results['status']}"
        )

        return drill_results


async def main():
    """Example usage."""
    logging.basicConfig(level=logging.INFO)

    # Initialize backup manager
    from .backup_manager import BackupManager

    manager = BackupManager(
        backup_root="/tmp/qbitel_backups",
        encryption_key="test_encryption_key",
        retention_days=30,
    )

    # Define backup schedules
    schedules = [
        BackupSchedule(
            name="daily_full_backup",
            source_paths=["/tmp/test_data"],
            backup_type=BackupType.FULL,
            schedule_cron="0 2 * * *",  # Daily at 2 AM
            retention_days=30,
            tags={"type": "daily", "priority": "high"},
        ),
        BackupSchedule(
            name="hourly_incremental",
            source_paths=["/tmp/test_data"],
            backup_type=BackupType.INCREMENTAL,
            schedule_cron="0 * * * *",  # Every hour
            retention_days=7,
            tags={"type": "incremental", "priority": "medium"},
        ),
    ]

    # Initialize automation
    automation = BackupAutomation(manager, schedules)

    # Run disaster recovery drill
    drill_results = await automation.run_disaster_recovery_drill()
    print(f"Disaster recovery drill results:")
    print(json.dumps(drill_results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
