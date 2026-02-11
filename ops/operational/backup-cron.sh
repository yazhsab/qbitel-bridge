#!/bin/bash
#
# QBITEL - Automated Backup Cron Job
#
# Install this cron job to run automated backups:
#
# 1. Copy this script to /usr/local/bin/qbitel-backup.sh
# 2. Make it executable: chmod +x /usr/local/bin/qbitel-backup.sh
# 3. Add to crontab: crontab -e
# 4. Add these lines:
#
#    # Daily full backup at 2 AM
#    0 2 * * * /usr/local/bin/qbitel-backup.sh daily >> /var/log/qbitel/backup.log 2>&1
#
#    # Weekly backup on Sunday at 3 AM
#    0 3 * * 0 /usr/local/bin/qbitel-backup.sh weekly >> /var/log/qbitel/backup.log 2>&1
#
#    # Monthly backup on 1st of month at 4 AM
#    0 4 1 * * /usr/local/bin/qbitel-backup.sh monthly >> /var/log/qbitel/backup.log 2>&1
#
#    # Cleanup old backups daily at 5 AM
#    0 5 * * * /usr/local/bin/qbitel-backup.sh cleanup >> /var/log/qbitel/backup.log 2>&1
#

set -euo pipefail

# Configuration
QBITEL_AI_HOME="${QBITEL_AI_HOME:-/opt/qbitel}"
BACKUP_SCRIPT="${QBITEL_AI_HOME}/scripts/backup_database.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_FILE="${LOG_FILE:-/var/log/qbitel/backup.log}"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Load environment variables from .env if exists
ENV_FILE="${QBITEL_AI_HOME}/.env.production"
if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Error handling
error_exit() {
    log "ERROR: $1"
    # Send notification (implement your notification method)
    # notify_failure "$1"
    exit 1
}

# Backup function
run_backup() {
    local backup_type="$1"
    local backup_name="${backup_type}_$(date +'%Y%m%d_%H%M%S')"

    log "Starting $backup_type backup: $backup_name"

    # Run backup
    if ! "$PYTHON_BIN" "$BACKUP_SCRIPT" --full --name "$backup_name"; then
        error_exit "Backup failed for $backup_name"
    fi

    log "✅ Backup completed successfully: $backup_name"
}

# Cleanup function
run_cleanup() {
    log "Starting backup cleanup..."

    if ! "$PYTHON_BIN" "$BACKUP_SCRIPT" --cleanup; then
        error_exit "Cleanup failed"
    fi

    log "✅ Cleanup completed successfully"
}

# Verify backups function
verify_backups() {
    log "Verifying recent backups..."

    if ! "$PYTHON_BIN" "$BACKUP_SCRIPT" --list | grep -q "✅"; then
        log "⚠️  Warning: Some backups may not be verified"
    else
        log "✅ Backup verification passed"
    fi
}

# Main execution
case "${1:-}" in
    daily)
        log "========== Daily Backup Started =========="
        run_backup "daily"
        verify_backups
        log "========== Daily Backup Completed =========="
        ;;

    weekly)
        log "========== Weekly Backup Started =========="
        run_backup "weekly"
        verify_backups
        log "========== Weekly Backup Completed =========="
        ;;

    monthly)
        log "========== Monthly Backup Started =========="
        run_backup "monthly"
        verify_backups
        log "========== Monthly Backup Completed =========="
        ;;

    cleanup)
        log "========== Cleanup Started =========="
        run_cleanup
        log "========== Cleanup Completed =========="
        ;;

    verify)
        log "========== Verification Started =========="
        verify_backups
        log "========== Verification Completed =========="
        ;;

    *)
        echo "Usage: $0 {daily|weekly|monthly|cleanup|verify}"
        echo ""
        echo "Options:"
        echo "  daily    - Create daily backup"
        echo "  weekly   - Create weekly backup"
        echo "  monthly  - Create monthly backup"
        echo "  cleanup  - Remove old backups based on retention policy"
        echo "  verify   - Verify recent backups"
        exit 1
        ;;
esac

exit 0
