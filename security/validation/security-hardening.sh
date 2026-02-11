#!/bin/bash

# QBITEL Security Hardening Script
# Automated security hardening for production deployment

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/var/log/qbitel/security-hardening.log"
CONFIG_DIR="/etc/qbitel/security"
BACKUP_DIR="/var/backups/qbitel/security"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $*" | tee -a "$LOG_FILE"
}

info() {
    log "INFO" "${BLUE}$*${NC}"
}

warn() {
    log "WARN" "${YELLOW}$*${NC}"
}

error() {
    log "ERROR" "${RED}$*${NC}"
}

success() {
    log "SUCCESS" "${GREEN}$*${NC}"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root"
        exit 1
    fi
}

# Create required directories
setup_directories() {
    info "Setting up security directories..."
    
    mkdir -p "$CONFIG_DIR"
    mkdir -p "$BACKUP_DIR"
    mkdir -p "/var/log/qbitel"
    mkdir -p "/etc/qbitel/certs"
    mkdir -p "/etc/qbitel/keys"
    
    # Set proper permissions
    chmod 755 "$CONFIG_DIR"
    chmod 700 "/etc/qbitel/keys"
    chmod 755 "/etc/qbitel/certs"
    
    success "Directories created successfully"
}

# System hardening
harden_kernel() {
    info "Applying kernel security hardening..."
    
    # Backup original sysctl configuration
    cp /etc/sysctl.conf "$BACKUP_DIR/sysctl.conf.backup.$(date +%s)"
    
    cat > /etc/sysctl.d/99-qbitel-security.conf << 'EOF'
# QBITEL Security Hardening - Kernel Parameters

# Network Security
net.ipv4.ip_forward=0
net.ipv4.conf.all.send_redirects=0
net.ipv4.conf.default.send_redirects=0
net.ipv4.conf.all.accept_redirects=0
net.ipv4.conf.default.accept_redirects=0
net.ipv4.conf.all.secure_redirects=0
net.ipv4.conf.default.secure_redirects=0
net.ipv4.conf.all.accept_source_route=0
net.ipv4.conf.default.accept_source_route=0
net.ipv4.conf.all.log_martians=1
net.ipv4.conf.default.log_martians=1
net.ipv4.icmp_echo_ignore_broadcasts=1
net.ipv4.icmp_ignore_bogus_error_responses=1
net.ipv4.tcp_syncookies=1
net.ipv4.tcp_max_syn_backlog=2048
net.ipv4.tcp_synack_retries=2
net.ipv4.tcp_syn_retries=5

# IPv6 Security
net.ipv6.conf.all.accept_redirects=0
net.ipv6.conf.default.accept_redirects=0
net.ipv6.conf.all.accept_source_route=0
net.ipv6.conf.default.accept_source_route=0

# Memory Protection
kernel.dmesg_restrict=1
kernel.kptr_restrict=2
kernel.yama.ptrace_scope=1
kernel.randomize_va_space=2

# File System Security
fs.suid_dumpable=0
fs.protected_hardlinks=1
fs.protected_symlinks=1
fs.protected_fifos=2
fs.protected_regular=2

# Process Security
kernel.core_uses_pid=1
kernel.ctrl-alt-del=0
EOF

    # Apply sysctl settings
    sysctl -p /etc/sysctl.d/99-qbitel-security.conf
    
    success "Kernel hardening applied successfully"
}

# Configure firewall
configure_firewall() {
    info "Configuring enterprise firewall rules..."
    
    # Backup current iptables rules
    iptables-save > "$BACKUP_DIR/iptables.backup.$(date +%s)"
    
    # Install iptables-persistent if not present
    if ! dpkg -l | grep -q iptables-persistent; then
        DEBIAN_FRONTEND=noninteractive apt-get install -y iptables-persistent
    fi
    
    # Flush existing rules
    iptables -F
    iptables -X
    iptables -t nat -F
    iptables -t nat -X
    iptables -t mangle -F
    iptables -t mangle -X
    
    # Set default policies
    iptables -P INPUT DROP
    iptables -P FORWARD DROP
    iptables -P OUTPUT ACCEPT
    
    # Allow loopback traffic
    iptables -A INPUT -i lo -j ACCEPT
    iptables -A OUTPUT -o lo -j ACCEPT
    
    # Allow established and related connections
    iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
    
    # Allow SSH (restrict to specific networks if needed)
    iptables -A INPUT -p tcp --dport 22 -m state --state NEW -j ACCEPT
    
    # Allow QBITEL services
    iptables -A INPUT -p tcp --dport 8000 -j ACCEPT  # AI Engine
    iptables -A INPUT -p tcp --dport 8080 -j ACCEPT  # Control Plane
    iptables -A INPUT -p tcp --dport 9090 -j ACCEPT  # Data Plane
    iptables -A INPUT -p tcp --dport 8001 -j ACCEPT  # Policy Engine
    iptables -A INPUT -p tcp --dport 8002 -j ACCEPT  # Management API
    
    # Allow monitoring and metrics
    iptables -A INPUT -p tcp --dport 9100 -j ACCEPT  # Node Exporter
    iptables -A INPUT -p tcp --dport 9090 -j ACCEPT  # Prometheus
    iptables -A INPUT -p tcp --dport 3000 -j ACCEPT  # Grafana
    
    # Allow Kubernetes API server (if applicable)
    iptables -A INPUT -p tcp --dport 6443 -j ACCEPT
    
    # Allow HTTPS and HTTP
    iptables -A INPUT -p tcp --dport 443 -j ACCEPT
    iptables -A INPUT -p tcp --dport 80 -j ACCEPT
    
    # Rate limiting for SSH
    iptables -A INPUT -p tcp --dport 22 -m recent --set --name SSH
    iptables -A INPUT -p tcp --dport 22 -m recent --update --seconds 60 --hitcount 4 --name SSH -j DROP
    
    # Log dropped packets
    iptables -A INPUT -j LOG --log-prefix "QBITEL-DROP: " --log-level 4
    
    # Save rules
    iptables-save > /etc/iptables/rules.v4
    
    success "Firewall rules configured successfully"
}

# Configure SSH hardening
harden_ssh() {
    info "Hardening SSH configuration..."
    
    # Backup original SSH config
    cp /etc/ssh/sshd_config "$BACKUP_DIR/sshd_config.backup.$(date +%s)"
    
    # Create hardened SSH configuration
    cat > /etc/ssh/sshd_config.d/99-qbitel-hardening.conf << 'EOF'
# QBITEL SSH Security Hardening

# Protocol and Encryption
Protocol 2
Port 22
AddressFamily inet

# Authentication
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
AuthenticationMethods publickey
MaxAuthTries 3
MaxSessions 5
LoginGraceTime 60

# Security Features
UsePAM yes
X11Forwarding no
AllowTcpForwarding no
AllowAgentForwarding no
PermitTunnel no
PermitUserEnvironment no
ClientAliveInterval 300
ClientAliveCountMax 2

# Ciphers and MACs
Ciphers chacha20-poly1305@openssh.com,aes256-gcm@openssh.com,aes128-gcm@openssh.com,aes256-ctr,aes192-ctr,aes128-ctr
MACs hmac-sha2-256-etm@openssh.com,hmac-sha2-512-etm@openssh.com,hmac-sha2-256,hmac-sha2-512
KexAlgorithms curve25519-sha256@libssh.org,ecdh-sha2-nistp521,ecdh-sha2-nistp384,ecdh-sha2-nistp256,diffie-hellman-group16-sha512,diffie-hellman-group18-sha512,diffie-hellman-group14-sha256

# Logging
SyslogFacility AUTH
LogLevel VERBOSE

# Banner
Banner /etc/issue.net
EOF

    # Create security banner
    cat > /etc/issue.net << 'EOF'
***************************************************************************
                        QBITEL SECURE SYSTEM
***************************************************************************
WARNING: This system is for authorized users only. All activities on this
system are monitored and recorded. Unauthorized access is prohibited and
will be prosecuted to the full extent of the law.
***************************************************************************
EOF

    # Restart SSH service
    systemctl restart sshd
    
    success "SSH hardening completed successfully"
}

# Configure auditd for security monitoring
configure_audit() {
    info "Configuring audit logging..."
    
    # Install auditd if not present
    if ! dpkg -l | grep -q auditd; then
        apt-get update
        apt-get install -y auditd audispd-plugins
    fi
    
    # Backup original audit configuration
    cp /etc/audit/auditd.conf "$BACKUP_DIR/auditd.conf.backup.$(date +%s)"
    
    # Configure auditd
    cat > /etc/audit/auditd.conf << 'EOF'
# QBITEL Audit Configuration

log_file = /var/log/audit/audit.log
log_format = RAW
log_group = adm
priority_boost = 4
flush = INCREMENTAL_ASYNC
freq = 50
num_logs = 10
disp_qos = lossy
dispatcher = /sbin/audispd
name_format = HOSTNAME
max_log_file = 100
max_log_file_action = ROTATE
space_left = 500
space_left_action = SYSLOG
admin_space_left = 100
admin_space_left_action = SUSPEND
disk_full_action = SUSPEND
disk_error_action = SUSPEND
use_libwrap = yes
tcp_listen_queue = 5
tcp_max_per_addr = 1
tcp_client_max_idle = 0
enable_krb5 = no
krb5_principal = auditd
EOF

    # Configure audit rules
    cat > /etc/audit/rules.d/99-qbitel-security.rules << 'EOF'
# QBITEL Security Audit Rules

# Delete all previous rules
-D

# Set buffer size
-b 8192

# Set failure mode
-f 1

# Monitor authentication events
-w /etc/passwd -p wa -k identity
-w /etc/group -p wa -k identity
-w /etc/shadow -p wa -k identity
-w /etc/sudoers -p wa -k identity

# Monitor system configuration changes
-w /etc/qbitel/ -p wa -k qbitel-config
-w /etc/ssl/ -p wa -k ssl-config
-w /etc/ssh/sshd_config -p wa -k ssh-config

# Monitor privilege escalation
-w /bin/su -p x -k privilege-escalation
-w /usr/bin/sudo -p x -k privilege-escalation
-w /etc/sudoers -p wa -k privilege-escalation

# Monitor network configuration
-w /etc/hosts -p wa -k network-config
-w /etc/network/ -p wa -k network-config

# Monitor system calls
-a always,exit -F arch=b64 -S adjtimex -S settimeofday -k time-change
-a always,exit -F arch=b32 -S adjtimex -S settimeofday -S stime -k time-change
-a always,exit -F arch=b64 -S clock_settime -k time-change
-a always,exit -F arch=b32 -S clock_settime -k time-change

# Monitor file access
-a always,exit -F arch=b64 -S chmod -S fchmod -S fchmodat -F auid>=1000 -F auid!=4294967295 -k perm-mod
-a always,exit -F arch=b32 -S chmod -S fchmod -S fchmodat -F auid>=1000 -F auid!=4294967295 -k perm-mod

# Make rules immutable
-e 2
EOF

    # Start auditd service
    systemctl enable auditd
    systemctl restart auditd
    
    success "Audit logging configured successfully"
}

# Install and configure fail2ban
configure_fail2ban() {
    info "Configuring fail2ban for intrusion prevention..."
    
    # Install fail2ban if not present
    if ! dpkg -l | grep -q fail2ban; then
        apt-get update
        apt-get install -y fail2ban
    fi
    
    # Create fail2ban configuration for QBITEL
    cat > /etc/fail2ban/jail.d/qbitel.conf << 'EOF'
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3
backend = auto
usedns = warn
logencoding = auto
enabled = false
mode = normal
filter = %(__name__)s[mode=%(mode)s]

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
bantime = 1800

[qbitel-auth]
enabled = true
port = 8000,8080,9090,8001,8002
filter = qbitel-auth
logpath = /var/log/qbitel/auth.log
maxretry = 5
bantime = 3600

[nginx-http-auth]
enabled = true
port = http,https
filter = nginx-http-auth
logpath = /var/log/nginx/error.log
maxretry = 3
bantime = 600
EOF

    # Create custom filter for QBITEL authentication
    cat > /etc/fail2ban/filter.d/qbitel-auth.conf << 'EOF'
[Definition]
failregex = ^.*Authentication failed for .* from <HOST>.*$
            ^.*Invalid login attempt from <HOST>.*$
            ^.*Failed authentication from <HOST>.*$
ignoreregex =
EOF

    # Start fail2ban service
    systemctl enable fail2ban
    systemctl restart fail2ban
    
    success "Fail2ban configured successfully"
}

# Configure log rotation and retention
configure_logging() {
    info "Configuring secure logging and retention..."
    
    # Create QBITEL log rotation configuration
    cat > /etc/logrotate.d/qbitel << 'EOF'
/var/log/qbitel/*.log {
    daily
    rotate 2555
    compress
    delaycompress
    missingok
    notifempty
    create 0640 qbitel qbitel
    postrotate
        systemctl reload rsyslog > /dev/null 2>&1 || true
    endscript
}

/var/log/qbitel/audit/*.log {
    daily
    rotate 2555
    compress
    delaycompress
    missingok
    notifempty
    create 0600 root root
    copytruncate
}

/var/log/qbitel/security/*.log {
    daily
    rotate 365
    compress
    delaycompress
    missingok
    notifempty
    create 0600 root root
    sharedscripts
    postrotate
        systemctl reload qbitel-security-monitor > /dev/null 2>&1 || true
    endscript
}
EOF

    # Configure rsyslog for QBITEL
    cat > /etc/rsyslog.d/99-qbitel.conf << 'EOF'
# QBITEL Logging Configuration

# Create separate log files for different components
:programname, isequal, "qbitel-dataplane"    /var/log/qbitel/dataplane.log
:programname, isequal, "qbitel-controlplane" /var/log/qbitel/controlplane.log
:programname, isequal, "qbitelengine"     /var/log/qbitel/aiengine.log
:programname, isequal, "qbitel-policy"       /var/log/qbitel/policy.log
:programname, isequal, "qbitel-mgmtapi"      /var/log/qbitel/mgmtapi.log

# Security events
:msg, contains, "QBITEL-SECURITY"         /var/log/qbitel/security.log
:msg, contains, "Authentication"             /var/log/qbitel/auth.log
:msg, contains, "QBITEL-DROP"             /var/log/qbitel/firewall.log

# Stop processing after writing to QBITEL logs
& stop
EOF

    # Restart rsyslog
    systemctl restart rsyslog
    
    success "Logging configuration completed successfully"
}

# Set file and directory permissions
secure_permissions() {
    info "Securing file and directory permissions..."
    
    # System directories
    chmod 755 /etc/qbitel
    chmod 700 /etc/qbitel/keys
    chmod 755 /etc/qbitel/certs
    chmod 644 /etc/qbitel/*.yaml 2>/dev/null || true
    
    # Log directories
    mkdir -p /var/log/qbitel/{security,audit,auth}
    chown -R root:adm /var/log/qbitel
    chmod -R 640 /var/log/qbitel/*.log 2>/dev/null || true
    chmod 600 /var/log/qbitel/security/*.log 2>/dev/null || true
    
    # Certificate directories
    if [ -d "/etc/ssl/certs" ]; then
        chmod 644 /etc/ssl/certs/*.crt 2>/dev/null || true
        chmod 644 /etc/ssl/certs/*.pem 2>/dev/null || true
    fi
    
    if [ -d "/etc/ssl/private" ]; then
        chmod 600 /etc/ssl/private/*.key 2>/dev/null || true
        chmod 600 /etc/ssl/private/*.pem 2>/dev/null || true
    fi
    
    # Temporary directories
    chmod 1777 /tmp
    chmod 1777 /var/tmp
    
    success "File permissions secured successfully"
}

# Install security tools
install_security_tools() {
    info "Installing essential security tools..."
    
    # Update package lists
    apt-get update
    
    # Install security and monitoring tools
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        aide \
        rkhunter \
        chkrootkit \
        lynis \
        nmap \
        tcpdump \
        iftop \
        htop \
        iotop \
        lsof \
        strace \
        ltrace \
        binutils \
        psmisc \
        fuser \
        lshw \
        hwinfo \
        dmidecode \
        pciutils \
        usbutils \
        acl \
        attr \
        cryptsetup \
        gnupg2 \
        openssl
    
    # Initialize AIDE database
    info "Initializing AIDE database..."
    aide --init
    mv /var/lib/aide/aide.db.new /var/lib/aide/aide.db
    
    # Configure rkhunter
    rkhunter --update
    rkhunter --propupd
    
    success "Security tools installed successfully"
}

# Create security monitoring service
create_monitoring_service() {
    info "Creating security monitoring service..."
    
    # Create monitoring script
    cat > /usr/local/bin/qbitel-security-monitor << 'EOF'
#!/bin/bash

# QBITEL Security Monitoring Script

LOG_FILE="/var/log/qbitel/security-monitor.log"
ALERT_FILE="/var/log/qbitel/security-alerts.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"
}

alert() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALERT: $*" >> "$ALERT_FILE"
    logger -p user.crit "QBITEL-SECURITY: $*"
}

# Check for failed login attempts
check_auth_failures() {
    local failures=$(grep "authentication failure" /var/log/auth.log | grep "$(date '+%b %d')" | wc -l)
    if [ "$failures" -gt 10 ]; then
        alert "High number of authentication failures detected: $failures"
    fi
}

# Check system integrity
check_system_integrity() {
    if [ -f "/var/lib/aide/aide.db" ]; then
        aide --check > /tmp/aide-check.log 2>&1
        if [ $? -ne 0 ]; then
            alert "System integrity check failed - files may have been modified"
        fi
    fi
}

# Check for suspicious processes
check_processes() {
    local suspicious_processes=(
        "nc"
        "netcat"
        "ncat"
        "socat"
        "wireshark"
        "tcpdump"
    )
    
    for process in "${suspicious_processes[@]}"; do
        if pgrep -f "$process" > /dev/null; then
            alert "Suspicious process detected: $process"
        fi
    done
}

# Check disk usage
check_disk_usage() {
    local usage=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ "$usage" -gt 90 ]; then
        alert "Root filesystem usage is at $usage%"
    fi
}

# Check network connections
check_network() {
    local established_connections=$(netstat -an | grep ESTABLISHED | wc -l)
    if [ "$established_connections" -gt 1000 ]; then
        alert "High number of established connections: $established_connections"
    fi
}

# Main monitoring loop
log "Security monitoring started"

check_auth_failures
check_system_integrity
check_processes
check_disk_usage
check_network

log "Security monitoring completed"
EOF

    chmod +x /usr/local/bin/qbitel-security-monitor
    
    # Create systemd service
    cat > /etc/systemd/system/qbitel-security-monitor.service << 'EOF'
[Unit]
Description=QBITEL Security Monitor
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/qbitel-security-monitor
User=root
StandardOutput=journal
StandardError=journal
EOF

    # Create systemd timer
    cat > /etc/systemd/system/qbitel-security-monitor.timer << 'EOF'
[Unit]
Description=Run QBITEL Security Monitor every 5 minutes
Requires=qbitel-security-monitor.service

[Timer]
OnCalendar=*:0/5
Persistent=true

[Install]
WantedBy=timers.target
EOF

    # Enable and start the timer
    systemctl daemon-reload
    systemctl enable qbitel-security-monitor.timer
    systemctl start qbitel-security-monitor.timer
    
    success "Security monitoring service created and started successfully"
}

# Generate security report
generate_security_report() {
    info "Generating security hardening report..."
    
    local report_file="/var/log/qbitel/security-hardening-report-$(date +%Y%m%d-%H%M%S).txt"
    
    cat > "$report_file" << EOF
QBITEL Security Hardening Report
Generated: $(date)
Hostname: $(hostname)
Kernel Version: $(uname -r)
OS Version: $(lsb_release -d | cut -f2)

============================================
HARDENING MEASURES APPLIED:
============================================

1. Kernel Security Parameters:
   - Network security parameters configured
   - Memory protection enabled
   - File system security hardened
   - Process security enhanced

2. Firewall Configuration:
   - Default deny policy implemented
   - Service-specific rules configured
   - Rate limiting enabled
   - Logging configured

3. SSH Hardening:
   - Root login disabled
   - Password authentication disabled
   - Strong ciphers and MACs configured
   - Connection limits applied

4. Audit Configuration:
   - Comprehensive audit rules configured
   - Log retention set to 7 years
   - System call monitoring enabled
   - File access monitoring configured

5. Intrusion Prevention:
   - Fail2ban configured and enabled
   - Custom filters for QBITEL services
   - Ban policies configured

6. Logging and Monitoring:
   - Centralized logging configured
   - Log rotation policies applied
   - Security monitoring service deployed
   - Real-time alerting enabled

7. System Tools:
   - Security scanning tools installed
   - File integrity monitoring enabled
   - Rootkit detection configured
   - System hardening assessment tools deployed

8. File Permissions:
   - Secure permissions applied
   - Certificate security configured
   - Log file permissions secured
   - Configuration file protection enabled

============================================
VERIFICATION COMMANDS:
============================================

Check firewall status: iptables -L -n
Check SSH configuration: sshd -T
Check audit status: systemctl status auditd
Check fail2ban status: fail2ban-client status
Check security monitoring: systemctl status qbitel-security-monitor.timer
Run security scan: lynis audit system

============================================
NEXT STEPS:
============================================

1. Run vulnerability assessment
2. Perform penetration testing
3. Implement continuous security monitoring
4. Schedule regular security audits
5. Update security policies and procedures

Report saved to: $report_file
EOF

    success "Security hardening report generated: $report_file"
}

# Main execution
main() {
    info "Starting QBITEL security hardening..."
    
    check_root
    setup_directories
    harden_kernel
    configure_firewall
    harden_ssh
    configure_audit
    configure_fail2ban
    configure_logging
    secure_permissions
    install_security_tools
    create_monitoring_service
    generate_security_report
    
    success "QBITEL security hardening completed successfully!"
    info "Please reboot the system to ensure all changes take effect."
    info "Review the security report in /var/log/qbitel/ for additional recommendations."
}

# Execute main function
main "$@"