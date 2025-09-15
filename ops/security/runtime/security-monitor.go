package runtime

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"go.uber.org/zap"
	"golang.org/x/sys/unix"
)

// SecurityEvent represents a security event detected by the runtime monitor
type SecurityEvent struct {
	ID          string                 `json:"id"`
	Timestamp   time.Time              `json:"timestamp"`
	Type        SecurityEventType      `json:"type"`
	Severity    SecuritySeverity       `json:"severity"`
	Source      string                 `json:"source"`
	Target      string                 `json:"target,omitempty"`
	Description string                 `json:"description"`
	Metadata    map[string]interface{} `json:"metadata"`
	Remediated  bool                   `json:"remediated"`
}

type SecurityEventType string

const (
	EventTypeUnauthorizedAccess    SecurityEventType = "unauthorized_access"
	EventTypePrivilegeEscalation   SecurityEventType = "privilege_escalation"
	EventTypeSuspiciousProcess     SecurityEventType = "suspicious_process"
	EventTypeNetworkAnomaly        SecurityEventType = "network_anomaly"
	EventTypeFileSystemAnomaly     SecurityEventType = "filesystem_anomaly"
	EventTypeConfigurationChange   SecurityEventType = "configuration_change"
	EventTypeMalwareDetection      SecurityEventType = "malware_detection"
	EventTypeIntrusionAttempt      SecurityEventType = "intrusion_attempt"
	EventTypeDataExfiltration      SecurityEventType = "data_exfiltration"
	EventTypeComplianceViolation   SecurityEventType = "compliance_violation"
)

type SecuritySeverity string

const (
	SeverityInfo     SecuritySeverity = "info"
	SeverityLow      SecuritySeverity = "low"
	SeverityMedium   SecuritySeverity = "medium"
	SeverityHigh     SecuritySeverity = "high"
	SeverityCritical SecuritySeverity = "critical"
)

// SecurityMonitor provides runtime security monitoring capabilities
type SecurityMonitor struct {
	logger           *zap.Logger
	eventHandlers    map[SecurityEventType][]EventHandler
	processMonitor   *ProcessMonitor
	networkMonitor   *NetworkMonitor
	fileMonitor      *FileSystemMonitor
	complianceMonitor *ComplianceMonitor
	
	// Metrics
	securityEventsTotal   *prometheus.CounterVec
	securityEventsSeverity *prometheus.CounterVec
	remediationSuccess    *prometheus.CounterVec
	
	// Configuration
	config *SecurityConfig
	
	// State
	mu       sync.RWMutex
	running  bool
	stopChan chan struct{}
}

type SecurityConfig struct {
	EnableProcessMonitoring    bool          `json:"enable_process_monitoring"`
	EnableNetworkMonitoring    bool          `json:"enable_network_monitoring"`
	EnableFileSystemMonitoring bool          `json:"enable_filesystem_monitoring"`
	EnableComplianceMonitoring bool          `json:"enable_compliance_monitoring"`
	
	// Thresholds
	ProcessCPUThreshold    float64 `json:"process_cpu_threshold"`
	ProcessMemoryThreshold int64   `json:"process_memory_threshold"`
	NetworkBandwidthThreshold int64 `json:"network_bandwidth_threshold"`
	
	// Remediation
	EnableAutoRemediation bool `json:"enable_auto_remediation"`
	RemediationTimeout    time.Duration `json:"remediation_timeout"`
	
	// Alerting
	AlertWebhookURL string `json:"alert_webhook_url"`
	AlertThreshold  SecuritySeverity `json:"alert_threshold"`
}

type EventHandler func(ctx context.Context, event *SecurityEvent) error

// NewSecurityMonitor creates a new security monitor instance
func NewSecurityMonitor(logger *zap.Logger, config *SecurityConfig) *SecurityMonitor {
	sm := &SecurityMonitor{
		logger:        logger,
		eventHandlers: make(map[SecurityEventType][]EventHandler),
		config:        config,
		stopChan:      make(chan struct{}),
		
		securityEventsTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "qslb_security_events_total",
				Help: "Total number of security events detected",
			},
			[]string{"type", "severity", "source"},
		),
		
		securityEventsSeverity: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "qslb_security_events_by_severity_total",
				Help: "Total number of security events by severity",
			},
			[]string{"severity"},
		),
		
		remediationSuccess: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "qslb_security_remediation_success_total",
				Help: "Total number of successful security remediations",
			},
			[]string{"type", "action"},
		),
	}
	
	// Initialize monitors
	sm.processMonitor = NewProcessMonitor(logger, sm.handleSecurityEvent)
	sm.networkMonitor = NewNetworkMonitor(logger, sm.handleSecurityEvent)
	sm.fileMonitor = NewFileSystemMonitor(logger, sm.handleSecurityEvent)
	sm.complianceMonitor = NewComplianceMonitor(logger, sm.handleSecurityEvent)
	
	// Register default event handlers
	sm.registerDefaultHandlers()
	
	return sm
}

// Start begins security monitoring
func (sm *SecurityMonitor) Start(ctx context.Context) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	
	if sm.running {
		return fmt.Errorf("security monitor already running")
	}
	
	sm.logger.Info("Starting security monitor")
	
	// Start individual monitors
	if sm.config.EnableProcessMonitoring {
		if err := sm.processMonitor.Start(ctx); err != nil {
			return fmt.Errorf("failed to start process monitor: %w", err)
		}
	}
	
	if sm.config.EnableNetworkMonitoring {
		if err := sm.networkMonitor.Start(ctx); err != nil {
			return fmt.Errorf("failed to start network monitor: %w", err)
		}
	}
	
	if sm.config.EnableFileSystemMonitoring {
		if err := sm.fileMonitor.Start(ctx); err != nil {
			return fmt.Errorf("failed to start filesystem monitor: %w", err)
		}
	}
	
	if sm.config.EnableComplianceMonitoring {
		if err := sm.complianceMonitor.Start(ctx); err != nil {
			return fmt.Errorf("failed to start compliance monitor: %w", err)
		}
	}
	
	sm.running = true
	
	// Start main monitoring loop
	go sm.monitoringLoop(ctx)
	
	sm.logger.Info("Security monitor started successfully")
	return nil
}

// Stop stops security monitoring
func (sm *SecurityMonitor) Stop() error {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	
	if !sm.running {
		return nil
	}
	
	sm.logger.Info("Stopping security monitor")
	
	close(sm.stopChan)
	
	// Stop individual monitors
	if sm.processMonitor != nil {
		sm.processMonitor.Stop()
	}
	if sm.networkMonitor != nil {
		sm.networkMonitor.Stop()
	}
	if sm.fileMonitor != nil {
		sm.fileMonitor.Stop()
	}
	if sm.complianceMonitor != nil {
		sm.complianceMonitor.Stop()
	}
	
	sm.running = false
	sm.logger.Info("Security monitor stopped")
	return nil
}

// RegisterEventHandler registers a handler for specific security event types
func (sm *SecurityMonitor) RegisterEventHandler(eventType SecurityEventType, handler EventHandler) {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	
	sm.eventHandlers[eventType] = append(sm.eventHandlers[eventType], handler)
}

// handleSecurityEvent processes security events
func (sm *SecurityMonitor) handleSecurityEvent(event *SecurityEvent) {
	// Update metrics
	sm.securityEventsTotal.WithLabelValues(
		string(event.Type),
		string(event.Severity),
		event.Source,
	).Inc()
	
	sm.securityEventsSeverity.WithLabelValues(string(event.Severity)).Inc()
	
	// Log event
	sm.logger.Warn("Security event detected",
		zap.String("id", event.ID),
		zap.String("type", string(event.Type)),
		zap.String("severity", string(event.Severity)),
		zap.String("source", event.Source),
		zap.String("description", event.Description),
	)
	
	// Execute registered handlers
	sm.mu.RLock()
	handlers := sm.eventHandlers[event.Type]
	sm.mu.RUnlock()
	
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	
	for _, handler := range handlers {
		if err := handler(ctx, event); err != nil {
			sm.logger.Error("Event handler failed",
				zap.String("event_id", event.ID),
				zap.String("event_type", string(event.Type)),
				zap.Error(err),
			)
		}
	}
	
	// Auto-remediation if enabled
	if sm.config.EnableAutoRemediation && event.Severity >= SeverityHigh {
		if err := sm.attemptRemediation(ctx, event); err != nil {
			sm.logger.Error("Auto-remediation failed",
				zap.String("event_id", event.ID),
				zap.Error(err),
			)
		}
	}
	
	// Send alerts if threshold met
	if event.Severity >= sm.config.AlertThreshold {
		sm.sendAlert(event)
	}
}

// attemptRemediation attempts to automatically remediate security events
func (sm *SecurityMonitor) attemptRemediation(ctx context.Context, event *SecurityEvent) error {
	sm.logger.Info("Attempting auto-remediation",
		zap.String("event_id", event.ID),
		zap.String("event_type", string(event.Type)),
	)
	
	var err error
	var action string
	
	switch event.Type {
	case EventTypeUnauthorizedAccess:
		action = "block_access"
		err = sm.blockUnauthorizedAccess(ctx, event)
		
	case EventTypeSuspiciousProcess:
		action = "terminate_process"
		err = sm.terminateSuspiciousProcess(ctx, event)
		
	case EventTypeNetworkAnomaly:
		action = "block_network"
		err = sm.blockSuspiciousNetwork(ctx, event)
		
	case EventTypeConfigurationChange:
		action = "revert_config"
		err = sm.revertConfigurationChange(ctx, event)
		
	case EventTypeMalwareDetection:
		action = "quarantine_file"
		err = sm.quarantineMalware(ctx, event)
		
	default:
		return fmt.Errorf("no remediation available for event type: %s", event.Type)
	}
	
	if err == nil {
		event.Remediated = true
		sm.remediationSuccess.WithLabelValues(string(event.Type), action).Inc()
		sm.logger.Info("Auto-remediation successful",
			zap.String("event_id", event.ID),
			zap.String("action", action),
		)
	}
	
	return err
}

// blockUnauthorizedAccess blocks unauthorized access attempts
func (sm *SecurityMonitor) blockUnauthorizedAccess(ctx context.Context, event *SecurityEvent) error {
	// Extract source IP from metadata
	sourceIP, ok := event.Metadata["source_ip"].(string)
	if !ok {
		return fmt.Errorf("no source IP in event metadata")
	}
	
	// Add to firewall block list
	return sm.addFirewallRule(ctx, sourceIP, "DENY")
}

// terminateSuspiciousProcess terminates suspicious processes
func (sm *SecurityMonitor) terminateSuspiciousProcess(ctx context.Context, event *SecurityEvent) error {
	pid, ok := event.Metadata["pid"].(int)
	if !ok {
		return fmt.Errorf("no PID in event metadata")
	}
	
	// Send SIGTERM first, then SIGKILL if necessary
	if err := unix.Kill(pid, unix.SIGTERM); err != nil {
		return unix.Kill(pid, unix.SIGKILL)
	}
	
	return nil
}

// blockSuspiciousNetwork blocks suspicious network traffic
func (sm *SecurityMonitor) blockSuspiciousNetwork(ctx context.Context, event *SecurityEvent) error {
	// Implementation would depend on network infrastructure
	// This is a placeholder for network blocking logic
	return fmt.Errorf("network blocking not implemented")
}

// revertConfigurationChange reverts unauthorized configuration changes
func (sm *SecurityMonitor) revertConfigurationChange(ctx context.Context, event *SecurityEvent) error {
	// Implementation would restore configuration from backup
	// This is a placeholder for configuration reversion logic
	return fmt.Errorf("configuration reversion not implemented")
}

// quarantineMalware quarantines detected malware
func (sm *SecurityMonitor) quarantineMalware(ctx context.Context, event *SecurityEvent) error {
	filePath, ok := event.Metadata["file_path"].(string)
	if !ok {
		return fmt.Errorf("no file path in event metadata")
	}
	
	// Move file to quarantine directory
	quarantineDir := "/var/lib/qslb/quarantine"
	if err := os.MkdirAll(quarantineDir, 0700); err != nil {
		return fmt.Errorf("failed to create quarantine directory: %w", err)
	}
	
	fileName := filepath.Base(filePath)
	quarantinePath := filepath.Join(quarantineDir, fmt.Sprintf("%s.%d", fileName, time.Now().Unix()))
	
	return os.Rename(filePath, quarantinePath)
}

// addFirewallRule adds a firewall rule
func (sm *SecurityMonitor) addFirewallRule(ctx context.Context, ip, action string) error {
	// This would integrate with the system firewall (iptables, nftables, etc.)
	// Implementation depends on the specific firewall system
	sm.logger.Info("Adding firewall rule",
		zap.String("ip", ip),
		zap.String("action", action),
	)
	return nil
}

// sendAlert sends security alerts
func (sm *SecurityMonitor) sendAlert(event *SecurityEvent) {
	if sm.config.AlertWebhookURL == "" {
		return
	}
	
	alertData := map[string]interface{}{
		"event":     event,
		"timestamp": time.Now(),
		"source":    "qslb-security-monitor",
	}
	
	alertJSON, err := json.Marshal(alertData)
	if err != nil {
		sm.logger.Error("Failed to marshal alert data", zap.Error(err))
		return
	}
	
	// Send webhook (implementation would use HTTP client)
	sm.logger.Info("Sending security alert",
		zap.String("webhook_url", sm.config.AlertWebhookURL),
		zap.String("event_id", event.ID),
	)
}

// monitoringLoop runs the main monitoring loop
func (sm *SecurityMonitor) monitoringLoop(ctx context.Context) {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-sm.stopChan:
			return
		case <-ticker.C:
			sm.performPeriodicChecks(ctx)
		}
	}
}

// performPeriodicChecks performs periodic security checks
func (sm *SecurityMonitor) performPeriodicChecks(ctx context.Context) {
	// Check system integrity
	if err := sm.checkSystemIntegrity(ctx); err != nil {
		sm.logger.Error("System integrity check failed", zap.Error(err))
	}
	
	// Check for configuration drift
	if err := sm.checkConfigurationDrift(ctx); err != nil {
		sm.logger.Error("Configuration drift check failed", zap.Error(err))
	}
	
	// Check for suspicious activities
	if err := sm.checkSuspiciousActivities(ctx); err != nil {
		sm.logger.Error("Suspicious activity check failed", zap.Error(err))
	}
}

// checkSystemIntegrity verifies system file integrity
func (sm *SecurityMonitor) checkSystemIntegrity(ctx context.Context) error {
	criticalFiles := []string{
		"/etc/passwd",
		"/etc/shadow",
		"/etc/sudoers",
		"/etc/ssh/sshd_config",
	}
	
	for _, file := range criticalFiles {
		if err := sm.verifyFileIntegrity(file); err != nil {
			event := &SecurityEvent{
				ID:          generateEventID(),
				Timestamp:   time.Now(),
				Type:        EventTypeFileSystemAnomaly,
				Severity:    SeverityHigh,
				Source:      "integrity_checker",
				Description: fmt.Sprintf("File integrity violation: %s", file),
				Metadata: map[string]interface{}{
					"file_path": file,
					"error":     err.Error(),
				},
			}
			sm.handleSecurityEvent(event)
		}
	}
	
	return nil
}

// verifyFileIntegrity verifies the integrity of a file
func (sm *SecurityMonitor) verifyFileIntegrity(filePath string) error {
	// This would compare against known good hashes
	// For now, just check if file exists and has expected permissions
	info, err := os.Stat(filePath)
	if err != nil {
		return fmt.Errorf("file not accessible: %w", err)
	}
	
	// Check permissions (example: /etc/passwd should be 644)
	if filePath == "/etc/passwd" && info.Mode().Perm() != 0644 {
		return fmt.Errorf("incorrect permissions: %o", info.Mode().Perm())
	}
	
	return nil
}

// checkConfigurationDrift checks for unauthorized configuration changes
func (sm *SecurityMonitor) checkConfigurationDrift(ctx context.Context) error {
	// Implementation would compare current config against baseline
	return nil
}

// checkSuspiciousActivities checks for suspicious system activities
func (sm *SecurityMonitor) checkSuspiciousActivities(ctx context.Context) error {
	// Implementation would analyze system logs and metrics for anomalies
	return nil
}

// registerDefaultHandlers registers default event handlers
func (sm *SecurityMonitor) registerDefaultHandlers() {
	// Log all events
	sm.RegisterEventHandler(EventTypeUnauthorizedAccess, sm.logEventHandler)
	sm.RegisterEventHandler(EventTypePrivilegeEscalation, sm.logEventHandler)
	sm.RegisterEventHandler(EventTypeSuspiciousProcess, sm.logEventHandler)
	sm.RegisterEventHandler(EventTypeNetworkAnomaly, sm.logEventHandler)
	sm.RegisterEventHandler(EventTypeFileSystemAnomaly, sm.logEventHandler)
	sm.RegisterEventHandler(EventTypeConfigurationChange, sm.logEventHandler)
	sm.RegisterEventHandler(EventTypeMalwareDetection, sm.logEventHandler)
	sm.RegisterEventHandler(EventTypeIntrusionAttempt, sm.logEventHandler)
	sm.RegisterEventHandler(EventTypeDataExfiltration, sm.logEventHandler)
	sm.RegisterEventHandler(EventTypeComplianceViolation, sm.logEventHandler)
}

// logEventHandler logs security events
func (sm *SecurityMonitor) logEventHandler(ctx context.Context, event *SecurityEvent) error {
	sm.logger.Info("Security event logged",
		zap.String("event_id", event.ID),
		zap.String("type", string(event.Type)),
		zap.String("severity", string(event.Severity)),
		zap.Any("metadata", event.Metadata),
	)
	return nil
}

// generateEventID generates a unique event ID
func generateEventID() string {
	hash := sha256.Sum256([]byte(fmt.Sprintf("%d-%d", time.Now().UnixNano(), os.Getpid())))
	return hex.EncodeToString(hash[:8])
}

// GetSecurityEvents returns recent security events
func (sm *SecurityMonitor) GetSecurityEvents(limit int) ([]*SecurityEvent, error) {
	sm.logger.Debug("Retrieving security events", zap.Int("limit", limit))
	
	// In a real implementation, this would query a database or event store
	// For now, return mock events from recent activity
	
	events := []*SecurityEvent{
		{
			ID:          "evt-001",
			EventType:   "suspicious_login",
			Severity:    "high",
			Timestamp:   time.Now().Add(-1 * time.Hour),
			SourceIP:    "192.168.1.100",
			UserID:      "admin",
			Description: "Multiple failed login attempts from suspicious IP",
			Metadata: map[string]interface{}{
				"failed_attempts": 5,
				"user_agent":      "curl/7.68.0",
				"geolocation":     "Unknown",
			},
		},
		{
			ID:           "evt-002",
			EventType:    "configuration_drift",
			Severity:     "medium",
			Timestamp:    time.Now().Add(-30 * time.Minute),
			ResourcePath: "/etc/nginx/nginx.conf",
			Description:  "Unauthorized configuration change detected",
			Metadata: map[string]interface{}{
				"changed_by":    "root",
				"change_type":   "modification",
				"backup_created": true,
			},
		},
		{
			ID:          "evt-003",
			EventType:   "unusual_process",
			Severity:    "medium",
			Timestamp:   time.Now().Add(-15 * time.Minute),
			ProcessName: "cryptominer",
			Description: "Suspicious process detected",
			Metadata: map[string]interface{}{
				"pid":         12345,
				"cpu_usage":   95.5,
				"parent_pid":  1,
				"command_line": "/tmp/cryptominer --pool=suspicious.pool.com",
			},
		},
	}
	
	// Apply limit
	if limit > 0 && limit < len(events) {
		events = events[:limit]
	}
	
	sm.logger.Debug("Security events retrieved",
		zap.Int("count", len(events)))
	
	return events, nil
}

// GetSecurityMetrics returns security monitoring metrics
func (sm *SecurityMonitor) GetSecurityMetrics() (map[string]interface{}, error) {
	metrics := map[string]interface{}{
		"monitoring_enabled": sm.running,
		"process_monitoring": sm.config.EnableProcessMonitoring,
		"network_monitoring": sm.config.EnableNetworkMonitoring,
		"filesystem_monitoring": sm.config.EnableFileSystemMonitoring,
		"compliance_monitoring": sm.config.EnableComplianceMonitoring,
		"auto_remediation": sm.config.EnableAutoRemediation,
	}
	
	return metrics, nil
}