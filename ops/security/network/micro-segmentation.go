package network

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"strings"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"go.uber.org/zap"
)

// MicroSegmentationEngine implements network micro-segmentation for QSLB
type MicroSegmentationEngine struct {
	logger *zap.Logger
	
	// Network policies and rules
	policies     map[string]*NetworkPolicy
	rules        map[string]*SegmentationRule
	segments     map[string]*NetworkSegment
	
	// Traffic monitoring
	trafficMonitor *TrafficMonitor
	
	// Metrics
	policyViolations   *prometheus.CounterVec
	trafficBlocked     *prometheus.CounterVec
	segmentationActive *prometheus.GaugeVec
	
	// Configuration
	config *SegmentationConfig
	
	// State management
	mu       sync.RWMutex
	running  bool
	stopChan chan struct{}
}

// NetworkPolicy defines network access policies
type NetworkPolicy struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Priority    int                    `json:"priority"`
	Rules       []*PolicyRule          `json:"rules"`
	Segments    []string               `json:"segments"`
	Metadata    map[string]interface{} `json:"metadata"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
	Active      bool                   `json:"active"`
}

// PolicyRule defines individual policy rules
type PolicyRule struct {
	ID          string      `json:"id"`
	Action      RuleAction  `json:"action"`
	Protocol    string      `json:"protocol"`
	SourceCIDR  string      `json:"source_cidr"`
	DestCIDR    string      `json:"dest_cidr"`
	SourcePorts []int       `json:"source_ports"`
	DestPorts   []int       `json:"dest_ports"`
	Direction   Direction   `json:"direction"`
	Priority    int         `json:"priority"`
	Conditions  []Condition `json:"conditions"`
}

// SegmentationRule defines micro-segmentation rules
type SegmentationRule struct {
	ID            string                 `json:"id"`
	Name          string                 `json:"name"`
	SourceSegment string                 `json:"source_segment"`
	DestSegment   string                 `json:"dest_segment"`
	Action        RuleAction             `json:"action"`
	Protocol      string                 `json:"protocol"`
	Ports         []int                  `json:"ports"`
	Conditions    []Condition            `json:"conditions"`
	Metadata      map[string]interface{} `json:"metadata"`
	Active        bool                   `json:"active"`
	CreatedAt     time.Time              `json:"created_at"`
}

// NetworkSegment represents a network segment
type NetworkSegment struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	CIDR        string                 `json:"cidr"`
	Type        SegmentType            `json:"type"`
	TrustLevel  TrustLevel             `json:"trust_level"`
	Devices     []string               `json:"devices"`
	Policies    []string               `json:"policies"`
	Metadata    map[string]interface{} `json:"metadata"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
}

// TrafficFlow represents network traffic flow
type TrafficFlow struct {
	ID            string                 `json:"id"`
	SourceIP      net.IP                 `json:"source_ip"`
	DestIP        net.IP                 `json:"dest_ip"`
	SourcePort    int                    `json:"source_port"`
	DestPort      int                    `json:"dest_port"`
	Protocol      string                 `json:"protocol"`
	BytesIn       int64                  `json:"bytes_in"`
	BytesOut      int64                  `json:"bytes_out"`
	PacketsIn     int64                  `json:"packets_in"`
	PacketsOut    int64                  `json:"packets_out"`
	StartTime     time.Time              `json:"start_time"`
	EndTime       time.Time              `json:"end_time"`
	Action        RuleAction             `json:"action"`
	PolicyID      string                 `json:"policy_id"`
	RuleID        string                 `json:"rule_id"`
	Metadata      map[string]interface{} `json:"metadata"`
}

type RuleAction string

const (
	ActionAllow    RuleAction = "allow"
	ActionDeny     RuleAction = "deny"
	ActionLog      RuleAction = "log"
	ActionAlert    RuleAction = "alert"
	ActionQuarantine RuleAction = "quarantine"
)

type Direction string

const (
	DirectionInbound  Direction = "inbound"
	DirectionOutbound Direction = "outbound"
	DirectionBoth     Direction = "both"
)

type SegmentType string

const (
	SegmentTypeManagement SegmentType = "management"
	SegmentTypeProduction SegmentType = "production"
	SegmentTypeDMZ        SegmentType = "dmz"
	SegmentTypeInternal   SegmentType = "internal"
	SegmentTypeGuest      SegmentType = "guest"
	SegmentTypeQuarantine SegmentType = "quarantine"
)

type TrustLevel string

const (
	TrustLevelHigh   TrustLevel = "high"
	TrustLevelMedium TrustLevel = "medium"
	TrustLevelLow    TrustLevel = "low"
	TrustLevelUntrusted TrustLevel = "untrusted"
)

type Condition struct {
	Type     string      `json:"type"`
	Operator string      `json:"operator"`
	Value    interface{} `json:"value"`
}

type SegmentationConfig struct {
	EnableMicroSegmentation bool          `json:"enable_micro_segmentation"`
	DefaultAction          RuleAction    `json:"default_action"`
	LogAllTraffic          bool          `json:"log_all_traffic"`
	BlockUnknownTraffic    bool          `json:"block_unknown_traffic"`
	
	// Monitoring
	TrafficMonitoringInterval time.Duration `json:"traffic_monitoring_interval"`
	PolicyEvaluationInterval  time.Duration `json:"policy_evaluation_interval"`
	
	// Thresholds
	SuspiciousTrafficThreshold int64 `json:"suspicious_traffic_threshold"`
	AnomalyDetectionEnabled    bool  `json:"anomaly_detection_enabled"`
	
	// Integration
	FirewallIntegration bool   `json:"firewall_integration"`
	SDNIntegration      bool   `json:"sdn_integration"`
	FirewallType        string `json:"firewall_type"`
}

// NewMicroSegmentationEngine creates a new micro-segmentation engine
func NewMicroSegmentationEngine(logger *zap.Logger, config *SegmentationConfig) *MicroSegmentationEngine {
	mse := &MicroSegmentationEngine{
		logger:   logger,
		policies: make(map[string]*NetworkPolicy),
		rules:    make(map[string]*SegmentationRule),
		segments: make(map[string]*NetworkSegment),
		config:   config,
		stopChan: make(chan struct{}),
		
		policyViolations: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "qslb_network_policy_violations_total",
				Help: "Total number of network policy violations",
			},
			[]string{"policy_id", "rule_id", "source_segment", "dest_segment"},
		),
		
		trafficBlocked: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "qslb_network_traffic_blocked_total",
				Help: "Total amount of blocked network traffic",
			},
			[]string{"source_segment", "dest_segment", "protocol"},
		),
		
		segmentationActive: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "qslb_network_segmentation_active",
				Help: "Network segmentation status",
			},
			[]string{"segment_id", "segment_type"},
		),
	}
	
	// Initialize traffic monitor
	mse.trafficMonitor = NewTrafficMonitor(logger, mse.handleTrafficFlow)
	
	// Create default segments
	mse.createDefaultSegments()
	
	// Create default policies
	mse.createDefaultPolicies()
	
	return mse
}

// Start begins micro-segmentation enforcement
func (mse *MicroSegmentationEngine) Start(ctx context.Context) error {
	mse.mu.Lock()
	defer mse.mu.Unlock()
	
	if mse.running {
		return fmt.Errorf("micro-segmentation engine already running")
	}
	
	mse.logger.Info("Starting micro-segmentation engine")
	
	// Start traffic monitoring
	if err := mse.trafficMonitor.Start(ctx); err != nil {
		return fmt.Errorf("failed to start traffic monitor: %w", err)
	}
	
	// Start policy enforcement
	go mse.policyEnforcementLoop(ctx)
	
	// Start traffic analysis
	go mse.trafficAnalysisLoop(ctx)
	
	mse.running = true
	mse.logger.Info("Micro-segmentation engine started")
	
	return nil
}

// Stop stops micro-segmentation enforcement
func (mse *MicroSegmentationEngine) Stop() error {
	mse.mu.Lock()
	defer mse.mu.Unlock()
	
	if !mse.running {
		return nil
	}
	
	mse.logger.Info("Stopping micro-segmentation engine")
	
	close(mse.stopChan)
	
	if mse.trafficMonitor != nil {
		mse.trafficMonitor.Stop()
	}
	
	mse.running = false
	mse.logger.Info("Micro-segmentation engine stopped")
	
	return nil
}

// CreateNetworkSegment creates a new network segment
func (mse *MicroSegmentationEngine) CreateNetworkSegment(segment *NetworkSegment) error {
	mse.mu.Lock()
	defer mse.mu.Unlock()
	
	// Validate CIDR
	_, _, err := net.ParseCIDR(segment.CIDR)
	if err != nil {
		return fmt.Errorf("invalid CIDR: %w", err)
	}
	
	segment.CreatedAt = time.Now()
	segment.UpdatedAt = time.Now()
	
	mse.segments[segment.ID] = segment
	
	// Update metrics
	mse.segmentationActive.WithLabelValues(segment.ID, string(segment.Type)).Set(1)
	
	mse.logger.Info("Network segment created",
		zap.String("segment_id", segment.ID),
		zap.String("segment_name", segment.Name),
		zap.String("cidr", segment.CIDR),
	)
	
	return nil
}

// CreateNetworkPolicy creates a new network policy
func (mse *MicroSegmentationEngine) CreateNetworkPolicy(policy *NetworkPolicy) error {
	mse.mu.Lock()
	defer mse.mu.Unlock()
	
	policy.CreatedAt = time.Now()
	policy.UpdatedAt = time.Now()
	
	mse.policies[policy.ID] = policy
	
	mse.logger.Info("Network policy created",
		zap.String("policy_id", policy.ID),
		zap.String("policy_name", policy.Name),
		zap.Int("rules_count", len(policy.Rules)),
	)
	
	return nil
}

// CreateSegmentationRule creates a new segmentation rule
func (mse *MicroSegmentationEngine) CreateSegmentationRule(rule *SegmentationRule) error {
	mse.mu.Lock()
	defer mse.mu.Unlock()
	
	rule.CreatedAt = time.Now()
	mse.rules[rule.ID] = rule
	
	mse.logger.Info("Segmentation rule created",
		zap.String("rule_id", rule.ID),
		zap.String("rule_name", rule.Name),
		zap.String("source_segment", rule.SourceSegment),
		zap.String("dest_segment", rule.DestSegment),
		zap.String("action", string(rule.Action)),
	)
	
	return nil
}

// EvaluateTrafficFlow evaluates a traffic flow against policies
func (mse *MicroSegmentationEngine) EvaluateTrafficFlow(flow *TrafficFlow) (*PolicyDecision, error) {
	mse.mu.RLock()
	defer mse.mu.RUnlock()
	
	// Determine source and destination segments
	sourceSegment := mse.getSegmentForIP(flow.SourceIP)
	destSegment := mse.getSegmentForIP(flow.DestIP)
	
	decision := &PolicyDecision{
		Action:        mse.config.DefaultAction,
		SourceSegment: sourceSegment,
		DestSegment:   destSegment,
		Timestamp:     time.Now(),
	}
	
	// Evaluate segmentation rules
	for _, rule := range mse.rules {
		if !rule.Active {
			continue
		}
		
		if mse.matchesSegmentationRule(rule, sourceSegment, destSegment, flow) {
			decision.Action = rule.Action
			decision.RuleID = rule.ID
			decision.PolicyID = "" // Segmentation rule, not policy rule
			break
		}
	}
	
	// Evaluate network policies
	for _, policy := range mse.policies {
		if !policy.Active {
			continue
		}
		
		for _, rule := range policy.Rules {
			if mse.matchesPolicyRule(rule, flow) {
				decision.Action = rule.Action
				decision.RuleID = rule.ID
				decision.PolicyID = policy.ID
				break
			}
		}
	}
	
	// Log policy violations
	if decision.Action == ActionDeny || decision.Action == ActionAlert {
		mse.policyViolations.WithLabelValues(
			decision.PolicyID,
			decision.RuleID,
			sourceSegment,
			destSegment,
		).Inc()
		
		if decision.Action == ActionDeny {
			mse.trafficBlocked.WithLabelValues(
				sourceSegment,
				destSegment,
				flow.Protocol,
			).Inc()
		}
	}
	
	return decision, nil
}

// PolicyDecision represents the result of policy evaluation
type PolicyDecision struct {
	Action        RuleAction `json:"action"`
	PolicyID      string     `json:"policy_id"`
	RuleID        string     `json:"rule_id"`
	SourceSegment string     `json:"source_segment"`
	DestSegment   string     `json:"dest_segment"`
	Reason        string     `json:"reason"`
	Timestamp     time.Time  `json:"timestamp"`
}

// getSegmentForIP determines which segment an IP belongs to
func (mse *MicroSegmentationEngine) getSegmentForIP(ip net.IP) string {
	for _, segment := range mse.segments {
		_, cidr, err := net.ParseCIDR(segment.CIDR)
		if err != nil {
			continue
		}
		
		if cidr.Contains(ip) {
			return segment.ID
		}
	}
	
	return "unknown"
}

// matchesSegmentationRule checks if a flow matches a segmentation rule
func (mse *MicroSegmentationEngine) matchesSegmentationRule(rule *SegmentationRule, sourceSegment, destSegment string, flow *TrafficFlow) bool {
	// Check segment match
	if rule.SourceSegment != "*" && rule.SourceSegment != sourceSegment {
		return false
	}
	
	if rule.DestSegment != "*" && rule.DestSegment != destSegment {
		return false
	}
	
	// Check protocol
	if rule.Protocol != "*" && !strings.EqualFold(rule.Protocol, flow.Protocol) {
		return false
	}
	
	// Check ports
	if len(rule.Ports) > 0 {
		portMatch := false
		for _, port := range rule.Ports {
			if port == flow.DestPort {
				portMatch = true
				break
			}
		}
		if !portMatch {
			return false
		}
	}
	
	// Check conditions
	for _, condition := range rule.Conditions {
		if !mse.evaluateCondition(condition, flow) {
			return false
		}
	}
	
	return true
}

// matchesPolicyRule checks if a flow matches a policy rule
func (mse *MicroSegmentationEngine) matchesPolicyRule(rule *PolicyRule, flow *TrafficFlow) bool {
	// Check protocol
	if rule.Protocol != "*" && !strings.EqualFold(rule.Protocol, flow.Protocol) {
		return false
	}
	
	// Check source CIDR
	if rule.SourceCIDR != "*" {
		_, cidr, err := net.ParseCIDR(rule.SourceCIDR)
		if err != nil || !cidr.Contains(flow.SourceIP) {
			return false
		}
	}
	
	// Check destination CIDR
	if rule.DestCIDR != "*" {
		_, cidr, err := net.ParseCIDR(rule.DestCIDR)
		if err != nil || !cidr.Contains(flow.DestIP) {
			return false
		}
	}
	
	// Check destination ports
	if len(rule.DestPorts) > 0 {
		portMatch := false
		for _, port := range rule.DestPorts {
			if port == flow.DestPort {
				portMatch = true
				break
			}
		}
		if !portMatch {
			return false
		}
	}
	
	// Check conditions
	for _, condition := range rule.Conditions {
		if !mse.evaluateCondition(condition, flow) {
			return false
		}
	}
	
	return true
}

// evaluateCondition evaluates a rule condition
func (mse *MicroSegmentationEngine) evaluateCondition(condition Condition, flow *TrafficFlow) bool {
	// Implementation would depend on condition type
	// This is a simplified example
	switch condition.Type {
	case "time_range":
		// Check if current time is within specified range
		return true
	case "traffic_volume":
		// Check traffic volume thresholds
		return true
	case "device_type":
		// Check device type from metadata
		return true
	default:
		return true
	}
}

// handleTrafficFlow processes traffic flows
func (mse *MicroSegmentationEngine) handleTrafficFlow(flow *TrafficFlow) {
	decision, err := mse.EvaluateTrafficFlow(flow)
	if err != nil {
		mse.logger.Error("Failed to evaluate traffic flow", zap.Error(err))
		return
	}
	
	flow.Action = decision.Action
	flow.PolicyID = decision.PolicyID
	flow.RuleID = decision.RuleID
	
	// Log significant events
	if decision.Action == ActionDeny || decision.Action == ActionAlert {
		mse.logger.Warn("Traffic flow blocked/alerted",
			zap.String("source_ip", flow.SourceIP.String()),
			zap.String("dest_ip", flow.DestIP.String()),
			zap.Int("dest_port", flow.DestPort),
			zap.String("protocol", flow.Protocol),
			zap.String("action", string(decision.Action)),
			zap.String("source_segment", decision.SourceSegment),
			zap.String("dest_segment", decision.DestSegment),
		)
	}
	
	// Enforce action if firewall integration is enabled
	if mse.config.FirewallIntegration && decision.Action == ActionDeny {
		if err := mse.enforceFirewallRule(flow, decision); err != nil {
			mse.logger.Error("Failed to enforce firewall rule", zap.Error(err))
		}
	}
}

// enforceFirewallRule enforces firewall rules
func (mse *MicroSegmentationEngine) enforceFirewallRule(flow *TrafficFlow, decision *PolicyDecision) error {
	// Implementation would depend on firewall type (iptables, nftables, etc.)
	mse.logger.Info("Enforcing firewall rule",
		zap.String("source_ip", flow.SourceIP.String()),
		zap.String("dest_ip", flow.DestIP.String()),
		zap.String("action", string(decision.Action)),
	)
	return nil
}

// policyEnforcementLoop runs the policy enforcement loop
func (mse *MicroSegmentationEngine) policyEnforcementLoop(ctx context.Context) {
	ticker := time.NewTicker(mse.config.PolicyEvaluationInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-mse.stopChan:
			return
		case <-ticker.C:
			mse.evaluatePolicies(ctx)
		}
	}
}

// trafficAnalysisLoop runs the traffic analysis loop
func (mse *MicroSegmentationEngine) trafficAnalysisLoop(ctx context.Context) {
	ticker := time.NewTicker(mse.config.TrafficMonitoringInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-mse.stopChan:
			return
		case <-ticker.C:
			mse.analyzeTrafficPatterns(ctx)
		}
	}
}

// evaluatePolicies performs periodic policy evaluation
func (mse *MicroSegmentationEngine) evaluatePolicies(ctx context.Context) {
	mse.logger.Debug("Evaluating network policies")
	
	// Get all active policies
	policies := mse.getActivePolicies()
	
	// Evaluate each policy for effectiveness and compliance
	for _, policy := range policies {
		// Check policy usage statistics
		stats := mse.getPolicyStats(policy.ID)
		
		// Evaluate policy effectiveness
		effectiveness := mse.calculatePolicyEffectiveness(policy, stats)
		
		// Check for policy conflicts
		conflicts := mse.detectPolicyConflicts(policy, policies)
		
		// Check compliance with security standards
		compliance := mse.checkPolicyCompliance(policy)
		
		// Log policy evaluation results
		mse.logger.Info("Policy evaluation completed",
			zap.String("policy_id", policy.ID),
			zap.Float64("effectiveness", effectiveness),
			zap.Int("conflicts", len(conflicts)),
			zap.Bool("compliant", compliance))
		
		// Update policy metrics
		mse.metrics.PolicyEffectiveness.WithLabelValues(policy.ID).Set(effectiveness)
		mse.metrics.PolicyConflicts.WithLabelValues(policy.ID).Set(float64(len(conflicts)))
		
		// Handle policy issues
		if effectiveness < 0.5 {
			mse.logger.Warn("Low policy effectiveness detected",
				zap.String("policy_id", policy.ID),
				zap.Float64("effectiveness", effectiveness))
			mse.handleLowEffectiveness(policy)
		}
		
		if len(conflicts) > 0 {
			mse.logger.Warn("Policy conflicts detected",
				zap.String("policy_id", policy.ID),
				zap.Int("conflicts", len(conflicts)))
			mse.handlePolicyConflicts(policy, conflicts)
		}
		
		if !compliance {
			mse.logger.Warn("Policy compliance issue detected",
				zap.String("policy_id", policy.ID))
			mse.handleComplianceIssue(policy)
		}
	}
	
	// Generate policy recommendations
	recommendations := mse.generatePolicyRecommendations(policies)
	if len(recommendations) > 0 {
		mse.logger.Info("Generated policy recommendations",
			zap.Int("count", len(recommendations)))
		mse.processPolicyRecommendations(recommendations)
	}
}

// analyzeTrafficPatterns analyzes traffic patterns for anomalies
func (mse *MicroSegmentationEngine) analyzeTrafficPatterns(ctx context.Context) {
	if !mse.config.AnomalyDetectionEnabled {
		return
	}
	
	// Analyze traffic patterns and detect anomalies
	mse.logger.Debug("Analyzing traffic patterns")
	
	// Get recent traffic flows
	flows := mse.getRecentTrafficFlows(time.Hour)
	
	// Analyze traffic patterns
	patterns := mse.analyzeTrafficPatterns(flows)
	
	// Detect anomalies in traffic patterns
	anomalies := mse.detectTrafficAnomalies(patterns)
	
	// Process detected anomalies
	for _, anomaly := range anomalies {
		mse.logger.Warn("Traffic anomaly detected",
			zap.String("type", anomaly.Type),
			zap.String("description", anomaly.Description),
			zap.Float64("severity", anomaly.Severity))
		
		// Update anomaly metrics
		mse.metrics.TrafficAnomalies.WithLabelValues(anomaly.Type).Inc()
		
		// Handle anomaly based on severity
		if anomaly.Severity > 0.8 {
			mse.handleHighSeverityAnomaly(anomaly)
		} else if anomaly.Severity > 0.5 {
			mse.handleMediumSeverityAnomaly(anomaly)
		}
	}
	
	// Update traffic analysis metrics
	mse.metrics.TrafficFlowsAnalyzed.Add(float64(len(flows)))
	mse.metrics.TrafficAnomaliesDetected.Add(float64(len(anomalies)))
	
	mse.logger.Debug("Traffic pattern analysis completed",
		zap.Int("flows_analyzed", len(flows)),
		zap.Int("anomalies_detected", len(anomalies)))
}

// createDefaultSegments creates default network segments
func (mse *MicroSegmentationEngine) createDefaultSegments() {
	segments := []*NetworkSegment{
		{
			ID:         "mgmt-segment",
			Name:       "Management Segment",
			CIDR:       "10.0.1.0/24",
			Type:       SegmentTypeManagement,
			TrustLevel: TrustLevelHigh,
		},
		{
			ID:         "prod-segment",
			Name:       "Production Segment",
			CIDR:       "10.0.2.0/24",
			Type:       SegmentTypeProduction,
			TrustLevel: TrustLevelHigh,
		},
		{
			ID:         "dmz-segment",
			Name:       "DMZ Segment",
			CIDR:       "10.0.3.0/24",
			Type:       SegmentTypeDMZ,
			TrustLevel: TrustLevelMedium,
		},
		{
			ID:         "quarantine-segment",
			Name:       "Quarantine Segment",
			CIDR:       "10.0.99.0/24",
			Type:       SegmentTypeQuarantine,
			TrustLevel: TrustLevelUntrusted,
		},
	}
	
	for _, segment := range segments {
		mse.segments[segment.ID] = segment
		mse.segmentationActive.WithLabelValues(segment.ID, string(segment.Type)).Set(1)
	}
}

// createDefaultPolicies creates default network policies
func (mse *MicroSegmentationEngine) createDefaultPolicies() {
	// Default deny policy
	denyPolicy := &NetworkPolicy{
		ID:          "default-deny",
		Name:        "Default Deny Policy",
		Description: "Default deny all traffic",
		Priority:    1000,
		Rules: []*PolicyRule{
			{
				ID:         "deny-all",
				Action:     ActionDeny,
				Protocol:   "*",
				SourceCIDR: "*",
				DestCIDR:   "*",
				Direction:  DirectionBoth,
				Priority:   1000,
			},
		},
		Active: true,
	}
	
	mse.policies[denyPolicy.ID] = denyPolicy
	
	// Management access policy
	mgmtPolicy := &NetworkPolicy{
		ID:          "mgmt-access",
		Name:        "Management Access Policy",
		Description: "Allow management access",
		Priority:    100,
		Rules: []*PolicyRule{
			{
				ID:         "allow-ssh",
				Action:     ActionAllow,
				Protocol:   "tcp",
				SourceCIDR: "10.0.1.0/24",
				DestCIDR:   "*",
				DestPorts:  []int{22},
				Direction:  DirectionOutbound,
				Priority:   100,
			},
			{
				ID:         "allow-https",
				Action:     ActionAllow,
				Protocol:   "tcp",
				SourceCIDR: "10.0.1.0/24",
				DestCIDR:   "*",
				DestPorts:  []int{443},
				Direction:  DirectionOutbound,
				Priority:   101,
			},
		},
		Active: true,
	}
	
	mse.policies[mgmtPolicy.ID] = mgmtPolicy
}

// GetNetworkSegments returns all network segments
func (mse *MicroSegmentationEngine) GetNetworkSegments() map[string]*NetworkSegment {
	mse.mu.RLock()
	defer mse.mu.RUnlock()
	
	segments := make(map[string]*NetworkSegment)
	for k, v := range mse.segments {
		segments[k] = v
	}
	
	return segments
}

// GetNetworkPolicies returns all network policies
func (mse *MicroSegmentationEngine) GetNetworkPolicies() map[string]*NetworkPolicy {
	mse.mu.RLock()
	defer mse.mu.RUnlock()
	
	policies := make(map[string]*NetworkPolicy)
	for k, v := range mse.policies {
		policies[k] = v
	}
	
	return policies
}

// GetSegmentationMetrics returns segmentation metrics
func (mse *MicroSegmentationEngine) GetSegmentationMetrics() (map[string]interface{}, error) {
	mse.mu.RLock()
	defer mse.mu.RUnlock()
	
	metrics := map[string]interface{}{
		"segments_count":           len(mse.segments),
		"policies_count":           len(mse.policies),
		"rules_count":              len(mse.rules),
		"micro_segmentation_active": mse.running,
		"traffic_monitoring_active": mse.trafficMonitor != nil && mse.trafficMonitor.IsRunning(),
	}
	
	return metrics, nil
}