
package performance

import (
	"context"
	"fmt"
	"math/rand"
	"net"
	"os"
	"os/exec"
	"sync"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"go.uber.org/zap"
)

// ChaosEngine implements chaos engineering experiments for CronosAI
type ChaosEngine struct {
	logger *zap.Logger
	
	// Experiment management
	experiments map[string]*ChaosExperiment
	
	// Metrics collection
	metrics *ChaosMetrics
	
	// Configuration
	config *ChaosConfig
	
	// State management
	mu       sync.RWMutex
	running  bool
	stopChan chan struct{}
}

// ChaosConfig defines chaos engineering configuration
type ChaosConfig struct {
	// Experiment settings
	EnableChaosEngineering bool          `json:"enable_chaos_engineering"`
	ExperimentInterval     time.Duration `json:"experiment_interval"`
	MaxConcurrentExperiments int         `json:"max_concurrent_experiments"`
	
	// Safety settings
	SafetyChecks        bool          `json:"safety_checks"`
	MaxImpactDuration   time.Duration `json:"max_impact_duration"`
	RecoveryTimeout     time.Duration `json:"recovery_timeout"`
	HealthCheckInterval time.Duration `json:"health_check_interval"`
	
	// Blast radius control
	MaxAffectedServices int     `json:"max_affected_services"`
	MaxAffectedNodes    int     `json:"max_affected_nodes"`
	ImpactPercentage    float64 `json:"impact_percentage"`
	
	// Experiment types
	EnableNetworkChaos    bool `json:"enable_network_chaos"`
	EnableResourceChaos   bool `json:"enable_resource_chaos"`
	EnableServiceChaos    bool `json:"enable_service_chaos"`
	EnableInfrastructureChaos bool `json:"enable_infrastructure_chaos"`
}

// ChaosExperiment defines a chaos engineering experiment
type ChaosExperiment struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Type        ExperimentType         `json:"type"`
	Category    ExperimentCategory     `json:"category"`
	Severity    ExperimentSeverity     `json:"severity"`
	
	// Targeting
	TargetServices []string           `json:"target_services"`
	TargetNodes    []string           `json:"target_nodes"`
	TargetFilters  map[string]string  `json:"target_filters"`
	
	// Execution parameters
	Duration       time.Duration      `json:"duration"`
	Parameters     map[string]interface{} `json:"parameters"`
	
	// Safety and rollback
	SafetyChecks   []SafetyCheck      `json:"safety_checks"`
	RollbackSteps  []RollbackStep     `json:"rollback_steps"`
	
	// Monitoring
	SuccessCriteria []SuccessCriterion `json:"success_criteria"`
	Metrics        []string           `json:"metrics"`
	
	// Execution state
	Status         ExperimentStatus   `json:"status"`
	StartTime      time.Time          `json:"start_time"`
	EndTime        time.Time          `json:"end_time"`
	Results        *ExperimentResults `json:"results"`
	
	// Metadata
	CreatedAt      time.Time          `json:"created_at"`
	CreatedBy      string             `json:"created_by"`
	Tags           []string           `json:"tags"`
}

type ExperimentType string

const (
	ExperimentTypeNetworkLatency     ExperimentType = "network_latency"
	ExperimentTypeNetworkPartition   ExperimentType = "network_partition"
	ExperimentTypeNetworkLoss        ExperimentType = "network_loss"
	ExperimentTypeCPUStress          ExperimentType = "cpu_stress"
	ExperimentTypeMemoryStress       ExperimentType = "memory_stress"
	ExperimentTypeDiskStress         ExperimentType = "disk_stress"
	ExperimentTypeServiceKill        ExperimentType = "service_kill"
	ExperimentTypeServiceRestart     ExperimentType = "service_restart"
	ExperimentTypeNodeFailure        ExperimentType = "node_failure"
	ExperimentTypeContainerKill      ExperimentType = "container_kill"
	ExperimentTypeDatabaseFailure    ExperimentType = "database_failure"
	ExperimentTypeClockSkew          ExperimentType = "clock_skew"
)

type ExperimentCategory string

const (
	CategoryNetwork        ExperimentCategory = "network"
	CategoryResource       ExperimentCategory = "resource"
	CategoryService        ExperimentCategory = "service"
	CategoryInfrastructure ExperimentCategory = "infrastructure"
	CategoryApplication    ExperimentCategory = "application"
)

type ExperimentSeverity string

const (
	SeverityLow      ExperimentSeverity = "low"
	SeverityMedium   ExperimentSeverity = "medium"
	SeverityHigh     ExperimentSeverity = "high"
	SeverityCritical ExperimentSeverity = "critical"
)

type ExperimentStatus string

const (
	StatusPending    ExperimentStatus = "pending"
	StatusRunning    ExperimentStatus = "running"
	StatusCompleted  ExperimentStatus = "completed"
	StatusFailed     ExperimentStatus = "failed"
	StatusAborted    ExperimentStatus = "aborted"
	StatusRolledBack ExperimentStatus = "rolled_back"
)

// SafetyCheck defines a safety check for experiments
type SafetyCheck struct {
	Name        string                 `json:"name"`
	Type        string                 `json:"type"`
	Parameters  map[string]interface{} `json:"parameters"`
	Threshold   float64                `json:"threshold"`
	Operator    string                 `json:"operator"`
	Description string                 `json:"description"`
}

// RollbackStep defines a rollback step
type RollbackStep struct {
	Name        string                 `json:"name"`
	Type        string                 `json:"type"`
	Parameters  map[string]interface{} `json:"parameters"`
	Order       int                    `json:"order"`
	Description string                 `json:"description"`
}

// SuccessCriterion defines success criteria for experiments
type SuccessCriterion struct {
	Name        string  `json:"name"`
	Metric      string  `json:"metric"`
	Threshold   float64 `json:"threshold"`
	Operator    string  `json:"operator"`
	Description string  `json:"description"`
}

// ExperimentResults contains experiment execution results
type ExperimentResults struct {
	Success           bool                   `json:"success"`
	ErrorMessage      string                 `json:"error_message,omitempty"`
	ImpactMeasured    bool                   `json:"impact_measured"`
	RecoveryTime      time.Duration          `json:"recovery_time"`
	MetricsCollected  map[string]interface{} `json:"metrics_collected"`
	SafetyViolations  []string               `json:"safety_violations"`
	Observations      []string               `json:"observations"`
	Recommendations   []string               `json:"recommendations"`
}

// ChaosMetrics contains chaos engineering metrics
type ChaosMetrics struct {
	ExperimentsTotal     *prometheus.CounterVec
	ExperimentDuration   *prometheus.HistogramVec
	ExperimentSuccess    *prometheus.CounterVec
	SafetyViolations     *prometheus.CounterVec
	RecoveryTime         *prometheus.HistogramVec
	SystemResilience     *prometheus.GaugeVec
}

// NewChaosEngine creates a new chaos engineering engine
func NewChaosEngine(logger *zap.Logger, config *ChaosConfig) *ChaosEngine {
	ce := &ChaosEngine{
		logger:      logger,
		config:      config,
		experiments: make(map[string]*ChaosExperiment),
		stopChan:    make(chan struct{}),
	}
	
	// Initialize metrics
	ce.initializeMetrics()
	
	// Load default experiments
	ce.loadDefaultExperiments()
	
	return ce
}

// initializeMetrics initializes Prometheus metrics
func (ce *ChaosEngine) initializeMetrics() {
	ce.metrics = &ChaosMetrics{
		ExperimentsTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "cronosai_chaos_experiments_total",
				Help: "Total number of chaos experiments executed",
			},
			[]string{"type", "category", "severity", "status"},
		),
		
		ExperimentDuration: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "cronosai_chaos_experiment_duration_seconds",
				Help:    "Duration of chaos experiments in seconds",
				Buckets: prometheus.ExponentialBuckets(1, 2, 10),
			},
			[]string{"type", "category"},
		),
		
		ExperimentSuccess: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "cronosai_chaos_experiment_success_total",
				Help: "Total number of successful chaos experiments",
			},
			[]string{"type", "category"},
		),
		
		SafetyViolations: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "cronosai_chaos_safety_violations_total",
				Help: "Total number of safety violations during chaos experiments",
			},
			[]string{"experiment_id", "violation_type"},
		),
		
		RecoveryTime: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "cronosai_chaos_recovery_time_seconds",
				Help:    "Time taken for system recovery after chaos experiments",
				Buckets: prometheus.ExponentialBuckets(1, 2, 10),
			},
			[]string{"type", "category"},
		),
		
		SystemResilience: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "cronosai_system_resilience_score",
				Help: "System resilience score based on chaos experiments",
			},
			[]string{"component", "metric"},
		),
	}
}

// Start begins chaos engineering experiments
func (ce *ChaosEngine) Start(ctx context.Context) error {
	ce.mu.Lock()
	defer ce.mu.Unlock()
	
	if ce.running {
		return fmt.Errorf("chaos engine already running")
	}
	
	if !ce.config.EnableChaosEngineering {
		ce.logger.Info("Chaos engineering is disabled")
		return nil
	}
	
	ce.logger.Info("Starting chaos engineering engine")
	ce.running = true
	
	// Start experiment scheduler
	go ce.experimentScheduler(ctx)
	
	// Start safety monitor
	go ce.safetyMonitor(ctx)
	
	ce.logger.Info("Chaos engineering engine started")
	return nil
}

// Stop stops chaos engineering experiments
func (ce *ChaosEngine) Stop() error {
	ce.mu.Lock()
	defer ce.mu.Unlock()
	
	if !ce.running {
		return nil
	}
	
	ce.logger.Info("Stopping chaos engineering engine")
	
	close(ce.stopChan)
	
	// Abort running experiments
	for _, experiment := range ce.experiments {
		if experiment.Status == StatusRunning {
			ce.abortExperiment(experiment.ID)
		}
	}
	
	ce.running = false
	ce.logger.Info("Chaos engineering engine stopped")
	
	return nil
}

// RunExperiment executes a specific chaos experiment
func (ce *ChaosEngine) RunExperiment(ctx context.Context, experimentID string) error {
	ce.mu.RLock()
	experiment, exists := ce.experiments[experimentID]
	ce.mu.RUnlock()
	
	if !exists {
		return fmt.Errorf("experiment not found: %s", experimentID)
	}
	
	if experiment.Status == StatusRunning {
		return fmt.Errorf("experiment already running: %s", experimentID)
	}
	
	ce.logger.Info("Starting chaos experiment",
		zap.String("experiment_id", experiment.ID),
		zap.String("experiment_name", experiment.Name),
		zap.String("type", string(experiment.Type)),
	)
	
	// Update experiment status
	experiment.Status = StatusRunning
	experiment.StartTime = time.Now()
	experiment.Results = &ExperimentResults{}
	
	// Record metrics
	ce.metrics.ExperimentsTotal.WithLabelValues(
		string(experiment.Type),
		string(experiment.Category),
		string(experiment.Severity),
		string(experiment.Status),
	).Inc()
	
	// Execute experiment with timeout
	experimentCtx, cancel := context.WithTimeout(ctx, experiment.Duration+ce.config.RecoveryTimeout)
	defer cancel()
	
	// Run safety checks before execution
	if ce.config.SafetyChecks {
		if err := ce.runSafetyChecks(experiment); err != nil {
			experiment.Status = StatusAborted
			experiment.Results.ErrorMessage = fmt.Sprintf("Safety check failed: %v", err)
			ce.logger.Error("Experiment aborted due to safety check failure",
				zap.String("experiment_id", experiment.ID),
				zap.Error(err),
			)
			return err
		}
	}
	
	// Execute the experiment
	err := ce.executeExperiment(experimentCtx, experiment)
	
	// Update final status and metrics
	experiment.EndTime = time.Now()
	duration := experiment.EndTime.Sub(experiment.StartTime)
	
	if err != nil {
		experiment.Status = StatusFailed
		experiment.Results.ErrorMessage = err.Error()
		ce.logger.Error("Chaos experiment failed",
			zap.String("experiment_id", experiment.ID),
			zap.Error(err),
		)
	} else {
		experiment.Status = StatusCompleted
		experiment.Results.Success = true
		ce.metrics.ExperimentSuccess.WithLabelValues(
			string(experiment.Type),
			string(experiment.Category),
		).Inc()
		ce.logger.Info("Chaos experiment completed successfully",
			zap.String("experiment_id", experiment.ID),
			zap.Duration("duration", duration),
		)
	}
	
	ce.metrics.ExperimentDuration.WithLabelValues(
		string(experiment.Type),
		string(experiment.Category),
	).Observe(duration.Seconds())
	
	return err
}

// executeExperiment executes the actual chaos experiment
func (ce *ChaosEngine) executeExperiment(ctx context.Context, experiment *ChaosExperiment) error {
	switch experiment.Type {
	case ExperimentTypeNetworkLatency:
		return ce.executeNetworkLatency(ctx, experiment)
	case ExperimentTypeNetworkPartition:
		return ce.executeNetworkPartition(ctx, experiment)
	case ExperimentTypeNetworkLoss:
		return ce.executeNetworkLoss(ctx, experiment)
	case ExperimentTypeCPUStress:
		return ce.executeCPUStress(ctx, experiment)
	case ExperimentTypeMemoryStress:
		return ce.executeMemoryStress(ctx, experiment)
	case ExperimentTypeDiskStress:
		return ce.executeDiskStress(ctx, experiment)
	case ExperimentTypeServiceKill:
		return ce.executeServiceKill(ctx, experiment)
	case ExperimentTypeServiceRestart:
		return ce.executeServiceRestart(ctx, experiment)
	case ExperimentTypeNodeFailure:
		return ce.executeNodeFailure(ctx, experiment)
	case ExperimentTypeContainerKill:
		return ce.executeContainerKill(ctx, experiment)
	case ExperimentTypeClockSkew:
		return ce.executeClockSkew(ctx, experiment)
	default:
		return fmt.Errorf("unsupported experiment type: %s", experiment.Type)
	}
}

// executeNetworkLatency introduces network latency
func (ce *ChaosEngine) executeNetworkLatency(ctx context.Context, experiment *ChaosExperiment) error {
	latency, ok := experiment.Parameters["latency"].(string)
	if !ok {
		latency = "100ms"
	}
	
	targetInterface, ok := experiment.Parameters["interface"].(string)
	if !ok {
		targetInterface = "eth0"
	}
	
	ce.logger.Info("Introducing network latency",
		zap.String("latency", latency),
		zap.String("interface", targetInterface),
	)
	
	// Add network latency using tc (traffic control)
	cmd := exec.CommandContext(ctx, "tc", "qdisc", "add", "dev", targetInterface, "root", "netem", "delay", latency)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to add network latency: %w", err)
	}
	
	// Wait for experiment duration
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(experiment.Duration):
	}
	
	// Remove network latency
	cleanupCmd := exec.Command("tc", "qdisc", "del", "dev", targetInterface, "root")
	if err := cleanupCmd.Run(); err != nil {
		ce.logger.Error("Failed to remove network latency", zap.Error(err))
	}
	
	return nil
}

// executeNetworkPartition creates network partition
func (ce *ChaosEngine) executeNetworkPartition(ctx context.Context, experiment *ChaosExperiment) error {
	targetIPs, ok := experiment.Parameters["target_ips"].([]string)
	if !ok {
		return fmt.Errorf("target_ips parameter required for network partition")
	}
	
	ce.logger.Info("Creating network partition", zap.Strings("target_ips", targetIPs))
	
	// Block traffic to target IPs using iptables
	var rules []string
	for _, ip := range targetIPs {
		rule := fmt.Sprintf("-A OUTPUT -d %s -j DROP", ip)
		rules = append(rules, rule)
		
		cmd := exec.CommandContext(ctx, "iptables", "-A", "OUTPUT", "-d", ip, "-j", "DROP")
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("failed to create network partition for %s: %w", ip, err)
		}
	}
	
	// Wait for experiment duration
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(experiment.Duration):
	}
	
	// Remove iptables rules
	for _, ip := range targetIPs {
		cleanupCmd := exec.Command("iptables", "-D", "OUTPUT", "-d", ip, "-j", "DROP")
		if err := cleanupCmd.Run(); err != nil {
			ce.logger.Error("Failed to remove network partition rule", zap.String("ip", ip), zap.Error(err))
		}
	}
	
	return nil
}

// executeNetworkLoss introduces packet loss
func (ce *ChaosEngine) executeNetworkLoss(ctx context.Context, experiment *ChaosExperiment) error {
	lossRate, ok := experiment.Parameters["loss_rate"].(string)
	if !ok {
		lossRate = "1%"
	}
	
	targetInterface, ok := experiment.Parameters["interface"].(string)
	if !ok {
		targetInterface = "eth0"
	}
	
	ce.logger.Info("Introducing packet loss",
		zap.String("loss_rate", lossRate),
		zap.String("interface", targetInterface),
	)
	
	// Add packet loss using tc
	cmd := exec.CommandContext(ctx, "tc", "qdisc", "add", "dev", targetInterface, "root", "netem", "loss", lossRate)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to add packet loss: %w", err)
	}
	
	// Wait for experiment duration
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(experiment.Duration):
	}
	
	// Remove packet loss
	cleanupCmd := exec.Command("tc", "qdisc", "del", "dev", targetInterface, "root")
	if err := cleanupCmd.Run(); err != nil {
		ce.logger.Error("Failed to remove packet loss", zap.Error(err))
	}
	
	return nil
}

// executeCPUStress creates CPU stress
func (ce *ChaosEngine) executeCPUStress(ctx context.Context, experiment *ChaosExperiment) error {
	workers, ok := experiment.Parameters["workers"].(int)
	if !ok {
		workers = 1
	}
	
	ce.logger.Info("Starting CPU stress test", zap.Int("workers", workers))
	
	// Start CPU stress using stress-ng or similar tool
	cmd := exec.CommandContext(ctx, "stress-ng", "--cpu", fmt.Sprintf("%d", workers), "--timeout", experiment.Duration.String())
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("CPU stress test failed: %w", err)
	}
	
	return nil
}

// executeMemoryStress creates memory stress
func (ce *ChaosEngine) executeMemoryStress(ctx context.Context, experiment *ChaosExperiment) error {
	memorySize, ok := experiment.Parameters["memory_size"].(string)
	if !ok {
		memorySize = "1G"
	}
	
	ce.logger.Info("Starting memory stress test", zap.String("memory_size", memorySize))
	
	// Start memory stress
	cmd := exec.CommandContext(ctx, "stress-ng", "--vm", "1", "--vm-bytes", memorySize, "--timeout", experiment.Duration.String())
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("memory stress test failed: %w", err)
	}
	
	return nil
}

// executeDiskStress creates disk I/O stress
func (ce *ChaosEngine) executeDiskStress(ctx context.Context, experiment *ChaosExperiment) error {
	workers, ok := experiment.Parameters["workers"].(int)
	if !ok {
		workers = 1
	}
	
	ce.logger.Info("Starting disk stress test", zap.Int("workers", workers))
	
	// Start disk I/O stress
	cmd := exec.CommandContext(ctx, "stress-ng", "--io", fmt.Sprintf("%d", workers), "--timeout", experiment.Duration.String())
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("disk stress test failed: %w", err)
	}
	
	return nil
}

// executeServiceKill kills a service process
func (ce *ChaosEngine) executeServiceKill(ctx context.Context, experiment *ChaosExperiment) error {
	serviceName, ok := experiment.Parameters["service_name"].(string)
	if !ok {
		return fmt.Errorf("service_name parameter required for service kill")
	}
	
	ce.logger.Info("Killing service", zap.String("service_name", serviceName))
	
	// Kill the service
	cmd := exec.CommandContext(ctx, "pkill", "-f", serviceName)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to kill service %s: %w", serviceName, err)
	}
	
	// Wait for experiment duration (service should restart automatically)
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(experiment.Duration):
	}
	
	return nil
}

// executeServiceRestart restarts a service
func (ce *ChaosEngine) executeServiceRestart(ctx context.Context, experiment *ChaosExperiment) error {
	serviceName, ok := experiment.Parameters["service_name"].(string)
	if !ok {
		return fmt.Errorf("service_name parameter required for service restart")
	}
	
	ce.logger.Info("Restarting service", zap.String("service_name", serviceName))
	
	// Restart the service
	cmd := exec.CommandContext(ctx, "systemctl", "restart", serviceName)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to restart service %s: %w", serviceName, err)
	}
	
	return nil
}

// executeNodeFailure simulates node failure
func (ce *ChaosEngine) executeNodeFailure(ctx context.Context, experiment *ChaosExperiment) error {
	// This is a placeholder - actual implementation would depend on infrastructure
	ce.logger.Info("Simulating node failure")
	
	// In a real implementation, this might:
	// - Shutdown network interfaces
	// - Stop critical services
	// - Simulate hardware failures
	
	time.Sleep(experiment.Duration)
	return nil
}

// executeContainerKill kills containers
func (ce *ChaosEngine) executeContainerKill(ctx context.Context, experiment *ChaosExperiment) error {
	containerName, ok := experiment.Parameters["container_name"].(string)
	if !ok {
		return fmt.Errorf("container_name parameter required for container kill")
	}
	
	ce.logger.Info("Killing container", zap.String("container_name", containerName))
	
	// Kill the container
	cmd := exec.CommandContext(ctx, "docker", "kill", containerName)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to kill container %s: %w", containerName, err)
	}
	
	return nil
}

// executeClockSkew introduces clock skew
func (ce *ChaosEngine) executeClockSkew(ctx context.Context, experiment *ChaosExperiment) error {
	skewAmount, ok := experiment.Parameters["skew_amount"].(string)
	if !ok {
		skewAmount = "+5min"
	}
	
	ce.logger.Info("Introducing clock skew", zap.String("skew_amount", skewAmount))
	
	// Get current time
	originalTime := time.Now()
	
	// Set skewed time
	cmd := exec.CommandContext(ctx, "date", "-s", skewAmount)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to set clock skew: %w", err)
	}
	
	// Wait for experiment duration
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(experiment.Duration):
	}
	
	// Restore original time
	restoreCmd := exec.Command("date", "-s", originalTime.Format("2006-01-02 15:04:05"))
	if err := restoreCmd.Run(); err != nil {
		ce.logger.Error("Failed to restore original time", zap.Error(err))
	}
	
	return nil
}

// runSafetyChecks runs safety checks before experiment execution
func (ce *ChaosEngine) runSafetyChecks(experiment *ChaosExperiment) error {
	for _, check := range experiment.SafetyChecks {
		if err := ce.executeSafetyCheck(check); err != nil {
			return fmt.Errorf("safety check '%s' failed: %w", check.Name, err)
		}
	}
	return nil
}

// executeSafetyCheck executes a single safety check
func (ce *ChaosEngine) executeSafetyCheck(check SafetyCheck) error {
	// Implementation would depend on check type
	// This is a simplified example
	switch check.Type {
	case "cpu_usage":
		// Check CPU usage is below threshold
		return nil
	case "memory_usage":
		// Check memory usage is below threshold
		return nil
	case "service_health":
		// Check service health
		return nil
	default:
		return fmt.Errorf("unknown safety check type: %s", check.Type)
	}
}

// abortExperiment aborts a running experiment
func (ce *ChaosEngine) abortExperiment(experimentID string) error {
	ce.mu.Lock()
	defer ce.mu.Unlock()
	
	experiment, exists := ce.experiments[experimentID]
	if !exists {
		return fmt.Errorf("experiment not found: %s", experimentID)
	}
	
	if experiment.Status != StatusRunning {
		return fmt.Errorf("experiment not running: %s", experimentID)
	}
	
	ce.logger.Info("Aborting chaos experiment", zap.String("experiment_id", experimentID))
	
	// Execute rollback steps
	for _, step := range experiment.RollbackSteps {
		if err := ce.executeRollbackStep(step); err != nil {
			ce.logger.Error("Rollback step failed",
				zap.String("experiment_id", experimentID),
				zap.String("step_name", step.Name),
				zap.Error(err),
			)
		}
	}
	
	experiment.Status = StatusAborted
	experiment.EndTime = time.Now()
	
	return nil
}

// executeRollbackStep executes a rollback step
func (ce *ChaosEngine) executeRollbackStep(step RollbackStep) error {
	ce.logger.Info("Executing rollback step",
		zap.String("step_name", step.Name),
		zap.String("step_type", step.Type))
	
	switch step.Type {
	case "restore_service":
		return ce.restoreService(step)
	case "restore_network":
		return ce.restoreNetwork(step)
	case "restore_resources":
		return ce.restoreResources(step)
	case "restore_configuration":
		return ce.restoreConfiguration(step)
	case "cleanup_chaos":
		return ce.cleanupChaos(step)
	default:
		return fmt.Errorf("unknown rollback step type: %s", step.Type)
	}
}

// Safety check implementations

func (ce *ChaosEngine) checkSystemHealth(check SafetyCheck) error {
	ce.logger.Debug("Checking system health")
	
	// Check CPU usage
	cpuUsage, err := ce.getCPUUsage()
	if err != nil {
		return fmt.Errorf("failed to get CPU usage: %w", err)
	}
	
	if cpuUsage > check.Thresholds.MaxCPUUsage {
		return fmt.Errorf("CPU usage too high: %.2f%% > %.2f%%", cpuUsage, check.Thresholds.MaxCPUUsage)
	}
	
	// Check memory usage
	memUsage, err := ce.getMemoryUsage()
	if err != nil {
		return fmt.Errorf("failed to get memory usage: %w", err)
	}
	
	if memUsage > check.Thresholds.MaxMemoryUsage {
		return fmt.Errorf("memory usage too high: %.2f%% > %.2f%%", memUsage, check.Thresholds.MaxMemoryUsage)
	}
	
	// Check disk usage
	diskUsage, err := ce.getDiskUsage()
	if err != nil {
		return fmt.Errorf("failed to get disk usage: %w", err)
	}
	
	if diskUsage > check.Thresholds.MaxDiskUsage {
		return fmt.Errorf("disk usage too high: %.2f%% > %.2f%%", diskUsage, check.Thresholds.MaxDiskUsage)
	}
	
	ce.logger.Info("System health check passed",
		zap.Float64("cpu_usage", cpuUsage),
		zap.Float64("memory_usage", memUsage),
		zap.Float64("disk_usage", diskUsage))
	
	return nil
}

func (ce *ChaosEngine) checkResourceAvailability(check SafetyCheck) error {
	ce.logger.Debug("Checking resource availability")
	
	// Check available memory
	availableMemory, err := ce.getAvailableMemory()
	if err != nil {
		return fmt.Errorf("failed to get available memory: %w", err)
	}
	
	if availableMemory < check.Thresholds.MinAvailableMemory {
		return fmt.Errorf("insufficient available memory: %d MB < %d MB",
			availableMemory, check.Thresholds.MinAvailableMemory)
	}
	
	// Check available disk space
	availableDisk, err := ce.getAvailableDiskSpace()
	if err != nil {
		return fmt.Errorf("failed to get available disk space: %w", err)
	}
	
	if availableDisk < check.Thresholds.MinAvailableDisk {
		return fmt.Errorf("insufficient available disk space: %d GB < %d GB",
			availableDisk, check.Thresholds.MinAvailableDisk)
	}
	
	ce.logger.Info("Resource availability check passed",
		zap.Int64("available_memory_mb", availableMemory),
		zap.Int64("available_disk_gb", availableDisk))
	
	return nil
}

func (ce *ChaosEngine) checkServiceStatus(check SafetyCheck) error {
	ce.logger.Debug("Checking service status")
	
	for _, service := range check.RequiredServices {
		status, err := ce.getServiceStatus(service)
		if err != nil {
			return fmt.Errorf("failed to get status for service %s: %w", service, err)
		}
		
		if status != "active" {
			return fmt.Errorf("service %s is not active: %s", service, status)
		}
	}
	
	ce.logger.Info("Service status check passed",
		zap.Strings("services", check.RequiredServices))
	
	return nil
}

func (ce *ChaosEngine) checkNetworkConnectivity(check SafetyCheck) error {
	ce.logger.Debug("Checking network connectivity")
	
	for _, endpoint := range check.NetworkEndpoints {
		if err := ce.pingEndpoint(endpoint); err != nil {
			return fmt.Errorf("network connectivity check failed for %s: %w", endpoint, err)
		}
	}
	
	ce.logger.Info("Network connectivity check passed",
		zap.Strings("endpoints", check.NetworkEndpoints))
	
	return nil
}

func (ce *ChaosEngine) checkDataIntegrity(check SafetyCheck) error {
	ce.logger.Debug("Checking data integrity")
	
	// Check database connectivity and basic queries
	for _, db := range check.Databases {
		if err := ce.checkDatabaseIntegrity(db); err != nil {
			return fmt.Errorf("data integrity check failed for %s: %w", db, err)
		}
	}
	
	ce.logger.Info("Data integrity check passed",
		zap.Strings("databases", check.Databases))
	
	return nil
}

func (ce *ChaosEngine) checkBackupStatus(check SafetyCheck) error {
	ce.logger.Debug("Checking backup status")
	
	// Check if recent backups exist
	lastBackup, err := ce.getLastBackupTime()
	if err != nil {
		return fmt.Errorf("failed to get last backup time: %w", err)
	}
	
	maxAge := time.Duration(check.Thresholds.MaxBackupAge) * time.Hour
	if time.Since(lastBackup) > maxAge {
		return fmt.Errorf("backup is too old: %v > %v", time.Since(lastBackup), maxAge)
	}
	
	ce.logger.Info("Backup status check passed",
		zap.Time("last_backup", lastBackup))
	
	return nil
}

// Rollback step implementations

func (ce *ChaosEngine) restoreService(step RollbackStep) error {
	ce.logger.Info("Restoring service", zap.String("service", step.Target))
	
	// Start the service
	if err := ce.startService(step.Target); err != nil {
		return fmt.Errorf("failed to start service %s: %w", step.Target, err)
	}
	
	// Wait for service to be healthy
	if err := ce.waitForServiceHealth(step.Target, step.Timeout); err != nil {
		return fmt.Errorf("service %s did not become healthy: %w", step.Target, err)
	}
	
	ce.logger.Info("Service restored successfully", zap.String("service", step.Target))
	return nil
}

func (ce *ChaosEngine) restoreNetwork(step RollbackStep) error {
	ce.logger.Info("Restoring network", zap.String("target", step.Target))
	
	// Remove network chaos rules
	if err := ce.removeNetworkChaos(step.Target); err != nil {
		return fmt.Errorf("failed to remove network chaos: %w", err)
	}
	
	// Verify network connectivity
	if err := ce.verifyNetworkConnectivity(step.Target); err != nil {
		return fmt.Errorf("network connectivity not restored: %w", err)
	}
	
	ce.logger.Info("Network restored successfully", zap.String("target", step.Target))
	return nil
}

func (ce *ChaosEngine) restoreResources(step RollbackStep) error {
	ce.logger.Info("Restoring resources", zap.String("target", step.Target))
	
	// Remove resource constraints
	if err := ce.removeResourceConstraints(step.Target); err != nil {
		return fmt.Errorf("failed to remove resource constraints: %w", err)
	}
	
	// Verify resource availability
	if err := ce.verifyResourceAvailability(step.Target); err != nil {
		return fmt.Errorf("resources not restored: %w", err)
	}
	
	ce.logger.Info("Resources restored successfully", zap.String("target", step.Target))
	return nil
}

func (ce *ChaosEngine) restoreConfiguration(step RollbackStep) error {
	ce.logger.Info("Restoring configuration", zap.String("target", step.Target))
	
	// Restore configuration from backup
	if err := ce.restoreConfigFromBackup(step.Target); err != nil {
		return fmt.Errorf("failed to restore configuration: %w", err)
	}
	
	// Reload configuration
	if err := ce.reloadConfiguration(step.Target); err != nil {
		return fmt.Errorf("failed to reload configuration: %w", err)
	}
	
	ce.logger.Info("Configuration restored successfully", zap.String("target", step.Target))
	return nil
}

func (ce *ChaosEngine) cleanupChaos(step RollbackStep) error {
	ce.logger.Info("Cleaning up chaos artifacts")
	
	// Remove temporary files
	if err := ce.removeTemporaryFiles(); err != nil {
		ce.logger.Warn("Failed to remove temporary files", zap.Error(err))
	}
	
	// Reset system state
	if err := ce.resetSystemState(); err != nil {
		return fmt.Errorf("failed to reset system state: %w", err)
	}
	
	ce.logger.Info("Chaos cleanup completed successfully")
	return nil
}

// Helper methods (simplified implementations)

func (ce *ChaosEngine) getCPUUsage() (float64, error) {
	// Mock implementation - in reality would read from /proc/stat or similar
	return 25.5, nil
}

func (ce *ChaosEngine) getMemoryUsage() (float64, error) {
	// Mock implementation - in reality would read from /proc/meminfo or similar
	return 45.2, nil
}

func (ce *ChaosEngine) getDiskUsage() (float64, error) {
	// Mock implementation - in reality would use syscall or df command
	return 60.8, nil
}

func (ce *ChaosEngine) getAvailableMemory() (int64, error) {
	// Mock implementation - return available memory in MB
	return 2048, nil
}

func (ce *ChaosEngine) getAvailableDiskSpace() (int64, error) {
	// Mock implementation - return available disk space in GB
	return 50, nil
}

func (ce *ChaosEngine) getServiceStatus(service string) (string, error) {
	// Mock implementation - in reality would use systemctl or similar
	return "active", nil
}

func (ce *ChaosEngine) pingEndpoint(endpoint string) error {
	// Mock implementation - in reality would perform actual network ping
	ce.logger.Debug("Pinging endpoint", zap.String("endpoint", endpoint))
	return nil
}

func (ce *ChaosEngine) checkDatabaseIntegrity(db string) error {
	// Mock implementation - in reality would perform database health checks
	ce.logger.Debug("Checking database integrity", zap.String("database", db))
	return nil
}

func (ce *ChaosEngine) getLastBackupTime() (time.Time, error) {
	// Mock implementation - return a recent backup time
	return time.Now().Add(-2 * time.Hour), nil
}

func (ce *ChaosEngine) startService(service string) error {
	// Mock implementation - in reality would use systemctl start
	ce.logger.Debug("Starting service", zap.String("service", service))
	return nil
}

func (ce *ChaosEngine) waitForServiceHealth(service string, timeout time.Duration) error {
	// Mock implementation - in reality would poll service status
	ce.logger.Debug("Waiting for service health",
		zap.String("service", service),
		zap.Duration("timeout", timeout))
	return nil
}

func (ce *ChaosEngine) removeNetworkChaos(target string) error {
	// Mock implementation - in reality would remove iptables rules or similar
	ce.logger.Debug("Removing network chaos", zap.String("target", target))
	return nil
}

func (ce *ChaosEngine) verifyNetworkConnectivity(target string) error {
	// Mock implementation - in reality would test network connectivity
	ce.logger.Debug("Verifying network connectivity", zap.String("target", target))
	return nil
}

func (ce *ChaosEngine) removeResourceConstraints(target string) error {
	// Mock implementation - in reality would remove cgroup limits or similar
	ce.logger.Debug("Removing resource constraints", zap.String("target", target))
	return nil
}

func (ce *ChaosEngine) verifyResourceAvailability(target string) error {
	// Mock implementation - in reality would check resource availability
	ce.logger.Debug("Verifying resource availability", zap.String("target", target))
	return nil
}

func (ce *ChaosEngine) restoreConfigFromBackup(target string) error {
	// Mock implementation - in reality would restore from actual backup
	ce.logger.Debug("Restoring config from backup", zap.String("target", target))
	return nil
}

func (ce *ChaosEngine) reloadConfiguration(target string) error {
	// Mock implementation - in reality would reload service configuration
	ce.logger.Debug("Reloading configuration", zap.String("target", target))
	return nil
}

func (ce *ChaosEngine) removeTemporaryFiles() error {
	// Mock implementation - in reality would clean up temp files
	ce.logger.Debug("Removing temporary files")
	return nil
}

func (ce *ChaosEngine) resetSystemState() error {
	// Mock implementation - in reality would reset system to clean state
	ce.logger.Debug("Resetting system state")
	return nil
}

// experimentScheduler schedules and runs experiments
func (ce *ChaosEngine) experimentScheduler(ctx context.Context) {
	ticker := time.NewTicker(ce.config.ExperimentInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ce.stopChan:
			return
		case <-ticker.C:
			ce.scheduleRandomExperiment(ctx)
		}
	}
}

// scheduleRandomExperiment schedules a random experiment
func (ce *ChaosEngine) scheduleRandomExperiment(ctx context.Context) {
	ce.mu.RLock()
	experiments := make([]*ChaosExperiment, 0, len(ce.experiments))
	for _, exp := range ce.experiments {
		if exp.Status == StatusPending {
			experiments = append(experiments, exp)
		}
	}
	ce.mu.RUnlock()
	
	if len(experiments) == 0 {
		return
	}
	
	// Select random experiment
	experiment := experiments[rand.Intn(len(experiments))]
	
	// Run experiment
	go func() {
		if err := ce.RunExperiment(ctx, experiment.ID); err != nil {
			ce.logger.Error("Scheduled experiment failed",
				zap.String("experiment_id", experiment.ID),
				zap.Error(err),
			)
		}
	}()
}

// safetyMonitor monitors system safety during experiments
func (ce *ChaosEngine) safetyMonitor(ctx context.Context) {
	ticker := time.NewTicker(ce.config.HealthCheckInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ce.stopChan:
			return