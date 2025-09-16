package soar

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"go.uber.org/zap"
)

// SOARPlatform provides Security Orchestration, Automation & Response capabilities
type SOARPlatform struct {
	logger *zap.Logger
	config *SOARConfig
	
	// Core components
	incidentManager    *IncidentManager
	playbookEngine     *PlaybookEngine
	orchestrator       *ResponseOrchestrator
	threatIntel        *ThreatIntelligence
	
	// Integration layers
	siemConnector      SIEMConnector
	ticketingSystem    TicketingSystem
	communicationHub   CommunicationHub
	
	// Workflow and automation
	workflowEngine     *WorkflowEngine
	actionExecutor     *ActionExecutor
	
	// Knowledge base
	knowledgeBase      *SecurityKnowledgeBase
	responseLibrary    *ResponseLibrary
	
	// Metrics
	incidentsProcessed    *prometheus.CounterVec
	responseTimeHistogram *prometheus.HistogramVec
	automationSuccess     *prometheus.CounterVec
	playbookExecutions    *prometheus.CounterVec
	mttrGauge             *prometheus.GaugeVec
	
	// State
	mu       sync.RWMutex
	running  bool
	stopChan chan struct{}
}

// SOARConfig holds SOAR platform configuration
type SOARConfig struct {
	// Core settings
	AutoResponseEnabled     bool          `json:"auto_response_enabled"`
	ResponseTimeout         time.Duration `json:"response_timeout"`
	EscalationTimeout       time.Duration `json:"escalation_timeout"`
	
	// Incident management
	IncidentClassification  ClassificationConfig `json:"incident_classification"`
	AutoTicketCreation      bool                `json:"auto_ticket_creation"`
	
	// Playbook settings
	PlaybookDirectory       string         `json:"playbook_directory"`
	DefaultPlaybooks        []string       `json:"default_playbooks"`
	CustomPlaybooks         []string       `json:"custom_playbooks"`
	
	// Integration settings
	SIEMIntegration         SIEMConfig     `json:"siem_integration"`
	TicketingIntegration    TicketingConfig `json:"ticketing_integration"`
	CommunicationConfig     CommConfig     `json:"communication"`
	ThreatIntelFeeds        []ThreatIntelFeed `json:"threat_intel_feeds"`
	
	// Automation settings
	MaxConcurrentPlaybooks  int            `json:"max_concurrent_playbooks"`
	AutoEnrichment          bool           `json:"auto_enrichment"`
	AutoContainment         bool           `json:"auto_containment"`
	
	// Compliance and audit
	ComplianceFrameworks    []string       `json:"compliance_frameworks"`
	AuditLogging            bool           `json:"audit_logging"`
	DataRetention           time.Duration  `json:"data_retention"`
}

// SecurityIncident represents a security incident
type SecurityIncident struct {
	ID          string                 `json:"id"`
	Title       string                 `json:"title"`
	Description string                 `json:"description"`
	
	// Classification
	Type        IncidentType          `json:"type"`
	Category    IncidentCategory      `json:"category"`
	Severity    IncidentSeverity      `json:"severity"`
	Priority    IncidentPriority      `json:"priority"`
	Confidence  float64               `json:"confidence"`
	
	// Status tracking
	Status      IncidentStatus        `json:"status"`
	Phase       ResponsePhase         `json:"phase"`
	
	// Timing
	DetectedAt  time.Time             `json:"detected_at"`
	ReportedAt  time.Time             `json:"reported_at"`
	StartedAt   time.Time             `json:"started_at"`
	ResolvedAt  *time.Time            `json:"resolved_at,omitempty"`
	ClosedAt    *time.Time            `json:"closed_at,omitempty"`
	
	// Impact assessment
	Impact      ImpactAssessment      `json:"impact"`
	Scope       IncidentScope         `json:"scope"`
	
	// Technical details
	Indicators  []IOC                 `json:"indicators"`
	Assets      []AffectedAsset       `json:"affected_assets"`
	Evidence    []Evidence            `json:"evidence"`
	Artifacts   []DigitalArtifact     `json:"artifacts"`
	
	// Response details
	Playbooks   []string              `json:"playbooks_executed"`
	Actions     []ResponseAction      `json:"actions_taken"`
	Timeline    []IncidentEvent       `json:"timeline"`
	
	// Assignment and communication
	AssignedTo  string                `json:"assigned_to"`
	Team        string                `json:"team"`
	Stakeholders []string             `json:"stakeholders"`
	
	// External references
	TicketID    string                `json:"ticket_id,omitempty"`
	CaseID      string                `json:"case_id,omitempty"`
	
	// Analysis and lessons learned
	RootCause   string                `json:"root_cause,omitempty"`
	LessonsLearned []string           `json:"lessons_learned,omitempty"`
	Recommendations []string          `json:"recommendations,omitempty"`
	
	// Metadata
	Tags        []string              `json:"tags"`
	Metadata    map[string]interface{} `json:"metadata"`
	CreatedBy   string                `json:"created_by"`
	CreatedAt   time.Time             `json:"created_at"`
	UpdatedAt   time.Time             `json:"updated_at"`
}

// ResponsePlaybook defines automated response procedures
type ResponsePlaybook struct {
	ID          string                `json:"id"`
	Name        string                `json:"name"`
	Description string                `json:"description"`
	Version     string                `json:"version"`
	
	// Triggers and conditions
	Triggers    []PlaybookTrigger     `json:"triggers"`
	Conditions  []PlaybookCondition   `json:"conditions"`
	
	// Workflow definition
	Workflow    WorkflowDefinition    `json:"workflow"`
	Actions     []AutomatedAction     `json:"actions"`
	
	// Configuration
	Timeout     time.Duration         `json:"timeout"`
	MaxRetries  int                   `json:"max_retries"`
	Parallel    bool                  `json:"parallel_execution"`
	
	// Approval and escalation
	RequiresApproval bool             `json:"requires_approval"`
	ApprovalTimeout  time.Duration    `json:"approval_timeout"`
	EscalationChain  []string         `json:"escalation_chain"`
	
	// Metadata
	Author      string                `json:"author"`
	Category    string                `json:"category"`
	Tags        []string              `json:"tags"`
	Active      bool                  `json:"active"`
	CreatedAt   time.Time             `json:"created_at"`
	UpdatedAt   time.Time             `json:"updated_at"`
}

// AutomatedAction represents an automated response action
type AutomatedAction struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Type        ActionType             `json:"type"`
	Category    ActionCategory         `json:"category"`
	
	// Execution details
	Command     string                 `json:"command,omitempty"`
	Script      string                 `json:"script,omitempty"`
	API         APICall                `json:"api,omitempty"`
	Parameters  map[string]interface{} `json:"parameters"`
	
	// Conditions and validation
	Preconditions []Condition          `json:"preconditions"`
	Validation    ValidationRule       `json:"validation"`
	
	// Error handling
	OnSuccess   []string               `json:"on_success"`
	OnFailure   []string               `json:"on_failure"`
	Rollback    *AutomatedAction       `json:"rollback,omitempty"`
	
	// Timing and retry
	Timeout     time.Duration          `json:"timeout"`
	MaxRetries  int                    `json:"max_retries"`
	RetryDelay  time.Duration          `json:"retry_delay"`
	
	// Impact and safety
	Impact      ImpactLevel            `json:"impact"`
	Reversible  bool                   `json:"reversible"`
	Destructive bool                   `json:"destructive"`
	
	// Approval requirements
	RequiresApproval bool              `json:"requires_approval"`
	ApprovalLevel    ApprovalLevel     `json:"approval_level"`
	
	// Metadata
	Description string                 `json:"description"`
	Tags        []string               `json:"tags"`
	Active      bool                   `json:"active"`
}

// WorkflowDefinition defines the workflow structure
type WorkflowDefinition struct {
	Steps       []WorkflowStep        `json:"steps"`
	Branches    []WorkflowBranch      `json:"branches"`
	Loops       []WorkflowLoop        `json:"loops"`
	Parallel    []ParallelExecution   `json:"parallel_blocks"`
	ErrorHandling ErrorHandlingConfig `json:"error_handling"`
}

// IOC (Indicator of Compromise) represents threat indicators
type IOC struct {
	ID          string                 `json:"id"`
	Type        IOCType                `json:"type"`
	Value       string                 `json:"value"`
	Context     string                 `json:"context"`
	Confidence  float64                `json:"confidence"`
	TLP         TLPLevel               `json:"tlp"` // Traffic Light Protocol
	Source      string                 `json:"source"`
	FirstSeen   time.Time              `json:"first_seen"`
	LastSeen    time.Time              `json:"last_seen"`
	ExpiresAt   *time.Time             `json:"expires_at,omitempty"`
	Tags        []string               `json:"tags"`
	References  []string               `json:"references"`
}

// Enums and types
type IncidentType string
const (
	IncidentTypeMalware           IncidentType = "malware"
	IncidentTypePhishing          IncidentType = "phishing"
	IncidentTypeDataBreach        IncidentType = "data_breach"
	IncidentTypeUnauthorizedAccess IncidentType = "unauthorized_access"
	IncidentTypeDenialOfService   IncidentType = "denial_of_service"
	IncidentTypeInsiderThreat     IncidentType = "insider_threat"
	IncidentTypeSystemCompromise  IncidentType = "system_compromise"
	IncidentTypeDataLoss          IncidentType = "data_loss"
	IncidentTypeCompliance        IncidentType = "compliance_violation"
	IncidentTypeOther             IncidentType = "other"
)

type IncidentSeverity string
const (
	SeverityCritical IncidentSeverity = "critical"
	SeverityHigh     IncidentSeverity = "high"
	SeverityMedium   IncidentSeverity = "medium"
	SeverityLow      IncidentSeverity = "low"
	SeverityInfo     IncidentSeverity = "info"
)

type IncidentStatus string
const (
	StatusNew         IncidentStatus = "new"
	StatusAssigned    IncidentStatus = "assigned"
	StatusInProgress  IncidentStatus = "in_progress"
	StatusContained   IncidentStatus = "contained"
	StatusEradicated  IncidentStatus = "eradicated"
	StatusRecovered   IncidentStatus = "recovered"
	StatusClosed      IncidentStatus = "closed"
)

type ResponsePhase string
const (
	PhaseDetection    ResponsePhase = "detection"
	PhaseAnalysis     ResponsePhase = "analysis"
	PhaseContainment  ResponsePhase = "containment"
	PhaseEradication  ResponsePhase = "eradication"
	PhaseRecovery     ResponsePhase = "recovery"
	PhaseLessonsLearned ResponsePhase = "lessons_learned"
)

type ActionType string
const (
	ActionTypeAPI        ActionType = "api"
	ActionTypeScript     ActionType = "script"
	ActionTypeCommand    ActionType = "command"
	ActionTypeEmail      ActionType = "email"
	ActionTypeWebhook    ActionType = "webhook"
	ActionTypeWorkflow   ActionType = "workflow"
	ActionTypeManual     ActionType = "manual"
)

type ActionCategory string
const (
	CategoryInvestigation ActionCategory = "investigation"
	CategoryContainment   ActionCategory = "containment"
	CategoryEradication   ActionCategory = "eradication"
	CategoryRecovery      ActionCategory = "recovery"
	CategoryNotification  ActionCategory = "notification"
	CategoryDocumentation ActionCategory = "documentation"
)

type IOCType string
const (
	IOCTypeIP        IOCType = "ip"
	IOCTypeDomain    IOCType = "domain"
	IOCTypeURL       IOCType = "url"
	IOCTypeHash      IOCType = "hash"
	IOCTypeEmail     IOCType = "email"
	IOCTypeFilename  IOCType = "filename"
	IOCTypeRegistry  IOCType = "registry"
	IOCTypeMutex     IOCType = "mutex"
	IOCTypeUserAgent IOCType = "user_agent"
)

// NewSOARPlatform creates a new SOAR platform instance
func NewSOARPlatform(logger *zap.Logger, config *SOARConfig) *SOARPlatform {
	platform := &SOARPlatform{
		logger:   logger,
		config:   config,
		stopChan: make(chan struct{}),
		
		incidentsProcessed: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "soar_incidents_processed_total",
				Help: "Total number of incidents processed by SOAR",
			},
			[]string{"type", "severity", "status"},
		),
		
		responseTimeHistogram: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name: "soar_response_time_seconds",
				Help: "Time taken for incident response",
				Buckets: []float64{60, 300, 900, 1800, 3600, 7200, 14400, 28800},
			},
			[]string{"phase", "severity"},
		),
		
		automationSuccess: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "soar_automation_success_total",
				Help: "Total number of successful automated actions",
			},
			[]string{"action_type", "playbook"},
		),
		
		playbookExecutions: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "soar_playbook_executions_total",
				Help: "Total number of playbook executions",
			},
			[]string{"playbook", "result"},
		),
		
		mttrGauge: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "soar_mttr_seconds",
				Help: "Mean Time To Recovery",
			},
			[]string{"incident_type", "severity"},
		),
	}
	
	// Initialize components
	platform.incidentManager = NewIncidentManager(logger, config)
	platform.playbookEngine = NewPlaybookEngine(logger, config)
	platform.orchestrator = NewResponseOrchestrator(logger, config)
	platform.threatIntel = NewThreatIntelligence(logger, config)
	platform.workflowEngine = NewWorkflowEngine(logger, config)
	platform.actionExecutor = NewActionExecutor(logger, config)
	platform.knowledgeBase = NewSecurityKnowledgeBase(logger, config)
	platform.responseLibrary = NewResponseLibrary(logger, config)
	
	// Load playbooks
	platform.loadPlaybooks()
	
	return platform
}

// Start begins SOAR platform operations
func (soar *SOARPlatform) Start(ctx context.Context) error {
	soar.mu.Lock()
	defer soar.mu.Unlock()
	
	if soar.running {
		return fmt.Errorf("SOAR platform already running")
	}
	
	soar.logger.Info("Starting SOAR platform")
	
	// Start core components
	if err := soar.incidentManager.Start(ctx); err != nil {
		return fmt.Errorf("failed to start incident manager: %w", err)
	}
	
	if err := soar.playbookEngine.Start(ctx); err != nil {
		return fmt.Errorf("failed to start playbook engine: %w", err)
	}
	
	if err := soar.orchestrator.Start(ctx); err != nil {
		return fmt.Errorf("failed to start response orchestrator: %w", err)
	}
	
	if err := soar.threatIntel.Start(ctx); err != nil {
		return fmt.Errorf("failed to start threat intelligence: %w", err)
	}
	
	if err := soar.workflowEngine.Start(ctx); err != nil {
		return fmt.Errorf("failed to start workflow engine: %w", err)
	}
	
	if err := soar.actionExecutor.Start(ctx); err != nil {
		return fmt.Errorf("failed to start action executor: %w", err)
	}
	
	// Initialize integrations
	if err := soar.initializeIntegrations(ctx); err != nil {
		soar.logger.Warn("failed to initialize some integrations", zap.Error(err))
	}
	
	// Start automation loops
	go soar.incidentProcessingLoop(ctx)
	go soar.automatedResponseLoop(ctx)
	go soar.metricsCollectionLoop(ctx)
	
	soar.running = true
	soar.logger.Info("SOAR platform started successfully")
	
	return nil
}

// Stop stops the SOAR platform
func (soar *SOARPlatform) Stop() error {
	soar.mu.Lock()
	defer soar.mu.Unlock()
	
	if !soar.running {
		return nil
	}
	
	soar.logger.Info("Stopping SOAR platform")
	
	close(soar.stopChan)
	
	// Stop components
	if soar.incidentManager != nil {
		soar.incidentManager.Stop()
	}
	if soar.playbookEngine != nil {
		soar.playbookEngine.Stop()
	}
	if soar.orchestrator != nil {
		soar.orchestrator.Stop()
	}
	if soar.threatIntel != nil {
		soar.threatIntel.Stop()
	}
	if soar.workflowEngine != nil {
		soar.workflowEngine.Stop()
	}
	if soar.actionExecutor != nil {
		soar.actionExecutor.Stop()
	}
	
	soar.running = false
	soar.logger.Info("SOAR platform stopped")
	
	return nil
}

// ProcessIncident processes a new security incident
func (soar *SOARPlatform) ProcessIncident(ctx context.Context, incident *SecurityIncident) error {
	start := time.Now()
	defer func() {
		soar.responseTimeHistogram.WithLabelValues(string(incident.Phase), string(incident.Severity)).Observe(time.Since(start).Seconds())
	}()
	
	soar.logger.Info("processing security incident",
		zap.String("incident_id", incident.ID),
		zap.String("type", string(incident.Type)),
		zap.String("severity", string(incident.Severity)))
	
	// Enrich incident with threat intelligence
	if err := soar.enrichIncident(ctx, incident); err != nil {
		soar.logger.Error("failed to enrich incident", zap.Error(err))
	}
	
	// Classify and prioritize incident
	if err := soar.classifyIncident(incident); err != nil {
		soar.logger.Error("failed to classify incident", zap.Error(err))
	}
	
	// Select and execute appropriate playbooks
	playbooks := soar.selectPlaybooks(incident)
	for _, playbookID := range playbooks {
		if err := soar.executePlaybook(ctx, incident, playbookID); err != nil {
			soar.logger.Error("failed to execute playbook",
				zap.String("playbook", playbookID),
				zap.Error(err))
			soar.playbookExecutions.WithLabelValues(playbookID, "failed").Inc()
		} else {
			soar.playbookExecutions.WithLabelValues(playbookID, "success").Inc()
		}
	}
	
	// Create ticket if enabled
	if soar.config.AutoTicketCreation && soar.ticketingSystem != nil {
		ticketID, err := soar.ticketingSystem.CreateTicket(incident)
		if err != nil {
			soar.logger.Error("failed to create ticket", zap.Error(err))
		} else {
			incident.TicketID = ticketID
			soar.logger.Info("created ticket", zap.String("ticket_id", ticketID))
		}
	}
	
	// Update metrics
	soar.incidentsProcessed.WithLabelValues(
		string(incident.Type),
		string(incident.Severity),
		string(incident.Status),
	).Inc()
	
	return nil
}

// ExecutePlaybook executes a response playbook
func (soar *SOARPlatform) executePlaybook(ctx context.Context, incident *SecurityIncident, playbookID string) error {
	playbook, err := soar.playbookEngine.GetPlaybook(playbookID)
	if err != nil {
		return fmt.Errorf("failed to get playbook %s: %w", playbookID, err)
	}
	
	soar.logger.Info("executing playbook",
		zap.String("playbook", playbookID),
		zap.String("incident", incident.ID))
	
	// Check if playbook is applicable
	if !soar.isPlaybookApplicable(playbook, incident) {
		return fmt.Errorf("playbook %s not applicable to incident", playbookID)
	}
	
	// Execute playbook workflow
	execution := &PlaybookExecution{
		ID:         fmt.Sprintf("exec-%s-%d", playbookID, time.Now().Unix()),
		PlaybookID: playbookID,
		IncidentID: incident.ID,
		StartTime:  time.Now(),
		Status:     ExecutionStatusRunning,
		Actions:    []ActionExecution{},
	}
	
	// Execute each action in the playbook
	for _, action := range playbook.Actions {
		actionExecution := soar.executeAction(ctx, &action, incident)
		execution.Actions = append(execution.Actions, *actionExecution)
		
		// Handle action failure
		if actionExecution.Status == ActionStatusFailed {
			if len(action.OnFailure) > 0 {
				// Execute failure handlers
				soar.logger.Info("executing failure handlers", zap.Strings("handlers", action.OnFailure))
			}
			
			// Stop execution if critical action fails
			if action.Impact == ImpactLevelCritical {
				execution.Status = ExecutionStatusFailed
				break
			}
		} else {
			// Execute success handlers
			if len(action.OnSuccess) > 0 {
				soar.logger.Info("executing success handlers", zap.Strings("handlers", action.OnSuccess))
			}
		}
		
		// Update automation success metrics
		if actionExecution.Status == ActionStatusCompleted {
			soar.automationSuccess.WithLabelValues(string(action.Type), playbookID).Inc()
		}
	}
	
	execution.EndTime = time.Now()
	if execution.Status != ExecutionStatusFailed {
		execution.Status = ExecutionStatusCompleted
	}
	
	// Add playbook to incident history
	incident.Playbooks = append(incident.Playbooks, playbookID)
	
	soar.logger.Info("playbook execution completed",
		zap.String("playbook", playbookID),
		zap.String("status", string(execution.Status)),
		zap.Duration("duration", execution.EndTime.Sub(execution.StartTime)))
	
	return nil
}

// executeAction executes a single automated action
func (soar *SOARPlatform) executeAction(ctx context.Context, action *AutomatedAction, incident *SecurityIncident) *ActionExecution {
	execution := &ActionExecution{
		ActionID:  action.ID,
		StartTime: time.Now(),
		Status:    ActionStatusRunning,
		Output:    make(map[string]interface{}),
	}
	
	soar.logger.Debug("executing action",
		zap.String("action_id", action.ID),
		zap.String("action_name", action.Name),
		zap.String("type", string(action.Type)))
	
	// Check preconditions
	if !soar.checkPreconditions(action.Preconditions, incident) {
		execution.Status = ActionStatusSkipped
		execution.Message = "Preconditions not met"
		execution.EndTime = time.Now()
		return execution
	}
	
	// Execute action based on type
	var err error
	switch action.Type {
	case ActionTypeAPI:
		err = soar.executeAPIAction(ctx, action, incident, execution)
	case ActionTypeScript:
		err = soar.executeScriptAction(ctx, action, incident, execution)
	case ActionTypeCommand:
		err = soar.executeCommandAction(ctx, action, incident, execution)
	case ActionTypeEmail:
		err = soar.executeEmailAction(ctx, action, incident, execution)
	case ActionTypeWebhook:
		err = soar.executeWebhookAction(ctx, action, incident, execution)
	default:
		err = fmt.Errorf("unsupported action type: %s", action.Type)
	}
	
	execution.EndTime = time.Now()
	if err != nil {
		execution.Status = ActionStatusFailed
		execution.Error = err.Error()
		soar.logger.Error("action execution failed",
			zap.String("action_id", action.ID),
			zap.Error(err))
	} else {
		execution.Status = ActionStatusCompleted
		soar.logger.Debug("action execution completed",
			zap.String("action_id", action.ID),
			zap.Duration("duration", execution.EndTime.Sub(execution.StartTime)))
	}
	
	return execution
}

// enrichIncident enriches incident with threat intelligence and context
func (soar *SOARPlatform) enrichIncident(ctx context.Context, incident *SecurityIncident) error {
	// Enrich with threat intelligence
	for i, ioc := range incident.Indicators {
		enrichedIOC, err := soar.threatIntel.EnrichIOC(ctx, ioc)
		if err != nil {
			soar.logger.Warn("failed to enrich IOC",
				zap.String("ioc", ioc.Value),
				zap.Error(err))
			continue
		}
		incident.Indicators[i] = *enrichedIOC
	}
	
	// Add additional context from knowledge base
	if relatedIncidents, err := soar.knowledgeBase.FindRelatedIncidents(incident); err == nil {
		if incident.Metadata == nil {
			incident.Metadata = make(map[string]interface{})
		}
		incident.Metadata["related_incidents"] = relatedIncidents
	}
	
	return nil
}

// Additional types and interfaces for SOAR implementation
type PlaybookExecution struct {
	ID         string            `json:"id"`
	PlaybookID string            `json:"playbook_id"`
	IncidentID string            `json:"incident_id"`
	StartTime  time.Time         `json:"start_time"`
	EndTime    time.Time         `json:"end_time"`
	Status     ExecutionStatus   `json:"status"`
	Actions    []ActionExecution `json:"actions"`
}

type ActionExecution struct {
	ActionID  string                 `json:"action_id"`
	StartTime time.Time              `json:"start_time"`
	EndTime   time.Time              `json:"end_time"`
	Status    ActionStatus           `json:"status"`
	Output    map[string]interface{} `json:"output"`
	Error     string                 `json:"error,omitempty"`
	Message   string                 `json:"message,omitempty"`
}

type ExecutionStatus string
const (
	ExecutionStatusPending   ExecutionStatus = "pending"
	ExecutionStatusRunning   ExecutionStatus = "running"
	ExecutionStatusCompleted ExecutionStatus = "completed"
	ExecutionStatusFailed    ExecutionStatus = "failed"
	ExecutionStatusCancelled ExecutionStatus = "cancelled"
)

type ActionStatus string
const (
	ActionStatusPending   ActionStatus = "pending"
	ActionStatusRunning   ActionStatus = "running"
	ActionStatusCompleted ActionStatus = "completed"
	ActionStatusFailed    ActionStatus = "failed"
	ActionStatusSkipped   ActionStatus = "skipped"
)

// Additional methods would continue here...
// Including: classifyIncident, selectPlaybooks, isPlaybookApplicable, executeAPIAction, etc.