package hsm

import (
	"context"
	"crypto"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"go.uber.org/zap"
)

// EnterpriseKeyManagementService provides HSM-based key management and escrow
type EnterpriseKeyManagementService struct {
	logger *zap.Logger
	config *HSMConfig
	
	// HSM providers
	primaryHSM   HSMProvider
	backupHSM    HSMProvider
	
	// Key stores
	keyStore     *KeyStore
	escrowVault  *EscrowVault
	
	// Key rotation and lifecycle
	keyRotator   *KeyRotator
	keyAuditor   *KeyAuditor
	
	// Compliance and governance
	complianceEngine *ComplianceEngine
	auditLogger      *SecurityAuditLogger
	
	// Metrics
	keyOperations    *prometheus.CounterVec
	keyRotations     *prometheus.CounterVec
	escrowOperations *prometheus.CounterVec
	hsmLatency       *prometheus.HistogramVec
	keyAgeGauge      *prometheus.GaugeVec
	
	// State
	mu       sync.RWMutex
	running  bool
	stopChan chan struct{}
}

// HSMConfig holds HSM configuration
type HSMConfig struct {
	// HSM settings
	PrimaryHSM   HSMProviderConfig `json:"primary_hsm"`
	BackupHSM    HSMProviderConfig `json:"backup_hsm"`
	
	// Key management settings
	DefaultKeySize       int           `json:"default_key_size"`
	KeyRotationInterval  time.Duration `json:"key_rotation_interval"`
	KeyExpirationPeriod  time.Duration `json:"key_expiration_period"`
	
	// Escrow settings
	EnableKeyEscrow      bool          `json:"enable_key_escrow"`
	EscrowThreshold      int           `json:"escrow_threshold"` // Number of key shares
	RecoveryThreshold    int           `json:"recovery_threshold"` // Min shares for recovery
	EscrowCustodians     []string      `json:"escrow_custodians"`
	
	// Compliance settings
	FIPSMode             bool          `json:"fips_mode"`
	CommonCriteria       bool          `json:"common_criteria"`
	ComplianceFrameworks []string      `json:"compliance_frameworks"`
	
	// Audit settings
	EnableKeyAudit       bool          `json:"enable_key_audit"`
	AuditLevel           string        `json:"audit_level"`
	AuditRetention       time.Duration `json:"audit_retention"`
	
	// Performance settings
	ConnectionPool       int           `json:"connection_pool"`
	RequestTimeout       time.Duration `json:"request_timeout"`
	RetryAttempts        int           `json:"retry_attempts"`
}

// HSMProviderConfig holds HSM provider-specific configuration
type HSMProviderConfig struct {
	Type        string            `json:"type"` // pkcs11, aws-kms, azure-kv, hashicorp-vault
	Endpoint    string            `json:"endpoint"`
	Region      string            `json:"region,omitempty"`
	Credentials map[string]string `json:"credentials"`
	Options     map[string]interface{} `json:"options"`
}

// KeyStore manages cryptographic keys
type KeyStore struct {
	keys    map[string]*ManagedKey
	indices map[string][]string // For efficient lookups
	mu      sync.RWMutex
}

// EscrowVault manages key escrow operations
type EscrowVault struct {
	escrowShares  map[string]*EscrowShare
	custodians    map[string]*EscrowCustodian
	policies      map[string]*EscrowPolicy
	mu            sync.RWMutex
}

// ManagedKey represents a managed cryptographic key
type ManagedKey struct {
	// Key identification
	ID          string    `json:"id"`
	Name        string    `json:"name"`
	Version     int       `json:"version"`
	
	// Key properties
	Algorithm   string    `json:"algorithm"`
	KeySize     int       `json:"key_size"`
	KeyType     KeyType   `json:"key_type"`
	Usage       []KeyUsage `json:"usage"`
	
	// Key material (references, not actual keys)
	PrimaryKeyRef   string `json:"primary_key_ref"`
	BackupKeyRef    string `json:"backup_key_ref,omitempty"`
	PublicKey       []byte `json:"public_key,omitempty"`
	
	// Lifecycle
	State       KeyState  `json:"state"`
	CreatedAt   time.Time `json:"created_at"`
	ActivatedAt time.Time `json:"activated_at,omitempty"`
	ExpiresAt   time.Time `json:"expires_at,omitempty"`
	RotatedAt   time.Time `json:"rotated_at,omitempty"`
	
	// Escrow information
	EscrowEnabled bool     `json:"escrow_enabled"`
	EscrowShares  []string `json:"escrow_shares,omitempty"`
	
	// Metadata
	Owner       string                 `json:"owner"`
	Application string                 `json:"application"`
	Environment string                 `json:"environment"`
	Tags        []string               `json:"tags"`
	Metadata    map[string]interface{} `json:"metadata"`
	
	// Audit trail
	AuditTrail  []KeyAuditEntry `json:"audit_trail"`
	
	// Compliance
	ComplianceLevel string   `json:"compliance_level"`
	Certifications  []string `json:"certifications"`
}

// EscrowShare represents a key escrow share
type EscrowShare struct {
	ID          string    `json:"id"`
	KeyID       string    `json:"key_id"`
	ShareIndex  int       `json:"share_index"`
	CustodianID string    `json:"custodian_id"`
	ShareData   []byte    `json:"share_data"` // Encrypted share
	CreatedAt   time.Time `json:"created_at"`
	AccessedAt  time.Time `json:"accessed_at,omitempty"`
}

// EscrowCustodian represents an escrow key custodian
type EscrowCustodian struct {
	ID          string    `json:"id"`
	Name        string    `json:"name"`
	Role        string    `json:"role"`
	Department  string    `json:"department"`
	ContactInfo string    `json:"contact_info"`
	PublicKey   []byte    `json:"public_key"`
	Active      bool      `json:"active"`
	CreatedAt   time.Time `json:"created_at"`
	LastAccess  time.Time `json:"last_access,omitempty"`
}

// EscrowPolicy defines key escrow policies
type EscrowPolicy struct {
	ID                string        `json:"id"`
	Name              string        `json:"name"`
	KeyTypes          []KeyType     `json:"key_types"`
	MinCustodians     int           `json:"min_custodians"`
	MaxCustodians     int           `json:"max_custodians"`
	RecoveryThreshold int           `json:"recovery_threshold"`
	ApprovalRequired  bool          `json:"approval_required"`
	AuditRequired     bool          `json:"audit_required"`
	RetentionPeriod   time.Duration `json:"retention_period"`
	Active            bool          `json:"active"`
}

// KeyAuditEntry represents a key audit log entry
type KeyAuditEntry struct {
	Timestamp   time.Time              `json:"timestamp"`
	Operation   string                 `json:"operation"`
	User        string                 `json:"user"`
	Source      string                 `json:"source"`
	Result      string                 `json:"result"`
	Details     map[string]interface{} `json:"details"`
	IPAddress   string                 `json:"ip_address,omitempty"`
	UserAgent   string                 `json:"user_agent,omitempty"`
}

// Enums and types
type KeyType string
const (
	KeyTypeSymmetric KeyType = "symmetric"
	KeyTypeAsymmetric KeyType = "asymmetric"
	KeyTypeHMAC      KeyType = "hmac"
	KeyTypeDerivation KeyType = "derivation"
)

type KeyUsage string
const (
	KeyUsageEncrypt    KeyUsage = "encrypt"
	KeyUsageDecrypt    KeyUsage = "decrypt"
	KeyUsageSign       KeyUsage = "sign"
	KeyUsageVerify     KeyUsage = "verify"
	KeyUsageDerive     KeyUsage = "derive"
	KeyUsageWrap       KeyUsage = "wrap"
	KeyUsageUnwrap     KeyUsage = "unwrap"
)

type KeyState string
const (
	KeyStatePending     KeyState = "pending"
	KeyStateActive      KeyState = "active"
	KeyStateRotating    KeyState = "rotating"
	KeyStateDeprecated  KeyState = "deprecated"
	KeyStateRevoked     KeyState = "revoked"
	KeyStateDestroyed   KeyState = "destroyed"
)

// HSMProvider interface for different HSM implementations
type HSMProvider interface {
	Connect(ctx context.Context) error
	Disconnect() error
	
	// Key operations
	GenerateKey(keyType KeyType, algorithm string, keySize int) (*KeyReference, error)
	ImportKey(keyMaterial []byte, keyType KeyType, algorithm string) (*KeyReference, error)
	ExportKey(keyRef *KeyReference, format string) ([]byte, error)
	DeleteKey(keyRef *KeyReference) error
	
	// Cryptographic operations
	Encrypt(keyRef *KeyReference, plaintext []byte) ([]byte, error)
	Decrypt(keyRef *KeyReference, ciphertext []byte) ([]byte, error)
	Sign(keyRef *KeyReference, data []byte, algorithm string) ([]byte, error)
	Verify(keyRef *KeyReference, data []byte, signature []byte, algorithm string) error
	
	// Key derivation
	DeriveKey(parentKeyRef *KeyReference, derivationData []byte) (*KeyReference, error)
	
	// Key wrapping
	WrapKey(keyRef *KeyReference, wrappingKeyRef *KeyReference) ([]byte, error)
	UnwrapKey(wrappedKey []byte, wrappingKeyRef *KeyReference) (*KeyReference, error)
	
	// Metadata
	GetKeyInfo(keyRef *KeyReference) (*KeyInfo, error)
	ListKeys() ([]*KeyReference, error)
	
	// Health and status
	GetStatus() (*HSMStatus, error)
	GetType() string
}

// KeyReference represents a reference to a key in HSM
type KeyReference struct {
	ID       string                 `json:"id"`
	Label    string                 `json:"label"`
	Handle   interface{}            `json:"-"` // HSM-specific handle
	Metadata map[string]interface{} `json:"metadata"`
}

// KeyInfo contains key information from HSM
type KeyInfo struct {
	Algorithm   string    `json:"algorithm"`
	KeySize     int       `json:"key_size"`
	KeyType     KeyType   `json:"key_type"`
	Usage       []KeyUsage `json:"usage"`
	CreatedAt   time.Time `json:"created_at"`
	ExpiresAt   *time.Time `json:"expires_at,omitempty"`
	Extractable bool      `json:"extractable"`
	Sensitive   bool      `json:"sensitive"`
}

// HSMStatus represents HSM health status
type HSMStatus struct {
	Connected      bool      `json:"connected"`
	Authenticated  bool      `json:"authenticated"`
	KeysLoaded     int       `json:"keys_loaded"`
	LastOperation  time.Time `json:"last_operation"`
	ErrorCount     int       `json:"error_count"`
	ResponseTime   time.Duration `json:"response_time"`
}

// NewEnterpriseKeyManagementService creates a new enterprise key management service
func NewEnterpriseKeyManagementService(logger *zap.Logger, config *HSMConfig) (*EnterpriseKeyManagementService, error) {
	service := &EnterpriseKeyManagementService{
		logger:   logger,
		config:   config,
		keyStore: &KeyStore{
			keys:    make(map[string]*ManagedKey),
			indices: make(map[string][]string),
		},
		escrowVault: &EscrowVault{
			escrowShares: make(map[string]*EscrowShare),
			custodians:   make(map[string]*EscrowCustodian),
			policies:     make(map[string]*EscrowPolicy),
		},
		stopChan: make(chan struct{}),
		
		keyOperations: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "hsm_key_operations_total",
				Help: "Total number of HSM key operations",
			},
			[]string{"operation", "key_type", "result"},
		),
		
		keyRotations: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "hsm_key_rotations_total",
				Help: "Total number of key rotations",
			},
			[]string{"key_type", "result"},
		),
		
		escrowOperations: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "hsm_escrow_operations_total",
				Help: "Total number of key escrow operations",
			},
			[]string{"operation", "result"},
		),
		
		hsmLatency: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name: "hsm_operation_duration_seconds",
				Help: "Duration of HSM operations",
				Buckets: prometheus.DefBuckets,
			},
			[]string{"hsm_type", "operation"},
		),
		
		keyAgeGauge: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "hsm_key_age_days",
				Help: "Age of keys in days",
			},
			[]string{"key_id", "key_type"},
		),
	}
	
	// Initialize HSM providers
	if err := service.initializeHSMProviders(); err != nil {
		return nil, fmt.Errorf("failed to initialize HSM providers: %w", err)
	}
	
	// Initialize components
	service.keyRotator = NewKeyRotator(logger, config, service)
	service.keyAuditor = NewKeyAuditor(logger, config)
	service.complianceEngine = NewComplianceEngine(logger, config)
	service.auditLogger = NewSecurityAuditLogger(logger, config)
	
	// Load existing keys and escrow data
	if err := service.loadExistingKeys(); err != nil {
		logger.Warn("failed to load existing keys", zap.Error(err))
	}
	
	return service, nil
}

// Start begins the key management service
func (ekms *EnterpriseKeyManagementService) Start(ctx context.Context) error {
	ekms.mu.Lock()
	defer ekms.mu.Unlock()
	
	if ekms.running {
		return fmt.Errorf("key management service already running")
	}
	
	ekms.logger.Info("Starting enterprise key management service")
	
	// Connect to HSM providers
	if err := ekms.primaryHSM.Connect(ctx); err != nil {
		return fmt.Errorf("failed to connect to primary HSM: %w", err)
	}
	
	if ekms.backupHSM != nil {
		if err := ekms.backupHSM.Connect(ctx); err != nil {
			ekms.logger.Warn("failed to connect to backup HSM", zap.Error(err))
		}
	}
	
	// Start key rotator
	if err := ekms.keyRotator.Start(ctx); err != nil {
		return fmt.Errorf("failed to start key rotator: %w", err)
	}
	
	// Start key auditor
	if err := ekms.keyAuditor.Start(ctx); err != nil {
		return fmt.Errorf("failed to start key auditor: %w", err)
	}
	
	// Start monitoring loops
	go ekms.keyLifecycleLoop(ctx)
	go ekms.complianceMonitoringLoop(ctx)
	go ekms.metricsUpdateLoop(ctx)
	
	ekms.running = true
	ekms.logger.Info("Enterprise key management service started successfully")
	
	return nil
}

// Stop stops the key management service
func (ekms *EnterpriseKeyManagementService) Stop() error {
	ekms.mu.Lock()
	defer ekms.mu.Unlock()
	
	if !ekms.running {
		return nil
	}
	
	ekms.logger.Info("Stopping enterprise key management service")
	
	close(ekms.stopChan)
	
	// Stop components
	if ekms.keyRotator != nil {
		ekms.keyRotator.Stop()
	}
	if ekms.keyAuditor != nil {
		ekms.keyAuditor.Stop()
	}
	
	// Disconnect from HSM providers
	if ekms.primaryHSM != nil {
		ekms.primaryHSM.Disconnect()
	}
	if ekms.backupHSM != nil {
		ekms.backupHSM.Disconnect()
	}
	
	ekms.running = false
	ekms.logger.Info("Enterprise key management service stopped")
	
	return nil
}

// CreateKey creates a new managed key with optional escrow
func (ekms *EnterpriseKeyManagementService) CreateKey(ctx context.Context, req *CreateKeyRequest) (*ManagedKey, error) {
	start := time.Now()
	defer func() {
		ekms.hsmLatency.WithLabelValues("primary", "create_key").Observe(time.Since(start).Seconds())
	}()
	
	// Validate request
	if err := ekms.validateCreateKeyRequest(req); err != nil {
		ekms.keyOperations.WithLabelValues("create", string(req.KeyType), "error").Inc()
		return nil, fmt.Errorf("invalid create key request: %w", err)
	}
	
	// Generate key in primary HSM
	keyRef, err := ekms.primaryHSM.GenerateKey(req.KeyType, req.Algorithm, req.KeySize)
	if err != nil {
		ekms.keyOperations.WithLabelValues("create", string(req.KeyType), "error").Inc()
		return nil, fmt.Errorf("failed to generate key in primary HSM: %w", err)
	}
	
	// Create managed key object
	managedKey := &ManagedKey{
		ID:            req.KeyID,
		Name:          req.Name,
		Version:       1,
		Algorithm:     req.Algorithm,
		KeySize:       req.KeySize,
		KeyType:       req.KeyType,
		Usage:         req.Usage,
		PrimaryKeyRef: keyRef.ID,
		State:         KeyStateActive,
		CreatedAt:     time.Now(),
		ActivatedAt:   time.Now(),
		Owner:         req.Owner,
		Application:   req.Application,
		Environment:   req.Environment,
		Tags:          req.Tags,
		Metadata:      req.Metadata,
		AuditTrail:    []KeyAuditEntry{},
		ComplianceLevel: req.ComplianceLevel,
		Certifications:  req.Certifications,
	}
	
	// Set expiration if specified
	if req.ExpirationPeriod > 0 {
		expiresAt := time.Now().Add(req.ExpirationPeriod)
		managedKey.ExpiresAt = expiresAt
	}
	
	// Create backup in secondary HSM if available
	if ekms.backupHSM != nil {
		backupKeyRef, err := ekms.createBackupKey(keyRef, req)
		if err != nil {
			ekms.logger.Warn("failed to create backup key", zap.Error(err))
		} else {
			managedKey.BackupKeyRef = backupKeyRef.ID
		}
	}
	
	// Create escrow shares if enabled
	if req.EnableEscrow || ekms.config.EnableKeyEscrow {
		if err := ekms.createEscrowShares(managedKey, keyRef); err != nil {
			ekms.logger.Error("failed to create escrow shares", zap.Error(err))
			// Continue without escrow - log the failure
		} else {
			managedKey.EscrowEnabled = true
		}
	}
	
	// Store managed key
	ekms.keyStore.mu.Lock()
	ekms.keyStore.keys[managedKey.ID] = managedKey
	ekms.keyStore.mu.Unlock()
	
	// Add audit entry
	auditEntry := KeyAuditEntry{
		Timestamp: time.Now(),
		Operation: "create",
		User:      req.Owner,
		Source:    "key-management-service",
		Result:    "success",
		Details: map[string]interface{}{
			"key_id":        managedKey.ID,
			"key_type":      string(managedKey.KeyType),
			"algorithm":     managedKey.Algorithm,
			"key_size":      managedKey.KeySize,
			"escrow_enabled": managedKey.EscrowEnabled,
		},
	}
	managedKey.AuditTrail = append(managedKey.AuditTrail, auditEntry)
	
	// Send to audit logger
	ekms.auditLogger.LogKeyOperation(auditEntry)
	
	// Update metrics
	ekms.keyOperations.WithLabelValues("create", string(req.KeyType), "success").Inc()
	ekms.keyAgeGauge.WithLabelValues(managedKey.ID, string(managedKey.KeyType)).Set(0)
	
	ekms.logger.Info("created new managed key",
		zap.String("key_id", managedKey.ID),
		zap.String("key_type", string(managedKey.KeyType)),
		zap.String("algorithm", managedKey.Algorithm),
		zap.Bool("escrow_enabled", managedKey.EscrowEnabled))
	
	return managedKey, nil
}

// CreateKeyRequest represents a key creation request
type CreateKeyRequest struct {
	KeyID             string                 `json:"key_id"`
	Name              string                 `json:"name"`
	KeyType           KeyType                `json:"key_type"`
	Algorithm         string                 `json:"algorithm"`
	KeySize           int                    `json:"key_size"`
	Usage             []KeyUsage             `json:"usage"`
	Owner             string                 `json:"owner"`
	Application       string                 `json:"application"`
	Environment       string                 `json:"environment"`
	Tags              []string               `json:"tags"`
	Metadata          map[string]interface{} `json:"metadata"`
	EnableEscrow      bool                   `json:"enable_escrow"`
	ExpirationPeriod  time.Duration          `json:"expiration_period"`
	ComplianceLevel   string                 `json:"compliance_level"`
	Certifications    []string               `json:"certifications"`
}

// RecoverKey recovers a key from escrow using custodian shares
func (ekms *EnterpriseKeyManagementService) RecoverKey(ctx context.Context, keyID string, custodianShares []*CustodianShare) (*ManagedKey, error) {
	start := time.Now()
	defer func() {
		ekms.hsmLatency.WithLabelValues("primary", "recover_key").Observe(time.Since(start).Seconds())
	}()
	
	// Get key information
	managedKey, exists := ekms.keyStore.keys[keyID]
	if !exists {
		ekms.escrowOperations.WithLabelValues("recover", "error").Inc()
		return nil, fmt.Errorf("key not found: %s", keyID)
	}
	
	if !managedKey.EscrowEnabled {
		ekms.escrowOperations.WithLabelValues("recover", "error").Inc()
		return nil, fmt.Errorf("key escrow not enabled for key: %s", keyID)
	}
	
	// Validate custodian shares
	if err := ekms.validateCustodianShares(managedKey, custodianShares); err != nil {
		ekms.escrowOperations.WithLabelValues("recover", "error").Inc()
		return nil, fmt.Errorf("invalid custodian shares: %w", err)
	}
	
	// Reconstruct key from shares using Shamir's Secret Sharing
	keyMaterial, err := ekms.reconstructKeyFromShares(managedKey, custodianShares)
	if err != nil {
		ekms.escrowOperations.WithLabelValues("recover", "error").Inc()
		return nil, fmt.Errorf("failed to reconstruct key: %w", err)
	}
	
	// Import recovered key into HSM
	keyRef, err := ekms.primaryHSM.ImportKey(keyMaterial, managedKey.KeyType, managedKey.Algorithm)
	if err != nil {
		ekms.escrowOperations.WithLabelValues("recover", "error").Inc()
		return nil, fmt.Errorf("failed to import recovered key: %w", err)
	}
	
	// Update managed key
	managedKey.PrimaryKeyRef = keyRef.ID
	managedKey.State = KeyStateActive
	managedKey.ActivatedAt = time.Now()
	
	// Add audit entry
	auditEntry := KeyAuditEntry{
		Timestamp: time.Now(),
		Operation: "recover",
		User:      "system", // This would come from context in real implementation
		Source:    "key-recovery-service",
		Result:    "success",
		Details: map[string]interface{}{
			"key_id":           managedKey.ID,
			"custodian_count":  len(custodianShares),
			"recovery_method":  "shamir_secret_sharing",
		},
	}
	managedKey.AuditTrail = append(managedKey.AuditTrail, auditEntry)
	
	// Send to audit logger
	ekms.auditLogger.LogKeyOperation(auditEntry)
	
	// Update metrics
	ekms.escrowOperations.WithLabelValues("recover", "success").Inc()
	
	ekms.logger.Info("successfully recovered key from escrow",
		zap.String("key_id", keyID),
		zap.Int("custodian_shares", len(custodianShares)))
	
	// Clear sensitive key material
	for i := range keyMaterial {
		keyMaterial[i] = 0
	}
	
	return managedKey, nil
}

// CustodianShare represents a custodian's key share for recovery
type CustodianShare struct {
	CustodianID string `json:"custodian_id"`
	ShareIndex  int    `json:"share_index"`
	ShareData   []byte `json:"share_data"`
	Signature   []byte `json:"signature"`
}

// Additional methods would continue here...
// Including: initializeHSMProviders, validateCreateKeyRequest, createEscrowShares, etc.