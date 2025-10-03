package devices

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
	"go.uber.org/zap"
)

// DeviceLifecycleManager manages the complete device lifecycle
type DeviceLifecycleManager struct {
	logger       *zap.Logger
	config       *LifecycleConfig
	devices      map[string]*Device
	enrollments  map[string]*EnrollmentSession
	certificates map[string]*DeviceCertificate
	policies     map[string]*DevicePolicy
	mu           sync.RWMutex
	attestor     *AttestationVerifier
	certManager  *CertificateManager
	policyEngine *PolicyEngine
}

// LifecycleConfig holds device lifecycle configuration
type LifecycleConfig struct {
	// Enrollment configuration
	EnrollmentTimeout     time.Duration `json:"enrollment_timeout"`
	RequireTPMAttestation bool          `json:"require_tpm_attestation"`
	RequireSecureBoot     bool          `json:"require_secure_boot"`
	AllowedManufacturers  []string      `json:"allowed_manufacturers"`
	TrustedRootCAPath     string        `json:"trusted_root_ca_path"`
	TrustedEKCertsPath    string        `json:"trusted_ek_certs_path"`
	PCRBaselinePath       string        `json:"pcr_baseline_path"`

	// Certificate configuration
	CertificateValidity    time.Duration `json:"certificate_validity"`
	CertificateRenewalTime time.Duration `json:"certificate_renewal_time"`
	RootCAPath             string        `json:"root_ca_path"`
	IntermediateCAPath     string        `json:"intermediate_ca_path"`

	// Policy configuration
	DefaultPolicySet       string        `json:"default_policy_set"`
	PolicyUpdateInterval   time.Duration `json:"policy_update_interval"`
	RequirePolicySignature bool          `json:"require_policy_signature"`

	// Health monitoring
	HealthCheckInterval   time.Duration `json:"health_check_interval"`
	HealthCheckTimeout    time.Duration `json:"health_check_timeout"`
	MaxMissedHealthChecks int           `json:"max_missed_health_checks"`

	// Compliance
	ComplianceCheckInterval time.Duration `json:"compliance_check_interval"`
	RequiredCompliance      []string      `json:"required_compliance"`

	// Operational
	MaxDevicesPerOrg      int           `json:"max_devices_per_org"`
	DeviceRetentionPeriod time.Duration `json:"device_retention_period"`
}

// Device represents a managed device
type Device struct {
	ID              string       `json:"id"`
	Name            string       `json:"name"`
	OrganizationID  string       `json:"organization_id"`
	DeviceType      DeviceType   `json:"device_type"`
	Status          DeviceStatus `json:"status"`
	Manufacturer    string       `json:"manufacturer"`
	Model           string       `json:"model"`
	SerialNumber    string       `json:"serial_number"`
	FirmwareVersion string       `json:"firmware_version"`
	HardwareVersion string       `json:"hardware_version"`

	// Identity and security
	PublicKey         []byte `json:"public_key"`
	CertificateID     string `json:"certificate_id"`
	TPMEndorsementKey []byte `json:"tpm_endorsement_key"`
	TPMAttestationKey []byte `json:"tpm_attestation_key"`

	// Lifecycle tracking
	EnrolledAt          time.Time `json:"enrolled_at"`
	LastSeen            time.Time `json:"last_seen"`
	LastHealthCheck     time.Time `json:"last_health_check"`
	LastPolicyUpdate    time.Time `json:"last_policy_update"`
	LastComplianceCheck time.Time `json:"last_compliance_check"`

	// Configuration
	Configuration map[string]interface{} `json:"configuration"`
	PolicySet     string                 `json:"policy_set"`
	Tags          []string               `json:"tags"`

	// Compliance and health
	ComplianceStatus ComplianceStatus `json:"compliance_status"`
	HealthStatus     HealthStatus     `json:"health_status"`
	Capabilities     []string         `json:"capabilities"`

	// Metadata
	Metadata  map[string]string `json:"metadata"`
	CreatedAt time.Time         `json:"created_at"`
	UpdatedAt time.Time         `json:"updated_at"`
}

// DeviceType represents the type of device
type DeviceType string

const (
	DeviceTypeGateway    DeviceType = "gateway"
	DeviceTypeEndpoint   DeviceType = "endpoint"
	DeviceTypeSensor     DeviceType = "sensor"
	DeviceTypeActuator   DeviceType = "actuator"
	DeviceTypeController DeviceType = "controller"
)

// DeviceStatus represents the current status of a device
type DeviceStatus string

const (
	DeviceStatusPending        DeviceStatus = "pending"
	DeviceStatusEnrolling      DeviceStatus = "enrolling"
	DeviceStatusActive         DeviceStatus = "active"
	DeviceStatusInactive       DeviceStatus = "inactive"
	DeviceStatusSuspended      DeviceStatus = "suspended"
	DeviceStatusDecommissioned DeviceStatus = "decommissioned"
	DeviceStatusError          DeviceStatus = "error"
)

// ComplianceStatus represents device compliance state
type ComplianceStatus string

const (
	ComplianceStatusCompliant    ComplianceStatus = "compliant"
	ComplianceStatusNonCompliant ComplianceStatus = "non_compliant"
	ComplianceStatusUnknown      ComplianceStatus = "unknown"
	ComplianceStatusChecking     ComplianceStatus = "checking"
)

// HealthStatus represents device health state
type HealthStatus string

const (
	HealthStatusHealthy   HealthStatus = "healthy"
	HealthStatusUnhealthy HealthStatus = "unhealthy"
	HealthStatusUnknown   HealthStatus = "unknown"
	HealthStatusChecking  HealthStatus = "checking"
)

// EnrollmentSession represents an active device enrollment
type EnrollmentSession struct {
	ID                 string                   `json:"id"`
	DeviceID           string                   `json:"device_id"`
	Challenge          []byte                   `json:"challenge"`
	Status             EnrollmentStatus         `json:"status"`
	CreatedAt          time.Time                `json:"created_at"`
	ExpiresAt          time.Time                `json:"expires_at"`
	AttestationData    *AttestationData         `json:"attestation_data,omitempty"`
	CertificateRequest *x509.CertificateRequest `json:"certificate_request,omitempty"`
	Metadata           map[string]string        `json:"metadata"`
}

// EnrollmentStatus represents enrollment session status
type EnrollmentStatus string

const (
	EnrollmentStatusPending    EnrollmentStatus = "pending"
	EnrollmentStatusChallenged EnrollmentStatus = "challenged"
	EnrollmentStatusVerifying  EnrollmentStatus = "verifying"
	EnrollmentStatusApproved   EnrollmentStatus = "approved"
	EnrollmentStatusRejected   EnrollmentStatus = "rejected"
	EnrollmentStatusExpired    EnrollmentStatus = "expired"
)

// AttestationData represents TPM attestation information
type AttestationData struct {
	Quote     []byte         `json:"quote"`
	Signature []byte         `json:"signature"`
	PCRValues map[int][]byte `json:"pcr_values"`
	EventLog  []byte         `json:"event_log"`
	EKCert    []byte         `json:"ek_cert"`
	AKCert    []byte         `json:"ak_cert"`
	Nonce     []byte         `json:"nonce"`
	Timestamp time.Time      `json:"timestamp"`
}

// DeviceCertificate represents a device certificate
type DeviceCertificate struct {
	ID               string            `json:"id"`
	DeviceID         string            `json:"device_id"`
	Certificate      []byte            `json:"certificate"`
	PrivateKey       []byte            `json:"private_key,omitempty"`
	SerialNumber     string            `json:"serial_number"`
	Subject          string            `json:"subject"`
	Issuer           string            `json:"issuer"`
	NotBefore        time.Time         `json:"not_before"`
	NotAfter         time.Time         `json:"not_after"`
	KeyUsage         []string          `json:"key_usage"`
	Status           CertificateStatus `json:"status"`
	CreatedAt        time.Time         `json:"created_at"`
	RevokedAt        *time.Time        `json:"revoked_at,omitempty"`
	RevocationReason string            `json:"revocation_reason,omitempty"`
	RenewedFrom      string            `json:"renewed_from,omitempty"`
}

// CertificateStatus represents certificate status
type CertificateStatus string

const (
	CertificateStatusActive    CertificateStatus = "active"
	CertificateStatusExpired   CertificateStatus = "expired"
	CertificateStatusRevoked   CertificateStatus = "revoked"
	CertificateStatusSuspended CertificateStatus = "suspended"
)

// DevicePolicy represents device-specific policies
type DevicePolicy struct {
	ID        string                 `json:"id"`
	DeviceID  string                 `json:"device_id"`
	PolicySet string                 `json:"policy_set"`
	Version   string                 `json:"version"`
	Policies  map[string]interface{} `json:"policies"`
	Signature string                 `json:"signature"`
	CreatedAt time.Time              `json:"created_at"`
	UpdatedAt time.Time              `json:"updated_at"`
	AppliedAt *time.Time             `json:"applied_at,omitempty"`
}

// NewDeviceLifecycleManager creates a new device lifecycle manager
func NewDeviceLifecycleManager(logger *zap.Logger, config *LifecycleConfig) (*DeviceLifecycleManager, error) {
	if config == nil {
		config = DefaultLifecycleConfig()
	}

	attestor, err := NewAttestationVerifier(logger, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create attestation verifier: %w", err)
	}

	certManager, err := NewCertificateManager(logger, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create certificate manager: %w", err)
	}

	policyEngine, err := NewPolicyEngine(logger, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create policy engine: %w", err)
	}

	dlm := &DeviceLifecycleManager{
		logger:       logger,
		config:       config,
		devices:      make(map[string]*Device),
		enrollments:  make(map[string]*EnrollmentSession),
		certificates: make(map[string]*DeviceCertificate),
		policies:     make(map[string]*DevicePolicy),
		attestor:     attestor,
		certManager:  certManager,
		policyEngine: policyEngine,
	}

	// Start background tasks
	go dlm.startHealthCheckTask()
	go dlm.startComplianceCheckTask()
	go dlm.startCertificateRenewalTask()
	go dlm.startCleanupTask()

	logger.Info("device lifecycle manager initialized",
		zap.Duration("enrollment_timeout", config.EnrollmentTimeout),
		zap.Bool("require_tpm_attestation", config.RequireTPMAttestation),
		zap.Duration("certificate_validity", config.CertificateValidity))

	return dlm, nil
}

// DefaultLifecycleConfig returns default lifecycle configuration
func DefaultLifecycleConfig() *LifecycleConfig {
	return &LifecycleConfig{
		EnrollmentTimeout:       time.Minute * 10,
		RequireTPMAttestation:   true,
		RequireSecureBoot:       true,
		AllowedManufacturers:    []string{"QSLB", "TrustedVendor"},
		TrustedRootCAPath:       "",
		TrustedEKCertsPath:      "",
		PCRBaselinePath:         "",
		CertificateValidity:     time.Hour * 24 * 365, // 1 year
		CertificateRenewalTime:  time.Hour * 24 * 30,  // 30 days before expiry
		DefaultPolicySet:        "default",
		PolicyUpdateInterval:    time.Hour * 24,
		RequirePolicySignature:  true,
		HealthCheckInterval:     time.Minute * 5,
		HealthCheckTimeout:      time.Second * 30,
		MaxMissedHealthChecks:   3,
		ComplianceCheckInterval: time.Hour * 24,
		RequiredCompliance:      []string{"secure_boot", "tpm_enabled", "firmware_verified"},
		MaxDevicesPerOrg:        10000,
		DeviceRetentionPeriod:   time.Hour * 24 * 365 * 7, // 7 years
	}
}

// StartEnrollment initiates device enrollment process
func (dlm *DeviceLifecycleManager) StartEnrollment(ctx context.Context, req *EnrollmentRequest) (*EnrollmentSession, error) {
	dlm.mu.Lock()
	defer dlm.mu.Unlock()

	// Validate enrollment request
	if err := dlm.validateEnrollmentRequest(req); err != nil {
		return nil, fmt.Errorf("invalid enrollment request: %w", err)
	}

	// Check organization device limits
	if err := dlm.checkDeviceLimits(req.OrganizationID); err != nil {
		return nil, fmt.Errorf("device limit exceeded: %w", err)
	}

	// Generate enrollment session
	sessionID := uuid.New().String()
	challenge := make([]byte, 32)
	if _, err := rand.Read(challenge); err != nil {
		return nil, fmt.Errorf("failed to generate challenge: %w", err)
	}

	session := &EnrollmentSession{
		ID:        sessionID,
		DeviceID:  req.DeviceID,
		Challenge: challenge,
		Status:    EnrollmentStatusChallenged,
		CreatedAt: time.Now(),
		ExpiresAt: time.Now().Add(dlm.config.EnrollmentTimeout),
		Metadata:  req.Metadata,
	}

	dlm.enrollments[sessionID] = session

	// Create pending device record
	device := &Device{
		ID:               req.DeviceID,
		Name:             req.DeviceName,
		OrganizationID:   req.OrganizationID,
		DeviceType:       req.DeviceType,
		Status:           DeviceStatusEnrolling,
		Manufacturer:     req.Manufacturer,
		Model:            req.Model,
		SerialNumber:     req.SerialNumber,
		FirmwareVersion:  req.FirmwareVersion,
		HardwareVersion:  req.HardwareVersion,
		Configuration:    req.Configuration,
		Tags:             req.Tags,
		ComplianceStatus: ComplianceStatusUnknown,
		HealthStatus:     HealthStatusUnknown,
		Capabilities:     req.Capabilities,
		Metadata:         req.Metadata,
		CreatedAt:        time.Now(),
		UpdatedAt:        time.Now(),
	}

	dlm.devices[req.DeviceID] = device

	dlm.logger.Info("enrollment session started",
		zap.String("session_id", sessionID),
		zap.String("device_id", req.DeviceID),
		zap.String("organization_id", req.OrganizationID))

	return session, nil
}

// SubmitAttestation processes device attestation data
func (dlm *DeviceLifecycleManager) SubmitAttestation(ctx context.Context, sessionID string, attestation *AttestationData) error {
	dlm.mu.Lock()
	defer dlm.mu.Unlock()

	session, exists := dlm.enrollments[sessionID]
	if !exists {
		return fmt.Errorf("enrollment session not found")
	}

	if session.Status != EnrollmentStatusChallenged {
		return fmt.Errorf("invalid session status: %s", session.Status)
	}

	if time.Now().After(session.ExpiresAt) {
		session.Status = EnrollmentStatusExpired
		return fmt.Errorf("enrollment session expired")
	}

	// Verify attestation
	session.Status = EnrollmentStatusVerifying
	session.AttestationData = attestation

	if dlm.config.RequireTPMAttestation {
		if err := dlm.attestor.VerifyAttestation(ctx, attestation, session.Challenge); err != nil {
			session.Status = EnrollmentStatusRejected
			dlm.logger.Warn("attestation verification failed",
				zap.String("session_id", sessionID),
				zap.Error(err))
			return fmt.Errorf("attestation verification failed: %w", err)
		}
	}

	// Verify compliance requirements
	if err := dlm.verifyComplianceRequirements(attestation); err != nil {
		session.Status = EnrollmentStatusRejected
		dlm.logger.Warn("compliance verification failed",
			zap.String("session_id", sessionID),
			zap.Error(err))
		return fmt.Errorf("compliance verification failed: %w", err)
	}

	session.Status = EnrollmentStatusApproved

	dlm.logger.Info("attestation verified successfully",
		zap.String("session_id", sessionID),
		zap.String("device_id", session.DeviceID))

	return nil
}

// CompleteEnrollment finalizes device enrollment
func (dlm *DeviceLifecycleManager) CompleteEnrollment(ctx context.Context, sessionID string, csr *x509.CertificateRequest) (*DeviceCertificate, error) {
	dlm.mu.Lock()
	defer dlm.mu.Unlock()

	if csr == nil {
		return nil, fmt.Errorf("certificate signing request is required")
	}

	if err := csr.CheckSignature(); err != nil {
		return nil, fmt.Errorf("invalid CSR signature: %w", err)
	}

	session, exists := dlm.enrollments[sessionID]
	if !exists {
		return nil, fmt.Errorf("enrollment session not found")
	}

	if session.Status != EnrollmentStatusApproved {
		return nil, fmt.Errorf("enrollment not approved")
	}

	device, exists := dlm.devices[session.DeviceID]
	if !exists {
		return nil, fmt.Errorf("device not found")
	}

	session.CertificateRequest = csr

	// Generate device certificate
	cert, err := dlm.certManager.IssueCertificate(ctx, csr, device)
	if err != nil {
		return nil, fmt.Errorf("failed to issue certificate: %w", err)
	}

	// Update device status
	device.Status = DeviceStatusActive
	device.CertificateID = cert.ID
	switch pk := csr.PublicKey.(type) {
	case *rsa.PublicKey:
		device.PublicKey = pk.N.Bytes()
	default:
		return nil, fmt.Errorf("unsupported public key type %T", csr.PublicKey)
	}
	device.EnrolledAt = time.Now()
	device.LastSeen = time.Now()
	device.UpdatedAt = time.Now()

	if session.AttestationData != nil {
		device.TPMEndorsementKey = session.AttestationData.EKCert
		device.TPMAttestationKey = session.AttestationData.AKCert
	}

	// Store certificate
	dlm.certificates[cert.ID] = cert

	// Apply default policies
	if err := dlm.applyDefaultPolicies(ctx, device); err != nil {
		dlm.logger.Warn("failed to apply default policies",
			zap.String("device_id", device.ID),
			zap.Error(err))
	}

	// Clean up enrollment session
	delete(dlm.enrollments, sessionID)

	dlm.logger.Info("device enrollment completed",
		zap.String("device_id", device.ID),
		zap.String("certificate_id", cert.ID),
		zap.String("organization_id", device.OrganizationID))

	return cert, nil
}

// GetEnrollmentSession retrieves enrollment session details
func (dlm *DeviceLifecycleManager) GetEnrollmentSession(sessionID string) (*EnrollmentSession, error) {
	dlm.mu.RLock()
	defer dlm.mu.RUnlock()

	session, exists := dlm.enrollments[sessionID]
	if !exists {
		return nil, fmt.Errorf("enrollment session not found")
	}

	copy := *session
	return &copy, nil
}

// ApproveEnrollment marks an enrollment session as approved
func (dlm *DeviceLifecycleManager) ApproveEnrollment(ctx context.Context, sessionID string) (*Device, error) {
	dlm.mu.Lock()
	defer dlm.mu.Unlock()

	session, exists := dlm.enrollments[sessionID]
	if !exists {
		return nil, fmt.Errorf("enrollment session not found")
	}

	if session.Status == EnrollmentStatusExpired {
		return nil, fmt.Errorf("enrollment session expired")
	}

	session.Status = EnrollmentStatusApproved
	if session.Metadata == nil {
		session.Metadata = make(map[string]string)
	}
	session.Metadata["approved_at"] = time.Now().Format(time.RFC3339)

	device, exists := dlm.devices[session.DeviceID]
	if !exists {
		return nil, fmt.Errorf("device not found")
	}

	device.Status = DeviceStatusActive
	device.UpdatedAt = time.Now()
	if device.Metadata == nil {
		device.Metadata = make(map[string]string)
	}
	device.Metadata["last_approved_session"] = sessionID

	return device, nil
}

// RejectEnrollment rejects an enrollment session and updates device state
func (dlm *DeviceLifecycleManager) RejectEnrollment(ctx context.Context, sessionID string, reason string) error {
	dlm.mu.Lock()
	defer dlm.mu.Unlock()

	session, exists := dlm.enrollments[sessionID]
	if !exists {
		return fmt.Errorf("enrollment session not found")
	}

	session.Status = EnrollmentStatusRejected
	if session.Metadata == nil {
		session.Metadata = make(map[string]string)
	}
	session.Metadata["rejection_reason"] = reason
	session.Metadata["rejected_at"] = time.Now().Format(time.RFC3339)

	if device, exists := dlm.devices[session.DeviceID]; exists {
		device.Status = DeviceStatusError
		device.UpdatedAt = time.Now()
		if device.Metadata == nil {
			device.Metadata = make(map[string]string)
		}
		device.Metadata["rejection_reason"] = reason
	}

	dlm.logger.Info("enrollment session rejected",
		zap.String("session_id", sessionID),
		zap.String("device_id", session.DeviceID),
		zap.String("reason", reason))

	return nil
}

// ListAllDevices returns devices matching optional filters
func (dlm *DeviceLifecycleManager) ListAllDevices(filters *DeviceFilters) ([]*Device, error) {
	dlm.mu.RLock()
	defer dlm.mu.RUnlock()

	var items []*Device
	for _, device := range dlm.devices {
		if filters == nil || dlm.matchesFilters(device, filters) {
			items = append(items, device)
		}
	}

	return items, nil
}

// GetDeviceCertificate returns the active certificate for a device
func (dlm *DeviceLifecycleManager) GetDeviceCertificate(deviceID string) (*DeviceCertificate, error) {
	dlm.mu.RLock()
	defer dlm.mu.RUnlock()

	device, exists := dlm.devices[deviceID]
	if !exists {
		return nil, fmt.Errorf("device not found")
	}

	if device.CertificateID == "" {
		return nil, fmt.Errorf("device has no certificate")
	}

	cert, exists := dlm.certificates[device.CertificateID]
	if !exists {
		return nil, fmt.Errorf("certificate not found")
	}

	return cert, nil
}

// RenewDeviceCertificate renews a device certificate using the provided CSR
func (dlm *DeviceLifecycleManager) RenewDeviceCertificate(ctx context.Context, deviceID string, csr *x509.CertificateRequest) (*DeviceCertificate, error) {
	dlm.mu.Lock()
	defer dlm.mu.Unlock()

	device, exists := dlm.devices[deviceID]
	if !exists {
		return nil, fmt.Errorf("device not found")
	}

	if device.CertificateID == "" {
		return nil, fmt.Errorf("device has no certificate to renew")
	}

	cert, exists := dlm.certificates[device.CertificateID]
	if !exists {
		return nil, fmt.Errorf("certificate not found")
	}

	renewed, err := dlm.certManager.RenewCertificate(ctx, cert.ID, csr)
	if err != nil {
		return nil, err
	}

	dlm.certificates[renewed.ID] = renewed
	device.CertificateID = renewed.ID
	if key, ok := csr.PublicKey.(*rsa.PublicKey); ok {
		device.PublicKey = key.N.Bytes()
	}
	device.UpdatedAt = time.Now()

	return renewed, nil
}

// RevokeDeviceCertificate revokes a device certificate and updates device metadata
func (dlm *DeviceLifecycleManager) RevokeDeviceCertificate(ctx context.Context, deviceID, reason string) error {
	dlm.mu.Lock()
	defer dlm.mu.Unlock()

	device, exists := dlm.devices[deviceID]
	if !exists {
		return fmt.Errorf("device not found")
	}

	if device.CertificateID == "" {
		return fmt.Errorf("device has no certificate to revoke")
	}

	cert, exists := dlm.certificates[device.CertificateID]
	if !exists {
		return fmt.Errorf("certificate not found")
	}

	if err := dlm.certManager.RevokeCertificate(ctx, cert, reason); err != nil {
		return err
	}

	now := time.Now()
	cert.Status = CertificateStatusRevoked
	cert.RevokedAt = &now
	cert.RevocationReason = reason
	device.Status = DeviceStatusInactive
	device.UpdatedAt = now
	return nil
}

// GetDevice retrieves device information
func (dlm *DeviceLifecycleManager) GetDevice(deviceID string) (*Device, error) {
	dlm.mu.RLock()
	defer dlm.mu.RUnlock()

	device, exists := dlm.devices[deviceID]
	if !exists {
		return nil, fmt.Errorf("device not found")
	}

	return device, nil
}

// ListDevices returns devices for an organization
func (dlm *DeviceLifecycleManager) ListDevices(organizationID string, filters *DeviceFilters) ([]*Device, error) {
	dlm.mu.RLock()
	defer dlm.mu.RUnlock()

	var devices []*Device
	for _, device := range dlm.devices {
		if device.OrganizationID == organizationID {
			if filters == nil || dlm.matchesFilters(device, filters) {
				devices = append(devices, device)
			}
		}
	}

	return devices, nil
}

// UpdateDeviceConfiguration updates device configuration
func (dlm *DeviceLifecycleManager) UpdateDeviceConfiguration(ctx context.Context, deviceID string, config map[string]interface{}) error {
	dlm.mu.Lock()
	defer dlm.mu.Unlock()

	device, exists := dlm.devices[deviceID]
	if !exists {
		return fmt.Errorf("device not found")
	}

	device.Configuration = config
	device.UpdatedAt = time.Now()

	dlm.logger.Info("device configuration updated",
		zap.String("device_id", deviceID))

	return nil
}

// SuspendDevice suspends a device
func (dlm *DeviceLifecycleManager) SuspendDevice(ctx context.Context, deviceID string, reason string) error {
	dlm.mu.Lock()
	defer dlm.mu.Unlock()

	device, exists := dlm.devices[deviceID]
	if !exists {
		return fmt.Errorf("device not found")
	}

	device.Status = DeviceStatusSuspended
	device.UpdatedAt = time.Now()
	device.Metadata["suspension_reason"] = reason
	device.Metadata["suspended_at"] = time.Now().Format(time.RFC3339)

	dlm.logger.Info("device suspended",
		zap.String("device_id", deviceID),
		zap.String("reason", reason))

	return nil
}

// ReactivateDevice reactivates a suspended device
func (dlm *DeviceLifecycleManager) ReactivateDevice(ctx context.Context, deviceID string) error {
	dlm.mu.Lock()
	defer dlm.mu.Unlock()

	device, exists := dlm.devices[deviceID]
	if !exists {
		return fmt.Errorf("device not found")
	}

	if device.Status != DeviceStatusSuspended {
		return fmt.Errorf("device is not suspended")
	}

	device.Status = DeviceStatusActive
	device.UpdatedAt = time.Now()
	delete(device.Metadata, "suspension_reason")
	delete(device.Metadata, "suspended_at")
	device.Metadata["reactivated_at"] = time.Now().Format(time.RFC3339)

	dlm.logger.Info("device reactivated",
		zap.String("device_id", deviceID))

	return nil
}

// DecommissionDevice decommissions a device
func (dlm *DeviceLifecycleManager) DecommissionDevice(ctx context.Context, deviceID string, reason string) error {
	dlm.mu.Lock()
	defer dlm.mu.Unlock()

	device, exists := dlm.devices[deviceID]
	if !exists {
		return fmt.Errorf("device not found")
	}

	// Revoke certificate
	if device.CertificateID != "" {
		if cert, exists := dlm.certificates[device.CertificateID]; exists {
			if err := dlm.certManager.RevokeCertificate(ctx, cert, reason); err != nil {
				dlm.logger.Warn("failed to revoke certificate",
					zap.String("device_id", deviceID),
					zap.String("certificate_id", device.CertificateID),
					zap.Error(err))
			} else {
				now := time.Now()
				cert.Status = CertificateStatusRevoked
				cert.RevokedAt = &now
				cert.RevocationReason = reason
			}
		} else {
			dlm.logger.Warn("device certificate not found in registry",
				zap.String("device_id", deviceID),
				zap.String("certificate_id", device.CertificateID))
		}
	}

	device.Status = DeviceStatusDecommissioned
	device.UpdatedAt = time.Now()
	device.Metadata["decommission_reason"] = reason
	device.Metadata["decommissioned_at"] = time.Now().Format(time.RFC3339)

	dlm.logger.Info("device decommissioned",
		zap.String("device_id", deviceID),
		zap.String("reason", reason))

	return nil
}

// Background task implementations would go here...
// (startHealthCheckTask, startComplianceCheckTask, etc.)

// Helper types and methods

// EnrollmentRequest represents a device enrollment request
type EnrollmentRequest struct {
	DeviceID        string                 `json:"device_id"`
	DeviceName      string                 `json:"device_name"`
	OrganizationID  string                 `json:"organization_id"`
	DeviceType      DeviceType             `json:"device_type"`
	Manufacturer    string                 `json:"manufacturer"`
	Model           string                 `json:"model"`
	SerialNumber    string                 `json:"serial_number"`
	FirmwareVersion string                 `json:"firmware_version"`
	HardwareVersion string                 `json:"hardware_version"`
	Configuration   map[string]interface{} `json:"configuration"`
	Tags            []string               `json:"tags"`
	Capabilities    []string               `json:"capabilities"`
	Metadata        map[string]string      `json:"metadata"`
}

// DeviceFilters represents device filtering options
type DeviceFilters struct {
	Status           []DeviceStatus     `json:"status,omitempty"`
	DeviceType       []DeviceType       `json:"device_type,omitempty"`
	Manufacturer     []string           `json:"manufacturer,omitempty"`
	Tags             []string           `json:"tags,omitempty"`
	ComplianceStatus []ComplianceStatus `json:"compliance_status,omitempty"`
	HealthStatus     []HealthStatus     `json:"health_status,omitempty"`
}

// Placeholder implementations for helper functions
func (dlm *DeviceLifecycleManager) validateEnrollmentRequest(req *EnrollmentRequest) error {
	if req.DeviceID == "" {
		return fmt.Errorf("device ID is required")
	}
	if req.OrganizationID == "" {
		return fmt.Errorf("organization ID is required")
	}
	// Add more validation as needed
	return nil
}

func (dlm *DeviceLifecycleManager) checkDeviceLimits(orgID string) error {
	count := 0
	for _, device := range dlm.devices {
		if device.OrganizationID == orgID && device.Status != DeviceStatusDecommissioned {
			count++
		}
	}
	if count >= dlm.config.MaxDevicesPerOrg {
		return fmt.Errorf("maximum devices limit reached: %d", dlm.config.MaxDevicesPerOrg)
	}
	return nil
}

func (dlm *DeviceLifecycleManager) verifyComplianceRequirements(attestation *AttestationData) error {
	// Implement compliance verification logic
	return nil
}

func (dlm *DeviceLifecycleManager) applyDefaultPolicies(ctx context.Context, device *Device) error {
	// Implement default policy application
	return nil
}

func (dlm *DeviceLifecycleManager) matchesFilters(device *Device, filters *DeviceFilters) bool {
	// Implement filter matching logic
	return true
}

func (dlm *DeviceLifecycleManager) startHealthCheckTask() {
	// Implement health check background task
}

func (dlm *DeviceLifecycleManager) startComplianceCheckTask() {
	// Implement compliance check background task
}

func (dlm *DeviceLifecycleManager) startCertificateRenewalTask() {
	// Implement certificate renewal background task
}

func (dlm *DeviceLifecycleManager) startCleanupTask() {
	// Implement cleanup background task
}
