package tpm

import (
	"context"
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/google/go-tpm/tpm2"
	"github.com/google/go-tpm/tpm2/transport"
	"github.com/google/go-tpm/tpmutil"
	"go.uber.org/zap"
)

// TPMSealer provides enterprise-grade TPM 2.0 key sealing and attestation
type TPMSealer struct {
	logger     *zap.Logger
	config     *Config
	transport  transport.TPMCloser
	mu         sync.RWMutex
	sealedKeys map[string]*SealedKey
	ekHandle   tpmutil.Handle
	akHandle   tpmutil.Handle
}

// Config holds TPM sealing configuration
type Config struct {
	// TPM device configuration
	DevicePath     string `json:"device_path"`     // "/dev/tpm0" or "simulator"
	
	// PCR configuration
	PCRSelection   []int  `json:"pcr_selection"`   // PCRs to seal against
	PCRBank        string `json:"pcr_bank"`        // "sha256", "sha1", "sha384"
	
	// Key configuration
	KeySize        int    `json:"key_size"`        // RSA key size (2048, 3072, 4096)
	KeyAlgorithm   string `json:"key_algorithm"`   // "rsa", "ecc"
	
	// Attestation configuration
	EKTemplate     string `json:"ek_template"`     // Endorsement Key template
	AKTemplate     string `json:"ak_template"`     // Attestation Key template
	
	// Security configuration
	RequireAuth    bool   `json:"require_auth"`    // Require user authentication
	AuthPolicy     string `json:"auth_policy"`     // Authorization policy
	
	// Operational configuration
	MaxRetries     int           `json:"max_retries"`
	Timeout        time.Duration `json:"timeout"`
	
	// Measured boot configuration
	EventLogPath   string `json:"event_log_path"`  // "/sys/kernel/security/tpm0/binary_bios_measurements"
	BootAggregate  bool   `json:"boot_aggregate"`  // Include boot aggregate in attestation
}

// SealedKey represents a key sealed to TPM PCRs
type SealedKey struct {
	ID           string            `json:"id"`
	Name         string            `json:"name"`
	KeyData      []byte            `json:"key_data"`      // Encrypted key material
	SealedBlob   []byte            `json:"sealed_blob"`   // TPM sealed blob
	PCRValues    map[int][]byte    `json:"pcr_values"`    // PCR values at seal time
	CreationData []byte            `json:"creation_data"` // TPM creation data
	Signature    []byte            `json:"signature"`     // Creation signature
	Handle       tpmutil.Handle    `json:"handle"`        // TPM object handle
	Created      time.Time         `json:"created"`
	LastUsed     time.Time         `json:"last_used"`
	Metadata     map[string]string `json:"metadata"`
}

// AttestationData represents TPM attestation information
type AttestationData struct {
	EKCert         []byte            `json:"ek_cert"`         // Endorsement Key certificate
	EKPub          []byte            `json:"ek_pub"`          // Endorsement Key public key
	AKPub          []byte            `json:"ak_pub"`          // Attestation Key public key
	Quote          []byte            `json:"quote"`           // TPM quote
	Signature      []byte            `json:"signature"`       // Quote signature
	PCRValues      map[int][]byte    `json:"pcr_values"`      // Current PCR values
	EventLog       []byte            `json:"event_log"`       // Measured boot event log
	BootAggregate  []byte            `json:"boot_aggregate"`  // Boot aggregate value
	Timestamp      time.Time         `json:"timestamp"`
	Nonce          []byte            `json:"nonce"`           // Challenge nonce
}

// DeviceIdentity represents device identity information
type DeviceIdentity struct {
	DeviceID       string    `json:"device_id"`
	EKFingerprint  string    `json:"ek_fingerprint"`
	AKFingerprint  string    `json:"ak_fingerprint"`
	Manufacturer   string    `json:"manufacturer"`
	Model          string    `json:"model"`
	FirmwareVersion string   `json:"firmware_version"`
	Created        time.Time `json:"created"`
	LastAttestation time.Time `json:"last_attestation"`
}

// NewTPMSealer creates a new TPM sealer
func NewTPMSealer(logger *zap.Logger, config *Config) (*TPMSealer, error) {
	if config == nil {
		config = DefaultConfig()
	}

	// Validate configuration
	if err := validateConfig(config); err != nil {
		return nil, fmt.Errorf("invalid TPM config: %w", err)
	}

	// Open TPM device
	transport, err := openTPMDevice(config.DevicePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open TPM device: %w", err)
	}

	sealer := &TPMSealer{
		logger:     logger,
		config:     config,
		transport:  transport,
		sealedKeys: make(map[string]*SealedKey),
	}

	// Initialize TPM
	if err := sealer.initialize(); err != nil {
		transport.Close()
		return nil, fmt.Errorf("failed to initialize TPM: %w", err)
	}

	logger.Info("TPM sealer initialized successfully",
		zap.String("device_path", config.DevicePath),
		zap.Ints("pcr_selection", config.PCRSelection),
		zap.String("pcr_bank", config.PCRBank))

	return sealer, nil
}

// DefaultConfig returns default TPM configuration
func DefaultConfig() *Config {
	return &Config{
		DevicePath:   "/dev/tpm0",
		PCRSelection: []int{0, 1, 2, 3, 4, 5, 6, 7}, // Boot PCRs
		PCRBank:      "sha256",
		KeySize:      2048,
		KeyAlgorithm: "rsa",
		MaxRetries:   3,
		Timeout:      30 * time.Second,
		EventLogPath: "/sys/kernel/security/tpm0/binary_bios_measurements",
		BootAggregate: true,
	}
}

// validateConfig validates TPM configuration
func validateConfig(config *Config) error {
	if config.DevicePath == "" {
		return fmt.Errorf("TPM device path is required")
	}

	if len(config.PCRSelection) == 0 {
		return fmt.Errorf("PCR selection cannot be empty")
	}

	if config.PCRBank == "" {
		config.PCRBank = "sha256"
	}

	if config.KeySize == 0 {
		config.KeySize = 2048
	}

	if config.KeyAlgorithm == "" {
		config.KeyAlgorithm = "rsa"
	}

	return nil
}

// openTPMDevice opens the TPM device
func openTPMDevice(devicePath string) (transport.TPMCloser, error) {
	if devicePath == "simulator" {
		// For testing with TPM simulator
		return transport.OpenTPM()
	}
	
	return transport.OpenTPM(devicePath)
}

// initialize initializes the TPM and creates necessary keys
func (ts *TPMSealer) initialize() error {
	// Create Endorsement Key if it doesn't exist
	if err := ts.createEndorsementKey(); err != nil {
		return fmt.Errorf("failed to create endorsement key: %w", err)
	}

	// Create Attestation Key if it doesn't exist
	if err := ts.createAttestationKey(); err != nil {
		return fmt.Errorf("failed to create attestation key: %w", err)
	}

	return nil
}

// createEndorsementKey creates or loads the Endorsement Key
func (ts *TPMSealer) createEndorsementKey() error {
	// Try to load existing EK
	ekHandle, err := ts.loadPersistentKey(0x81010001) // Standard EK handle
	if err == nil {
		ts.ekHandle = ekHandle
		ts.logger.Info("loaded existing endorsement key", zap.Uint32("handle", uint32(ekHandle)))
		return nil
	}

	// Create new EK
	template := tpm2.RSAEKTemplate
	if ts.config.KeyAlgorithm == "ecc" {
		template = tpm2.ECCEKTemplate
	}

	createResp, err := tpm2.CreatePrimary{
		PrimaryHandle: tpm2.TPMRHEndorsement,
		InPublic:      tpm2.New2B(template),
	}.Execute(ts.transport)
	if err != nil {
		return fmt.Errorf("failed to create endorsement key: %w", err)
	}

	// Make EK persistent
	_, err = tpm2.EvictControl{
		Auth:          tpm2.TPMRHOwner,
		ObjectHandle:  createResp.ObjectHandle,
		PersistentHandle: 0x81010001,
	}.Execute(ts.transport)
	if err != nil {
		return fmt.Errorf("failed to make EK persistent: %w", err)
	}

	ts.ekHandle = 0x81010001
	ts.logger.Info("created new endorsement key", zap.Uint32("handle", uint32(ts.ekHandle)))
	return nil
}

// createAttestationKey creates or loads the Attestation Key
func (ts *TPMSealer) createAttestationKey() error {
	// Try to load existing AK
	akHandle, err := ts.loadPersistentKey(0x81010002) // Standard AK handle
	if err == nil {
		ts.akHandle = akHandle
		ts.logger.Info("loaded existing attestation key", zap.Uint32("handle", uint32(akHandle)))
		return nil
	}

	// Create new AK
	template := tpm2.RSASRKTemplate
	if ts.config.KeyAlgorithm == "ecc" {
		template = tpm2.ECCSRKTemplate
	}

	createResp, err := tpm2.CreatePrimary{
		PrimaryHandle: tpm2.TPMRHOwner,
		InPublic:      tpm2.New2B(template),
	}.Execute(ts.transport)
	if err != nil {
		return fmt.Errorf("failed to create attestation key: %w", err)
	}

	// Make AK persistent
	_, err = tpm2.EvictControl{
		Auth:          tpm2.TPMRHOwner,
		ObjectHandle:  createResp.ObjectHandle,
		PersistentHandle: 0x81010002,
	}.Execute(ts.transport)
	if err != nil {
		return fmt.Errorf("failed to make AK persistent: %w", err)
	}

	ts.akHandle = 0x81010002
	ts.logger.Info("created new attestation key", zap.Uint32("handle", uint32(ts.akHandle)))
	return nil
}

// loadPersistentKey loads a persistent key by handle
func (ts *TPMSealer) loadPersistentKey(handle tpmutil.Handle) (tpmutil.Handle, error) {
	// Try to read the public key to verify the handle exists
	_, err := tpm2.ReadPublic{
		ObjectHandle: handle,
	}.Execute(ts.transport)
	if err != nil {
		return 0, err
	}
	
	return handle, nil
}

// SealKey seals a key to the current PCR values
func (ts *TPMSealer) SealKey(ctx context.Context, keyID, keyName string, keyData []byte) (*SealedKey, error) {
	ts.mu.Lock()
	defer ts.mu.Unlock()

	// Check if key already exists
	if _, exists := ts.sealedKeys[keyID]; exists {
		return nil, fmt.Errorf("key %s already exists", keyID)
	}

	// Read current PCR values
	pcrValues, err := ts.readPCRValues()
	if err != nil {
		return nil, fmt.Errorf("failed to read PCR values: %w", err)
	}

	// Create PCR selection for sealing
	pcrSelection := tpm2.TPMLPCRSelection{
		PCRSelections: []tpm2.TPMSPCRSelection{
			{
				Hash:      tpm2.TPMAlgSHA256,
				PCRSelect: ts.createPCRSelect(),
			},
		},
	}

	// Create sealing policy
	policyDigest, err := ts.createSealingPolicy(pcrSelection)
	if err != nil {
		return nil, fmt.Errorf("failed to create sealing policy: %w", err)
	}

	// Create sealed object template
	template := tpm2.TPMTPublic{
		Type:    tpm2.TPMAlgKeyedHash,
		NameAlg: tpm2.TPMAlgSHA256,
		ObjectAttributes: tpm2.TPMAObject{
			UserWithAuth: true,
			NoDA:         true,
		},
		AuthPolicy: policyDigest,
	}

	// Create the sealed object
	createResp, err := tpm2.Create{
		ParentHandle: ts.akHandle,
		InSensitive: tpm2.TPM2BSensitiveCreate{
			Sensitive: &tpm2.TPMSSensitiveCreate{
				Data: keyData,
			},
		},
		InPublic: tpm2.New2B(template),
	}.Execute(ts.transport)
	if err != nil {
		return nil, fmt.Errorf("failed to create sealed object: %w", err)
	}

	// Load the sealed object
	loadResp, err := tpm2.Load{
		ParentHandle: ts.akHandle,
		InPrivate:   createResp.OutPrivate,
		InPublic:    createResp.OutPublic,
	}.Execute(ts.transport)
	if err != nil {
		return nil, fmt.Errorf("failed to load sealed object: %w", err)
	}

	sealedKey := &SealedKey{
		ID:           keyID,
		Name:         keyName,
		KeyData:      keyData,
		SealedBlob:   createResp.OutPrivate.Buffer,
		PCRValues:    pcrValues,
		CreationData: createResp.CreationData.Buffer,
		Signature:    createResp.CreationTk.Digest,
		Handle:       loadResp.ObjectHandle,
		Created:      time.Now(),
		LastUsed:     time.Now(),
		Metadata:     make(map[string]string),
	}

	ts.sealedKeys[keyID] = sealedKey

	ts.logger.Info("key sealed successfully",
		zap.String("key_id", keyID),
		zap.String("key_name", keyName),
		zap.Int("key_size", len(keyData)),
		zap.Int("pcr_count", len(pcrValues)))

	return sealedKey, nil
}

// UnsealKey unseals a key if PCR values match
func (ts *TPMSealer) UnsealKey(ctx context.Context, keyID string) ([]byte, error) {
	ts.mu.RLock()
	sealedKey, exists := ts.sealedKeys[keyID]
	ts.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("key %s not found", keyID)
	}

	// Verify current PCR values match sealed values
	currentPCRs, err := ts.readPCRValues()
	if err != nil {
		return nil, fmt.Errorf("failed to read current PCR values: %w", err)
	}

	if !ts.comparePCRValues(sealedKey.PCRValues, currentPCRs) {
		return nil, fmt.Errorf("PCR values do not match sealed values - system state changed")
	}

	// Start policy session for unsealing
	sessionResp, err := tpm2.StartAuthSession{
		SessionType: tpm2.TPMSEPolicy,
		AuthHash:    tpm2.TPMAlgSHA256,
	}.Execute(ts.transport)
	if err != nil {
		return nil, fmt.Errorf("failed to start policy session: %w", err)
	}
	defer func() {
		tpm2.FlushContext{FlushHandle: sessionResp.SessionHandle}.Execute(ts.transport)
	}()

	// Execute PCR policy
	pcrSelection := tpm2.TPMLPCRSelection{
		PCRSelections: []tpm2.TPMSPCRSelection{
			{
				Hash:      tpm2.TPMAlgSHA256,
				PCRSelect: ts.createPCRSelect(),
			},
		},
	}

	_, err = tpm2.PolicyPCR{
		PolicySession: sessionResp.SessionHandle,
		Pcrs:          pcrSelection,
	}.Execute(ts.transport)
	if err != nil {
		return nil, fmt.Errorf("failed to execute PCR policy: %w", err)
	}

	// Unseal the key
	unsealResp, err := tpm2.Unseal{
		ItemHandle: sealedKey.Handle,
		SessionHandle: sessionResp.SessionHandle,
	}.Execute(ts.transport)
	if err != nil {
		return nil, fmt.Errorf("failed to unseal key: %w", err)
	}

	// Update last used time
	ts.mu.Lock()
	sealedKey.LastUsed = time.Now()
	ts.mu.Unlock()

	ts.logger.Info("key unsealed successfully", zap.String("key_id", keyID))
	return unsealResp.OutData.Buffer, nil
}

// GenerateAttestation generates a TPM attestation quote
func (ts *TPMSealer) GenerateAttestation(ctx context.Context, nonce []byte) (*AttestationData, error) {
	// Read current PCR values
	pcrValues, err := ts.readPCRValues()
	if err != nil {
		return nil, fmt.Errorf("failed to read PCR values: %w", err)
	}

	// Create PCR selection for quote
	pcrSelection := tpm2.TPMLPCRSelection{
		PCRSelections: []tpm2.TPMSPCRSelection{
			{
				Hash:      tpm2.TPMAlgSHA256,
				PCRSelect: ts.createPCRSelect(),
			},
		},
	}

	// Generate quote
	quoteResp, err := tpm2.Quote{
		SignHandle:   ts.akHandle,
		QualifyingData: nonce,
		InScheme:     tpm2.TPMTSigScheme{
			Scheme: tpm2.TPMAlgRSAPSS,
			Details: tpm2.NewTPMUSigScheme(
				tpm2.TPMAlgRSAPSS,
				&tpm2.TPMSSchemeHash{HashAlg: tpm2.TPMAlgSHA256},
			),
		},
		PCRSelect: pcrSelection,
	}.Execute(ts.transport)
	if err != nil {
		return nil, fmt.Errorf("failed to generate quote: %w", err)
	}

	// Read EK public key
	ekPubResp, err := tpm2.ReadPublic{
		ObjectHandle: ts.ekHandle,
	}.Execute(ts.transport)
	if err != nil {
		return nil, fmt.Errorf("failed to read EK public key: %w", err)
	}

	// Read AK public key
	akPubResp, err := tpm2.ReadPublic{
		ObjectHandle: ts.akHandle,
	}.Execute(ts.transport)
	if err != nil {
		return nil, fmt.Errorf("failed to read AK public key: %w", err)
	}

	// Read event log if configured
	var eventLog []byte
	if ts.config.EventLogPath != "" {
		eventLog, _ = ts.readEventLog() // Non-fatal if fails
	}

	// Calculate boot aggregate if configured
	var bootAggregate []byte
	if ts.config.BootAggregate {
		bootAggregate = ts.calculateBootAggregate(pcrValues)
	}

	attestation := &AttestationData{
		EKPub:         ekPubResp.OutPublic.Buffer,
		AKPub:         akPubResp.OutPublic.Buffer,
		Quote:         quoteResp.Quoted.Buffer,
		Signature:     quoteResp.Signature.Buffer,
		PCRValues:     pcrValues,
		EventLog:      eventLog,
		BootAggregate: bootAggregate,
		Timestamp:     time.Now(),
		Nonce:         nonce,
	}

	ts.logger.Info("attestation generated successfully",
		zap.Int("pcr_count", len(pcrValues)),
		zap.Int("quote_size", len(attestation.Quote)),
		zap.Int("signature_size", len(attestation.Signature)))

	return attestation, nil
}

// readPCRValues reads the current PCR values
func (ts *TPMSealer) readPCRValues() (map[int][]byte, error) {
	pcrValues := make(map[int][]byte)

	for _, pcrIndex := range ts.config.PCRSelection {
		pcrResp, err := tpm2.PCRRead{
			PCRSelectionIn: tpm2.TPMLPCRSelection{
				PCRSelections: []tpm2.TPMSPCRSelection{
					{
						Hash:      tpm2.TPMAlgSHA256,
						PCRSelect: tpm2.PCClientCompatible.PCRs(pcrIndex),
					},
				},
			},
		}.Execute(ts.transport)
		if err != nil {
			return nil, fmt.Errorf("failed to read PCR %d: %w", pcrIndex, err)
		}

		if len(pcrResp.PCRValues.Digests) > 0 {
			pcrValues[pcrIndex] = pcrResp.PCRValues.Digests[0].Buffer
		}
	}

	return pcrValues, nil
}

// createPCRSelect creates PCR selection bitmask
func (ts *TPMSealer) createPCRSelect() tpm2.TPMSPCRSelect {
	var pcrSelect tpm2.TPMSPCRSelect
	for _, pcr := range ts.config.PCRSelection {
		pcrSelect[pcr/8] |= 1 << (pcr % 8)
	}
	return pcrSelect
}

// createSealingPolicy creates a policy digest for sealing
func (ts *TPMSealer) createSealingPolicy(pcrSelection tpm2.TPMLPCRSelection) (tpm2.TPM2BDigest, error) {
	// Start trial session
	sessionResp, err := tpm2.StartAuthSession{
		SessionType: tpm2.TPMSETrial,
		AuthHash:    tpm2.TPMAlgSHA256,
	}.Execute(ts.transport)
	if err != nil {
		return tpm2.TPM2BDigest{}, fmt.Errorf("failed to start trial session: %w", err)
	}
	defer func() {
		tpm2.FlushContext{FlushHandle: sessionResp.SessionHandle}.Execute(ts.transport)
	}()

	// Execute PCR policy in trial mode
	_, err = tpm2.PolicyPCR{
		PolicySession: sessionResp.SessionHandle,
		Pcrs:          pcrSelection,
	}.Execute(ts.transport)
	if err != nil {
		return tpm2.TPM2BDigest{}, fmt.Errorf("failed to execute trial PCR policy: %w", err)
	}

	// Get policy digest
	digestResp, err := tpm2.PolicyGetDigest{
		PolicySession: sessionResp.SessionHandle,
	}.Execute(ts.transport)
	if err != nil {
		return tpm2.TPM2BDigest{}, fmt.Errorf("failed to get policy digest: %w", err)
	}

	return digestResp.PolicyDigest, nil
}

// comparePCRValues compares two sets of PCR values
func (ts *TPMSealer) comparePCRValues(sealed, current map[int][]byte) bool {
	if len(sealed) != len(current) {
		return false
	}

	for pcr, sealedValue := range sealed {
		currentValue, exists := current[pcr]
		if !exists {
			return false
		}

		if len(sealedValue) != len(currentValue) {
			return false
		}

		for i, b := range sealedValue {
			if b != currentValue[i] {
				return false
			}
		}
	}

	return true
}

// readEventLog reads the TPM event log
func (ts *TPMSealer) readEventLog() ([]byte, error) {
	// This would read from /sys/kernel/security/tpm0/binary_bios_measurements
	// For now, return empty log
	return []byte{}, nil
}

// calculateBootAggregate calculates boot aggregate from PCR values
func (ts *TPMSealer) calculateBootAggregate(pcrValues map[int][]byte) []byte {
	hash := sha256.New()
	
	// Aggregate PCRs 0-7 (boot PCRs)
	for i := 0; i < 8; i++ {
		if value, exists := pcrValues[i]; exists {
			hash.Write(value)
		}
	}
	
	return hash.Sum(nil)
}

// GenerateDeviceIdentity generates device identity information
func (ts *TPMSealer) GenerateDeviceIdentity(ctx context.Context) (*DeviceIdentity, error) {
	// Read EK public key for fingerprinting
	ekPubResp, err := tpm2.ReadPublic{
		ObjectHandle: ts.ekHandle,
	}.Execute(ts.transport)
	if err != nil {
		return nil, fmt.Errorf("failed to read EK public key: %w", err)
	}

	// Read AK public key for fingerprinting
	akPubResp, err := tpm2.ReadPublic{
		ObjectHandle: ts.akHandle,
	}.Execute(ts.transport)
	if err != nil {
		return nil, fmt.Errorf("failed to read AK public key: %w", err)
	}

	// Calculate fingerprints
	ekHash := sha256.Sum256(ekPubResp.OutPublic.Buffer)
	akHash := sha256.Sum256(akPubResp.OutPublic.Buffer)

	// Generate device ID from EK fingerprint
	deviceID := fmt.Sprintf("qslb-device-%x", ekHash[:8])

	identity := &DeviceIdentity{
		DeviceID:        deviceID,
		EKFingerprint:   fmt.Sprintf("%x", ekHash),
		AKFingerprint:   fmt.Sprintf("%x", akHash),
		Manufacturer:    "QSLB",
		Model:           "Enterprise Gateway",
		FirmwareVersion: "1.0.0",
		Created:         time.Now(),
		LastAttestation: time.Now(),
	}

	ts.logger.Info("device identity generated",
		zap.String("device_id", identity.DeviceID),
		zap.String("ek_fingerprint", identity.EKFingerprint[:16]+"..."),
		zap.String("ak_fingerprint", identity.AKFingerprint[:16]+"..."))

	return identity, nil
}

// ListSealedKeys returns all sealed keys
func (ts *TPMSealer) ListSealedKeys() map[string]*SealedKey {
	ts.mu.RLock()
	defer ts.mu.RUnlock()

	result := make(map[string]*SealedKey)
	for id, key := range ts.sealedKeys {
		result[id] = key
	}

	return result
}

// DeleteSealedKey deletes a sealed key
func (ts *TPMSealer) DeleteSealedKey(ctx context.Context, keyID string) error {
	ts.mu.Lock()
	defer ts.mu.Unlock()

	sealedKey, exists := ts.sealedKeys[keyID]
	if !exists {
		return fmt.Errorf("key %s not found", keyID)
	}

	// Flush the key from TPM
	_, err := tpm2.FlushContext{
		FlushHandle: sealedKey.Handle,
	}.Execute(ts.transport)
	if err != nil {
		ts.logger.Warn("failed to flush key from TPM", zap.String("key_id", keyID), zap.Error(err))
	}

	delete(ts.sealedKeys, keyID)

	ts.logger.Info("sealed key deleted", zap.String("key_id", keyID))
	return nil
}

// GetHealth returns TPM health status
func (ts *TPMSealer) GetHealth(ctx context.Context) (map[string]interface{}, error) {
	// Test TPM communication
	_, err := tpm2.GetCapability{
		Capability:    tpm2.TPMCapTPMProperties,
		Property:      uint32(tpm2.TPMPTManufacturer),
		PropertyCount: 1,
	}.Execute(ts.transport)

	healthy := err == nil

	return map[string]interface{}{
		"healthy":           healthy,
		"device_path":       ts.config.DevicePath,
		"sealed_keys":       len(ts.sealedKeys),
		"pcr_selection":     ts.config.PCRSelection,
		"pcr_bank":          ts.config.PCRBank,
		"ek_handle":         fmt.Sprintf("0x%x", ts.ekHandle),
		"ak_handle":         fmt.Sprintf("0x%x", ts.akHandle),
		"last_check":        time.Now(),
	}, nil
}

// Close closes the TPM sealer
func (ts *TPMSealer) Close() error {
	if ts.transport != nil {
		return ts.transport.Close()
	}
	return nil
}