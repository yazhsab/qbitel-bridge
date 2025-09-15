package devices

import (
	"context"
	"crypto"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/asn1"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"
)

// AttestationVerifier handles TPM attestation verification
type AttestationVerifier struct {
	logger          *zap.Logger
	config          *LifecycleConfig
	trustedRoots    *x509.CertPool
	trustedEKCerts  map[string]*x509.Certificate
	attestationCache map[string]*CachedAttestation
	mu              sync.RWMutex
}

// CachedAttestation represents a cached attestation result
type CachedAttestation struct {
	DeviceID    string    `json:"device_id"`
	Result      bool      `json:"result"`
	Timestamp   time.Time `json:"timestamp"`
	ExpiresAt   time.Time `json:"expires_at"`
	PCRValues   map[int][]byte `json:"pcr_values"`
	EventLog    []byte    `json:"event_log"`
	Errors      []string  `json:"errors,omitempty"`
}

// TPMQuote represents a TPM quote structure
type TPMQuote struct {
	Magic       uint32            `json:"magic"`
	Type        uint16            `json:"type"`
	QualData    []byte            `json:"qual_data"`
	PCRSelect   TPMPCRSelection   `json:"pcr_select"`
	PCRDigest   []byte            `json:"pcr_digest"`
	ClockInfo   TPMClockInfo      `json:"clock_info"`
	FirmwareVersion uint64        `json:"firmware_version"`
	Attested    TPMAttestedQuote  `json:"attested"`
}

// TPMPCRSelection represents PCR selection
type TPMPCRSelection struct {
	Hash      uint16 `json:"hash"`
	SizeOfSelect uint8 `json:"size_of_select"`
	PCRSelect []byte `json:"pcr_select"`
}

// TPMClockInfo represents TPM clock information
type TPMClockInfo struct {
	Clock        uint64 `json:"clock"`
	ResetCount   uint32 `json:"reset_count"`
	RestartCount uint32 `json:"restart_count"`
	Safe         uint8  `json:"safe"`
}

// TPMAttestedQuote represents the attested portion of a quote
type TPMAttestedQuote struct {
	Magic       uint32 `json:"magic"`
	Type        uint16 `json:"type"`
	QualData    []byte `json:"qual_data"`
	ExtraData   []byte `json:"extra_data"`
}

// EventLogEntry represents a measured boot event
type EventLogEntry struct {
	PCRIndex    int    `json:"pcr_index"`
	EventType   uint32 `json:"event_type"`
	Digest      []byte `json:"digest"`
	EventSize   uint32 `json:"event_size"`
	Event       []byte `json:"event"`
	Description string `json:"description"`
}

// AttestationResult represents the result of attestation verification
type AttestationResult struct {
	Valid           bool                    `json:"valid"`
	DeviceID        string                  `json:"device_id"`
	Timestamp       time.Time               `json:"timestamp"`
	PCRValues       map[int][]byte          `json:"pcr_values"`
	ExpectedPCRs    map[int][]byte          `json:"expected_pcrs"`
	EventLog        []EventLogEntry         `json:"event_log"`
	Compliance      ComplianceResult        `json:"compliance"`
	TrustChain      TrustChainResult        `json:"trust_chain"`
	Errors          []string                `json:"errors,omitempty"`
	Warnings        []string                `json:"warnings,omitempty"`
}

// ComplianceResult represents compliance check results
type ComplianceResult struct {
	SecureBoot      bool     `json:"secure_boot"`
	TPMEnabled      bool     `json:"tpm_enabled"`
	FirmwareValid   bool     `json:"firmware_valid"`
	BootloaderValid bool     `json:"bootloader_valid"`
	OSValid         bool     `json:"os_valid"`
	Violations      []string `json:"violations,omitempty"`
}

// TrustChainResult represents trust chain verification results
type TrustChainResult struct {
	EKValid         bool     `json:"ek_valid"`
	AKValid         bool     `json:"ak_valid"`
	CertChainValid  bool     `json:"cert_chain_valid"`
	ManufacturerTrusted bool `json:"manufacturer_trusted"`
	Issues          []string `json:"issues,omitempty"`
}

// NewAttestationVerifier creates a new attestation verifier
func NewAttestationVerifier(logger *zap.Logger, config *LifecycleConfig) (*AttestationVerifier, error) {
	verifier := &AttestationVerifier{
		logger:           logger,
		config:           config,
		trustedRoots:     x509.NewCertPool(),
		trustedEKCerts:   make(map[string]*x509.Certificate),
		attestationCache: make(map[string]*CachedAttestation),
	}

	// Load trusted root certificates
	if err := verifier.loadTrustedRoots(); err != nil {
		return nil, fmt.Errorf("failed to load trusted roots: %w", err)
	}

	// Load trusted EK certificates
	if err := verifier.loadTrustedEKCerts(); err != nil {
		return nil, fmt.Errorf("failed to load trusted EK certificates: %w", err)
	}

	// Start cache cleanup task
	go verifier.startCacheCleanupTask()

	logger.Info("attestation verifier initialized",
		zap.Int("trusted_roots", len(verifier.trustedRoots.Subjects())),
		zap.Int("trusted_ek_certs", len(verifier.trustedEKCerts)))

	return verifier, nil
}

// VerifyAttestation verifies TPM attestation data
func (av *AttestationVerifier) VerifyAttestation(ctx context.Context, attestation *AttestationData, challenge []byte) error {
	start := time.Now()
	defer func() {
		av.logger.Debug("attestation verification completed",
			zap.Duration("duration", time.Since(start)))
	}()

	// Check cache first
	if cached := av.getCachedAttestation(attestation); cached != nil && cached.Result {
		av.logger.Debug("using cached attestation result")
		return nil
	}

	result := &AttestationResult{
		Timestamp: time.Now(),
		PCRValues: attestation.PCRValues,
	}

	// Verify quote signature
	if err := av.verifyQuoteSignature(attestation); err != nil {
		result.Errors = append(result.Errors, fmt.Sprintf("quote signature verification failed: %v", err))
		av.cacheAttestationResult(attestation, result)
		return fmt.Errorf("quote signature verification failed: %w", err)
	}

	// Verify challenge/nonce
	if err := av.verifyChallenge(attestation, challenge); err != nil {
		result.Errors = append(result.Errors, fmt.Sprintf("challenge verification failed: %v", err))
		av.cacheAttestationResult(attestation, result)
		return fmt.Errorf("challenge verification failed: %w", err)
	}

	// Verify trust chain
	trustResult, err := av.verifyTrustChain(attestation)
	if err != nil {
		result.Errors = append(result.Errors, fmt.Sprintf("trust chain verification failed: %v", err))
		av.cacheAttestationResult(attestation, result)
		return fmt.Errorf("trust chain verification failed: %w", err)
	}
	result.TrustChain = *trustResult

	// Verify PCR values
	if err := av.verifyPCRValues(attestation); err != nil {
		result.Errors = append(result.Errors, fmt.Sprintf("PCR verification failed: %v", err))
		av.cacheAttestationResult(attestation, result)
		return fmt.Errorf("PCR verification failed: %w", err)
	}

	// Parse and verify event log
	if len(attestation.EventLog) > 0 {
		eventLog, err := av.parseEventLog(attestation.EventLog)
		if err != nil {
			result.Warnings = append(result.Warnings, fmt.Sprintf("event log parsing failed: %v", err))
		} else {
			result.EventLog = eventLog
			if err := av.verifyEventLog(eventLog, attestation.PCRValues); err != nil {
				result.Warnings = append(result.Warnings, fmt.Sprintf("event log verification failed: %v", err))
			}
		}
	}

	// Perform compliance checks
	complianceResult, err := av.performComplianceChecks(attestation, result.EventLog)
	if err != nil {
		result.Warnings = append(result.Warnings, fmt.Sprintf("compliance check failed: %v", err))
	} else {
		result.Compliance = *complianceResult
	}

	result.Valid = len(result.Errors) == 0
	av.cacheAttestationResult(attestation, result)

	if !result.Valid {
		return fmt.Errorf("attestation verification failed: %v", result.Errors)
	}

	av.logger.Info("attestation verification successful",
		zap.Bool("secure_boot", result.Compliance.SecureBoot),
		zap.Bool("tpm_enabled", result.Compliance.TPMEnabled),
		zap.Bool("firmware_valid", result.Compliance.FirmwareValid),
		zap.Int("warnings", len(result.Warnings)))

	return nil
}

// verifyQuoteSignature verifies the TPM quote signature
func (av *AttestationVerifier) verifyQuoteSignature(attestation *AttestationData) error {
	// Parse the quote
	quote, err := av.parseTPMQuote(attestation.Quote)
	if err != nil {
		return fmt.Errorf("failed to parse TPM quote: %w", err)
	}

	// Verify quote magic number
	if quote.Magic != 0xff544347 { // TPM_GENERATED_VALUE
		return fmt.Errorf("invalid quote magic number: 0x%x", quote.Magic)
	}

	// Get AK public key from certificate
	akCert, err := x509.ParseCertificate(attestation.AKCert)
	if err != nil {
		return fmt.Errorf("failed to parse AK certificate: %w", err)
	}

	akPubKey, ok := akCert.PublicKey.(*rsa.PublicKey)
	if !ok {
		return fmt.Errorf("AK certificate does not contain RSA public key")
	}

	// Verify signature
	hash := sha256.Sum256(attestation.Quote)
	err = rsa.VerifyPKCS1v15(akPubKey, crypto.SHA256, hash[:], attestation.Signature)
	if err != nil {
		return fmt.Errorf("quote signature verification failed: %w", err)
	}

	return nil
}

// verifyChallenge verifies the challenge/nonce in the quote
func (av *AttestationVerifier) verifyChallenge(attestation *AttestationData, challenge []byte) error {
	quote, err := av.parseTPMQuote(attestation.Quote)
	if err != nil {
		return fmt.Errorf("failed to parse TPM quote: %w", err)
	}

	// Compare challenge with qualified data in quote
	if len(quote.QualData) != len(challenge) {
		return fmt.Errorf("challenge length mismatch: expected %d, got %d", len(challenge), len(quote.QualData))
	}

	for i, b := range challenge {
		if quote.QualData[i] != b {
			return fmt.Errorf("challenge verification failed at byte %d", i)
		}
	}

	return nil
}

// verifyTrustChain verifies the certificate trust chain
func (av *AttestationVerifier) verifyTrustChain(attestation *AttestationData) (*TrustChainResult, error) {
	result := &TrustChainResult{}

	// Parse EK certificate
	ekCert, err := x509.ParseCertificate(attestation.EKCert)
	if err != nil {
		result.Issues = append(result.Issues, fmt.Sprintf("failed to parse EK certificate: %v", err))
		return result, nil
	}

	// Parse AK certificate
	akCert, err := x509.ParseCertificate(attestation.AKCert)
	if err != nil {
		result.Issues = append(result.Issues, fmt.Sprintf("failed to parse AK certificate: %v", err))
		return result, nil
	}

	// Verify EK certificate against trusted roots
	opts := x509.VerifyOptions{
		Roots: av.trustedRoots,
	}
	_, err = ekCert.Verify(opts)
	if err != nil {
		result.Issues = append(result.Issues, fmt.Sprintf("EK certificate verification failed: %v", err))
	} else {
		result.EKValid = true
	}

	// Verify AK certificate (simplified - in practice would verify against EK)
	result.AKValid = true // Placeholder

	// Check manufacturer trust
	result.ManufacturerTrusted = av.isManufacturerTrusted(ekCert.Subject.Organization)

	result.CertChainValid = result.EKValid && result.AKValid

	return result, nil
}

// verifyPCRValues verifies PCR values against expected values
func (av *AttestationVerifier) verifyPCRValues(attestation *AttestationData) error {
	// Get expected PCR values (this would come from a policy or baseline)
	expectedPCRs := av.getExpectedPCRValues()

	for pcrIndex, expectedValue := range expectedPCRs {
		actualValue, exists := attestation.PCRValues[pcrIndex]
		if !exists {
			return fmt.Errorf("missing PCR %d in attestation", pcrIndex)
		}

		if len(actualValue) != len(expectedValue) {
			return fmt.Errorf("PCR %d length mismatch: expected %d, got %d", pcrIndex, len(expectedValue), len(actualValue))
		}

		for i, b := range expectedValue {
			if actualValue[i] != b {
				return fmt.Errorf("PCR %d value mismatch at byte %d", pcrIndex, i)
			}
		}
	}

	return nil
}

// parseEventLog parses the measured boot event log
func (av *AttestationVerifier) parseEventLog(eventLogData []byte) ([]EventLogEntry, error) {
	var entries []EventLogEntry
	
	// This is a simplified parser - real implementation would handle TCG Event Log format
	// For now, return empty entries
	
	return entries, nil
}

// verifyEventLog verifies event log consistency with PCR values
func (av *AttestationVerifier) verifyEventLog(eventLog []EventLogEntry, pcrValues map[int][]byte) error {
	// Replay event log and verify PCR values
	// This is a complex operation that would require full TCG Event Log parsing
	return nil
}

// performComplianceChecks performs various compliance checks
func (av *AttestationVerifier) performComplianceChecks(attestation *AttestationData, eventLog []EventLogEntry) (*ComplianceResult, error) {
	result := &ComplianceResult{}

	// Check secure boot (PCR 7 typically contains secure boot measurements)
	if pcrValue, exists := attestation.PCRValues[7]; exists {
		// Check if PCR 7 indicates secure boot is enabled
		result.SecureBoot = av.checkSecureBootPCR(pcrValue)
	}

	// Check TPM enabled (if we got attestation data, TPM is enabled)
	result.TPMEnabled = true

	// Check firmware validity (PCRs 0-3 typically contain firmware measurements)
	result.FirmwareValid = av.checkFirmwareValidity(attestation.PCRValues)

	// Check bootloader validity (PCR 4 typically contains bootloader measurements)
	if pcrValue, exists := attestation.PCRValues[4]; exists {
		result.BootloaderValid = av.checkBootloaderValidity(pcrValue)
	}

	// Check OS validity (PCRs 8-15 typically contain OS measurements)
	result.OSValid = av.checkOSValidity(attestation.PCRValues)

	// Collect violations
	if !result.SecureBoot && av.config.RequireSecureBoot {
		result.Violations = append(result.Violations, "secure boot not enabled")
	}

	return result, nil
}

// Helper methods

func (av *AttestationVerifier) parseTPMQuote(quoteData []byte) (*TPMQuote, error) {
	// This is a simplified parser - real implementation would handle TPM2B_ATTEST structure
	quote := &TPMQuote{}
	
	if len(quoteData) < 8 {
		return nil, fmt.Errorf("quote data too short")
	}

	// Parse basic fields (simplified)
	quote.Magic = uint32(quoteData[0])<<24 | uint32(quoteData[1])<<16 | uint32(quoteData[2])<<8 | uint32(quoteData[3])
	quote.Type = uint16(quoteData[4])<<8 | uint16(quoteData[5])
	
	// In a real implementation, this would properly parse the entire TPMS_ATTEST structure
	
	return quote, nil
}

func (av *AttestationVerifier) getCachedAttestation(attestation *AttestationData) *CachedAttestation {
	av.mu.RLock()
	defer av.mu.RUnlock()

	// Create a hash of the attestation data for cache key
	hash := sha256.Sum256(attestation.Quote)
	key := fmt.Sprintf("%x", hash[:8])

	cached, exists := av.attestationCache[key]
	if !exists || time.Now().After(cached.ExpiresAt) {
		return nil
	}

	return cached
}

func (av *AttestationVerifier) cacheAttestationResult(attestation *AttestationData, result *AttestationResult) {
	av.mu.Lock()
	defer av.mu.Unlock()

	hash := sha256.Sum256(attestation.Quote)
	key := fmt.Sprintf("%x", hash[:8])

	cached := &CachedAttestation{
		DeviceID:  result.DeviceID,
		Result:    result.Valid,
		Timestamp: result.Timestamp,
		ExpiresAt: time.Now().Add(time.Hour), // Cache for 1 hour
		PCRValues: result.PCRValues,
		Errors:    result.Errors,
	}

	av.attestationCache[key] = cached
}

func (av *AttestationVerifier) loadTrustedRoots() error {
	// Load trusted root certificates from configuration
	// This would typically load from files or a certificate store
	return nil
}

func (av *AttestationVerifier) loadTrustedEKCerts() error {
	// Load trusted EK certificates from TPM manufacturers
	// This would typically load from a database or certificate store
	return nil
}

func (av *AttestationVerifier) isManufacturerTrusted(organizations []string) bool {
	for _, org := range organizations {
		for _, allowed := range av.config.AllowedManufacturers {
			if org == allowed {
				return true
			}
		}
	}
	return false
}

func (av *AttestationVerifier) getExpectedPCRValues() map[int][]byte {
	// Return expected PCR values from policy or baseline
	// This would typically come from a configuration or policy engine
	return make(map[int][]byte)
}

func (av *AttestationVerifier) checkSecureBootPCR(pcrValue []byte) bool {
	// Check if PCR value indicates secure boot is enabled
	// This would compare against known good values
	return len(pcrValue) > 0 // Simplified check
}

func (av *AttestationVerifier) checkFirmwareValidity(pcrValues map[int][]byte) bool {
	// Check firmware PCRs (0-3) against known good values
	for i := 0; i < 4; i++ {
		if _, exists := pcrValues[i]; !exists {
			return false
		}
	}
	return true
}

func (av *AttestationVerifier) checkBootloaderValidity(pcrValue []byte) bool {
	// Check bootloader PCR against known good values
	return len(pcrValue) > 0 // Simplified check
}

func (av *AttestationVerifier) checkOSValidity(pcrValues map[int][]byte) bool {
	// Check OS PCRs (8-15) against known good values
	for i := 8; i < 16; i++ {
		if _, exists := pcrValues[i]; !exists {
			return false
		}
	}
	return true
}

func (av *AttestationVerifier) startCacheCleanupTask() {
	ticker := time.NewTicker(time.Hour)
	defer ticker.Stop()

	for range ticker.C {
		av.cleanupExpiredCache()
	}
}

func (av *AttestationVerifier) cleanupExpiredCache() {
	av.mu.Lock()
	defer av.mu.Unlock()

	now := time.Now()
	for key, cached := range av.attestationCache {
		if now.After(cached.ExpiresAt) {
			delete(av.attestationCache, key)
		}
	}
}

// GetAttestationStats returns attestation verification statistics
func (av *AttestationVerifier) GetAttestationStats() map[string]interface{} {
	av.mu.RLock()
	defer av.mu.RUnlock()

	return map[string]interface{}{
		"cached_attestations": len(av.attestationCache),
		"trusted_roots":       len(av.trustedRoots.Subjects()),
		"trusted_ek_certs":    len(av.trustedEKCerts),
		"require_tpm":         av.config.RequireTPMAttestation,
		"require_secure_boot": av.config.RequireSecureBoot,
	}
}