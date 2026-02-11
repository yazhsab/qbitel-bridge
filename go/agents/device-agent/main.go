package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/json"
	"fmt"
	"math/big"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"go.uber.org/zap"
)

// DeviceAgent handles TPM attestation and device lifecycle
type DeviceAgent struct {
	logger              *zap.Logger
	deviceID            string
	controlURL          string
	attestationInterval time.Duration
	httpClient          *http.Client
}

// AttestationRequest represents a request for TPM attestation
type AttestationRequest struct {
	DeviceID  string            `json:"device_id"`
	Challenge []byte            `json:"challenge"`
	PCRValues map[int][]byte    `json:"pcr_values"`
	Quote     []byte            `json:"quote"`
	Signature []byte            `json:"signature"`
	EKCert    []byte            `json:"ek_cert"`
	AKCert    []byte            `json:"ak_cert"`
	EventLog  []byte            `json:"event_log"`
	Timestamp time.Time         `json:"timestamp"`
}

// EnrollmentRequest represents a device enrollment request
type EnrollmentRequest struct {
	DeviceID     string `json:"device_id"`
	DeviceType   string `json:"device_type"`
	EKCert       []byte `json:"ek_cert"`
	Manufacturer string `json:"manufacturer"`
	Model        string `json:"model"`
	SerialNumber string `json:"serial_number"`
}

func main() {
	logger, _ := zap.NewProduction()
	defer logger.Sync()
	
	logger.Info("device agent starting (enroll/attest/ota)")
	
	// Initialize device agent
	agent := &DeviceAgent{
		logger:              logger,
		deviceID:            getDeviceID(),
		controlURL:          getControlURL(),
		attestationInterval: 5 * time.Minute,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
	
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	
	// Handle shutdown signals
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	
	go func() {
		<-sigChan
		logger.Info("shutdown signal received")
		cancel()
	}()
	
	// Start device agent
	if err := agent.Run(ctx); err != nil {
		logger.Fatal("device agent failed", zap.Error(err))
	}
	
	logger.Info("device agent stopped")
}

// Run starts the device agent main loop
func (da *DeviceAgent) Run(ctx context.Context) error {
	// Perform initial enrollment
	if err := da.enroll(ctx); err != nil {
		return fmt.Errorf("enrollment failed: %w", err)
	}
	
	// Start attestation loop
	ticker := time.NewTicker(da.attestationInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return nil
		case <-ticker.C:
			if err := da.performAttestation(ctx); err != nil {
				da.logger.Error("attestation failed", zap.Error(err))
			}
		}
	}
}

// enroll performs device enrollment with the control plane
func (da *DeviceAgent) enroll(ctx context.Context) error {
	da.logger.Info("starting device enrollment")
	
	// Generate or load EK certificate
	ekCert, err := da.getEKCertificate()
	if err != nil {
		return fmt.Errorf("failed to get EK certificate: %w", err)
	}
	
	// Create enrollment request
	req := &EnrollmentRequest{
		DeviceID:     da.deviceID,
		DeviceType:   "qbitel-device",
		EKCert:       ekCert,
		Manufacturer: "QBITEL",
		Model:        "v1.0",
		SerialNumber: da.deviceID,
	}
	
	// Send enrollment request
	if err := da.sendEnrollmentRequest(ctx, req); err != nil {
		return fmt.Errorf("enrollment request failed: %w", err)
	}
	
	da.logger.Info("device enrollment completed")
	return nil
}

// performAttestation performs TPM attestation
func (da *DeviceAgent) performAttestation(ctx context.Context) error {
	da.logger.Debug("performing TPM attestation")
	
	// Get challenge from control plane
	challenge, err := da.getChallenge(ctx)
	if err != nil {
		return fmt.Errorf("failed to get challenge: %w", err)
	}
	
	// Read PCR values
	pcrValues, err := da.readPCRValues()
	if err != nil {
		return fmt.Errorf("failed to read PCR values: %w", err)
	}
	
	// Generate TPM quote
	quote, signature, err := da.generateQuote(challenge, pcrValues)
	if err != nil {
		return fmt.Errorf("failed to generate quote: %w", err)
	}
	
	// Get certificates
	ekCert, err := da.getEKCertificate()
	if err != nil {
		return fmt.Errorf("failed to get EK certificate: %w", err)
	}
	
	akCert, err := da.getAKCertificate()
	if err != nil {
		return fmt.Errorf("failed to get AK certificate: %w", err)
	}
	
	// Get event log
	eventLog, err := da.getEventLog()
	if err != nil {
		da.logger.Warn("failed to get event log", zap.Error(err))
		eventLog = []byte{} // Continue without event log
	}
	
	// Create attestation request
	attestReq := &AttestationRequest{
		DeviceID:  da.deviceID,
		Challenge: challenge,
		PCRValues: pcrValues,
		Quote:     quote,
		Signature: signature,
		EKCert:    ekCert,
		AKCert:    akCert,
		EventLog:  eventLog,
		Timestamp: time.Now(),
	}
	
	// Send attestation request
	if err := da.sendAttestationRequest(ctx, attestReq); err != nil {
		return fmt.Errorf("attestation request failed: %w", err)
	}
	
	da.logger.Debug("TPM attestation completed successfully")
	return nil
}

// Helper methods for TPM operations (simplified implementations)

func (da *DeviceAgent) getEKCertificate() ([]byte, error) {
	// In a real implementation, this would read the EK certificate from TPM
	// For now, generate a mock certificate
	return da.generateMockCertificate("EK")
}

func (da *DeviceAgent) getAKCertificate() ([]byte, error) {
	// In a real implementation, this would read the AK certificate from TPM
	// For now, generate a mock certificate
	return da.generateMockCertificate("AK")
}

func (da *DeviceAgent) generateMockCertificate(certType string) ([]byte, error) {
	// Generate a mock RSA key pair
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return nil, err
	}
	
	// Create a self-signed certificate template
	template := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject: pkix.Name{
			CommonName:   fmt.Sprintf("%s-%s", certType, da.deviceID),
			Organization: []string{"QBITEL"},
		},
		NotBefore:    time.Now(),
		NotAfter:     time.Now().Add(365 * 24 * time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}
	
	// Create the certificate
	certDER, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	if err != nil {
		return nil, err
	}
	
	return certDER, nil
}

func (da *DeviceAgent) readPCRValues() (map[int][]byte, error) {
	// In a real implementation, this would read PCR values from TPM
	// For now, return mock PCR values
	pcrValues := make(map[int][]byte)
	
	for i := 0; i < 24; i++ {
		hash := sha256.Sum256([]byte(fmt.Sprintf("pcr-%d-%s", i, da.deviceID)))
		pcrValues[i] = hash[:]
	}
	
	return pcrValues, nil
}

func (da *DeviceAgent) generateQuote(challenge []byte, pcrValues map[int][]byte) ([]byte, []byte, error) {
	// In a real implementation, this would use TPM to generate a quote
	// For now, create a mock quote structure
	
	quoteData := struct {
		Magic     uint32            `json:"magic"`
		Challenge []byte            `json:"challenge"`
		PCRValues map[int][]byte    `json:"pcr_values"`
		Timestamp time.Time         `json:"timestamp"`
	}{
		Magic:     0xff544347, // TPM_GENERATED_VALUE
		Challenge: challenge,
		PCRValues: pcrValues,
		Timestamp: time.Now(),
	}
	
	quote, err := json.Marshal(quoteData)
	if err != nil {
		return nil, nil, err
	}
	
	// Generate mock signature
	hash := sha256.Sum256(quote)
	signature := hash[:] // Simplified signature
	
	return quote, signature, nil
}

func (da *DeviceAgent) getEventLog() ([]byte, error) {
	// In a real implementation, this would read the TPM event log
	// For now, return empty event log
	return []byte{}, nil
}

func (da *DeviceAgent) getChallenge(ctx context.Context) ([]byte, error) {
	// Request challenge from control plane
	url := fmt.Sprintf("%s/api/v1/devices/%s/challenge", da.controlURL, da.deviceID)
	
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}
	
	resp, err := da.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("challenge request failed with status: %d", resp.StatusCode)
	}
	
	var challengeResp struct {
		Challenge []byte `json:"challenge"`
	}
	
	if err := json.NewDecoder(resp.Body).Decode(&challengeResp); err != nil {
		return nil, err
	}
	
	return challengeResp.Challenge, nil
}

func (da *DeviceAgent) sendEnrollmentRequest(ctx context.Context, req *EnrollmentRequest) error {
	url := fmt.Sprintf("%s/api/v1/devices/enroll", da.controlURL)
	return da.sendJSONRequest(ctx, "POST", url, req)
}

func (da *DeviceAgent) sendAttestationRequest(ctx context.Context, req *AttestationRequest) error {
	url := fmt.Sprintf("%s/api/v1/devices/%s/attest", da.controlURL, da.deviceID)
	return da.sendJSONRequest(ctx, "POST", url, req)
}

func (da *DeviceAgent) sendJSONRequest(ctx context.Context, method, url string, payload interface{}) error {
	jsonData, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	
	req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return err
	}
	
	req.Header.Set("Content-Type", "application/json")
	
	resp, err := da.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("request failed with status: %d", resp.StatusCode)
	}
	
	return nil
}

// Utility functions

func getDeviceID() string {
	if id := os.Getenv("DEVICE_ID"); id != "" {
		return id
	}
	// Generate a device ID based on system information
	hostname, _ := os.Hostname()
	return fmt.Sprintf("device-%s", hostname)
}

func getControlURL() string {
	if url := os.Getenv("CONTROL_URL"); url != "" {
		return url
	}
	return "https://control.qbitel.local"
}
