package devices

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"math/big"
	"strings"
	"sync"
	"time"

	"go.uber.org/zap"
)

// CertificateManager handles device certificate lifecycle
type CertificateManager struct {
	logger     *zap.Logger
	config     *LifecycleConfig
	rootCA     *x509.Certificate
	rootKey    *rsa.PrivateKey
	intermCA   *x509.Certificate
	intermKey  *rsa.PrivateKey
	crl        *x509.RevocationList
	mu         sync.RWMutex
	serialNum  *big.Int
}

// NewCertificateManager creates a new certificate manager
func NewCertificateManager(logger *zap.Logger, config *LifecycleConfig) (*CertificateManager, error) {
	cm := &CertificateManager{
		logger:    logger,
		config:    config,
		serialNum: big.NewInt(1),
	}

	// Load or create CA certificates
	if err := cm.initializeCAs(); err != nil {
		return nil, fmt.Errorf("failed to initialize CAs: %w", err)
	}

	// Initialize CRL
	if err := cm.initializeCRL(); err != nil {
		return nil, fmt.Errorf("failed to initialize CRL: %w", err)
	}

	logger.Info("certificate manager initialized",
		zap.String("root_ca_subject", cm.rootCA.Subject.String()),
		zap.String("intermediate_ca_subject", cm.intermCA.Subject.String()))

	return cm, nil
}

// IssueCertificate issues a new device certificate
func (cm *CertificateManager) IssueCertificate(ctx context.Context, csr *x509.CertificateRequest, device *Device) (*DeviceCertificate, error) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	// Validate CSR
	if err := cm.validateCSR(csr); err != nil {
		return nil, fmt.Errorf("CSR validation failed: %w", err)
	}

	// Generate serial number
	serialNumber := new(big.Int).Set(cm.serialNum)
	cm.serialNum.Add(cm.serialNum, big.NewInt(1))

	// Create certificate template
	template := &x509.Certificate{
		SerialNumber: serialNumber,
		Subject: pkix.Name{
			CommonName:         fmt.Sprintf("device-%s", device.ID),
			Organization:       []string{device.OrganizationID},
			OrganizationalUnit: []string{"QSLB Devices"},
			Country:            []string{"US"},
			Province:           []string{"CA"},
			Locality:           []string{"San Francisco"},
		},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(cm.config.CertificateValidity),
		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageKeyEncipherment,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth, x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		IsCA:                  false,
	}

	// Add device-specific extensions
	cm.addDeviceExtensions(template, device)

	// Add SAN entries
	template.DNSNames = []string{
		fmt.Sprintf("%s.devices.qslb.local", device.ID),
		fmt.Sprintf("%s.%s.devices.qslb.local", device.ID, device.OrganizationID),
	}

	// Sign certificate
	certDER, err := x509.CreateCertificate(rand.Reader, template, cm.intermCA, csr.PublicKey, cm.intermKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create certificate: %w", err)
	}

	// Parse the created certificate
	cert, err := x509.ParseCertificate(certDER)
	if err != nil {
		return nil, fmt.Errorf("failed to parse created certificate: %w", err)
	}

	// Create device certificate record
	deviceCert := &DeviceCertificate{
		ID:           fmt.Sprintf("cert-%s", device.ID),
		DeviceID:     device.ID,
		Certificate:  certDER,
		SerialNumber: serialNumber.String(),
		Subject:      cert.Subject.String(),
		Issuer:       cert.Issuer.String(),
		NotBefore:    cert.NotBefore,
		NotAfter:     cert.NotAfter,
		KeyUsage:     []string{"digital_signature", "key_encipherment", "client_auth", "server_auth"},
		Status:       CertificateStatusActive,
		CreatedAt:    time.Now(),
	}

	cm.logger.Info("device certificate issued",
		zap.String("device_id", device.ID),
		zap.String("certificate_id", deviceCert.ID),
		zap.String("serial_number", deviceCert.SerialNumber),
		zap.Time("not_after", deviceCert.NotAfter))

	return deviceCert, nil
}

// RevokeCertificate revokes a device certificate
func (cm *CertificateManager) RevokeCertificate(ctx context.Context, certID string, reason string) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	// In a real implementation, this would look up the certificate from storage
	// For now, we'll simulate the revocation process

	// Add to CRL
	revokedCert := pkix.RevokedCertificate{
		SerialNumber:   big.NewInt(1), // Would be actual serial number
		RevocationTime: time.Now(),
		Extensions: []pkix.Extension{
			{
				Id:    []int{2, 5, 29, 21}, // CRL Reason Code
				Value: []byte{1},           // Key compromise
			},
		},
	}

	// Update CRL
	cm.crl.RevokedCertificates = append(cm.crl.RevokedCertificates, revokedCert)
	cm.crl.NextUpdate = time.Now().Add(24 * time.Hour)

	// Regenerate CRL
	crlDER, err := x509.CreateRevocationList(rand.Reader, cm.crl, cm.intermCA, cm.intermKey)
	if err != nil {
		return fmt.Errorf("failed to create CRL: %w", err)
	}

	// In a real implementation, this would be published to a CRL distribution point
	_ = crlDER

	cm.logger.Info("certificate revoked",
		zap.String("certificate_id", certID),
		zap.String("reason", reason))

	return nil
}

// RenewCertificate renews a device certificate
func (cm *CertificateManager) RenewCertificate(ctx context.Context, certID string, csr *x509.CertificateRequest) (*DeviceCertificate, error) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	cm.logger.Info("certificate renewal requested",
		zap.String("certificate_id", certID))

	// In a real implementation, this would look up the existing certificate from storage
	// For now, we'll simulate the lookup and validation process
	
	// Validate the renewal request
	if err := cm.validateRenewalRequest(certID, csr); err != nil {
		return nil, fmt.Errorf("renewal validation failed: %w", err)
	}

	// Extract device information from certificate ID
	deviceID := extractDeviceIDFromCertID(certID)
	if deviceID == "" {
		return nil, fmt.Errorf("invalid certificate ID format")
	}

	// Create a mock device for renewal (in real implementation, this would be fetched from storage)
	device := &Device{
		ID:             deviceID,
		OrganizationID: "default-org",
		DeviceType:     "qslb-device",
		Status:         DeviceStatusActive,
	}

	// Generate new serial number
	serialNumber := new(big.Int).Set(cm.serialNum)
	cm.serialNum.Add(cm.serialNum, big.NewInt(1))

	// Create certificate template with extended validity
	template := &x509.Certificate{
		SerialNumber: serialNumber,
		Subject: pkix.Name{
			CommonName:         fmt.Sprintf("device-%s", device.ID),
			Organization:       []string{device.OrganizationID},
			OrganizationalUnit: []string{"QSLB Devices"},
			Country:            []string{"US"},
			Province:           []string{"CA"},
			Locality:           []string{"San Francisco"},
		},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(cm.config.CertificateValidity),
		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageKeyEncipherment,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth, x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		IsCA:                  false,
	}

	// Add device-specific extensions
	cm.addDeviceExtensions(template, device)

	// Add SAN entries
	template.DNSNames = []string{
		fmt.Sprintf("%s.devices.qslb.local", device.ID),
		fmt.Sprintf("%s.%s.devices.qslb.local", device.ID, device.OrganizationID),
	}

	// Sign the renewed certificate
	certDER, err := x509.CreateCertificate(rand.Reader, template, cm.intermCA, csr.PublicKey, cm.intermKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create renewed certificate: %w", err)
	}

	// Parse the created certificate
	cert, err := x509.ParseCertificate(certDER)
	if err != nil {
		return nil, fmt.Errorf("failed to parse renewed certificate: %w", err)
	}

	// Create new device certificate record
	newCertID := fmt.Sprintf("cert-%s-%d", device.ID, time.Now().Unix())
	deviceCert := &DeviceCertificate{
		ID:           newCertID,
		DeviceID:     device.ID,
		Certificate:  certDER,
		SerialNumber: serialNumber.String(),
		Subject:      cert.Subject.String(),
		Issuer:       cert.Issuer.String(),
		NotBefore:    cert.NotBefore,
		NotAfter:     cert.NotAfter,
		KeyUsage:     []string{"digital_signature", "key_encipherment", "client_auth", "server_auth"},
		Status:       CertificateStatusActive,
		CreatedAt:    time.Now(),
		RenewedFrom:  certID, // Track the original certificate
	}

	// In a real implementation, we would:
	// 1. Store the new certificate in the database
	// 2. Mark the old certificate as superseded
	// 3. Optionally add the old certificate to CRL

	cm.logger.Info("certificate renewed successfully",
		zap.String("old_certificate_id", certID),
		zap.String("new_certificate_id", deviceCert.ID),
		zap.String("device_id", device.ID),
		zap.String("serial_number", deviceCert.SerialNumber),
		zap.Time("new_not_after", deviceCert.NotAfter))

	return deviceCert, nil
}

// GetCertificateChain returns the certificate chain for validation
func (cm *CertificateManager) GetCertificateChain() ([]*x509.Certificate, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	return []*x509.Certificate{cm.intermCA, cm.rootCA}, nil
}

// GetCRL returns the current Certificate Revocation List
func (cm *CertificateManager) GetCRL() ([]byte, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	crlDER, err := x509.CreateRevocationList(rand.Reader, cm.crl, cm.intermCA, cm.intermKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create CRL: %w", err)
	}

	return crlDER, nil
}

// ValidateCertificate validates a device certificate
func (cm *CertificateManager) ValidateCertificate(certDER []byte) error {
	cert, err := x509.ParseCertificate(certDER)
	if err != nil {
		return fmt.Errorf("failed to parse certificate: %w", err)
	}

	// Create certificate pool with our CA
	roots := x509.NewCertPool()
	roots.AddCert(cm.rootCA)

	intermediates := x509.NewCertPool()
	intermediates.AddCert(cm.intermCA)

	// Verify certificate
	opts := x509.VerifyOptions{
		Roots:         roots,
		Intermediates: intermediates,
		KeyUsages:     []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}

	_, err = cert.Verify(opts)
	if err != nil {
		return fmt.Errorf("certificate verification failed: %w", err)
	}

	// Check if certificate is revoked
	if cm.isCertificateRevoked(cert.SerialNumber) {
		return fmt.Errorf("certificate is revoked")
	}

	return nil
}

// Helper methods

func (cm *CertificateManager) initializeCAs() error {
	// In a real implementation, this would load existing CA certificates
	// For now, we'll create self-signed CAs for demonstration

	// Create root CA
	rootTemplate := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject: pkix.Name{
			CommonName:   "QSLB Root CA",
			Organization: []string{"QSLB"},
			Country:      []string{"US"},
		},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(10 * 365 * 24 * time.Hour), // 10 years
		KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageCRLSign,
		BasicConstraintsValid: true,
		IsCA:                  true,
		MaxPathLen:            1,
	}

	// Generate root CA key
	rootKey, err := rsa.GenerateKey(rand.Reader, 4096)
	if err != nil {
		return fmt.Errorf("failed to generate root CA key: %w", err)
	}

	// Create root CA certificate
	rootCertDER, err := x509.CreateCertificate(rand.Reader, rootTemplate, rootTemplate, &rootKey.PublicKey, rootKey)
	if err != nil {
		return fmt.Errorf("failed to create root CA certificate: %w", err)
	}

	rootCert, err := x509.ParseCertificate(rootCertDER)
	if err != nil {
		return fmt.Errorf("failed to parse root CA certificate: %w", err)
	}

	cm.rootCA = rootCert
	cm.rootKey = rootKey

	// Create intermediate CA
	intermTemplate := &x509.Certificate{
		SerialNumber: big.NewInt(2),
		Subject: pkix.Name{
			CommonName:   "QSLB Intermediate CA",
			Organization: []string{"QSLB"},
			Country:      []string{"US"},
		},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(5 * 365 * 24 * time.Hour), // 5 years
		KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageCRLSign,
		BasicConstraintsValid: true,
		IsCA:                  true,
		MaxPathLen:            0,
	}

	// Generate intermediate CA key
	intermKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return fmt.Errorf("failed to generate intermediate CA key: %w", err)
	}

	// Create intermediate CA certificate
	intermCertDER, err := x509.CreateCertificate(rand.Reader, intermTemplate, rootCert, &intermKey.PublicKey, rootKey)
	if err != nil {
		return fmt.Errorf("failed to create intermediate CA certificate: %w", err)
	}

	intermCert, err := x509.ParseCertificate(intermCertDER)
	if err != nil {
		return fmt.Errorf("failed to parse intermediate CA certificate: %w", err)
	}

	cm.intermCA = intermCert
	cm.intermKey = intermKey

	return nil
}

func (cm *CertificateManager) initializeCRL() error {
	cm.crl = &x509.RevocationList{
		SignatureAlgorithm:  x509.SHA256WithRSA,
		RevokedCertificates: []pkix.RevokedCertificate{},
		Number:              big.NewInt(1),
		ThisUpdate:          time.Now(),
		NextUpdate:          time.Now().Add(24 * time.Hour),
	}

	return nil
}

func (cm *CertificateManager) validateCSR(csr *x509.CertificateRequest) error {
	// Verify CSR signature
	if err := csr.CheckSignature(); err != nil {
		return fmt.Errorf("CSR signature verification failed: %w", err)
	}

	// Validate key size
	if rsaKey, ok := csr.PublicKey.(*rsa.PublicKey); ok {
		if rsaKey.Size() < 256 { // 2048 bits minimum
			return fmt.Errorf("RSA key size too small: %d bits", rsaKey.Size()*8)
		}
	}

	// Validate subject
	if csr.Subject.CommonName == "" {
		return fmt.Errorf("CSR must have a common name")
	}

	return nil
}

func (cm *CertificateManager) addDeviceExtensions(template *x509.Certificate, device *Device) {
	// Add device-specific extensions
	// This could include device type, capabilities, etc.
	
	// Example: Add device ID as a custom extension
	deviceIDExt := pkix.Extension{
		Id:       []int{1, 3, 6, 1, 4, 1, 12345, 1, 1}, // Custom OID
		Critical: false,
		Value:    []byte(device.ID),
	}
	template.ExtraExtensions = append(template.ExtraExtensions, deviceIDExt)

	// Add device type extension
	deviceTypeExt := pkix.Extension{
		Id:       []int{1, 3, 6, 1, 4, 1, 12345, 1, 2}, // Custom OID
		Critical: false,
		Value:    []byte(device.DeviceType),
	}
	template.ExtraExtensions = append(template.ExtraExtensions, deviceTypeExt)
}

func (cm *CertificateManager) isCertificateRevoked(serialNumber *big.Int) bool {
	for _, revoked := range cm.crl.RevokedCertificates {
		if revoked.SerialNumber.Cmp(serialNumber) == 0 {
			return true
		}
	}
	return false
}

// GetCertificateStats returns certificate management statistics
func (cm *CertificateManager) GetCertificateStats() map[string]interface{} {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	return map[string]interface{}{
		"root_ca_subject":        cm.rootCA.Subject.String(),
		"intermediate_ca_subject": cm.intermCA.Subject.String(),
		"root_ca_expires":        cm.rootCA.NotAfter,
		"intermediate_ca_expires": cm.intermCA.NotAfter,
		"revoked_certificates":   len(cm.crl.RevokedCertificates),
		"next_serial_number":     cm.serialNum.String(),
		"crl_next_update":        cm.crl.NextUpdate,
	}
}

// ExportCACertificates exports CA certificates in PEM format
func (cm *CertificateManager) ExportCACertificates() ([]byte, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	var pemData []byte

	// Export root CA
	rootPEM := pem.EncodeToMemory(&pem.Block{
		Type:  "CERTIFICATE",
		Bytes: cm.rootCA.Raw,
	})
	pemData = append(pemData, rootPEM...)

	// Export intermediate CA
	intermPEM := pem.EncodeToMemory(&pem.Block{
		Type:  "CERTIFICATE",
		Bytes: cm.intermCA.Raw,
	})
	pemData = append(pemData, intermPEM...)

	return pemData, nil
}

// PolicyEngine placeholder - would be implemented separately
type PolicyEngine struct {
	logger *zap.Logger
	config *LifecycleConfig
}

func NewPolicyEngine(logger *zap.Logger, config *LifecycleConfig) (*PolicyEngine, error) {
	return &PolicyEngine{
		logger: logger,
		config: config,
	}, nil
}

// validateRenewalRequest validates a certificate renewal request
func (cm *CertificateManager) validateRenewalRequest(certID string, csr *x509.CertificateRequest) error {
	// Validate CSR
	if err := cm.validateCSR(csr); err != nil {
		return fmt.Errorf("CSR validation failed: %w", err)
	}

	// Validate certificate ID format
	if certID == "" {
		return fmt.Errorf("certificate ID cannot be empty")
	}

	// In a real implementation, this would:
	// 1. Check if the certificate exists and is valid for renewal
	// 2. Verify the device is authorized to renew this certificate
	// 3. Check renewal policies (e.g., not too early, not expired too long)
	// 4. Validate that the CSR public key matches or is authorized

	cm.logger.Debug("certificate renewal request validated",
		zap.String("certificate_id", certID))

	return nil
}

// extractDeviceIDFromCertID extracts device ID from certificate ID
func extractDeviceIDFromCertID(certID string) string {
	// Simple extraction assuming format "cert-{deviceID}" or "cert-{deviceID}-{timestamp}"
	if len(certID) > 5 && certID[:5] == "cert-" {
		remaining := certID[5:]
		// Find the first dash after "cert-" to handle timestamps
		if dashIndex := strings.Index(remaining, "-"); dashIndex > 0 {
			return remaining[:dashIndex]
		}
		return remaining
	}
	return ""
}

// Additional certificate management helper methods

// GetCertificateByID retrieves a certificate by its ID (placeholder implementation)
func (cm *CertificateManager) GetCertificateByID(certID string) (*DeviceCertificate, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	// In a real implementation, this would query the database
	// For now, return a mock certificate
	deviceID := extractDeviceIDFromCertID(certID)
	if deviceID == "" {
		return nil, fmt.Errorf("certificate not found")
	}

	return &DeviceCertificate{
		ID:           certID,
		DeviceID:     deviceID,
		SerialNumber: "1",
		Status:       CertificateStatusActive,
		CreatedAt:    time.Now().Add(-24 * time.Hour),
		NotBefore:    time.Now().Add(-24 * time.Hour),
		NotAfter:     time.Now().Add(365 * 24 * time.Hour),
	}, nil
}

// ListCertificatesByDevice lists all certificates for a device
func (cm *CertificateManager) ListCertificatesByDevice(deviceID string) ([]*DeviceCertificate, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	// In a real implementation, this would query the database
	// For now, return a mock list
	return []*DeviceCertificate{
		{
			ID:           fmt.Sprintf("cert-%s", deviceID),
			DeviceID:     deviceID,
			SerialNumber: "1",
			Status:       CertificateStatusActive,
			CreatedAt:    time.Now().Add(-24 * time.Hour),
			NotBefore:    time.Now().Add(-24 * time.Hour),
			NotAfter:     time.Now().Add(365 * 24 * time.Hour),
		},
	}, nil
}

// CheckCertificateExpiry checks for certificates nearing expiry
func (cm *CertificateManager) CheckCertificateExpiry(warningDays int) ([]*DeviceCertificate, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	// In a real implementation, this would query the database for certificates
	// expiring within the warning period
	warningTime := time.Now().Add(time.Duration(warningDays) * 24 * time.Hour)
	
	var expiringCerts []*DeviceCertificate
	
	// This would be replaced with actual database query
	cm.logger.Info("checking for expiring certificates",
		zap.Time("warning_threshold", warningTime),
		zap.Int("warning_days", warningDays))

	return expiringCerts, nil
}