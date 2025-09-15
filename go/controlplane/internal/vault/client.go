package vault

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"sync"
	"time"

	"github.com/hashicorp/vault/api"
	"go.uber.org/zap"
)

// VaultClient provides enterprise-grade Vault integration
type VaultClient struct {
	client     *api.Client
	logger     *zap.Logger
	config     *Config
	mu         sync.RWMutex
	token      string
	tokenExp   time.Time
	renewTimer *time.Timer
}

// Config holds Vault client configuration
type Config struct {
	// Connection configuration
	Address    string `json:"address"`
	Namespace  string `json:"namespace,omitempty"`
	
	// Authentication configuration
	AuthMethod     string `json:"auth_method"`     // "token", "cert", "kubernetes", "aws"
	TokenPath      string `json:"token_path,omitempty"`
	CertPath       string `json:"cert_path,omitempty"`
	KeyPath        string `json:"key_path,omitempty"`
	CACertPath     string `json:"ca_cert_path,omitempty"`
	
	// TLS configuration
	TLSSkipVerify  bool   `json:"tls_skip_verify"`
	TLSServerName  string `json:"tls_server_name,omitempty"`
	
	// Key management configuration
	KeyMountPath   string `json:"key_mount_path"`
	TransitPath    string `json:"transit_path"`
	PKIPath        string `json:"pki_path"`
	
	// Operational configuration
	MaxRetries     int           `json:"max_retries"`
	Timeout        time.Duration `json:"timeout"`
	RenewThreshold time.Duration `json:"renew_threshold"`
	
	// HSM integration
	HSMEnabled     bool   `json:"hsm_enabled"`
	HSMSlotID      int    `json:"hsm_slot_id"`
	HSMPin         string `json:"hsm_pin,omitempty"`
}

// KeyMetadata holds metadata about managed keys
type KeyMetadata struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Type        string            `json:"type"`        // "rsa", "ec", "ed25519", "aes"
	Usage       []string          `json:"usage"`       // "encrypt", "decrypt", "sign", "verify"
	Algorithm   string            `json:"algorithm"`
	KeySize     int               `json:"key_size"`
	Created     time.Time         `json:"created"`
	Expires     *time.Time        `json:"expires,omitempty"`
	Rotated     *time.Time        `json:"rotated,omitempty"`
	Version     int               `json:"version"`
	HSMBacked   bool              `json:"hsm_backed"`
	Metadata    map[string]string `json:"metadata"`
}

// KeyRotationPolicy defines key rotation policies
type KeyRotationPolicy struct {
	KeyName         string        `json:"key_name"`
	RotationPeriod  time.Duration `json:"rotation_period"`
	MinDecryptions  int           `json:"min_decryptions"`
	AutoRotate      bool          `json:"auto_rotate"`
	NotifyBefore    time.Duration `json:"notify_before"`
	RetainVersions  int           `json:"retain_versions"`
}

// NewVaultClient creates a new enterprise Vault client
func NewVaultClient(logger *zap.Logger, config *Config) (*VaultClient, error) {
	if config == nil {
		return nil, fmt.Errorf("vault config is required")
	}

	// Validate configuration
	if err := validateConfig(config); err != nil {
		return nil, fmt.Errorf("invalid vault config: %w", err)
	}

	// Create Vault API client
	vaultConfig := api.DefaultConfig()
	vaultConfig.Address = config.Address
	vaultConfig.MaxRetries = config.MaxRetries
	vaultConfig.Timeout = config.Timeout

	// Configure TLS
	if err := configureTLS(vaultConfig, config); err != nil {
		return nil, fmt.Errorf("failed to configure TLS: %w", err)
	}

	client, err := api.NewClient(vaultConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create vault client: %w", err)
	}

	// Set namespace if provided
	if config.Namespace != "" {
		client.SetNamespace(config.Namespace)
	}

	vc := &VaultClient{
		client: client,
		logger: logger,
		config: config,
	}

	// Authenticate
	if err := vc.authenticate(context.Background()); err != nil {
		return nil, fmt.Errorf("vault authentication failed: %w", err)
	}

	// Start token renewal
	vc.startTokenRenewal()

	logger.Info("vault client initialized successfully",
		zap.String("address", config.Address),
		zap.String("auth_method", config.AuthMethod),
		zap.Bool("hsm_enabled", config.HSMEnabled))

	return vc, nil
}

// validateConfig validates Vault configuration
func validateConfig(config *Config) error {
	if config.Address == "" {
		return fmt.Errorf("vault address is required")
	}

	if config.AuthMethod == "" {
		return fmt.Errorf("auth method is required")
	}

	if config.KeyMountPath == "" {
		config.KeyMountPath = "secret"
	}

	if config.TransitPath == "" {
		config.TransitPath = "transit"
	}

	if config.PKIPath == "" {
		config.PKIPath = "pki"
	}

	if config.MaxRetries == 0 {
		config.MaxRetries = 3
	}

	if config.Timeout == 0 {
		config.Timeout = 30 * time.Second
	}

	if config.RenewThreshold == 0 {
		config.RenewThreshold = 10 * time.Minute
	}

	return nil
}

// configureTLS configures TLS settings for Vault client
func configureTLS(vaultConfig *api.Config, config *Config) error {
	tlsConfig := &tls.Config{
		InsecureSkipVerify: config.TLSSkipVerify,
		ServerName:         config.TLSServerName,
	}

	// Load CA certificate if provided
	if config.CACertPath != "" {
		caCert, err := ioutil.ReadFile(config.CACertPath)
		if err != nil {
			return fmt.Errorf("failed to read CA certificate: %w", err)
		}

		caCertPool := x509.NewCertPool()
		if !caCertPool.AppendCertsFromPEM(caCert) {
			return fmt.Errorf("failed to parse CA certificate")
		}
		tlsConfig.RootCAs = caCertPool
	}

	// Load client certificate if provided
	if config.CertPath != "" && config.KeyPath != "" {
		cert, err := tls.LoadX509KeyPair(config.CertPath, config.KeyPath)
		if err != nil {
			return fmt.Errorf("failed to load client certificate: %w", err)
		}
		tlsConfig.Certificates = []tls.Certificate{cert}
	}

	vaultConfig.HttpClient.Transport = &http.Transport{
		TLSClientConfig: tlsConfig,
	}

	return nil
}

// authenticate performs initial authentication with Vault
func (vc *VaultClient) authenticate(ctx context.Context) error {
	switch vc.config.AuthMethod {
	case "token":
		return vc.authenticateWithToken()
	case "cert":
		return vc.authenticateWithCert(ctx)
	case "kubernetes":
		return vc.authenticateWithKubernetes(ctx)
	case "aws":
		return vc.authenticateWithAWS(ctx)
	default:
		return fmt.Errorf("unsupported auth method: %s", vc.config.AuthMethod)
	}
}

// authenticateWithToken authenticates using a token
func (vc *VaultClient) authenticateWithToken() error {
	if vc.config.TokenPath == "" {
		return fmt.Errorf("token path is required for token auth")
	}

	tokenBytes, err := ioutil.ReadFile(vc.config.TokenPath)
	if err != nil {
		return fmt.Errorf("failed to read token: %w", err)
	}

	token := string(tokenBytes)
	vc.client.SetToken(token)

	// Verify token
	secret, err := vc.client.Auth().Token().LookupSelf()
	if err != nil {
		return fmt.Errorf("token verification failed: %w", err)
	}

	vc.mu.Lock()
	vc.token = token
	if ttl, ok := secret.Data["ttl"].(json.Number); ok {
		if ttlInt, err := ttl.Int64(); err == nil {
			vc.tokenExp = time.Now().Add(time.Duration(ttlInt) * time.Second)
		}
	}
	vc.mu.Unlock()

	vc.logger.Info("authenticated with token", zap.Time("expires", vc.tokenExp))
	return nil
}

// authenticateWithCert authenticates using client certificates
func (vc *VaultClient) authenticateWithCert(ctx context.Context) error {
	data := map[string]interface{}{
		"name": "qslb-client",
	}

	secret, err := vc.client.Logical().WriteWithContext(ctx, "auth/cert/login", data)
	if err != nil {
		return fmt.Errorf("cert authentication failed: %w", err)
	}

	if secret.Auth == nil {
		return fmt.Errorf("no auth info returned")
	}

	vc.mu.Lock()
	vc.token = secret.Auth.ClientToken
	vc.tokenExp = time.Now().Add(time.Duration(secret.Auth.LeaseDuration) * time.Second)
	vc.mu.Unlock()

	vc.client.SetToken(vc.token)
	vc.logger.Info("authenticated with certificate", zap.Time("expires", vc.tokenExp))
	return nil
}

// authenticateWithKubernetes authenticates using Kubernetes service account
func (vc *VaultClient) authenticateWithKubernetes(ctx context.Context) error {
	jwt, err := ioutil.ReadFile("/var/run/secrets/kubernetes.io/serviceaccount/token")
	if err != nil {
		return fmt.Errorf("failed to read service account token: %w", err)
	}

	data := map[string]interface{}{
		"role": "qslb-service",
		"jwt":  string(jwt),
	}

	secret, err := vc.client.Logical().WriteWithContext(ctx, "auth/kubernetes/login", data)
	if err != nil {
		return fmt.Errorf("kubernetes authentication failed: %w", err)
	}

	if secret.Auth == nil {
		return fmt.Errorf("no auth info returned")
	}

	vc.mu.Lock()
	vc.token = secret.Auth.ClientToken
	vc.tokenExp = time.Now().Add(time.Duration(secret.Auth.LeaseDuration) * time.Second)
	vc.mu.Unlock()

	vc.client.SetToken(vc.token)
	vc.logger.Info("authenticated with kubernetes", zap.Time("expires", vc.tokenExp))
	return nil
}

// authenticateWithAWS authenticates using AWS IAM
func (vc *VaultClient) authenticateWithAWS(ctx context.Context) error {
	// This would integrate with AWS SDK for IAM authentication
	// For now, return not implemented
	return fmt.Errorf("AWS authentication not implemented")
}

// startTokenRenewal starts automatic token renewal
func (vc *VaultClient) startTokenRenewal() {
	vc.mu.Lock()
	defer vc.mu.Unlock()

	if vc.tokenExp.IsZero() {
		return
	}

	renewAt := vc.tokenExp.Add(-vc.config.RenewThreshold)
	if renewAt.Before(time.Now()) {
		// Token expires too soon, renew immediately
		go vc.renewToken()
		return
	}

	vc.renewTimer = time.AfterFunc(time.Until(renewAt), func() {
		vc.renewToken()
	})
}

// renewToken renews the Vault token
func (vc *VaultClient) renewToken() {
	ctx, cancel := context.WithTimeout(context.Background(), vc.config.Timeout)
	defer cancel()

	secret, err := vc.client.Auth().Token().RenewSelfWithContext(ctx, 0)
	if err != nil {
		vc.logger.Error("failed to renew token", zap.Error(err))
		// Try to re-authenticate
		if authErr := vc.authenticate(ctx); authErr != nil {
			vc.logger.Error("failed to re-authenticate", zap.Error(authErr))
		}
		return
	}

	vc.mu.Lock()
	if secret.Auth != nil {
		vc.tokenExp = time.Now().Add(time.Duration(secret.Auth.LeaseDuration) * time.Second)
	}
	vc.mu.Unlock()

	vc.logger.Info("token renewed successfully", zap.Time("expires", vc.tokenExp))
	vc.startTokenRenewal() // Schedule next renewal
}

// GenerateKey generates a new cryptographic key
func (vc *VaultClient) GenerateKey(ctx context.Context, keyName, keyType string, keySize int) (*KeyMetadata, error) {
	path := fmt.Sprintf("%s/keys/%s", vc.config.TransitPath, keyName)
	
	data := map[string]interface{}{
		"type": keyType,
	}
	
	if keySize > 0 {
		data["key_size"] = keySize
	}

	// Enable HSM if configured
	if vc.config.HSMEnabled {
		data["managed_key_name"] = fmt.Sprintf("hsm-%s", keyName)
		data["managed_key_id"] = fmt.Sprintf("hsm-slot-%d", vc.config.HSMSlotID)
	}

	secret, err := vc.client.Logical().WriteWithContext(ctx, path, data)
	if err != nil {
		return nil, fmt.Errorf("failed to generate key: %w", err)
	}

	metadata := &KeyMetadata{
		ID:        keyName,
		Name:      keyName,
		Type:      keyType,
		KeySize:   keySize,
		Created:   time.Now(),
		Version:   1,
		HSMBacked: vc.config.HSMEnabled,
		Usage:     []string{"encrypt", "decrypt"},
		Metadata:  make(map[string]string),
	}

	vc.logger.Info("key generated successfully",
		zap.String("key_name", keyName),
		zap.String("key_type", keyType),
		zap.Int("key_size", keySize),
		zap.Bool("hsm_backed", vc.config.HSMEnabled))

	return metadata, nil
}

// RotateKey rotates an existing key
func (vc *VaultClient) RotateKey(ctx context.Context, keyName string) (*KeyMetadata, error) {
	path := fmt.Sprintf("%s/keys/%s/rotate", vc.config.TransitPath, keyName)

	secret, err := vc.client.Logical().WriteWithContext(ctx, path, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to rotate key: %w", err)
	}

	// Get updated key info
	return vc.GetKeyMetadata(ctx, keyName)
}

// GetKeyMetadata retrieves key metadata
func (vc *VaultClient) GetKeyMetadata(ctx context.Context, keyName string) (*KeyMetadata, error) {
	path := fmt.Sprintf("%s/keys/%s", vc.config.TransitPath, keyName)

	secret, err := vc.client.Logical().ReadWithContext(ctx, path)
	if err != nil {
		return nil, fmt.Errorf("failed to get key metadata: %w", err)
	}

	if secret == nil || secret.Data == nil {
		return nil, fmt.Errorf("key not found: %s", keyName)
	}

	metadata := &KeyMetadata{
		ID:   keyName,
		Name: keyName,
	}

	if keyType, ok := secret.Data["type"].(string); ok {
		metadata.Type = keyType
	}

	if version, ok := secret.Data["latest_version"].(json.Number); ok {
		if v, err := version.Int64(); err == nil {
			metadata.Version = int(v)
		}
	}

	if created, ok := secret.Data["creation_time"].(string); ok {
		if t, err := time.Parse(time.RFC3339, created); err == nil {
			metadata.Created = t
		}
	}

	return metadata, nil
}

// Encrypt encrypts data using a managed key
func (vc *VaultClient) Encrypt(ctx context.Context, keyName string, plaintext []byte) ([]byte, error) {
	path := fmt.Sprintf("%s/encrypt/%s", vc.config.TransitPath, keyName)

	data := map[string]interface{}{
		"plaintext": plaintext,
	}

	secret, err := vc.client.Logical().WriteWithContext(ctx, path, data)
	if err != nil {
		return nil, fmt.Errorf("encryption failed: %w", err)
	}

	if secret.Data == nil {
		return nil, fmt.Errorf("no data returned from encryption")
	}

	ciphertext, ok := secret.Data["ciphertext"].(string)
	if !ok {
		return nil, fmt.Errorf("invalid ciphertext format")
	}

	return []byte(ciphertext), nil
}

// Decrypt decrypts data using a managed key
func (vc *VaultClient) Decrypt(ctx context.Context, keyName string, ciphertext []byte) ([]byte, error) {
	path := fmt.Sprintf("%s/decrypt/%s", vc.config.TransitPath, keyName)

	data := map[string]interface{}{
		"ciphertext": string(ciphertext),
	}

	secret, err := vc.client.Logical().WriteWithContext(ctx, path, data)
	if err != nil {
		return nil, fmt.Errorf("decryption failed: %w", err)
	}

	if secret.Data == nil {
		return nil, fmt.Errorf("no data returned from decryption")
	}

	plaintext, ok := secret.Data["plaintext"].(string)
	if !ok {
		return nil, fmt.Errorf("invalid plaintext format")
	}

	return []byte(plaintext), nil
}

// Sign signs data using a managed key
func (vc *VaultClient) Sign(ctx context.Context, keyName string, data []byte, algorithm string) ([]byte, error) {
	path := fmt.Sprintf("%s/sign/%s", vc.config.TransitPath, keyName)

	requestData := map[string]interface{}{
		"input": data,
	}

	if algorithm != "" {
		requestData["signature_algorithm"] = algorithm
	}

	secret, err := vc.client.Logical().WriteWithContext(ctx, path, requestData)
	if err != nil {
		return nil, fmt.Errorf("signing failed: %w", err)
	}

	if secret.Data == nil {
		return nil, fmt.Errorf("no data returned from signing")
	}

	signature, ok := secret.Data["signature"].(string)
	if !ok {
		return nil, fmt.Errorf("invalid signature format")
	}

	return []byte(signature), nil
}

// Verify verifies a signature using a managed key
func (vc *VaultClient) Verify(ctx context.Context, keyName string, data, signature []byte, algorithm string) (bool, error) {
	path := fmt.Sprintf("%s/verify/%s", vc.config.TransitPath, keyName)

	requestData := map[string]interface{}{
		"input":     data,
		"signature": string(signature),
	}

	if algorithm != "" {
		requestData["signature_algorithm"] = algorithm
	}

	secret, err := vc.client.Logical().WriteWithContext(ctx, path, requestData)
	if err != nil {
		return false, fmt.Errorf("verification failed: %w", err)
	}

	if secret.Data == nil {
		return false, fmt.Errorf("no data returned from verification")
	}

	valid, ok := secret.Data["valid"].(bool)
	if !ok {
		return false, fmt.Errorf("invalid verification result format")
	}

	return valid, nil
}

// GenerateCertificate generates a certificate using PKI backend
func (vc *VaultClient) GenerateCertificate(ctx context.Context, role string, commonName string, altNames []string, ttl time.Duration) ([]byte, []byte, error) {
	path := fmt.Sprintf("%s/issue/%s", vc.config.PKIPath, role)

	data := map[string]interface{}{
		"common_name": commonName,
		"ttl":         ttl.String(),
	}

	if len(altNames) > 0 {
		data["alt_names"] = altNames
	}

	secret, err := vc.client.Logical().WriteWithContext(ctx, path, data)
	if err != nil {
		return nil, nil, fmt.Errorf("certificate generation failed: %w", err)
	}

	if secret.Data == nil {
		return nil, nil, fmt.Errorf("no data returned from certificate generation")
	}

	cert, ok := secret.Data["certificate"].(string)
	if !ok {
		return nil, nil, fmt.Errorf("invalid certificate format")
	}

	privateKey, ok := secret.Data["private_key"].(string)
	if !ok {
		return nil, nil, fmt.Errorf("invalid private key format")
	}

	return []byte(cert), []byte(privateKey), nil
}

// SetupKeyRotationPolicy sets up automatic key rotation
func (vc *VaultClient) SetupKeyRotationPolicy(ctx context.Context, policy *KeyRotationPolicy) error {
	if !policy.AutoRotate {
		return nil
	}

	// This would integrate with a job scheduler or cron system
	// For now, we'll just log the policy setup
	vc.logger.Info("key rotation policy configured",
		zap.String("key_name", policy.KeyName),
		zap.Duration("rotation_period", policy.RotationPeriod),
		zap.Bool("auto_rotate", policy.AutoRotate))

	return nil
}

// ListKeys lists all managed keys
func (vc *VaultClient) ListKeys(ctx context.Context) ([]string, error) {
	path := fmt.Sprintf("%s/keys", vc.config.TransitPath)

	secret, err := vc.client.Logical().ListWithContext(ctx, path)
	if err != nil {
		return nil, fmt.Errorf("failed to list keys: %w", err)
	}

	if secret == nil || secret.Data == nil {
		return []string{}, nil
	}

	keys, ok := secret.Data["keys"].([]interface{})
	if !ok {
		return []string{}, nil
	}

	result := make([]string, 0, len(keys))
	for _, key := range keys {
		if keyName, ok := key.(string); ok {
			result = append(result, keyName)
		}
	}

	return result, nil
}

// DeleteKey deletes a managed key
func (vc *VaultClient) DeleteKey(ctx context.Context, keyName string) error {
	// First, update key to allow deletion
	configPath := fmt.Sprintf("%s/keys/%s/config", vc.config.TransitPath, keyName)
	configData := map[string]interface{}{
		"deletion_allowed": true,
	}

	if _, err := vc.client.Logical().WriteWithContext(ctx, configPath, configData); err != nil {
		return fmt.Errorf("failed to configure key for deletion: %w", err)
	}

	// Delete the key
	deletePath := fmt.Sprintf("%s/keys/%s", vc.config.TransitPath, keyName)
	if _, err := vc.client.Logical().DeleteWithContext(ctx, deletePath); err != nil {
		return fmt.Errorf("failed to delete key: %w", err)
	}

	vc.logger.Info("key deleted successfully", zap.String("key_name", keyName))
	return nil
}

// GetHealth checks Vault health status
func (vc *VaultClient) GetHealth(ctx context.Context) (map[string]interface{}, error) {
	health, err := vc.client.Sys().HealthWithContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get health status: %w", err)
	}

	return map[string]interface{}{
		"initialized":    health.Initialized,
		"sealed":         health.Sealed,
		"standby":        health.Standby,
		"version":        health.Version,
		"cluster_name":   health.ClusterName,
		"cluster_id":     health.ClusterID,
		"replication_dr": health.ReplicationDRMode,
		"replication_pr": health.ReplicationPerfMode,
	}, nil
}

// Close closes the Vault client and stops token renewal
func (vc *VaultClient) Close() {
	vc.mu.Lock()
	defer vc.mu.Unlock()

	if vc.renewTimer != nil {
		vc.renewTimer.Stop()
		vc.renewTimer = nil
	}

	vc.logger.Info("vault client closed")
}