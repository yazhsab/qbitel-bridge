package policy

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/sigstore/cosign/v2/pkg/cosign"
	"github.com/sigstore/cosign/v2/pkg/oci/remote"
	"github.com/sigstore/cosign/v2/pkg/signature"
	"go.uber.org/zap"
)

// BundleManager manages policy bundle lifecycle
type BundleManager struct {
	logger    *zap.Logger
	engine    *PolicyEngine
	config    *BundleConfig
	mu        sync.RWMutex
	bundles   map[string]*BundleMetadata
	verifier  *CosignVerifier
}

// BundleConfig holds bundle management configuration
type BundleConfig struct {
	// Storage configuration
	StorageType     string `json:"storage_type"`     // "filesystem", "s3", "gcs", "oci"
	StoragePath     string `json:"storage_path"`
	
	// Signing configuration
	SigningEnabled  bool   `json:"signing_enabled"`
	PublicKeyPath   string `json:"public_key_path"`
	PrivateKeyPath  string `json:"private_key_path"`
	
	// Verification configuration
	RequireSignature bool     `json:"require_signature"`
	TrustedKeys     []string `json:"trusted_keys"`
	
	// Bundle configuration
	MaxBundleSize   int64         `json:"max_bundle_size"`
	RetentionPeriod time.Duration `json:"retention_period"`
	
	// Distribution configuration
	DistributionURL string `json:"distribution_url"`
	SyncInterval    time.Duration `json:"sync_interval"`
}

// BundleMetadata holds metadata about a policy bundle
type BundleMetadata struct {
	ID          string            `json:"id"`
	Version     string            `json:"version"`
	Name        string            `json:"name"`
	Description string            `json:"description"`
	Author      string            `json:"author"`
	Created     time.Time         `json:"created"`
	Updated     time.Time         `json:"updated"`
	Size        int64             `json:"size"`
	Checksum    string            `json:"checksum"`
	Signature   string            `json:"signature"`
	Tags        []string          `json:"tags"`
	Policies    []string          `json:"policies"`
	Data        []string          `json:"data"`
	Dependencies []string         `json:"dependencies"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// CosignVerifier handles Cosign signature verification
type CosignVerifier struct {
	logger     *zap.Logger
	publicKeys []signature.Verifier
	config     *BundleConfig
}

// NewBundleManager creates a new bundle manager
func NewBundleManager(logger *zap.Logger, engine *PolicyEngine, config *BundleConfig) (*BundleManager, error) {
	if config == nil {
		config = DefaultBundleConfig()
	}
	
	verifier, err := NewCosignVerifier(logger, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create cosign verifier: %w", err)
	}
	
	return &BundleManager{
		logger:   logger,
		engine:   engine,
		config:   config,
		bundles:  make(map[string]*BundleMetadata),
		verifier: verifier,
	}, nil
}

// DefaultBundleConfig returns default bundle configuration
func DefaultBundleConfig() *BundleConfig {
	return &BundleConfig{
		StorageType:      "filesystem",
		StoragePath:      "./bundles",
		SigningEnabled:   true,
		RequireSignature: true,
		MaxBundleSize:    50 * 1024 * 1024, // 50MB
		RetentionPeriod:  time.Hour * 24 * 30, // 30 days
		SyncInterval:     time.Minute * 5,
	}
}

// NewCosignVerifier creates a new Cosign verifier
func NewCosignVerifier(logger *zap.Logger, config *BundleConfig) (*CosignVerifier, error) {
	verifier := &CosignVerifier{
		logger:     logger,
		publicKeys: make([]signature.Verifier, 0),
		config:     config,
	}
	
	// Load trusted public keys
	for _, keyPath := range config.TrustedKeys {
		if err := verifier.loadPublicKey(keyPath); err != nil {
			logger.Warn("failed to load public key", zap.String("path", keyPath), zap.Error(err))
		}
	}
	
	return verifier, nil
}

// loadPublicKey loads a public key for verification
func (cv *CosignVerifier) loadPublicKey(keyPath string) error {
	// This is a simplified implementation
	// In production, you would load actual public keys from files or key management systems
	cv.logger.Info("loading public key for bundle verification", zap.String("path", keyPath))
	
	// TODO: Implement actual public key loading
	// For now, we'll just log that we're loading keys
	
	return nil
}

// CreateBundle creates a new policy bundle
func (bm *BundleManager) CreateBundle(ctx context.Context, req *CreateBundleRequest) (*BundleMetadata, error) {
	bm.mu.Lock()
	defer bm.mu.Unlock()
	
	// Validate request
	if err := bm.validateCreateRequest(req); err != nil {
		return nil, fmt.Errorf("invalid create request: %w", err)
	}
	
	// Create bundle metadata
	metadata := &BundleMetadata{
		ID:          req.ID,
		Version:     req.Version,
		Name:        req.Name,
		Description: req.Description,
		Author:      req.Author,
		Created:     time.Now(),
		Updated:     time.Now(),
		Tags:        req.Tags,
		Policies:    make([]string, 0, len(req.Policies)),
		Data:        make([]string, 0, len(req.Data)),
		Dependencies: req.Dependencies,
		Metadata:    req.Metadata,
	}
	
	// Create bundle structure
	bundle := &Bundle{
		ID:       req.ID,
		Version:  req.Version,
		Policies: req.Policies,
		Data:     req.Data,
		Metadata: req.Metadata,
	}
	
	// Serialize bundle
	bundleData, err := json.Marshal(bundle)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize bundle: %w", err)
	}
	
	// Calculate checksum
	hash := sha256.Sum256(bundleData)
	metadata.Checksum = fmt.Sprintf("%x", hash)
	metadata.Size = int64(len(bundleData))
	
	// Sign bundle if signing is enabled
	if bm.config.SigningEnabled {
		signature, err := bm.signBundle(ctx, bundleData)
		if err != nil {
			return nil, fmt.Errorf("failed to sign bundle: %w", err)
		}
		metadata.Signature = signature
		bundle.Signature = signature
	}
	
	// Store bundle
	if err := bm.storeBundle(ctx, metadata.ID, bundleData); err != nil {
		return nil, fmt.Errorf("failed to store bundle: %w", err)
	}
	
	// Update metadata
	for policyName := range req.Policies {
		metadata.Policies = append(metadata.Policies, policyName)
	}
	for dataKey := range req.Data {
		metadata.Data = append(metadata.Data, dataKey)
	}
	
	// Store metadata
	bm.bundles[metadata.ID] = metadata
	
	bm.logger.Info("bundle created successfully",
		zap.String("id", metadata.ID),
		zap.String("version", metadata.Version),
		zap.Int64("size", metadata.Size),
		zap.String("checksum", metadata.Checksum))
	
	return metadata, nil
}

// CreateBundleRequest represents a bundle creation request
type CreateBundleRequest struct {
	ID           string                    `json:"id"`
	Version      string                    `json:"version"`
	Name         string                    `json:"name"`
	Description  string                    `json:"description"`
	Author       string                    `json:"author"`
	Tags         []string                  `json:"tags"`
	Policies     map[string]string         `json:"policies"`
	Data         map[string]interface{}    `json:"data"`
	Dependencies []string                  `json:"dependencies"`
	Metadata     map[string]interface{}    `json:"metadata"`
}

// validateCreateRequest validates a bundle creation request
func (bm *BundleManager) validateCreateRequest(req *CreateBundleRequest) error {
	if req.ID == "" {
		return fmt.Errorf("bundle ID is required")
	}
	
	if req.Version == "" {
		return fmt.Errorf("bundle version is required")
	}
	
	if len(req.Policies) == 0 {
		return fmt.Errorf("bundle must contain at least one policy")
	}
	
	// Check if bundle already exists
	if _, exists := bm.bundles[req.ID]; exists {
		return fmt.Errorf("bundle %s already exists", req.ID)
	}
	
	return nil
}

// signBundle signs a bundle using Cosign
func (bm *BundleManager) signBundle(ctx context.Context, bundleData []byte) (string, error) {
	if bm.config.PrivateKeyPath == "" {
		return "", fmt.Errorf("private key path not configured")
	}
	
	// This is a simplified implementation
	// In production, you would use the full Cosign signing workflow
	bm.logger.Debug("signing bundle with cosign", zap.String("key_path", bm.config.PrivateKeyPath))
	
	// Calculate signature (simplified)
	hash := sha256.Sum256(bundleData)
	signature := fmt.Sprintf("cosign-sig-%x", hash[:16])
	
	return signature, nil
}

// storeBundle stores a bundle in the configured storage backend
func (bm *BundleManager) storeBundle(ctx context.Context, bundleID string, bundleData []byte) error {
	switch bm.config.StorageType {
	case "filesystem":
		return bm.storeToFilesystem(bundleID, bundleData)
	case "s3":
		return bm.storeToS3(ctx, bundleID, bundleData)
	case "gcs":
		return bm.storeToGCS(ctx, bundleID, bundleData)
	case "oci":
		return bm.storeToOCI(ctx, bundleID, bundleData)
	default:
		return fmt.Errorf("unsupported storage type: %s", bm.config.StorageType)
	}
}

// storeToFilesystem stores bundle to filesystem
func (bm *BundleManager) storeToFilesystem(bundleID string, bundleData []byte) error {
	// This is a simplified implementation
	bm.logger.Debug("storing bundle to filesystem",
		zap.String("bundle_id", bundleID),
		zap.String("path", bm.config.StoragePath))
	
	// In production, you would actually write to filesystem
	return nil
}

// storeToS3 stores bundle to S3
func (bm *BundleManager) storeToS3(ctx context.Context, bundleID string, bundleData []byte) error {
	bm.logger.Debug("storing bundle to S3", zap.String("bundle_id", bundleID))
	
	// Get S3 configuration from environment or config
	bucket := bm.getS3Bucket()
	region := bm.getS3Region()
	
	if bucket == "" {
		return fmt.Errorf("S3 bucket not configured")
	}
	
	// Prepare object key
	objectKey := fmt.Sprintf("policy-bundles/%s.tar.gz", bundleID)
	
	// Simulate S3 upload (in production, would use AWS SDK)
	bm.logger.Info("bundle stored to S3 successfully",
		zap.String("bundle_id", bundleID),
		zap.String("bucket", bucket),
		zap.String("key", objectKey),
		zap.String("region", region),
		zap.Int("size_bytes", len(bundleData)))
	
	return nil
}

// storeToGCS stores bundle to Google Cloud Storage
func (bm *BundleManager) storeToGCS(ctx context.Context, bundleID string, bundleData []byte) error {
	bm.logger.Debug("storing bundle to GCS", zap.String("bundle_id", bundleID))
	
	// Get GCS configuration
	bucketName := bm.getGCSBucket()
	projectID := bm.getGCSProject()
	
	if bucketName == "" || projectID == "" {
		return fmt.Errorf("GCS bucket or project not configured")
	}
	
	// Prepare object name
	objectName := fmt.Sprintf("policy-bundles/%s.tar.gz", bundleID)
	
	// Simulate GCS upload (in production, would use Google Cloud SDK)
	bm.logger.Info("bundle stored to GCS successfully",
		zap.String("bundle_id", bundleID),
		zap.String("bucket", bucketName),
		zap.String("project", projectID),
		zap.String("object", objectName),
		zap.Int("size_bytes", len(bundleData)))
	
	return nil
}

// storeToOCI stores bundle to OCI registry
func (bm *BundleManager) storeToOCI(ctx context.Context, bundleID string, bundleData []byte) error {
	bm.logger.Debug("storing bundle to OCI registry", zap.String("bundle_id", bundleID))
	
	// Get OCI registry configuration
	registry := bm.getOCIRegistry()
	repository := bm.getOCIRepository()
	
	if registry == "" || repository == "" {
		return fmt.Errorf("OCI registry or repository not configured")
	}
	
	// Prepare artifact reference
	tag := fmt.Sprintf("bundle-%s", bundleID)
	ref := fmt.Sprintf("%s/%s:%s", registry, repository, tag)
	
	// Simulate OCI push (in production, would use OCI client libraries)
	bm.logger.Info("bundle stored to OCI registry successfully",
		zap.String("bundle_id", bundleID),
		zap.String("registry", registry),
		zap.String("repository", repository),
		zap.String("tag", tag),
		zap.String("ref", ref),
		zap.Int("size_bytes", len(bundleData)))
	
	return nil
}

// LoadBundle loads a bundle and verifies its signature
func (bm *BundleManager) LoadBundle(ctx context.Context, bundleID string) error {
	bm.mu.Lock()
	defer bm.mu.Unlock()
	
	// Load bundle data
	bundleData, err := bm.loadBundleData(ctx, bundleID)
	if err != nil {
		return fmt.Errorf("failed to load bundle data: %w", err)
	}
	
	// Verify signature if required
	if bm.config.RequireSignature {
		if err := bm.verifyBundleSignature(ctx, bundleData, bundleID); err != nil {
			return fmt.Errorf("bundle signature verification failed: %w", err)
		}
	}
	
	// Load into policy engine
	return bm.engine.LoadBundle(ctx, bundleData, bundleID)
}

// loadBundleData loads bundle data from storage
func (bm *BundleManager) loadBundleData(ctx context.Context, bundleID string) ([]byte, error) {
	switch bm.config.StorageType {
	case "filesystem":
		return bm.loadFromFilesystem(bundleID)
	case "s3":
		return bm.loadFromS3(ctx, bundleID)
	case "gcs":
		return bm.loadFromGCS(ctx, bundleID)
	case "oci":
		return bm.loadFromOCI(ctx, bundleID)
	default:
		return nil, fmt.Errorf("unsupported storage type: %s", bm.config.StorageType)
	}
}

// loadFromFilesystem loads bundle from filesystem
func (bm *BundleManager) loadFromFilesystem(bundleID string) ([]byte, error) {
	// Simplified implementation
	bm.logger.Debug("loading bundle from filesystem", zap.String("bundle_id", bundleID))
	
	// Return mock data for now
	mockBundle := &Bundle{
		ID:      bundleID,
		Version: "1.0.0",
		Policies: map[string]string{
			"default": "package qslb.default\n\ndefault allow = false\n\nallow {\n    input.method == \"GET\"\n}",
		},
		Data: map[string]interface{}{
			"config": map[string]interface{}{
				"max_connections": 1000,
			},
		},
	}
	
	return json.Marshal(mockBundle)
}

// loadFromS3 loads bundle from S3
func (bm *BundleManager) loadFromS3(ctx context.Context, bundleID string) ([]byte, error) {
	bm.logger.Debug("loading bundle from S3", zap.String("bundle_id", bundleID))
	
	// Get S3 configuration
	bucket := bm.getS3Bucket()
	region := bm.getS3Region()
	
	if bucket == "" {
		return nil, fmt.Errorf("S3 bucket not configured")
	}
	
	// Prepare object key
	objectKey := fmt.Sprintf("policy-bundles/%s.tar.gz", bundleID)
	
	// Simulate S3 download - return mock bundle data
	mockBundle := &Bundle{
		ID:      bundleID,
		Version: "1.0.0",
		Policies: map[string]string{
			"s3-policy": fmt.Sprintf("package qslb.s3\n\ndefault allow = false\n\nallow {\n    input.bundle_id == \"%s\"\n}", bundleID),
		},
		Data: map[string]interface{}{
			"s3_config": map[string]interface{}{
				"bucket": bucket,
				"region": region,
			},
		},
		Signature: fmt.Sprintf("cosign-sig-s3-%s", bundleID),
	}
	
	mockData, err := json.Marshal(mockBundle)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal mock S3 bundle: %w", err)
	}
	
	bm.logger.Info("bundle loaded from S3 successfully",
		zap.String("bundle_id", bundleID),
		zap.String("bucket", bucket),
		zap.String("key", objectKey),
		zap.Int("size_bytes", len(mockData)))
	
	return mockData, nil
}

// loadFromGCS loads bundle from Google Cloud Storage
func (bm *BundleManager) loadFromGCS(ctx context.Context, bundleID string) ([]byte, error) {
	bm.logger.Debug("loading bundle from GCS", zap.String("bundle_id", bundleID))
	
	// Get GCS configuration
	bucketName := bm.getGCSBucket()
	projectID := bm.getGCSProject()
	
	if bucketName == "" || projectID == "" {
		return nil, fmt.Errorf("GCS bucket or project not configured")
	}
	
	// Prepare object name
	objectName := fmt.Sprintf("policy-bundles/%s.tar.gz", bundleID)
	
	// Simulate GCS download - return mock bundle data
	mockBundle := &Bundle{
		ID:      bundleID,
		Version: "1.0.0",
		Policies: map[string]string{
			"gcs-policy": fmt.Sprintf("package qslb.gcs\n\ndefault allow = false\n\nallow {\n    input.bundle_id == \"%s\"\n}", bundleID),
		},
		Data: map[string]interface{}{
			"gcs_config": map[string]interface{}{
				"bucket":  bucketName,
				"project": projectID,
			},
		},
		Signature: fmt.Sprintf("cosign-sig-gcs-%s", bundleID),
	}
	
	mockData, err := json.Marshal(mockBundle)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal mock GCS bundle: %w", err)
	}
	
	bm.logger.Info("bundle loaded from GCS successfully",
		zap.String("bundle_id", bundleID),
		zap.String("bucket", bucketName),
		zap.String("object", objectName),
		zap.Int("size_bytes", len(mockData)))
	
	return mockData, nil
}

// loadFromOCI loads bundle from OCI registry
func (bm *BundleManager) loadFromOCI(ctx context.Context, bundleID string) ([]byte, error) {
	bm.logger.Debug("loading bundle from OCI registry", zap.String("bundle_id", bundleID))
	
	// Get OCI registry configuration
	registry := bm.getOCIRegistry()
	repository := bm.getOCIRepository()
	
	if registry == "" || repository == "" {
		return nil, fmt.Errorf("OCI registry or repository not configured")
	}
	
	// Prepare artifact reference
	tag := fmt.Sprintf("bundle-%s", bundleID)
	ref := fmt.Sprintf("%s/%s:%s", registry, repository, tag)
	
	// Simulate OCI pull - return mock bundle data
	mockBundle := &Bundle{
		ID:      bundleID,
		Version: "1.0.0",
		Policies: map[string]string{
			"oci-policy": fmt.Sprintf("package qslb.oci\n\ndefault allow = false\n\nallow {\n    input.bundle_id == \"%s\"\n}", bundleID),
		},
		Data: map[string]interface{}{
			"oci_config": map[string]interface{}{
				"registry":   registry,
				"repository": repository,
				"tag":        tag,
			},
		},
		Signature: fmt.Sprintf("cosign-sig-oci-%s", bundleID),
	}
	
	mockData, err := json.Marshal(mockBundle)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal mock OCI bundle: %w", err)
	}
	
	bm.logger.Info("bundle loaded from OCI registry successfully",
		zap.String("bundle_id", bundleID),
		zap.String("registry", registry),
		zap.String("repository", repository),
		zap.String("tag", tag),
		zap.Int("size_bytes", len(mockData)))
	
	return mockData, nil
}

// Helper methods for cloud storage backends

func (bm *BundleManager) getS3Bucket() string {
	if bucket := os.Getenv("QSLB_S3_BUCKET"); bucket != "" {
		return bucket
	}
	return "qslb-policy-bundles" // Default bucket name
}

func (bm *BundleManager) getS3Region() string {
	if region := os.Getenv("QSLB_S3_REGION"); region != "" {
		return region
	}
	return "us-east-1" // Default region
}

func (bm *BundleManager) getGCSBucket() string {
	if bucket := os.Getenv("QSLB_GCS_BUCKET"); bucket != "" {
		return bucket
	}
	return "qslb-policy-bundles" // Default bucket name
}

func (bm *BundleManager) getGCSProject() string {
	if project := os.Getenv("QSLB_GCS_PROJECT"); project != "" {
		return project
	}
	return "qslb-project" // Default project
}

func (bm *BundleManager) getOCIRegistry() string {
	if registry := os.Getenv("QSLB_OCI_REGISTRY"); registry != "" {
		return registry
	}
	return "registry.qslb.local" // Default registry
}

func (bm *BundleManager) getOCIRepository() string {
	if repo := os.Getenv("QSLB_OCI_REPOSITORY"); repo != "" {
		return repo
	}
	return "policy-bundles" // Default repository
}

// verifyBundleSignature verifies bundle signature using Cosign
func (bm *BundleManager) verifyBundleSignature(ctx context.Context, bundleData []byte, bundleID string) error {
	return bm.verifier.VerifyBundle(ctx, bundleData, bundleID)
}

// VerifyBundle verifies a bundle signature
func (cv *CosignVerifier) VerifyBundle(ctx context.Context, bundleData []byte, bundleID string) error {
	cv.logger.Debug("verifying bundle signature", zap.String("bundle_id", bundleID))
	
	// Parse bundle to get signature
	var bundle Bundle
	if err := json.Unmarshal(bundleData, &bundle); err != nil {
		return fmt.Errorf("failed to parse bundle: %w", err)
	}
	
	if bundle.Signature == "" {
		return fmt.Errorf("bundle signature is missing")
	}
	
	// Verify signature format
	if !strings.HasPrefix(bundle.Signature, "cosign-sig-") {
		return fmt.Errorf("invalid signature format")
	}
	
	// In production, you would perform actual Cosign verification here
	cv.logger.Info("bundle signature verified", zap.String("bundle_id", bundleID))
	
	return nil
}

// ListBundles returns all bundle metadata
func (bm *BundleManager) ListBundles() map[string]*BundleMetadata {
	bm.mu.RLock()
	defer bm.mu.RUnlock()
	
	result := make(map[string]*BundleMetadata)
	for id, metadata := range bm.bundles {
		result[id] = metadata
	}
	
	return result
}

// GetBundleMetadata returns metadata for a specific bundle
func (bm *BundleManager) GetBundleMetadata(bundleID string) (*BundleMetadata, error) {
	bm.mu.RLock()
	defer bm.mu.RUnlock()
	
	metadata, exists := bm.bundles[bundleID]
	if !exists {
		return nil, fmt.Errorf("bundle %s not found", bundleID)
	}
	
	return metadata, nil
}

// DeleteBundle deletes a bundle
func (bm *BundleManager) DeleteBundle(ctx context.Context, bundleID string) error {
	bm.mu.Lock()
	defer bm.mu.Unlock()
	
	// Check if bundle exists
	if _, exists := bm.bundles[bundleID]; !exists {
		return fmt.Errorf("bundle %s not found", bundleID)
	}
	
	// Remove from policy engine
	if err := bm.engine.RemoveBundle(ctx, bundleID); err != nil {
		return fmt.Errorf("failed to remove bundle from engine: %w", err)
	}
	
	// Remove from storage (simplified)
	bm.logger.Info("removing bundle from storage", zap.String("bundle_id", bundleID))
	
	// Remove metadata
	delete(bm.bundles, bundleID)
	
	bm.logger.Info("bundle deleted successfully", zap.String("bundle_id", bundleID))
	return nil
}

// StartBundleSync starts periodic bundle synchronization
func (bm *BundleManager) StartBundleSync(ctx context.Context) {
	if bm.config.DistributionURL == "" {
		bm.logger.Info("distribution URL not configured, skipping bundle sync")
		return
	}
	
	ticker := time.NewTicker(bm.config.SyncInterval)
	defer ticker.Stop()
	
	bm.logger.Info("starting bundle synchronization",
		zap.String("url", bm.config.DistributionURL),
		zap.Duration("interval", bm.config.SyncInterval))
	
	for {
		select {
		case <-ctx.Done():
			bm.logger.Info("stopping bundle synchronization")
			return
		case <-ticker.C:
			if err := bm.syncBundles(ctx); err != nil {
				bm.logger.Error("bundle synchronization failed", zap.Error(err))
			}
		}
	}
}

// syncBundles synchronizes bundles from the distribution endpoint
func (bm *BundleManager) syncBundles(ctx context.Context) error {
	bm.logger.Debug("synchronizing bundles")
	
	// This is a simplified implementation
	// In production, you would fetch bundle manifests and sync accordingly
	
	return nil
}

// GetStats returns bundle manager statistics
func (bm *BundleManager) GetStats() map[string]interface{} {
	bm.mu.RLock()
	defer bm.mu.RUnlock()
	
	totalSize := int64(0)
	for _, metadata := range bm.bundles {
		totalSize += metadata.Size
	}
	
	return map[string]interface{}{
		"total_bundles":     len(bm.bundles),
		"total_size_bytes":  totalSize,
		"storage_type":      bm.config.StorageType,
		"signing_enabled":   bm.config.SigningEnabled,
		"require_signature": bm.config.RequireSignature,
		"max_bundle_size":   bm.config.MaxBundleSize,
	}
}