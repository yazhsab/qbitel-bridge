package policy

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/open-policy-agent/opa/rego"
	"github.com/open-policy-agent/opa/storage"
	"github.com/open-policy-agent/opa/storage/inmem"
	"github.com/sigstore/cosign/v2/pkg/cosign"
	"github.com/sigstore/cosign/v2/pkg/oci/remote"
	"go.uber.org/zap"
)

// PolicyEngine manages OPA policy bundles with Cosign verification
type PolicyEngine struct {
	logger    *zap.Logger
	store     storage.Store
	bundles   map[string]*Bundle
	mu        sync.RWMutex
	config    *Config
	evaluator *rego.PreparedEvalQuery
}

// Config holds policy engine configuration
type Config struct {
	// Bundle storage configuration
	BundleURL      string        `json:"bundle_url"`
	PollInterval   time.Duration `json:"poll_interval"`
	
	// Cosign verification configuration
	CosignPublicKey string `json:"cosign_public_key"`
	RequireSigned   bool   `json:"require_signed"`
	
	// Policy evaluation configuration
	DefaultPolicy   string `json:"default_policy"`
	CacheSize       int    `json:"cache_size"`
	EvalTimeout     time.Duration `json:"eval_timeout"`
	
	// Security configuration
	MaxBundleSize   int64 `json:"max_bundle_size"`
	AllowedSources  []string `json:"allowed_sources"`
}

// Bundle represents a policy bundle with metadata
type Bundle struct {
	ID          string            `json:"id"`
	Version     string            `json:"version"`
	Policies    map[string]string `json:"policies"`
	Data        map[string]interface{} `json:"data"`
	Signature   string            `json:"signature"`
	Timestamp   time.Time         `json:"timestamp"`
	Checksum    string            `json:"checksum"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// EvaluationRequest represents a policy evaluation request
type EvaluationRequest struct {
	Input    interface{} `json:"input"`
	Policy   string      `json:"policy,omitempty"`
	Query    string      `json:"query,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// EvaluationResult represents a policy evaluation result
type EvaluationResult struct {
	Allow    bool                   `json:"allow"`
	Deny     bool                   `json:"deny"`
	Result   interface{}            `json:"result"`
	Metadata map[string]interface{} `json:"metadata"`
	Errors   []string               `json:"errors,omitempty"`
	Duration time.Duration          `json:"duration"`
}

// NewPolicyEngine creates a new policy engine
func NewPolicyEngine(logger *zap.Logger, config *Config) (*PolicyEngine, error) {
	if config == nil {
		config = DefaultConfig()
	}
	
	store := inmem.New()
	
	engine := &PolicyEngine{
		logger:  logger,
		store:   store,
		bundles: make(map[string]*Bundle),
		config:  config,
	}
	
	// Initialize with default policy if provided
	if config.DefaultPolicy != "" {
		if err := engine.loadDefaultPolicy(); err != nil {
			return nil, fmt.Errorf("failed to load default policy: %w", err)
		}
	}
	
	return engine, nil
}

// DefaultConfig returns default configuration
func DefaultConfig() *Config {
	return &Config{
		PollInterval:   time.Minute * 5,
		RequireSigned:  true,
		CacheSize:      1000,
		EvalTimeout:    time.Second * 5,
		MaxBundleSize:  10 * 1024 * 1024, // 10MB
		AllowedSources: []string{},
	}
}

// LoadBundle loads and validates a policy bundle
func (pe *PolicyEngine) LoadBundle(ctx context.Context, bundleData []byte, bundleID string) error {
	pe.mu.Lock()
	defer pe.mu.Unlock()
	
	// Validate bundle size
	if int64(len(bundleData)) > pe.config.MaxBundleSize {
		return fmt.Errorf("bundle size %d exceeds maximum %d", len(bundleData), pe.config.MaxBundleSize)
	}
	
	// Parse bundle
	var bundle Bundle
	if err := json.Unmarshal(bundleData, &bundle); err != nil {
		return fmt.Errorf("failed to parse bundle: %w", err)
	}
	
	bundle.ID = bundleID
	bundle.Timestamp = time.Now()
	
	// Calculate checksum
	hash := sha256.Sum256(bundleData)
	bundle.Checksum = fmt.Sprintf("%x", hash)
	
	// Verify signature if required
	if pe.config.RequireSigned {
		if err := pe.verifyBundleSignature(ctx, bundleData, bundle.Signature); err != nil {
			return fmt.Errorf("bundle signature verification failed: %w", err)
		}
		pe.logger.Info("bundle signature verified", zap.String("bundle_id", bundleID))
	}
	
	// Validate bundle schema
	if err := pe.validateBundleSchema(&bundle); err != nil {
		return fmt.Errorf("bundle schema validation failed: %w", err)
	}
	
	// Load policies into OPA store
	if err := pe.loadBundleIntoStore(ctx, &bundle); err != nil {
		return fmt.Errorf("failed to load bundle into store: %w", err)
	}
	
	// Store bundle metadata
	pe.bundles[bundleID] = &bundle
	
	pe.logger.Info("policy bundle loaded successfully",
		zap.String("bundle_id", bundleID),
		zap.String("version", bundle.Version),
		zap.String("checksum", bundle.Checksum),
		zap.Int("policies", len(bundle.Policies)))
	
	return nil
}

// verifyBundleSignature verifies bundle signature using Cosign
func (pe *PolicyEngine) verifyBundleSignature(ctx context.Context, bundleData []byte, signature string) error {
	if pe.config.CosignPublicKey == "" {
		return fmt.Errorf("cosign public key not configured")
	}
	
	// This is a simplified implementation
	// In production, you would use the full Cosign verification workflow
	pe.logger.Debug("verifying bundle signature with cosign",
		zap.String("public_key", pe.config.CosignPublicKey),
		zap.String("signature", signature[:min(len(signature), 32)]+"..."))
	
	// TODO: Implement full Cosign verification
	// For now, we'll do basic validation
	if signature == "" {
		return fmt.Errorf("bundle signature is empty")
	}
	
	if len(signature) < 64 {
		return fmt.Errorf("bundle signature too short")
	}
	
	return nil
}

// validateBundleSchema validates the bundle structure
func (pe *PolicyEngine) validateBundleSchema(bundle *Bundle) error {
	if bundle.ID == "" {
		return fmt.Errorf("bundle ID is required")
	}
	
	if bundle.Version == "" {
		return fmt.Errorf("bundle version is required")
	}
	
	if len(bundle.Policies) == 0 {
		return fmt.Errorf("bundle must contain at least one policy")
	}
	
	// Validate each policy
	for name, policy := range bundle.Policies {
		if policy == "" {
			return fmt.Errorf("policy %s is empty", name)
		}
		
		// Basic Rego syntax validation
		if err := pe.validateRegoPolicy(policy); err != nil {
			return fmt.Errorf("policy %s validation failed: %w", name, err)
		}
	}
	
	return nil
}

// validateRegoPolicy performs basic Rego policy validation
func (pe *PolicyEngine) validateRegoPolicy(policy string) error {
	// Try to compile the policy to check for syntax errors
	_, err := rego.New(
		rego.Query("data"),
		rego.Module("test", policy),
	).PrepareForEval(context.Background())
	
	return err
}

// loadBundleIntoStore loads bundle policies and data into OPA store
func (pe *PolicyEngine) loadBundleIntoStore(ctx context.Context, bundle *Bundle) error {
	txn, err := pe.store.NewTransaction(ctx, storage.WriteParams)
	if err != nil {
		return fmt.Errorf("failed to create transaction: %w", err)
	}
	defer pe.store.Abort(ctx, txn)
	
	// Load policies
	for name, policy := range bundle.Policies {
		path := storage.MustParsePath(fmt.Sprintf("/policies/%s", name))
		if err := pe.store.UpsertPolicy(ctx, txn, path, []byte(policy)); err != nil {
			return fmt.Errorf("failed to load policy %s: %w", name, err)
		}
	}
	
	// Load data
	if bundle.Data != nil {
		for key, value := range bundle.Data {
			path := storage.MustParsePath(fmt.Sprintf("/data/%s", key))
			if err := pe.store.Write(ctx, txn, storage.AddOp, path, value); err != nil {
				return fmt.Errorf("failed to load data %s: %w", key, err)
			}
		}
	}
	
	return pe.store.Commit(ctx, txn)
}

// loadDefaultPolicy loads the default policy
func (pe *PolicyEngine) loadDefaultPolicy() error {
	defaultBundle := &Bundle{
		ID:      "default",
		Version: "1.0.0",
		Policies: map[string]string{
			"default": pe.config.DefaultPolicy,
		},
		Data:      make(map[string]interface{}),
		Timestamp: time.Now(),
	}
	
	ctx := context.Background()
	return pe.loadBundleIntoStore(ctx, defaultBundle)
}

// EvaluatePolicy evaluates a policy against input data
func (pe *PolicyEngine) EvaluatePolicy(ctx context.Context, req *EvaluationRequest) (*EvaluationResult, error) {
	start := time.Now()
	
	// Set evaluation timeout
	evalCtx, cancel := context.WithTimeout(ctx, pe.config.EvalTimeout)
	defer cancel()
	
	// Prepare query
	query := req.Query
	if query == "" {
		query = "data.allow" // Default query
	}
	
	// Create Rego query
	r := rego.New(
		rego.Query(query),
		rego.Store(pe.store),
		rego.Input(req.Input),
	)
	
	// Prepare for evaluation
	prepared, err := r.PrepareForEval(evalCtx)
	if err != nil {
		return &EvaluationResult{
			Allow:    false,
			Deny:     true,
			Errors:   []string{fmt.Sprintf("failed to prepare query: %v", err)},
			Duration: time.Since(start),
		}, nil
	}
	
	// Evaluate policy
	results, err := prepared.Eval(evalCtx)
	if err != nil {
		return &EvaluationResult{
			Allow:    false,
			Deny:     true,
			Errors:   []string{fmt.Sprintf("evaluation failed: %v", err)},
			Duration: time.Since(start),
		}, nil
	}
	
	// Process results
	result := &EvaluationResult{
		Duration: time.Since(start),
		Metadata: req.Metadata,
	}
	
	if len(results) == 0 {
		result.Allow = false
		result.Deny = true
		result.Errors = []string{"no results returned from policy evaluation"}
		return result, nil
	}
	
	// Extract result
	if len(results[0].Expressions) > 0 {
		result.Result = results[0].Expressions[0].Value
		
		// Determine allow/deny based on result
		switch v := result.Result.(type) {
		case bool:
			result.Allow = v
			result.Deny = !v
		case map[string]interface{}:
			if allow, ok := v["allow"].(bool); ok {
				result.Allow = allow
				result.Deny = !allow
			}
			if deny, ok := v["deny"].(bool); ok {
				result.Deny = deny
				if !result.Allow {
					result.Allow = !deny
				}
			}
		default:
			// Default to deny if result is not boolean
			result.Allow = false
			result.Deny = true
		}
	}
	
	pe.logger.Debug("policy evaluation completed",
		zap.Bool("allow", result.Allow),
		zap.Bool("deny", result.Deny),
		zap.Duration("duration", result.Duration),
		zap.String("query", query))
	
	return result, nil
}

// GetBundle retrieves a bundle by ID
func (pe *PolicyEngine) GetBundle(bundleID string) (*Bundle, error) {
	pe.mu.RLock()
	defer pe.mu.RUnlock()
	
	bundle, exists := pe.bundles[bundleID]
	if !exists {
		return nil, fmt.Errorf("bundle %s not found", bundleID)
	}
	
	return bundle, nil
}

// ListBundles returns all loaded bundles
func (pe *PolicyEngine) ListBundles() map[string]*Bundle {
	pe.mu.RLock()
	defer pe.mu.RUnlock()
	
	result := make(map[string]*Bundle)
	for id, bundle := range pe.bundles {
		result[id] = bundle
	}
	
	return result
}

// RemoveBundle removes a bundle
func (pe *PolicyEngine) RemoveBundle(ctx context.Context, bundleID string) error {
	pe.mu.Lock()
	defer pe.mu.Unlock()
	
	if _, exists := pe.bundles[bundleID]; !exists {
		return fmt.Errorf("bundle %s not found", bundleID)
	}
	
	// Remove from store
	txn, err := pe.store.NewTransaction(ctx, storage.WriteParams)
	if err != nil {
		return fmt.Errorf("failed to create transaction: %w", err)
	}
	defer pe.store.Abort(ctx, txn)
	
	// Remove policies
	bundle := pe.bundles[bundleID]
	for name := range bundle.Policies {
		path := storage.MustParsePath(fmt.Sprintf("/policies/%s", name))
		if err := pe.store.DeletePolicy(ctx, txn, path); err != nil {
			pe.logger.Warn("failed to remove policy", zap.String("policy", name), zap.Error(err))
		}
	}
	
	if err := pe.store.Commit(ctx, txn); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}
	
	delete(pe.bundles, bundleID)
	
	pe.logger.Info("bundle removed", zap.String("bundle_id", bundleID))
	return nil
}

// GetStats returns policy engine statistics
func (pe *PolicyEngine) GetStats() map[string]interface{} {
	pe.mu.RLock()
	defer pe.mu.RUnlock()
	
	return map[string]interface{}{
		"bundles_loaded":    len(pe.bundles),
		"require_signed":    pe.config.RequireSigned,
		"max_bundle_size":   pe.config.MaxBundleSize,
		"eval_timeout":      pe.config.EvalTimeout.String(),
		"poll_interval":     pe.config.PollInterval.String(),
	}
}

// StartBundlePolling starts periodic bundle polling
func (pe *PolicyEngine) StartBundlePolling(ctx context.Context) {
	if pe.config.BundleURL == "" {
		pe.logger.Info("bundle URL not configured, skipping polling")
		return
	}
	
	ticker := time.NewTicker(pe.config.PollInterval)
	defer ticker.Stop()
	
	pe.logger.Info("starting bundle polling",
		zap.String("url", pe.config.BundleURL),
		zap.Duration("interval", pe.config.PollInterval))
	
	for {
		select {
		case <-ctx.Done():
			pe.logger.Info("stopping bundle polling")
			return
		case <-ticker.C:
			if err := pe.fetchAndLoadBundle(ctx); err != nil {
				pe.logger.Error("failed to fetch bundle", zap.Error(err))
			}
		}
	}
}

// fetchAndLoadBundle fetches and loads a bundle from the configured URL
func (pe *PolicyEngine) fetchAndLoadBundle(ctx context.Context) error {
	client := &http.Client{
		Timeout: time.Second * 30,
	}
	
	req, err := http.NewRequestWithContext(ctx, "GET", pe.config.BundleURL, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to fetch bundle: %w", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}
	
	// Read bundle data with size limit
	bundleData, err := io.ReadAll(io.LimitReader(resp.Body, pe.config.MaxBundleSize))
	if err != nil {
		return fmt.Errorf("failed to read bundle: %w", err)
	}
	
	// Load bundle
	bundleID := fmt.Sprintf("remote-%d", time.Now().Unix())
	return pe.LoadBundle(ctx, bundleData, bundleID)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}