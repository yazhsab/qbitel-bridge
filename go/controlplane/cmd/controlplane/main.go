package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"

	"github.com/qslb/go/controlplane/internal/policy"
)

// PolicyValidationRequest represents a policy validation request
type PolicyValidationRequest struct {
	BundleID string                 `json:"bundle_id"`
	Query    string                 `json:"query"`
	Input    map[string]interface{} `json:"input"`
}

// PolicyValidationResponse represents a policy validation response
type PolicyValidationResponse struct {
	Valid  bool                   `json:"valid"`
	Result map[string]interface{} `json:"result,omitempty"`
	Error  string                 `json:"error,omitempty"`
}

// BundleUploadRequest represents a bundle upload request
type BundleUploadRequest struct {
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

// ControlPlaneServer represents the control plane server
type ControlPlaneServer struct {
	logger        *zap.Logger
	policyEngine  *policy.PolicyEngine
	bundleManager *policy.BundleManager
	router        *gin.Engine
}

func main() {
	logger, err := zap.NewProduction()
	if err != nil {
		panic(fmt.Sprintf("failed to create logger: %v", err))
	}
	defer logger.Sync()

	// Create policy engine
	policyConfig := policy.DefaultPolicyConfig()
	policyEngine, err := policy.NewPolicyEngine(logger, policyConfig)
	if err != nil {
		logger.Fatal("failed to create policy engine", zap.Error(err))
	}

	// Create bundle manager
	bundleConfig := policy.DefaultBundleConfig()
	bundleConfig.StorageType = getenv("BUNDLE_STORAGE_TYPE", "filesystem")
	bundleConfig.StoragePath = getenv("BUNDLE_STORAGE_PATH", "./bundles")
	bundleConfig.SigningEnabled = getenvBool("BUNDLE_SIGNING_ENABLED", true)
	bundleConfig.RequireSignature = getenvBool("BUNDLE_REQUIRE_SIGNATURE", true)
	bundleConfig.DistributionURL = getenv("BUNDLE_DISTRIBUTION_URL", "")
	
	bundleManager, err := policy.NewBundleManager(logger, policyEngine, bundleConfig)
	if err != nil {
		logger.Fatal("failed to create bundle manager", zap.Error(err))
	}

	// Create server
	server := &ControlPlaneServer{
		logger:        logger,
		policyEngine:  policyEngine,
		bundleManager: bundleManager,
	}

	// Setup routes
	server.setupRoutes()

	// Start bundle synchronization if configured
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if bundleConfig.DistributionURL != "" {
		go bundleManager.StartBundleSync(ctx)
	}

	// Load default bundles
	if err := server.loadDefaultBundles(ctx); err != nil {
		logger.Warn("failed to load default bundles", zap.Error(err))
	}

	// Start server
	addr := getenv("CONTROL_ADDR", ":8080")
	logger.Info("control plane starting", 
		zap.String("addr", addr),
		zap.String("storage_type", bundleConfig.StorageType),
		zap.Bool("signing_enabled", bundleConfig.SigningEnabled))

	// Setup graceful shutdown
	srv := &http.Server{
		Addr:    addr,
		Handler: server.router,
	}

	// Start server in goroutine
	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatal("failed to start server", zap.Error(err))
		}
	}()

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Info("shutting down control plane")

	// Cancel context to stop background tasks
	cancel()

	// Shutdown server with timeout
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()

	if err := srv.Shutdown(shutdownCtx); err != nil {
		logger.Error("server forced to shutdown", zap.Error(err))
	}

	logger.Info("control plane stopped")
}

// setupRoutes configures the HTTP routes
func (s *ControlPlaneServer) setupRoutes() {
	s.router = gin.Default()

	// Add logging middleware
	s.router.Use(gin.LoggerWithConfig(gin.LoggerConfig{
		Formatter: func(param gin.LogFormatterParams) string {
			return fmt.Sprintf("%s - [%s] \"%s %s %s %d %s \"%s\" %s\"\n",
				param.ClientIP,
				param.TimeStamp.Format(time.RFC3339),
				param.Method,
				param.Path,
				param.Request.Proto,
				param.StatusCode,
				param.Latency,
				param.Request.UserAgent(),
				param.ErrorMessage,
			)
		},
	}))

	// Health check
	s.router.GET("/healthz", s.handleHealthCheck)

	// Policy validation endpoints
	s.router.POST("/policy/validate", s.handlePolicyValidation)
	s.router.POST("/policy/evaluate", s.handlePolicyEvaluation)

	// Bundle management endpoints
	s.router.POST("/bundles", s.handleBundleUpload)
	s.router.GET("/bundles", s.handleListBundles)
	s.router.GET("/bundles/:id", s.handleGetBundle)
	s.router.DELETE("/bundles/:id", s.handleDeleteBundle)
	s.router.POST("/bundles/:id/load", s.handleLoadBundle)

	// Statistics endpoints
	s.router.GET("/stats/policy", s.handlePolicyStats)
	s.router.GET("/stats/bundles", s.handleBundleStats)

	// Admin endpoints
	s.router.GET("/admin/status", s.handleAdminStatus)
}

// handleHealthCheck handles health check requests
func (s *ControlPlaneServer) handleHealthCheck(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":    "ok",
		"timestamp": time.Now().UTC().Format(time.RFC3339),
		"version":   "1.0.0",
	})
}

// handlePolicyValidation handles policy validation requests (legacy endpoint)
func (s *ControlPlaneServer) handlePolicyValidation(c *gin.Context) {
	var req PolicyValidationRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, PolicyValidationResponse{
			Valid: false,
			Error: fmt.Sprintf("invalid request: %v", err),
		})
		return
	}

	ctx := c.Request.Context()
	result, err := s.policyEngine.EvaluatePolicy(ctx, req.Query, req.Input)
	if err != nil {
		s.logger.Error("policy evaluation failed", 
			zap.String("bundle_id", req.BundleID),
			zap.String("query", req.Query),
			zap.Error(err))
		
		c.JSON(http.StatusInternalServerError, PolicyValidationResponse{
			Valid: false,
			Error: fmt.Sprintf("evaluation failed: %v", err),
		})
		return
	}

	// Check if result contains 'allow' field for backward compatibility
	valid := false
	if allow, ok := result["allow"].(bool); ok {
		valid = allow
	}

	c.JSON(http.StatusOK, PolicyValidationResponse{
		Valid:  valid,
		Result: result,
	})
}

// handlePolicyEvaluation handles policy evaluation requests
func (s *ControlPlaneServer) handlePolicyEvaluation(c *gin.Context) {
	var req PolicyValidationRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": fmt.Sprintf("invalid request: %v", err),
		})
		return
	}

	ctx := c.Request.Context()
	result, err := s.policyEngine.EvaluatePolicy(ctx, req.Query, req.Input)
	if err != nil {
		s.logger.Error("policy evaluation failed", 
			zap.String("bundle_id", req.BundleID),
			zap.String("query", req.Query),
			zap.Error(err))
		
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": fmt.Sprintf("evaluation failed: %v", err),
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"result": result,
	})
}

// handleBundleUpload handles bundle upload requests
func (s *ControlPlaneServer) handleBundleUpload(c *gin.Context) {
	var req BundleUploadRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": fmt.Sprintf("invalid request: %v", err),
		})
		return
	}

	createReq := &policy.CreateBundleRequest{
		ID:           req.ID,
		Version:      req.Version,
		Name:         req.Name,
		Description:  req.Description,
		Author:       req.Author,
		Tags:         req.Tags,
		Policies:     req.Policies,
		Data:         req.Data,
		Dependencies: req.Dependencies,
		Metadata:     req.Metadata,
	}

	ctx := c.Request.Context()
	metadata, err := s.bundleManager.CreateBundle(ctx, createReq)
	if err != nil {
		s.logger.Error("bundle creation failed", 
			zap.String("bundle_id", req.ID),
			zap.Error(err))
		
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": fmt.Sprintf("bundle creation failed: %v", err),
		})
		return
	}

	s.logger.Info("bundle created successfully", 
		zap.String("bundle_id", metadata.ID),
		zap.String("version", metadata.Version))

	c.JSON(http.StatusCreated, metadata)
}

// handleListBundles handles bundle listing requests
func (s *ControlPlaneServer) handleListBundles(c *gin.Context) {
	bundles := s.bundleManager.ListBundles()
	c.JSON(http.StatusOK, gin.H{
		"bundles": bundles,
		"count":   len(bundles),
	})
}

// handleGetBundle handles bundle retrieval requests
func (s *ControlPlaneServer) handleGetBundle(c *gin.Context) {
	bundleID := c.Param("id")
	if bundleID == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "bundle ID is required",
		})
		return
	}

	metadata, err := s.bundleManager.GetBundleMetadata(bundleID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{
			"error": fmt.Sprintf("bundle not found: %v", err),
		})
		return
	}

	c.JSON(http.StatusOK, metadata)
}

// handleDeleteBundle handles bundle deletion requests
func (s *ControlPlaneServer) handleDeleteBundle(c *gin.Context) {
	bundleID := c.Param("id")
	if bundleID == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "bundle ID is required",
		})
		return
	}

	ctx := c.Request.Context()
	if err := s.bundleManager.DeleteBundle(ctx, bundleID); err != nil {
		s.logger.Error("bundle deletion failed", 
			zap.String("bundle_id", bundleID),
			zap.Error(err))
		
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": fmt.Sprintf("bundle deletion failed: %v", err),
		})
		return
	}

	s.logger.Info("bundle deleted successfully", zap.String("bundle_id", bundleID))
	c.JSON(http.StatusOK, gin.H{
		"message": "bundle deleted successfully",
	})
}

// handleLoadBundle handles bundle loading requests
func (s *ControlPlaneServer) handleLoadBundle(c *gin.Context) {
	bundleID := c.Param("id")
	if bundleID == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "bundle ID is required",
		})
		return
	}

	ctx := c.Request.Context()
	if err := s.bundleManager.LoadBundle(ctx, bundleID); err != nil {
		s.logger.Error("bundle loading failed", 
			zap.String("bundle_id", bundleID),
			zap.Error(err))
		
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": fmt.Sprintf("bundle loading failed: %v", err),
		})
		return
	}

	s.logger.Info("bundle loaded successfully", zap.String("bundle_id", bundleID))
	c.JSON(http.StatusOK, gin.H{
		"message": "bundle loaded successfully",
	})
}

// handlePolicyStats handles policy statistics requests
func (s *ControlPlaneServer) handlePolicyStats(c *gin.Context) {
	stats := s.policyEngine.GetStats()
	c.JSON(http.StatusOK, stats)
}

// handleBundleStats handles bundle statistics requests
func (s *ControlPlaneServer) handleBundleStats(c *gin.Context) {
	stats := s.bundleManager.GetStats()
	c.JSON(http.StatusOK, stats)
}

// handleAdminStatus handles admin status requests
func (s *ControlPlaneServer) handleAdminStatus(c *gin.Context) {
	policyStats := s.policyEngine.GetStats()
	bundleStats := s.bundleManager.GetStats()

	c.JSON(http.StatusOK, gin.H{
		"status":        "running",
		"timestamp":     time.Now().UTC().Format(time.RFC3339),
		"policy_stats":  policyStats,
		"bundle_stats":  bundleStats,
		"version":       "1.0.0",
	})
}

// loadDefaultBundles loads default policy bundles
func (s *ControlPlaneServer) loadDefaultBundles(ctx context.Context) error {
	// Create a default security bundle
	defaultBundle := &policy.CreateBundleRequest{
		ID:          "qslb-default",
		Version:     "1.0.0",
		Name:        "QSLB Default Security Policies",
		Description: "Default security policies for QSLB",
		Author:      "QSLB System",
		Tags:        []string{"default", "security"},
		Policies: map[string]string{
			"default": `
				package qslb.default
				
				import rego.v1
				
				# Default deny policy
				default allow := false
				
				# Allow health checks
				allow if {
					input.path == "/healthz"
					input.method == "GET"
				}
				
				# Allow authenticated admin requests
				allow if {
					input.path startswith "/admin"
					input.user.role == "admin"
					input.user.authenticated == true
				}
				
				# Allow policy evaluation for authenticated users
				allow if {
					input.path startswith "/policy"
					input.user.authenticated == true
				}
				
				# Allow bundle operations for admin users
				allow if {
					input.path startswith "/bundles"
					input.user.role == "admin"
					input.user.authenticated == true
				}
			`,
			"rate_limiting": `
				package qslb.rate_limiting
				
				import rego.v1
				
				# Rate limiting policy
				default rate_limit := {"allowed": true, "limit": 1000}
				
				rate_limit := {"allowed": false, "limit": 100} if {
					input.user.role == "guest"
				}
				
				rate_limit := {"allowed": true, "limit": 10000} if {
					input.user.role == "admin"
				}
			`,
		},
		Data: map[string]interface{}{
			"config": map[string]interface{}{
				"max_requests_per_minute": 1000,
				"admin_max_requests":      10000,
				"guest_max_requests":      100,
			},
		},
		Metadata: map[string]interface{}{
			"created_by": "system",
			"auto_load":  true,
		},
	}

	// Create the bundle
	metadata, err := s.bundleManager.CreateBundle(ctx, defaultBundle)
	if err != nil {
		// If bundle already exists, try to load it
		if err := s.bundleManager.LoadBundle(ctx, defaultBundle.ID); err != nil {
			return fmt.Errorf("failed to create or load default bundle: %w", err)
		}
		s.logger.Info("loaded existing default bundle", zap.String("bundle_id", defaultBundle.ID))
	} else {
		s.logger.Info("created default bundle", 
			zap.String("bundle_id", metadata.ID),
			zap.String("version", metadata.Version))
	}

	return nil
}

// getenv returns environment variable value or default
func getenv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// getenvBool returns environment variable as boolean or default
func getenvBool(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		return value == "true" || value == "1" || value == "yes"
	}
	return defaultValue
}
