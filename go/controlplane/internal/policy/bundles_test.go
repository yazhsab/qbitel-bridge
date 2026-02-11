package policy

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap/zaptest"
)

func TestNewBundleManager(t *testing.T) {
	logger := zaptest.NewLogger(t)
	engine, err := NewPolicyEngine(logger, DefaultPolicyConfig())
	require.NoError(t, err)
	
	config := DefaultBundleConfig()
	
	manager, err := NewBundleManager(logger, engine, config)
	require.NoError(t, err)
	require.NotNil(t, manager)
	
	assert.Equal(t, config, manager.config)
	assert.NotNil(t, manager.verifier)
	assert.Empty(t, manager.bundles)
}

func TestBundleManager_CreateBundle(t *testing.T) {
	logger := zaptest.NewLogger(t)
	engine, err := NewPolicyEngine(logger, DefaultPolicyConfig())
	require.NoError(t, err)
	
	config := DefaultBundleConfig()
	config.SigningEnabled = false // Disable signing for this test
	
	manager, err := NewBundleManager(logger, engine, config)
	require.NoError(t, err)
	
	req := &CreateBundleRequest{
		ID:          "test-bundle",
		Version:     "1.0.0",
		Name:        "Test Bundle",
		Description: "A test bundle",
		Author:      "test-author",
		Tags:        []string{"test", "example"},
		Policies: map[string]string{
			"test": `
				package qbitel.test
				
				default allow = false
				
				allow {
					input.method == "GET"
				}
			`,
		},
		Data: map[string]interface{}{
			"config": map[string]interface{}{
				"max_requests": 100,
			},
		},
		Dependencies: []string{},
		Metadata: map[string]interface{}{
			"environment": "test",
		},
	}
	
	ctx := context.Background()
	metadata, err := manager.CreateBundle(ctx, req)
	require.NoError(t, err)
	require.NotNil(t, metadata)
	
	assert.Equal(t, req.ID, metadata.ID)
	assert.Equal(t, req.Version, metadata.Version)
	assert.Equal(t, req.Name, metadata.Name)
	assert.Equal(t, req.Description, metadata.Description)
	assert.Equal(t, req.Author, metadata.Author)
	assert.Equal(t, req.Tags, metadata.Tags)
	assert.Contains(t, metadata.Policies, "test")
	assert.Contains(t, metadata.Data, "config")
	assert.NotEmpty(t, metadata.Checksum)
	assert.Greater(t, metadata.Size, int64(0))
	assert.Empty(t, metadata.Signature) // Signing disabled
	
	// Verify bundle is stored in manager
	storedMetadata, err := manager.GetBundleMetadata("test-bundle")
	require.NoError(t, err)
	assert.Equal(t, metadata, storedMetadata)
}

func TestBundleManager_CreateBundleWithSigning(t *testing.T) {
	logger := zaptest.NewLogger(t)
	engine, err := NewPolicyEngine(logger, DefaultPolicyConfig())
	require.NoError(t, err)
	
	config := DefaultBundleConfig()
	config.SigningEnabled = true
	config.PrivateKeyPath = "/path/to/private/key"
	
	manager, err := NewBundleManager(logger, engine, config)
	require.NoError(t, err)
	
	req := &CreateBundleRequest{
		ID:      "signed-bundle",
		Version: "1.0.0",
		Name:    "Signed Bundle",
		Policies: map[string]string{
			"test": "package qbitel.test\ndefault allow = false",
		},
	}
	
	ctx := context.Background()
	metadata, err := manager.CreateBundle(ctx, req)
	require.NoError(t, err)
	
	assert.NotEmpty(t, metadata.Signature)
	assert.Contains(t, metadata.Signature, "cosign-sig-")
}

func TestBundleManager_CreateBundleValidation(t *testing.T) {
	logger := zaptest.NewLogger(t)
	engine, err := NewPolicyEngine(logger, DefaultPolicyConfig())
	require.NoError(t, err)
	
	manager, err := NewBundleManager(logger, engine, DefaultBundleConfig())
	require.NoError(t, err)
	
	ctx := context.Background()
	
	tests := []struct {
		name        string
		req         *CreateBundleRequest
		expectError bool
		errorMsg    string
	}{
		{
			name: "missing ID",
			req: &CreateBundleRequest{
				Version: "1.0.0",
				Policies: map[string]string{
					"test": "package qbitel.test\ndefault allow = false",
				},
			},
			expectError: true,
			errorMsg:    "bundle ID is required",
		},
		{
			name: "missing version",
			req: &CreateBundleRequest{
				ID: "test-bundle",
				Policies: map[string]string{
					"test": "package qbitel.test\ndefault allow = false",
				},
			},
			expectError: true,
			errorMsg:    "bundle version is required",
		},
		{
			name: "missing policies",
			req: &CreateBundleRequest{
				ID:      "test-bundle",
				Version: "1.0.0",
			},
			expectError: true,
			errorMsg:    "bundle must contain at least one policy",
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := manager.CreateBundle(ctx, tt.req)
			assert.Error(t, err)
			assert.Contains(t, err.Error(), tt.errorMsg)
		})
	}
}

func TestBundleManager_CreateBundleDuplicate(t *testing.T) {
	logger := zaptest.NewLogger(t)
	engine, err := NewPolicyEngine(logger, DefaultPolicyConfig())
	require.NoError(t, err)
	
	config := DefaultBundleConfig()
	config.SigningEnabled = false
	
	manager, err := NewBundleManager(logger, engine, config)
	require.NoError(t, err)
	
	req := &CreateBundleRequest{
		ID:      "duplicate-bundle",
		Version: "1.0.0",
		Policies: map[string]string{
			"test": "package qbitel.test\ndefault allow = false",
		},
	}
	
	ctx := context.Background()
	
	// Create first bundle
	_, err = manager.CreateBundle(ctx, req)
	require.NoError(t, err)
	
	// Try to create duplicate
	_, err = manager.CreateBundle(ctx, req)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "bundle duplicate-bundle already exists")
}

func TestBundleManager_LoadBundle(t *testing.T) {
	logger := zaptest.NewLogger(t)
	engine, err := NewPolicyEngine(logger, DefaultPolicyConfig())
	require.NoError(t, err)
	
	config := DefaultBundleConfig()
	config.RequireSignature = false // Disable signature verification for this test
	
	manager, err := NewBundleManager(logger, engine, config)
	require.NoError(t, err)
	
	ctx := context.Background()
	err = manager.LoadBundle(ctx, "test-bundle")
	assert.NoError(t, err)
}

func TestBundleManager_LoadBundleWithSignatureVerification(t *testing.T) {
	logger := zaptest.NewLogger(t)
	engine, err := NewPolicyEngine(logger, DefaultPolicyConfig())
	require.NoError(t, err)
	
	config := DefaultBundleConfig()
	config.RequireSignature = true
	config.TrustedKeys = []string{"/path/to/public/key"}
	
	manager, err := NewBundleManager(logger, engine, config)
	require.NoError(t, err)
	
	ctx := context.Background()
	err = manager.LoadBundle(ctx, "signed-bundle")
	assert.NoError(t, err)
}

func TestBundleManager_DeleteBundle(t *testing.T) {
	logger := zaptest.NewLogger(t)
	engine, err := NewPolicyEngine(logger, DefaultPolicyConfig())
	require.NoError(t, err)
	
	config := DefaultBundleConfig()
	config.SigningEnabled = false
	
	manager, err := NewBundleManager(logger, engine, config)
	require.NoError(t, err)
	
	// Create a bundle first
	req := &CreateBundleRequest{
		ID:      "delete-test",
		Version: "1.0.0",
		Policies: map[string]string{
			"test": "package qbitel.test\ndefault allow = false",
		},
	}
	
	ctx := context.Background()
	_, err = manager.CreateBundle(ctx, req)
	require.NoError(t, err)
	
	// Verify bundle exists
	_, err = manager.GetBundleMetadata("delete-test")
	require.NoError(t, err)
	
	// Delete bundle
	err = manager.DeleteBundle(ctx, "delete-test")
	assert.NoError(t, err)
	
	// Verify bundle is deleted
	_, err = manager.GetBundleMetadata("delete-test")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "bundle delete-test not found")
}

func TestBundleManager_DeleteBundleNotFound(t *testing.T) {
	logger := zaptest.NewLogger(t)
	engine, err := NewPolicyEngine(logger, DefaultPolicyConfig())
	require.NoError(t, err)
	
	manager, err := NewBundleManager(logger, engine, DefaultBundleConfig())
	require.NoError(t, err)
	
	ctx := context.Background()
	err = manager.DeleteBundle(ctx, "nonexistent-bundle")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "bundle nonexistent-bundle not found")
}

func TestBundleManager_ListBundles(t *testing.T) {
	logger := zaptest.NewLogger(t)
	engine, err := NewPolicyEngine(logger, DefaultPolicyConfig())
	require.NoError(t, err)
	
	config := DefaultBundleConfig()
	config.SigningEnabled = false
	
	manager, err := NewBundleManager(logger, engine, config)
	require.NoError(t, err)
	
	// Initially no bundles
	bundles := manager.ListBundles()
	assert.Empty(t, bundles)
	
	// Create test bundles
	ctx := context.Background()
	for i := 1; i <= 3; i++ {
		req := &CreateBundleRequest{
			ID:      fmt.Sprintf("list-test-%d", i),
			Version: "1.0.0",
			Policies: map[string]string{
				"test": "package qbitel.test\ndefault allow = false",
			},
		}
		
		_, err = manager.CreateBundle(ctx, req)
		require.NoError(t, err)
	}
	
	// List bundles
	bundles = manager.ListBundles()
	assert.Len(t, bundles, 3)
	
	expectedIDs := []string{"list-test-1", "list-test-2", "list-test-3"}
	for _, expectedID := range expectedIDs {
		_, exists := bundles[expectedID]
		assert.True(t, exists, "bundle %s should be in the list", expectedID)
	}
}

func TestBundleManager_GetStats(t *testing.T) {
	logger := zaptest.NewLogger(t)
	engine, err := NewPolicyEngine(logger, DefaultPolicyConfig())
	require.NoError(t, err)
	
	config := DefaultBundleConfig()
	config.SigningEnabled = false
	
	manager, err := NewBundleManager(logger, engine, config)
	require.NoError(t, err)
	
	// Get initial stats
	stats := manager.GetStats()
	assert.Equal(t, 0, stats["total_bundles"])
	assert.Equal(t, int64(0), stats["total_size_bytes"])
	assert.Equal(t, config.StorageType, stats["storage_type"])
	assert.Equal(t, config.SigningEnabled, stats["signing_enabled"])
	assert.Equal(t, config.RequireSignature, stats["require_signature"])
	assert.Equal(t, config.MaxBundleSize, stats["max_bundle_size"])
	
	// Create test bundles
	ctx := context.Background()
	for i := 1; i <= 2; i++ {
		req := &CreateBundleRequest{
			ID:      fmt.Sprintf("stats-test-%d", i),
			Version: "1.0.0",
			Policies: map[string]string{
				"test": "package qbitel.test\ndefault allow = false",
			},
		}
		
		_, err = manager.CreateBundle(ctx, req)
		require.NoError(t, err)
	}
	
	// Get updated stats
	stats = manager.GetStats()
	assert.Equal(t, 2, stats["total_bundles"])
	assert.Greater(t, stats["total_size_bytes"].(int64), int64(0))
}

func TestCosignVerifier_VerifyBundle(t *testing.T) {
	logger := zaptest.NewLogger(t)
	config := DefaultBundleConfig()
	config.TrustedKeys = []string{"/path/to/public/key"}
	
	verifier, err := NewCosignVerifier(logger, config)
	require.NoError(t, err)
	
	// Create test bundle with signature
	bundle := &Bundle{
		ID:        "verify-test",
		Version:   "1.0.0",
		Signature: "cosign-sig-1234567890abcdef",
		Policies: map[string]string{
			"test": "package qbitel.test\ndefault allow = false",
		},
	}
	
	bundleData, err := json.Marshal(bundle)
	require.NoError(t, err)
	
	ctx := context.Background()
	err = verifier.VerifyBundle(ctx, bundleData, "verify-test")
	assert.NoError(t, err)
}

func TestCosignVerifier_VerifyBundleInvalidSignature(t *testing.T) {
	logger := zaptest.NewLogger(t)
	config := DefaultBundleConfig()
	
	verifier, err := NewCosignVerifier(logger, config)
	require.NoError(t, err)
	
	tests := []struct {
		name      string
		bundle    *Bundle
		errorMsg  string
	}{
		{
			name: "missing signature",
			bundle: &Bundle{
				ID:      "no-sig-test",
				Version: "1.0.0",
				Policies: map[string]string{
					"test": "package qbitel.test\ndefault allow = false",
				},
			},
			errorMsg: "bundle signature is missing",
		},
		{
			name: "invalid signature format",
			bundle: &Bundle{
				ID:        "invalid-sig-test",
				Version:   "1.0.0",
				Signature: "invalid-signature-format",
				Policies: map[string]string{
					"test": "package qbitel.test\ndefault allow = false",
				},
			},
			errorMsg: "invalid signature format",
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bundleData, err := json.Marshal(tt.bundle)
			require.NoError(t, err)
			
			ctx := context.Background()
			err = verifier.VerifyBundle(ctx, bundleData, tt.bundle.ID)
			assert.Error(t, err)
			assert.Contains(t, err.Error(), tt.errorMsg)
		})
	}
}

func TestBundleManager_StorageTypes(t *testing.T) {
	logger := zaptest.NewLogger(t)
	engine, err := NewPolicyEngine(logger, DefaultPolicyConfig())
	require.NoError(t, err)
	
	storageTypes := []string{"filesystem", "s3", "gcs", "oci"}
	
	for _, storageType := range storageTypes {
		t.Run(storageType, func(t *testing.T) {
			config := DefaultBundleConfig()
			config.StorageType = storageType
			config.SigningEnabled = false
			
			manager, err := NewBundleManager(logger, engine, config)
			require.NoError(t, err)
			
			ctx := context.Background()
			
			// Test storing bundle
			err = manager.storeBundle(ctx, "test-bundle", []byte("test data"))
			if storageType == "filesystem" {
				assert.NoError(t, err)
			} else {
				// Other storage types are not fully implemented yet
				assert.NoError(t, err) // They return nil for now
			}
			
			// Test loading bundle
			_, err = manager.loadBundleData(ctx, "test-bundle")
			if storageType == "filesystem" {
				assert.NoError(t, err)
			} else {
				// Other storage types return not implemented error
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "not implemented")
			}
		})
	}
}

func TestBundleManager_UnsupportedStorageType(t *testing.T) {
	logger := zaptest.NewLogger(t)
	engine, err := NewPolicyEngine(logger, DefaultPolicyConfig())
	require.NoError(t, err)
	
	config := DefaultBundleConfig()
	config.StorageType = "unsupported"
	
	manager, err := NewBundleManager(logger, engine, config)
	require.NoError(t, err)
	
	ctx := context.Background()
	
	// Test storing bundle with unsupported storage type
	err = manager.storeBundle(ctx, "test-bundle", []byte("test data"))
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported storage type: unsupported")
	
	// Test loading bundle with unsupported storage type
	_, err = manager.loadBundleData(ctx, "test-bundle")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unsupported storage type: unsupported")
}

func TestDefaultBundleConfig(t *testing.T) {
	config := DefaultBundleConfig()
	
	assert.Equal(t, "filesystem", config.StorageType)
	assert.Equal(t, "./bundles", config.StoragePath)
	assert.True(t, config.SigningEnabled)
	assert.True(t, config.RequireSignature)
	assert.Equal(t, int64(50*1024*1024), config.MaxBundleSize)
	assert.Equal(t, time.Hour*24*30, config.RetentionPeriod)
	assert.Equal(t, time.Minute*5, config.SyncInterval)
}

// Benchmark tests
func BenchmarkBundleManager_CreateBundle(b *testing.B) {
	logger := zaptest.NewLogger(b)
	engine, err := NewPolicyEngine(logger, DefaultPolicyConfig())
	require.NoError(b, err)
	
	config := DefaultBundleConfig()
	config.SigningEnabled = false
	
	manager, err := NewBundleManager(logger, engine, config)
	require.NoError(b, err)
	
	req := &CreateBundleRequest{
		Version: "1.0.0",
		Policies: map[string]string{
			"test": "package qbitel.test\ndefault allow = false",
		},
	}
	
	ctx := context.Background()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req.ID = fmt.Sprintf("benchmark-bundle-%d", i)
		_, err := manager.CreateBundle(ctx, req)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkBundleManager_LoadBundle(b *testing.B) {
	logger := zaptest.NewLogger(b)
	engine, err := NewPolicyEngine(logger, DefaultPolicyConfig())
	require.NoError(b, err)
	
	config := DefaultBundleConfig()
	config.RequireSignature = false
	
	manager, err := NewBundleManager(logger, engine, config)
	require.NoError(b, err)
	
	ctx := context.Background()
	
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			err := manager.LoadBundle(ctx, "benchmark-bundle")
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}