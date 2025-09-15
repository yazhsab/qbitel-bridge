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

func TestNewPolicyEngine(t *testing.T) {
	logger := zaptest.NewLogger(t)
	config := DefaultPolicyConfig()
	
	engine, err := NewPolicyEngine(logger, config)
	require.NoError(t, err)
	require.NotNil(t, engine)
	
	assert.Equal(t, config, engine.config)
	assert.NotNil(t, engine.opa)
	assert.NotNil(t, engine.bundles)
}

func TestPolicyEngine_LoadBundle(t *testing.T) {
	logger := zaptest.NewLogger(t)
	config := DefaultPolicyConfig()
	
	engine, err := NewPolicyEngine(logger, config)
	require.NoError(t, err)
	
	// Create test bundle
	bundle := &Bundle{
		ID:      "test-bundle",
		Version: "1.0.0",
		Policies: map[string]string{
			"test": `
				package qslb.test
				
				default allow = false
				
				allow {
					input.method == "GET"
					input.path == "/api/test"
				}
			`,
		},
		Data: map[string]interface{}{
			"config": map[string]interface{}{
				"max_requests": 100,
			},
		},
	}
	
	bundleData, err := json.Marshal(bundle)
	require.NoError(t, err)
	
	ctx := context.Background()
	err = engine.LoadBundle(ctx, bundleData, "test-bundle")
	assert.NoError(t, err)
	
	// Verify bundle is loaded
	loadedBundle, exists := engine.bundles["test-bundle"]
	assert.True(t, exists)
	assert.Equal(t, bundle.ID, loadedBundle.ID)
	assert.Equal(t, bundle.Version, loadedBundle.Version)
}

func TestPolicyEngine_LoadBundleInvalidJSON(t *testing.T) {
	logger := zaptest.NewLogger(t)
	config := DefaultPolicyConfig()
	
	engine, err := NewPolicyEngine(logger, config)
	require.NoError(t, err)
	
	ctx := context.Background()
	err = engine.LoadBundle(ctx, []byte("invalid json"), "test-bundle")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to parse bundle")
}

func TestPolicyEngine_EvaluatePolicy(t *testing.T) {
	logger := zaptest.NewLogger(t)
	config := DefaultPolicyConfig()
	
	engine, err := NewPolicyEngine(logger, config)
	require.NoError(t, err)
	
	// Load test bundle
	bundle := &Bundle{
		ID:      "test-bundle",
		Version: "1.0.0",
		Policies: map[string]string{
			"test": `
				package qslb.test
				
				default allow = false
				
				allow {
					input.method == "GET"
					input.path == "/api/test"
				}
				
				allow {
					input.method == "POST"
					input.path == "/api/create"
					input.user.role == "admin"
				}
			`,
		},
	}
	
	bundleData, err := json.Marshal(bundle)
	require.NoError(t, err)
	
	ctx := context.Background()
	err = engine.LoadBundle(ctx, bundleData, "test-bundle")
	require.NoError(t, err)
	
	tests := []struct {
		name     string
		query    string
		input    map[string]interface{}
		expected bool
	}{
		{
			name:  "allow GET request",
			query: "data.qslb.test.allow",
			input: map[string]interface{}{
				"method": "GET",
				"path":   "/api/test",
			},
			expected: true,
		},
		{
			name:  "deny GET request to wrong path",
			query: "data.qslb.test.allow",
			input: map[string]interface{}{
				"method": "GET",
				"path":   "/api/wrong",
			},
			expected: false,
		},
		{
			name:  "allow POST request for admin",
			query: "data.qslb.test.allow",
			input: map[string]interface{}{
				"method": "POST",
				"path":   "/api/create",
				"user": map[string]interface{}{
					"role": "admin",
				},
			},
			expected: true,
		},
		{
			name:  "deny POST request for non-admin",
			query: "data.qslb.test.allow",
			input: map[string]interface{}{
				"method": "POST",
				"path":   "/api/create",
				"user": map[string]interface{}{
					"role": "user",
				},
			},
			expected: false,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := engine.EvaluatePolicy(ctx, tt.query, tt.input)
			require.NoError(t, err)
			
			allowed, ok := result["allow"].(bool)
			require.True(t, ok, "result should contain 'allow' boolean field")
			assert.Equal(t, tt.expected, allowed)
		})
	}
}

func TestPolicyEngine_EvaluatePolicyWithTimeout(t *testing.T) {
	logger := zaptest.NewLogger(t)
	config := DefaultPolicyConfig()
	config.EvaluationTimeout = time.Millisecond * 10 // Very short timeout
	
	engine, err := NewPolicyEngine(logger, config)
	require.NoError(t, err)
	
	// Load bundle with potentially slow policy
	bundle := &Bundle{
		ID:      "slow-bundle",
		Version: "1.0.0",
		Policies: map[string]string{
			"slow": `
				package qslb.slow
				
				default result = false
				
				result {
					# This could potentially be slow in a real scenario
					input.value > 0
				}
			`,
		},
	}
	
	bundleData, err := json.Marshal(bundle)
	require.NoError(t, err)
	
	ctx := context.Background()
	err = engine.LoadBundle(ctx, bundleData, "slow-bundle")
	require.NoError(t, err)
	
	input := map[string]interface{}{
		"value": 42,
	}
	
	// This should complete within timeout for this simple policy
	result, err := engine.EvaluatePolicy(ctx, "data.qslb.slow.result", input)
	assert.NoError(t, err)
	assert.NotNil(t, result)
}

func TestPolicyEngine_RemoveBundle(t *testing.T) {
	logger := zaptest.NewLogger(t)
	config := DefaultPolicyConfig()
	
	engine, err := NewPolicyEngine(logger, config)
	require.NoError(t, err)
	
	// Load test bundle
	bundle := &Bundle{
		ID:      "test-bundle",
		Version: "1.0.0",
		Policies: map[string]string{
			"test": "package qslb.test\ndefault allow = false",
		},
	}
	
	bundleData, err := json.Marshal(bundle)
	require.NoError(t, err)
	
	ctx := context.Background()
	err = engine.LoadBundle(ctx, bundleData, "test-bundle")
	require.NoError(t, err)
	
	// Verify bundle is loaded
	_, exists := engine.bundles["test-bundle"]
	assert.True(t, exists)
	
	// Remove bundle
	err = engine.RemoveBundle(ctx, "test-bundle")
	assert.NoError(t, err)
	
	// Verify bundle is removed
	_, exists = engine.bundles["test-bundle"]
	assert.False(t, exists)
}

func TestPolicyEngine_RemoveBundleNotFound(t *testing.T) {
	logger := zaptest.NewLogger(t)
	config := DefaultPolicyConfig()
	
	engine, err := NewPolicyEngine(logger, config)
	require.NoError(t, err)
	
	ctx := context.Background()
	err = engine.RemoveBundle(ctx, "nonexistent-bundle")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "bundle nonexistent-bundle not found")
}

func TestPolicyEngine_ListBundles(t *testing.T) {
	logger := zaptest.NewLogger(t)
	config := DefaultPolicyConfig()
	
	engine, err := NewPolicyEngine(logger, config)
	require.NoError(t, err)
	
	// Initially no bundles
	bundles := engine.ListBundles()
	assert.Empty(t, bundles)
	
	// Load test bundles
	for i := 1; i <= 3; i++ {
		bundle := &Bundle{
			ID:      fmt.Sprintf("test-bundle-%d", i),
			Version: "1.0.0",
			Policies: map[string]string{
				"test": "package qslb.test\ndefault allow = false",
			},
		}
		
		bundleData, err := json.Marshal(bundle)
		require.NoError(t, err)
		
		ctx := context.Background()
		err = engine.LoadBundle(ctx, bundleData, bundle.ID)
		require.NoError(t, err)
	}
	
	// Verify bundles are listed
	bundles = engine.ListBundles()
	assert.Len(t, bundles, 3)
	
	expectedIDs := []string{"test-bundle-1", "test-bundle-2", "test-bundle-3"}
	for _, expectedID := range expectedIDs {
		found := false
		for _, bundle := range bundles {
			if bundle.ID == expectedID {
				found = true
				break
			}
		}
		assert.True(t, found, "bundle %s should be in the list", expectedID)
	}
}

func TestPolicyEngine_GetStats(t *testing.T) {
	logger := zaptest.NewLogger(t)
	config := DefaultPolicyConfig()
	
	engine, err := NewPolicyEngine(logger, config)
	require.NoError(t, err)
	
	// Get initial stats
	stats := engine.GetStats()
	assert.Equal(t, 0, stats["total_bundles"])
	assert.Equal(t, 0, stats["total_policies"])
	assert.Equal(t, int64(0), stats["total_evaluations"])
	
	// Load test bundle
	bundle := &Bundle{
		ID:      "test-bundle",
		Version: "1.0.0",
		Policies: map[string]string{
			"policy1": "package qslb.policy1\ndefault allow = false",
			"policy2": "package qslb.policy2\ndefault allow = true",
		},
	}
	
	bundleData, err := json.Marshal(bundle)
	require.NoError(t, err)
	
	ctx := context.Background()
	err = engine.LoadBundle(ctx, bundleData, "test-bundle")
	require.NoError(t, err)
	
	// Perform some evaluations
	input := map[string]interface{}{"test": true}
	_, err = engine.EvaluatePolicy(ctx, "data.qslb.policy1.allow", input)
	require.NoError(t, err)
	_, err = engine.EvaluatePolicy(ctx, "data.qslb.policy2.allow", input)
	require.NoError(t, err)
	
	// Get updated stats
	stats = engine.GetStats()
	assert.Equal(t, 1, stats["total_bundles"])
	assert.Equal(t, 2, stats["total_policies"])
	assert.Equal(t, int64(2), stats["total_evaluations"])
}

func TestPolicyEngine_ValidateBundle(t *testing.T) {
	logger := zaptest.NewLogger(t)
	config := DefaultPolicyConfig()
	
	engine, err := NewPolicyEngine(logger, config)
	require.NoError(t, err)
	
	tests := []struct {
		name        string
		bundle      *Bundle
		expectError bool
		errorMsg    string
	}{
		{
			name: "valid bundle",
			bundle: &Bundle{
				ID:      "valid-bundle",
				Version: "1.0.0",
				Policies: map[string]string{
					"test": "package qslb.test\ndefault allow = false",
				},
			},
			expectError: false,
		},
		{
			name: "bundle without ID",
			bundle: &Bundle{
				Version: "1.0.0",
				Policies: map[string]string{
					"test": "package qslb.test\ndefault allow = false",
				},
			},
			expectError: true,
			errorMsg:    "bundle ID is required",
		},
		{
			name: "bundle without version",
			bundle: &Bundle{
				ID: "test-bundle",
				Policies: map[string]string{
					"test": "package qslb.test\ndefault allow = false",
				},
			},
			expectError: true,
			errorMsg:    "bundle version is required",
		},
		{
			name: "bundle without policies",
			bundle: &Bundle{
				ID:      "test-bundle",
				Version: "1.0.0",
			},
			expectError: true,
			errorMsg:    "bundle must contain at least one policy",
		},
		{
			name: "bundle with invalid policy syntax",
			bundle: &Bundle{
				ID:      "test-bundle",
				Version: "1.0.0",
				Policies: map[string]string{
					"invalid": "this is not valid rego syntax {{{",
				},
			},
			expectError: true,
			errorMsg:    "policy validation failed",
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := engine.validateBundle(tt.bundle)
			
			if tt.expectError {
				assert.Error(t, err)
				if tt.errorMsg != "" {
					assert.Contains(t, err.Error(), tt.errorMsg)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestPolicyEngine_ConcurrentEvaluations(t *testing.T) {
	logger := zaptest.NewLogger(t)
	config := DefaultPolicyConfig()
	
	engine, err := NewPolicyEngine(logger, config)
	require.NoError(t, err)
	
	// Load test bundle
	bundle := &Bundle{
		ID:      "concurrent-test",
		Version: "1.0.0",
		Policies: map[string]string{
			"test": `
				package qslb.test
				
				default allow = false
				
				allow {
					input.id >= 0
					input.id < 100
				}
			`,
		},
	}
	
	bundleData, err := json.Marshal(bundle)
	require.NoError(t, err)
	
	ctx := context.Background()
	err = engine.LoadBundle(ctx, bundleData, "concurrent-test")
	require.NoError(t, err)
	
	// Run concurrent evaluations
	const numGoroutines = 10
	const numEvaluations = 100
	
	results := make(chan error, numGoroutines*numEvaluations)
	
	for i := 0; i < numGoroutines; i++ {
		go func(goroutineID int) {
			for j := 0; j < numEvaluations; j++ {
				input := map[string]interface{}{
					"id": goroutineID*numEvaluations + j,
				}
				
				_, err := engine.EvaluatePolicy(ctx, "data.qslb.test.allow", input)
				results <- err
			}
		}(i)
	}
	
	// Collect results
	for i := 0; i < numGoroutines*numEvaluations; i++ {
		err := <-results
		assert.NoError(t, err)
	}
	
	// Verify stats
	stats := engine.GetStats()
	assert.Equal(t, int64(numGoroutines*numEvaluations), stats["total_evaluations"])
}

// Benchmark tests
func BenchmarkPolicyEngine_EvaluatePolicy(b *testing.B) {
	logger := zaptest.NewLogger(b)
	config := DefaultPolicyConfig()
	
	engine, err := NewPolicyEngine(logger, config)
	require.NoError(b, err)
	
	// Load test bundle
	bundle := &Bundle{
		ID:      "benchmark-test",
		Version: "1.0.0",
		Policies: map[string]string{
			"test": `
				package qslb.test
				
				default allow = false
				
				allow {
					input.method == "GET"
					input.path == "/api/test"
					input.user.role == "admin"
				}
			`,
		},
	}
	
	bundleData, err := json.Marshal(bundle)
	require.NoError(b, err)
	
	ctx := context.Background()
	err = engine.LoadBundle(ctx, bundleData, "benchmark-test")
	require.NoError(b, err)
	
	input := map[string]interface{}{
		"method": "GET",
		"path":   "/api/test",
		"user": map[string]interface{}{
			"role": "admin",
		},
	}
	
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, err := engine.EvaluatePolicy(ctx, "data.qslb.test.allow", input)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}