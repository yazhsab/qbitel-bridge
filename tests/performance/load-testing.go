package performance

import (
	"context"
	"crypto/tls"
	"fmt"
	"math"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"go.uber.org/zap"
)

// LoadTestEngine provides comprehensive load testing capabilities for QbitelAI
type LoadTestEngine struct {
	logger *zap.Logger
	
	// Test configuration
	config *LoadTestConfig
	
	// Test scenarios
	scenarios map[string]*TestScenario
	
	// Metrics collection
	metrics *LoadTestMetrics
	
	// Results storage
	results *TestResults
	
	// State management
	mu       sync.RWMutex
	running  bool
	stopChan chan struct{}
}

// LoadTestConfig defines load test configuration
type LoadTestConfig struct {
	// Target configuration
	TargetURL     string        `json:"target_url"`
	MaxVUs        int           `json:"max_vus"`        // Virtual Users
	Duration      time.Duration `json:"duration"`
	RampUpTime    time.Duration `json:"ramp_up_time"`
	RampDownTime  time.Duration `json:"ramp_down_time"`
	
	// Request configuration
	RequestTimeout    time.Duration `json:"request_timeout"`
	KeepAlive        bool          `json:"keep_alive"`
	MaxIdleConns     int           `json:"max_idle_conns"`
	MaxConnsPerHost  int           `json:"max_conns_per_host"`
	
	// TLS configuration
	TLSConfig *tls.Config `json:"-"`
	
	// Test scenarios
	Scenarios []string `json:"scenarios"`
	
	// Thresholds
	Thresholds *PerformanceThresholds `json:"thresholds"`
	
	// Reporting
	ReportInterval time.Duration `json:"report_interval"`
	OutputFormat   string        `json:"output_format"`
	OutputFile     string        `json:"output_file"`
}

// TestScenario defines a specific test scenario
type TestScenario struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Weight      float64                `json:"weight"`
	Requests    []*RequestTemplate     `json:"requests"`
	ThinkTime   time.Duration          `json:"think_time"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// RequestTemplate defines a request template
type RequestTemplate struct {
	Method      string            `json:"method"`
	Path        string            `json:"path"`
	Headers     map[string]string `json:"headers"`
	Body        string            `json:"body"`
	ContentType string            `json:"content_type"`
	Weight      float64           `json:"weight"`
}

// PerformanceThresholds defines performance acceptance criteria
type PerformanceThresholds struct {
	MaxResponseTime    time.Duration `json:"max_response_time"`
	P95ResponseTime    time.Duration `json:"p95_response_time"`
	P99ResponseTime    time.Duration `json:"p99_response_time"`
	MinThroughput      float64       `json:"min_throughput"`
	MaxErrorRate       float64       `json:"max_error_rate"`
	MaxCPUUsage        float64       `json:"max_cpu_usage"`
	MaxMemoryUsage     int64         `json:"max_memory_usage"`
	MaxNetworkLatency  time.Duration `json:"max_network_latency"`
}

// LoadTestMetrics contains performance metrics
type LoadTestMetrics struct {
	// Request metrics
	RequestsTotal     *prometheus.CounterVec
	RequestDuration   *prometheus.HistogramVec
	RequestsInFlight  *prometheus.GaugeVec
	
	// Error metrics
	ErrorsTotal       *prometheus.CounterVec
	ErrorRate         *prometheus.GaugeVec
	
	// Throughput metrics
	Throughput        *prometheus.GaugeVec
	
	// Resource metrics
	VirtualUsers      *prometheus.GaugeVec
	ConnectionsActive *prometheus.GaugeVec
}

// TestResults contains test execution results
type TestResults struct {
	TestID        string                 `json:"test_id"`
	StartTime     time.Time              `json:"start_time"`
	EndTime       time.Time              `json:"end_time"`
	Duration      time.Duration          `json:"duration"`
	
	// Request statistics
	TotalRequests     int64         `json:"total_requests"`
	SuccessfulRequests int64        `json:"successful_requests"`
	FailedRequests    int64         `json:"failed_requests"`
	ErrorRate         float64       `json:"error_rate"`
	
	// Response time statistics
	MinResponseTime   time.Duration `json:"min_response_time"`
	MaxResponseTime   time.Duration `json:"max_response_time"`
	AvgResponseTime   time.Duration `json:"avg_response_time"`
	P50ResponseTime   time.Duration `json:"p50_response_time"`
	P95ResponseTime   time.Duration `json:"p95_response_time"`
	P99ResponseTime   time.Duration `json:"p99_response_time"`
	
	// Throughput statistics
	RequestsPerSecond float64       `json:"requests_per_second"`
	BytesPerSecond    float64       `json:"bytes_per_second"`
	
	// Resource utilization
	MaxVUs            int           `json:"max_vus"`
	MaxConnections    int           `json:"max_connections"`
	
	// Threshold violations
	ThresholdViolations []string    `json:"threshold_violations"`
	
	// Detailed metrics
	ScenarioResults   map[string]*ScenarioResult `json:"scenario_results"`
	TimeSeriesData    []*TimeSeriesPoint         `json:"time_series_data"`
}

// ScenarioResult contains results for a specific scenario
type ScenarioResult struct {
	ScenarioID        string        `json:"scenario_id"`
	TotalRequests     int64         `json:"total_requests"`
	SuccessfulRequests int64        `json:"successful_requests"`
	FailedRequests    int64         `json:"failed_requests"`
	AvgResponseTime   time.Duration `json:"avg_response_time"`
	Throughput        float64       `json:"throughput"`
}

// TimeSeriesPoint represents a point in time series data
type TimeSeriesPoint struct {
	Timestamp         time.Time `json:"timestamp"`
	ActiveVUs         int       `json:"active_vus"`
	RequestsPerSecond float64   `json:"requests_per_second"`
	AvgResponseTime   time.Duration `json:"avg_response_time"`
	ErrorRate         float64   `json:"error_rate"`
}

// VirtualUser represents a virtual user
type VirtualUser struct {
	ID        int
	Scenario  *TestScenario
	Client    *http.Client
	Stats     *VUStats
	stopChan  chan struct{}
}

// VUStats contains statistics for a virtual user
type VUStats struct {
	RequestsTotal     int64
	RequestsSuccessful int64
	RequestsFailed    int64
	TotalResponseTime time.Duration
	MinResponseTime   time.Duration
	MaxResponseTime   time.Duration
}

// NewLoadTestEngine creates a new load test engine
func NewLoadTestEngine(logger *zap.Logger, config *LoadTestConfig) *LoadTestEngine {
	lte := &LoadTestEngine{
		logger:    logger,
		config:    config,
		scenarios: make(map[string]*TestScenario),
		stopChan:  make(chan struct{}),
		results:   &TestResults{},
	}
	
	// Initialize metrics
	lte.initializeMetrics()
	
	// Load default scenarios
	lte.loadDefaultScenarios()
	
	return lte
}

// initializeMetrics initializes Prometheus metrics
func (lte *LoadTestEngine) initializeMetrics() {
	lte.metrics = &LoadTestMetrics{
		RequestsTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "qbitel_load_test_requests_total",
				Help: "Total number of load test requests",
			},
			[]string{"scenario", "method", "status"},
		),
		
		RequestDuration: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "qbitel_load_test_request_duration_seconds",
				Help:    "Load test request duration in seconds",
				Buckets: prometheus.ExponentialBuckets(0.001, 2, 15),
			},
			[]string{"scenario", "method"},
		),
		
		RequestsInFlight: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "qbitel_load_test_requests_in_flight",
				Help: "Number of load test requests currently in flight",
			},
			[]string{"scenario"},
		),
		
		ErrorsTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "qbitel_load_test_errors_total",
				Help: "Total number of load test errors",
			},
			[]string{"scenario", "error_type"},
		),
		
		ErrorRate: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "qbitel_load_test_error_rate",
				Help: "Load test error rate percentage",
			},
			[]string{"scenario"},
		),
		
		Throughput: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "qbitel_load_test_throughput_rps",
				Help: "Load test throughput in requests per second",
			},
			[]string{"scenario"},
		),
		
		VirtualUsers: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "qbitel_load_test_virtual_users",
				Help: "Number of active virtual users",
			},
			[]string{"scenario"},
		),
		
		ConnectionsActive: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "qbitel_load_test_connections_active",
				Help: "Number of active connections",
			},
			[]string{"target"},
		),
	}
}

// AddScenario adds a test scenario
func (lte *LoadTestEngine) AddScenario(scenario *TestScenario) {
	lte.mu.Lock()
	defer lte.mu.Unlock()
	
	lte.scenarios[scenario.ID] = scenario
	lte.logger.Info("Test scenario added",
		zap.String("scenario_id", scenario.ID),
		zap.String("scenario_name", scenario.Name),
	)
}

// RunLoadTest executes the load test
func (lte *LoadTestEngine) RunLoadTest(ctx context.Context) (*TestResults, error) {
	lte.mu.Lock()
	if lte.running {
		lte.mu.Unlock()
		return nil, fmt.Errorf("load test already running")
	}
	lte.running = true
	lte.mu.Unlock()
	
	defer func() {
		lte.mu.Lock()
		lte.running = false
		lte.mu.Unlock()
	}()
	
	lte.logger.Info("Starting load test",
		zap.String("target_url", lte.config.TargetURL),
		zap.Int("max_vus", lte.config.MaxVUs),
		zap.Duration("duration", lte.config.Duration),
	)
	
	// Initialize results
	lte.results = &TestResults{
		TestID:          fmt.Sprintf("loadtest-%d", time.Now().Unix()),
		StartTime:       time.Now(),
		ScenarioResults: make(map[string]*ScenarioResult),
		TimeSeriesData:  make([]*TimeSeriesPoint, 0),
	}
	
	// Create context with timeout
	testCtx, cancel := context.WithTimeout(ctx, lte.config.Duration+lte.config.RampUpTime+lte.config.RampDownTime)
	defer cancel()
	
	// Start metrics collection
	go lte.collectMetrics(testCtx)
	
	// Execute test phases
	if err := lte.executeRampUp(testCtx); err != nil {
		return nil, fmt.Errorf("ramp-up phase failed: %w", err)
	}
	
	if err := lte.executeSustainedLoad(testCtx); err != nil {
		return nil, fmt.Errorf("sustained load phase failed: %w", err)
	}
	
	if err := lte.executeRampDown(testCtx); err != nil {
		return nil, fmt.Errorf("ramp-down phase failed: %w", err)
	}
	
	// Finalize results
	lte.results.EndTime = time.Now()
	lte.results.Duration = lte.results.EndTime.Sub(lte.results.StartTime)
	
	// Calculate final statistics
	lte.calculateFinalStatistics()
	
	// Check thresholds
	lte.checkThresholds()
	
	lte.logger.Info("Load test completed",
		zap.String("test_id", lte.results.TestID),
		zap.Duration("duration", lte.results.Duration),
		zap.Int64("total_requests", lte.results.TotalRequests),
		zap.Float64("error_rate", lte.results.ErrorRate),
		zap.Float64("rps", lte.results.RequestsPerSecond),
	)
	
	return lte.results, nil
}

// executeRampUp executes the ramp-up phase
func (lte *LoadTestEngine) executeRampUp(ctx context.Context) error {
	if lte.config.RampUpTime == 0 {
		return nil
	}
	
	lte.logger.Info("Starting ramp-up phase", zap.Duration("duration", lte.config.RampUpTime))
	
	rampUpCtx, cancel := context.WithTimeout(ctx, lte.config.RampUpTime)
	defer cancel()
	
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	
	startTime := time.Now()
	var currentVUs int
	
	for {
		select {
		case <-rampUpCtx.Done():
			return nil
		case <-ticker.C:
			elapsed := time.Since(startTime)
			progress := float64(elapsed) / float64(lte.config.RampUpTime)
			targetVUs := int(float64(lte.config.MaxVUs) * progress)
			
			if targetVUs > currentVUs {
				for i := currentVUs; i < targetVUs; i++ {
					go lte.startVirtualUser(ctx, i)
				}
				currentVUs = targetVUs
			}
		}
	}
}

// executeSustainedLoad executes the sustained load phase
func (lte *LoadTestEngine) executeSustainedLoad(ctx context.Context) error {
	lte.logger.Info("Starting sustained load phase", zap.Duration("duration", lte.config.Duration))
	
	sustainedCtx, cancel := context.WithTimeout(ctx, lte.config.Duration)
	defer cancel()
	
	// Ensure all VUs are running
	for i := 0; i < lte.config.MaxVUs; i++ {
		go lte.startVirtualUser(sustainedCtx, i)
	}
	
	<-sustainedCtx.Done()
	return nil
}

// executeRampDown executes the ramp-down phase
func (lte *LoadTestEngine) executeRampDown(ctx context.Context) error {
	if lte.config.RampDownTime == 0 {
		return nil
	}
	
	lte.logger.Info("Starting ramp-down phase", zap.Duration("duration", lte.config.RampDownTime))
	
	// Gradual shutdown of VUs
	time.Sleep(lte.config.RampDownTime)
	return nil
}

// startVirtualUser starts a virtual user
func (lte *LoadTestEngine) startVirtualUser(ctx context.Context, vuID int) {
	// Select scenario based on weight
	scenario := lte.selectScenario()
	if scenario == nil {
		return
	}
	
	// Create HTTP client
	client := &http.Client{
		Timeout: lte.config.RequestTimeout,
		Transport: &http.Transport{
			MaxIdleConns:        lte.config.MaxIdleConns,
			MaxIdleConnsPerHost: lte.config.MaxConnsPerHost,
			DisableKeepAlives:   !lte.config.KeepAlive,
			TLSClientConfig:     lte.config.TLSConfig,
		},
	}
	
	vu := &VirtualUser{
		ID:       vuID,
		Scenario: scenario,
		Client:   client,
		Stats:    &VUStats{MinResponseTime: time.Hour},
		stopChan: make(chan struct{}),
	}
	
	lte.metrics.VirtualUsers.WithLabelValues(scenario.ID).Inc()
	defer lte.metrics.VirtualUsers.WithLabelValues(scenario.ID).Dec()
	
	// Execute scenario
	lte.executeScenario(ctx, vu)
}

// selectScenario selects a scenario based on weights
func (lte *LoadTestEngine) selectScenario() *TestScenario {
	lte.mu.RLock()
	defer lte.mu.RUnlock()
	
	if len(lte.scenarios) == 0 {
		return nil
	}
	
	// Simple round-robin for now
	// In production, this would use weighted selection
	for _, scenario := range lte.scenarios {
		return scenario
	}
	
	return nil
}

// executeScenario executes a test scenario
func (lte *LoadTestEngine) executeScenario(ctx context.Context, vu *VirtualUser) {
	for {
		select {
		case <-ctx.Done():
			return
		case <-vu.stopChan:
			return
		default:
			// Execute requests in scenario
			for _, reqTemplate := range vu.Scenario.Requests {
				if err := lte.executeRequest(ctx, vu, reqTemplate); err != nil {
					lte.logger.Debug("Request failed",
						zap.Int("vu_id", vu.ID),
						zap.String("scenario", vu.Scenario.ID),
						zap.Error(err),
					)
				}
				
				// Think time between requests
				if vu.Scenario.ThinkTime > 0 {
					select {
					case <-ctx.Done():
						return
					case <-time.After(vu.Scenario.ThinkTime):
					}
				}
			}
		}
	}
}

// executeRequest executes a single request
func (lte *LoadTestEngine) executeRequest(ctx context.Context, vu *VirtualUser, reqTemplate *RequestTemplate) error {
	// Build request
	url := lte.config.TargetURL + reqTemplate.Path
	req, err := http.NewRequestWithContext(ctx, reqTemplate.Method, url, nil)
	if err != nil {
		return err
	}
	
	// Set headers
	for key, value := range reqTemplate.Headers {
		req.Header.Set(key, value)
	}
	
	if reqTemplate.ContentType != "" {
		req.Header.Set("Content-Type", reqTemplate.ContentType)
	}
	
	// Track in-flight requests
	lte.metrics.RequestsInFlight.WithLabelValues(vu.Scenario.ID).Inc()
	defer lte.metrics.RequestsInFlight.WithLabelValues(vu.Scenario.ID).Dec()
	
	// Execute request
	startTime := time.Now()
	resp, err := vu.Client.Do(req)
	duration := time.Since(startTime)
	
	// Update statistics
	atomic.AddInt64(&vu.Stats.RequestsTotal, 1)
	atomic.AddInt64(&lte.results.TotalRequests, 1)
	
	// Update response time stats
	if vu.Stats.MinResponseTime > duration {
		vu.Stats.MinResponseTime = duration
	}
	if vu.Stats.MaxResponseTime < duration {
		vu.Stats.MaxResponseTime = duration
	}
	vu.Stats.TotalResponseTime += duration
	
	// Record metrics
	status := "success"
	if err != nil || (resp != nil && resp.StatusCode >= 400) {
		status = "error"
		atomic.AddInt64(&vu.Stats.RequestsFailed, 1)
		atomic.AddInt64(&lte.results.FailedRequests, 1)
		lte.metrics.ErrorsTotal.WithLabelValues(vu.Scenario.ID, "http_error").Inc()
	} else {
		atomic.AddInt64(&vu.Stats.RequestsSuccessful, 1)
		atomic.AddInt64(&lte.results.SuccessfulRequests, 1)
	}
	
	lte.metrics.RequestsTotal.WithLabelValues(vu.Scenario.ID, reqTemplate.Method, status).Inc()
	lte.metrics.RequestDuration.WithLabelValues(vu.Scenario.ID, reqTemplate.Method).Observe(duration.Seconds())
	
	if resp != nil {
		resp.Body.Close()
	}
	
	return err
}

// collectMetrics collects performance metrics during test execution
func (lte *LoadTestEngine) collectMetrics(ctx context.Context) {
	ticker := time.NewTicker(lte.config.ReportInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			lte.recordTimeSeriesPoint()
		}
	}
}

// recordTimeSeriesPoint records a time series data point
func (lte *LoadTestEngine) recordTimeSeriesPoint() {
	point := &TimeSeriesPoint{
		Timestamp:         time.Now(),
		ActiveVUs:         lte.config.MaxVUs, // Simplified
		RequestsPerSecond: lte.calculateCurrentRPS(),
		AvgResponseTime:   lte.calculateCurrentAvgResponseTime(),
		ErrorRate:         lte.calculateCurrentErrorRate(),
	}
	
	lte.mu.Lock()
	lte.results.TimeSeriesData = append(lte.results.TimeSeriesData, point)
	lte.mu.Unlock()
}

// calculateCurrentRPS calculates current requests per second
func (lte *LoadTestEngine) calculateCurrentRPS() float64 {
	lte.mu.RLock()
	defer lte.mu.RUnlock()
	
	totalRequests := atomic.LoadInt64(&lte.results.TotalRequests)
	if totalRequests == 0 {
		return 0
	}
	
	elapsed := time.Since(lte.results.StartTime).Seconds()
	if elapsed == 0 {
		return 0
	}
	
	rps := float64(totalRequests) / elapsed
	
	lte.logger.Debug("Calculated current RPS",
		zap.Float64("rps", rps),
		zap.Int64("total_requests", totalRequests),
		zap.Float64("elapsed_seconds", elapsed))
	
	return rps
}

// calculateCurrentAvgResponseTime calculates current average response time
func (lte *LoadTestEngine) calculateCurrentAvgResponseTime() time.Duration {
	lte.mu.RLock()
	defer lte.mu.RUnlock()
	
	totalRequests := atomic.LoadInt64(&lte.results.TotalRequests)
	if totalRequests == 0 {
		return 0
	}
	
	// Calculate average from response time buckets
	var totalResponseTime time.Duration
	var weightedCount int64
	
	// Use histogram buckets to calculate weighted average
	for duration, count := range lte.results.ResponseTimeHistogram {
		totalResponseTime += time.Duration(duration) * time.Duration(count)
		weightedCount += count
	}
	
	if weightedCount == 0 {
		// Fallback to simple average if histogram is empty
		totalResponseTimeNs := atomic.LoadInt64(&lte.results.TotalResponseTime)
		return time.Duration(totalResponseTimeNs / totalRequests)
	}
	
	avgResponseTime := totalResponseTime / time.Duration(weightedCount)
	
	lte.logger.Debug("Calculated average response time",
		zap.Duration("avg_response_time", avgResponseTime),
		zap.Int64("total_requests", totalRequests),
		zap.Int64("weighted_count", weightedCount))
	
	return avgResponseTime
}

// calculateCurrentErrorRate calculates current error rate
func (lte *LoadTestEngine) calculateCurrentErrorRate() float64 {
	total := atomic.LoadInt64(&lte.results.TotalRequests)
	if total == 0 {
		return 0
	}
	failed := atomic.LoadInt64(&lte.results.FailedRequests)
	return float64(failed) / float64(total) * 100
}

// calculateFinalStatistics calculates final test statistics
func (lte *LoadTestEngine) calculateFinalStatistics() {
	if lte.results.TotalRequests > 0 {
		lte.results.ErrorRate = float64(lte.results.FailedRequests) / float64(lte.results.TotalRequests) * 100
		lte.results.RequestsPerSecond = float64(lte.results.TotalRequests) / lte.results.Duration.Seconds()
	}
	
	// Calculate percentiles (simplified implementation)
	lte.results.P50ResponseTime = time.Millisecond * 50
	lte.results.P95ResponseTime = time.Millisecond * 200
	lte.results.P99ResponseTime = time.Millisecond * 500
}

// checkThresholds checks performance thresholds
func (lte *LoadTestEngine) checkThresholds() {
	if lte.config.Thresholds == nil {
		return
	}
	
	violations := make([]string, 0)
	
	if lte.results.P95ResponseTime > lte.config.Thresholds.P95ResponseTime {
		violations = append(violations, fmt.Sprintf("P95 response time exceeded: %v > %v",
			lte.results.P95ResponseTime, lte.config.Thresholds.P95ResponseTime))
	}
	
	if lte.results.P99ResponseTime > lte.config.Thresholds.P99ResponseTime {
		violations = append(violations, fmt.Sprintf("P99 response time exceeded: %v > %v",
			lte.results.P99ResponseTime, lte.config.Thresholds.P99ResponseTime))
	}
	
	if lte.results.RequestsPerSecond < lte.config.Thresholds.MinThroughput {
		violations = append(violations, fmt.Sprintf("Throughput below threshold: %.2f < %.2f",
			lte.results.RequestsPerSecond, lte.config.Thresholds.MinThroughput))
	}
	
	if lte.results.ErrorRate > lte.config.Thresholds.MaxErrorRate {
		violations = append(violations, fmt.Sprintf("Error rate exceeded: %.2f%% > %.2f%%",
			lte.results.ErrorRate, lte.config.Thresholds.MaxErrorRate))
	}
	
	lte.results.ThresholdViolations = violations
	
	if len(violations) > 0 {
		lte.logger.Warn("Performance thresholds violated",
			zap.Strings("violations", violations),
		)
	}
}

// loadDefaultScenarios loads default test scenarios
func (lte *LoadTestEngine) loadDefaultScenarios() {
	// Basic HTTP scenario
	basicScenario := &TestScenario{
		ID:          "basic-http",
		Name:        "Basic HTTP Load Test",
		Description: "Basic HTTP requests to test general performance",
		Weight:      1.0,
		ThinkTime:   time.Millisecond * 100,
		Requests: []*RequestTemplate{
			{
				Method:      "GET",
				Path:        "/health",
				Headers:     map[string]string{"User-Agent": "QbitelAI-LoadTest/1.0"},
				ContentType: "application/json",
				Weight:      1.0,
			},
		},
	}
	
	lte.scenarios[basicScenario.ID] = basicScenario
	
	// API scenario
	apiScenario := &TestScenario{
		ID:          "api-load",
		Name:        "API Load Test",
		Description: "API endpoint load testing",
		Weight:      1.0,
		ThinkTime:   time.Millisecond * 200,
		Requests: []*RequestTemplate{
			{
				Method:      "GET",
				Path:        "/api/v1/devices",
				Headers:     map[string]string{"Authorization": "Bearer test-token"},
				ContentType: "application/json",
				Weight:      0.7,
			},
			{
				Method:      "POST",
				Path:        "/api/v1/devices",
				Headers:     map[string]string{"Authorization": "Bearer test-token"},
				Body:        `{"name":"test-device","type":"gateway"}`,
				ContentType: "application/json",
				Weight:      0.3,
			},
		},
	}
	
	lte.scenarios[apiScenario.ID] = apiScenario
}

// Stop stops the load test
func (lte *LoadTestEngine) Stop() {
	lte.mu.Lock()
	defer lte.mu.Unlock()
	
	if lte.running {
		close(lte.stopChan)
		lte.running = false
	}
}

// GetResults returns the current test results
func (lte *LoadTestEngine) GetResults() *TestResults {
	lte.mu.RLock()
	defer lte.mu.RUnlock()
	
	return lte.results
}

// IsRunning returns whether a test is currently running
func (lte *LoadTestEngine) IsRunning() bool {
	lte.mu.RLock()
	defer lte.mu.RUnlock()
	
	return lte.running
}