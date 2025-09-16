package threatdetection

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"go.uber.org/zap"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// MLBehavioralAnalyzer implements machine learning-based behavioral analysis
type MLBehavioralAnalyzer struct {
	logger *zap.Logger
	config *ThreatDetectionConfig
	
	// ML Models
	anomalyDetector     *AnomalyDetectionModel
	behaviorClassifier  *BehaviorClassificationModel
	riskPredictor      *RiskPredictionModel
	
	// Data storage
	behaviorProfiles   map[string]*BehaviorProfile
	trainingData       *TrainingDataStore
	modelStore         *ModelStore
	
	// Feature extraction
	featureExtractor   *FeatureExtractor
	
	// Metrics
	anomaliesDetected   *prometheus.CounterVec
	behaviorScores      *prometheus.HistogramVec
	modelAccuracy       *prometheus.GaugeVec
	predictionLatency   *prometheus.HistogramVec
	
	// State
	mu       sync.RWMutex
	running  bool
	stopChan chan struct{}
}

// BehaviorProfile represents user/device behavioral patterns
type BehaviorProfile struct {
	EntityID     string                 `json:"entity_id"`
	EntityType   string                 `json:"entity_type"` // "user" or "device"
	
	// Baseline behavior patterns
	BaselineFeatures  FeatureVector      `json:"baseline_features"`
	NormalPatterns   []BehaviorPattern   `json:"normal_patterns"`
	
	// Current behavior
	RecentFeatures   FeatureVector       `json:"recent_features"`
	CurrentAnomalies []Anomaly           `json:"current_anomalies"`
	
	// Risk assessment
	RiskScore        float64             `json:"risk_score"`
	ThreatLevel      ThreatLevel         `json:"threat_level"`
	Confidence       float64             `json:"confidence"`
	
	// Statistics
	TotalEvents      int64               `json:"total_events"`
	AnomalyCount     int64               `json:"anomaly_count"`
	LastUpdate       time.Time           `json:"last_update"`
	LastAnomaly      time.Time           `json:"last_anomaly"`
	
	// Training data
	TrainingExamples []TrainingExample   `json:"training_examples"`
	ModelVersion     string              `json:"model_version"`
}

// FeatureVector represents extracted behavioral features
type FeatureVector struct {
	// Temporal features
	LoginTimes          []float64          `json:"login_times"`
	SessionDurations    []float64          `json:"session_durations"`
	ActivityFrequency   map[string]float64 `json:"activity_frequency"`
	TimeBasedPatterns   map[int]float64    `json:"time_based_patterns"` // Hour of day
	
	// Access patterns
	ResourceAccess      map[string]float64 `json:"resource_access"`
	LocationPatterns    map[string]float64 `json:"location_patterns"`
	DevicePatterns      map[string]float64 `json:"device_patterns"`
	NetworkPatterns     map[string]float64 `json:"network_patterns"`
	
	// Behavioral metrics
	TypingPatterns      []float64          `json:"typing_patterns"`
	ClickPatterns       []float64          `json:"click_patterns"`
	NavigationPatterns  []float64          `json:"navigation_patterns"`
	
	// Security events
	FailedAttempts      float64            `json:"failed_attempts"`
	SecurityViolations  float64            `json:"security_violations"`
	PolicyViolations    float64            `json:"policy_violations"`
	
	// Network behavior
	DataTransfer        float64            `json:"data_transfer"`
	ConnectionPatterns  map[string]float64 `json:"connection_patterns"`
	ProtocolDistribution map[string]float64 `json:"protocol_distribution"`
}

// BehaviorPattern represents a detected behavior pattern
type BehaviorPattern struct {
	ID          string                 `json:"id"`
	Type        PatternType            `json:"type"`
	Confidence  float64                `json:"confidence"`
	Frequency   float64                `json:"frequency"`
	Features    map[string]float64     `json:"features"`
	Context     map[string]interface{} `json:"context"`
	FirstSeen   time.Time              `json:"first_seen"`
	LastSeen    time.Time              `json:"last_seen"`
}

// Anomaly represents a detected behavioral anomaly
type Anomaly struct {
	ID           string                 `json:"id"`
	Type         AnomalyType            `json:"type"`
	Severity     float64                `json:"severity"`
	Confidence   float64                `json:"confidence"`
	Description  string                 `json:"description"`
	Features     FeatureVector          `json:"features"`
	Baseline     FeatureVector          `json:"baseline"`
	Deviation    float64                `json:"deviation"`
	Context      map[string]interface{} `json:"context"`
	DetectedAt   time.Time              `json:"detected_at"`
	Acknowledged bool                   `json:"acknowledged"`
	FalsePositive bool                  `json:"false_positive"`
}

// AnomalyDetectionModel implements unsupervised anomaly detection
type AnomalyDetectionModel struct {
	ModelType    string              `json:"model_type"`
	Parameters   map[string]float64  `json:"parameters"`
	Threshold    float64             `json:"threshold"`
	
	// Isolation Forest parameters
	NumTrees     int                 `json:"num_trees"`
	SubsampleSize int                `json:"subsample_size"`
	
	// One-Class SVM parameters
	Gamma        float64             `json:"gamma"`
	Nu           float64             `json:"nu"`
	
	// Autoencoder parameters
	InputDim     int                 `json:"input_dim"`
	HiddenDims   []int               `json:"hidden_dims"`
	
	// Training metadata
	TrainedOn    time.Time           `json:"trained_on"`
	TrainingSamples int              `json:"training_samples"`
	Accuracy     float64             `json:"accuracy"`
	Version      string              `json:"version"`
}

// BehaviorClassificationModel implements supervised behavior classification
type BehaviorClassificationModel struct {
	ModelType    string              `json:"model_type"`
	Classes      []string            `json:"classes"`
	Features     []string            `json:"features"`
	Parameters   map[string]float64  `json:"parameters"`
	
	// Random Forest parameters
	NumTrees     int                 `json:"num_trees"`
	MaxDepth     int                 `json:"max_depth"`
	MinSamples   int                 `json:"min_samples"`
	
	// Neural Network parameters
	Architecture []int               `json:"architecture"`
	ActivationFunc string            `json:"activation_func"`
	LearningRate float64             `json:"learning_rate"`
	
	// Training metadata
	TrainedOn    time.Time           `json:"trained_on"`
	TrainingSamples int              `json:"training_samples"`
	Accuracy     float64             `json:"accuracy"`
	F1Score      float64             `json:"f1_score"`
	Version      string              `json:"version"`
}

// RiskPredictionModel implements risk score prediction
type RiskPredictionModel struct {
	ModelType    string              `json:"model_type"`
	Features     []string            `json:"features"`
	Parameters   map[string]float64  `json:"parameters"`
	
	// Gradient Boosting parameters
	NumEstimators int                `json:"num_estimators"`
	LearningRate float64             `json:"learning_rate"`
	MaxDepth     int                 `json:"max_depth"`
	
	// Training metadata
	TrainedOn    time.Time           `json:"trained_on"`
	TrainingSamples int              `json:"training_samples"`
	MAE          float64             `json:"mae"` // Mean Absolute Error
	RMSE         float64             `json:"rmse"` // Root Mean Square Error
	Version      string              `json:"version"`
}

// TrainingExample represents a labeled training example
type TrainingExample struct {
	Features    FeatureVector          `json:"features"`
	Label       string                 `json:"label"`
	Weight      float64                `json:"weight"`
	Timestamp   time.Time              `json:"timestamp"`
	Context     map[string]interface{} `json:"context"`
}

// Types and enums
type PatternType string
const (
	PatternTemporal     PatternType = "temporal"
	PatternAccess       PatternType = "access"
	PatternLocation     PatternType = "location"
	PatternDevice       PatternType = "device"
	PatternNetwork      PatternType = "network"
	PatternBehavioral   PatternType = "behavioral"
)

type AnomalyType string
const (
	AnomalyOutlier      AnomalyType = "outlier"
	AnomalyNovelty      AnomalyType = "novelty"
	AnomalyDrift        AnomalyType = "drift"
	AnomalyFrequency    AnomalyType = "frequency"
	AnomalySequence     AnomalyType = "sequence"
	AnomalyCorrelation  AnomalyType = "correlation"
)

type ThreatLevel string
const (
	ThreatLevelLow      ThreatLevel = "low"
	ThreatLevelMedium   ThreatLevel = "medium"
	ThreatLevelHigh     ThreatLevel = "high"
	ThreatLevelCritical ThreatLevel = "critical"
)

// ThreatDetectionConfig holds configuration for threat detection
type ThreatDetectionConfig struct {
	// Analysis settings
	AnalysisInterval        time.Duration `json:"analysis_interval"`
	MinEventsForAnalysis    int           `json:"min_events_for_analysis"`
	BaselinePeriod          time.Duration `json:"baseline_period"`
	
	// Anomaly detection settings
	AnomalyThreshold        float64       `json:"anomaly_threshold"`
	SensitivityLevel        float64       `json:"sensitivity_level"`
	FalsePositiveThreshold  float64       `json:"false_positive_threshold"`
	
	// Model settings
	ModelUpdateInterval     time.Duration `json:"model_update_interval"`
	AutoRetrain             bool          `json:"auto_retrain"`
	MinTrainingSamples      int           `json:"min_training_samples"`
	
	// Feature extraction settings
	FeatureExtractionWindow time.Duration `json:"feature_extraction_window"`
	MaxFeatureHistory       int           `json:"max_feature_history"`
	
	// Performance settings
	MaxConcurrentAnalyses   int           `json:"max_concurrent_analyses"`
	CacheEnabled            bool          `json:"cache_enabled"`
	CacheTTL                time.Duration `json:"cache_ttl"`
}

// NewMLBehavioralAnalyzer creates a new ML-based behavioral analyzer
func NewMLBehavioralAnalyzer(logger *zap.Logger, config *ThreatDetectionConfig) *MLBehavioralAnalyzer {
	analyzer := &MLBehavioralAnalyzer{
		logger:           logger,
		config:           config,
		behaviorProfiles: make(map[string]*BehaviorProfile),
		stopChan:         make(chan struct{}),
		
		anomaliesDetected: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "ml_anomalies_detected_total",
				Help: "Total number of ML-detected anomalies",
			},
			[]string{"entity_type", "anomaly_type", "severity"},
		),
		
		behaviorScores: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name: "ml_behavior_scores",
				Help: "Distribution of behavioral risk scores",
				Buckets: []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
			},
			[]string{"entity_type", "entity_id"},
		),
		
		modelAccuracy: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "ml_model_accuracy",
				Help: "Accuracy of ML models",
			},
			[]string{"model_type", "model_version"},
		),
		
		predictionLatency: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name: "ml_prediction_latency_seconds",
				Help: "Latency of ML predictions",
				Buckets: prometheus.DefBuckets,
			},
			[]string{"model_type", "entity_type"},
		),
	}
	
	// Initialize components
	analyzer.featureExtractor = NewFeatureExtractor(logger, config)
	analyzer.trainingData = NewTrainingDataStore(logger, config)
	analyzer.modelStore = NewModelStore(logger, config)
	
	// Initialize ML models
	analyzer.initializeModels()
	
	return analyzer
}

// Start begins ML-based behavioral analysis
func (mba *MLBehavioralAnalyzer) Start(ctx context.Context) error {
	mba.mu.Lock()
	defer mba.mu.Unlock()
	
	if mba.running {
		return fmt.Errorf("ML behavioral analyzer already running")
	}
	
	mba.logger.Info("Starting ML behavioral analyzer")
	
	// Start feature extractor
	if err := mba.featureExtractor.Start(ctx); err != nil {
		return fmt.Errorf("failed to start feature extractor: %w", err)
	}
	
	// Start training data store
	if err := mba.trainingData.Start(ctx); err != nil {
		return fmt.Errorf("failed to start training data store: %w", err)
	}
	
	// Start analysis loops
	go mba.behaviorAnalysisLoop(ctx)
	go mba.modelTrainingLoop(ctx)
	go mba.anomalyDetectionLoop(ctx)
	
	mba.running = true
	mba.logger.Info("ML behavioral analyzer started successfully")
	
	return nil
}

// Stop stops the ML behavioral analyzer
func (mba *MLBehavioralAnalyzer) Stop() error {
	mba.mu.Lock()
	defer mba.mu.Unlock()
	
	if !mba.running {
		return nil
	}
	
	mba.logger.Info("Stopping ML behavioral analyzer")
	
	close(mba.stopChan)
	
	if mba.featureExtractor != nil {
		mba.featureExtractor.Stop()
	}
	if mba.trainingData != nil {
		mba.trainingData.Stop()
	}
	
	mba.running = false
	mba.logger.Info("ML behavioral analyzer stopped")
	
	return nil
}

// AnalyzeBehavior analyzes behavioral data and detects anomalies
func (mba *MLBehavioralAnalyzer) AnalyzeBehavior(ctx context.Context, entityID, entityType string, events []SecurityEvent) (*BehaviorAnalysisResult, error) {
	start := time.Now()
	defer func() {
		mba.predictionLatency.WithLabelValues("analysis", entityType).Observe(time.Since(start).Seconds())
	}()
	
	// Get or create behavior profile
	profile := mba.getBehaviorProfile(entityID, entityType)
	
	// Extract features from events
	features, err := mba.featureExtractor.ExtractFeatures(events)
	if err != nil {
		return nil, fmt.Errorf("feature extraction failed: %w", err)
	}
	
	// Update profile with new features
	profile.RecentFeatures = *features
	profile.TotalEvents += int64(len(events))
	profile.LastUpdate = time.Now()
	
	result := &BehaviorAnalysisResult{
		EntityID:   entityID,
		EntityType: entityType,
		Timestamp:  time.Now(),
		Features:   *features,
		Anomalies:  []Anomaly{},
	}
	
	// Detect anomalies using ML models
	if mba.anomalyDetector != nil {
		anomalies := mba.detectAnomalies(profile, features)
		result.Anomalies = append(result.Anomalies, anomalies...)
		profile.CurrentAnomalies = anomalies
		profile.AnomalyCount += int64(len(anomalies))
		
		if len(anomalies) > 0 {
			profile.LastAnomaly = time.Now()
		}
	}
	
	// Classify behavior patterns
	if mba.behaviorClassifier != nil {
		patterns := mba.classifyBehavior(profile, features)
		result.BehaviorPatterns = patterns
	}
	
	// Predict risk score
	if mba.riskPredictor != nil {
		riskScore := mba.predictRisk(profile, features)
		result.RiskScore = riskScore
		profile.RiskScore = riskScore
		
		// Determine threat level
		profile.ThreatLevel = mba.determineThreatLevel(riskScore)
		result.ThreatLevel = profile.ThreatLevel
	}
	
	// Update metrics
	mba.behaviorScores.WithLabelValues(entityType, entityID).Observe(result.RiskScore)
	
	for _, anomaly := range result.Anomalies {
		severity := "low"
		if anomaly.Severity > 0.7 {
			severity = "high"
		} else if anomaly.Severity > 0.4 {
			severity = "medium"
		}
		
		mba.anomaliesDetected.WithLabelValues(entityType, string(anomaly.Type), severity).Inc()
	}
	
	// Store updated profile
	mba.storeBehaviorProfile(profile)
	
	mba.logger.Debug("behavior analysis completed",
		zap.String("entity_id", entityID),
		zap.String("entity_type", entityType),
		zap.Int("events", len(events)),
		zap.Int("anomalies", len(result.Anomalies)),
		zap.Float64("risk_score", result.RiskScore))
	
	return result, nil
}

// detectAnomalies detects behavioral anomalies using ML
func (mba *MLBehavioralAnalyzer) detectAnomalies(profile *BehaviorProfile, features *FeatureVector) []Anomaly {
	var anomalies []Anomaly
	
	if len(profile.BaselineFeatures.LoginTimes) == 0 {
		// Not enough baseline data
		return anomalies
	}
	
	// Detect temporal anomalies
	if len(features.LoginTimes) > 0 && len(profile.BaselineFeatures.LoginTimes) > 0 {
		deviation := mba.calculateTemporalDeviation(features.LoginTimes, profile.BaselineFeatures.LoginTimes)
		if deviation > mba.config.AnomalyThreshold {
			anomalies = append(anomalies, Anomaly{
				ID:          fmt.Sprintf("temporal-%s-%d", profile.EntityID, time.Now().Unix()),
				Type:        AnomalyOutlier,
				Severity:    math.Min(deviation, 1.0),
				Confidence:  0.8,
				Description: "Unusual login timing pattern detected",
				Deviation:   deviation,
				DetectedAt:  time.Now(),
			})
		}
	}
	
	// Detect access pattern anomalies
	accessDeviation := mba.calculateAccessDeviation(features.ResourceAccess, profile.BaselineFeatures.ResourceAccess)
	if accessDeviation > mba.config.AnomalyThreshold {
		anomalies = append(anomalies, Anomaly{
			ID:          fmt.Sprintf("access-%s-%d", profile.EntityID, time.Now().Unix()),
			Type:        AnomalyNovelty,
			Severity:    math.Min(accessDeviation, 1.0),
			Confidence:  0.7,
			Description: "Unusual resource access pattern detected",
			Deviation:   accessDeviation,
			DetectedAt:  time.Now(),
		})
	}
	
	// Detect location anomalies
	locationDeviation := mba.calculateLocationDeviation(features.LocationPatterns, profile.BaselineFeatures.LocationPatterns)
	if locationDeviation > mba.config.AnomalyThreshold {
		anomalies = append(anomalies, Anomaly{
			ID:          fmt.Sprintf("location-%s-%d", profile.EntityID, time.Now().Unix()),
			Type:        AnomalyFrequency,
			Severity:    math.Min(locationDeviation, 1.0),
			Confidence:  0.9,
			Description: "Unusual location access pattern detected",
			Deviation:   locationDeviation,
			DetectedAt:  time.Now(),
		})
	}
	
	// Use Isolation Forest for multivariate anomaly detection
	if mba.anomalyDetector != nil {
		multiVariateScore := mba.calculateIsolationScore(features, profile.BaselineFeatures)
		if multiVariateScore > mba.config.AnomalyThreshold {
			anomalies = append(anomalies, Anomaly{
				ID:          fmt.Sprintf("multivariate-%s-%d", profile.EntityID, time.Now().Unix()),
				Type:        AnomalyOutlier,
				Severity:    math.Min(multiVariateScore, 1.0),
				Confidence:  0.85,
				Description: "Multivariate behavioral anomaly detected",
				Deviation:   multiVariateScore,
				DetectedAt:  time.Now(),
			})
		}
	}
	
	return anomalies
}

// calculateTemporalDeviation calculates deviation in temporal patterns
func (mba *MLBehavioralAnalyzer) calculateTemporalDeviation(current, baseline []float64) float64 {
	if len(current) == 0 || len(baseline) == 0 {
		return 0.0
	}
	
	// Calculate mean and standard deviation for baseline
	baselineMean := stat.Mean(baseline, nil)
	baselineStdDev := stat.StdDev(baseline, nil)
	
	// Calculate Z-score for current values
	var zScores []float64
	for _, value := range current {
		if baselineStdDev > 0 {
			zScore := math.Abs((value - baselineMean) / baselineStdDev)
			zScores = append(zScores, zScore)
		}
	}
	
	if len(zScores) == 0 {
		return 0.0
	}
	
	// Return maximum Z-score as deviation
	sort.Float64s(zScores)
	return math.Min(zScores[len(zScores)-1]/3.0, 1.0) // Normalize to 0-1
}

// calculateAccessDeviation calculates deviation in access patterns
func (mba *MLBehavioralAnalyzer) calculateAccessDeviation(current, baseline map[string]float64) float64 {
	if len(current) == 0 || len(baseline) == 0 {
		return 0.0
	}
	
	// Calculate cosine similarity between current and baseline patterns
	similarity := mba.calculateCosineSimilarity(current, baseline)
	
	// Convert similarity to deviation (1 - similarity)
	return 1.0 - similarity
}

// calculateLocationDeviation calculates deviation in location patterns
func (mba *MLBehavioralAnalyzer) calculateLocationDeviation(current, baseline map[string]float64) float64 {
	if len(current) == 0 || len(baseline) == 0 {
		return 0.0
	}
	
	// Check for new locations not in baseline
	newLocations := 0
	for location := range current {
		if _, exists := baseline[location]; !exists {
			newLocations++
		}
	}
	
	// Calculate novelty score
	noveltyScore := float64(newLocations) / float64(len(current))
	
	// Also calculate frequency deviation
	frequencyDeviation := mba.calculateAccessDeviation(current, baseline)
	
	// Combine novelty and frequency deviations
	return (noveltyScore + frequencyDeviation) / 2.0
}

// calculateCosineSimilarity calculates cosine similarity between two feature maps
func (mba *MLBehavioralAnalyzer) calculateCosineSimilarity(a, b map[string]float64) float64 {
	// Get union of all keys
	allKeys := make(map[string]bool)
	for k := range a {
		allKeys[k] = true
	}
	for k := range b {
		allKeys[k] = true
	}
	
	// Convert to vectors
	var vecA, vecB []float64
	for key := range allKeys {
		vecA = append(vecA, a[key])
		vecB = append(vecB, b[key])
	}
	
	if len(vecA) == 0 {
		return 0.0
	}
	
	// Calculate dot product
	dotProduct := 0.0
	for i := range vecA {
		dotProduct += vecA[i] * vecB[i]
	}
	
	// Calculate magnitudes
	magA := 0.0
	magB := 0.0
	for i := range vecA {
		magA += vecA[i] * vecA[i]
		magB += vecB[i] * vecB[i]
	}
	
	magA = math.Sqrt(magA)
	magB = math.Sqrt(magB)
	
	if magA == 0.0 || magB == 0.0 {
		return 0.0
	}
	
	return dotProduct / (magA * magB)
}

// calculateIsolationScore calculates isolation forest anomaly score
func (mba *MLBehavioralAnalyzer) calculateIsolationScore(features, baseline FeatureVector) float64 {
	// This is a simplified implementation
	// In practice, this would use a trained Isolation Forest model
	
	// Calculate feature-wise deviations
	var deviations []float64
	
	// Temporal deviations
	if len(features.LoginTimes) > 0 && len(baseline.LoginTimes) > 0 {
		deviation := mba.calculateTemporalDeviation(features.LoginTimes, baseline.LoginTimes)
		deviations = append(deviations, deviation)
	}
	
	// Access pattern deviations
	accessDev := mba.calculateAccessDeviation(features.ResourceAccess, baseline.ResourceAccess)
	deviations = append(deviations, accessDev)
	
	// Location pattern deviations
	locationDev := mba.calculateLocationDeviation(features.LocationPatterns, baseline.LocationPatterns)
	deviations = append(deviations, locationDev)
	
	// Network pattern deviations
	networkDev := mba.calculateAccessDeviation(features.NetworkPatterns, baseline.NetworkPatterns)
	deviations = append(deviations, networkDev)
	
	if len(deviations) == 0 {
		return 0.0
	}
	
	// Return average deviation as isolation score
	sum := 0.0
	for _, dev := range deviations {
		sum += dev
	}
	
	return sum / float64(len(deviations))
}

// Additional methods would continue here...
// Including: initializeModels, getBehaviorProfile, classifyBehavior, predictRisk, etc.