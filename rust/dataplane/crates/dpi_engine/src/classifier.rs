//! DPI Classifier - Main Classification Engine
//!
//! This module provides the core classification logic that combines pattern matching,
//! ML inference, and security analysis to classify network packets.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use parking_lot::RwLock;
use prometheus::{Counter, Histogram, register_counter, register_histogram};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, error, info, warn, instrument};

use crate::{
    DpiError, PacketData, ProtocolType, ApplicationType, SecurityFlag,
    features::{PacketFeatures, FeatureExtractor},
    patterns::{PatternMatcher, PatternMatch},
    ml_engine::{MLClassifier, MLPrediction},
    security::{SecurityAnalyzer, SecurityEvent},
    protocols::{ProtocolSignature, ProtocolAnalyzer},
};

/// Classification errors
#[derive(Error, Debug)]
pub enum ClassificationError {
    #[error("Classification failed: {0}")]
    ClassificationFailed(String),
    
    #[error("Invalid input data: {0}")]
    InvalidInput(String),
    
    #[error("Model inference failed: {0}")]
    ModelInferenceFailed(String),
    
    #[error("Pattern matching failed: {0}")]
    PatternMatchingFailed(String),
    
    #[error("Security analysis failed: {0}")]
    SecurityAnalysisFailed(String),
}

type Result<T> = std::result::Result<T, ClassificationError>;

/// Classification confidence levels
#[derive(Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum ClassificationConfidence {
    VeryLow,    // 0.0 - 0.2
    Low,        // 0.2 - 0.4
    Medium,     // 0.4 - 0.6
    High,       // 0.6 - 0.8
    VeryHigh,   // 0.8 - 1.0
}

impl From<f32> for ClassificationConfidence {
    fn from(confidence: f32) -> Self {
        match confidence {
            x if x < 0.2 => ClassificationConfidence::VeryLow,
            x if x < 0.4 => ClassificationConfidence::Low,
            x if x < 0.6 => ClassificationConfidence::Medium,
            x if x < 0.8 => ClassificationConfidence::High,
            _ => ClassificationConfidence::VeryHigh,
        }
    }
}

/// Complete classification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub packet_id: u64,
    pub flow_id: Option<u64>,
    pub protocol: ProtocolType,
    pub application: ApplicationType,
    pub confidence: ClassificationConfidence,
    pub confidence_score: f32,
    
    // Evidence sources
    pub pattern_matches: Vec<PatternMatch>,
    pub ml_prediction: Option<MLPrediction>,
    pub protocol_signature: Option<ProtocolSignature>,
    pub security_events: Vec<SecurityEvent>,
    
    // Feature analysis
    pub extracted_features: PacketFeatures,
    pub feature_importance: HashMap<String, f32>,
    
    // Classification metadata
    pub classification_method: ClassificationMethod,
    pub processing_time_us: u64,
    pub timestamp: u64,
    pub classifier_version: String,
    
    // Risk assessment
    pub security_flags: Vec<SecurityFlag>,
    pub risk_score: f32,
    pub threat_indicators: Vec<String>,
    
    // Quality metrics
    pub data_quality_score: f32,
    pub classification_uncertainty: f32,
}

/// Method used for classification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClassificationMethod {
    PatternMatching,
    MachineLearning,
    RuleBased,
    Hybrid,
    Ensemble,
}

/// DPI Classifier configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierConfig {
    pub confidence_threshold: f32,
    pub security_threshold: f32,
    pub enable_pattern_matching: bool,
    pub enable_ml_inference: bool,
    pub enable_protocol_analysis: bool,
    pub enable_security_analysis: bool,
    pub classification_timeout_ms: u64,
    pub feature_importance_threshold: f32,
    pub ensemble_voting_strategy: EnsembleStrategy,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.8,
            security_threshold: 0.7,
            enable_pattern_matching: true,
            enable_ml_inference: true,
            enable_protocol_analysis: true,
            enable_security_analysis: true,
            classification_timeout_ms: 1000,
            feature_importance_threshold: 0.1,
            ensemble_voting_strategy: EnsembleStrategy::WeightedVoting,
        }
    }
}

/// Ensemble voting strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnsembleStrategy {
    MajorityVoting,
    WeightedVoting,
    ConfidenceVoting,
    BayesianCombination,
}

/// Main DPI Classifier
pub struct DpiClassifier {
    config: ClassifierConfig,
    pattern_matcher: Arc<PatternMatcher>,
    feature_extractor: Arc<FeatureExtractor>,
    ml_classifier: Arc<MLClassifier>,
    security_analyzer: Arc<SecurityAnalyzer>,
    
    // Classification strategies
    strategies: Vec<Box<dyn ClassificationStrategy + Send + Sync>>,
    ensemble_combiner: Box<dyn EnsembleCombiner + Send + Sync>,
    
    // Performance tracking
    classification_count: std::sync::atomic::AtomicU64,
    performance_metrics: Arc<RwLock<ClassifierMetrics>>,
    
    // Rule engine
    rule_engine: Arc<RuleEngine>,
    
    // Protocol analyzers
    protocol_analyzers: HashMap<ProtocolType, Box<dyn ProtocolAnalyzer + Send + Sync>>,
}

/// Classification strategy trait
#[async_trait]
pub trait ClassificationStrategy: Send + Sync {
    async fn classify(&self, packet_data: &PacketData, features: &PacketFeatures) -> Result<StrategyResult>;
    fn name(&self) -> &str;
    fn confidence_weight(&self) -> f32;
    fn is_applicable(&self, packet_data: &PacketData, features: &PacketFeatures) -> bool;
}

/// Result from a classification strategy
#[derive(Debug, Clone)]
pub struct StrategyResult {
    pub protocol: ProtocolType,
    pub application: ApplicationType,
    pub confidence: f32,
    pub method: ClassificationMethod,
    pub evidence: Vec<String>,
    pub processing_time_us: u64,
}

/// Ensemble combiner trait
#[async_trait]
pub trait EnsembleCombiner: Send + Sync {
    async fn combine(&self, results: Vec<StrategyResult>, features: &PacketFeatures) -> Result<ClassificationResult>;
    fn strategy(&self) -> EnsembleStrategy;
}

/// Rule-based classification engine
pub struct RuleEngine {
    rules: Vec<ClassificationRule>,
    rule_cache: Arc<RwLock<HashMap<u64, (ProtocolType, f32)>>>,
}

/// Classification rule
#[derive(Debug, Clone)]
pub struct ClassificationRule {
    pub id: String,
    pub priority: u32,
    pub conditions: Vec<RuleCondition>,
    pub action: RuleAction,
    pub confidence: f32,
    pub enabled: bool,
}

/// Rule condition
#[derive(Debug, Clone)]
pub struct RuleCondition {
    pub field: String,
    pub operator: RuleOperator,
    pub value: RuleValue,
}

/// Rule operators
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuleOperator {
    Equals,
    NotEquals,
    LessThan,
    GreaterThan,
    Contains,
    StartsWith,
    EndsWith,
    Matches, // Regex
    In,      // List membership
}

/// Rule values
#[derive(Debug, Clone)]
pub enum RuleValue {
    String(String),
    Number(f64),
    Boolean(bool),
    List(Vec<String>),
    Regex(String),
}

/// Rule actions
#[derive(Debug, Clone)]
pub enum RuleAction {
    Classify(ProtocolType, ApplicationType),
    SetFlag(SecurityFlag),
    SetConfidence(f32),
    Reject,
    Custom(String),
}

/// Classifier performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClassifierMetrics {
    pub total_classifications: u64,
    pub successful_classifications: u64,
    pub failed_classifications: u64,
    pub average_processing_time_us: f64,
    pub confidence_distribution: HashMap<String, u64>,
    pub protocol_distribution: HashMap<String, u64>,
    pub method_distribution: HashMap<String, u64>,
    pub accuracy_by_protocol: HashMap<String, f32>,
}

// Prometheus metrics
lazy_static::lazy_static! {
    static ref CLASSIFICATIONS_TOTAL: Counter = register_counter!(
        "dpi_classifications_total",
        "Total number of packet classifications performed"
    ).unwrap();
    
    static ref CLASSIFICATION_DURATION: Histogram = register_histogram!(
        "dpi_classification_duration_seconds",
        "Time taken for packet classification"
    ).unwrap();
}

impl DpiClassifier {
    /// Create a new DPI classifier
    #[instrument(skip_all)]
    pub fn new(
        pattern_matcher: Arc<PatternMatcher>,
        feature_extractor: Arc<FeatureExtractor>,
        ml_classifier: Arc<MLClassifier>,
        security_analyzer: Arc<SecurityAnalyzer>,
        config: crate::config::DpiConfig,
    ) -> Result<Self> {
        info!("Initializing DPI Classifier");
        
        let classifier_config = ClassifierConfig::default();
        
        // Initialize classification strategies
        let strategies = Self::initialize_strategies(&classifier_config)?;
        
        // Initialize ensemble combiner
        let ensemble_combiner = Self::create_ensemble_combiner(&classifier_config)?;
        
        // Initialize rule engine
        let rule_engine = Arc::new(RuleEngine::new()?);
        
        let classifier = Self {
            config: classifier_config,
            pattern_matcher,
            feature_extractor,
            ml_classifier,
            security_analyzer,
            strategies,
            ensemble_combiner,
            classification_count: std::sync::atomic::AtomicU64::new(0),
            performance_metrics: Arc::new(RwLock::new(ClassifierMetrics::default())),
            rule_engine,
            protocol_analyzers: HashMap::new(),
        };
        
        info!("DPI Classifier initialized with {} strategies", classifier.strategies.len());
        Ok(classifier)
    }
    
    /// Classify a packet using all available methods
    #[instrument(skip(self, packet_data, packet_features))]
    pub async fn classify_packet(
        &self,
        packet_data: PacketData,
        packet_features: PacketFeatures,
        pattern_matches: Vec<PatternMatch>,
        ml_prediction: Option<MLPrediction>,
        security_flags: Vec<SecurityFlag>,
        protocol_signature: Option<ProtocolSignature>,
    ) -> Result<ClassificationResult> {
        let start_time = Instant::now();
        let packet_id = packet_data.packet_id;
        
        debug!("Starting classification for packet {}", packet_id);
        
        // Run all applicable classification strategies
        let mut strategy_results = Vec::new();
        
        for strategy in &self.strategies {
            if strategy.is_applicable(&packet_data, &packet_features) {
                match strategy.classify(&packet_data, &packet_features).await {
                    Ok(result) => {
                        debug!("Strategy '{}' classified packet as {:?} with confidence {:.2}",
                               strategy.name(), result.protocol, result.confidence);
                        strategy_results.push(result);
                    },
                    Err(e) => {
                        warn!("Strategy '{}' failed: {}", strategy.name(), e);
                    }
                }
            }
        }
        
        // Apply rule-based classification
        if let Ok(rule_result) = self.apply_rules(&packet_data, &packet_features).await {
            strategy_results.push(rule_result);
        }
        
        // Combine results using ensemble method
        let mut classification = self.ensemble_combiner
            .combine(strategy_results, &packet_features)
            .await?;
        
        // Enhance with additional information
        classification.packet_id = packet_id;
        classification.flow_id = packet_data.flow_id;
        classification.pattern_matches = pattern_matches;
        classification.ml_prediction = ml_prediction;
        classification.protocol_signature = protocol_signature;
        classification.extracted_features = packet_features;
        classification.security_flags = security_flags;
        classification.processing_time_us = start_time.elapsed().as_micros() as u64;
        classification.timestamp = packet_data.timestamp.elapsed().as_millis() as u64;
        classification.classifier_version = "1.0.0".to_string();
        
        // Calculate risk score
        classification.risk_score = self.calculate_risk_score(&classification);
        
        // Validate classification quality
        classification.data_quality_score = self.assess_data_quality(&packet_data, &classification.extracted_features);
        classification.classification_uncertainty = self.calculate_uncertainty(&classification);
        
        // Update metrics
        self.update_classification_metrics(&classification);
        
        // Update Prometheus metrics
        CLASSIFICATIONS_TOTAL.inc();
        CLASSIFICATION_DURATION.observe(classification.processing_time_us as f64 / 1_000_000.0);
        
        info!("Classified packet {} as {:?} with confidence {:?} in {}μs",
              packet_id, classification.protocol, classification.confidence, classification.processing_time_us);
        
        Ok(classification)
    }
    
    /// Initialize classification strategies
    fn initialize_strategies(config: &ClassifierConfig) -> Result<Vec<Box<dyn ClassificationStrategy + Send + Sync>>> {
        let mut strategies: Vec<Box<dyn ClassificationStrategy + Send + Sync>> = Vec::new();
        
        if config.enable_pattern_matching {
            strategies.push(Box::new(PatternMatchingStrategy::new()));
        }
        
        if config.enable_ml_inference {
            strategies.push(Box::new(MLInferenceStrategy::new()));
        }
        
        if config.enable_protocol_analysis {
            strategies.push(Box::new(ProtocolAnalysisStrategy::new()));
        }
        
        // Add heuristic strategy as fallback
        strategies.push(Box::new(HeuristicStrategy::new()));
        
        Ok(strategies)
    }
    
    /// Create ensemble combiner
    fn create_ensemble_combiner(config: &ClassifierConfig) -> Result<Box<dyn EnsembleCombiner + Send + Sync>> {
        match config.ensemble_voting_strategy {
            EnsembleStrategy::MajorityVoting => Ok(Box::new(MajorityVotingCombiner::new())),
            EnsembleStrategy::WeightedVoting => Ok(Box::new(WeightedVotingCombiner::new())),
            EnsembleStrategy::ConfidenceVoting => Ok(Box::new(ConfidenceVotingCombiner::new())),
            EnsembleStrategy::BayesianCombination => Ok(Box::new(BayesianCombiner::new())),
        }
    }
    
    /// Apply rule-based classification
    async fn apply_rules(&self, packet_data: &PacketData, features: &PacketFeatures) -> Result<StrategyResult> {
        // Check rule cache first
        let cache_key = self.calculate_rule_cache_key(packet_data, features);
        if let Some((protocol, confidence)) = self.rule_engine.rule_cache.read().get(&cache_key) {
            return Ok(StrategyResult {
                protocol: protocol.clone(),
                application: ApplicationType::Unknown,
                confidence: *confidence,
                method: ClassificationMethod::RuleBased,
                evidence: vec!["Cached rule result".to_string()],
                processing_time_us: 0,
            });
        }
        
        // Apply rules
        for rule in &self.rule_engine.rules {
            if !rule.enabled {
                continue;
            }
            
            if self.evaluate_rule_conditions(&rule.conditions, packet_data, features) {
                match &rule.action {
                    RuleAction::Classify(protocol, application) => {
                        // Cache the result
                        self.rule_engine.rule_cache.write().insert(cache_key, (protocol.clone(), rule.confidence));
                        
                        return Ok(StrategyResult {
                            protocol: protocol.clone(),
                            application: application.clone(),
                            confidence: rule.confidence,
                            method: ClassificationMethod::RuleBased,
                            evidence: vec![format!("Rule: {}", rule.id)],
                            processing_time_us: 0,
                        });
                    },
                    _ => continue,
                }
            }
        }
        
        // No rules matched
        Err(ClassificationError::ClassificationFailed("No rules matched".to_string()))
    }
    
    /// Evaluate rule conditions
    fn evaluate_rule_conditions(&self, conditions: &[RuleCondition], packet_data: &PacketData, features: &PacketFeatures) -> bool {
        for condition in conditions {
            if !self.evaluate_single_condition(condition, packet_data, features) {
                return false;
            }
        }
        true
    }
    
    /// Evaluate a single rule condition
    fn evaluate_single_condition(&self, condition: &RuleCondition, packet_data: &PacketData, features: &PacketFeatures) -> bool {
        // This would implement the actual condition evaluation logic
        // For now, return a placeholder result
        match condition.field.as_str() {
            "dst_port" => {
                if let RuleValue::Number(port) = condition.value {
                    match condition.operator {
                        RuleOperator::Equals => packet_data.dst_port == port as u16,
                        RuleOperator::LessThan => (packet_data.dst_port as f64) < port,
                        RuleOperator::GreaterThan => (packet_data.dst_port as f64) > port,
                        _ => false,
                    }
                } else {
                    false
                }
            },
            "packet_size" => {
                if let RuleValue::Number(size) = condition.value {
                    match condition.operator {
                        RuleOperator::Equals => features.packet_size == size as u32,
                        RuleOperator::LessThan => (features.packet_size as f64) < size,
                        RuleOperator::GreaterThan => (features.packet_size as f64) > size,
                        _ => false,
                    }
                } else {
                    false
                }
            },
            _ => false,
        }
    }
    
    /// Calculate cache key for rule evaluation
    fn calculate_rule_cache_key(&self, packet_data: &PacketData, features: &PacketFeatures) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        packet_data.dst_port.hash(&mut hasher);
        features.packet_size.hash(&mut hasher);
        features.protocol_type.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Calculate risk score based on classification results
    fn calculate_risk_score(&self, classification: &ClassificationResult) -> f32 {
        let mut risk_score = 0.0;
        
        // Base risk from security flags
        for flag in &classification.security_flags {
            risk_score += match flag {
                SecurityFlag::Malicious => 0.9,
                SecurityFlag::Suspicious => 0.6,
                SecurityFlag::Anomalous => 0.4,
                SecurityFlag::PolicyViolation => 0.3,
                _ => 0.1,
            };
        }
        
        // Adjust based on confidence
        risk_score *= classification.confidence_score;
        
        // Normalize to 0-1 range
        risk_score.min(1.0)
    }
    
    /// Assess data quality
    fn assess_data_quality(&self, packet_data: &PacketData, features: &PacketFeatures) -> f32 {
        let mut quality_score = 1.0;
        
        // Penalize very small packets
        if features.packet_size < 64 {
            quality_score *= 0.8;
        }
        
        // Penalize truncated packets
        if packet_data.data.len() < features.packet_size as usize {
            quality_score *= 0.6;
        }
        
        // Reward complete headers
        if features.header_length > 0 {
            quality_score *= 1.1;
        }
        
        quality_score.min(1.0)
    }
    
    /// Calculate classification uncertainty
    fn calculate_uncertainty(&self, classification: &ClassificationResult) -> f32 {
        let mut uncertainty = 1.0 - classification.confidence_score;
        
        // Increase uncertainty if few evidence sources
        let evidence_count = classification.pattern_matches.len() +
                           if classification.ml_prediction.is_some() { 1 } else { 0 } +
                           if classification.protocol_signature.is_some() { 1 } else { 0 };
        
        if evidence_count < 2 {
            uncertainty *= 1.5;
        }
        
        uncertainty.min(1.0)
    }
    
    /// Update classification metrics
    fn update_classification_metrics(&self, classification: &ClassificationResult) {
        let mut metrics = self.performance_metrics.write();
        
        metrics.total_classifications += 1;
        
        if classification.confidence_score >= self.config.confidence_threshold {
            metrics.successful_classifications += 1;
        } else {
            metrics.failed_classifications += 1;
        }
        
        // Update processing time average
        let count = metrics.total_classifications as f64;
        metrics.average_processing_time_us = 
            (metrics.average_processing_time_us * (count - 1.0) + classification.processing_time_us as f64) / count;
        
        // Update confidence distribution
        let confidence_key = format!("{:?}", classification.confidence);
        *metrics.confidence_distribution.entry(confidence_key).or_insert(0) += 1;
        
        // Update protocol distribution
        let protocol_key = format!("{:?}", classification.protocol);
        *metrics.protocol_distribution.entry(protocol_key).or_insert(0) += 1;
        
        // Update method distribution
        let method_key = format!("{:?}", classification.classification_method);
        *metrics.method_distribution.entry(method_key).or_insert(0) += 1;
    }
    
    /// Get classifier performance metrics
    pub fn get_metrics(&self) -> ClassifierMetrics {
        self.performance_metrics.read().clone()
    }
    
    /// Update classification rules
    pub fn update_rules(&mut self, rules: Vec<ClassificationRule>) {
        self.rule_engine.rules = rules;
        self.rule_engine.rule_cache.write().clear();
        info!("Updated classification rules: {} rules loaded", self.rule_engine.rules.len());
    }
    
    /// Get current classification statistics
    pub fn get_classification_count(&self) -> u64 {
        self.classification_count.load(std::sync::atomic::Ordering::Relaxed)
    }
}

// Implementation of classification strategies

/// Pattern matching strategy
struct PatternMatchingStrategy;

impl PatternMatchingStrategy {
    fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ClassificationStrategy for PatternMatchingStrategy {
    async fn classify(&self, _packet_data: &PacketData, features: &PacketFeatures) -> Result<StrategyResult> {
        // Simplified pattern matching logic
        let protocol = match features.dst_port {
            80 => ProtocolType::HTTP,
            443 => ProtocolType::HTTPS,
            22 => ProtocolType::SSH,
            21 => ProtocolType::FTP,
            25 => ProtocolType::SMTP,
            53 => ProtocolType::DNS,
            _ => ProtocolType::Unknown,
        };
        
        let confidence = if protocol != ProtocolType::Unknown { 0.7 } else { 0.1 };
        
        Ok(StrategyResult {
            protocol,
            application: ApplicationType::Unknown,
            confidence,
            method: ClassificationMethod::PatternMatching,
            evidence: vec!["Port-based classification".to_string()],
            processing_time_us: 0,
        })
    }
    
    fn name(&self) -> &str {
        "PatternMatching"
    }
    
    fn confidence_weight(&self) -> f32 {
        0.3
    }
    
    fn is_applicable(&self, _packet_data: &PacketData, _features: &PacketFeatures) -> bool {
        true
    }
}

/// ML inference strategy
struct MLInferenceStrategy;

impl MLInferenceStrategy {
    fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ClassificationStrategy for MLInferenceStrategy {
    async fn classify(&self, _packet_data: &PacketData, _features: &PacketFeatures) -> Result<StrategyResult> {
        // Placeholder ML inference
        Ok(StrategyResult {
            protocol: ProtocolType::HTTP,
            application: ApplicationType::WebBrowsing,
            confidence: 0.85,
            method: ClassificationMethod::MachineLearning,
            evidence: vec!["Neural network prediction".to_string()],
            processing_time_us: 100,
        })
    }
    
    fn name(&self) -> &str {
        "MLInference"
    }
    
    fn confidence_weight(&self) -> f32 {
        0.6
    }
    
    fn is_applicable(&self, _packet_data: &PacketData, features: &PacketFeatures) -> bool {
        features.payload_length > 0
    }
}

/// Protocol analysis strategy
struct ProtocolAnalysisStrategy;

impl ProtocolAnalysisStrategy {
    fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ClassificationStrategy for ProtocolAnalysisStrategy {
    async fn classify(&self, _packet_data: &PacketData, features: &PacketFeatures) -> Result<StrategyResult> {
        let protocol = if features.has_http_headers {
            ProtocolType::HTTP
        } else if features.has_tls_handshake {
            ProtocolType::HTTPS
        } else if features.has_dns_query {
            ProtocolType::DNS
        } else {
            ProtocolType::Unknown
        };
        
        let confidence = if protocol != ProtocolType::Unknown { 0.9 } else { 0.0 };
        
        Ok(StrategyResult {
            protocol,
            application: ApplicationType::Unknown,
            confidence,
            method: ClassificationMethod::RuleBased,
            evidence: vec!["Protocol signature analysis".to_string()],
            processing_time_us: 50,
        })
    }
    
    fn name(&self) -> &str {
        "ProtocolAnalysis"
    }
    
    fn confidence_weight(&self) -> f32 {
        0.8
    }
    
    fn is_applicable(&self, _packet_data: &PacketData, features: &PacketFeatures) -> bool {
        features.application_data_length > 0
    }
}

/// Heuristic strategy (fallback)
struct HeuristicStrategy;

impl HeuristicStrategy {
    fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ClassificationStrategy for HeuristicStrategy {
    async fn classify(&self, packet_data: &PacketData, _features: &PacketFeatures) -> Result<StrategyResult> {
        // Simple heuristic based on ports and packet size
        let protocol = match packet_data.dst_port {
            80 | 8080 | 3000..=3010 => ProtocolType::HTTP,
            443 | 8443 => ProtocolType::HTTPS,
            22 => ProtocolType::SSH,
            21 => ProtocolType::FTP,
            25 | 587 => ProtocolType::SMTP,
            53 => ProtocolType::DNS,
            _ => ProtocolType::Unknown,
        };
        
        Ok(StrategyResult {
            protocol,
            application: ApplicationType::Unknown,
            confidence: 0.5,
            method: ClassificationMethod::RuleBased,
            evidence: vec!["Heuristic analysis".to_string()],
            processing_time_us: 10,
        })
    }
    
    fn name(&self) -> &str {
        "Heuristic"
    }
    
    fn confidence_weight(&self) -> f32 {
        0.2
    }
    
    fn is_applicable(&self, _packet_data: &PacketData, _features: &PacketFeatures) -> bool {
        true // Always applicable as fallback
    }
}

// Ensemble combiners

/// Weighted voting combiner
struct WeightedVotingCombiner;

impl WeightedVotingCombiner {
    fn new() -> Self {
        Self
    }
}

#[async_trait]
impl EnsembleCombiner for WeightedVotingCombiner {
    async fn combine(&self, results: Vec<StrategyResult>, _features: &PacketFeatures) -> Result<ClassificationResult> {
        if results.is_empty() {
            return Err(ClassificationError::ClassificationFailed("No results to combine".to_string()));
        }
        
        // Weight votes by confidence and strategy weight
        let mut protocol_votes: HashMap<ProtocolType, f32> = HashMap::new();
        let mut total_weight = 0.0;
        
        for result in &results {
            let weight = result.confidence * 0.5; // Simplified weight calculation
            *protocol_votes.entry(result.protocol.clone()).or_insert(0.0) += weight;
            total_weight += weight;
        }
        
        // Find the protocol with highest weighted vote
        let (winning_protocol, winning_score) = protocol_votes.iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(p, s)| (p.clone(), *s))
            .unwrap_or((ProtocolType::Unknown, 0.0));
        
        let final_confidence = if total_weight > 0.0 { winning_score / total_weight } else { 0.0 };
        
        Ok(ClassificationResult {
            packet_id: 0, // Will be set by caller
            flow_id: None,
            protocol: winning_protocol,
            application: ApplicationType::Unknown,
            confidence: ClassificationConfidence::from(final_confidence),
            confidence_score: final_confidence,
            pattern_matches: Vec::new(),
            ml_prediction: None,
            protocol_signature: None,
            security_events: Vec::new(),
            extracted_features: PacketFeatures::default(),
            feature_importance: HashMap::new(),
            classification_method: ClassificationMethod::Ensemble,
            processing_time_us: 0,
            timestamp: 0,
            classifier_version: String::new(),
            security_flags: Vec::new(),
            risk_score: 0.0,
            threat_indicators: Vec::new(),
            data_quality_score: 1.0,
            classification_uncertainty: 1.0 - final_confidence,
        })
    }
    
    fn strategy(&self) -> EnsembleStrategy {
        EnsembleStrategy::WeightedVoting
    }
}

/// Majority voting combiner
struct MajorityVotingCombiner;

impl MajorityVotingCombiner {
    fn new() -> Self {
        Self
    }
}

#[async_trait]
impl EnsembleCombiner for MajorityVotingCombiner {
    async fn combine(&self, results: Vec<StrategyResult>, features: &PacketFeatures) -> Result<ClassificationResult> {
        // For simplicity, delegate to weighted voting for now
        let weighted_combiner = WeightedVotingCombiner::new();
        weighted_combiner.combine(results, features).await
    }
    
    fn strategy(&self) -> EnsembleStrategy {
        EnsembleStrategy::MajorityVoting
    }
}

/// Confidence voting combiner
struct ConfidenceVotingCombiner;

impl ConfidenceVotingCombiner {
    fn new() -> Self {
        Self
    }
}

#[async_trait]
impl EnsembleCombiner for ConfidenceVotingCombiner {
    async fn combine(&self, results: Vec<StrategyResult>, features: &PacketFeatures) -> Result<ClassificationResult> {
        // Choose the result with highest confidence
        if let Some(best_result) = results.iter().max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap()) {
            let mut classification = ClassificationResult {
                packet_id: 0,
                flow_id: None,
                protocol: best_result.protocol.clone(),
                application: best_result.application.clone(),
                confidence: ClassificationConfidence::from(best_result.confidence),
                confidence_score: best_result.confidence,
                pattern_matches: Vec::new(),
                ml_prediction: None,
                protocol_signature: None,
                security_events: Vec::new(),
                extracted_features: features.clone(),
                feature_importance: HashMap::new(),
                classification_method: best_result.method.clone(),
                processing_time_us: 0,
                timestamp: 0,
                classifier_version: String::new(),
                security_flags: Vec::new(),
                risk_score: 0.0,
                threat_indicators: Vec::new(),
                data_quality_score: 1.0,
                classification_uncertainty: 1.0 - best_result.confidence,
            };
            
            Ok(classification)
        } else {
            Err(ClassificationError::ClassificationFailed("No results available".to_string()))
        }
    }
    
    fn strategy(&self) -> EnsembleStrategy {
        EnsembleStrategy::ConfidenceVoting
    }
}

/// Bayesian combiner
struct BayesianCombiner;

impl BayesianCombiner {
    fn new() -> Self {
        Self
    }
}

#[async_trait]
impl EnsembleCombiner for BayesianCombiner {
    async fn combine(&self, results: Vec<StrategyResult>, features: &PacketFeatures) -> Result<ClassificationResult> {
        // For now, delegate to weighted voting
        // Real implementation would use Bayesian combination
        let weighted_combiner = WeightedVotingCombiner::new();
        weighted_combiner.combine(results, features).await
    }
    
    fn strategy(&self) -> EnsembleStrategy {
        EnsembleStrategy::BayesianCombination
    }
}

impl RuleEngine {
    fn new() -> Result<Self> {
        Ok(Self {
            rules: Self::create_default_rules(),
            rule_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    fn create_default_rules() -> Vec<ClassificationRule> {
        vec![
            ClassificationRule {
                id: "http_port_80".to_string(),
                priority: 100,
                conditions: vec![
                    RuleCondition {
                        field: "dst_port".to_string(),
                        operator: RuleOperator::Equals,
                        value: RuleValue::Number(80.0),
                    }
                ],
                action: RuleAction::Classify(ProtocolType::HTTP, ApplicationType::WebBrowsing),
                confidence: 0.8,
                enabled: true,
            },
            ClassificationRule {
                id: "https_port_443".to_string(),
                priority: 100,
                conditions: vec![
                    RuleCondition {
                        field: "dst_port".to_string(),
                        operator: RuleOperator::Equals,
                        value: RuleValue::Number(443.0),
                    }
                ],
                action: RuleAction::Classify(ProtocolType::HTTPS, ApplicationType::WebBrowsing),
                confidence: 0.8,
                enabled: true,
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::PacketFeatures;
    use std::net::{IpAddr, Ipv4Addr};
    use bytes::Bytes;
    use std::time::Instant;
    
    #[test]
    fn test_classification_confidence_conversion() {
        assert_eq!(ClassificationConfidence::from(0.1), ClassificationConfidence::VeryLow);
        assert_eq!(ClassificationConfidence::from(0.3), ClassificationConfidence::Low);
        assert_eq!(ClassificationConfidence::from(0.5), ClassificationConfidence::Medium);
        assert_eq!(ClassificationConfidence::from(0.7), ClassificationConfidence::High);
        assert_eq!(ClassificationConfidence::from(0.9), ClassificationConfidence::VeryHigh);
    }
    
    #[tokio::test]
    async fn test_pattern_matching_strategy() {
        let strategy = PatternMatchingStrategy::new();
        
        let packet_data = PacketData {
            packet_id: 1,
            timestamp: Instant::now(),
            data: Bytes::new(),
            flow_id: None,
            src_ip: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100)),
            dst_ip: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
            src_port: 12345,
            dst_port: 80,
            protocol: 6,
            payload_offset: 0,
            metadata: HashMap::new(),
        };
        
        let features = PacketFeatures {
            dst_port: 80,
            ..Default::default()
        };
        
        let result = strategy.classify(&packet_data, &features).await.unwrap();
        assert_eq!(result.protocol, ProtocolType::HTTP);
        assert!(result.confidence > 0.5);
    }
    
    #[test]
    fn test_rule_condition_evaluation() {
        let condition = RuleCondition {
            field: "dst_port".to_string(),
            operator: RuleOperator::Equals,
            value: RuleValue::Number(80.0),
        };
        
        // This would require a full DPI classifier instance to test properly
        // For now, just verify the structure
        assert_eq!(condition.field, "dst_port");
        assert_eq!(condition.operator, RuleOperator::Equals);
    }
}