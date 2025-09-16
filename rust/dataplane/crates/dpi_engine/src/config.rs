//! DPI Engine Configuration
//!
//! This module provides comprehensive configuration management for the DPI engine,
//! including loading from files, environment variables, and runtime updates.

use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, error, info, warn};

/// DPI configuration errors
#[derive(Error, Debug)]
pub enum DpiError {
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("File not found: {0}")]
    FileNotFound(String),
    
    #[error("Parse error: {0}")]
    ParseError(String),
    
    #[error("Validation error: {0}")]
    ValidationError(String),
    
    #[error("Runtime error: {0}")]
    RuntimeError(String),
}

type Result<T> = std::result::Result<T, DpiError>;

/// Main DPI configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DpiConfig {
    // Core engine settings
    pub engine: EngineConfig,
    
    // Pattern matching configuration
    pub pattern_matching: PatternMatchingConfig,
    
    // ML classification configuration
    pub ml_classification: MLClassificationConfig,
    
    // Security analysis configuration
    pub security: SecurityConfig,
    
    // Protocol analysis configuration
    pub protocol_analysis: ProtocolAnalysisConfig,
    
    // Performance and resource settings
    pub performance: PerformanceConfig,
    
    // Logging and monitoring
    pub logging: LoggingConfig,
    
    // Feature extraction settings
    pub feature_extraction: FeatureExtractionConfig,
    
    // Custom extensions
    pub extensions: HashMap<String, serde_json::Value>,
}

impl Default for DpiConfig {
    fn default() -> Self {
        Self {
            engine: EngineConfig::default(),
            pattern_matching: PatternMatchingConfig::default(),
            ml_classification: MLClassificationConfig::default(),
            security: SecurityConfig::default(),
            protocol_analysis: ProtocolAnalysisConfig::default(),
            performance: PerformanceConfig::default(),
            logging: LoggingConfig::default(),
            feature_extraction: FeatureExtractionConfig::default(),
            extensions: HashMap::new(),
        }
    }
}

/// Core engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub worker_threads: usize,
    pub batch_size: usize,
    pub queue_size: usize,
    pub enable_async_processing: bool,
    pub processing_timeout_ms: u64,
    pub shutdown_timeout_ms: u64,
    pub max_concurrent_packets: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            worker_threads: num_cpus::get(),
            batch_size: 32,
            queue_size: 1024,
            enable_async_processing: true,
            processing_timeout_ms: 1000,
            shutdown_timeout_ms: 5000,
            max_concurrent_packets: 10000,
        }
    }
}

/// Pattern matching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMatchingConfig {
    pub enabled: bool,
    pub engine: PatternEngine,
    pub pattern_database_path: String,
    pub max_pattern_length: usize,
    pub case_sensitive: bool,
    pub enable_regex: bool,
    pub max_matches_per_packet: usize,
    pub cache_size: usize,
    pub update_interval_minutes: u32,
}

impl Default for PatternMatchingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            engine: PatternEngine::AhoCorasick,
            pattern_database_path: "./patterns".to_string(),
            max_pattern_length: 1000,
            case_sensitive: true,
            enable_regex: false,
            max_matches_per_packet: 100,
            cache_size: 10000,
            update_interval_minutes: 60,
        }
    }
}

/// Pattern matching engines
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternEngine {
    AhoCorasick,
    #[cfg(feature = "hardware-acceleration")]
    Hyperscan,
    Regex,
    Custom(String),
}

/// ML classification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLClassificationConfig {
    pub enabled: bool,
    pub model_path: String,
    pub model_type: ModelType,
    pub confidence_threshold: f32,
    pub batch_inference: bool,
    pub max_batch_size: usize,
    pub model_cache_size: usize,
    pub enable_gpu: bool,
    pub gpu_device_id: Option<u32>,
    pub inference_timeout_ms: u64,
    pub feature_normalization: bool,
    pub ensemble_voting: EnsembleVoting,
}

impl Default for MLClassificationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            model_path: "./models".to_string(),
            model_type: ModelType::NeuralNetwork,
            confidence_threshold: 0.8,
            batch_inference: true,
            max_batch_size: 64,
            model_cache_size: 5,
            enable_gpu: false,
            gpu_device_id: None,
            inference_timeout_ms: 100,
            feature_normalization: true,
            ensemble_voting: EnsembleVoting::WeightedAverage,
        }
    }
}

/// ML model types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    NeuralNetwork,
    RandomForest,
    SVM,
    XGBoost,
    LSTM,
    Transformer,
    CNN,
    Ensemble,
    Custom(String),
}

/// Ensemble voting strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnsembleVoting {
    Majority,
    WeightedAverage,
    HighestConfidence,
    Bayesian,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub enabled: bool,
    pub threat_detection: bool,
    pub anomaly_detection: bool,
    pub policy_enforcement: bool,
    pub threat_confidence_threshold: f32,
    pub anomaly_threshold: f32,
    pub max_events_per_second: u32,
    pub correlation_window_seconds: u64,
    pub threat_intelligence: ThreatIntelligenceConfig,
    pub behavioral_analysis: BehavioralAnalysisConfig,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threat_detection: true,
            anomaly_detection: true,
            policy_enforcement: true,
            threat_confidence_threshold: 0.7,
            anomaly_threshold: 2.0,
            max_events_per_second: 1000,
            correlation_window_seconds: 300,
            threat_intelligence: ThreatIntelligenceConfig::default(),
            behavioral_analysis: BehavioralAnalysisConfig::default(),
        }
    }
}

/// Threat intelligence configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatIntelligenceConfig {
    pub enabled: bool,
    pub feeds: Vec<ThreatFeedConfig>,
    pub update_interval_minutes: u32,
    pub cache_ttl_hours: u32,
    pub max_indicators: usize,
}

impl Default for ThreatIntelligenceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            feeds: Vec::new(),
            update_interval_minutes: 60,
            cache_ttl_hours: 24,
            max_indicators: 100000,
        }
    }
}

/// Threat feed configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatFeedConfig {
    pub name: String,
    pub url: String,
    pub format: FeedFormat,
    pub api_key: Option<String>,
    pub enabled: bool,
    pub priority: u32,
}

/// Threat feed formats
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeedFormat {
    Json,
    Xml,
    Csv,
    Stix,
    Taxii,
    Custom(String),
}

/// Behavioral analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralAnalysisConfig {
    pub enabled: bool,
    pub baseline_learning_period_hours: u32,
    pub anomaly_detection_algorithms: Vec<AnomalyAlgorithm>,
    pub statistical_window_minutes: u32,
    pub min_samples_for_baseline: u32,
}

impl Default for BehavioralAnalysisConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            baseline_learning_period_hours: 24,
            anomaly_detection_algorithms: vec![
                AnomalyAlgorithm::ZScore,
                AnomalyAlgorithm::IsolationForest,
            ],
            statistical_window_minutes: 60,
            min_samples_for_baseline: 1000,
        }
    }
}

/// Anomaly detection algorithms
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalyAlgorithm {
    ZScore,
    IsolationForest,
    OneClassSVM,
    LocalOutlierFactor,
    DBSCAN,
    AutoEncoder,
    Custom(String),
}

/// Protocol analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolAnalysisConfig {
    pub enabled: bool,
    pub deep_packet_inspection: bool,
    pub protocol_parsers: HashMap<String, ProtocolParserConfig>,
    pub signature_matching: bool,
    pub behavioral_analysis: bool,
    pub application_detection: bool,
    pub version_detection: bool,
}

impl Default for ProtocolAnalysisConfig {
    fn default() -> Self {
        let mut parsers = HashMap::new();
        parsers.insert("http".to_string(), ProtocolParserConfig::default());
        parsers.insert("https".to_string(), ProtocolParserConfig::default());
        parsers.insert("dns".to_string(), ProtocolParserConfig::default());
        parsers.insert("ssh".to_string(), ProtocolParserConfig::default());
        
        Self {
            enabled: true,
            deep_packet_inspection: true,
            protocol_parsers: parsers,
            signature_matching: true,
            behavioral_analysis: true,
            application_detection: true,
            version_detection: true,
        }
    }
}

/// Protocol parser configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolParserConfig {
    pub enabled: bool,
    pub max_parse_depth: usize,
    pub extract_metadata: bool,
    pub validate_syntax: bool,
    pub custom_rules: Vec<String>,
}

impl Default for ProtocolParserConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_parse_depth: 10,
            extract_metadata: true,
            validate_syntax: true,
            custom_rules: Vec::new(),
        }
    }
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub max_memory_usage_mb: usize,
    pub cache_sizes: CacheConfig,
    pub thread_affinity: bool,
    pub numa_awareness: bool,
    pub prefetch_optimization: bool,
    pub zero_copy: bool,
    pub batch_processing: bool,
    pub pipeline_depth: usize,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_memory_usage_mb: 1024,
            cache_sizes: CacheConfig::default(),
            thread_affinity: false,
            numa_awareness: false,
            prefetch_optimization: true,
            zero_copy: true,
            batch_processing: true,
            pipeline_depth: 4,
        }
    }
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub classification_cache_size: usize,
    pub pattern_cache_size: usize,
    pub feature_cache_size: usize,
    pub model_cache_size: usize,
    pub ttl_seconds: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            classification_cache_size: 10000,
            pattern_cache_size: 5000,
            feature_cache_size: 20000,
            model_cache_size: 5,
            ttl_seconds: 300,
        }
    }
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub format: LogFormat,
    pub output: LogOutput,
    pub enable_metrics: bool,
    pub metrics_interval_seconds: u64,
    pub enable_tracing: bool,
    pub trace_sample_rate: f32,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: LogFormat::Json,
            output: LogOutput::Stdout,
            enable_metrics: true,
            metrics_interval_seconds: 60,
            enable_tracing: false,
            trace_sample_rate: 0.01,
        }
    }
}

/// Log formats
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogFormat {
    Json,
    Plain,
    Structured,
}

/// Log outputs
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogOutput {
    Stdout,
    Stderr,
    File(String),
    Syslog,
    Network(String),
}

/// Feature extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureExtractionConfig {
    pub enabled: bool,
    pub extract_packet_features: bool,
    pub extract_flow_features: bool,
    pub extract_timing_features: bool,
    pub extract_statistical_features: bool,
    pub max_payload_analysis_bytes: usize,
    pub flow_timeout_seconds: u64,
    pub enable_deep_payload_inspection: bool,
    pub entropy_calculation: EntropyMethod,
    pub feature_selection: bool,
    pub dimensionality_reduction: bool,
}

impl Default for FeatureExtractionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            extract_packet_features: true,
            extract_flow_features: true,
            extract_timing_features: true,
            extract_statistical_features: true,
            max_payload_analysis_bytes: 1500,
            flow_timeout_seconds: 600,
            enable_deep_payload_inspection: false,
            entropy_calculation: EntropyMethod::Shannon,
            feature_selection: true,
            dimensionality_reduction: false,
        }
    }
}

/// Entropy calculation methods
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntropyMethod {
    Shannon,
    Renyi,
    Kolmogorov,
    ApproximateEntropy,
}

impl DpiConfig {
    /// Load configuration from file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        info!("Loading DPI configuration from: {}", path.display());
        
        if !path.exists() {
            return Err(DpiError::FileNotFound(path.display().to_string()));
        }
        
        let content = std::fs::read_to_string(path)
            .map_err(|e| DpiError::ConfigError(format!("Failed to read config file: {}", e)))?;
        
        let config = match path.extension().and_then(|s| s.to_str()) {
            Some("toml") => toml::from_str(&content)
                .map_err(|e| DpiError::ParseError(format!("TOML parse error: {}", e)))?,
            Some("json") => serde_json::from_str(&content)
                .map_err(|e| DpiError::ParseError(format!("JSON parse error: {}", e)))?,
            Some("yaml") | Some("yml") => serde_yaml::from_str(&content)
                .map_err(|e| DpiError::ParseError(format!("YAML parse error: {}", e)))?,
            _ => return Err(DpiError::ConfigError("Unsupported config file format".to_string())),
        };
        
        info!("Successfully loaded DPI configuration");
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        info!("Saving DPI configuration to: {}", path.display());
        
        let content = match path.extension().and_then(|s| s.to_str()) {
            Some("toml") => toml::to_string_pretty(self)
                .map_err(|e| DpiError::ConfigError(format!("TOML serialize error: {}", e)))?,
            Some("json") => serde_json::to_string_pretty(self)
                .map_err(|e| DpiError::ConfigError(format!("JSON serialize error: {}", e)))?,
            Some("yaml") | Some("yml") => serde_yaml::to_string(self)
                .map_err(|e| DpiError::ConfigError(format!("YAML serialize error: {}", e)))?,
            _ => return Err(DpiError::ConfigError("Unsupported config file format".to_string())),
        };
        
        std::fs::write(path, content)
            .map_err(|e| DpiError::ConfigError(format!("Failed to write config file: {}", e)))?;
        
        info!("Successfully saved DPI configuration");
        Ok(())
    }
    
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self> {
        info!("Loading DPI configuration from environment variables");
        
        let mut config = Self::default();
        
        // Core engine settings
        if let Ok(threads) = std::env::var("DPI_WORKER_THREADS") {
            config.engine.worker_threads = threads.parse()
                .map_err(|e| DpiError::ValidationError(format!("Invalid worker threads: {}", e)))?;
        }
        
        if let Ok(batch_size) = std::env::var("DPI_BATCH_SIZE") {
            config.engine.batch_size = batch_size.parse()
                .map_err(|e| DpiError::ValidationError(format!("Invalid batch size: {}", e)))?;
        }
        
        // Pattern matching
        if let Ok(enabled) = std::env::var("DPI_PATTERN_MATCHING_ENABLED") {
            config.pattern_matching.enabled = enabled.parse()
                .map_err(|e| DpiError::ValidationError(format!("Invalid pattern matching enabled: {}", e)))?;
        }
        
        if let Ok(db_path) = std::env::var("DPI_PATTERN_DATABASE_PATH") {
            config.pattern_matching.pattern_database_path = db_path;
        }
        
        // ML classification
        if let Ok(enabled) = std::env::var("DPI_ML_ENABLED") {
            config.ml_classification.enabled = enabled.parse()
                .map_err(|e| DpiError::ValidationError(format!("Invalid ML enabled: {}", e)))?;
        }
        
        if let Ok(model_path) = std::env::var("DPI_MODEL_PATH") {
            config.ml_classification.model_path = model_path;
        }
        
        if let Ok(threshold) = std::env::var("DPI_CONFIDENCE_THRESHOLD") {
            config.ml_classification.confidence_threshold = threshold.parse()
                .map_err(|e| DpiError::ValidationError(format!("Invalid confidence threshold: {}", e)))?;
        }
        
        // Security
        if let Ok(enabled) = std::env::var("DPI_SECURITY_ENABLED") {
            config.security.enabled = enabled.parse()
                .map_err(|e| DpiError::ValidationError(format!("Invalid security enabled: {}", e)))?;
        }
        
        // Logging
        if let Ok(level) = std::env::var("DPI_LOG_LEVEL") {
            config.logging.level = level;
        }
        
        info!("Successfully loaded DPI configuration from environment");
        Ok(config)
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        info!("Validating DPI configuration");
        
        // Validate engine settings
        if self.engine.worker_threads == 0 {
            return Err(DpiError::ValidationError("Worker threads must be greater than 0".to_string()));
        }
        
        if self.engine.batch_size == 0 {
            return Err(DpiError::ValidationError("Batch size must be greater than 0".to_string()));
        }
        
        if self.engine.queue_size == 0 {
            return Err(DpiError::ValidationError("Queue size must be greater than 0".to_string()));
        }
        
        // Validate ML settings
        if self.ml_classification.enabled {
            if self.ml_classification.confidence_threshold < 0.0 || self.ml_classification.confidence_threshold > 1.0 {
                return Err(DpiError::ValidationError("Confidence threshold must be between 0.0 and 1.0".to_string()));
            }
            
            if self.ml_classification.max_batch_size == 0 {
                return Err(DpiError::ValidationError("ML batch size must be greater than 0".to_string()));
            }
        }
        
        // Validate security settings
        if self.security.enabled {
            if self.security.threat_confidence_threshold < 0.0 || self.security.threat_confidence_threshold > 1.0 {
                return Err(DpiError::ValidationError("Threat confidence threshold must be between 0.0 and 1.0".to_string()));
            }
            
            if self.security.anomaly_threshold < 0.0 {
                return Err(DpiError::ValidationError("Anomaly threshold must be non-negative".to_string()));
            }
        }
        
        // Validate performance settings
        if self.performance.max_memory_usage_mb == 0 {
            return Err(DpiError::ValidationError("Max memory usage must be greater than 0".to_string()));
        }
        
        // Validate feature extraction settings
        if self.feature_extraction.max_payload_analysis_bytes == 0 {
            return Err(DpiError::ValidationError("Max payload analysis bytes must be greater than 0".to_string()));
        }
        
        info!("DPI configuration validation successful");
        Ok(())
    }
    
    /// Merge with another configuration (other takes precedence)
    pub fn merge(&mut self, other: &DpiConfig) {
        // This would implement a deep merge of configurations
        // For simplicity, just copy some key fields
        self.engine.worker_threads = other.engine.worker_threads;
        self.ml_classification.enabled = other.ml_classification.enabled;
        self.security.enabled = other.security.enabled;
        self.pattern_matching.enabled = other.pattern_matching.enabled;
        
        info!("Merged DPI configuration");
    }
    
    /// Get configuration summary for logging
    pub fn summary(&self) -> String {
        format!(
            "DPI Config: threads={}, ml={}, security={}, patterns={}, batch_size={}",
            self.engine.worker_threads,
            self.ml_classification.enabled,
            self.security.enabled,
            self.pattern_matching.enabled,
            self.engine.batch_size
        )
    }
}

/// Configuration manager for runtime updates
pub struct ConfigManager {
    current_config: parking_lot::RwLock<DpiConfig>,
    config_watchers: Vec<Box<dyn ConfigWatcher + Send + Sync>>,
}

/// Trait for configuration change notifications
pub trait ConfigWatcher: Send + Sync {
    fn on_config_changed(&self, old_config: &DpiConfig, new_config: &DpiConfig);
    fn name(&self) -> &str;
}

impl ConfigManager {
    pub fn new(config: DpiConfig) -> Self {
        Self {
            current_config: parking_lot::RwLock::new(config),
            config_watchers: Vec::new(),
        }
    }
    
    pub fn get_config(&self) -> DpiConfig {
        self.current_config.read().clone()
    }
    
    pub fn update_config(&self, new_config: DpiConfig) -> Result<()> {
        new_config.validate()?;
        
        let old_config = {
            let mut config = self.current_config.write();
            let old = config.clone();
            *config = new_config.clone();
            old
        };
        
        // Notify watchers
        for watcher in &self.config_watchers {
            watcher.on_config_changed(&old_config, &new_config);
        }
        
        info!("Updated DPI configuration: {}", new_config.summary());
        Ok(())
    }
    
    pub fn add_watcher(&mut self, watcher: Box<dyn ConfigWatcher + Send + Sync>) {
        info!("Added config watcher: {}", watcher.name());
        self.config_watchers.push(watcher);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_default_config() {
        let config = DpiConfig::default();
        assert!(config.engine.worker_threads > 0);
        assert!(config.ml_classification.enabled);
        assert!(config.security.enabled);
        assert!(config.pattern_matching.enabled);
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = DpiConfig::default();
        assert!(config.validate().is_ok());
        
        // Test invalid worker threads
        config.engine.worker_threads = 0;
        assert!(config.validate().is_err());
        
        // Test invalid confidence threshold
        config.engine.worker_threads = 4;
        config.ml_classification.confidence_threshold = 1.5;
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_config_file_operations() {
        let config = DpiConfig::default();
        
        // Test JSON format
        let json_file = NamedTempFile::with_suffix(".json").unwrap();
        config.to_file(&json_file).unwrap();
        let loaded_config = DpiConfig::from_file(&json_file).unwrap();
        assert_eq!(config.engine.worker_threads, loaded_config.engine.worker_threads);
    }
    
    #[test]
    fn test_config_manager() {
        let config = DpiConfig::default();
        let manager = ConfigManager::new(config.clone());
        
        let retrieved = manager.get_config();
        assert_eq!(config.engine.worker_threads, retrieved.engine.worker_threads);
        
        let mut new_config = config.clone();
        new_config.engine.worker_threads = 8;
        assert!(manager.update_config(new_config).is_ok());
        
        let updated = manager.get_config();
        assert_eq!(updated.engine.worker_threads, 8);
    }
}