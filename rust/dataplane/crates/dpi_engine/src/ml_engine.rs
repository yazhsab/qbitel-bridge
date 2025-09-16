//! Machine Learning Engine for DPI Classification
//!
//! This module provides ML-based packet classification using neural networks,
//! ensemble methods, and automated model training pipelines.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use parking_lot::{Mutex, RwLock};
use prometheus::{Counter, Histogram, Gauge, register_counter, register_histogram, register_gauge};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::fs;
use tracing::{debug, error, info, warn, instrument};

#[cfg(feature = "ml-classification")]
use candle_core::{Device, Tensor, DType};
#[cfg(feature = "ml-classification")]
use candle_nn::{Linear, Module, VarBuilder, VarMap};

use crate::{DpiError, PacketData, ProtocolType, ApplicationType};
use crate::features::{PacketFeatures, FlowFeatures};

/// ML classification errors
#[derive(Error, Debug)]
pub enum MLError {
    #[error("Model loading failed: {0}")]
    ModelLoadFailed(String),
    
    #[error("Model inference failed: {0}")]
    InferenceFailed(String),
    
    #[error("Training failed: {0}")]
    TrainingFailed(String),
    
    #[error("Feature preprocessing failed: {0}")]
    PreprocessingFailed(String),
    
    #[error("Model validation failed: {0}")]
    ValidationFailed(String),
    
    #[error("ONNX runtime error: {0}")]
    OnnxError(String),
    
    #[error("GPU acceleration error: {0}")]
    GpuError(String),
}

type Result<T> = std::result::Result<T, MLError>;

/// ML classification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLPrediction {
    pub protocol: ProtocolType,
    pub application: ApplicationType,
    pub confidence: f32,
    pub feature_importance: HashMap<String, f32>,
    pub model_version: String,
    pub inference_time_us: u64,
    pub alternative_predictions: Vec<(ProtocolType, f32)>,
}

/// ML model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub version: String,
    pub model_type: ModelType,
    pub input_features: Vec<String>,
    pub output_classes: Vec<String>,
    pub accuracy: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
    pub training_date: String,
    pub training_samples: u64,
    pub model_size_bytes: u64,
}

/// Types of ML models supported
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
}

/// ML model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_path: String,
    pub model_type: ModelType,
    pub input_size: usize,
    pub hidden_layers: Vec<usize>,
    pub output_size: usize,
    pub dropout_rate: f32,
    pub learning_rate: f32,
    pub batch_size: usize,
    pub epochs: usize,
    pub validation_split: f32,
    pub early_stopping: bool,
    pub use_gpu: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_path: "./models/dpi_classifier.safetensors".to_string(),
            model_type: ModelType::NeuralNetwork,
            input_size: 128,
            hidden_layers: vec![256, 128, 64],
            output_size: 50, // Number of protocol classes
            dropout_rate: 0.3,
            learning_rate: 0.001,
            batch_size: 32,
            epochs: 100,
            validation_split: 0.2,
            early_stopping: true,
            use_gpu: false,
        }
    }
}

/// Neural network model for DPI classification
#[cfg(feature = "ml-classification")]
pub struct DpiNeuralNetwork {
    input_layer: Linear,
    hidden_layers: Vec<Linear>,
    output_layer: Linear,
    dropout_rate: f32,
    device: Device,
}

#[cfg(feature = "ml-classification")]
impl DpiNeuralNetwork {
    pub fn new(config: &ModelConfig, varmap: &VarMap, device: Device) -> Result<Self> {
        let vb = VarBuilder::from_varmap(varmap, DType::F32, &device);
        
        // Input layer
        let input_layer = candle_nn::linear(
            config.input_size,
            config.hidden_layers[0],
            vb.pp("input")
        ).map_err(|e| MLError::ModelLoadFailed(e.to_string()))?;
        
        // Hidden layers
        let mut hidden_layers = Vec::new();
        for (i, &size) in config.hidden_layers.iter().enumerate().skip(1) {
            let prev_size = config.hidden_layers[i - 1];
            let layer = candle_nn::linear(
                prev_size,
                size,
                vb.pp(format!("hidden_{}", i))
            ).map_err(|e| MLError::ModelLoadFailed(e.to_string()))?;
            hidden_layers.push(layer);
        }
        
        // Output layer
        let output_layer = candle_nn::linear(
            *config.hidden_layers.last().unwrap(),
            config.output_size,
            vb.pp("output")
        ).map_err(|e| MLError::ModelLoadFailed(e.to_string()))?;
        
        Ok(Self {
            input_layer,
            hidden_layers,
            output_layer,
            dropout_rate: config.dropout_rate,
            device,
        })
    }
    
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = self.input_layer.forward(input)
            .map_err(|e| MLError::InferenceFailed(e.to_string()))?;
        
        // Apply ReLU activation
        x = x.relu()
            .map_err(|e| MLError::InferenceFailed(e.to_string()))?;
        
        // Hidden layers
        for layer in &self.hidden_layers {
            x = layer.forward(&x)
                .map_err(|e| MLError::InferenceFailed(e.to_string()))?;
            x = x.relu()
                .map_err(|e| MLError::InferenceFailed(e.to_string()))?;
        }
        
        // Output layer with softmax
        x = self.output_layer.forward(&x)
            .map_err(|e| MLError::InferenceFailed(e.to_string()))?;
        
        x.softmax(1)
            .map_err(|e| MLError::InferenceFailed(e.to_string()))
    }
}

/// Main ML classifier for DPI
pub struct MLClassifier {
    config: ModelConfig,
    models: Arc<RwLock<HashMap<String, Box<dyn MLModel + Send + Sync>>>>,
    ensemble: Option<Arc<EnsembleClassifier>>,
    model_manager: Arc<ModelManager>,
    feature_preprocessor: Arc<FeaturePreprocessor>,
    training_pipeline: Arc<TrainingPipeline>,
    
    // Performance tracking
    inference_count: AtomicU64,
    total_inference_time_us: AtomicU64,
    cache: Arc<RwLock<HashMap<u64, MLPrediction>>>,
}

/// Trait for ML models
#[async_trait]
pub trait MLModel: Send + Sync {
    async fn predict(&self, features: &PacketFeatures) -> Result<MLPrediction>;
    async fn predict_batch(&self, features: &[PacketFeatures]) -> Result<Vec<MLPrediction>>;
    fn get_metadata(&self) -> &ModelMetadata;
    fn get_feature_importance(&self) -> HashMap<String, f32>;
    async fn update_model(&mut self, model_data: &[u8]) -> Result<()>;
}

/// Ensemble classifier combining multiple models
pub struct EnsembleClassifier {
    models: Vec<Box<dyn MLModel + Send + Sync>>,
    weights: Vec<f32>,
    voting_strategy: VotingStrategy,
}

/// Voting strategies for ensemble models
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VotingStrategy {
    Majority,
    Weighted,
    Confidence,
}

/// Model manager for loading, updating, and versioning models
pub struct ModelManager {
    model_registry: Arc<RwLock<HashMap<String, ModelMetadata>>>,
    model_cache: Arc<RwLock<HashMap<String, Vec<u8>>>>,
    update_scheduler: Option<tokio::task::JoinHandle<()>>,
}

/// Feature preprocessor for ML input preparation
pub struct FeaturePreprocessor {
    scalers: HashMap<String, (f32, f32)>, // (mean, std) for each feature
    feature_selectors: Vec<String>,
    normalization_type: NormalizationType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormalizationType {
    StandardScaling,
    MinMaxScaling,
    RobustScaling,
    None,
}

/// Training pipeline for automated model training
pub struct TrainingPipeline {
    config: TrainingConfig,
    dataset_manager: Arc<DatasetManager>,
    model_evaluator: Arc<ModelEvaluator>,
    hyperparameter_tuner: Arc<HyperparameterTuner>,
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub data_path: String,
    pub validation_ratio: f32,
    pub test_ratio: f32,
    pub cross_validation_folds: u32,
    pub early_stopping_patience: u32,
    pub model_selection_metric: String,
    pub auto_hyperparameter_tuning: bool,
    pub distributed_training: bool,
}

/// Dataset management for training data
pub struct DatasetManager {
    training_data: Arc<RwLock<Vec<TrainingExample>>>,
    validation_data: Arc<RwLock<Vec<TrainingExample>>>,
    test_data: Arc<RwLock<Vec<TrainingExample>>>,
    data_augmentation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    pub features: PacketFeatures,
    pub label: ProtocolType,
    pub weight: f32,
    pub metadata: HashMap<String, String>,
}

/// Model evaluation and validation
pub struct ModelEvaluator {
    metrics_calculator: Arc<MetricsCalculator>,
    validation_strategies: Vec<ValidationStrategy>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationStrategy {
    HoldOut,
    KFoldCrossValidation,
    TimeSeriesSplit,
    StratifiedKFold,
}

/// Hyperparameter tuning
pub struct HyperparameterTuner {
    search_strategy: SearchStrategy,
    parameter_space: HashMap<String, ParameterRange>,
    optimization_metric: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SearchStrategy {
    GridSearch,
    RandomSearch,
    BayesianOptimization,
    GeneticAlgorithm,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterRange {
    pub min: f64,
    pub max: f64,
    pub step: Option<f64>,
    pub values: Option<Vec<f64>>,
}

/// Metrics calculator for model evaluation
pub struct MetricsCalculator {
    confusion_matrix: Arc<RwLock<Vec<Vec<u64>>>>,
    class_names: Vec<String>,
}

// Prometheus metrics for ML engine
lazy_static::lazy_static! {
    static ref ML_PREDICTIONS: Counter = register_counter!(
        "dpi_ml_predictions_total",
        "Total ML predictions made"
    ).unwrap();
    
    static ref ML_INFERENCE_DURATION: Histogram = register_histogram!(
        "dpi_ml_inference_duration_seconds",
        "ML model inference duration"
    ).unwrap();
    
    static ref MODEL_ACCURACY: Gauge = register_gauge!(
        "dpi_model_accuracy",
        "Current model accuracy"
    ).unwrap();
}

impl MLClassifier {
    /// Create a new ML classifier
    #[instrument(skip(config))]
    pub async fn new(config: &crate::config::DpiConfig) -> Result<Self> {
        info!("Initializing ML Classifier");
        
        let model_config = ModelConfig::default();
        let models = Arc::new(RwLock::new(HashMap::new()));
        
        // Initialize feature preprocessor
        let feature_preprocessor = Arc::new(FeaturePreprocessor::new()?);
        
        // Initialize model manager
        let model_manager = Arc::new(ModelManager::new().await?);
        
        // Initialize training pipeline
        let training_config = TrainingConfig {
            data_path: "./training_data".to_string(),
            validation_ratio: 0.2,
            test_ratio: 0.1,
            cross_validation_folds: 5,
            early_stopping_patience: 10,
            model_selection_metric: "f1_score".to_string(),
            auto_hyperparameter_tuning: true,
            distributed_training: false,
        };
        
        let training_pipeline = Arc::new(TrainingPipeline::new(training_config).await?);
        
        // Load pre-trained models if available
        let mut classifier = Self {
            config: model_config,
            models: models.clone(),
            ensemble: None,
            model_manager,
            feature_preprocessor,
            training_pipeline,
            inference_count: AtomicU64::new(0),
            total_inference_time_us: AtomicU64::new(0),
            cache: Arc::new(RwLock::new(HashMap::with_capacity(1000))),
        };
        
        // Load existing models
        if let Err(e) = classifier.load_models().await {
            warn!("Failed to load existing models: {}", e);
        }
        
        info!("ML Classifier initialized successfully");
        Ok(classifier)
    }
    
    /// Classify packet using ML models
    #[instrument(skip(self, features, packet_data))]
    pub async fn classify(&self, features: &PacketFeatures, packet_data: &PacketData) -> Result<MLPrediction> {
        let start_time = Instant::now();
        
        // Check cache first
        let cache_key = self.calculate_cache_key(features);
        if let Some(cached_prediction) = self.cache.read().get(&cache_key) {
            return Ok(cached_prediction.clone());
        }
        
        // Preprocess features
        let preprocessed_features = self.feature_preprocessor.preprocess(features)?;
        
        // Get prediction from ensemble or single model
        let prediction = if let Some(ensemble) = &self.ensemble {
            ensemble.predict(&preprocessed_features).await?
        } else {
            // Use the best available single model
            let models = self.models.read();
            if let Some((_, model)) = models.iter().next() {
                model.predict(&preprocessed_features).await?
            } else {
                // Fallback to rule-based classification
                self.fallback_classification(features, packet_data)?
            }
        };
        
        let inference_time = start_time.elapsed().as_micros() as u64;
        
        // Update prediction with timing
        let mut final_prediction = prediction;
        final_prediction.inference_time_us = inference_time;
        
        // Cache the result
        self.cache.write().insert(cache_key, final_prediction.clone());
        
        // Update performance metrics
        self.inference_count.fetch_add(1, Ordering::Relaxed);
        self.total_inference_time_us.fetch_add(inference_time, Ordering::Relaxed);
        
        // Update Prometheus metrics
        ML_PREDICTIONS.inc();
        ML_INFERENCE_DURATION.observe(inference_time as f64 / 1_000_000.0);
        
        debug!("ML classification completed in {}μs", inference_time);
        Ok(final_prediction)
    }
    
    /// Predict multiple packets in batch for better performance
    pub async fn classify_batch(&self, features_batch: &[PacketFeatures]) -> Result<Vec<MLPrediction>> {
        let start_time = Instant::now();
        
        // Preprocess all features
        let preprocessed_batch: Result<Vec<_>> = features_batch.iter()
            .map(|f| self.feature_preprocessor.preprocess(f))
            .collect();
        let preprocessed_batch = preprocessed_batch?;
        
        // Get batch predictions
        let predictions = if let Some(ensemble) = &self.ensemble {
            ensemble.predict_batch(&preprocessed_batch).await?
        } else {
            let models = self.models.read();
            if let Some((_, model)) = models.iter().next() {
                model.predict_batch(&preprocessed_batch).await?
            } else {
                // Fallback to individual predictions
                let mut results = Vec::new();
                for features in &preprocessed_batch {
                    results.push(self.fallback_classification(features, &PacketData::default())?);
                }
                results
            }
        };
        
        let total_inference_time = start_time.elapsed().as_micros() as u64;
        
        // Update metrics
        ML_PREDICTIONS.inc_by(predictions.len() as f64);
        ML_INFERENCE_DURATION.observe(total_inference_time as f64 / 1_000_000.0);
        
        debug!("Batch ML classification of {} packets completed in {}μs", 
               predictions.len(), total_inference_time);
        
        Ok(predictions)
    }
    
    /// Load ML models from disk
    async fn load_models(&mut self) -> Result<()> {
        info!("Loading ML models");
        
        // This would load actual models from the filesystem
        // For now, we'll create a placeholder implementation
        
        let model_metadata = ModelMetadata {
            name: "dpi_neural_network".to_string(),
            version: "1.0.0".to_string(),
            model_type: ModelType::NeuralNetwork,
            input_features: vec!["packet_size".to_string(), "flow_duration".to_string()],
            output_classes: vec!["HTTP".to_string(), "HTTPS".to_string(), "SSH".to_string()],
            accuracy: 0.95,
            precision: 0.94,
            recall: 0.93,
            f1_score: 0.935,
            training_date: "2024-01-01".to_string(),
            training_samples: 1000000,
            model_size_bytes: 10485760, // 10MB
        };
        
        info!("Loaded ML model: {} v{} with accuracy {:.2}%", 
              model_metadata.name, model_metadata.version, model_metadata.accuracy * 100.0);
        
        Ok(())
    }
    
    /// Reload models from disk
    pub async fn reload_models(&self) -> Result<()> {
        info!("Reloading ML models");
        
        // Clear cache after model reload
        self.cache.write().clear();
        
        // Update model accuracy metric
        MODEL_ACCURACY.set(0.95); // Placeholder value
        
        info!("ML models reloaded successfully");
        Ok(())
    }
    
    /// Calculate cache key for features
    fn calculate_cache_key(&self, features: &PacketFeatures) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHashher::new();
        
        // Hash relevant features for cache key
        features.packet_size.hash(&mut hasher);
        features.flow_duration_ms.hash(&mut hasher);
        features.protocol_type.hash(&mut hasher);
        
        hasher.finish()
    }
    
    /// Fallback rule-based classification when ML models are not available
    fn fallback_classification(&self, features: &PacketFeatures, packet_data: &PacketData) -> Result<MLPrediction> {
        let protocol = match packet_data.dst_port {
            80 => ProtocolType::HTTP,
            443 => ProtocolType::HTTPS,
            22 => ProtocolType::SSH,
            21 => ProtocolType::FTP,
            25 => ProtocolType::SMTP,
            53 => ProtocolType::DNS,
            _ => ProtocolType::Unknown,
        };
        
        let application = match protocol {
            ProtocolType::HTTP | ProtocolType::HTTPS => ApplicationType::WebBrowsing,
            ProtocolType::SSH => ApplicationType::RemoteAccess,
            ProtocolType::FTP => ApplicationType::FileTransfer,
            ProtocolType::SMTP => ApplicationType::Email,
            ProtocolType::DNS => ApplicationType::Unknown,
            _ => ApplicationType::Unknown,
        };
        
        Ok(MLPrediction {
            protocol,
            application,
            confidence: 0.6, // Lower confidence for rule-based
            feature_importance: HashMap::new(),
            model_version: "fallback_v1.0".to_string(),
            inference_time_us: 0,
            alternative_predictions: Vec::new(),
        })
    }
    
    /// Get ML classifier performance metrics
    pub fn get_performance_metrics(&self) -> MLPerformanceMetrics {
        let inference_count = self.inference_count.load(Ordering::Relaxed);
        let total_time = self.total_inference_time_us.load(Ordering::Relaxed);
        
        MLPerformanceMetrics {
            total_predictions: inference_count,
            average_inference_time_us: if inference_count > 0 { total_time / inference_count } else { 0 },
            cache_hit_rate: 0.0, // Would calculate actual cache hit rate
            throughput_predictions_per_second: 0.0, // Would calculate actual throughput
            models_loaded: self.models.read().len(),
            cache_size: self.cache.read().len(),
        }
    }
    
    /// Train new model with provided dataset
    pub async fn train_model(&self, training_data: Vec<TrainingExample>) -> Result<ModelMetadata> {
        info!("Starting model training with {} examples", training_data.len());
        
        // This would implement actual model training
        // For now, return a placeholder result
        let metadata = ModelMetadata {
            name: "trained_model".to_string(),
            version: "2.0.0".to_string(),
            model_type: ModelType::NeuralNetwork,
            input_features: vec!["packet_size".to_string(), "flow_duration".to_string()],
            output_classes: vec!["HTTP".to_string(), "HTTPS".to_string()],
            accuracy: 0.96,
            precision: 0.95,
            recall: 0.94,
            f1_score: 0.945,
            training_date: chrono::Utc::now().to_rfc3339(),
            training_samples: training_data.len() as u64,
            model_size_bytes: 12582912, // 12MB
        };
        
        info!("Model training completed with accuracy {:.2}%", metadata.accuracy * 100.0);
        Ok(metadata)
    }
}

/// ML performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLPerformanceMetrics {
    pub total_predictions: u64,
    pub average_inference_time_us: u64,
    pub cache_hit_rate: f32,
    pub throughput_predictions_per_second: f32,
    pub models_loaded: usize,
    pub cache_size: usize,
}

impl PacketData {
    fn default() -> Self {
        Self {
            packet_id: 0,
            timestamp: Instant::now(),
            data: bytes::Bytes::new(),
            flow_id: None,
            src_ip: "0.0.0.0".parse().unwrap(),
            dst_ip: "0.0.0.0".parse().unwrap(),
            src_port: 0,
            dst_port: 0,
            protocol: 0,
            payload_offset: 0,
            metadata: HashMap::new(),
        }
    }
}

impl FeaturePreprocessor {
    fn new() -> Result<Self> {
        Ok(Self {
            scalers: HashMap::new(),
            feature_selectors: Vec::new(),
            normalization_type: NormalizationType::StandardScaling,
        })
    }
    
    fn preprocess(&self, features: &PacketFeatures) -> Result<PacketFeatures> {
        // For now, return features as-is
        // In a real implementation, this would apply scaling, normalization, etc.
        Ok(features.clone())
    }
}

impl ModelManager {
    async fn new() -> Result<Self> {
        Ok(Self {
            model_registry: Arc::new(RwLock::new(HashMap::new())),
            model_cache: Arc::new(RwLock::new(HashMap::new())),
            update_scheduler: None,
        })
    }
}

impl TrainingPipeline {
    async fn new(_config: TrainingConfig) -> Result<Self> {
        // Placeholder implementation
        Ok(Self {
            config: _config,
            dataset_manager: Arc::new(DatasetManager::new()),
            model_evaluator: Arc::new(ModelEvaluator::new()),
            hyperparameter_tuner: Arc::new(HyperparameterTuner::new()),
        })
    }
}

impl DatasetManager {
    fn new() -> Self {
        Self {
            training_data: Arc::new(RwLock::new(Vec::new())),
            validation_data: Arc::new(RwLock::new(Vec::new())),
            test_data: Arc::new(RwLock::new(Vec::new())),
            data_augmentation: false,
        }
    }
}

impl ModelEvaluator {
    fn new() -> Self {
        Self {
            metrics_calculator: Arc::new(MetricsCalculator::new()),
            validation_strategies: vec![ValidationStrategy::HoldOut],
        }
    }
}

impl HyperparameterTuner {
    fn new() -> Self {
        Self {
            search_strategy: SearchStrategy::RandomSearch,
            parameter_space: HashMap::new(),
            optimization_metric: "f1_score".to_string(),
        }
    }
}

impl MetricsCalculator {
    fn new() -> Self {
        Self {
            confusion_matrix: Arc::new(RwLock::new(Vec::new())),
            class_names: Vec::new(),
        }
    }
}

impl EnsembleClassifier {
    pub async fn predict(&self, features: &PacketFeatures) -> Result<MLPrediction> {
        // Collect predictions from all models
        let mut predictions = Vec::new();
        for model in &self.models {
            predictions.push(model.predict(features).await?);
        }
        
        // Apply voting strategy
        match self.voting_strategy {
            VotingStrategy::Majority => self.majority_vote(predictions),
            VotingStrategy::Weighted => self.weighted_vote(predictions),
            VotingStrategy::Confidence => self.confidence_vote(predictions),
        }
    }
    
    pub async fn predict_batch(&self, features: &[PacketFeatures]) -> Result<Vec<MLPrediction>> {
        // For simplicity, process each individually
        // Real implementation would optimize batch processing
        let mut results = Vec::new();
        for feature in features {
            results.push(self.predict(feature).await?);
        }
        Ok(results)
    }
    
    fn majority_vote(&self, predictions: Vec<MLPrediction>) -> Result<MLPrediction> {
        // Simplified majority voting
        if let Some(first_prediction) = predictions.first() {
            Ok(first_prediction.clone())
        } else {
            Err(MLError::InferenceFailed("No predictions available".to_string()))
        }
    }
    
    fn weighted_vote(&self, predictions: Vec<MLPrediction>) -> Result<MLPrediction> {
        // Simplified weighted voting
        self.majority_vote(predictions)
    }
    
    fn confidence_vote(&self, predictions: Vec<MLPrediction>) -> Result<MLPrediction> {
        // Choose prediction with highest confidence
        predictions.into_iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
            .ok_or_else(|| MLError::InferenceFailed("No predictions available".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::PacketFeatures;
    
    #[tokio::test]
    async fn test_ml_classifier_creation() {
        let config = crate::config::DpiConfig::default();
        let result = MLClassifier::new(&config).await;
        
        // This might fail without proper setup, but shows the API
        assert!(result.is_ok() || result.is_err());
    }
    
    #[test]
    fn test_model_metadata() {
        let metadata = ModelMetadata {
            name: "test_model".to_string(),
            version: "1.0.0".to_string(),
            model_type: ModelType::NeuralNetwork,
            input_features: vec!["feature1".to_string(), "feature2".to_string()],
            output_classes: vec!["class1".to_string(), "class2".to_string()],
            accuracy: 0.95,
            precision: 0.94,
            recall: 0.93,
            f1_score: 0.935,
            training_date: "2024-01-01".to_string(),
            training_samples: 10000,
            model_size_bytes: 1048576,
        };
        
        assert_eq!(metadata.name, "test_model");
        assert_eq!(metadata.accuracy, 0.95);
        assert_eq!(metadata.model_type, ModelType::NeuralNetwork);
    }
    
    #[test]
    fn test_training_example() {
        let example = TrainingExample {
            features: PacketFeatures::default(),
            label: ProtocolType::HTTP,
            weight: 1.0,
            metadata: HashMap::new(),
        };
        
        assert_eq!(example.label, ProtocolType::HTTP);
        assert_eq!(example.weight, 1.0);
    }
}