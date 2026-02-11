//! QBITEL Bridge Bridge - Integration between Rust dataplane and Python AI engine
//!
//! This crate provides a high-performance bridge between the Rust dataplane components
//! and the Python-based AI engine for protocol discovery and classification.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use bytes::Bytes;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use pyo3_asyncio::tokio::future_into_py;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info, warn};

use adapter_sdk::{AdapterError, L7Adapter};

/// Errors that can occur in the AI bridge
#[derive(Error, Debug)]
pub enum AiBridgeError {
    #[error("Python runtime error: {0}")]
    Python(#[from] PyErr),
    
    #[error("Adapter error: {0}")]
    Adapter(#[from] AdapterError),
    
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("Discovery timeout: {0:?}")]
    Timeout(Duration),
    
    #[error("Protocol not supported: {0}")]
    UnsupportedProtocol(String),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Configuration for the AI bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiBridgeConfig {
    /// Python module path for the AI engine
    pub ai_module_path: String,
    
    /// Maximum time to wait for AI discovery
    pub discovery_timeout: Duration,
    
    /// Confidence threshold for protocol classification
    pub confidence_threshold: f64,
    
    /// Enable caching of discovery results
    pub enable_caching: bool,
    
    /// Maximum cache size
    pub max_cache_size: usize,
    
    /// Enable training mode for continuous learning
    pub enable_training: bool,
    
    /// Batch size for processing multiple messages
    pub batch_size: usize,
}

impl Default for AiBridgeConfig {
    fn default() -> Self {
        Self {
            ai_module_path: "ai_engine.discovery.protocol_discovery_orchestrator".to_string(),
            discovery_timeout: Duration::from_secs(30),
            confidence_threshold: 0.7,
            enable_caching: true,
            max_cache_size: 10000,
            enable_training: false,
            batch_size: 10,
        }
    }
}

/// Result of protocol discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryResult {
    pub protocol_type: String,
    pub confidence: f64,
    pub processing_time: f64,
    pub metadata: HashMap<String, serde_json::Value>,
    pub validation_result: Option<ValidationResult>,
}

/// Result of message validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub confidence: f64,
    pub issues: Vec<ValidationIssue>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Validation issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    pub severity: String,
    pub code: String,
    pub message: String,
    pub position: Option<usize>,
    pub field_name: Option<String>,
}

/// Cache entry for discovery results
#[derive(Debug, Clone)]
struct CacheEntry {
    result: DiscoveryResult,
    created_at: Instant,
    access_count: u64,
}

/// Statistics for the AI bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeStatistics {
    pub total_discoveries: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub average_discovery_time: f64,
    pub protocols_discovered: HashMap<String, u64>,
    pub error_count: u64,
    pub python_call_failures: u64,
}

/// High-performance AI bridge for protocol discovery
pub struct AiBridge {
    config: AiBridgeConfig,
    python_interpreter: Arc<Mutex<Python>>,
    orchestrator: Arc<Mutex<Option<PyObject>>>,
    cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    statistics: Arc<RwLock<BridgeStatistics>>,
    initialized: Arc<RwLock<bool>>,
}

impl AiBridge {
    /// Create a new AI bridge with default configuration
    pub fn new() -> Self {
        Self::with_config(AiBridgeConfig::default())
    }
    
    /// Create a new AI bridge with custom configuration
    pub fn with_config(config: AiBridgeConfig) -> Self {
        let py = Python::acquire_gil().python();
        
        Self {
            config,
            python_interpreter: Arc::new(Mutex::new(py)),
            orchestrator: Arc::new(Mutex::new(None)),
            cache: Arc::new(RwLock::new(HashMap::new())),
            statistics: Arc::new(RwLock::new(BridgeStatistics::default())),
            initialized: Arc::new(RwLock::new(false)),
        }
    }
    
    /// Initialize the AI bridge
    pub async fn initialize(&self) -> Result<(), AiBridgeError> {
        let mut initialized = self.initialized.write().await;
        if *initialized {
            warn!("AI Bridge already initialized");
            return Ok(());
        }
        
        info!("Initializing AI Bridge");
        let start_time = Instant::now();
        
        // Initialize Python interpreter and AI components
        let py = Python::acquire_gil().python();
        let mut orchestrator_guard = self.orchestrator.lock().await;
        
        // Import the AI engine modules
        let ai_module = py.import(&self.config.ai_module_path)
            .map_err(|e| AiBridgeError::Config(format!("Failed to import AI module: {}", e)))?;
        
        // Create orchestrator instance
        let orchestrator_class = ai_module.getattr("ProtocolDiscoveryOrchestrator")?;
        
        // Create a dummy config object for now
        let config_dict = PyDict::new(py);
        config_dict.set_item("inference", PyDict::new(py))?;
        
        let orchestrator_instance = orchestrator_class.call1((config_dict,))?;
        
        // Initialize the orchestrator
        let init_future = orchestrator_instance.call_method0("initialize")?;
        pyo3_asyncio::tokio::into_future(init_future).await?;
        
        *orchestrator_guard = Some(orchestrator_instance.to_object(py));
        
        *initialized = true;
        
        let init_time = start_time.elapsed();
        info!("AI Bridge initialized successfully in {:?}", init_time);
        
        Ok(())
    }
    
    /// Discover protocol from message data
    pub async fn discover_protocol(
        &self,
        messages: &[Bytes],
        known_protocol: Option<&str>,
        enable_validation: bool,
    ) -> Result<DiscoveryResult, AiBridgeError> {
        if !*self.initialized.read().await {
            return Err(AiBridgeError::Config("AI Bridge not initialized".to_string()));
        }
        
        let start_time = Instant::now();
        let mut stats = self.statistics.write().await;
        stats.total_discoveries += 1;
        drop(stats);
        
        // Check cache first
        if self.config.enable_caching {
            let cache_key = self.generate_cache_key(messages, known_protocol);
            let cache = self.cache.read().await;
            
            if let Some(entry) = cache.get(&cache_key) {
                let mut stats = self.statistics.write().await;
                stats.cache_hits += 1;
                drop(stats);
                
                debug!("Cache hit for protocol discovery");
                return Ok(entry.result.clone());
            }
            
            drop(cache);
            let mut stats = self.statistics.write().await;
            stats.cache_misses += 1;
            drop(stats);
        }
        
        // Perform discovery using Python AI engine
        let result = self.call_python_discovery(messages, known_protocol, enable_validation).await?;
        
        // Update statistics
        let processing_time = start_time.elapsed().as_secs_f64();
        let mut stats = self.statistics.write().await;
        stats.average_discovery_time = (stats.average_discovery_time * (stats.total_discoveries - 1) as f64 + processing_time) / stats.total_discoveries as f64;
        *stats.protocols_discovered.entry(result.protocol_type.clone()).or_insert(0) += 1;
        drop(stats);
        
        // Cache the result
        if self.config.enable_caching {
            let cache_key = self.generate_cache_key(messages, known_protocol);
            let mut cache = self.cache.write().await;
            
            // Implement simple LRU eviction
            if cache.len() >= self.config.max_cache_size {
                if let Some(oldest_key) = cache.keys().next().cloned() {
                    cache.remove(&oldest_key);
                }
            }
            
            cache.insert(cache_key, CacheEntry {
                result: result.clone(),
                created_at: Instant::now(),
                access_count: 1,
            });
        }
        
        Ok(result)
    }
    
    /// Call Python discovery orchestrator
    async fn call_python_discovery(
        &self,
        messages: &[Bytes],
        known_protocol: Option<&str>,
        enable_validation: bool,
    ) -> Result<DiscoveryResult, AiBridgeError> {
        let py = Python::acquire_gil().python();
        let orchestrator_guard = self.orchestrator.lock().await;
        
        let orchestrator = orchestrator_guard.as_ref()
            .ok_or_else(|| AiBridgeError::Config("Orchestrator not initialized".to_string()))?;
        
        // Convert messages to Python bytes objects
        let py_messages = PyList::new(py, messages.iter().map(|msg| PyBytes::new(py, msg)));
        
        // Create discovery request
        let request_dict = PyDict::new(py);
        request_dict.set_item("messages", py_messages)?;
        
        if let Some(protocol) = known_protocol {
            request_dict.set_item("known_protocol", protocol)?;
        }
        
        request_dict.set_item("training_mode", false)?;
        request_dict.set_item("confidence_threshold", self.config.confidence_threshold)?;
        request_dict.set_item("generate_parser", true)?;
        request_dict.set_item("validate_results", enable_validation)?;
        
        // Create DiscoveryRequest object
        let discovery_request_class = py.import("ai_engine.discovery.protocol_discovery_orchestrator")?
            .getattr("DiscoveryRequest")?;
        let request_obj = discovery_request_class.call((), Some(request_dict))?;
        
        // Call discover_protocol method
        let discovery_future = orchestrator.call_method1(py, "discover_protocol", (request_obj,))?;
        
        // Convert to Rust future and await with timeout
        let result_obj = tokio::time::timeout(
            self.config.discovery_timeout,
            pyo3_asyncio::tokio::into_future(discovery_future),
        ).await
        .map_err(|_| AiBridgeError::Timeout(self.config.discovery_timeout))??;
        
        // Convert Python result to Rust struct
        self.convert_python_result(py, result_obj)
    }
    
    /// Convert Python discovery result to Rust struct
    fn convert_python_result(&self, py: Python, result_obj: PyObject) -> Result<DiscoveryResult, AiBridgeError> {
        let result = result_obj.as_ref(py);
        
        let protocol_type = result.getattr("protocol_type")?
            .extract::<String>()?;
        let confidence = result.getattr("confidence")?
            .extract::<f64>()?;
        let processing_time = result.getattr("processing_time")?
            .extract::<f64>()?;
        
        // Convert metadata
        let metadata_py = result.getattr("metadata")?;
        let metadata = if metadata_py.is_none() {
            HashMap::new()
        } else {
            let metadata_dict = metadata_py.downcast::<PyDict>()?;
            let mut metadata = HashMap::new();
            
            for (key, value) in metadata_dict.iter() {
                let key_str = key.extract::<String>()?;
                let value_json = serde_json::to_value(value.extract::<String>().unwrap_or_else(|_| "unknown".to_string()))?;
                metadata.insert(key_str, value_json);
            }
            metadata
        };
        
        // Convert validation result if present
        let validation_result = if let Ok(validation_obj) = result.getattr("validation_result") {
            if !validation_obj.is_none() {
                Some(self.convert_validation_result(py, validation_obj)?)
            } else {
                None
            }
        } else {
            None
        };
        
        Ok(DiscoveryResult {
            protocol_type,
            confidence,
            processing_time,
            metadata,
            validation_result,
        })
    }
    
    /// Convert Python validation result to Rust struct
    fn convert_validation_result(&self, py: Python, validation_obj: PyObject) -> Result<ValidationResult, AiBridgeError> {
        let validation = validation_obj.as_ref(py);
        
        let is_valid = validation.getattr("is_valid")?.extract::<bool>()?;
        let confidence = validation.getattr("confidence")?.extract::<f64>()?;
        
        // Convert issues
        let issues_py = validation.getattr("issues")?;
        let mut issues = Vec::new();
        
        if let Ok(issues_list) = issues_py.downcast::<PyList>() {
            for issue_obj in issues_list.iter() {
                let severity = issue_obj.getattr("severity")?
                    .getattr("value")?  // Enum value
                    .extract::<String>()?;
                let code = issue_obj.getattr("code")?.extract::<String>()?;
                let message = issue_obj.getattr("message")?.extract::<String>()?;
                
                let position = issue_obj.getattr("position")?
                    .extract::<Option<usize>>().unwrap_or(None);
                let field_name = issue_obj.getattr("field_name")?
                    .extract::<Option<String>>().unwrap_or(None);
                
                issues.push(ValidationIssue {
                    severity,
                    code,
                    message,
                    position,
                    field_name,
                });
            }
        }
        
        // Convert metadata
        let metadata_py = validation.getattr("metadata")?;
        let metadata = if let Ok(metadata_dict) = metadata_py.downcast::<PyDict>() {
            let mut metadata = HashMap::new();
            for (key, value) in metadata_dict.iter() {
                let key_str = key.extract::<String>()?;
                let value_json = serde_json::to_value(value.extract::<String>().unwrap_or_else(|_| "unknown".to_string()))?;
                metadata.insert(key_str, value_json);
            }
            metadata
        } else {
            HashMap::new()
        };
        
        Ok(ValidationResult {
            is_valid,
            confidence,
            issues,
            metadata,
        })
    }
    
    /// Generate cache key for messages and parameters
    fn generate_cache_key(&self, messages: &[Bytes], known_protocol: Option<&str>) -> String {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        
        // Hash first few messages
        for (i, msg) in messages.iter().take(3).enumerate() {
            i.hash(&mut hasher);
            msg.hash(&mut hasher);
        }
        
        // Hash parameters
        if let Some(protocol) = known_protocol {
            protocol.hash(&mut hasher);
        }
        
        format!("cache_{:x}", hasher.finish())
    }
    
    /// Get bridge statistics
    pub async fn get_statistics(&self) -> BridgeStatistics {
        self.statistics.read().await.clone()
    }
    
    /// Clear the discovery cache
    pub async fn clear_cache(&self) -> usize {
        let mut cache = self.cache.write().await;
        let size = cache.len();
        cache.clear();
        info!("Cleared {} cached discovery results", size);
        size
    }
    
    /// Train the classifier with new samples
    pub async fn train_classifier(&self, samples: Vec<(Bytes, String)>) -> Result<(), AiBridgeError> {
        if !*self.initialized.read().await {
            return Err(AiBridgeError::Config("AI Bridge not initialized".to_string()));
        }
        
        if !self.config.enable_training {
            return Err(AiBridgeError::Config("Training not enabled in configuration".to_string()));
        }
        
        let py = Python::acquire_gil().python();
        let orchestrator_guard = self.orchestrator.lock().await;
        
        let orchestrator = orchestrator_guard.as_ref()
            .ok_or_else(|| AiBridgeError::Config("Orchestrator not initialized".to_string()))?;
        
        // Convert samples to Python objects
        let py_samples = PyList::empty(py);
        
        for (data, label) in samples {
            let sample_dict = PyDict::new(py);
            sample_dict.set_item("data", PyBytes::new(py, &data))?;
            sample_dict.set_item("label", label)?;
            
            // Create ProtocolSample object
            let sample_class = py.import("ai_engine.discovery.protocol_classifier")?
                .getattr("ProtocolSample")?;
            let sample_obj = sample_class.call((), Some(sample_dict))?;
            
            py_samples.append(sample_obj)?;
        }
        
        // Call train_classifier method
        let training_future = orchestrator.call_method1(py, "train_classifier", (py_samples,))?;
        pyo3_asyncio::tokio::into_future(training_future).await?;
        
        info!("Classifier training completed with {} samples", py_samples.len());
        Ok(())
    }
    
    /// Shutdown the AI bridge
    pub async fn shutdown(&self) -> Result<(), AiBridgeError> {
        info!("Shutting down AI Bridge");
        
        let orchestrator_guard = self.orchestrator.lock().await;
        if let Some(orchestrator) = orchestrator_guard.as_ref() {
            let py = Python::acquire_gil().python();
            let shutdown_future = orchestrator.call_method0(py, "shutdown")?;
            pyo3_asyncio::tokio::into_future(shutdown_future).await?;
        }
        
        // Clear cache
        self.clear_cache().await;
        
        *self.initialized.write().await = false;
        info!("AI Bridge shutdown completed");
        Ok(())
    }
}

impl Default for BridgeStatistics {
    fn default() -> Self {
        Self {
            total_discoveries: 0,
            cache_hits: 0,
            cache_misses: 0,
            average_discovery_time: 0.0,
            protocols_discovered: HashMap::new(),
            error_count: 0,
            python_call_failures: 0,
        }
    }
}

/// AI-powered L7 adapter that uses protocol discovery
pub struct AiL7Adapter {
    bridge: Arc<AiBridge>,
    protocol_cache: Arc<RwLock<HashMap<String, String>>>, // message hash -> protocol
    config: AiBridgeConfig,
}

impl AiL7Adapter {
    /// Create a new AI-powered L7 adapter
    pub async fn new(config: AiBridgeConfig) -> Result<Self, AiBridgeError> {
        let bridge = Arc::new(AiBridge::with_config(config.clone()));
        bridge.initialize().await?;
        
        Ok(Self {
            bridge,
            protocol_cache: Arc::new(RwLock::new(HashMap::new())),
            config,
        })
    }
    
    /// Get the discovered protocol for a message
    async fn get_protocol(&self, data: &Bytes) -> Result<String, AiBridgeError> {
        // Check protocol cache first
        let message_hash = format!("{:x}", md5::compute(data));
        {
            let cache = self.protocol_cache.read().await;
            if let Some(protocol) = cache.get(&message_hash) {
                return Ok(protocol.clone());
            }
        }
        
        // Discover protocol using AI bridge
        let discovery_result = self.bridge.discover_protocol(
            &[data.clone()],
            None,
            false, // Skip validation for performance
        ).await?;
        
        // Cache the result if confidence is high enough
        if discovery_result.confidence >= self.config.confidence_threshold {
            let mut cache = self.protocol_cache.write().await;
            cache.insert(message_hash, discovery_result.protocol_type.clone());
        }
        
        Ok(discovery_result.protocol_type)
    }
}

#[async_trait]
impl L7Adapter for AiL7Adapter {
    async fn to_upstream(&self, input: Bytes) -> Result<Bytes, AdapterError> {
        debug!("AI L7 Adapter processing upstream data: {} bytes", input.len());
        
        match self.get_protocol(&input).await {
            Ok(protocol) => {
                debug!("Discovered protocol: {} for upstream data", protocol);
                
                // For now, pass through the data unchanged
                // In a full implementation, you would apply protocol-specific transformations
                match protocol.as_str() {
                    "http" => {
                        // Apply HTTP-specific processing
                        Ok(input)
                    },
                    "json" => {
                        // Apply JSON-specific processing  
                        Ok(input)
                    },
                    _ => {
                        // Default pass-through
                        Ok(input)
                    }
                }
            }
            Err(e) => {
                error!("Protocol discovery failed: {}", e);
                // Fall back to pass-through on error
                Ok(input)
            }
        }
    }
    
    async fn to_client(&self, input: Bytes) -> Result<Bytes, AdapterError> {
        debug!("AI L7 Adapter processing client data: {} bytes", input.len());
        
        match self.get_protocol(&input).await {
            Ok(protocol) => {
                debug!("Discovered protocol: {} for client data", protocol);
                
                // Apply protocol-specific transformations
                match protocol.as_str() {
                    "http" => {
                        // Apply HTTP-specific processing
                        Ok(input)
                    },
                    "json" => {
                        // Apply JSON-specific processing
                        Ok(input)
                    },
                    _ => {
                        // Default pass-through
                        Ok(input)
                    }
                }
            }
            Err(e) => {
                error!("Protocol discovery failed: {}", e);
                // Fall back to pass-through on error
                Ok(input)
            }
        }
    }
    
    fn name(&self) -> &'static str {
        "ai_l7_adapter"
    }
}

/// Initialize Python environment for the AI bridge
pub fn initialize_python_env() -> Result<(), AiBridgeError> {
    pyo3::prepare_freethreaded_python();
    
    Python::with_gil(|py| {
        // Add the AI engine path to sys.path
        let sys = py.import("sys")?;
        let path = sys.getattr("path")?;
        path.call_method1("append", ("../../ai_engine",))?;
        
        // Initialize asyncio event loop
        let asyncio = py.import("asyncio")?;
        let loop = asyncio.call_method0("new_event_loop")?;
        asyncio.call_method1("set_event_loop", (loop,))?;
        
        Ok::<(), PyErr>(())
    }).map_err(AiBridgeError::Python)?;
    
    info!("Python environment initialized for AI bridge");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;
    
    #[test]
    async fn test_ai_bridge_initialization() {
        let config = AiBridgeConfig::default();
        let bridge = AiBridge::with_config(config);
        
        // Note: This test would require a proper Python environment setup
        // In a real deployment, you'd have integration tests with the full Python stack
        assert!(!*bridge.initialized.read().await);
    }
    
    #[test]
    async fn test_cache_key_generation() {
        let bridge = AiBridge::new();
        let messages = vec![
            Bytes::from("test message 1"),
            Bytes::from("test message 2"),
        ];
        
        let key1 = bridge.generate_cache_key(&messages, None);
        let key2 = bridge.generate_cache_key(&messages, Some("http"));
        
        assert_ne!(key1, key2);
    }
    
    #[test]
    async fn test_statistics_default() {
        let stats = BridgeStatistics::default();
        assert_eq!(stats.total_discoveries, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.error_count, 0);
    }
}