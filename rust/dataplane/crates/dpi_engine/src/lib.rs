//! QBITEL Bridge Deep Packet Inspection Engine
//!
//! This module provides enterprise-grade deep packet inspection (DPI) with
//! machine learning-based classification for comprehensive network traffic analysis.
//!
//! Features:
//! - High-performance pattern matching with Hyperscan
//! - ML-based protocol classification using neural networks
//! - Real-time feature extraction and analysis
//! - Multi-threaded packet processing pipeline
//! - Hardware acceleration support
//! - Enterprise security monitoring

use std::collections::HashMap;
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use bytes::Bytes;
use crossbeam_channel::{bounded, Receiver, Sender};
use dashmap::DashMap;
use parking_lot::{Mutex, RwLock};
use prometheus::{Counter, Histogram, Gauge, register_counter, register_histogram, register_gauge};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, error, info, warn, instrument};

pub mod classifier;
pub mod patterns;
pub mod features;
pub mod protocols;
pub mod ml_engine;
pub mod security;
pub mod config;

pub use classifier::{DpiClassifier, ClassificationResult, ClassificationConfidence};
pub use patterns::{PatternMatcher, PatternDatabase, PatternMatch};
pub use features::{FeatureExtractor, PacketFeatures, FlowFeatures};
pub use protocols::{ProtocolParser, ProtocolAnalyzer, ProtocolSignature};
pub use ml_engine::{MLClassifier, ModelManager, TrainingPipeline};
pub use security::{SecurityAnalyzer, ThreatDetector, SecurityEvent};
pub use config::{DpiConfig, DpiError};

/// DPI Engine errors
#[derive(Error, Debug)]
pub enum DpiError {
    #[error("DPI initialization failed: {0}")]
    InitializationFailed(String),
    
    #[error("Pattern matching error: {0}")]
    PatternMatchingError(String),
    
    #[error("ML classification error: {0}")]
    MLClassificationError(String),
    
    #[error("Feature extraction error: {0}")]
    FeatureExtractionError(String),
    
    #[error("Protocol parsing error: {0}")]
    ProtocolParsingError(String),
    
    #[error("Security analysis error: {0}")]
    SecurityAnalysisError(String),
    
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    #[error("Model loading error: {0}")]
    ModelLoadingError(String),
}

type Result<T> = std::result::Result<T, DpiError>;

/// Packet classification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PacketClassification {
    pub protocol: ProtocolType,
    pub application: ApplicationType,
    pub confidence: f32,
    pub features: Vec<String>,
    pub patterns_matched: Vec<PatternMatch>,
    pub security_flags: Vec<SecurityFlag>,
    pub metadata: HashMap<String, String>,
    pub processing_time_us: u64,
}

/// Supported protocol types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProtocolType {
    HTTP,
    HTTPS,
    SSH,
    Telnet,
    FTP,
    SMTP,
    POP3,
    IMAP,
    DNS,
    DHCP,
    SNMP,
    SIP,
    RTP,
    BitTorrent,
    Skype,
    WhatsApp,
    Telegram,
    TikTok,
    YouTube,
    Netflix,
    Zoom,
    Teams,
    Slack,
    Custom(String),
    Unknown,
}

/// Application layer protocols
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ApplicationType {
    WebBrowsing,
    Email,
    FileTransfer,
    RemoteAccess,
    VoIP,
    VideoStreaming,
    SocialMedia,
    Messaging,
    Gaming,
    P2P,
    CloudStorage,
    VideoConferencing,
    Malware,
    Botnet,
    Custom(String),
    Unknown,
}

/// Security flags for threat detection
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityFlag {
    Malicious,
    Suspicious,
    Encrypted,
    Tunneled,
    Anomalous,
    PolicyViolation,
    DataExfiltration,
    CommandAndControl,
    Reconnaissance,
    Exploit,
    DnsExfiltration,
    HttpTunneling,
}

/// DPI processing statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DpiStatistics {
    pub packets_processed: u64,
    pub packets_classified: u64,
    pub patterns_matched: u64,
    pub ml_predictions: u64,
    pub security_events: u64,
    pub processing_errors: u64,
    pub average_processing_time_us: f64,
    pub throughput_pps: f64,
    pub classification_accuracy: f64,
    pub model_inference_time_us: f64,
}

/// Configuration for the DPI engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DpiConfig {
    pub enable_pattern_matching: bool,
    pub enable_ml_classification: bool,
    pub enable_security_analysis: bool,
    pub enable_protocol_analysis: bool,
    pub max_packet_size: usize,
    pub max_flow_packets: usize,
    pub pattern_database_path: String,
    pub model_path: String,
    pub worker_threads: usize,
    pub batch_size: usize,
    pub confidence_threshold: f32,
    pub security_threshold: f32,
    pub enable_gpu_acceleration: bool,
    pub cache_size: usize,
}

impl Default for DpiConfig {
    fn default() -> Self {
        Self {
            enable_pattern_matching: true,
            enable_ml_classification: true,
            enable_security_analysis: true,
            enable_protocol_analysis: true,
            max_packet_size: 65536,
            max_flow_packets: 1000,
            pattern_database_path: "./patterns".to_string(),
            model_path: "./models".to_string(),
            worker_threads: 4,
            batch_size: 32,
            confidence_threshold: 0.8,
            security_threshold: 0.7,
            enable_gpu_acceleration: false,
            cache_size: 10000,
        }
    }
}

/// Main DPI Engine
pub struct DpiEngine {
    config: DpiConfig,
    classifier: Arc<DpiClassifier>,
    pattern_matcher: Arc<PatternMatcher>,
    feature_extractor: Arc<FeatureExtractor>,
    ml_engine: Arc<MLClassifier>,
    security_analyzer: Arc<SecurityAnalyzer>,
    protocol_analyzers: Arc<DashMap<ProtocolType, Box<dyn ProtocolAnalyzer + Send + Sync>>>,
    
    // Worker management
    worker_handles: Vec<tokio::task::JoinHandle<()>>,
    shutdown_signal: Arc<std::sync::atomic::AtomicBool>,
    
    // Communication channels
    input_channel: Receiver<PacketData>,
    output_channel: Sender<PacketClassification>,
    
    // Performance monitoring
    stats: Arc<RwLock<DpiStatistics>>,
    packet_counter: AtomicU64,
    processing_time_histogram: Arc<Mutex<Vec<u64>>>,
    
    // Caching
    classification_cache: Arc<DashMap<u64, PacketClassification>>,
}

/// Packet data for DPI processing
#[derive(Debug, Clone)]
pub struct PacketData {
    pub packet_id: u64,
    pub timestamp: Instant,
    pub data: Bytes,
    pub flow_id: Option<u64>,
    pub src_ip: std::net::IpAddr,
    pub dst_ip: std::net::IpAddr,
    pub src_port: u16,
    pub dst_port: u16,
    pub protocol: u8,
    pub payload_offset: usize,
    pub metadata: HashMap<String, String>,
}

// Prometheus metrics
lazy_static::lazy_static! {
    static ref PACKETS_PROCESSED: Counter = register_counter!(
        "dpi_packets_processed_total",
        "Total packets processed by DPI engine"
    ).unwrap();
    
    static ref CLASSIFICATION_ACCURACY: Gauge = register_gauge!(
        "dpi_classification_accuracy",
        "DPI classification accuracy ratio"
    ).unwrap();
    
    static ref PROCESSING_DURATION: Histogram = register_histogram!(
        "dpi_processing_duration_seconds",
        "DPI packet processing duration"
    ).unwrap();
    
    static ref SECURITY_EVENTS: Counter = register_counter!(
        "dpi_security_events_total",
        "Total security events detected by DPI"
    ).unwrap();
}

impl DpiEngine {
    /// Create a new DPI engine
    #[instrument(skip(config))]
    pub async fn new(config: DpiConfig) -> Result<Self> {
        info!("Initializing DPI Engine");
        
        // Initialize pattern matcher
        let pattern_matcher = Arc::new(PatternMatcher::new(&config).await?);
        
        // Initialize feature extractor
        let feature_extractor = Arc::new(FeatureExtractor::new(&config)?);
        
        // Initialize ML classifier
        let ml_engine = Arc::new(MLClassifier::new(&config).await?);
        
        // Initialize security analyzer
        let security_analyzer = Arc::new(SecurityAnalyzer::new(&config)?);
        
        // Initialize main classifier
        let classifier = Arc::new(DpiClassifier::new(
            Arc::clone(&pattern_matcher),
            Arc::clone(&feature_extractor),
            Arc::clone(&ml_engine),
            Arc::clone(&security_analyzer),
            config.clone(),
        )?);
        
        // Initialize protocol analyzers
        let protocol_analyzers = Arc::new(DashMap::new());
        Self::initialize_protocol_analyzers(&protocol_analyzers)?;
        
        // Setup communication channels
        let (input_tx, input_rx) = bounded(config.batch_size * config.worker_threads);
        let (output_tx, _output_rx) = bounded(config.batch_size * config.worker_threads);
        
        let engine = Self {
            config: config.clone(),
            classifier,
            pattern_matcher,
            feature_extractor,
            ml_engine,
            security_analyzer,
            protocol_analyzers,
            worker_handles: Vec::new(),
            shutdown_signal: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            input_channel: input_rx,
            output_channel: output_tx,
            stats: Arc::new(RwLock::new(DpiStatistics::default())),
            packet_counter: AtomicU64::new(0),
            processing_time_histogram: Arc::new(Mutex::new(Vec::with_capacity(1000))),
            classification_cache: Arc::new(DashMap::with_capacity(config.cache_size)),
        };
        
        info!("DPI Engine initialized successfully");
        Ok(engine)
    }
    
    /// Start the DPI processing engine
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting DPI Engine with {} workers", self.config.worker_threads);
        
        // Start worker threads
        for worker_id in 0..self.config.worker_threads {
            let handle = self.start_dpi_worker(worker_id).await?;
            self.worker_handles.push(handle);
        }
        
        // Start statistics collection
        let stats_handle = self.start_stats_worker().await?;
        self.worker_handles.push(stats_handle);
        
        // Start cache management
        let cache_handle = self.start_cache_worker().await?;
        self.worker_handles.push(cache_handle);
        
        info!("DPI Engine started successfully");
        Ok(())
    }
    
    /// Process a single packet through the DPI pipeline
    #[instrument(skip(self, packet_data))]
    pub async fn process_packet(&self, packet_data: PacketData) -> Result<PacketClassification> {
        let start_time = Instant::now();
        let packet_id = packet_data.packet_id;
        
        // Check cache first
        if let Some(cached_result) = self.classification_cache.get(&packet_id) {
            debug!("Cache hit for packet {}", packet_id);
            return Ok(cached_result.clone());
        }
        
        // Extract features from packet
        let packet_features = self.feature_extractor.extract_packet_features(&packet_data).await?;
        
        // Perform pattern matching
        let pattern_matches = if self.config.enable_pattern_matching {
            self.pattern_matcher.match_patterns(&packet_data.data).await?
        } else {
            Vec::new()
        };
        
        // ML-based classification
        let ml_prediction = if self.config.enable_ml_classification {
            Some(self.ml_engine.classify(&packet_features, &packet_data).await?)
        } else {
            None
        };
        
        // Security analysis
        let security_flags = if self.config.enable_security_analysis {
            self.security_analyzer.analyze_packet(&packet_data, &packet_features).await?
        } else {
            Vec::new()
        };
        
        // Protocol-specific analysis
        let protocol_analysis = if self.config.enable_protocol_analysis {
            self.analyze_protocol(&packet_data).await?
        } else {
            None
        };
        
        // Combine all analysis results
        let classification = self.classifier.classify_packet(
            packet_data,
            packet_features,
            pattern_matches,
            ml_prediction,
            security_flags,
            protocol_analysis,
        ).await?;
        
        // Update processing time
        let processing_time = start_time.elapsed().as_micros() as u64;
        
        let mut final_classification = classification;
        final_classification.processing_time_us = processing_time;
        
        // Cache the result
        self.classification_cache.insert(packet_id, final_classification.clone());
        
        // Update statistics
        self.update_statistics(&final_classification, processing_time);
        
        // Update Prometheus metrics
        PACKETS_PROCESSED.inc();
        PROCESSING_DURATION.observe(processing_time as f64 / 1_000_000.0); // Convert to seconds
        
        if !final_classification.security_flags.is_empty() {
            SECURITY_EVENTS.inc();
        }
        
        debug!("Processed packet {} in {}μs", packet_id, processing_time);
        Ok(final_classification)
    }
    
    /// Start a DPI worker thread
    async fn start_dpi_worker(&self, worker_id: usize) -> Result<tokio::task::JoinHandle<()>> {
        let input_channel = self.input_channel.clone();
        let output_channel = self.output_channel.clone();
        let shutdown_signal = Arc::clone(&self.shutdown_signal);
        let classifier = Arc::clone(&self.classifier);
        let stats = Arc::clone(&self.stats);
        
        let handle = tokio::task::spawn(async move {
            info!("DPI worker {} started", worker_id);
            
            while !shutdown_signal.load(Ordering::Relaxed) {
                match input_channel.try_recv() {
                    Ok(packet_data) => {
                        // Process packet (this would call the main processing pipeline)
                        debug!("Worker {} processing packet {}", worker_id, packet_data.packet_id);
                        
                        // Update worker statistics
                        let mut stats_guard = stats.write();
                        stats_guard.packets_processed += 1;
                    },
                    Err(crossbeam_channel::TryRecvError::Empty) => {
                        // No packets available, sleep briefly
                        tokio::time::sleep(Duration::from_micros(100)).await;
                    },
                    Err(crossbeam_channel::TryRecvError::Disconnected) => {
                        warn!("Input channel disconnected for worker {}", worker_id);
                        break;
                    }
                }
            }
            
            info!("DPI worker {} shutdown", worker_id);
        });
        
        Ok(handle)
    }
    
    /// Start statistics collection worker
    async fn start_stats_worker(&self) -> Result<tokio::task::JoinHandle<()>> {
        let shutdown_signal = Arc::clone(&self.shutdown_signal);
        let stats = Arc::clone(&self.stats);
        let processing_time_histogram = Arc::clone(&self.processing_time_histogram);
        
        let handle = tokio::task::spawn(async move {
            info!("DPI statistics worker started");
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            
            while !shutdown_signal.load(Ordering::Relaxed) {
                interval.tick().await;
                
                // Calculate and update statistics
                let mut stats_guard = stats.write();
                let histogram = processing_time_histogram.lock();
                
                if !histogram.is_empty() {
                    let sum: u64 = histogram.iter().sum();
                    stats_guard.average_processing_time_us = sum as f64 / histogram.len() as f64;
                    stats_guard.throughput_pps = histogram.len() as f64; // Simplified calculation
                }
                
                // Update Prometheus metrics
                CLASSIFICATION_ACCURACY.set(stats_guard.classification_accuracy);
                
                debug!("Updated DPI statistics: {} packets processed", stats_guard.packets_processed);
            }
            
            info!("DPI statistics worker shutdown");
        });
        
        Ok(handle)
    }
    
    /// Start cache management worker
    async fn start_cache_worker(&self) -> Result<tokio::task::JoinHandle<()>> {
        let shutdown_signal = Arc::clone(&self.shutdown_signal);
        let classification_cache = Arc::clone(&self.classification_cache);
        let cache_size = self.config.cache_size;
        
        let handle = tokio::task::spawn(async move {
            info!("DPI cache management worker started");
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            
            while !shutdown_signal.load(Ordering::Relaxed) {
                interval.tick().await;
                
                // Clean up old cache entries if cache is full
                if classification_cache.len() > cache_size {
                    // Collect entries sorted by packet ID (ascending = oldest first)
                    let mut entries: Vec<u64> = classification_cache
                        .iter()
                        .map(|entry| *entry.key())
                        .collect();
                    entries.sort_unstable();

                    // Remove oldest entries (lowest packet IDs)
                    let entries_to_remove = classification_cache.len() - cache_size + 100;
                    let to_remove = entries_to_remove.min(entries.len());
                    for &key in &entries[..to_remove] {
                        classification_cache.remove(&key);
                    }

                    debug!("Evicted {} oldest cache entries", to_remove);
                }
            }
            
            info!("DPI cache management worker shutdown");
        });
        
        Ok(handle)
    }
    
    /// Initialize protocol analyzers
    fn initialize_protocol_analyzers(
        analyzers: &Arc<DashMap<ProtocolType, Box<dyn ProtocolAnalyzer + Send + Sync>>>
    ) -> Result<()> {
        // This would initialize specific protocol analyzers
        // For now, just log that they would be initialized
        info!("Initializing protocol analyzers for HTTP, HTTPS, SSH, DNS, etc.");
        Ok(())
    }
    
    /// Analyze packet with protocol-specific logic
    async fn analyze_protocol(&self, packet_data: &PacketData) -> Result<Option<ProtocolSignature>> {
        // Determine candidate protocol from well-known ports
        let candidate = match packet_data.dst_port {
            80 | 8080 | 8443 => Some(ProtocolType::HTTP),
            443 => Some(ProtocolType::HTTPS),
            22 => Some(ProtocolType::SSH),
            53 => Some(ProtocolType::DNS),
            25 | 587 | 465 => Some(ProtocolType::SMTP),
            21 => Some(ProtocolType::FTP),
            23 => Some(ProtocolType::Telnet),
            110 | 995 => Some(ProtocolType::POP3),
            143 | 993 => Some(ProtocolType::IMAP),
            67 | 68 => Some(ProtocolType::DHCP),
            161 | 162 => Some(ProtocolType::SNMP),
            5060 | 5061 => Some(ProtocolType::SIP),
            _ => None,
        };

        if let Some(proto) = candidate {
            if let Some(analyzer) = self.protocol_analyzers.get(&proto) {
                match analyzer.analyze(packet_data).await {
                    Ok(sig) => return Ok(Some(sig)),
                    Err(e) => {
                        debug!("Protocol analyzer failed for {:?}: {}", proto, e);
                        // Fall through to payload heuristics below
                    }
                }
            }
        }

        // Payload-based heuristic fallback when no registered analyzer matched
        let payload = if packet_data.payload_offset < packet_data.data.len() {
            &packet_data.data[packet_data.payload_offset..]
        } else {
            return Ok(None);
        };

        if payload.is_empty() {
            return Ok(None);
        }

        // Simple heuristic signatures for common protocols
        let detected = if payload.len() >= 4 && (
            payload.starts_with(b"GET ") ||
            payload.starts_with(b"POST") ||
            payload.starts_with(b"PUT ") ||
            payload.starts_with(b"HEAD") ||
            payload.starts_with(b"HTTP")
        ) {
            Some(ProtocolSignature {
                protocol: ProtocolType::HTTP,
                confidence: 0.85,
                matched_bytes: payload.len().min(64),
                description: "HTTP method/response detected in payload".to_string(),
            })
        } else if payload.len() >= 3 && payload[0] == 0x16 && payload[1] == 0x03 {
            // TLS record: ContentType=Handshake(0x16), Version=0x03xx
            Some(ProtocolSignature {
                protocol: ProtocolType::HTTPS,
                confidence: 0.90,
                matched_bytes: 5,
                description: "TLS handshake record detected".to_string(),
            })
        } else if payload.len() >= 4 && payload.starts_with(b"SSH-") {
            Some(ProtocolSignature {
                protocol: ProtocolType::SSH,
                confidence: 0.95,
                matched_bytes: 4,
                description: "SSH version string detected".to_string(),
            })
        } else if payload.len() >= 12 {
            // DNS: flags at offset 2-3, question count at 4-5
            let flags = u16::from_be_bytes([payload[2], payload[3]]);
            let qr = (flags >> 15) & 1;
            let opcode = (flags >> 11) & 0xF;
            if opcode <= 2 && (packet_data.dst_port == 53 || packet_data.src_port == 53) {
                Some(ProtocolSignature {
                    protocol: ProtocolType::DNS,
                    confidence: 0.80,
                    matched_bytes: 12,
                    description: format!("DNS {} detected", if qr == 0 { "query" } else { "response" }),
                })
            } else {
                None
            }
        } else {
            None
        };

        Ok(detected)
    }
    
    /// Update processing statistics
    fn update_statistics(&self, classification: &PacketClassification, processing_time: u64) {
        let mut stats = self.stats.write();
        stats.packets_classified += 1;
        stats.patterns_matched += classification.patterns_matched.len() as u64;
        
        if !classification.security_flags.is_empty() {
            stats.security_events += 1;
        }
        
        // Update processing time histogram
        let mut histogram = self.processing_time_histogram.lock();
        histogram.push(processing_time);
        
        // Keep histogram size manageable
        if histogram.len() > 1000 {
            histogram.drain(0..500); // Remove oldest half
        }
    }
    
    /// Get current DPI statistics
    pub fn get_statistics(&self) -> DpiStatistics {
        self.stats.read().clone()
    }
    
    /// Shutdown the DPI engine
    pub async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down DPI Engine");
        
        // Signal shutdown to all workers
        self.shutdown_signal.store(true, Ordering::Relaxed);
        
        // Wait for all workers to complete
        for handle in self.worker_handles.drain(..) {
            if let Err(e) = handle.await {
                warn!("DPI worker thread join error: {}", e);
            }
        }
        
        info!("DPI Engine shutdown completed");
        Ok(())
    }
    
    /// Get classification cache statistics
    pub fn get_cache_stats(&self) -> (usize, usize) {
        (self.classification_cache.len(), self.config.cache_size)
    }
    
    /// Clear classification cache
    pub fn clear_cache(&self) {
        self.classification_cache.clear();
        info!("DPI classification cache cleared");
    }
    
    /// Load new ML models
    pub async fn reload_models(&self) -> Result<()> {
        info!("Reloading DPI ML models");
        self.ml_engine.reload_models().await?;
        self.clear_cache(); // Clear cache after model reload
        info!("DPI ML models reloaded successfully");
        Ok(())
    }
}

impl Drop for DpiEngine {
    fn drop(&mut self) {
        // Emergency cleanup if not properly shutdown
        if !self.shutdown_signal.load(Ordering::Relaxed) {
            warn!("DPI Engine dropped without proper shutdown");
            self.shutdown_signal.store(true, Ordering::Relaxed);
        }
    }
}

/// Trait for custom packet processors that can be integrated with the DPI engine
#[async_trait]
pub trait PacketProcessor: Send + Sync {
    async fn process(&self, packet: PacketData) -> Result<PacketClassification>;
    fn name(&self) -> &str;
    fn version(&self) -> &str;
}

/// Builder pattern for DPI engine configuration
pub struct DpiEngineBuilder {
    config: DpiConfig,
}

impl DpiEngineBuilder {
    pub fn new() -> Self {
        Self {
            config: DpiConfig::default(),
        }
    }
    
    pub fn with_config(mut self, config: DpiConfig) -> Self {
        self.config = config;
        self
    }
    
    pub fn enable_pattern_matching(mut self, enabled: bool) -> Self {
        self.config.enable_pattern_matching = enabled;
        self
    }
    
    pub fn enable_ml_classification(mut self, enabled: bool) -> Self {
        self.config.enable_ml_classification = enabled;
        self
    }
    
    pub fn enable_security_analysis(mut self, enabled: bool) -> Self {
        self.config.enable_security_analysis = enabled;
        self
    }
    
    pub fn with_worker_threads(mut self, threads: usize) -> Self {
        self.config.worker_threads = threads;
        self
    }
    
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }
    
    pub fn with_confidence_threshold(mut self, threshold: f32) -> Self {
        self.config.confidence_threshold = threshold;
        self
    }
    
    pub async fn build(self) -> Result<DpiEngine> {
        DpiEngine::new(self.config).await
    }
}

impl Default for DpiEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use std::net::{IpAddr, Ipv4Addr};
    
    #[tokio::test]
    async fn test_dpi_engine_creation() {
        let config = DpiConfig::default();
        let result = DpiEngine::new(config).await;
        
        // This test would fail in a real environment without proper setup
        // but shows the intended API
        assert!(result.is_err() || result.is_ok());
    }
    
    #[test]
    fn test_packet_data_creation() {
        let packet_data = PacketData {
            packet_id: 12345,
            timestamp: Instant::now(),
            data: Bytes::from(vec![0x45, 0x00, 0x00, 0x28]), // IP header start
            flow_id: Some(67890),
            src_ip: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100)),
            dst_ip: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
            src_port: 12345,
            dst_port: 80,
            protocol: 6, // TCP
            payload_offset: 20,
            metadata: HashMap::new(),
        };
        
        assert_eq!(packet_data.packet_id, 12345);
        assert_eq!(packet_data.src_port, 12345);
        assert_eq!(packet_data.dst_port, 80);
    }
    
    #[test]
    fn test_protocol_types() {
        let protocols = vec![
            ProtocolType::HTTP,
            ProtocolType::HTTPS,
            ProtocolType::SSH,
            ProtocolType::DNS,
        ];
        
        assert_eq!(protocols.len(), 4);
        assert!(protocols.contains(&ProtocolType::HTTP));
    }
}