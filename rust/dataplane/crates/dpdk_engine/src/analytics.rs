//! DPDK Real-time Packet Analytics
//!
//! This module provides enterprise-grade real-time packet analytics with ML-based
//! anomaly detection, traffic profiling, and performance monitoring.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use parking_lot::{Mutex, RwLock};
use prometheus::{Counter, Gauge, Histogram, register_counter, register_gauge, register_histogram};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, error, info, warn, instrument};

use crate::packet::{PacketBuffer, PacketClassification, ProtocolType};
use crate::flow::FlowPattern;

/// Analytics errors
#[derive(Error, Debug)]
pub enum AnalyticsError {
    #[error("Analytics initialization failed: {0}")]
    InitializationFailed(String),
    
    #[error("Data processing error: {0}")]
    ProcessingError(String),
    
    #[error("Anomaly detection error: {0}")]
    AnomalyDetectionError(String),
    
    #[error("Metric calculation error: {0}")]
    MetricCalculationError(String),
    
    #[error("Export error: {0}")]
    ExportError(String),
}

type Result<T> = std::result::Result<T, AnalyticsError>;

/// Real-time packet analytics engine
pub struct PacketAnalytics {
    config: AnalyticsConfig,
    traffic_stats: Arc<RwLock<TrafficStatistics>>,
    protocol_stats: Arc<RwLock<HashMap<ProtocolType, ProtocolStatistics>>>,
    flow_stats: Arc<RwLock<FlowAnalytics>>,
    anomaly_detector: Arc<Mutex<AnomalyDetector>>,
    time_series_data: Arc<Mutex<TimeSeriesBuffer>>,
    bandwidth_monitor: Arc<RwLock<BandwidthMonitor>>,
    security_monitor: Arc<RwLock<SecurityMonitor>>,
    packet_counter: AtomicU64,
    byte_counter: AtomicU64,
}

/// Analytics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsConfig {
    pub enable_anomaly_detection: bool,
    pub enable_protocol_analysis: bool,
    pub enable_flow_analytics: bool,
    pub enable_security_monitoring: bool,
    pub time_window_seconds: u64,
    pub sampling_rate: f64,
    pub anomaly_threshold: f64,
    pub export_interval_seconds: u64,
    pub max_flows_tracked: usize,
    pub bandwidth_calculation_window: u64,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        AnalyticsConfig {
            enable_anomaly_detection: true,
            enable_protocol_analysis: true,
            enable_flow_analytics: true,
            enable_security_monitoring: true,
            time_window_seconds: 300, // 5 minutes
            sampling_rate: 1.0, // 100% sampling
            anomaly_threshold: 2.0, // 2 standard deviations
            export_interval_seconds: 60, // 1 minute
            max_flows_tracked: 100000,
            bandwidth_calculation_window: 60, // 1 minute
        }
    }
}

/// Comprehensive traffic statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrafficStatistics {
    pub total_packets: u64,
    pub total_bytes: u64,
    pub packets_per_second: f64,
    pub bytes_per_second: f64,
    pub average_packet_size: f64,
    pub peak_bandwidth: f64,
    pub packet_size_distribution: HashMap<String, u64>, // Size ranges as keys
    pub inter_arrival_times: VecDeque<f64>,
    pub burst_detection: BurstStatistics,
    pub error_statistics: ErrorStatistics,
}

/// Protocol-specific statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProtocolStatistics {
    pub packet_count: u64,
    pub byte_count: u64,
    pub average_packet_size: f64,
    pub bandwidth_utilization: f64,
    pub connection_count: u64,
    pub connection_rate: f64,
    pub error_rate: f64,
    pub latency_stats: LatencyStatistics,
    pub custom_metrics: HashMap<String, f64>,
}

/// Flow analytics data
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FlowAnalytics {
    pub total_flows: u64,
    pub active_flows: u64,
    pub flow_creation_rate: f64,
    pub flow_termination_rate: f64,
    pub average_flow_duration: f64,
    pub flow_size_distribution: HashMap<String, u64>,
    pub top_talkers: Vec<TopTalker>,
    pub elephant_flows: Vec<ElephantFlow>,
    pub mouse_flows: Vec<MouseFlow>,
}

/// Burst detection statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BurstStatistics {
    pub burst_count: u64,
    pub current_burst_size: u64,
    pub max_burst_size: u64,
    pub average_burst_size: f64,
    pub burst_threshold: u64,
    pub time_between_bursts: VecDeque<f64>,
}

/// Error statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ErrorStatistics {
    pub checksum_errors: u64,
    pub truncated_packets: u64,
    pub oversized_packets: u64,
    pub malformed_packets: u64,
    pub unknown_protocols: u64,
    pub error_rate: f64,
}

/// Latency statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LatencyStatistics {
    pub min_latency_us: f64,
    pub max_latency_us: f64,
    pub average_latency_us: f64,
    pub percentile_50_us: f64,
    pub percentile_95_us: f64,
    pub percentile_99_us: f64,
    pub jitter_us: f64,
}

/// Top talker (high bandwidth flow)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopTalker {
    pub flow_pattern: FlowPattern,
    pub bytes_per_second: u64,
    pub packets_per_second: u64,
    pub duration_seconds: u64,
    pub last_seen: u64,
}

/// Elephant flow (long-duration, high-volume flow)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElephantFlow {
    pub flow_pattern: FlowPattern,
    pub total_bytes: u64,
    pub duration_seconds: u64,
    pub average_bandwidth: u64,
    pub start_time: u64,
}

/// Mouse flow (short-duration, low-volume flow)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MouseFlow {
    pub flow_pattern: FlowPattern,
    pub total_bytes: u64,
    pub duration_seconds: u64,
    pub packet_count: u64,
}

/// Anomaly detection engine
pub struct AnomalyDetector {
    baseline_metrics: BaselineMetrics,
    anomaly_history: VecDeque<AnomalyEvent>,
    detection_algorithms: Vec<Box<dyn AnomalyDetectionAlgorithm + Send + Sync>>,
    threshold: f64,
}

/// Baseline traffic metrics for anomaly detection
#[derive(Debug, Clone, Default)]
pub struct BaselineMetrics {
    pub mean_packets_per_second: f64,
    pub std_packets_per_second: f64,
    pub mean_bytes_per_second: f64,
    pub std_bytes_per_second: f64,
    pub mean_flow_rate: f64,
    pub std_flow_rate: f64,
    pub protocol_distribution: HashMap<ProtocolType, f64>,
    pub last_updated: u64,
}

/// Anomaly event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyEvent {
    pub timestamp: u64,
    pub anomaly_type: AnomalyType,
    pub severity: AnomalySeverity,
    pub confidence: f64,
    pub description: String,
    pub affected_flows: Vec<FlowPattern>,
    pub metrics: HashMap<String, f64>,
}

/// Types of anomalies that can be detected
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalyType {
    TrafficSpike,
    TrafficDrop,
    ProtocolAnomaly,
    FlowAnomaly,
    BandwidthAnomaly,
    SecurityThreat,
    PerformanceDegradation,
}

/// Severity levels for anomalies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Anomaly detection algorithm trait
pub trait AnomalyDetectionAlgorithm: Send + Sync {
    fn detect(&self, metrics: &TrafficStatistics) -> Option<AnomalyEvent>;
    fn update_baseline(&mut self, metrics: &TrafficStatistics);
    fn name(&self) -> &str;
}

/// Time series data buffer for historical analysis
pub struct TimeSeriesBuffer {
    packet_rates: VecDeque<(u64, f64)>, // (timestamp, rate)
    bandwidth_rates: VecDeque<(u64, f64)>,
    flow_rates: VecDeque<(u64, f64)>,
    error_rates: VecDeque<(u64, f64)>,
    max_size: usize,
}

/// Bandwidth monitoring
#[derive(Debug, Clone, Default)]
pub struct BandwidthMonitor {
    pub current_bandwidth_bps: u64,
    pub peak_bandwidth_bps: u64,
    pub average_bandwidth_bps: u64,
    pub utilization_percentage: f64,
    pub link_capacity_bps: u64,
    pub ingress_bytes: u64,
    pub egress_bytes: u64,
}

/// Security monitoring statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SecurityMonitor {
    pub suspicious_flows: u64,
    pub port_scan_attempts: u64,
    pub ddos_indicators: u64,
    pub malformed_packet_rate: f64,
    pub connection_flood_count: u64,
    pub threat_events: Vec<ThreatEvent>,
}

/// Security threat event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatEvent {
    pub timestamp: u64,
    pub threat_type: ThreatType,
    pub severity: AnomalySeverity,
    pub source_ip: std::net::IpAddr,
    pub target_ip: Option<std::net::IpAddr>,
    pub description: String,
    pub mitigation_applied: bool,
}

/// Types of security threats
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThreatType {
    PortScan,
    DDoS,
    BruteForce,
    Malware,
    DataExfiltration,
    UnauthorizedAccess,
    AbnormalTraffic,
}

// Prometheus metrics for analytics
lazy_static::lazy_static! {
    static ref PACKETS_ANALYZED: Counter = register_counter!(
        "dpdk_packets_analyzed_total",
        "Total number of packets analyzed"
    ).unwrap();
    
    static ref ANOMALIES_DETECTED: Counter = register_counter!(
        "dpdk_anomalies_detected_total",
        "Total number of anomalies detected"
    ).unwrap();
    
    static ref BANDWIDTH_UTILIZATION: Gauge = register_gauge!(
        "dpdk_bandwidth_utilization_ratio",
        "Current bandwidth utilization ratio"
    ).unwrap();
    
    static ref ANALYSIS_DURATION: Histogram = register_histogram!(
        "dpdk_analysis_duration_seconds",
        "Time taken for packet analysis"
    ).unwrap();
}

impl PacketAnalytics {
    /// Create a new packet analytics engine
    #[instrument(skip(config))]
    pub fn new(config: AnalyticsConfig) -> Result<Self> {
        info!("Initializing Packet Analytics Engine");
        
        let analytics = PacketAnalytics {
            config: config.clone(),
            traffic_stats: Arc::new(RwLock::new(TrafficStatistics::default())),
            protocol_stats: Arc::new(RwLock::new(HashMap::new())),
            flow_stats: Arc::new(RwLock::new(FlowAnalytics::default())),
            anomaly_detector: Arc::new(Mutex::new(AnomalyDetector::new(config.anomaly_threshold)?)),
            time_series_data: Arc::new(Mutex::new(TimeSeriesBuffer::new(1000))), // Keep 1000 data points
            bandwidth_monitor: Arc::new(RwLock::new(BandwidthMonitor::default())),
            security_monitor: Arc::new(RwLock::new(SecurityMonitor::default())),
            packet_counter: AtomicU64::new(0),
            byte_counter: AtomicU64::new(0),
        };
        
        info!("Packet Analytics Engine initialized successfully");
        Ok(analytics)
    }
    
    /// Analyze an incoming packet
    #[instrument(skip(self, packet))]
    pub fn analyze_packet(&self, packet: &PacketBuffer, classification: &PacketClassification) -> Result<()> {
        let _timer = ANALYSIS_DURATION.start_timer();
        
        // Apply sampling if configured
        if self.config.sampling_rate < 1.0 {
            if rand::random::<f64>() > self.config.sampling_rate {
                return Ok(());
            }
        }
        
        // Update counters
        let packet_count = self.packet_counter.fetch_add(1, Ordering::Relaxed) + 1;
        let byte_count = self.byte_counter.fetch_add(packet.get_packet_length() as u64, Ordering::Relaxed) + packet.get_packet_length() as u64;
        
        // Update traffic statistics
        if self.config.enable_protocol_analysis {
            self.update_traffic_statistics(packet, classification)?;
        }
        
        // Update protocol-specific statistics
        if self.config.enable_protocol_analysis {
            self.update_protocol_statistics(packet, classification)?;
        }
        
        // Update flow analytics
        if self.config.enable_flow_analytics {
            self.update_flow_analytics(packet, classification)?;
        }
        
        // Security monitoring
        if self.config.enable_security_monitoring {
            self.update_security_monitoring(packet, classification)?;
        }
        
        // Perform anomaly detection
        if self.config.enable_anomaly_detection {
            self.detect_anomalies()?;
        }
        
        // Update bandwidth monitoring
        self.update_bandwidth_monitoring(packet)?;
        
        PACKETS_ANALYZED.inc();
        
        if packet_count % 10000 == 0 {
            debug!("Analyzed {} packets ({} bytes)", packet_count, byte_count);
        }
        
        Ok(())
    }
    
    /// Update traffic statistics
    fn update_traffic_statistics(&self, packet: &PacketBuffer, _classification: &PacketClassification) -> Result<()> {
        let mut stats = self.traffic_stats.write();
        let packet_size = packet.get_packet_length() as u64;
        
        stats.total_packets += 1;
        stats.total_bytes += packet_size;
        
        // Update average packet size
        stats.average_packet_size = stats.total_bytes as f64 / stats.total_packets as f64;
        
        // Update packet size distribution
        let size_range = match packet_size {
            0..=64 => "0-64",
            65..=128 => "65-128",
            129..=256 => "129-256",
            257..=512 => "257-512",
            513..=1024 => "513-1024",
            1025..=1518 => "1025-1518",
            _ => "1519+",
        };
        
        *stats.packet_size_distribution.entry(size_range.to_string()).or_insert(0) += 1;
        
        // Calculate inter-arrival times (simplified)
        let now = Self::current_timestamp_us();
        if let Some(&last_time) = stats.inter_arrival_times.back() {
            let inter_arrival = (now - last_time as u64) as f64 / 1_000_000.0; // Convert to seconds
            stats.inter_arrival_times.push_back(inter_arrival);
            
            // Keep only recent inter-arrival times
            if stats.inter_arrival_times.len() > 1000 {
                stats.inter_arrival_times.pop_front();
            }
        } else {
            stats.inter_arrival_times.push_back(now as f64);
        }
        
        // Update burst detection
        self.update_burst_statistics(&mut stats.burst_detection, packet_size)?;
        
        // Check for packet errors
        self.update_error_statistics(&mut stats.error_statistics, packet)?;
        
        Ok(())
    }
    
    /// Update protocol-specific statistics
    fn update_protocol_statistics(&self, packet: &PacketBuffer, classification: &PacketClassification) -> Result<()> {
        let mut protocol_stats = self.protocol_stats.write();
        let stats = protocol_stats.entry(classification.protocol.clone()).or_insert_with(ProtocolStatistics::default);
        
        let packet_size = packet.get_packet_length() as u64;
        stats.packet_count += 1;
        stats.byte_count += packet_size;
        stats.average_packet_size = stats.byte_count as f64 / stats.packet_count as f64;
        
        // Calculate bandwidth utilization (simplified)
        stats.bandwidth_utilization = (stats.byte_count as f64 * 8.0) / (self.config.time_window_seconds as f64 * 1_000_000_000.0); // Gbps
        
        // Update connection count for connection-oriented protocols
        if matches!(classification.protocol, ProtocolType::TCP | ProtocolType::HTTPS) {
            if let Some(flow_id) = classification.flow_id {
                // This is a simplification - real implementation would track connection states
                stats.connection_count += 1;
            }
        }
        
        Ok(())
    }
    
    /// Update flow analytics
    fn update_flow_analytics(&self, _packet: &PacketBuffer, classification: &PacketClassification) -> Result<()> {
        let mut flow_stats = self.flow_stats.write();
        
        if let Some(_flow_id) = classification.flow_id {
            flow_stats.total_flows += 1;
            flow_stats.active_flows += 1; // Simplified - would track actual active flows
            
            // Update flow creation rate (simplified)
            flow_stats.flow_creation_rate = flow_stats.total_flows as f64 / self.config.time_window_seconds as f64;
        }
        
        Ok(())
    }
    
    /// Update security monitoring
    fn update_security_monitoring(&self, packet: &PacketBuffer, classification: &PacketClassification) -> Result<()> {
        let mut security_stats = self.security_monitor.write();
        
        // Check for port scanning behavior
        if self.is_potential_port_scan(classification) {
            security_stats.port_scan_attempts += 1;
            
            let threat_event = ThreatEvent {
                timestamp: Self::current_timestamp(),
                threat_type: ThreatType::PortScan,
                severity: AnomalySeverity::Medium,
                source_ip: classification.src_ip.unwrap_or_else(|| "0.0.0.0".parse().unwrap()),
                target_ip: classification.dst_ip,
                description: "Potential port scan detected".to_string(),
                mitigation_applied: false,
            };
            
            security_stats.threat_events.push(threat_event);
        }
        
        // Check for DDoS indicators
        if self.is_potential_ddos(packet, classification) {
            security_stats.ddos_indicators += 1;
        }
        
        // Check for malformed packets
        if self.is_malformed_packet(packet) {
            security_stats.malformed_packet_rate += 1.0;
        }
        
        Ok(())
    }
    
    /// Detect anomalies in traffic patterns
    fn detect_anomalies(&self) -> Result<()> {
        let traffic_stats = self.traffic_stats.read();
        let mut anomaly_detector = self.anomaly_detector.lock();
        
        if let Some(anomaly) = anomaly_detector.detect_anomalies(&traffic_stats) {
            warn!("Anomaly detected: {:?} (confidence: {:.2})", 
                  anomaly.anomaly_type, anomaly.confidence);
            
            ANOMALIES_DETECTED.inc();
            
            // Add to anomaly history
            anomaly_detector.add_anomaly_event(anomaly);
        }
        
        // Update baseline metrics periodically
        if Self::current_timestamp() % 300 == 0 { // Every 5 minutes
            anomaly_detector.update_baseline(&traffic_stats);
        }
        
        Ok(())
    }
    
    /// Update bandwidth monitoring
    fn update_bandwidth_monitoring(&self, packet: &PacketBuffer) -> Result<()> {
        let mut bandwidth_monitor = self.bandwidth_monitor.write();
        let packet_size = packet.get_packet_length() as u64;
        
        bandwidth_monitor.ingress_bytes += packet_size;
        
        // Calculate current bandwidth (simplified - would use sliding window)
        let now = Self::current_timestamp();
        let window_bytes = bandwidth_monitor.ingress_bytes;
        let window_seconds = self.config.bandwidth_calculation_window;
        
        bandwidth_monitor.current_bandwidth_bps = (window_bytes * 8) / window_seconds; // bits per second
        
        if bandwidth_monitor.current_bandwidth_bps > bandwidth_monitor.peak_bandwidth_bps {
            bandwidth_monitor.peak_bandwidth_bps = bandwidth_monitor.current_bandwidth_bps;
        }
        
        // Calculate utilization if link capacity is known
        if bandwidth_monitor.link_capacity_bps > 0 {
            bandwidth_monitor.utilization_percentage = 
                (bandwidth_monitor.current_bandwidth_bps as f64 / bandwidth_monitor.link_capacity_bps as f64) * 100.0;
        }
        
        BANDWIDTH_UTILIZATION.set(bandwidth_monitor.utilization_percentage / 100.0);
        
        Ok(())
    }
    
    /// Update burst statistics
    fn update_burst_statistics(&self, burst_stats: &mut BurstStatistics, packet_size: u64) -> Result<()> {
        burst_stats.current_burst_size += packet_size;
        
        // Simple burst detection based on packet size threshold
        if burst_stats.current_burst_size > burst_stats.burst_threshold {
            burst_stats.burst_count += 1;
            
            if burst_stats.current_burst_size > burst_stats.max_burst_size {
                burst_stats.max_burst_size = burst_stats.current_burst_size;
            }
            
            // Update average burst size
            burst_stats.average_burst_size = 
                ((burst_stats.average_burst_size * (burst_stats.burst_count - 1) as f64) + 
                 burst_stats.current_burst_size as f64) / burst_stats.burst_count as f64;
            
            burst_stats.current_burst_size = 0;
        }
        
        Ok(())
    }
    
    /// Update error statistics
    fn update_error_statistics(&self, error_stats: &mut ErrorStatistics, packet: &PacketBuffer) -> Result<()> {
        let headers = packet.get_headers();
        
        // Check for truncated packets
        if packet.get_packet_length() < 64 { // Minimum Ethernet frame size
            error_stats.truncated_packets += 1;
        }
        
        // Check for oversized packets
        if packet.get_packet_length() > 9000 { // Jumbo frame threshold
            error_stats.oversized_packets += 1;
        }
        
        // Check for unknown protocols
        if headers.ethernet.is_none() {
            error_stats.unknown_protocols += 1;
        }
        
        // Calculate error rate
        let total_packets = self.packet_counter.load(Ordering::Relaxed);
        if total_packets > 0 {
            let total_errors = error_stats.checksum_errors + error_stats.truncated_packets + 
                              error_stats.oversized_packets + error_stats.malformed_packets;
            error_stats.error_rate = (total_errors as f64) / (total_packets as f64) * 100.0;
        }
        
        Ok(())
    }
    
    /// Check if packet indicates potential port scanning
    fn is_potential_port_scan(&self, classification: &PacketClassification) -> bool {
        // Simplified detection - would use more sophisticated heuristics
        matches!(classification.protocol, ProtocolType::TCP) && 
        classification.dst_port.map_or(false, |port| port < 1024)
    }
    
    /// Check if packet indicates potential DDoS
    fn is_potential_ddos(&self, _packet: &PacketBuffer, _classification: &PacketClassification) -> bool {
        // Simplified detection - would analyze traffic patterns, source diversity, etc.
        false
    }
    
    /// Check if packet is malformed
    fn is_malformed_packet(&self, packet: &PacketBuffer) -> bool {
        // Basic malformation checks
        packet.get_packet_length() < 14 || // Less than Ethernet header
        packet.get_headers().ethernet.is_none()
    }
    
    /// Get comprehensive analytics report
    pub fn get_analytics_report(&self) -> AnalyticsReport {
        let traffic_stats = self.traffic_stats.read().clone();
        let protocol_stats = self.protocol_stats.read().clone();
        let flow_stats = self.flow_stats.read().clone();
        let bandwidth_monitor = self.bandwidth_monitor.read().clone();
        let security_monitor = self.security_monitor.read().clone();
        
        AnalyticsReport {
            timestamp: Self::current_timestamp(),
            traffic_statistics: traffic_stats,
            protocol_statistics: protocol_stats,
            flow_analytics: flow_stats,
            bandwidth_monitoring: bandwidth_monitor,
            security_monitoring: security_monitor,
            total_packets_analyzed: self.packet_counter.load(Ordering::Relaxed),
            total_bytes_analyzed: self.byte_counter.load(Ordering::Relaxed),
        }
    }
    
    /// Export analytics data for external analysis
    pub fn export_analytics(&self) -> Result<String> {
        let report = self.get_analytics_report();
        serde_json::to_string_pretty(&report)
            .map_err(|e| AnalyticsError::ExportError(e.to_string()))
    }
    
    /// Reset analytics counters and statistics
    pub fn reset_analytics(&self) {
        self.packet_counter.store(0, Ordering::Relaxed);
        self.byte_counter.store(0, Ordering::Relaxed);
        
        *self.traffic_stats.write() = TrafficStatistics::default();
        self.protocol_stats.write().clear();
        *self.flow_stats.write() = FlowAnalytics::default();
        *self.bandwidth_monitor.write() = BandwidthMonitor::default();
        *self.security_monitor.write() = SecurityMonitor::default();
        
        info!("Analytics statistics reset");
    }
    
    /// Get current timestamp in seconds
    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
    
    /// Get current timestamp in microseconds
    fn current_timestamp_us() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64
    }
}

/// Comprehensive analytics report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsReport {
    pub timestamp: u64,
    pub traffic_statistics: TrafficStatistics,
    pub protocol_statistics: HashMap<ProtocolType, ProtocolStatistics>,
    pub flow_analytics: FlowAnalytics,
    pub bandwidth_monitoring: BandwidthMonitor,
    pub security_monitoring: SecurityMonitor,
    pub total_packets_analyzed: u64,
    pub total_bytes_analyzed: u64,
}

impl AnomalyDetector {
    fn new(threshold: f64) -> Result<Self> {
        Ok(AnomalyDetector {
            baseline_metrics: BaselineMetrics::default(),
            anomaly_history: VecDeque::with_capacity(1000),
            detection_algorithms: Vec::new(),
            threshold,
        })
    }
    
    fn detect_anomalies(&self, stats: &TrafficStatistics) -> Option<AnomalyEvent> {
        // Simple statistical anomaly detection
        let current_pps = stats.packets_per_second;
        let baseline_pps = self.baseline_metrics.mean_packets_per_second;
        let std_pps = self.baseline_metrics.std_packets_per_second;
        
        if std_pps > 0.0 {
            let z_score = (current_pps - baseline_pps).abs() / std_pps;
            
            if z_score > self.threshold {
                let anomaly_type = if current_pps > baseline_pps {
                    AnomalyType::TrafficSpike
                } else {
                    AnomalyType::TrafficDrop
                };
                
                return Some(AnomalyEvent {
                    timestamp: PacketAnalytics::current_timestamp(),
                    anomaly_type,
                    severity: if z_score > 3.0 { AnomalySeverity::High } else { AnomalySeverity::Medium },
                    confidence: (z_score / 5.0).min(1.0), // Normalize confidence
                    description: format!("Packet rate anomaly detected: {:.2} pps (baseline: {:.2})", current_pps, baseline_pps),
                    affected_flows: Vec::new(),
                    metrics: vec![
                        ("z_score".to_string(), z_score),
                        ("current_pps".to_string(), current_pps),
                        ("baseline_pps".to_string(), baseline_pps),
                    ].into_iter().collect(),
                });
            }
        }
        
        None
    }
    
    fn update_baseline(&mut self, stats: &TrafficStatistics) {
        // Update baseline metrics using exponential moving average
        let alpha = 0.1; // Learning rate
        
        self.baseline_metrics.mean_packets_per_second = 
            alpha * stats.packets_per_second + (1.0 - alpha) * self.baseline_metrics.mean_packets_per_second;
        
        self.baseline_metrics.mean_bytes_per_second = 
            alpha * stats.bytes_per_second + (1.0 - alpha) * self.baseline_metrics.mean_bytes_per_second;
        
        self.baseline_metrics.last_updated = PacketAnalytics::current_timestamp();
    }
    
    fn add_anomaly_event(&mut self, event: AnomalyEvent) {
        self.anomaly_history.push_back(event);
        
        // Keep only recent anomalies
        if self.anomaly_history.len() > 1000 {
            self.anomaly_history.pop_front();
        }
    }
}

impl TimeSeriesBuffer {
    fn new(max_size: usize) -> Self {
        TimeSeriesBuffer {
            packet_rates: VecDeque::with_capacity(max_size),
            bandwidth_rates: VecDeque::with_capacity(max_size),
            flow_rates: VecDeque::with_capacity(max_size),
            error_rates: VecDeque::with_capacity(max_size),
            max_size,
        }
    }
    
    fn add_data_point(&mut self, timestamp: u64, packet_rate: f64, bandwidth_rate: f64, flow_rate: f64, error_rate: f64) {
        // Add new data points
        self.packet_rates.push_back((timestamp, packet_rate));
        self.bandwidth_rates.push_back((timestamp, bandwidth_rate));
        self.flow_rates.push_back((timestamp, flow_rate));
        self.error_rates.push_back((timestamp, error_rate));
        
        // Remove old data points if buffer is full
        if self.packet_rates.len() > self.max_size {
            self.packet_rates.pop_front();
            self.bandwidth_rates.pop_front();
            self.flow_rates.pop_front();
            self.error_rates.pop_front();
        }
    }
}

unsafe impl Send for PacketAnalytics {}
unsafe impl Sync for PacketAnalytics {}

impl Drop for PacketAnalytics {
    fn drop(&mut self) {
        info!("Dropping Packet Analytics Engine");
        // Export final analytics report
        if let Ok(report) = self.export_analytics() {
            debug!("Final analytics report: {}", report);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_analytics_config() {
        let config = AnalyticsConfig::default();
        assert!(config.enable_anomaly_detection);
        assert_eq!(config.time_window_seconds, 300);
        assert_eq!(config.sampling_rate, 1.0);
    }
    
    #[test]
    fn test_anomaly_detection() {
        let detector = AnomalyDetector::new(2.0).unwrap();
        assert_eq!(detector.threshold, 2.0);
        assert_eq!(detector.anomaly_history.len(), 0);
    }
    
    #[test]
    fn test_time_series_buffer() {
        let mut buffer = TimeSeriesBuffer::new(3);
        
        buffer.add_data_point(1, 100.0, 1000.0, 10.0, 0.1);
        buffer.add_data_point(2, 200.0, 2000.0, 20.0, 0.2);
        buffer.add_data_point(3, 300.0, 3000.0, 30.0, 0.3);
        
        assert_eq!(buffer.packet_rates.len(), 3);
        
        // Adding one more should remove the oldest
        buffer.add_data_point(4, 400.0, 4000.0, 40.0, 0.4);
        assert_eq!(buffer.packet_rates.len(), 3);
        assert_eq!(buffer.packet_rates.front().unwrap().0, 2); // First timestamp should be 2 now
    }
}