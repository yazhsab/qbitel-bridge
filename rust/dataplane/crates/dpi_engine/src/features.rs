//! Feature Extraction for DPI Classification
//!
//! This module provides comprehensive feature extraction from network packets
//! and flows for machine learning-based classification and analysis.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use bytes::Bytes;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, error, info, warn, instrument};

use crate::{DpiError, PacketData, ProtocolType};

/// Feature extraction errors
#[derive(Error, Debug)]
pub enum FeatureError {
    #[error("Feature extraction failed: {0}")]
    ExtractionFailed(String),
    
    #[error("Invalid packet data: {0}")]
    InvalidPacketData(String),
    
    #[error("Flow tracking error: {0}")]
    FlowTrackingError(String),
    
    #[error("Statistical calculation error: {0}")]
    StatisticalError(String),
}

type Result<T> = std::result::Result<T, FeatureError>;

/// Comprehensive packet features for ML classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PacketFeatures {
    // Basic packet features
    pub packet_size: u32,
    pub header_length: u16,
    pub payload_length: u32,
    pub protocol_type: u8,
    
    // IP layer features
    pub ip_version: u8,
    pub ttl: u8,
    pub fragment_flags: u8,
    pub dscp: u8,
    pub ecn: u8,
    pub identification: u16,
    pub fragment_offset: u16,
    
    // Transport layer features
    pub src_port: u16,
    pub dst_port: u16,
    pub tcp_flags: Option<u8>,
    pub tcp_window_size: Option<u16>,
    pub tcp_sequence_number: Option<u32>,
    pub tcp_acknowledgment_number: Option<u32>,
    pub udp_checksum: Option<u16>,
    
    // Payload features
    pub payload_entropy: f32,
    pub payload_printable_ratio: f32,
    pub payload_ascii_ratio: f32,
    pub payload_binary_ratio: f32,
    pub payload_null_bytes: u32,
    pub payload_most_frequent_byte: u8,
    pub payload_byte_frequency: Vec<u32>, // 256 elements for each byte value
    
    // Timing features
    pub timestamp_ms: u64,
    pub inter_arrival_time_ms: Option<u64>,
    
    // Statistical features
    pub packet_direction: PacketDirection,
    pub is_retransmission: bool,
    pub is_out_of_order: bool,
    
    // Application layer features
    pub application_data_length: u32,
    pub has_http_headers: bool,
    pub has_tls_handshake: bool,
    pub has_dns_query: bool,
    
    // Custom features
    pub custom_features: HashMap<String, f32>,
}

impl Default for PacketFeatures {
    fn default() -> Self {
        Self {
            packet_size: 0,
            header_length: 0,
            payload_length: 0,
            protocol_type: 0,
            ip_version: 4,
            ttl: 64,
            fragment_flags: 0,
            dscp: 0,
            ecn: 0,
            identification: 0,
            fragment_offset: 0,
            src_port: 0,
            dst_port: 0,
            tcp_flags: None,
            tcp_window_size: None,
            tcp_sequence_number: None,
            tcp_acknowledgment_number: None,
            udp_checksum: None,
            payload_entropy: 0.0,
            payload_printable_ratio: 0.0,
            payload_ascii_ratio: 0.0,
            payload_binary_ratio: 0.0,
            payload_null_bytes: 0,
            payload_most_frequent_byte: 0,
            payload_byte_frequency: vec![0; 256],
            timestamp_ms: 0,
            inter_arrival_time_ms: None,
            packet_direction: PacketDirection::Unknown,
            is_retransmission: false,
            is_out_of_order: false,
            application_data_length: 0,
            has_http_headers: false,
            has_tls_handshake: false,
            has_dns_query: false,
            custom_features: HashMap::new(),
        }
    }
}

/// Flow-level features aggregated from multiple packets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowFeatures {
    // Flow identification
    pub flow_id: u64,
    pub src_ip: std::net::IpAddr,
    pub dst_ip: std::net::IpAddr,
    pub src_port: u16,
    pub dst_port: u16,
    pub protocol: u8,
    
    // Flow timing
    pub start_time: Instant,
    pub end_time: Option<Instant>,
    pub flow_duration_ms: u64,
    pub idle_time_ms: u64,
    pub active_time_ms: u64,
    
    // Packet count features
    pub total_packets: u64,
    pub forward_packets: u64,
    pub backward_packets: u64,
    pub packets_per_second: f32,
    
    // Byte count features
    pub total_bytes: u64,
    pub forward_bytes: u64,
    pub backward_bytes: u64,
    pub bytes_per_second: f32,
    pub avg_packet_size: f32,
    pub max_packet_size: u32,
    pub min_packet_size: u32,
    pub std_packet_size: f32,
    
    // Inter-arrival time features
    pub avg_inter_arrival_time_ms: f32,
    pub std_inter_arrival_time_ms: f32,
    pub max_inter_arrival_time_ms: u64,
    pub min_inter_arrival_time_ms: u64,
    
    // TCP-specific features
    pub tcp_flags_summary: Option<TcpFlagsStatistics>,
    pub tcp_window_sizes: Option<Vec<u16>>,
    pub connection_state: ConnectionState,
    pub syn_flag_count: u32,
    pub fin_flag_count: u32,
    pub rst_flag_count: u32,
    pub ack_flag_count: u32,
    pub urg_flag_count: u32,
    pub psh_flag_count: u32,
    
    // Payload features
    pub total_payload_bytes: u64,
    pub avg_payload_entropy: f32,
    pub payload_type_distribution: HashMap<PayloadType, u32>,
    
    // Behavioral features
    pub bidirectional_flow: bool,
    pub bulk_transfer_indicator: bool,
    pub interactive_flow_indicator: bool,
    pub burst_count: u32,
    pub burst_sizes: Vec<u32>,
    
    // Statistical features
    pub jitter_ms: f32,
    pub throughput_variability: f32,
    pub flow_symmetry_ratio: f32,
    
    // Application layer features
    pub application_protocol: Option<ProtocolType>,
    pub tls_version: Option<String>,
    pub http_methods: Vec<String>,
    pub dns_query_types: Vec<u16>,
    
    // Custom flow features
    pub custom_flow_features: HashMap<String, f32>,
}

/// Packet direction in a flow
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PacketDirection {
    Forward,
    Backward,
    Unknown,
}

/// Connection state for TCP flows
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionState {
    Established,
    SynSent,
    SynReceived,
    FinWait1,
    FinWait2,
    TimeWait,
    Closed,
    CloseWait,
    LastAck,
    Listen,
    Closing,
    Unknown,
}

/// TCP flags statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TcpFlagsStatistics {
    pub syn_count: u32,
    pub ack_count: u32,
    pub fin_count: u32,
    pub rst_count: u32,
    pub psh_count: u32,
    pub urg_count: u32,
    pub ece_count: u32,
    pub cwr_count: u32,
}

/// Payload type classification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PayloadType {
    Text,
    Binary,
    Encrypted,
    Compressed,
    Multimedia,
    Protocol,
    Unknown,
}

/// Feature extractor for packets and flows
pub struct FeatureExtractor {
    flow_tracker: RwLock<HashMap<u64, FlowState>>,
    configuration: FeatureExtractionConfig,
    statistics: RwLock<ExtractionStatistics>,
}

/// Configuration for feature extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureExtractionConfig {
    pub extract_payload_features: bool,
    pub extract_timing_features: bool,
    pub extract_statistical_features: bool,
    pub max_payload_analysis_bytes: usize,
    pub flow_timeout_ms: u64,
    pub enable_deep_payload_inspection: bool,
    pub entropy_calculation_method: EntropyMethod,
}

impl Default for FeatureExtractionConfig {
    fn default() -> Self {
        Self {
            extract_payload_features: true,
            extract_timing_features: true,
            extract_statistical_features: true,
            max_payload_analysis_bytes: 1500,
            flow_timeout_ms: 600000, // 10 minutes
            enable_deep_payload_inspection: false,
            entropy_calculation_method: EntropyMethod::Shannon,
        }
    }
}

/// Methods for calculating entropy
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntropyMethod {
    Shannon,
    Renyi,
    Kolmogorov,
}

/// Internal flow state tracking
struct FlowState {
    features: FlowFeatures,
    last_packet_time: Instant,
    packet_times: Vec<Instant>,
    packet_sizes: Vec<u32>,
    tcp_sequence_numbers: Vec<u32>,
}

/// Feature extraction statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExtractionStatistics {
    pub packets_processed: u64,
    pub flows_tracked: u64,
    pub features_extracted: u64,
    pub extraction_errors: u64,
    pub average_extraction_time_us: f64,
}

impl FeatureExtractor {
    /// Create a new feature extractor
    pub fn new(config: &crate::config::DpiConfig) -> Result<Self> {
        info!("Initializing Feature Extractor");
        
        let extraction_config = FeatureExtractionConfig::default();
        
        let extractor = Self {
            flow_tracker: RwLock::new(HashMap::new()),
            configuration: extraction_config,
            statistics: RwLock::new(ExtractionStatistics::default()),
        };
        
        info!("Feature Extractor initialized successfully");
        Ok(extractor)
    }
    
    /// Extract features from a single packet
    #[instrument(skip(self, packet_data))]
    pub async fn extract_packet_features(&self, packet_data: &PacketData) -> Result<PacketFeatures> {
        let start_time = Instant::now();
        
        let mut features = PacketFeatures {
            packet_size: packet_data.data.len() as u32,
            timestamp_ms: packet_data.timestamp.elapsed().as_millis() as u64,
            protocol_type: packet_data.protocol,
            src_port: packet_data.src_port,
            dst_port: packet_data.dst_port,
            ..Default::default()
        };
        
        // Extract IP layer features
        self.extract_ip_features(&mut features, &packet_data.data)?;
        
        // Extract transport layer features
        self.extract_transport_features(&mut features, &packet_data.data, packet_data.protocol)?;
        
        // Extract payload features if enabled
        if self.configuration.extract_payload_features {
            self.extract_payload_features(&mut features, &packet_data.data, packet_data.payload_offset)?;
        }
        
        // Extract application layer features
        self.extract_application_features(&mut features, &packet_data.data, packet_data.payload_offset)?;
        
        // Update flow tracking and extract flow-based features
        if let Some(flow_id) = packet_data.flow_id {
            self.update_flow_tracking(flow_id, packet_data, &features)?;
            features.inter_arrival_time_ms = self.get_inter_arrival_time(flow_id);
            features.packet_direction = self.determine_packet_direction(flow_id, packet_data)?;
        }
        
        // Update statistics
        let extraction_time = start_time.elapsed().as_micros() as u64;
        let mut stats = self.statistics.write();
        stats.packets_processed += 1;
        stats.features_extracted += 1;
        stats.average_extraction_time_us = 
            (stats.average_extraction_time_us * (stats.packets_processed - 1) as f64 + extraction_time as f64) 
            / stats.packets_processed as f64;
        
        debug!("Extracted features for packet {} in {}μs", packet_data.packet_id, extraction_time);
        Ok(features)
    }
    
    /// Extract IP layer features
    fn extract_ip_features(&self, features: &mut PacketFeatures, packet_data: &Bytes) -> Result<()> {
        if packet_data.len() < 20 {
            return Err(FeatureError::InvalidPacketData("Packet too short for IP header".to_string()));
        }
        
        // Parse IP header (simplified)
        let ip_version = (packet_data[0] >> 4) & 0x0F;
        let header_length = ((packet_data[0] & 0x0F) * 4) as u16;
        let dscp = (packet_data[1] >> 2) & 0x3F;
        let ecn = packet_data[1] & 0x03;
        let total_length = u16::from_be_bytes([packet_data[2], packet_data[3]]);
        let identification = u16::from_be_bytes([packet_data[4], packet_data[5]]);
        let fragment_flags = (packet_data[6] >> 5) & 0x07;
        let fragment_offset = u16::from_be_bytes([packet_data[6], packet_data[7]]) & 0x1FFF;
        let ttl = packet_data[8];
        
        features.ip_version = ip_version;
        features.header_length = header_length;
        features.dscp = dscp;
        features.ecn = ecn;
        features.identification = identification;
        features.fragment_flags = fragment_flags;
        features.fragment_offset = fragment_offset;
        features.ttl = ttl;
        features.payload_length = total_length.saturating_sub(header_length as u16) as u32;
        
        Ok(())
    }
    
    /// Extract transport layer features
    fn extract_transport_features(&self, features: &mut PacketFeatures, packet_data: &Bytes, protocol: u8) -> Result<()> {
        let ip_header_length = features.header_length as usize;
        
        if packet_data.len() < ip_header_length + 8 {
            return Ok(()); // Not enough data for transport header
        }
        
        match protocol {
            6 => { // TCP
                self.extract_tcp_features(features, &packet_data[ip_header_length..])?;
            },
            17 => { // UDP
                self.extract_udp_features(features, &packet_data[ip_header_length..])?;
            },
            1 => { // ICMP
                self.extract_icmp_features(features, &packet_data[ip_header_length..])?;
            },
            _ => {
                // Other protocols
            }
        }
        
        Ok(())
    }
    
    /// Extract TCP-specific features
    fn extract_tcp_features(&self, features: &mut PacketFeatures, tcp_data: &[u8]) -> Result<()> {
        if tcp_data.len() < 20 {
            return Ok(());
        }
        
        let src_port = u16::from_be_bytes([tcp_data[0], tcp_data[1]]);
        let dst_port = u16::from_be_bytes([tcp_data[2], tcp_data[3]]);
        let sequence_number = u32::from_be_bytes([tcp_data[4], tcp_data[5], tcp_data[6], tcp_data[7]]);
        let acknowledgment_number = u32::from_be_bytes([tcp_data[8], tcp_data[9], tcp_data[10], tcp_data[11]]);
        let tcp_flags = tcp_data[13];
        let window_size = u16::from_be_bytes([tcp_data[14], tcp_data[15]]);
        
        features.src_port = src_port;
        features.dst_port = dst_port;
        features.tcp_flags = Some(tcp_flags);
        features.tcp_window_size = Some(window_size);
        features.tcp_sequence_number = Some(sequence_number);
        features.tcp_acknowledgment_number = Some(acknowledgment_number);
        
        Ok(())
    }
    
    /// Extract UDP-specific features
    fn extract_udp_features(&self, features: &mut PacketFeatures, udp_data: &[u8]) -> Result<()> {
        if udp_data.len() < 8 {
            return Ok(());
        }
        
        let src_port = u16::from_be_bytes([udp_data[0], udp_data[1]]);
        let dst_port = u16::from_be_bytes([udp_data[2], udp_data[3]]);
        let checksum = u16::from_be_bytes([udp_data[6], udp_data[7]]);
        
        features.src_port = src_port;
        features.dst_port = dst_port;
        features.udp_checksum = Some(checksum);
        
        Ok(())
    }
    
    /// Extract ICMP-specific features
    fn extract_icmp_features(&self, _features: &mut PacketFeatures, _icmp_data: &[u8]) -> Result<()> {
        // ICMP feature extraction would go here
        Ok(())
    }
    
    /// Extract payload features
    fn extract_payload_features(&self, features: &mut PacketFeatures, packet_data: &Bytes, payload_offset: usize) -> Result<()> {
        if payload_offset >= packet_data.len() {
            return Ok(()); // No payload
        }
        
        let payload = &packet_data[payload_offset..];
        let max_analysis_bytes = self.configuration.max_payload_analysis_bytes.min(payload.len());
        let analysis_payload = &payload[..max_analysis_bytes];
        
        // Calculate payload entropy
        features.payload_entropy = self.calculate_entropy(analysis_payload)?;
        
        // Calculate character ratios
        let (printable_count, ascii_count, null_count) = self.analyze_payload_characters(analysis_payload);
        let total_bytes = analysis_payload.len() as f32;
        
        features.payload_printable_ratio = printable_count as f32 / total_bytes;
        features.payload_ascii_ratio = ascii_count as f32 / total_bytes;
        features.payload_binary_ratio = 1.0 - features.payload_ascii_ratio;
        features.payload_null_bytes = null_count;
        
        // Calculate byte frequency distribution
        features.payload_byte_frequency = self.calculate_byte_frequency(analysis_payload);
        features.payload_most_frequent_byte = self.find_most_frequent_byte(&features.payload_byte_frequency);
        
        Ok(())
    }
    
    /// Extract application layer features
    fn extract_application_features(&self, features: &mut PacketFeatures, packet_data: &Bytes, payload_offset: usize) -> Result<()> {
        if payload_offset >= packet_data.len() {
            return Ok(());
        }
        
        let payload = &packet_data[payload_offset..];
        features.application_data_length = payload.len() as u32;
        
        // Check for HTTP headers
        features.has_http_headers = self.detect_http_headers(payload);
        
        // Check for TLS handshake
        features.has_tls_handshake = self.detect_tls_handshake(payload);
        
        // Check for DNS query
        features.has_dns_query = self.detect_dns_query(payload, features.dst_port);
        
        Ok(())
    }
    
    /// Calculate entropy of payload data
    fn calculate_entropy(&self, data: &[u8]) -> Result<f32> {
        if data.is_empty() {
            return Ok(0.0);
        }
        
        match self.configuration.entropy_calculation_method {
            EntropyMethod::Shannon => self.calculate_shannon_entropy(data),
            EntropyMethod::Renyi => self.calculate_renyi_entropy(data),
            EntropyMethod::Kolmogorov => self.calculate_kolmogorov_entropy(data),
        }
    }
    
    /// Calculate Shannon entropy
    fn calculate_shannon_entropy(&self, data: &[u8]) -> Result<f32> {
        let mut frequency = [0u32; 256];
        for &byte in data {
            frequency[byte as usize] += 1;
        }
        
        let data_len = data.len() as f32;
        let mut entropy = 0.0f32;
        
        for &count in frequency.iter() {
            if count > 0 {
                let probability = count as f32 / data_len;
                entropy -= probability * probability.log2();
            }
        }
        
        Ok(entropy)
    }
    
    /// Calculate Renyi entropy (simplified)
    fn calculate_renyi_entropy(&self, data: &[u8]) -> Result<f32> {
        // Simplified implementation - in practice would use different alpha values
        self.calculate_shannon_entropy(data)
    }
    
    /// Calculate Kolmogorov entropy (approximation)
    fn calculate_kolmogorov_entropy(&self, data: &[u8]) -> Result<f32> {
        // This is a very simplified approximation
        // Real Kolmogorov complexity is uncomputable
        let unique_bytes = data.iter().collect::<std::collections::HashSet<_>>().len();
        Ok((unique_bytes as f32).log2())
    }
    
    /// Analyze payload characters
    fn analyze_payload_characters(&self, payload: &[u8]) -> (usize, usize, u32) {
        let mut printable_count = 0;
        let mut ascii_count = 0;
        let mut null_count = 0;
        
        for &byte in payload {
            if byte == 0 {
                null_count += 1;
            }
            if byte < 128 {
                ascii_count += 1;
            }
            if byte >= 32 && byte <= 126 {
                printable_count += 1;
            }
        }
        
        (printable_count, ascii_count, null_count)
    }
    
    /// Calculate byte frequency distribution
    fn calculate_byte_frequency(&self, payload: &[u8]) -> Vec<u32> {
        let mut frequency = vec![0u32; 256];
        for &byte in payload {
            frequency[byte as usize] += 1;
        }
        frequency
    }
    
    /// Find most frequent byte
    fn find_most_frequent_byte(&self, frequency: &[u32]) -> u8 {
        frequency.iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(byte, _)| byte as u8)
            .unwrap_or(0)
    }
    
    /// Detect HTTP headers in payload
    fn detect_http_headers(&self, payload: &[u8]) -> bool {
        if payload.len() < 4 {
            return false;
        }
        
        // Check for common HTTP methods and response codes
        let payload_str = std::str::from_utf8(&payload[..payload.len().min(100)]).unwrap_or("");
        
        payload_str.starts_with("GET ") ||
        payload_str.starts_with("POST ") ||
        payload_str.starts_with("PUT ") ||
        payload_str.starts_with("DELETE ") ||
        payload_str.starts_with("HEAD ") ||
        payload_str.starts_with("OPTIONS ") ||
        payload_str.starts_with("HTTP/")
    }
    
    /// Detect TLS handshake in payload
    fn detect_tls_handshake(&self, payload: &[u8]) -> bool {
        if payload.len() < 5 {
            return false;
        }
        
        // Check for TLS record header
        payload[0] == 0x16 && // Handshake
        payload[1] == 0x03 && // TLS version major
        (payload[2] >= 0x01 && payload[2] <= 0x04) // TLS version minor (1.0 to 1.3)
    }
    
    /// Detect DNS query in payload
    fn detect_dns_query(&self, payload: &[u8], dst_port: u16) -> bool {
        dst_port == 53 && payload.len() >= 12 // DNS header is 12 bytes minimum
    }
    
    /// Update flow tracking information
    fn update_flow_tracking(&self, flow_id: u64, packet_data: &PacketData, packet_features: &PacketFeatures) -> Result<()> {
        let mut flow_tracker = self.flow_tracker.write();
        
        let flow_state = flow_tracker.entry(flow_id).or_insert_with(|| {
            FlowState {
                features: FlowFeatures {
                    flow_id,
                    src_ip: packet_data.src_ip,
                    dst_ip: packet_data.dst_ip,
                    src_port: packet_data.src_port,
                    dst_port: packet_data.dst_port,
                    protocol: packet_data.protocol,
                    start_time: packet_data.timestamp,
                    end_time: None,
                    flow_duration_ms: 0,
                    idle_time_ms: 0,
                    active_time_ms: 0,
                    total_packets: 0,
                    forward_packets: 0,
                    backward_packets: 0,
                    packets_per_second: 0.0,
                    total_bytes: 0,
                    forward_bytes: 0,
                    backward_bytes: 0,
                    bytes_per_second: 0.0,
                    avg_packet_size: 0.0,
                    max_packet_size: 0,
                    min_packet_size: u32::MAX,
                    std_packet_size: 0.0,
                    avg_inter_arrival_time_ms: 0.0,
                    std_inter_arrival_time_ms: 0.0,
                    max_inter_arrival_time_ms: 0,
                    min_inter_arrival_time_ms: u64::MAX,
                    tcp_flags_summary: None,
                    tcp_window_sizes: None,
                    connection_state: ConnectionState::Unknown,
                    syn_flag_count: 0,
                    fin_flag_count: 0,
                    rst_flag_count: 0,
                    ack_flag_count: 0,
                    urg_flag_count: 0,
                    psh_flag_count: 0,
                    total_payload_bytes: 0,
                    avg_payload_entropy: 0.0,
                    payload_type_distribution: HashMap::new(),
                    bidirectional_flow: false,
                    bulk_transfer_indicator: false,
                    interactive_flow_indicator: false,
                    burst_count: 0,
                    burst_sizes: Vec::new(),
                    jitter_ms: 0.0,
                    throughput_variability: 0.0,
                    flow_symmetry_ratio: 0.0,
                    application_protocol: None,
                    tls_version: None,
                    http_methods: Vec::new(),
                    dns_query_types: Vec::new(),
                    custom_flow_features: HashMap::new(),
                },
                last_packet_time: packet_data.timestamp,
                packet_times: Vec::new(),
                packet_sizes: Vec::new(),
                tcp_sequence_numbers: Vec::new(),
            }
        });
        
        // Update flow statistics
        flow_state.features.total_packets += 1;
        flow_state.features.total_bytes += packet_features.packet_size as u64;
        flow_state.features.end_time = Some(packet_data.timestamp);
        flow_state.features.flow_duration_ms = packet_data.timestamp
            .duration_since(flow_state.features.start_time)
            .as_millis() as u64;
        
        // Update packet sizes
        flow_state.packet_sizes.push(packet_features.packet_size);
        flow_state.features.max_packet_size = flow_state.features.max_packet_size.max(packet_features.packet_size);
        flow_state.features.min_packet_size = flow_state.features.min_packet_size.min(packet_features.packet_size);
        
        // Update timing information
        flow_state.packet_times.push(packet_data.timestamp);
        flow_state.last_packet_time = packet_data.timestamp;
        
        // Update TCP-specific information
        if let Some(tcp_flags) = packet_features.tcp_flags {
            self.update_tcp_flow_features(&mut flow_state.features, tcp_flags);
        }
        
        // Update sequence numbers for TCP
        if let Some(seq_num) = packet_features.tcp_sequence_number {
            flow_state.tcp_sequence_numbers.push(seq_num);
        }
        
        Ok(())
    }
    
    /// Update TCP-specific flow features
    fn update_tcp_flow_features(&self, flow_features: &mut FlowFeatures, tcp_flags: u8) {
        if tcp_flags & 0x02 != 0 { flow_features.syn_flag_count += 1; } // SYN
        if tcp_flags & 0x10 != 0 { flow_features.ack_flag_count += 1; } // ACK
        if tcp_flags & 0x01 != 0 { flow_features.fin_flag_count += 1; } // FIN
        if tcp_flags & 0x04 != 0 { flow_features.rst_flag_count += 1; } // RST
        if tcp_flags & 0x08 != 0 { flow_features.psh_flag_count += 1; } // PSH
        if tcp_flags & 0x20 != 0 { flow_features.urg_flag_count += 1; } // URG
        
        // Update connection state based on flags
        flow_features.connection_state = self.determine_connection_state(flow_features);
    }
    
    /// Determine TCP connection state
    fn determine_connection_state(&self, flow_features: &FlowFeatures) -> ConnectionState {
        if flow_features.syn_flag_count > 0 && flow_features.ack_flag_count > 0 && flow_features.fin_flag_count == 0 {
            ConnectionState::Established
        } else if flow_features.syn_flag_count > 0 && flow_features.ack_flag_count == 0 {
            ConnectionState::SynSent
        } else if flow_features.fin_flag_count > 0 {
            ConnectionState::FinWait1
        } else if flow_features.rst_flag_count > 0 {
            ConnectionState::Closed
        } else {
            ConnectionState::Unknown
        }
    }
    
    /// Get inter-arrival time for a flow
    fn get_inter_arrival_time(&self, flow_id: u64) -> Option<u64> {
        let flow_tracker = self.flow_tracker.read();
        if let Some(flow_state) = flow_tracker.get(&flow_id) {
            if flow_state.packet_times.len() >= 2 {
                let last_two = &flow_state.packet_times[flow_state.packet_times.len() - 2..];
                return Some(last_two[1].duration_since(last_two[0]).as_millis() as u64);
            }
        }
        None
    }
    
    /// Determine packet direction in flow
    fn determine_packet_direction(&self, flow_id: u64, packet_data: &PacketData) -> Result<PacketDirection> {
        let flow_tracker = self.flow_tracker.read();
        if let Some(flow_state) = flow_tracker.get(&flow_id) {
            if packet_data.src_ip == flow_state.features.src_ip && 
               packet_data.src_port == flow_state.features.src_port {
                Ok(PacketDirection::Forward)
            } else {
                Ok(PacketDirection::Backward)
            }
        } else {
            Ok(PacketDirection::Unknown)
        }
    }
    
    /// Get flow features for a specific flow
    pub fn get_flow_features(&self, flow_id: u64) -> Option<FlowFeatures> {
        let flow_tracker = self.flow_tracker.read();
        flow_tracker.get(&flow_id).map(|state| state.features.clone())
    }
    
    /// Get extraction statistics
    pub fn get_statistics(&self) -> ExtractionStatistics {
        self.statistics.read().clone()
    }
    
    /// Clean up expired flows
    pub fn cleanup_expired_flows(&self) {
        let mut flow_tracker = self.flow_tracker.write();
        let now = Instant::now();
        let timeout = Duration::from_millis(self.configuration.flow_timeout_ms);
        
        flow_tracker.retain(|_, state| {
            now.duration_since(state.last_packet_time) < timeout
        });
        
        let mut stats = self.statistics.write();
        stats.flows_tracked = flow_tracker.len() as u64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use std::net::{IpAddr, Ipv4Addr};
    
    #[tokio::test]
    async fn test_feature_extraction() {
        let config = crate::config::DpiConfig::default();
        let extractor = FeatureExtractor::new(&config).unwrap();
        
        let packet_data = PacketData {
            packet_id: 1,
            timestamp: Instant::now(),
            data: Bytes::from(vec![
                0x45, 0x00, 0x00, 0x28, // IP header start
                0x00, 0x00, 0x40, 0x00,
                0x40, 0x06, 0x00, 0x00, // TTL=64, Protocol=6 (TCP)
                0xC0, 0xA8, 0x01, 0x64, // Source IP: 192.168.1.100
                0xC0, 0xA8, 0x01, 0x01, // Dest IP: 192.168.1.1
            ]),
            flow_id: Some(12345),
            src_ip: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100)),
            dst_ip: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
            src_port: 12345,
            dst_port: 80,
            protocol: 6,
            payload_offset: 20,
            metadata: HashMap::new(),
        };
        
        let features = extractor.extract_packet_features(&packet_data).await.unwrap();
        
        assert_eq!(features.ip_version, 4);
        assert_eq!(features.protocol_type, 6);
        assert_eq!(features.src_port, 12345);
        assert_eq!(features.dst_port, 80);
    }
    
    #[test]
    fn test_entropy_calculation() {
        let config = crate::config::DpiConfig::default();
        let extractor = FeatureExtractor::new(&config).unwrap();
        
        // Test with uniform data (maximum entropy)
        let uniform_data = (0u8..=255u8).collect::<Vec<_>>();
        let entropy = extractor.calculate_shannon_entropy(&uniform_data).unwrap();
        assert!(entropy > 7.9); // Should be close to 8 bits
        
        // Test with single byte (minimum entropy)
        let single_byte_data = vec![0u8; 100];
        let entropy = extractor.calculate_shannon_entropy(&single_byte_data).unwrap();
        assert_eq!(entropy, 0.0);
    }
    
    #[test]
    fn test_payload_analysis() {
        let config = crate::config::DpiConfig::default();
        let extractor = FeatureExtractor::new(&config).unwrap();
        
        let text_payload = b"Hello, World! This is a text payload.";
        let (printable, ascii, null) = extractor.analyze_payload_characters(text_payload);
        
        assert_eq!(ascii, text_payload.len());
        assert!(printable > 0);
        assert_eq!(null, 0);
    }
}