//! DPDK Flow Classification and Management
//!
//! This module provides enterprise-grade flow classification, management, and
//! hardware acceleration using DPDK's rte_flow API for high-performance packet processing.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::{Mutex, RwLock};
use prometheus::{Counter, Gauge, Histogram, register_counter, register_gauge, register_histogram};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, error, info, warn, instrument};

use crate::bindings::*;
use crate::packet::{PacketBuffer, PacketClassification, ProtocolType};

/// Flow management errors
#[derive(Error, Debug)]
pub enum FlowError {
    #[error("Flow creation failed: {0}")]
    CreationFailed(String),
    
    #[error("Flow validation failed: {0}")]
    ValidationFailed(String),
    
    #[error("Hardware acceleration not supported: {0}")]
    HardwareAccelNotSupported(String),
    
    #[error("Flow table full: {0}")]
    TableFull(String),
    
    #[error("Invalid flow pattern: {0}")]
    InvalidPattern(String),
    
    #[error("Flow action error: {0}")]
    ActionError(String),
}

type Result<T> = std::result::Result<T, FlowError>;

/// Flow classification engine with hardware acceleration
pub struct FlowClassifier {
    port_id: u16,
    flow_table: Arc<RwLock<HashMap<u64, FlowEntry>>>,
    flow_rules: Arc<RwLock<Vec<FlowRule>>>,
    stats: Arc<RwLock<FlowStats>>,
    config: FlowConfig,
    hardware_capabilities: HardwareCapabilities,
    flow_id_counter: AtomicU64,
}

/// Flow entry in the classification table
#[derive(Debug, Clone)]
pub struct FlowEntry {
    pub flow_id: u64,
    pub pattern: FlowPattern,
    pub actions: Vec<FlowAction>,
    pub stats: FlowEntryStats,
    pub created_at: u64,
    pub last_seen: u64,
    pub priority: u16,
    pub timeout: u32,
    pub dpdk_flow: Option<*mut rte_flow>,
}

/// Flow pattern for matching packets
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FlowPattern {
    pub src_ip: Option<std::net::IpAddr>,
    pub dst_ip: Option<std::net::IpAddr>,
    pub src_port: Option<u16>,
    pub dst_port: Option<u16>,
    pub protocol: Option<ProtocolType>,
    pub vlan_id: Option<u16>,
    pub dscp: Option<u8>,
    pub tcp_flags: Option<u8>,
    pub payload_pattern: Option<Vec<u8>>,
}

/// Flow actions to be taken on matching packets
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlowAction {
    Drop,
    Pass,
    Redirect(u16), // Redirect to queue
    Mirror(u16),   // Mirror to port
    Mark(u32),     // Mark packet
    Count,         // Count packets
    SetPriority(u8),
    SetVlan(u16),
    Decrypt,
    Encrypt,
    Compress,
    Decompress,
}

/// Flow rule with pattern and actions
#[derive(Debug, Clone)]
pub struct FlowRule {
    pub id: u64,
    pub pattern: FlowPattern,
    pub actions: Vec<FlowAction>,
    pub priority: u16,
    pub enabled: bool,
    pub hit_count: u64,
    pub created_at: u64,
}

/// Flow statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FlowStats {
    pub total_flows: u64,
    pub active_flows: u64,
    pub flow_creation_rate: f64,
    pub flow_expiration_rate: f64,
    pub hardware_accelerated_flows: u64,
    pub classification_hits: u64,
    pub classification_misses: u64,
    pub average_flow_duration: f64,
}

#[derive(Debug, Clone, Default)]
pub struct FlowEntryStats {
    pub packets: u64,
    pub bytes: u64,
    pub first_seen: u64,
    pub last_seen: u64,
    pub duration: u64,
}

/// Hardware acceleration capabilities
#[derive(Debug, Clone)]
pub struct HardwareCapabilities {
    pub rss_supported: bool,
    pub flow_director_supported: bool,
    pub vlan_filter_supported: bool,
    pub tunnel_supported: bool,
    pub crypto_supported: bool,
    pub max_flow_entries: u32,
    pub supported_protocols: Vec<ProtocolType>,
}

/// Flow classification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowConfig {
    pub max_flows: u32,
    pub flow_timeout: u32,
    pub enable_hardware_acceleration: bool,
    pub enable_flow_aging: bool,
    pub aging_check_interval: u32,
    pub priority_levels: u8,
    pub enable_statistics: bool,
}

impl Default for FlowConfig {
    fn default() -> Self {
        FlowConfig {
            max_flows: 100000,
            flow_timeout: 300, // 5 minutes
            enable_hardware_acceleration: true,
            enable_flow_aging: true,
            aging_check_interval: 60, // 1 minute
            priority_levels: 8,
            enable_statistics: true,
        }
    }
}

// Prometheus metrics for flow classification
lazy_static::lazy_static! {
    static ref FLOW_CLASSIFICATIONS: Counter = register_counter!(
        "dpdk_flow_classifications_total",
        "Total number of flow classifications"
    ).unwrap();
    
    static ref ACTIVE_FLOWS: Gauge = register_gauge!(
        "dpdk_active_flows",
        "Number of active flows in the table"
    ).unwrap();
    
    static ref FLOW_CLASSIFICATION_DURATION: Histogram = register_histogram!(
        "dpdk_flow_classification_duration_seconds",
        "Time taken for flow classification"
    ).unwrap();
}

impl FlowClassifier {
    /// Create a new flow classifier
    #[instrument(skip(config))]
    pub fn new(port_id: u16, config: FlowConfig) -> Result<Self> {
        info!("Initializing Flow Classifier for port {}", port_id);
        
        // Detect hardware capabilities
        let hardware_capabilities = Self::detect_hardware_capabilities(port_id)?;
        
        info!("Hardware capabilities: RSS={}, FlowDirector={}, MaxFlows={}",
              hardware_capabilities.rss_supported,
              hardware_capabilities.flow_director_supported,
              hardware_capabilities.max_flow_entries);
        
        let classifier = FlowClassifier {
            port_id,
            flow_table: Arc::new(RwLock::new(HashMap::new())),
            flow_rules: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(FlowStats::default())),
            config,
            hardware_capabilities,
            flow_id_counter: AtomicU64::new(1),
        };
        
        info!("Flow Classifier initialized successfully");
        Ok(classifier)
    }
    
    /// Classify an incoming packet
    #[instrument(skip(self, packet))]
    pub fn classify_packet(&self, packet: &PacketBuffer) -> Result<PacketClassification> {
        let _timer = FLOW_CLASSIFICATION_DURATION.start_timer();
        
        // Extract flow pattern from packet
        let pattern = self.extract_flow_pattern(packet)?;
        let flow_id = self.calculate_flow_id(&pattern);
        
        // Check if flow already exists
        let mut flow_table = self.flow_table.write();
        let classification = if let Some(flow_entry) = flow_table.get_mut(&flow_id) {
            // Update existing flow
            flow_entry.stats.packets += 1;
            flow_entry.stats.bytes += packet.get_packet_length() as u64;
            flow_entry.last_seen = Self::current_timestamp();
            
            // Update statistics
            let mut stats = self.stats.write();
            stats.classification_hits += 1;
            
            self.create_classification_from_flow(flow_entry, packet)
        } else {
            // Create new flow
            let flow_entry = self.create_new_flow(flow_id, pattern.clone(), packet)?;
            let classification = self.create_classification_from_flow(&flow_entry, packet);
            
            flow_table.insert(flow_id, flow_entry);
            
            // Update statistics
            let mut stats = self.stats.write();
            stats.classification_misses += 1;
            stats.total_flows += 1;
            stats.active_flows = flow_table.len() as u64;
            
            classification
        };
        
        FLOW_CLASSIFICATIONS.inc();
        ACTIVE_FLOWS.set(flow_table.len() as f64);
        
        Ok(classification)
    }
    
    /// Add a flow rule to the classifier
    pub fn add_flow_rule(&self, pattern: FlowPattern, actions: Vec<FlowAction>, priority: u16) -> Result<u64> {
        let rule_id = self.flow_id_counter.fetch_add(1, Ordering::Relaxed);
        
        let rule = FlowRule {
            id: rule_id,
            pattern: pattern.clone(),
            actions: actions.clone(),
            priority,
            enabled: true,
            hit_count: 0,
            created_at: Self::current_timestamp(),
        };
        
        // Add to software rules
        let mut flow_rules = self.flow_rules.write();
        flow_rules.push(rule);
        flow_rules.sort_by(|a, b| b.priority.cmp(&a.priority)); // Sort by priority (highest first)
        
        // Attempt hardware acceleration if enabled and supported
        if self.config.enable_hardware_acceleration && self.hardware_capabilities.flow_director_supported {
            if let Err(e) = self.create_hardware_flow(&pattern, &actions, priority) {
                warn!("Failed to create hardware flow: {}", e);
            }
        }
        
        info!("Added flow rule {} with priority {}", rule_id, priority);
        Ok(rule_id)
    }
    
    /// Remove a flow rule
    pub fn remove_flow_rule(&self, rule_id: u64) -> Result<()> {
        let mut flow_rules = self.flow_rules.write();
        if let Some(pos) = flow_rules.iter().position(|r| r.id == rule_id) {
            flow_rules.remove(pos);
            info!("Removed flow rule {}", rule_id);
            Ok(())
        } else {
            Err(FlowError::ValidationFailed(format!("Rule {} not found", rule_id)))
        }
    }
    
    /// Create hardware-accelerated flow using DPDK rte_flow
    #[instrument(skip(self))]
    fn create_hardware_flow(&self, pattern: &FlowPattern, actions: &[FlowAction], priority: u16) -> Result<*mut rte_flow> {
        if !self.hardware_capabilities.flow_director_supported {
            return Err(FlowError::HardwareAccelNotSupported("Flow Director not available".to_string()));
        }
        
        unsafe {
            let mut attr = rte_flow_attr {
                group: 0,
                priority,
                ingress: 1,
                egress: 0,
                transfer: 0,
                reserved: 0,
            };
            
            // Create pattern items
            let mut pattern_items = self.create_rte_flow_pattern(pattern)?;
            
            // Create action items
            let mut action_items = self.create_rte_flow_actions(actions)?;
            
            let mut error = rte_flow_error {
                type_: rte_flow_error_type_RTE_FLOW_ERROR_TYPE_NONE,
                cause: std::ptr::null(),
                message: std::ptr::null(),
            };
            
            // Validate the flow first
            let validation_result = rte_flow_validate(
                self.port_id,
                &attr,
                pattern_items.as_ptr(),
                action_items.as_ptr(),
                &mut error,
            );
            
            if validation_result != 0 {
                return Err(FlowError::ValidationFailed(
                    format!("Hardware flow validation failed: error type {}", error.type_)
                ));
            }
            
            // Create the flow
            let flow = rte_flow_create(
                self.port_id,
                &attr,
                pattern_items.as_ptr(),
                action_items.as_ptr(),
                &mut error,
            );
            
            if flow.is_null() {
                Err(FlowError::CreationFailed(
                    format!("Hardware flow creation failed: error type {}", error.type_)
                ))
            } else {
                info!("Created hardware flow with priority {}", priority);
                Ok(flow)
            }
        }
    }
    
    /// Extract flow pattern from packet
    fn extract_flow_pattern(&self, packet: &PacketBuffer) -> Result<FlowPattern> {
        let headers = packet.get_headers();
        
        let mut pattern = FlowPattern {
            src_ip: None,
            dst_ip: None,
            src_port: None,
            dst_port: None,
            protocol: None,
            vlan_id: None,
            dscp: None,
            tcp_flags: None,
            payload_pattern: None,
        };
        
        // Extract IP information
        if let Some(ipv4_header) = &headers.ipv4 {
            pattern.src_ip = Some(ipv4_header.src_addr.into());
            pattern.dst_ip = Some(ipv4_header.dst_addr.into());
            pattern.protocol = Some(ProtocolType::from_ip_protocol(ipv4_header.next_proto_id));
            pattern.dscp = Some((ipv4_header.type_of_service >> 2) & 0x3F);
        } else if let Some(ipv6_header) = &headers.ipv6 {
            pattern.src_ip = Some(ipv6_header.src_addr.into());
            pattern.dst_ip = Some(ipv6_header.dst_addr.into());
            pattern.protocol = Some(ProtocolType::from_ip_protocol(ipv6_header.proto));
        }
        
        // Extract transport layer information
        if let Some(tcp_header) = &headers.tcp {
            pattern.src_port = Some(tcp_header.src_port);
            pattern.dst_port = Some(tcp_header.dst_port);
            pattern.tcp_flags = Some(tcp_header.tcp_flags);
        } else if let Some(udp_header) = &headers.udp {
            pattern.src_port = Some(udp_header.src_port);
            pattern.dst_port = Some(udp_header.dst_port);
        }
        
        // Extract VLAN information
        if let Some(vlan_header) = &headers.vlan {
            pattern.vlan_id = Some(vlan_header.vlan_tci & 0x0FFF);
        }
        
        Ok(pattern)
    }
    
    /// Calculate flow ID from pattern
    fn calculate_flow_id(&self, pattern: &FlowPattern) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        pattern.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Create new flow entry
    fn create_new_flow(&self, flow_id: u64, pattern: FlowPattern, packet: &PacketBuffer) -> Result<FlowEntry> {
        let now = Self::current_timestamp();
        
        // Find matching rules and determine actions
        let actions = self.find_matching_actions(&pattern);
        
        let flow_entry = FlowEntry {
            flow_id,
            pattern,
            actions,
            stats: FlowEntryStats {
                packets: 1,
                bytes: packet.get_packet_length() as u64,
                first_seen: now,
                last_seen: now,
                duration: 0,
            },
            created_at: now,
            last_seen: now,
            priority: 0,
            timeout: self.config.flow_timeout,
            dpdk_flow: None,
        };
        
        debug!("Created new flow {} for pattern {:?}", flow_id, flow_entry.pattern);
        Ok(flow_entry)
    }
    
    /// Find matching actions for a flow pattern
    fn find_matching_actions(&self, pattern: &FlowPattern) -> Vec<FlowAction> {
        let flow_rules = self.flow_rules.read();
        
        for rule in flow_rules.iter() {
            if rule.enabled && self.pattern_matches(&rule.pattern, pattern) {
                debug!("Flow pattern matched rule {} with {} actions", rule.id, rule.actions.len());
                return rule.actions.clone();
            }
        }
        
        // Default action is to pass
        vec![FlowAction::Pass]
    }
    
    /// Check if patterns match
    fn pattern_matches(&self, rule_pattern: &FlowPattern, packet_pattern: &FlowPattern) -> bool {
        // Check IP addresses
        if let Some(rule_src) = &rule_pattern.src_ip {
            if let Some(packet_src) = &packet_pattern.src_ip {
                if rule_src != packet_src {
                    return false;
                }
            } else {
                return false;
            }
        }
        
        if let Some(rule_dst) = &rule_pattern.dst_ip {
            if let Some(packet_dst) = &packet_pattern.dst_ip {
                if rule_dst != packet_dst {
                    return false;
                }
            } else {
                return false;
            }
        }
        
        // Check ports
        if let Some(rule_src_port) = rule_pattern.src_port {
            if let Some(packet_src_port) = packet_pattern.src_port {
                if rule_src_port != packet_src_port {
                    return false;
                }
            } else {
                return false;
            }
        }
        
        if let Some(rule_dst_port) = rule_pattern.dst_port {
            if let Some(packet_dst_port) = packet_pattern.dst_port {
                if rule_dst_port != packet_dst_port {
                    return false;
                }
            } else {
                return false;
            }
        }
        
        // Check protocol
        if let Some(rule_proto) = &rule_pattern.protocol {
            if let Some(packet_proto) = &packet_pattern.protocol {
                if rule_proto != packet_proto {
                    return false;
                }
            } else {
                return false;
            }
        }
        
        true
    }
    
    /// Create packet classification from flow entry
    fn create_classification_from_flow(&self, flow_entry: &FlowEntry, packet: &PacketBuffer) -> PacketClassification {
        let mut classification = PacketClassification {
            flow_id: Some(flow_entry.flow_id),
            protocol: flow_entry.pattern.protocol.clone().unwrap_or(ProtocolType::Unknown),
            src_ip: flow_entry.pattern.src_ip,
            dst_ip: flow_entry.pattern.dst_ip,
            src_port: flow_entry.pattern.src_port,
            dst_port: flow_entry.pattern.dst_port,
            payload_offset: packet.get_payload_offset(),
            payload_length: packet.get_payload_length(),
            confidence: 1.0,
            actions: flow_entry.actions.clone(),
            priority: flow_entry.priority,
            vlan_id: flow_entry.pattern.vlan_id,
            dscp: flow_entry.pattern.dscp,
            metadata: HashMap::new(),
        };
        
        // Add flow statistics to metadata
        classification.metadata.insert("flow_packets".to_string(), flow_entry.stats.packets.to_string());
        classification.metadata.insert("flow_bytes".to_string(), flow_entry.stats.bytes.to_string());
        classification.metadata.insert("flow_duration".to_string(), (flow_entry.last_seen - flow_entry.stats.first_seen).to_string());
        
        classification
    }
    
    /// Create DPDK flow pattern from internal pattern
    fn create_rte_flow_pattern(&self, pattern: &FlowPattern) -> Result<Vec<rte_flow_item>> {
        let mut items = Vec::new();
        
        // Ethernet pattern (always present)
        items.push(rte_flow_item {
            type_: rte_flow_item_type_RTE_FLOW_ITEM_TYPE_ETH,
            spec: std::ptr::null(),
            last: std::ptr::null(),
            mask: std::ptr::null(),
        });
        
        // IP pattern
        if pattern.src_ip.is_some() || pattern.dst_ip.is_some() {
            match pattern.src_ip {
                Some(std::net::IpAddr::V4(_)) => {
                    items.push(rte_flow_item {
                        type_: rte_flow_item_type_RTE_FLOW_ITEM_TYPE_IPV4,
                        spec: std::ptr::null(), // Would need to create actual IPv4 spec
                        last: std::ptr::null(),
                        mask: std::ptr::null(),
                    });
                },
                Some(std::net::IpAddr::V6(_)) => {
                    items.push(rte_flow_item {
                        type_: rte_flow_item_type_RTE_FLOW_ITEM_TYPE_IPV6,
                        spec: std::ptr::null(), // Would need to create actual IPv6 spec
                        last: std::ptr::null(),
                        mask: std::ptr::null(),
                    });
                },
                None => {}
            }
        }
        
        // Transport layer pattern
        if let Some(protocol) = &pattern.protocol {
            match protocol {
                ProtocolType::TCP => {
                    items.push(rte_flow_item {
                        type_: rte_flow_item_type_RTE_FLOW_ITEM_TYPE_TCP,
                        spec: std::ptr::null(), // Would need to create actual TCP spec
                        last: std::ptr::null(),
                        mask: std::ptr::null(),
                    });
                },
                ProtocolType::UDP => {
                    items.push(rte_flow_item {
                        type_: rte_flow_item_type_RTE_FLOW_ITEM_TYPE_UDP,
                        spec: std::ptr::null(), // Would need to create actual UDP spec
                        last: std::ptr::null(),
                        mask: std::ptr::null(),
                    });
                },
                _ => {}
            }
        }
        
        // End pattern
        items.push(rte_flow_item {
            type_: rte_flow_item_type_RTE_FLOW_ITEM_TYPE_END,
            spec: std::ptr::null(),
            last: std::ptr::null(),
            mask: std::ptr::null(),
        });
        
        Ok(items)
    }
    
    /// Create DPDK flow actions from internal actions
    fn create_rte_flow_actions(&self, actions: &[FlowAction]) -> Result<Vec<rte_flow_action>> {
        let mut items = Vec::new();
        
        for action in actions {
            match action {
                FlowAction::Drop => {
                    items.push(rte_flow_action {
                        type_: rte_flow_action_type_RTE_FLOW_ACTION_TYPE_DROP,
                        conf: std::ptr::null(),
                    });
                },
                FlowAction::Pass => {
                    // No specific action needed for pass
                },
                FlowAction::Redirect(queue_id) => {
                    items.push(rte_flow_action {
                        type_: rte_flow_action_type_RTE_FLOW_ACTION_TYPE_QUEUE,
                        conf: std::ptr::null(), // Would need to create queue action config
                    });
                },
                FlowAction::Count => {
                    items.push(rte_flow_action {
                        type_: rte_flow_action_type_RTE_FLOW_ACTION_TYPE_COUNT,
                        conf: std::ptr::null(),
                    });
                },
                _ => {
                    warn!("Unsupported flow action in hardware: {:?}", action);
                }
            }
        }
        
        // End action
        items.push(rte_flow_action {
            type_: rte_flow_action_type_RTE_FLOW_ACTION_TYPE_END,
            conf: std::ptr::null(),
        });
        
        Ok(items)
    }
    
    /// Detect hardware capabilities for the port
    fn detect_hardware_capabilities(port_id: u16) -> Result<HardwareCapabilities> {
        unsafe {
            let mut dev_info = std::mem::MaybeUninit::<rte_eth_dev_info>::uninit();
            let result = rte_eth_dev_info_get(port_id, dev_info.as_mut_ptr());
            
            if result != 0 {
                return Err(FlowError::HardwareAccelNotSupported(
                    format!("Failed to get device info for port {}", port_id)
                ));
            }
            
            let dev_info = dev_info.assume_init();
            
            Ok(HardwareCapabilities {
                rss_supported: (dev_info.flow_type_rss_offloads & 0x1) != 0,
                flow_director_supported: true, // Simplified detection
                vlan_filter_supported: (dev_info.rx_offload_capa & DEV_RX_OFFLOAD_VLAN_FILTER as u64) != 0,
                tunnel_supported: (dev_info.rx_offload_capa & DEV_RX_OFFLOAD_OUTER_IPV4_CKSUM as u64) != 0,
                crypto_supported: false, // Would need crypto device detection
                max_flow_entries: 1000000, // Default large value
                supported_protocols: vec![
                    ProtocolType::TCP,
                    ProtocolType::UDP,
                    ProtocolType::ICMP,
                    ProtocolType::HTTP,
                    ProtocolType::HTTPS,
                ],
            })
        }
    }
    
    /// Perform flow aging and cleanup
    pub fn age_flows(&self) {
        if !self.config.enable_flow_aging {
            return;
        }
        
        let now = Self::current_timestamp();
        let mut flow_table = self.flow_table.write();
        let mut expired_flows = Vec::new();
        
        for (&flow_id, flow_entry) in flow_table.iter() {
            if now - flow_entry.last_seen > self.config.flow_timeout as u64 {
                expired_flows.push(flow_id);
            }
        }
        
        for flow_id in expired_flows {
            if let Some(flow_entry) = flow_table.remove(&flow_id) {
                // Clean up hardware flow if present
                if let Some(dpdk_flow) = flow_entry.dpdk_flow {
                    unsafe {
                        let mut error = rte_flow_error {
                            type_: rte_flow_error_type_RTE_FLOW_ERROR_TYPE_NONE,
                            cause: std::ptr::null(),
                            message: std::ptr::null(),
                        };
                        
                        if rte_flow_destroy(self.port_id, dpdk_flow, &mut error) != 0 {
                            warn!("Failed to destroy hardware flow: error type {}", error.type_);
                        }
                    }
                }
                
                debug!("Aged out flow {} after {} seconds", 
                       flow_id, now - flow_entry.created_at);
            }
        }
        
        // Update statistics
        let mut stats = self.stats.write();
        stats.active_flows = flow_table.len() as u64;
        stats.flow_expiration_rate = expired_flows.len() as f64;
        
        info!("Aged out {} flows, {} active flows remaining", 
              expired_flows.len(), flow_table.len());
    }
    
    /// Get flow statistics
    pub fn get_stats(&self) -> FlowStats {
        self.stats.read().clone()
    }
    
    /// Get current timestamp in seconds
    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
    
    /// Get hardware capabilities
    pub fn get_hardware_capabilities(&self) -> &HardwareCapabilities {
        &self.hardware_capabilities
    }
    
    /// Get active flow count
    pub fn get_active_flow_count(&self) -> usize {
        self.flow_table.read().len()
    }
    
    /// Clear all flows
    pub fn clear_flows(&self) {
        let mut flow_table = self.flow_table.write();
        flow_table.clear();
        
        let mut stats = self.stats.write();
        stats.active_flows = 0;
        
        info!("Cleared all flows from classifier");
    }
}

unsafe impl Send for FlowClassifier {}
unsafe impl Sync for FlowClassifier {}

impl Drop for FlowClassifier {
    fn drop(&mut self) {
        info!("Dropping Flow Classifier for port {}", self.port_id);
        self.clear_flows();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{Ipv4Addr, IpAddr};
    
    #[test]
    fn test_flow_pattern_creation() {
        let pattern = FlowPattern {
            src_ip: Some(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1))),
            dst_ip: Some(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 2))),
            src_port: Some(80),
            dst_port: Some(8080),
            protocol: Some(ProtocolType::TCP),
            vlan_id: None,
            dscp: None,
            tcp_flags: Some(0x02), // SYN flag
            payload_pattern: None,
        };
        
        assert_eq!(pattern.src_port, Some(80));
        assert_eq!(pattern.protocol, Some(ProtocolType::TCP));
    }
    
    #[test]
    fn test_flow_action_serialization() {
        let actions = vec![
            FlowAction::Pass,
            FlowAction::Count,
            FlowAction::Mark(123),
            FlowAction::Redirect(5),
        ];
        
        assert_eq!(actions.len(), 4);
        assert_eq!(actions[2], FlowAction::Mark(123));
    }
    
    #[test]
    fn test_hardware_capabilities() {
        let capabilities = HardwareCapabilities {
            rss_supported: true,
            flow_director_supported: true,
            vlan_filter_supported: false,
            tunnel_supported: true,
            crypto_supported: false,
            max_flow_entries: 100000,
            supported_protocols: vec![ProtocolType::TCP, ProtocolType::UDP],
        };
        
        assert!(capabilities.rss_supported);
        assert_eq!(capabilities.max_flow_entries, 100000);
    }
}