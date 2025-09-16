//! High-Performance Packet Processing
//!
//! This module provides zero-copy packet processing capabilities with DPDK integration,
//! including packet parsing, classification, and forwarding with enterprise-grade features.

use std::collections::HashMap;
use std::fmt;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::sync::Arc;

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, error, info, warn, instrument};

use crate::bindings::*;
use crate::memory::MemoryPool;

/// Packet Processing Errors
#[derive(Error, Debug)]
pub enum PacketError {
    #[error("Invalid packet format: {0}")]
    InvalidFormat(String),
    
    #[error("Packet too small: {0} bytes")]
    PacketTooSmall(usize),
    
    #[error("Packet too large: {0} bytes")]
    PacketTooLarge(usize),
    
    #[error("Unsupported protocol: {0}")]
    UnsupportedProtocol(String),
    
    #[error("Memory allocation failed")]
    MemoryAllocationFailed,
    
    #[error("Checksum verification failed")]
    ChecksumFailed,
    
    #[error("Packet parsing failed: {0}")]
    ParsingFailed(String),
}

type Result<T> = std::result::Result<T, PacketError>;

/// High-performance packet buffer with zero-copy operations
pub struct PacketBuffer {
    mbuf: *mut rte_mbuf,
    data: *mut u8,
    data_len: usize,
    buffer_len: usize,
    metadata: PacketMetadata,
    pool: Arc<MemoryPool>,
}

/// Comprehensive packet metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PacketMetadata {
    pub timestamp: u64,
    pub port_id: u16,
    pub queue_id: u16,
    pub packet_type: PacketType,
    pub l2_header: Option<L2Header>,
    pub l3_header: Option<L3Header>,
    pub l4_header: Option<L4Header>,
    pub flow_hash: Option<u32>,
    pub vlan_tci: Option<u16>,
    pub tunnel_info: Option<TunnelInfo>,
    pub classification: PacketClassification,
    pub qos_priority: u8,
    pub security_flags: SecurityFlags,
}

/// Packet type identification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PacketType {
    IPv4,
    IPv6,
    ARP,
    LLDP,
    Unknown(u16), // EtherType for unknown protocols
}

/// Layer 2 (Ethernet) header information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L2Header {
    pub src_mac: [u8; 6],
    pub dst_mac: [u8; 6],
    pub ether_type: u16,
    pub vlan_tag: Option<VlanTag>,
}

/// VLAN tag information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VlanTag {
    pub tci: u16,
    pub vid: u16,
    pub pcp: u8,
    pub dei: bool,
}

/// Layer 3 (IP) header information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L3Header {
    pub src_ip: IpAddr,
    pub dst_ip: IpAddr,
    pub protocol: u8,
    pub ttl: u8,
    pub tos: u8,
    pub length: u16,
    pub fragment_info: Option<FragmentInfo>,
}

/// IP fragmentation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentInfo {
    pub id: u16,
    pub flags: u8,
    pub fragment_offset: u16,
}

/// Layer 4 (Transport) header information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum L4Header {
    TCP {
        src_port: u16,
        dst_port: u16,
        seq_num: u32,
        ack_num: u32,
        flags: u16,
        window: u16,
        options: Option<Vec<u8>>,
    },
    UDP {
        src_port: u16,
        dst_port: u16,
        length: u16,
        checksum: u16,
    },
    ICMP {
        icmp_type: u8,
        code: u8,
        checksum: u16,
        data: Vec<u8>,
    },
}

/// Tunnel information for encapsulated packets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunnelInfo {
    pub tunnel_type: TunnelType,
    pub outer_l3: Option<L3Header>,
    pub outer_l4: Option<L4Header>,
    pub tunnel_id: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TunnelType {
    VXLAN,
    GRE,
    IPIP,
    L2TP,
    NVGRE,
}

/// Advanced packet classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PacketClassification {
    pub application_protocol: Option<String>,
    pub traffic_class: TrafficClass,
    pub flow_direction: FlowDirection,
    pub confidence: f32,
    pub signatures: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrafficClass {
    Web,
    Email,
    FileTransfer,
    Streaming,
    Gaming,
    VoIP,
    Database,
    Messaging,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlowDirection {
    Ingress,
    Egress,
    Internal,
}

/// Security-related packet flags
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SecurityFlags {
    pub is_malicious: bool,
    pub is_suspicious: bool,
    pub has_anomaly: bool,
    pub blocked: bool,
    pub threat_score: u8,
    pub detection_rules: Vec<String>,
}

/// Packet processor trait for extensible packet handling
#[async_trait]
pub trait PacketProcessor: Send + Sync {
    /// Process a single packet
    async fn process_packet(&self, packet: &mut PacketBuffer) -> Result<ProcessingDecision>;
    
    /// Process a batch of packets for better performance
    async fn process_batch(&self, packets: &mut [PacketBuffer]) -> Result<Vec<ProcessingDecision>>;
    
    /// Get processor statistics
    fn get_stats(&self) -> ProcessorStats;
    
    /// Reset processor state
    async fn reset(&self);
}

/// Packet processing decision
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessingDecision {
    Forward { next_hop: Option<String>, port: Option<u16> },
    Drop { reason: String },
    Redirect { target: String },
    Mirror { targets: Vec<String> },
    Queue { queue_id: u16, priority: u8 },
}

/// Packet processor statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProcessorStats {
    pub packets_processed: u64,
    pub packets_forwarded: u64,
    pub packets_dropped: u64,
    pub packets_redirected: u64,
    pub processing_errors: u64,
    pub average_processing_time_ns: u64,
    pub classification_hits: HashMap<String, u64>,
}

impl PacketBuffer {
    /// Create a new packet buffer from DPDK mbuf
    pub fn from_mbuf(mbuf: *mut rte_mbuf, pool: Arc<MemoryPool>) -> Result<Self> {
        if mbuf.is_null() {
            return Err(PacketError::MemoryAllocationFailed);
        }
        
        unsafe {
            let data = rte_pktmbuf_mtod(mbuf, *mut u8);
            let data_len = (*mbuf).data_len as usize;
            let buffer_len = (*mbuf).buf_len as usize;
            
            Ok(PacketBuffer {
                mbuf,
                data,
                data_len,
                buffer_len,
                metadata: PacketMetadata::new(),
                pool,
            })
        }
    }
    
    /// Allocate a new packet buffer from memory pool
    pub fn allocate(pool: Arc<MemoryPool>, size: usize) -> Result<Self> {
        let mbuf = pool.allocate_mbuf()?;
        if mbuf.is_null() {
            return Err(PacketError::MemoryAllocationFailed);
        }
        
        unsafe {
            // Reserve space for headers
            let data = rte_pktmbuf_append(mbuf, size as u16) as *mut u8;
            if data.is_null() {
                pool.free_mbuf(mbuf);
                return Err(PacketError::MemoryAllocationFailed);
            }
            
            Ok(PacketBuffer {
                mbuf,
                data,
                data_len: size,
                buffer_len: (*mbuf).buf_len as usize,
                metadata: PacketMetadata::new(),
                pool,
            })
        }
    }
    
    /// Get packet data as a slice
    pub fn data(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.data, self.data_len) }
    }
    
    /// Get mutable packet data
    pub fn data_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.data, self.data_len) }
    }
    
    /// Get packet length
    pub fn len(&self) -> usize {
        self.data_len
    }
    
    /// Check if packet is empty
    pub fn is_empty(&self) -> bool {
        self.data_len == 0
    }
    
    /// Parse packet headers and populate metadata
    #[instrument(skip(self))]
    pub fn parse_headers(&mut self) -> Result<()> {
        if self.data_len < 14 {
            return Err(PacketError::PacketTooSmall(self.data_len));
        }
        
        let data = self.data();
        
        // Parse Ethernet header
        let l2_header = self.parse_ethernet_header(data)?;
        self.metadata.packet_type = PacketType::from_ether_type(l2_header.ether_type);
        self.metadata.l2_header = Some(l2_header);
        
        // Parse L3 header based on EtherType
        let l3_offset = if self.metadata.l2_header.as_ref().unwrap().vlan_tag.is_some() { 18 } else { 14 };
        
        match self.metadata.packet_type {
            PacketType::IPv4 => {
                if data.len() >= l3_offset + 20 {
                    let l3_header = self.parse_ipv4_header(&data[l3_offset..])?;
                    let l4_offset = l3_offset + ((data[l3_offset] & 0x0F) as usize * 4);
                    
                    // Parse L4 header
                    if data.len() > l4_offset {
                        match l3_header.protocol {
                            6 => { // TCP
                                if data.len() >= l4_offset + 20 {
                                    self.metadata.l4_header = Some(self.parse_tcp_header(&data[l4_offset..])?);
                                }
                            },
                            17 => { // UDP
                                if data.len() >= l4_offset + 8 {
                                    self.metadata.l4_header = Some(self.parse_udp_header(&data[l4_offset..])?);
                                }
                            },
                            1 => { // ICMP
                                if data.len() >= l4_offset + 8 {
                                    self.metadata.l4_header = Some(self.parse_icmp_header(&data[l4_offset..])?);
                                }
                            },
                            _ => {}
                        }
                    }
                    
                    self.metadata.l3_header = Some(l3_header);
                }
            },
            PacketType::IPv6 => {
                if data.len() >= l3_offset + 40 {
                    let l3_header = self.parse_ipv6_header(&data[l3_offset..])?;
                    self.metadata.l3_header = Some(l3_header);
                    // TODO: Handle IPv6 extension headers
                }
            },
            PacketType::ARP => {
                // ARP parsing would go here
            },
            _ => {}
        }
        
        // Calculate flow hash for RSS
        self.calculate_flow_hash();
        
        debug!("Parsed packet: {:?}", self.metadata.packet_type);
        Ok(())
    }
    
    fn parse_ethernet_header(&self, data: &[u8]) -> Result<L2Header> {
        if data.len() < 14 {
            return Err(PacketError::InvalidFormat("Ethernet header too short".to_string()));
        }
        
        let mut dst_mac = [0u8; 6];
        let mut src_mac = [0u8; 6];
        
        dst_mac.copy_from_slice(&data[0..6]);
        src_mac.copy_from_slice(&data[6..12]);
        
        let ether_type = u16::from_be_bytes([data[12], data[13]]);
        
        // Check for VLAN tag
        let (ether_type, vlan_tag) = if ether_type == 0x8100 {
            if data.len() < 18 {
                return Err(PacketError::InvalidFormat("VLAN header truncated".to_string()));
            }
            
            let tci = u16::from_be_bytes([data[14], data[15]]);
            let inner_ether_type = u16::from_be_bytes([data[16], data[17]]);
            
            let vlan_tag = VlanTag {
                tci,
                vid: tci & 0x0FFF,
                pcp: ((tci & 0xE000) >> 13) as u8,
                dei: (tci & 0x1000) != 0,
            };
            
            (inner_ether_type, Some(vlan_tag))
        } else {
            (ether_type, None)
        };
        
        Ok(L2Header {
            dst_mac,
            src_mac,
            ether_type,
            vlan_tag,
        })
    }
    
    fn parse_ipv4_header(&self, data: &[u8]) -> Result<L3Header> {
        if data.len() < 20 {
            return Err(PacketError::InvalidFormat("IPv4 header too short".to_string()));
        }
        
        let version = (data[0] & 0xF0) >> 4;
        if version != 4 {
            return Err(PacketError::InvalidFormat(format!("Invalid IP version: {}", version)));
        }
        
        let ihl = (data[0] & 0x0F) as usize * 4;
        if data.len() < ihl {
            return Err(PacketError::InvalidFormat("IPv4 header truncated".to_string()));
        }
        
        let tos = data[1];
        let total_length = u16::from_be_bytes([data[2], data[3]]);
        let id = u16::from_be_bytes([data[4], data[5]]);
        let flags_and_offset = u16::from_be_bytes([data[6], data[7]]);
        let ttl = data[8];
        let protocol = data[9];
        
        let src_ip = IpAddr::V4(Ipv4Addr::new(data[12], data[13], data[14], data[15]));
        let dst_ip = IpAddr::V4(Ipv4Addr::new(data[16], data[17], data[18], data[19]));
        
        let fragment_info = if (flags_and_offset & 0x3FFF) != 0 {
            Some(FragmentInfo {
                id,
                flags: ((flags_and_offset & 0xE000) >> 13) as u8,
                fragment_offset: flags_and_offset & 0x1FFF,
            })
        } else {
            None
        };
        
        Ok(L3Header {
            src_ip,
            dst_ip,
            protocol,
            ttl,
            tos,
            length: total_length,
            fragment_info,
        })
    }
    
    fn parse_ipv6_header(&self, data: &[u8]) -> Result<L3Header> {
        if data.len() < 40 {
            return Err(PacketError::InvalidFormat("IPv6 header too short".to_string()));
        }
        
        let version = (data[0] & 0xF0) >> 4;
        if version != 6 {
            return Err(PacketError::InvalidFormat(format!("Invalid IP version: {}", version)));
        }
        
        let traffic_class = ((data[0] & 0x0F) << 4) | ((data[1] & 0xF0) >> 4);
        let payload_length = u16::from_be_bytes([data[4], data[5]]);
        let next_header = data[6];
        let hop_limit = data[7];
        
        let mut src_addr = [0u8; 16];
        let mut dst_addr = [0u8; 16];
        src_addr.copy_from_slice(&data[8..24]);
        dst_addr.copy_from_slice(&data[24..40]);
        
        let src_ip = IpAddr::V6(Ipv6Addr::from(src_addr));
        let dst_ip = IpAddr::V6(Ipv6Addr::from(dst_addr));
        
        Ok(L3Header {
            src_ip,
            dst_ip,
            protocol: next_header,
            ttl: hop_limit,
            tos: traffic_class,
            length: payload_length,
            fragment_info: None, // IPv6 fragmentation is handled differently
        })
    }
    
    fn parse_tcp_header(&self, data: &[u8]) -> Result<L4Header> {
        if data.len() < 20 {
            return Err(PacketError::InvalidFormat("TCP header too short".to_string()));
        }
        
        let src_port = u16::from_be_bytes([data[0], data[1]]);
        let dst_port = u16::from_be_bytes([data[2], data[3]]);
        let seq_num = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
        let ack_num = u32::from_be_bytes([data[8], data[9], data[10], data[11]]);
        let data_offset = ((data[12] & 0xF0) >> 4) as usize * 4;
        let flags = u16::from_be_bytes([data[12], data[13]]) & 0x1FF;
        let window = u16::from_be_bytes([data[14], data[15]]);
        
        let options = if data_offset > 20 && data.len() >= data_offset {
            Some(data[20..data_offset].to_vec())
        } else {
            None
        };
        
        Ok(L4Header::TCP {
            src_port,
            dst_port,
            seq_num,
            ack_num,
            flags,
            window,
            options,
        })
    }
    
    fn parse_udp_header(&self, data: &[u8]) -> Result<L4Header> {
        if data.len() < 8 {
            return Err(PacketError::InvalidFormat("UDP header too short".to_string()));
        }
        
        let src_port = u16::from_be_bytes([data[0], data[1]]);
        let dst_port = u16::from_be_bytes([data[2], data[3]]);
        let length = u16::from_be_bytes([data[4], data[5]]);
        let checksum = u16::from_be_bytes([data[6], data[7]]);
        
        Ok(L4Header::UDP {
            src_port,
            dst_port,
            length,
            checksum,
        })
    }
    
    fn parse_icmp_header(&self, data: &[u8]) -> Result<L4Header> {
        if data.len() < 8 {
            return Err(PacketError::InvalidFormat("ICMP header too short".to_string()));
        }
        
        let icmp_type = data[0];
        let code = data[1];
        let checksum = u16::from_be_bytes([data[2], data[3]]);
        let data = data[4..].to_vec();
        
        Ok(L4Header::ICMP {
            icmp_type,
            code,
            checksum,
            data,
        })
    }
    
    fn calculate_flow_hash(&mut self) {
        // Simple 5-tuple hash for RSS
        if let (Some(l3), Some(l4)) = (&self.metadata.l3_header, &self.metadata.l4_header) {
            let mut hash_input = Vec::new();
            
            match l3.src_ip {
                IpAddr::V4(addr) => hash_input.extend_from_slice(&addr.octets()),
                IpAddr::V6(addr) => hash_input.extend_from_slice(&addr.octets()),
            }
            
            match l3.dst_ip {
                IpAddr::V4(addr) => hash_input.extend_from_slice(&addr.octets()),
                IpAddr::V6(addr) => hash_input.extend_from_slice(&addr.octets()),
            }
            
            hash_input.push(l3.protocol);
            
            match l4 {
                L4Header::TCP { src_port, dst_port, .. } => {
                    hash_input.extend_from_slice(&src_port.to_be_bytes());
                    hash_input.extend_from_slice(&dst_port.to_be_bytes());
                },
                L4Header::UDP { src_port, dst_port, .. } => {
                    hash_input.extend_from_slice(&src_port.to_be_bytes());
                    hash_input.extend_from_slice(&dst_port.to_be_bytes());
                },
                _ => {}
            }
            
            // Simple hash calculation (in production, use hardware RSS or jhash)
            let mut hash = 0u32;
            for &byte in &hash_input {
                hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
            }
            
            self.metadata.flow_hash = Some(hash);
        }
    }
    
    /// Clone packet data (not zero-copy)
    pub fn clone_data(&self) -> Bytes {
        Bytes::copy_from_slice(self.data())
    }
    
    /// Get packet timestamp
    pub fn timestamp(&self) -> u64 {
        self.metadata.timestamp
    }
    
    /// Get packet metadata
    pub fn metadata(&self) -> &PacketMetadata {
        &self.metadata
    }
    
    /// Get mutable packet metadata
    pub fn metadata_mut(&mut self) -> &mut PacketMetadata {
        &mut self.metadata
    }
}

impl PacketMetadata {
    pub fn new() -> Self {
        Self {
            timestamp: 0,
            port_id: 0,
            queue_id: 0,
            packet_type: PacketType::Unknown(0),
            l2_header: None,
            l3_header: None,
            l4_header: None,
            flow_hash: None,
            vlan_tci: None,
            tunnel_info: None,
            classification: PacketClassification::new(),
            qos_priority: 0,
            security_flags: SecurityFlags::default(),
        }
    }
}

impl Default for PacketMetadata {
    fn default() -> Self {
        Self::new()
    }
}

impl PacketClassification {
    pub fn new() -> Self {
        Self {
            application_protocol: None,
            traffic_class: TrafficClass::Unknown,
            flow_direction: FlowDirection::Internal,
            confidence: 0.0,
            signatures: Vec::new(),
        }
    }
}

impl Default for PacketClassification {
    fn default() -> Self {
        Self::new()
    }
}

impl PacketType {
    pub fn from_ether_type(ether_type: u16) -> Self {
        match ether_type {
            0x0800 => PacketType::IPv4,
            0x86DD => PacketType::IPv6,
            0x0806 => PacketType::ARP,
            0x88CC => PacketType::LLDP,
            other => PacketType::Unknown(other),
        }
    }
}

impl Drop for PacketBuffer {
    fn drop(&mut self) {
        if !self.mbuf.is_null() {
            self.pool.free_mbuf(self.mbuf);
        }
    }
}

impl fmt::Debug for PacketBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PacketBuffer")
            .field("data_len", &self.data_len)
            .field("buffer_len", &self.buffer_len)
            .field("metadata", &self.metadata)
            .finish()
    }
}

/// Default packet processor implementation
pub struct DefaultPacketProcessor {
    stats: parking_lot::RwLock<ProcessorStats>,
}

impl DefaultPacketProcessor {
    pub fn new() -> Self {
        Self {
            stats: parking_lot::RwLock::new(ProcessorStats::default()),
        }
    }
}

#[async_trait]
impl PacketProcessor for DefaultPacketProcessor {
    async fn process_packet(&self, packet: &mut PacketBuffer) -> Result<ProcessingDecision> {
        let start_time = std::time::Instant::now();
        
        // Parse packet if not already done
        if packet.metadata.l2_header.is_none() {
            packet.parse_headers()?;
        }
        
        // Basic forwarding decision
        let decision = match &packet.metadata.packet_type {
            PacketType::IPv4 | PacketType::IPv6 => {
                ProcessingDecision::Forward { next_hop: None, port: None }
            },
            PacketType::ARP => {
                ProcessingDecision::Forward { next_hop: None, port: None }
            },
            PacketType::Unknown(_) => {
                ProcessingDecision::Drop { reason: "Unknown protocol".to_string() }
            },
            _ => {
                ProcessingDecision::Forward { next_hop: None, port: None }
            }
        };
        
        // Update statistics
        let processing_time = start_time.elapsed().as_nanos() as u64;
        let mut stats = self.stats.write();
        stats.packets_processed += 1;
        
        match &decision {
            ProcessingDecision::Forward { .. } => stats.packets_forwarded += 1,
            ProcessingDecision::Drop { .. } => stats.packets_dropped += 1,
            ProcessingDecision::Redirect { .. } => stats.packets_redirected += 1,
            _ => {}
        }
        
        // Update average processing time
        if stats.packets_processed == 1 {
            stats.average_processing_time_ns = processing_time;
        } else {
            stats.average_processing_time_ns = 
                (stats.average_processing_time_ns * (stats.packets_processed - 1) + processing_time) / stats.packets_processed;
        }
        
        Ok(decision)
    }
    
    async fn process_batch(&self, packets: &mut [PacketBuffer]) -> Result<Vec<ProcessingDecision>> {
        let mut decisions = Vec::with_capacity(packets.len());
        
        for packet in packets.iter_mut() {
            let decision = self.process_packet(packet).await?;
            decisions.push(decision);
        }
        
        Ok(decisions)
    }
    
    fn get_stats(&self) -> ProcessorStats {
        self.stats.read().clone()
    }
    
    async fn reset(&self) {
        let mut stats = self.stats.write();
        *stats = ProcessorStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    
    // Mock memory pool for testing
    struct MockMemoryPool;
    
    impl MockMemoryPool {
        fn new() -> Arc<Self> {
            Arc::new(Self)
        }
        
        fn allocate_mbuf(&self) -> Result<*mut rte_mbuf> {
            // This would need actual DPDK initialization in real tests
            Err(PacketError::MemoryAllocationFailed)
        }
        
        fn free_mbuf(&self, _mbuf: *mut rte_mbuf) {
            // Mock implementation
        }
    }
    
    #[test]
    fn test_packet_type_from_ether_type() {
        assert_eq!(PacketType::from_ether_type(0x0800), PacketType::IPv4);
        assert_eq!(PacketType::from_ether_type(0x86DD), PacketType::IPv6);
        assert_eq!(PacketType::from_ether_type(0x0806), PacketType::ARP);
        assert_eq!(PacketType::from_ether_type(0x1234), PacketType::Unknown(0x1234));
    }
    
    #[test]
    fn test_vlan_tag_parsing() {
        let tci = 0x8100; // PCP=4, DEI=0, VID=256
        let vlan_tag = VlanTag {
            tci,
            vid: tci & 0x0FFF,
            pcp: ((tci & 0xE000) >> 13) as u8,
            dei: (tci & 0x1000) != 0,
        };
        
        assert_eq!(vlan_tag.vid, 256);
        assert_eq!(vlan_tag.pcp, 4);
        assert_eq!(vlan_tag.dei, false);
    }
    
    #[tokio::test]
    async fn test_default_packet_processor() {
        let processor = DefaultPacketProcessor::new();
        
        // Test initial stats
        let stats = processor.get_stats();
        assert_eq!(stats.packets_processed, 0);
        
        // Test reset
        processor.reset().await;
        let stats = processor.get_stats();
        assert_eq!(stats.packets_processed, 0);
    }
}