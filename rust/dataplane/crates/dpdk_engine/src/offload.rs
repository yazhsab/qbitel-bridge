//! DPDK Hardware Offload Management
//!
//! This module provides enterprise-grade hardware offload management for DPDK,
//! including RSS, checksum offloading, crypto acceleration, and other hardware features.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::{Mutex, RwLock};
use prometheus::{Counter, Gauge, register_counter, register_gauge};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, error, info, warn, instrument};

use crate::bindings::*;
use crate::packet::PacketBuffer;

/// Hardware offload errors
#[derive(Error, Debug)]
pub enum OffloadError {
    #[error("Offload initialization failed: {0}")]
    InitializationFailed(String),
    
    #[error("Hardware feature not supported: {0}")]
    FeatureNotSupported(String),
    
    #[error("RSS configuration error: {0}")]
    RssConfigError(String),
    
    #[error("Checksum offload error: {0}")]
    ChecksumOffloadError(String),
    
    #[error("Crypto offload error: {0}")]
    CryptoOffloadError(String),
    
    #[error("TSO/LRO configuration error: {0}")]
    TsoLroError(String),
}

type Result<T> = std::result::Result<T, OffloadError>;

/// Hardware offload manager
pub struct HardwareOffloadManager {
    port_id: u16,
    config: OffloadConfig,
    capabilities: HardwareCapabilities,
    rss_manager: Arc<RssManager>,
    checksum_manager: Arc<ChecksumOffloadManager>,
    crypto_manager: Arc<CryptoOffloadManager>,
    tso_lro_manager: Arc<TsoLroManager>,
    stats: Arc<RwLock<OffloadStatistics>>,
}

/// Offload configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffloadConfig {
    pub enable_rss: bool,
    pub enable_rx_checksum_offload: bool,
    pub enable_tx_checksum_offload: bool,
    pub enable_tso: bool,
    pub enable_lro: bool,
    pub enable_crypto_offload: bool,
    pub enable_vlan_offload: bool,
    pub rss_hash_functions: Vec<RssHashFunction>,
    pub rss_queue_count: u16,
    pub tso_max_size: u32,
    pub lro_max_size: u32,
}

impl Default for OffloadConfig {
    fn default() -> Self {
        OffloadConfig {
            enable_rss: true,
            enable_rx_checksum_offload: true,
            enable_tx_checksum_offload: true,
            enable_tso: true,
            enable_lro: true,
            enable_crypto_offload: false, // Usually requires special hardware
            enable_vlan_offload: true,
            rss_hash_functions: vec![
                RssHashFunction::IPv4,
                RssHashFunction::IPv6,
                RssHashFunction::TcpIpv4,
                RssHashFunction::TcpIpv6,
                RssHashFunction::UdpIpv4,
                RssHashFunction::UdpIpv6,
            ],
            rss_queue_count: 4,
            tso_max_size: 65536,
            lro_max_size: 65536,
        }
    }
}

/// Hardware capabilities detected from the NIC
#[derive(Debug, Clone)]
pub struct HardwareCapabilities {
    pub rss_supported: bool,
    pub rx_checksum_supported: bool,
    pub tx_checksum_supported: bool,
    pub tso_supported: bool,
    pub lro_supported: bool,
    pub crypto_supported: bool,
    pub vlan_supported: bool,
    pub jumbo_frame_supported: bool,
    pub scatter_gather_supported: bool,
    pub max_queues: u16,
    pub max_tso_size: u32,
    pub max_lro_size: u32,
    pub supported_hash_functions: Vec<RssHashFunction>,
}

/// RSS (Receive Side Scaling) hash functions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RssHashFunction {
    IPv4,
    IPv6,
    TcpIpv4,
    TcpIpv6,
    UdpIpv4,
    UdpIpv6,
    SctpIpv4,
    SctpIpv6,
    TunnelIpv4,
    TunnelIpv6,
}

/// RSS Manager for load balancing across queues
pub struct RssManager {
    port_id: u16,
    config: RssConfig,
    hash_key: Vec<u8>,
    reta_table: Vec<u16>, // Redirection table
    stats: Arc<RwLock<RssStatistics>>,
}

/// RSS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RssConfig {
    pub hash_functions: Vec<RssHashFunction>,
    pub queue_count: u16,
    pub hash_key_length: usize,
    pub reta_size: usize,
    pub symmetric_hash: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RssStatistics {
    pub total_packets_distributed: u64,
    pub queue_distribution: HashMap<u16, u64>,
    pub hash_collisions: u64,
    pub load_balance_efficiency: f64,
}

/// Checksum offload manager
pub struct ChecksumOffloadManager {
    port_id: u16,
    rx_offload_enabled: bool,
    tx_offload_enabled: bool,
    stats: Arc<RwLock<ChecksumStatistics>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChecksumStatistics {
    pub rx_checksum_good: u64,
    pub rx_checksum_bad: u64,
    pub rx_checksum_none: u64,
    pub tx_checksum_offloaded: u64,
    pub checksum_error_rate: f64,
}

/// Crypto offload manager for hardware encryption/decryption
pub struct CryptoOffloadManager {
    port_id: u16,
    crypto_devices: Vec<CryptoDevice>,
    sessions: HashMap<u32, CryptoSession>,
    stats: Arc<RwLock<CryptoStatistics>>,
}

#[derive(Debug, Clone)]
pub struct CryptoDevice {
    pub device_id: u8,
    pub device_name: String,
    pub supported_algorithms: Vec<CryptoAlgorithm>,
    pub max_sessions: u32,
    pub queue_pairs: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CryptoAlgorithm {
    Aes128Cbc,
    Aes256Cbc,
    Aes128Gcm,
    Aes256Gcm,
    ChaCha20Poly1305,
    Sha256,
    Sha384,
    Sha512,
}

#[derive(Debug, Clone)]
pub struct CryptoSession {
    pub session_id: u32,
    pub algorithm: CryptoAlgorithm,
    pub key: Vec<u8>,
    pub direction: CryptoDirection,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CryptoDirection {
    Encrypt,
    Decrypt,
    Auth,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CryptoStatistics {
    pub packets_encrypted: u64,
    pub packets_decrypted: u64,
    pub packets_authenticated: u64,
    pub crypto_errors: u64,
    pub average_crypto_latency_us: f64,
}

/// TSO/LRO manager for TCP segmentation and reassembly offload
pub struct TsoLroManager {
    port_id: u16,
    tso_enabled: bool,
    lro_enabled: bool,
    tso_max_size: u32,
    lro_max_size: u32,
    stats: Arc<RwLock<TsoLroStatistics>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TsoLroStatistics {
    pub tso_packets: u64,
    pub tso_bytes_saved: u64,
    pub lro_packets: u64,
    pub lro_bytes_saved: u64,
    pub tso_efficiency: f64,
    pub lro_efficiency: f64,
}

/// Comprehensive offload statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OffloadStatistics {
    pub rss_stats: RssStatistics,
    pub checksum_stats: ChecksumStatistics,
    pub crypto_stats: CryptoStatistics,
    pub tso_lro_stats: TsoLroStatistics,
    pub total_offloaded_operations: u64,
    pub hardware_utilization: f64,
}

// Prometheus metrics for hardware offloads
lazy_static::lazy_static! {
    static ref OFFLOAD_OPERATIONS: Counter = register_counter!(
        "dpdk_offload_operations_total",
        "Total number of hardware offload operations"
    ).unwrap();
    
    static ref HARDWARE_UTILIZATION: Gauge = register_gauge!(
        "dpdk_hardware_utilization_ratio",
        "Hardware offload utilization ratio"
    ).unwrap();
    
    static ref RSS_EFFICIENCY: Gauge = register_gauge!(
        "dpdk_rss_load_balance_efficiency",
        "RSS load balancing efficiency"
    ).unwrap();
}

impl HardwareOffloadManager {
    /// Create a new hardware offload manager
    #[instrument(skip(config))]
    pub fn new(port_id: u16, config: OffloadConfig) -> Result<Self> {
        info!("Initializing Hardware Offload Manager for port {}", port_id);
        
        // Detect hardware capabilities
        let capabilities = Self::detect_capabilities(port_id)?;
        
        info!("Hardware capabilities: RSS={}, Checksum={}, TSO={}, Crypto={}",
              capabilities.rss_supported,
              capabilities.rx_checksum_supported,
              capabilities.tso_supported,
              capabilities.crypto_supported);
        
        // Initialize RSS manager
        let rss_manager = Arc::new(RssManager::new(port_id, &config, &capabilities)?);
        
        // Initialize checksum offload manager
        let checksum_manager = Arc::new(ChecksumOffloadManager::new(port_id, &config, &capabilities)?);
        
        // Initialize crypto offload manager
        let crypto_manager = Arc::new(CryptoOffloadManager::new(port_id, &config, &capabilities)?);
        
        // Initialize TSO/LRO manager
        let tso_lro_manager = Arc::new(TsoLroManager::new(port_id, &config, &capabilities)?);
        
        let manager = HardwareOffloadManager {
            port_id,
            config,
            capabilities,
            rss_manager,
            checksum_manager,
            crypto_manager,
            tso_lro_manager,
            stats: Arc::new(RwLock::new(OffloadStatistics::default())),
        };
        
        // Configure hardware offloads
        manager.configure_hardware_offloads()?;
        
        info!("Hardware Offload Manager initialized successfully");
        Ok(manager)
    }
    
    /// Configure hardware offloads on the NIC
    #[instrument(skip(self))]
    fn configure_hardware_offloads(&self) -> Result<()> {
        unsafe {
            // Get current device configuration
            let mut dev_conf = std::mem::MaybeUninit::<rte_eth_conf>::uninit();
            let result = rte_eth_dev_configure(self.port_id, 1, 1, dev_conf.as_mut_ptr());
            
            if result != 0 {
                return Err(OffloadError::InitializationFailed(
                    format!("Failed to get device configuration for port {}", self.port_id)
                ));
            }
            
            let mut dev_conf = dev_conf.assume_init();
            
            // Configure RSS
            if self.config.enable_rss && self.capabilities.rss_supported {
                dev_conf.rxmode.mq_mode = ETH_MQ_RX_RSS;
                dev_conf.rx_adv_conf.rss_conf.rss_hf = self.get_rss_hash_functions();
                info!("RSS enabled with hash functions: 0x{:x}", dev_conf.rx_adv_conf.rss_conf.rss_hf);
            }
            
            // Configure RX checksum offload
            if self.config.enable_rx_checksum_offload && self.capabilities.rx_checksum_supported {
                dev_conf.rxmode.offloads |= DEV_RX_OFFLOAD_CHECKSUM as u64;
                info!("RX checksum offload enabled");
            }
            
            // Configure TX checksum offload
            if self.config.enable_tx_checksum_offload && self.capabilities.tx_checksum_supported {
                dev_conf.txmode.offloads |= (DEV_TX_OFFLOAD_IPV4_CKSUM | DEV_TX_OFFLOAD_TCP_CKSUM | DEV_TX_OFFLOAD_UDP_CKSUM) as u64;
                info!("TX checksum offload enabled");
            }
            
            // Configure TSO
            if self.config.enable_tso && self.capabilities.tso_supported {
                dev_conf.txmode.offloads |= DEV_TX_OFFLOAD_TCP_TSO as u64;
                info!("TSO enabled with max size: {}", self.config.tso_max_size);
            }
            
            // Configure LRO
            if self.config.enable_lro && self.capabilities.lro_supported {
                dev_conf.rxmode.offloads |= DEV_RX_OFFLOAD_TCP_LRO as u64;
                info!("LRO enabled with max size: {}", self.config.lro_max_size);
            }
            
            // Configure VLAN offload
            if self.config.enable_vlan_offload && self.capabilities.vlan_supported {
                dev_conf.rxmode.offloads |= (DEV_RX_OFFLOAD_VLAN_STRIP | DEV_RX_OFFLOAD_VLAN_FILTER) as u64;
                dev_conf.txmode.offloads |= DEV_TX_OFFLOAD_VLAN_INSERT as u64;
                info!("VLAN offload enabled");
            }
            
            // Apply configuration
            let result = rte_eth_dev_configure(
                self.port_id,
                self.config.rss_queue_count,
                1, // One TX queue for simplicity
                &dev_conf
            );
            
            if result != 0 {
                return Err(OffloadError::InitializationFailed(
                    format!("Failed to configure offloads for port {}: error {}", self.port_id, result)
                ));
            }
        }
        
        info!("Hardware offloads configured successfully for port {}", self.port_id);
        Ok(())
    }
    
    /// Convert RSS hash functions to DPDK format
    fn get_rss_hash_functions(&self) -> u64 {
        let mut hash_functions = 0u64;
        
        for func in &self.config.rss_hash_functions {
            hash_functions |= match func {
                RssHashFunction::IPv4 => ETH_RSS_IPV4,
                RssHashFunction::IPv6 => ETH_RSS_IPV6,
                RssHashFunction::TcpIpv4 => ETH_RSS_NONFRAG_IPV4_TCP,
                RssHashFunction::TcpIpv6 => ETH_RSS_NONFRAG_IPV6_TCP,
                RssHashFunction::UdpIpv4 => ETH_RSS_NONFRAG_IPV4_UDP,
                RssHashFunction::UdpIpv6 => ETH_RSS_NONFRAG_IPV6_UDP,
                RssHashFunction::SctpIpv4 => ETH_RSS_NONFRAG_IPV4_SCTP,
                RssHashFunction::SctpIpv6 => ETH_RSS_NONFRAG_IPV6_SCTP,
                _ => 0, // Tunnel functions would need additional mapping
            } as u64;
        }
        
        hash_functions
    }
    
    /// Detect hardware capabilities from the NIC
    fn detect_capabilities(port_id: u16) -> Result<HardwareCapabilities> {
        unsafe {
            let mut dev_info = std::mem::MaybeUninit::<rte_eth_dev_info>::uninit();
            let result = rte_eth_dev_info_get(port_id, dev_info.as_mut_ptr());
            
            if result != 0 {
                return Err(OffloadError::InitializationFailed(
                    format!("Failed to get device info for port {}", port_id)
                ));
            }
            
            let dev_info = dev_info.assume_init();
            
            let capabilities = HardwareCapabilities {
                rss_supported: (dev_info.flow_type_rss_offloads & ETH_RSS_IP as u64) != 0,
                rx_checksum_supported: (dev_info.rx_offload_capa & DEV_RX_OFFLOAD_CHECKSUM as u64) != 0,
                tx_checksum_supported: (dev_info.tx_offload_capa & DEV_TX_OFFLOAD_IPV4_CKSUM as u64) != 0,
                tso_supported: (dev_info.tx_offload_capa & DEV_TX_OFFLOAD_TCP_TSO as u64) != 0,
                lro_supported: (dev_info.rx_offload_capa & DEV_RX_OFFLOAD_TCP_LRO as u64) != 0,
                crypto_supported: false, // Would require crypto device detection
                vlan_supported: (dev_info.rx_offload_capa & DEV_RX_OFFLOAD_VLAN_STRIP as u64) != 0,
                jumbo_frame_supported: (dev_info.rx_offload_capa & DEV_RX_OFFLOAD_JUMBO_FRAME as u64) != 0,
                scatter_gather_supported: (dev_info.tx_offload_capa & DEV_TX_OFFLOAD_MULTI_SEGS as u64) != 0,
                max_queues: dev_info.max_rx_queues.min(dev_info.max_tx_queues),
                max_tso_size: dev_info.tx_desc_lim.nb_seg_max * dev_info.tx_desc_lim.nb_mtu_seg_max,
                max_lro_size: 65536, // Default value
                supported_hash_functions: vec![
                    RssHashFunction::IPv4,
                    RssHashFunction::IPv6,
                    RssHashFunction::TcpIpv4,
                    RssHashFunction::TcpIpv6,
                    RssHashFunction::UdpIpv4,
                    RssHashFunction::UdpIpv6,
                ],
            };
            
            info!("Detected capabilities for port {}: RSS={}, Checksum={}, TSO={}", 
                  port_id, capabilities.rss_supported, capabilities.rx_checksum_supported, capabilities.tso_supported);
            
            Ok(capabilities)
        }
    }
    
    /// Process packet with hardware offloads
    pub fn process_packet_offloads(&self, packet: &mut PacketBuffer) -> Result<()> {
        let mut stats = self.stats.write();
        
        // RSS processing
        if self.config.enable_rss {
            self.rss_manager.process_packet(packet)?;
        }
        
        // Checksum processing
        if self.config.enable_rx_checksum_offload || self.config.enable_tx_checksum_offload {
            self.checksum_manager.process_packet(packet)?;
        }
        
        // TSO/LRO processing
        if self.config.enable_tso || self.config.enable_lro {
            self.tso_lro_manager.process_packet(packet)?;
        }
        
        stats.total_offloaded_operations += 1;
        OFFLOAD_OPERATIONS.inc();
        
        Ok(())
    }
    
    /// Get comprehensive offload statistics
    pub fn get_offload_statistics(&self) -> OffloadStatistics {
        let mut stats = self.stats.read().clone();
        
        stats.rss_stats = self.rss_manager.get_statistics();
        stats.checksum_stats = self.checksum_manager.get_statistics();
        stats.crypto_stats = self.crypto_manager.get_statistics();
        stats.tso_lro_stats = self.tso_lro_manager.get_statistics();
        
        // Calculate hardware utilization
        let total_operations = stats.rss_stats.total_packets_distributed +
                              stats.checksum_stats.rx_checksum_good +
                              stats.checksum_stats.tx_checksum_offloaded +
                              stats.tso_lro_stats.tso_packets +
                              stats.tso_lro_stats.lro_packets;
        
        if stats.total_offloaded_operations > 0 {
            stats.hardware_utilization = (total_operations as f64) / (stats.total_offloaded_operations as f64);
        }
        
        HARDWARE_UTILIZATION.set(stats.hardware_utilization);
        RSS_EFFICIENCY.set(stats.rss_stats.load_balance_efficiency);
        
        stats
    }
    
    /// Get hardware capabilities
    pub fn get_capabilities(&self) -> &HardwareCapabilities {
        &self.capabilities
    }
    
    /// Update RSS redirection table
    pub fn update_rss_reta(&self, reta_table: Vec<u16>) -> Result<()> {
        self.rss_manager.update_reta_table(reta_table)
    }
    
    /// Create crypto session for hardware encryption/decryption
    pub fn create_crypto_session(&self, algorithm: CryptoAlgorithm, key: Vec<u8>, direction: CryptoDirection) -> Result<u32> {
        self.crypto_manager.create_session(algorithm, key, direction)
    }
    
    /// Destroy crypto session
    pub fn destroy_crypto_session(&self, session_id: u32) -> Result<()> {
        self.crypto_manager.destroy_session(session_id)
    }
}

impl RssManager {
    fn new(port_id: u16, config: &OffloadConfig, capabilities: &HardwareCapabilities) -> Result<Self> {
        if !config.enable_rss || !capabilities.rss_supported {
            info!("RSS not enabled or supported for port {}", port_id);
            return Ok(RssManager {
                port_id,
                config: RssConfig {
                    hash_functions: Vec::new(),
                    queue_count: 1,
                    hash_key_length: 0,
                    reta_size: 0,
                    symmetric_hash: false,
                },
                hash_key: Vec::new(),
                reta_table: Vec::new(),
                stats: Arc::new(RwLock::new(RssStatistics::default())),
            });
        }
        
        let rss_config = RssConfig {
            hash_functions: config.rss_hash_functions.clone(),
            queue_count: config.rss_queue_count.min(capabilities.max_queues),
            hash_key_length: 52, // Standard RSS key length
            reta_size: 128,     // Standard RETA size
            symmetric_hash: false,
        };
        
        // Generate RSS hash key
        let hash_key = Self::generate_rss_key(rss_config.hash_key_length);
        
        // Initialize RETA table for load balancing
        let reta_table = Self::generate_reta_table(rss_config.reta_size, rss_config.queue_count);
        
        info!("RSS configured for port {} with {} queues", port_id, rss_config.queue_count);
        
        Ok(RssManager {
            port_id,
            config: rss_config,
            hash_key,
            reta_table,
            stats: Arc::new(RwLock::new(RssStatistics::default())),
        })
    }
    
    fn generate_rss_key(length: usize) -> Vec<u8> {
        // Default RSS key from Intel DPDK documentation
        vec![
            0x6d, 0x5a, 0x6d, 0x5a, 0x6d, 0x5a, 0x6d, 0x5a,
            0x6d, 0x5a, 0x6d, 0x5a, 0x6d, 0x5a, 0x6d, 0x5a,
            0x6d, 0x5a, 0x6d, 0x5a, 0x6d, 0x5a, 0x6d, 0x5a,
            0x6d, 0x5a, 0x6d, 0x5a, 0x6d, 0x5a, 0x6d, 0x5a,
            0x6d, 0x5a, 0x6d, 0x5a, 0x6d, 0x5a, 0x6d, 0x5a,
            0x6d, 0x5a, 0x6d, 0x5a, 0x6d, 0x5a, 0x6d, 0x5a,
            0x6d, 0x5a, 0x6d, 0x5a,
        ][..length].to_vec()
    }
    
    fn generate_reta_table(reta_size: usize, queue_count: u16) -> Vec<u16> {
        (0..reta_size).map(|i| (i as u16) % queue_count).collect()
    }
    
    fn process_packet(&self, _packet: &PacketBuffer) -> Result<()> {
        let mut stats = self.stats.write();
        stats.total_packets_distributed += 1;
        
        // In a real implementation, this would examine the RSS hash
        // and update queue distribution statistics
        
        Ok(())
    }
    
    fn update_reta_table(&self, _reta_table: Vec<u16>) -> Result<()> {
        // Would update the hardware RETA table
        Ok(())
    }
    
    fn get_statistics(&self) -> RssStatistics {
        self.stats.read().clone()
    }
}

impl ChecksumOffloadManager {
    fn new(port_id: u16, config: &OffloadConfig, capabilities: &HardwareCapabilities) -> Result<Self> {
        let rx_enabled = config.enable_rx_checksum_offload && capabilities.rx_checksum_supported;
        let tx_enabled = config.enable_tx_checksum_offload && capabilities.tx_checksum_supported;
        
        info!("Checksum offload for port {}: RX={}, TX={}", port_id, rx_enabled, tx_enabled);
        
        Ok(ChecksumOffloadManager {
            port_id,
            rx_offload_enabled: rx_enabled,
            tx_offload_enabled: tx_enabled,
            stats: Arc::new(RwLock::new(ChecksumStatistics::default())),
        })
    }
    
    fn process_packet(&self, _packet: &PacketBuffer) -> Result<()> {
        let mut stats = self.stats.write();
        
        // In a real implementation, this would check packet mbuf flags
        // and update checksum statistics accordingly
        
        if self.rx_offload_enabled {
            stats.rx_checksum_good += 1;
        }
        
        if self.tx_offload_enabled {
            stats.tx_checksum_offloaded += 1;
        }
        
        Ok(())
    }
    
    fn get_statistics(&self) -> ChecksumStatistics {
        self.stats.read().clone()
    }
}

impl CryptoOffloadManager {
    fn new(port_id: u16, config: &OffloadConfig, capabilities: &HardwareCapabilities) -> Result<Self> {
        let manager = CryptoOffloadManager {
            port_id,
            crypto_devices: Vec::new(),
            sessions: HashMap::new(),
            stats: Arc::new(RwLock::new(CryptoStatistics::default())),
        };
        
        if config.enable_crypto_offload && capabilities.crypto_supported {
            info!("Crypto offload enabled for port {}", port_id);
            // Would initialize crypto devices here
        }
        
        Ok(manager)
    }
    
    fn create_session(&self, algorithm: CryptoAlgorithm, key: Vec<u8>, direction: CryptoDirection) -> Result<u32> {
        // Would create actual crypto session with hardware
        let session_id = rand::random::<u32>();
        info!("Created crypto session {} with algorithm {:?}", session_id, algorithm);
        Ok(session_id)
    }
    
    fn destroy_session(&self, session_id: u32) -> Result<()> {
        info!("Destroyed crypto session {}", session_id);
        Ok(())
    }
    
    fn get_statistics(&self) -> CryptoStatistics {
        self.stats.read().clone()
    }
}

impl TsoLroManager {
    fn new(port_id: u16, config: &OffloadConfig, capabilities: &HardwareCapabilities) -> Result<Self> {
        let tso_enabled = config.enable_tso && capabilities.tso_supported;
        let lro_enabled = config.enable_lro && capabilities.lro_supported;
        
        info!("TSO/LRO for port {}: TSO={}, LRO={}", port_id, tso_enabled, lro_enabled);
        
        Ok(TsoLroManager {
            port_id,
            tso_enabled,
            lro_enabled,
            tso_max_size: config.tso_max_size.min(capabilities.max_tso_size),
            lro_max_size: config.lro_max_size.min(capabilities.max_lro_size),
            stats: Arc::new(RwLock::new(TsoLroStatistics::default())),
        })
    }
    
    fn process_packet(&self, _packet: &PacketBuffer) -> Result<()> {
        let mut stats = self.stats.write();
        
        // In a real implementation, this would check for TSO/LRO opportunities
        // and update statistics accordingly
        
        if self.tso_enabled {
            stats.tso_packets += 1;
        }
        
        if self.lro_enabled {
            stats.lro_packets += 1;
        }
        
        Ok(())
    }
    
    fn get_statistics(&self) -> TsoLroStatistics {
        self.stats.read().clone()
    }
}

unsafe impl Send for HardwareOffloadManager {}
unsafe impl Sync for HardwareOffloadManager {}

impl Drop for HardwareOffloadManager {
    fn drop(&mut self) {
        info!("Dropping Hardware Offload Manager for port {}", self.port_id);
        
        // Log final statistics
        let stats = self.get_offload_statistics();
        info!("Final offload statistics: {} operations, {:.2}% hardware utilization",
              stats.total_offloaded_operations, stats.hardware_utilization * 100.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_offload_config() {
        let config = OffloadConfig::default();
        assert!(config.enable_rss);
        assert!(config.enable_rx_checksum_offload);
        assert_eq!(config.rss_queue_count, 4);
    }
    
    #[test]
    fn test_rss_hash_functions() {
        let functions = vec![
            RssHashFunction::IPv4,
            RssHashFunction::TcpIpv4,
            RssHashFunction::UdpIpv4,
        ];
        
        assert_eq!(functions.len(), 3);
        assert_eq!(functions[0], RssHashFunction::IPv4);
    }
    
    #[test]
    fn test_crypto_algorithms() {
        let algorithms = vec![
            CryptoAlgorithm::Aes128Cbc,
            CryptoAlgorithm::Aes256Gcm,
            CryptoAlgorithm::ChaCha20Poly1305,
        ];
        
        assert_eq!(algorithms.len(), 3);
        assert_eq!(algorithms[1], CryptoAlgorithm::Aes256Gcm);
    }
    
    #[test]
    fn test_hardware_capabilities() {
        let capabilities = HardwareCapabilities {
            rss_supported: true,
            rx_checksum_supported: true,
            tx_checksum_supported: true,
            tso_supported: false,
            lro_supported: false,
            crypto_supported: false,
            vlan_supported: true,
            jumbo_frame_supported: true,
            scatter_gather_supported: false,
            max_queues: 8,
            max_tso_size: 65536,
            max_lro_size: 65536,
            supported_hash_functions: vec![RssHashFunction::IPv4, RssHashFunction::IPv6],
        };
        
        assert!(capabilities.rss_supported);
        assert!(!capabilities.crypto_supported);
        assert_eq!(capabilities.max_queues, 8);
    }
}