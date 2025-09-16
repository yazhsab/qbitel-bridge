//! DPDK Configuration Management
//!
//! This module provides comprehensive configuration management for DPDK engine
//! with validation, serialization, and environment-based configuration loading.

use std::collections::HashSet;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// DPDK Configuration Error
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Invalid core mask: {0}")]
    InvalidCoreMask(String),
    
    #[error("Invalid memory configuration: {0}")]
    InvalidMemoryConfig(String),
    
    #[error("Invalid port configuration: {0}")]
    InvalidPortConfig(String),
    
    #[error("Configuration validation failed: {0}")]
    ValidationFailed(String),
    
    #[error("Environment variable error: {0}")]
    EnvironmentError(String),
}

/// Comprehensive DPDK Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DpdkConfig {
    // EAL Configuration
    pub core_mask: String,
    pub memory_channels: u32,
    pub use_huge_pages: bool,
    pub huge_page_dir: String,
    pub numa_sockets: Vec<u32>,
    
    // PCI Device Configuration
    pub pci_whitelist: Vec<String>,
    pub pci_blacklist: Vec<String>,
    
    // Memory Configuration
    pub mbuf_pool_size: usize,
    pub mbuf_cache_size: usize,
    pub ring_size: usize,
    
    // Worker Configuration
    pub worker_threads: usize,
    pub channel_size: usize,
    pub cpu_affinity: Vec<CpuAffinity>,
    
    // Performance Configuration
    pub flow_table_size: usize,
    pub hash_table_size: usize,
    pub burst_size: usize,
    pub prefetch_offset: usize,
    
    // Network Configuration
    pub ports: Vec<PortConfiguration>,
    pub default_mtu: u16,
    pub enable_jumbo_frames: bool,
    pub max_packet_size: usize,
    
    // Offload Configuration
    pub checksum_offload: bool,
    pub tso_offload: bool,
    pub rss_offload: bool,
    pub flow_director: bool,
    
    // Security Configuration
    pub enable_packet_capture: bool,
    pub capture_buffer_size: usize,
    pub enable_flow_inspection: bool,
    
    // Monitoring Configuration
    pub enable_telemetry: bool,
    pub telemetry_port: u16,
    pub metrics_interval: u64,
    pub enable_debug_stats: bool,
    
    // Quality of Service
    pub enable_qos: bool,
    pub qos_profiles: Vec<QoSProfile>,
    
    // High Availability
    pub enable_failover: bool,
    pub backup_interfaces: Vec<String>,
    pub heartbeat_interval: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuAffinity {
    pub worker_id: usize,
    pub cpu_cores: Vec<u32>,
    pub numa_socket: u32,
    pub isolation_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortConfiguration {
    pub port_id: u16,
    pub pci_address: String,
    pub driver: String,
    pub rx_queues: u16,
    pub tx_queues: u16,
    pub rx_descriptors: u16,
    pub tx_descriptors: u16,
    pub mtu: u16,
    pub promiscuous: bool,
    pub rss_hash_key: Option<Vec<u8>>,
    pub flow_control: FlowControlConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowControlConfig {
    pub enable_rx_pause: bool,
    pub enable_tx_pause: bool,
    pub pause_time: u16,
    pub low_water: u32,
    pub high_water: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QoSProfile {
    pub name: String,
    pub priority: u8,
    pub bandwidth_limit: u64, // Mbps
    pub burst_size: u32,
    pub dscp_marking: u8,
    pub traffic_class: String,
}

impl Default for DpdkConfig {
    fn default() -> Self {
        Self {
            core_mask: "0xf".to_string(), // Use first 4 cores
            memory_channels: 4,
            use_huge_pages: true,
            huge_page_dir: "/dev/hugepages".to_string(),
            numa_sockets: vec![0],
            
            pci_whitelist: Vec::new(),
            pci_blacklist: Vec::new(),
            
            mbuf_pool_size: 16384,
            mbuf_cache_size: 256,
            ring_size: 1024,
            
            worker_threads: 4,
            channel_size: 1024,
            cpu_affinity: Vec::new(),
            
            flow_table_size: 65536,
            hash_table_size: 32768,
            burst_size: 32,
            prefetch_offset: 3,
            
            ports: Vec::new(),
            default_mtu: 1500,
            enable_jumbo_frames: false,
            max_packet_size: 9600,
            
            checksum_offload: true,
            tso_offload: true,
            rss_offload: true,
            flow_director: true,
            
            enable_packet_capture: false,
            capture_buffer_size: 1024 * 1024, // 1MB
            enable_flow_inspection: true,
            
            enable_telemetry: true,
            telemetry_port: 9090,
            metrics_interval: 1000, // 1 second
            enable_debug_stats: false,
            
            enable_qos: false,
            qos_profiles: Vec::new(),
            
            enable_failover: false,
            backup_interfaces: Vec::new(),
            heartbeat_interval: 5000, // 5 seconds
        }
    }
}

impl Default for FlowControlConfig {
    fn default() -> Self {
        Self {
            enable_rx_pause: true,
            enable_tx_pause: true,
            pause_time: 0x1337,
            low_water: 0x40,
            high_water: 0x80,
        }
    }
}

impl DpdkConfig {
    /// Load configuration from environment variables and files
    pub fn from_environment() -> Result<Self, ConfigError> {
        let mut config = Self::default();
        
        // Load from environment variables
        if let Ok(core_mask) = std::env::var("DPDK_CORE_MASK") {
            config.core_mask = core_mask;
        }
        
        if let Ok(memory_channels) = std::env::var("DPDK_MEMORY_CHANNELS") {
            config.memory_channels = memory_channels.parse()
                .map_err(|e| ConfigError::EnvironmentError(format!("Invalid DPDK_MEMORY_CHANNELS: {}", e)))?;
        }
        
        if let Ok(huge_page_dir) = std::env::var("DPDK_HUGE_PAGE_DIR") {
            config.huge_page_dir = huge_page_dir;
        }
        
        if let Ok(worker_threads) = std::env::var("DPDK_WORKER_THREADS") {
            config.worker_threads = worker_threads.parse()
                .map_err(|e| ConfigError::EnvironmentError(format!("Invalid DPDK_WORKER_THREADS: {}", e)))?;
        }
        
        if let Ok(flow_table_size) = std::env::var("DPDK_FLOW_TABLE_SIZE") {
            config.flow_table_size = flow_table_size.parse()
                .map_err(|e| ConfigError::EnvironmentError(format!("Invalid DPDK_FLOW_TABLE_SIZE: {}", e)))?;
        }
        
        // Load PCI whitelist
        if let Ok(pci_whitelist) = std::env::var("DPDK_PCI_WHITELIST") {
            config.pci_whitelist = pci_whitelist.split(',')
                .map(|s| s.trim().to_string())
                .collect();
        }
        
        // Load PCI blacklist
        if let Ok(pci_blacklist) = std::env::var("DPDK_PCI_BLACKLIST") {
            config.pci_blacklist = pci_blacklist.split(',')
                .map(|s| s.trim().to_string())
                .collect();
        }
        
        // Validate configuration
        config.validate()?;
        
        Ok(config)
    }
    
    /// Load configuration from a TOML file
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| ConfigError::ValidationFailed(format!("Failed to read config file: {}", e)))?;
        
        let config: DpdkConfig = toml::from_str(&content)
            .map_err(|e| ConfigError::ValidationFailed(format!("Failed to parse config: {}", e)))?;
        
        config.validate()?;
        Ok(config)
    }
    
    /// Validate the configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Validate core mask
        self.validate_core_mask()?;
        
        // Validate memory configuration
        self.validate_memory_config()?;
        
        // Validate port configuration
        self.validate_ports_config()?;
        
        // Validate worker configuration
        self.validate_worker_config()?;
        
        // Validate performance configuration
        self.validate_performance_config()?;
        
        Ok(())
    }
    
    fn validate_core_mask(&self) -> Result<(), ConfigError> {
        if self.core_mask.is_empty() {
            return Err(ConfigError::InvalidCoreMask("Core mask cannot be empty".to_string()));
        }
        
        // Validate hex format
        let core_mask = if self.core_mask.starts_with("0x") {
            &self.core_mask[2..]
        } else {
            &self.core_mask
        };
        
        if core_mask.chars().any(|c| !c.is_ascii_hexdigit()) {
            return Err(ConfigError::InvalidCoreMask(
                "Core mask must be a valid hexadecimal value".to_string()
            ));
        }
        
        // Parse and validate core count
        let mask_value = u64::from_str_radix(core_mask, 16)
            .map_err(|_| ConfigError::InvalidCoreMask("Invalid core mask format".to_string()))?;
        
        let core_count = mask_value.count_ones() as usize;
        if core_count < 2 {
            return Err(ConfigError::InvalidCoreMask(
                "At least 2 cores must be specified".to_string()
            ));
        }
        
        if core_count > 64 {
            return Err(ConfigError::InvalidCoreMask(
                "Core mask supports maximum 64 cores".to_string()
            ));
        }
        
        Ok(())
    }
    
    fn validate_memory_config(&self) -> Result<(), ConfigError> {
        if self.memory_channels == 0 || self.memory_channels > 8 {
            return Err(ConfigError::InvalidMemoryConfig(
                "Memory channels must be between 1 and 8".to_string()
            ));
        }
        
        if self.mbuf_pool_size < 1024 {
            return Err(ConfigError::InvalidMemoryConfig(
                "Mbuf pool size must be at least 1024".to_string()
            ));
        }
        
        if self.mbuf_cache_size > self.mbuf_pool_size / 10 {
            return Err(ConfigError::InvalidMemoryConfig(
                "Mbuf cache size must be less than 10% of pool size".to_string()
            ));
        }
        
        if self.ring_size == 0 || !self.ring_size.is_power_of_two() {
            return Err(ConfigError::InvalidMemoryConfig(
                "Ring size must be a power of 2".to_string()
            ));
        }
        
        Ok(())
    }
    
    fn validate_ports_config(&self) -> Result<(), ConfigError> {
        let mut port_ids = HashSet::new();
        let mut pci_addresses = HashSet::new();
        
        for port in &self.ports {
            if !port_ids.insert(port.port_id) {
                return Err(ConfigError::InvalidPortConfig(
                    format!("Duplicate port ID: {}", port.port_id)
                ));
            }
            
            if !pci_addresses.insert(&port.pci_address) {
                return Err(ConfigError::InvalidPortConfig(
                    format!("Duplicate PCI address: {}", port.pci_address)
                ));
            }
            
            if port.rx_queues == 0 || port.tx_queues == 0 {
                return Err(ConfigError::InvalidPortConfig(
                    "Port must have at least 1 RX and 1 TX queue".to_string()
                ));
            }
            
            if port.rx_descriptors < 64 || port.tx_descriptors < 64 {
                return Err(ConfigError::InvalidPortConfig(
                    "Descriptors must be at least 64".to_string()
                ));
            }
            
            if port.mtu < 64 || port.mtu > 9600 {
                return Err(ConfigError::InvalidPortConfig(
                    "MTU must be between 64 and 9600".to_string()
                ));
            }
        }
        
        Ok(())
    }
    
    fn validate_worker_config(&self) -> Result<(), ConfigError> {
        if self.worker_threads == 0 || self.worker_threads > 128 {
            return Err(ConfigError::ValidationFailed(
                "Worker threads must be between 1 and 128".to_string()
            ));
        }
        
        if self.channel_size < 64 || !self.channel_size.is_power_of_two() {
            return Err(ConfigError::ValidationFailed(
                "Channel size must be a power of 2 and at least 64".to_string()
            ));
        }
        
        // Validate CPU affinity configuration
        for affinity in &self.cpu_affinity {
            if affinity.worker_id >= self.worker_threads {
                return Err(ConfigError::ValidationFailed(
                    format!("Worker ID {} exceeds worker thread count", affinity.worker_id)
                ));
            }
            
            if affinity.cpu_cores.is_empty() {
                return Err(ConfigError::ValidationFailed(
                    "CPU affinity must specify at least one core".to_string()
                ));
            }
        }
        
        Ok(())
    }
    
    fn validate_performance_config(&self) -> Result<(), ConfigError> {
        if self.flow_table_size == 0 || !self.flow_table_size.is_power_of_two() {
            return Err(ConfigError::ValidationFailed(
                "Flow table size must be a power of 2".to_string()
            ));
        }
        
        if self.hash_table_size == 0 || !self.hash_table_size.is_power_of_two() {
            return Err(ConfigError::ValidationFailed(
                "Hash table size must be a power of 2".to_string()
            ));
        }
        
        if self.burst_size == 0 || self.burst_size > 256 {
            return Err(ConfigError::ValidationFailed(
                "Burst size must be between 1 and 256".to_string()
            ));
        }
        
        if self.prefetch_offset > self.burst_size {
            return Err(ConfigError::ValidationFailed(
                "Prefetch offset must not exceed burst size".to_string()
            ));
        }
        
        Ok(())
    }
    
    /// Get the number of CPU cores from core mask
    pub fn get_core_count(&self) -> usize {
        let core_mask = if self.core_mask.starts_with("0x") {
            &self.core_mask[2..]
        } else {
            &self.core_mask
        };
        
        u64::from_str_radix(core_mask, 16)
            .map(|mask| mask.count_ones() as usize)
            .unwrap_or(0)
    }
    
    /// Get list of CPU cores from core mask
    pub fn get_core_list(&self) -> Vec<u32> {
        let core_mask = if self.core_mask.starts_with("0x") {
            &self.core_mask[2..]
        } else {
            &self.core_mask
        };
        
        let mask_value = u64::from_str_radix(core_mask, 16).unwrap_or(0);
        let mut cores = Vec::new();
        
        for i in 0..64 {
            if (mask_value >> i) & 1 == 1 {
                cores.push(i);
            }
        }
        
        cores
    }
    
    /// Check if huge pages are properly configured
    pub fn check_huge_pages(&self) -> Result<(), ConfigError> {
        if !self.use_huge_pages {
            return Ok(());
        }
        
        let huge_page_path = PathBuf::from(&self.huge_page_dir);
        if !huge_page_path.exists() {
            return Err(ConfigError::ValidationFailed(
                format!("Huge page directory does not exist: {}", self.huge_page_dir)
            ));
        }
        
        // Check if huge pages are mounted
        let mount_info = std::fs::read_to_string("/proc/mounts")
            .map_err(|_| ConfigError::ValidationFailed("Cannot read /proc/mounts".to_string()))?;
        
        if !mount_info.contains("hugetlbfs") {
            return Err(ConfigError::ValidationFailed(
                "Huge pages not properly mounted".to_string()
            ));
        }
        
        Ok(())
    }
    
    /// Generate DPDK EAL arguments
    pub fn to_eal_args(&self) -> Vec<String> {
        let mut args = Vec::new();
        
        args.push("cronos-dpdk".to_string());
        
        // Core mask
        args.push("-c".to_string());
        args.push(self.core_mask.clone());
        
        // Memory channels
        args.push("-n".to_string());
        args.push(self.memory_channels.to_string());
        
        // Huge pages
        if self.use_huge_pages {
            args.push("--huge-dir".to_string());
            args.push(self.huge_page_dir.clone());
        }
        
        // PCI whitelist
        for pci in &self.pci_whitelist {
            args.push("-w".to_string());
            args.push(pci.clone());
        }
        
        // PCI blacklist
        for pci in &self.pci_blacklist {
            args.push("-b".to_string());
            args.push(pci.clone());
        }
        
        args
    }
}

/// Configuration builder for easier setup
pub struct DpdkConfigBuilder {
    config: DpdkConfig,
}

impl DpdkConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: DpdkConfig::default(),
        }
    }
    
    pub fn core_mask<S: Into<String>>(mut self, core_mask: S) -> Self {
        self.config.core_mask = core_mask.into();
        self
    }
    
    pub fn memory_channels(mut self, channels: u32) -> Self {
        self.config.memory_channels = channels;
        self
    }
    
    pub fn huge_pages<S: Into<String>>(mut self, dir: S) -> Self {
        self.config.use_huge_pages = true;
        self.config.huge_page_dir = dir.into();
        self
    }
    
    pub fn worker_threads(mut self, threads: usize) -> Self {
        self.config.worker_threads = threads;
        self
    }
    
    pub fn flow_table_size(mut self, size: usize) -> Self {
        self.config.flow_table_size = size;
        self
    }
    
    pub fn add_port(mut self, port: PortConfiguration) -> Self {
        self.config.ports.push(port);
        self
    }
    
    pub fn enable_offloads(mut self) -> Self {
        self.config.checksum_offload = true;
        self.config.tso_offload = true;
        self.config.rss_offload = true;
        self.config.flow_director = true;
        self
    }
    
    pub fn build(self) -> Result<DpdkConfig, ConfigError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for DpdkConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_config_validation() {
        let config = DpdkConfig::default();
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_core_mask_validation() {
        let mut config = DpdkConfig::default();
        
        // Valid core masks
        config.core_mask = "0xf".to_string();
        assert!(config.validate().is_ok());
        
        config.core_mask = "ff".to_string();
        assert!(config.validate().is_ok());
        
        // Invalid core masks
        config.core_mask = "".to_string();
        assert!(config.validate().is_err());
        
        config.core_mask = "0xg".to_string();
        assert!(config.validate().is_err());
        
        config.core_mask = "0x1".to_string(); // Only 1 core
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_get_core_list() {
        let config = DpdkConfig {
            core_mask: "0xf".to_string(), // Cores 0, 1, 2, 3
            ..DpdkConfig::default()
        };
        
        let cores = config.get_core_list();
        assert_eq!(cores, vec![0, 1, 2, 3]);
        
        let core_count = config.get_core_count();
        assert_eq!(core_count, 4);
    }
    
    #[test]
    fn test_config_builder() {
        let config = DpdkConfigBuilder::new()
            .core_mask("0xff")
            .memory_channels(4)
            .worker_threads(8)
            .flow_table_size(32768)
            .enable_offloads()
            .build()
            .unwrap();
        
        assert_eq!(config.core_mask, "0xff");
        assert_eq!(config.memory_channels, 4);
        assert_eq!(config.worker_threads, 8);
        assert_eq!(config.flow_table_size, 32768);
        assert!(config.checksum_offload);
        assert!(config.tso_offload);
    }
}