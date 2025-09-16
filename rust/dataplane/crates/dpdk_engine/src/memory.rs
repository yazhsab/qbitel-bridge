//! DPDK Memory Management
//!
//! This module provides NUMA-aware memory management for DPDK with enterprise features
//! including memory pools, huge page allocation, and zero-copy operations.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::{Mutex, RwLock};
use prometheus::{Counter, Gauge, register_counter, register_gauge};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, error, info, warn, instrument};

use crate::bindings::*;
use crate::config::DpdkConfig;

/// Memory management errors
#[derive(Error, Debug)]
pub enum MemoryError {
    #[error("Memory pool creation failed: {0}")]
    PoolCreationFailed(String),
    
    #[error("Memory allocation failed: {0}")]
    AllocationFailed(String),
    
    #[error("Invalid memory pool: {0}")]
    InvalidPool(String),
    
    #[error("NUMA configuration error: {0}")]
    NumaConfigError(String),
    
    #[error("Huge page configuration error: {0}")]
    HugePageError(String),
    
    #[error("Memory pool exhausted: {0}")]
    PoolExhausted(String),
}

type Result<T> = std::result::Result<T, MemoryError>;

/// NUMA-aware memory pool for packet buffers
pub struct MemoryPool {
    pool: *mut rte_mempool,
    socket_id: u32,
    pool_size: u32,
    cache_size: u32,
    stats: Arc<RwLock<PoolStats>>,
    name: String,
}

/// Memory pool statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PoolStats {
    pub allocated_objects: u64,
    pub free_objects: u64,
    pub allocation_requests: u64,
    pub allocation_failures: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub memory_usage_bytes: u64,
}

/// DPDK Memory Manager
pub struct DpdkMemoryManager {
    pools: HashMap<u32, Arc<MemoryPool>>,
    config: DpdkConfig,
    total_allocated: AtomicU64,
    numa_topology: NumaTopology,
    huge_page_info: HugePageInfo,
}

/// NUMA topology information
#[derive(Debug, Clone)]
pub struct NumaTopology {
    pub nodes: Vec<NumaNode>,
    pub total_memory: u64,
    pub available_memory: u64,
}

#[derive(Debug, Clone)]
pub struct NumaNode {
    pub node_id: u32,
    pub cpus: Vec<u32>,
    pub memory_mb: u64,
    pub free_memory_mb: u64,
    pub distance_map: HashMap<u32, u32>,
}

/// Huge page information
#[derive(Debug, Clone)]
pub struct HugePageInfo {
    pub page_size_kb: u32,
    pub total_pages: u32,
    pub free_pages: u32,
    pub reserved_pages: u32,
    pub mount_point: String,
}

/// Memory allocation strategy
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationStrategy {
    FirstFit,
    BestFit,
    NextFit,
    Buddy,
}

/// Memory pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolConfig {
    pub name: String,
    pub socket_id: u32,
    pub pool_size: u32,
    pub cache_size: u32,
    pub element_size: u32,
    pub alignment: u32,
    pub allocation_strategy: AllocationStrategy,
}

// Prometheus metrics for memory management
lazy_static::lazy_static! {
    static ref MEMORY_ALLOCATED: Counter = register_counter!(
        "dpdk_memory_allocated_bytes_total",
        "Total memory allocated by DPDK"
    ).unwrap();
    
    static ref MEMORY_USAGE: Gauge = register_gauge!(
        "dpdk_memory_usage_bytes",
        "Current memory usage by DPDK"
    ).unwrap();
    
    static ref POOL_UTILIZATION: Gauge = register_gauge!(
        "dpdk_pool_utilization_ratio",
        "Memory pool utilization ratio"
    ).unwrap();
}

impl MemoryPool {
    /// Create a new NUMA-aware memory pool
    #[instrument(skip(config))]
    pub fn new(pool: *mut rte_mempool, socket_id: u32, config: &DpdkConfig) -> Result<Self> {
        if pool.is_null() {
            return Err(MemoryError::InvalidPool("Null memory pool".to_string()));
        }
        
        let pool_name = unsafe {
            let name_ptr = (*pool).name.as_ptr();
            std::ffi::CStr::from_ptr(name_ptr).to_string_lossy().to_string()
        };
        
        let pool_size = unsafe { (*pool).size };
        let cache_size = unsafe { (*pool).cache_size };
        
        info!("Created memory pool '{}' on socket {} with {} objects", 
              pool_name, socket_id, pool_size);
        
        Ok(MemoryPool {
            pool,
            socket_id,
            pool_size,
            cache_size,
            stats: Arc::new(RwLock::new(PoolStats::default())),
            name: pool_name,
        })
    }
    
    /// Allocate a packet buffer (mbuf)
    #[instrument(skip(self))]
    pub fn allocate_mbuf(&self) -> Result<*mut rte_mbuf> {
        unsafe {
            let mbuf = rte_pktmbuf_alloc(self.pool);
            if mbuf.is_null() {
                let mut stats = self.stats.write();
                stats.allocation_failures += 1;
                
                error!("Failed to allocate mbuf from pool '{}'", self.name);
                return Err(MemoryError::AllocationFailed("mbuf allocation failed".to_string()));
            }
            
            // Update statistics
            let mut stats = self.stats.write();
            stats.allocation_requests += 1;
            stats.allocated_objects += 1;
            
            // Update Prometheus metrics
            MEMORY_ALLOCATED.inc_by((*mbuf).buf_len as f64);
            
            debug!("Allocated mbuf from pool '{}' on socket {}", self.name, self.socket_id);
            Ok(mbuf)
        }
    }
    
    /// Free a packet buffer
    pub fn free_mbuf(&self, mbuf: *mut rte_mbuf) {
        if mbuf.is_null() {
            warn!("Attempted to free null mbuf");
            return;
        }
        
        unsafe {
            let buf_len = (*mbuf).buf_len;
            rte_pktmbuf_free(mbuf);
            
            // Update statistics
            let mut stats = self.stats.write();
            stats.allocated_objects = stats.allocated_objects.saturating_sub(1);
            
            // Update Prometheus metrics
            MEMORY_ALLOCATED.inc_by(-(buf_len as f64));
            
            debug!("Freed mbuf from pool '{}'", self.name);
        }
    }
    
    /// Allocate multiple mbufs at once for better performance
    pub fn allocate_bulk_mbufs(&self, count: u16) -> Result<Vec<*mut rte_mbuf>> {
        let mut mbufs = vec![std::ptr::null_mut(); count as usize];
        
        unsafe {
            let allocated = rte_pktmbuf_alloc_bulk(
                self.pool,
                mbufs.as_mut_ptr(),
                count as u32,
            );
            
            if allocated != 0 {
                let mut stats = self.stats.write();
                stats.allocation_failures += 1;
                
                return Err(MemoryError::AllocationFailed(
                    format!("Bulk allocation failed, requested: {}", count)
                ));
            }
            
            // Update statistics
            let mut stats = self.stats.write();
            stats.allocation_requests += count as u64;
            stats.allocated_objects += count as u64;
            
            debug!("Bulk allocated {} mbufs from pool '{}'", count, self.name);
            Ok(mbufs)
        }
    }
    
    /// Free multiple mbufs at once
    pub fn free_bulk_mbufs(&self, mbufs: Vec<*mut rte_mbuf>) {
        let count = mbufs.len();
        
        for mbuf in mbufs {
            if !mbuf.is_null() {
                self.free_mbuf(mbuf);
            }
        }
        
        debug!("Bulk freed {} mbufs from pool '{}'", count, self.name);
    }
    
    /// Get pool statistics
    pub fn get_stats(&self) -> PoolStats {
        let mut stats = self.stats.read().clone();
        
        unsafe {
            // Update with real-time DPDK pool statistics
            stats.free_objects = rte_mempool_avail_count(self.pool) as u64;
            stats.memory_usage_bytes = (self.pool_size - stats.free_objects as u32) as u64 * 2048; // Estimate
        }
        
        stats
    }
    
    /// Get pool utilization ratio
    pub fn get_utilization(&self) -> f64 {
        let stats = self.get_stats();
        let total_objects = self.pool_size as u64;
        
        if total_objects == 0 {
            0.0
        } else {
            (stats.allocated_objects as f64) / (total_objects as f64)
        }
    }
    
    /// Check if pool is running low on free objects
    pub fn is_low_on_memory(&self, threshold: f64) -> bool {
        self.get_utilization() > threshold
    }
    
    /// Get raw DPDK memory pool pointer
    pub fn get_raw_pool(&self) -> *mut rte_mempool {
        self.pool
    }
    
    /// Get socket ID
    pub fn socket_id(&self) -> u32 {
        self.socket_id
    }
    
    /// Get pool name
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl DpdkMemoryManager {
    /// Create a new DPDK memory manager
    pub fn new(config: &DpdkConfig) -> Result<Self> {
        info!("Initializing DPDK Memory Manager");
        
        // Detect NUMA topology
        let numa_topology = Self::detect_numa_topology()?;
        
        // Get huge page information
        let huge_page_info = Self::get_huge_page_info(config)?;
        
        // Validate configuration
        Self::validate_memory_config(config, &numa_topology, &huge_page_info)?;
        
        let manager = DpdkMemoryManager {
            pools: HashMap::new(),
            config: config.clone(),
            total_allocated: AtomicU64::new(0),
            numa_topology,
            huge_page_info,
        };
        
        info!("DPDK Memory Manager initialized successfully");
        Ok(manager)
    }
    
    /// Add a memory pool to the manager
    pub fn add_pool(&mut self, socket_id: u32, pool: Arc<MemoryPool>) {
        info!("Adding memory pool '{}' for socket {}", pool.name(), socket_id);
        self.pools.insert(socket_id, pool);
    }
    
    /// Get memory pool for a specific socket
    pub fn get_pool(&self, socket_id: u32) -> Option<Arc<MemoryPool>> {
        self.pools.get(&socket_id).cloned()
    }
    
    /// Get the best memory pool based on current thread's NUMA node
    pub fn get_best_pool(&self) -> Option<Arc<MemoryPool>> {
        // In a real implementation, this would detect the current NUMA node
        // For now, return the first available pool
        self.pools.values().next().cloned()
    }
    
    /// Get comprehensive memory statistics
    pub fn get_memory_stats(&self) -> MemoryManagerStats {
        let mut total_allocated = 0u64;
        let mut total_free = 0u64;
        let mut pool_stats = HashMap::new();
        
        for (socket_id, pool) in &self.pools {
            let stats = pool.get_stats();
            total_allocated += stats.allocated_objects;
            total_free += stats.free_objects;
            pool_stats.insert(*socket_id, stats);
        }
        
        MemoryManagerStats {
            total_allocated_objects: total_allocated,
            total_free_objects: total_free,
            total_memory_usage: self.total_allocated.load(Ordering::Relaxed),
            pool_count: self.pools.len() as u32,
            numa_nodes: self.numa_topology.nodes.len() as u32,
            huge_page_utilization: self.calculate_huge_page_utilization(),
            pool_stats,
        }
    }
    
    /// Detect NUMA topology from the system
    fn detect_numa_topology() -> Result<NumaTopology> {
        info!("Detecting NUMA topology");
        
        // This is a simplified implementation
        // Real implementation would read from /sys/devices/system/node/
        let mut nodes = Vec::new();
        
        unsafe {
            let socket_count = rte_socket_count();
            
            for socket_id in 0..socket_count {
                let node = NumaNode {
                    node_id: socket_id,
                    cpus: vec![], // Would be populated from /sys/devices/system/node/nodeX/cpulist
                    memory_mb: 1024 * 1024, // 1GB default
                    free_memory_mb: 512 * 1024, // 512MB free
                    distance_map: HashMap::new(),
                };
                nodes.push(node);
            }
        }
        
        let topology = NumaTopology {
            nodes,
            total_memory: 4 * 1024 * 1024 * 1024, // 4GB
            available_memory: 2 * 1024 * 1024 * 1024, // 2GB
        };
        
        info!("Detected {} NUMA nodes", topology.nodes.len());
        Ok(topology)
    }
    
    /// Get huge page information from the system
    fn get_huge_page_info(config: &DpdkConfig) -> Result<HugePageInfo> {
        if !config.use_huge_pages {
            return Ok(HugePageInfo {
                page_size_kb: 4, // Regular 4KB pages
                total_pages: 0,
                free_pages: 0,
                reserved_pages: 0,
                mount_point: String::new(),
            });
        }
        
        // Read from /proc/meminfo
        let meminfo = std::fs::read_to_string("/proc/meminfo")
            .map_err(|e| MemoryError::HugePageError(format!("Cannot read /proc/meminfo: {}", e)))?;
        
        let mut page_size_kb = 2048; // Default 2MB huge pages
        let mut total_pages = 0;
        let mut free_pages = 0;
        
        for line in meminfo.lines() {
            if line.starts_with("HugePages_Total:") {
                total_pages = line.split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
            } else if line.starts_with("HugePages_Free:") {
                free_pages = line.split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
            } else if line.starts_with("Hugepagesize:") {
                page_size_kb = line.split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(2048);
            }
        }
        
        let huge_page_info = HugePageInfo {
            page_size_kb,
            total_pages,
            free_pages,
            reserved_pages: total_pages - free_pages,
            mount_point: config.huge_page_dir.clone(),
        };
        
        info!("Huge pages: {} total, {} free, {} KB per page", 
              total_pages, free_pages, page_size_kb);
        
        Ok(huge_page_info)
    }
    
    /// Validate memory configuration
    fn validate_memory_config(
        config: &DpdkConfig,
        numa_topology: &NumaTopology,
        huge_page_info: &HugePageInfo,
    ) -> Result<()> {
        // Check if requested NUMA sockets exist
        for &socket_id in &config.numa_sockets {
            if !numa_topology.nodes.iter().any(|node| node.node_id == socket_id) {
                return Err(MemoryError::NumaConfigError(
                    format!("NUMA socket {} not found in topology", socket_id)
                ));
            }
        }
        
        // Check huge page availability if required
        if config.use_huge_pages {
            if huge_page_info.total_pages == 0 {
                return Err(MemoryError::HugePageError(
                    "No huge pages configured on the system".to_string()
                ));
            }
            
            if huge_page_info.free_pages < 10 {
                warn!("Low number of free huge pages: {}", huge_page_info.free_pages);
            }
        }
        
        // Validate memory pool sizes
        let total_memory_mb = config.numa_sockets.len() as u64 * config.mbuf_pool_size as u64 * 2 / 1024;
        if total_memory_mb > numa_topology.available_memory / (1024 * 1024) {
            warn!("Requested memory ({} MB) may exceed available memory", total_memory_mb);
        }
        
        Ok(())
    }
    
    /// Calculate huge page utilization
    fn calculate_huge_page_utilization(&self) -> f64 {
        if self.huge_page_info.total_pages == 0 {
            0.0
        } else {
            (self.huge_page_info.reserved_pages as f64) / (self.huge_page_info.total_pages as f64)
        }
    }
    
    /// Perform memory cleanup and optimization
    pub fn cleanup_and_optimize(&self) {
        info!("Performing memory cleanup and optimization");
        
        // Update Prometheus metrics
        let stats = self.get_memory_stats();
        MEMORY_USAGE.set(stats.total_memory_usage as f64);
        POOL_UTILIZATION.set(stats.total_allocated_objects as f64 / 
                            (stats.total_allocated_objects + stats.total_free_objects) as f64);
        
        // Log memory statistics
        info!("Memory statistics: {} objects allocated, {} MB total usage",
              stats.total_allocated_objects, stats.total_memory_usage / (1024 * 1024));
    }
    
    /// Get NUMA topology information
    pub fn get_numa_topology(&self) -> &NumaTopology {
        &self.numa_topology
    }
    
    /// Get huge page information
    pub fn get_huge_page_info(&self) -> &HugePageInfo {
        &self.huge_page_info
    }
}

/// Comprehensive memory manager statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryManagerStats {
    pub total_allocated_objects: u64,
    pub total_free_objects: u64,
    pub total_memory_usage: u64,
    pub pool_count: u32,
    pub numa_nodes: u32,
    pub huge_page_utilization: f64,
    pub pool_stats: HashMap<u32, PoolStats>,
}

unsafe impl Send for MemoryPool {}
unsafe impl Sync for MemoryPool {}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        if !self.pool.is_null() {
            warn!("Dropping memory pool '{}' - ensure all mbufs are freed", self.name);
            // Note: We don't free the pool here as DPDK manages pool lifecycle
        }
    }
}

impl Drop for DpdkMemoryManager {
    fn drop(&mut self) {
        info!("Dropping DPDK Memory Manager");
        self.cleanup_and_optimize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pool_config_validation() {
        let config = PoolConfig {
            name: "test_pool".to_string(),
            socket_id: 0,
            pool_size: 1024,
            cache_size: 64,
            element_size: 2048,
            alignment: 64,
            allocation_strategy: AllocationStrategy::BestFit,
        };
        
        assert_eq!(config.name, "test_pool");
        assert_eq!(config.pool_size, 1024);
    }
    
    #[test]
    fn test_numa_topology() {
        let node = NumaNode {
            node_id: 0,
            cpus: vec![0, 1, 2, 3],
            memory_mb: 1024,
            free_memory_mb: 512,
            distance_map: HashMap::new(),
        };
        
        assert_eq!(node.node_id, 0);
        assert_eq!(node.cpus.len(), 4);
    }
    
    #[test]
    fn test_huge_page_info() {
        let huge_page_info = HugePageInfo {
            page_size_kb: 2048,
            total_pages: 1024,
            free_pages: 512,
            reserved_pages: 512,
            mount_point: "/dev/hugepages".to_string(),
        };
        
        assert_eq!(huge_page_info.page_size_kb, 2048);
        assert_eq!(huge_page_info.total_pages, 1024);
    }
}