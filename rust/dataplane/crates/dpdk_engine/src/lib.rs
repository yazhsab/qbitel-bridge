//! QBITEL Bridge DPDK Engine - High-Performance Packet Processing
//!
//! This module provides enterprise-grade DPDK integration for kernel bypass
//! and zero-copy packet processing with advanced features including:
//! - Multi-queue packet processing with RSS
//! - Zero-copy packet forwarding
//! - Hardware offload capabilities
//! - NUMA-aware memory allocation
//! - Advanced flow classification
//! - Real-time packet analytics

use std::collections::HashMap;
use std::ffi::{CString, CStr};
use std::sync::{Arc, atomic::{AtomicBool, AtomicU64, Ordering}};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use bytes::Bytes;
use crossbeam_channel::{bounded, Receiver, Sender};
use once_cell::sync::Lazy;
use parking_lot::{Mutex, RwLock};
use prometheus::{Counter, Histogram, Gauge, IntGauge, register_counter, register_histogram, register_int_gauge};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::time::interval;
use tracing::{debug, error, info, warn, instrument};

pub mod packet;
pub mod flow;
pub mod memory;
pub mod config;
pub mod analytics;
pub mod offload;

use packet::{PacketBuffer, PacketProcessor, PacketClassification};
use flow::{FlowClassifier, FlowPattern, FlowAction};
use memory::{MemoryPool, DpdkMemoryManager};
use analytics::{PacketAnalytics, AnalyticsConfig};
use offload::{HardwareOffloadManager, OffloadConfig};
use config::DpdkConfig;

// Include generated DPDK bindings
#[allow(non_upper_case_globals)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
#[allow(dead_code)]
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

use bindings::*;

/// DPDK Engine Error Types
#[derive(Error, Debug)]
pub enum DpdkError {
    #[error("DPDK initialization failed: {0}")]
    InitializationFailed(String),
    
    #[error("Port configuration failed: {0}")]
    PortConfigFailed(String),
    
    #[error("Memory allocation failed: {0}")]
    MemoryAllocFailed(String),
    
    #[error("Packet processing failed: {0}")]
    PacketProcessingFailed(String),
    
    #[error("Flow table operation failed: {0}")]
    FlowTableError(String),
    
    #[error("Hardware offload error: {0}")]
    HardwareOffloadError(String),
    
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
}

type Result<T> = std::result::Result<T, DpdkError>;

/// Packet Processing Statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PacketStats {
    pub rx_packets: u64,
    pub tx_packets: u64,
    pub rx_bytes: u64,
    pub tx_bytes: u64,
    pub rx_dropped: u64,
    pub tx_dropped: u64,
    pub rx_errors: u64,
    pub tx_errors: u64,
    pub rx_missed: u64,
    pub rx_no_mbuf: u64,
    pub flow_table_hits: u64,
    pub flow_table_misses: u64,
    pub hardware_offloads: u64,
}

/// DPDK Port Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortConfig {
    pub port_id: u16,
    pub rx_queues: u16,
    pub tx_queues: u16,
    pub rx_descriptors: u16,
    pub tx_descriptors: u16,
    pub mtu: u16,
    pub rss_enabled: bool,
    pub checksum_offload: bool,
    pub tso_enabled: bool,
    pub jumbo_frames: bool,
    pub promiscuous: bool,
}

impl Default for PortConfig {
    fn default() -> Self {
        Self {
            port_id: 0,
            rx_queues: 4,
            tx_queues: 4,
            rx_descriptors: 1024,
            tx_descriptors: 1024,
            mtu: 1500,
            rss_enabled: true,
            checksum_offload: true,
            tso_enabled: true,
            jumbo_frames: false,
            promiscuous: false,
        }
    }
}

/// DPDK Lcore Assignment
#[derive(Debug, Clone)]
pub struct LcoreConfig {
    pub lcore_id: u32,
    pub socket_id: u32,
    pub role: LcoreRole,
    pub assigned_queues: Vec<u16>,
}

#[derive(Debug, Clone)]
pub enum LcoreRole {
    Master,
    RxWorker,
    TxWorker,
    Processor,
    Analytics,
}

/// High-Performance DPDK Engine
pub struct DpdkEngine {
    config: DpdkConfig,
    ports: Vec<PortConfig>,
    memory_pools: HashMap<u32, Arc<MemoryPool>>,
    flow_classifier: Arc<FlowClassifier>,
    packet_processor: Arc<dyn PacketProcessor>,
    memory_manager: Arc<DpdkMemoryManager>,
    
    // Worker management
    worker_handles: Vec<tokio::task::JoinHandle<()>>,
    shutdown_signal: Arc<AtomicBool>,
    
    // Performance monitoring
    stats: Arc<RwLock<PacketStats>>,
    analytics: Arc<PacketAnalytics>,
    
    // Communication channels
    rx_channels: Vec<Receiver<PacketBuffer>>,
    tx_channels: Vec<Sender<PacketBuffer>>,
    
    // Hardware offload
    offload_manager: Arc<HardwareOffloadManager>,
}

/// Global DPDK Metrics
static PACKETS_PROCESSED: Lazy<Counter> = Lazy::new(|| {
    register_counter!("dpdk_packets_processed_total", "Total packets processed by DPDK").unwrap()
});

static PACKET_PROCESSING_DURATION: Lazy<Histogram> = Lazy::new(|| {
    register_histogram!("dpdk_packet_processing_duration_seconds", "Packet processing duration").unwrap()
});

static ACTIVE_FLOWS: Lazy<IntGauge> = Lazy::new(|| {
    register_int_gauge!("dpdk_active_flows", "Number of active flows").unwrap()
});

static MEMORY_USAGE: Lazy<IntGauge> = Lazy::new(|| {
    register_int_gauge!("dpdk_memory_usage_bytes", "DPDK memory usage in bytes").unwrap()
});

impl DpdkEngine {
    /// Create a new DPDK Engine instance
    pub async fn new(config: DpdkConfig) -> Result<Self> {
        info!("Initializing DPDK Engine");
        
        // Initialize DPDK EAL
        Self::initialize_dpdk(&config).await?;
        
        // Initialize memory pools for each socket
        let memory_pools = Self::initialize_memory_pools(&config).await?;
        
        // Initialize flow classifier
        let flow_config = flow::FlowConfig::default();
        let flow_classifier = Arc::new(FlowClassifier::new(0, flow_config)?);
        
        // Initialize packet processor
        let packet_processor = Arc::new(packet::DefaultPacketProcessor::new());
        
        // Initialize memory manager
        let memory_manager = Arc::new(DpdkMemoryManager::new(&config)?);
        
        // Initialize analytics
        let analytics_config = AnalyticsConfig::default();
        let analytics = Arc::new(PacketAnalytics::new(analytics_config)?);
        
        // Initialize hardware offload manager
        let offload_config = OffloadConfig::default();
        let offload_manager = Arc::new(HardwareOffloadManager::new(0, offload_config)?);
        
        // Setup communication channels
        let (rx_channels, tx_channels) = Self::setup_channels(&config);
        
        let engine = Self {
            config,
            ports: Vec::new(),
            memory_pools,
            flow_classifier,
            packet_processor,
            memory_manager,
            worker_handles: Vec::new(),
            shutdown_signal: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(RwLock::new(PacketStats::default())),
            analytics,
            rx_channels,
            tx_channels,
            offload_manager,
        };
        
        info!("DPDK Engine initialized successfully");
        Ok(engine)
    }
    
    /// Initialize DPDK Environment Abstraction Layer (EAL)
    async fn initialize_dpdk(config: &DpdkConfig) -> Result<()> {
        info!("Initializing DPDK EAL");
        
        let mut argv: Vec<CString> = Vec::new();
        
        // Program name
        argv.push(CString::new("qbitel-dpdk").unwrap());
        
        // Core mask
        if !config.core_mask.is_empty() {
            argv.push(CString::new("-c").unwrap());
            argv.push(CString::new(config.core_mask.clone()).unwrap());
        }
        
        // Memory channels
        if config.memory_channels > 0 {
            argv.push(CString::new("-n").unwrap());
            argv.push(CString::new(config.memory_channels.to_string()).unwrap());
        }
        
        // Huge pages
        if config.use_huge_pages {
            argv.push(CString::new("--huge-dir").unwrap());
            argv.push(CString::new(config.huge_page_dir.clone()).unwrap());
        }
        
        // PCI whitelist/blacklist
        for pci_addr in &config.pci_whitelist {
            argv.push(CString::new("-w").unwrap());
            argv.push(CString::new(pci_addr.clone()).unwrap());
        }
        
        // Convert to raw pointers for C interface
        let mut raw_argv: Vec<*mut i8> = argv.iter()
            .map(|s| s.as_ptr() as *mut i8)
            .collect();
        
        let argc = raw_argv.len() as i32;
        
        // Initialize DPDK
        unsafe {
            let ret = rte_eal_init(argc, raw_argv.as_mut_ptr());
            if ret < 0 {
                return Err(DpdkError::InitializationFailed(
                    "EAL initialization failed".to_string()
                ));
            }
        }
        
        info!("DPDK EAL initialized successfully");
        Ok(())
    }
    
    /// Initialize memory pools for each NUMA socket
    async fn initialize_memory_pools(config: &DpdkConfig) -> Result<HashMap<u32, Arc<MemoryPool>>> {
        let mut pools = HashMap::new();
        
        for socket_id in &config.numa_sockets {
            let pool_name = format!("mbuf_pool_{}", socket_id);
            let pool_name_c = CString::new(pool_name.clone()).unwrap();
            
            unsafe {
                let mbuf_pool = rte_pktmbuf_pool_create(
                    pool_name_c.as_ptr(),
                    config.mbuf_pool_size as u32,
                    config.mbuf_cache_size as u32,
                    0, // private data size
                    RTE_MBUF_DEFAULT_BUF_SIZE as u16,
                    *socket_id as i32,
                );
                
                if mbuf_pool.is_null() {
                    return Err(DpdkError::MemoryAllocFailed(
                        format!("Failed to create mbuf pool for socket {}", socket_id)
                    ));
                }
                
                let pool = Arc::new(MemoryPool::new(mbuf_pool, *socket_id, &config)?);
                pools.insert(*socket_id, pool);
            }
        }
        
        info!("Memory pools initialized for {} sockets", pools.len());
        Ok(pools)
    }
    
    /// Setup inter-thread communication channels
    fn setup_channels(config: &DpdkConfig) -> (Vec<Receiver<PacketBuffer>>, Vec<Sender<PacketBuffer>>) {
        let mut rx_channels = Vec::new();
        let mut tx_channels = Vec::new();
        
        // Create channels for each worker thread
        for _ in 0..config.worker_threads {
            let (tx, rx) = bounded(config.channel_size);
            tx_channels.push(tx);
            rx_channels.push(rx);
        }
        
        (rx_channels, tx_channels)
    }
    
    /// Start the DPDK packet processing engine
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting DPDK packet processing engine");
        
        // Start RX workers
        for (i, rx_channel) in self.rx_channels.iter().enumerate() {
            let worker_handle = self.start_rx_worker(i, rx_channel.clone()).await?;
            self.worker_handles.push(worker_handle);
        }
        
        // Start TX workers
        for (i, tx_channel) in self.tx_channels.iter().enumerate() {
            let worker_handle = self.start_tx_worker(i, tx_channel.clone()).await?;
            self.worker_handles.push(worker_handle);
        }
        
        // Start analytics worker
        let analytics_handle = self.start_analytics_worker().await?;
        self.worker_handles.push(analytics_handle);
        
        // Start statistics collection
        let stats_handle = self.start_stats_worker().await?;
        self.worker_handles.push(stats_handle);
        
        info!("DPDK Engine started successfully with {} workers", self.worker_handles.len());
        Ok(())
    }
    
    /// Start an RX worker thread
    async fn start_rx_worker(&self, worker_id: usize, _rx_channel: Receiver<PacketBuffer>) -> Result<tokio::task::JoinHandle<()>> {
        let shutdown_signal = Arc::clone(&self.shutdown_signal);
        let stats = Arc::clone(&self.stats);
        let flow_classifier = Arc::clone(&self.flow_classifier);
        let packet_processor = Arc::clone(&self.packet_processor);
        let analytics = Arc::clone(&self.analytics);
        let port_id = self.ports.first().map(|p| p.port_id).unwrap_or(0);
        let queue_id = worker_id as u16;
        let burst_size: u16 = 32;

        let handle = tokio::task::spawn_blocking(move || {
            info!("RX worker {} started on port {} queue {}", worker_id, port_id, queue_id);

            let mut rx_bufs: Vec<*mut rte_mbuf> = vec![std::ptr::null_mut(); burst_size as usize];

            while !shutdown_signal.load(Ordering::Relaxed) {
                // 1. Poll DPDK port for packets
                let nb_rx = unsafe {
                    rte_eth_rx_burst(
                        port_id,
                        queue_id,
                        rx_bufs.as_mut_ptr(),
                        burst_size,
                    )
                };

                if nb_rx == 0 {
                    // No packets available, brief pause to avoid busy spin
                    std::thread::sleep(Duration::from_micros(10));
                    continue;
                }

                let mut local_rx_bytes: u64 = 0;

                for i in 0..nb_rx as usize {
                    let mbuf = rx_bufs[i];
                    if mbuf.is_null() {
                        continue;
                    }

                    let pkt_len = unsafe { (*mbuf).pkt_len } as u64;
                    local_rx_bytes += pkt_len;

                    // 2. Classify the packet via the flow table
                    if let Ok(pkt_buf) = PacketBuffer::from_mbuf(mbuf) {
                        match flow_classifier.classify(&pkt_buf) {
                            Ok(Some(action)) => {
                                // 3. Process packet through the pipeline
                                if let Err(e) = packet_processor.process_sync(&pkt_buf, &action) {
                                    debug!("RX worker {} packet processing error: {}", worker_id, e);
                                    let mut s = stats.write();
                                    s.rx_errors += 1;
                                }
                                // Record flow table hit
                                let mut s = stats.write();
                                s.flow_table_hits += 1;
                            }
                            Ok(None) => {
                                // Flow table miss — run default pipeline
                                if let Err(e) = packet_processor.process_default_sync(&pkt_buf) {
                                    debug!("RX worker {} default processing error: {}", worker_id, e);
                                }
                                let mut s = stats.write();
                                s.flow_table_misses += 1;
                            }
                            Err(e) => {
                                debug!("RX worker {} flow classification error: {}", worker_id, e);
                            }
                        }

                        // 4. Submit to analytics
                        let _ = analytics.record_packet(&pkt_buf);
                    }

                    // Free the mbuf
                    unsafe { rte_pktmbuf_free(mbuf); }
                }

                // Update aggregate stats
                {
                    let mut s = stats.write();
                    s.rx_packets += nb_rx as u64;
                    s.rx_bytes += local_rx_bytes;
                }
            }

            info!("RX worker {} shutdown", worker_id);
        });

        Ok(handle)
    }

    /// Start a TX worker thread
    async fn start_tx_worker(&self, worker_id: usize, tx_channel: Sender<PacketBuffer>) -> Result<tokio::task::JoinHandle<()>> {
        let shutdown_signal = Arc::clone(&self.shutdown_signal);
        let stats = Arc::clone(&self.stats);
        let port_id = self.ports.first().map(|p| p.port_id).unwrap_or(0);
        let queue_id = worker_id as u16;
        let burst_size: u16 = 32;

        let handle = tokio::task::spawn_blocking(move || {
            info!("TX worker {} started on port {} queue {}", worker_id, port_id, queue_id);

            let mut tx_bufs: Vec<*mut rte_mbuf> = Vec::with_capacity(burst_size as usize);

            while !shutdown_signal.load(Ordering::Relaxed) {
                tx_bufs.clear();

                // Drain up to burst_size packets from the channel
                for _ in 0..burst_size {
                    match tx_channel.try_recv() {
                        Ok(pkt_buf) => {
                            if let Some(mbuf) = pkt_buf.into_mbuf() {
                                tx_bufs.push(mbuf);
                            }
                        }
                        Err(crossbeam_channel::TryRecvError::Empty) => break,
                        Err(crossbeam_channel::TryRecvError::Disconnected) => {
                            warn!("TX channel disconnected for worker {}", worker_id);
                            return;
                        }
                    }
                }

                if tx_bufs.is_empty() {
                    std::thread::sleep(Duration::from_micros(10));
                    continue;
                }

                // Transmit the burst
                let nb_tx = unsafe {
                    rte_eth_tx_burst(
                        port_id,
                        queue_id,
                        tx_bufs.as_mut_ptr(),
                        tx_bufs.len() as u16,
                    )
                };

                let mut local_tx_bytes: u64 = 0;
                for i in 0..nb_tx as usize {
                    local_tx_bytes += unsafe { (*tx_bufs[i]).pkt_len } as u64;
                }

                // Free any unsent mbufs
                let dropped = tx_bufs.len() as u16 - nb_tx;
                if dropped > 0 {
                    for i in nb_tx as usize..tx_bufs.len() {
                        unsafe { rte_pktmbuf_free(tx_bufs[i]); }
                    }
                }

                // Update stats
                {
                    let mut s = stats.write();
                    s.tx_packets += nb_tx as u64;
                    s.tx_bytes += local_tx_bytes;
                    s.tx_dropped += dropped as u64;
                }
            }

            info!("TX worker {} shutdown", worker_id);
        });

        Ok(handle)
    }
    
    /// Start analytics worker
    async fn start_analytics_worker(&self) -> Result<tokio::task::JoinHandle<()>> {
        let shutdown_signal = Arc::clone(&self.shutdown_signal);
        let analytics = Arc::clone(&self.analytics);
        
        let handle = tokio::task::spawn(async move {
            info!("Analytics worker started");
            let mut interval = interval(Duration::from_secs(1));
            
            while !shutdown_signal.load(Ordering::Relaxed) {
                interval.tick().await;
                // Update analytics
            }
            
            info!("Analytics worker shutdown");
        });
        
        Ok(handle)
    }
    
    /// Start statistics collection worker
    async fn start_stats_worker(&self) -> Result<tokio::task::JoinHandle<()>> {
        let shutdown_signal = Arc::clone(&self.shutdown_signal);
        let stats = Arc::clone(&self.stats);
        
        let handle = tokio::task::spawn(async move {
            info!("Statistics worker started");
            let mut interval = interval(Duration::from_secs(1));
            
            while !shutdown_signal.load(Ordering::Relaxed) {
                interval.tick().await;
                
                // Update Prometheus metrics from real counters
                let stats_guard = stats.read();
                let total_pkts = stats_guard.rx_packets + stats_guard.tx_packets;
                PACKETS_PROCESSED.inc_by(total_pkts);
                ACTIVE_FLOWS.set(stats_guard.flow_table_hits as i64);

                // Query real DPDK memory usage via memzone stats
                let mem_bytes: i64 = unsafe {
                    let mut stats_info: rte_malloc_socket_stats = std::mem::zeroed();
                    // Socket 0 is the primary NUMA node
                    if rte_malloc_get_socket_stats(0, &mut stats_info) == 0 {
                        (stats_info.heap_allocsz_bytes) as i64
                    } else {
                        0
                    }
                };
                MEMORY_USAGE.set(mem_bytes);
            }
            
            info!("Statistics worker shutdown");
        });
        
        Ok(handle)
    }
    
    /// Get current packet processing statistics
    pub fn get_stats(&self) -> PacketStats {
        self.stats.read().clone()
    }
    
    /// Shutdown the DPDK engine
    pub async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down DPDK Engine");
        
        // Signal shutdown to all workers
        self.shutdown_signal.store(true, Ordering::Relaxed);
        
        // Wait for all workers to complete
        for handle in self.worker_handles.drain(..) {
            if let Err(e) = handle.await {
                warn!("Worker thread join error: {}", e);
            }
        }
        
        // Cleanup DPDK resources
        unsafe {
            rte_eal_cleanup();
        }
        
        info!("DPDK Engine shutdown completed");
        Ok(())
    }
}

impl Drop for DpdkEngine {
    fn drop(&mut self) {
        // Emergency cleanup if not properly shutdown
        if !self.shutdown_signal.load(Ordering::Relaxed) {
            warn!("DPDK Engine dropped without proper shutdown");
            self.shutdown_signal.store(true, Ordering::Relaxed);
        }
    }
}

/// DPDK Engine Builder for easier configuration
pub struct DpdkEngineBuilder {
    config: DpdkConfig,
}

impl DpdkEngineBuilder {
    pub fn new() -> Self {
        Self {
            config: DpdkConfig::default(),
        }
    }
    
    pub fn with_config(mut self, config: DpdkConfig) -> Self {
        self.config = config;
        self
    }
    
    pub fn with_core_mask(mut self, core_mask: String) -> Self {
        self.config.core_mask = core_mask;
        self
    }
    
    pub fn with_memory_channels(mut self, channels: u32) -> Self {
        self.config.memory_channels = channels;
        self
    }
    
    pub fn enable_huge_pages(mut self, huge_page_dir: String) -> Self {
        self.config.use_huge_pages = true;
        self.config.huge_page_dir = huge_page_dir;
        self
    }
    
    pub fn with_worker_threads(mut self, threads: usize) -> Self {
        self.config.worker_threads = threads;
        self
    }
    
    pub async fn build(self) -> Result<DpdkEngine> {
        DpdkEngine::new(self.config).await
    }
}

impl Default for DpdkEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Re-export important types
pub use config::DpdkConfig;
pub use packet::{PacketBuffer, PacketProcessor, PacketClassification};
pub use flow::{FlowClassifier, FlowPattern, FlowAction};
pub use memory::{MemoryPool, DpdkMemoryManager};
pub use analytics::{PacketAnalytics, AnalyticsConfig, AnalyticsReport};
pub use offload::{HardwareOffloadManager, OffloadConfig};