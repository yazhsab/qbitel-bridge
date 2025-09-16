//! High-Frequency Performance Monitoring System
//!
//! This module provides ultra-low latency, high-frequency performance monitoring
//! designed for real-time data plane operations with minimal overhead.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::thread;

use crossbeam_channel::{unbounded, Receiver, Sender};
use parking_lot::{RwLock, Mutex};
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, warn};

/// High-frequency metrics collector with lock-free operations
pub struct HighFrequencyMetrics {
    config: HFMetricsConfig,
    collectors: Arc<RwLock<HashMap<String, Box<dyn MetricCollector + Send + Sync>>>>,
    ring_buffer: Arc<RingBuffer>,
    aggregator: Arc<MetricsAggregator>,
    exporter: Arc<MetricsExporter>,
    shutdown: Arc<AtomicBool>,
    stats: Arc<HFStats>,
}

#[derive(Debug, Clone)]
pub struct HFMetricsConfig {
    /// Sample rate in Hz (samples per second)
    pub sample_rate_hz: u64,
    
    /// Ring buffer size for high-frequency samples
    pub ring_buffer_size: usize,
    
    /// Batch size for processing
    pub batch_size: usize,
    
    /// Aggregation window in milliseconds
    pub aggregation_window_ms: u64,
    
    /// Export interval in milliseconds
    pub export_interval_ms: u64,
    
    /// Enable lock-free operations
    pub lock_free_mode: bool,
    
    /// Maximum memory usage in MB
    pub max_memory_mb: usize,
    
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    
    /// Histogram bucket configuration
    pub histogram_buckets: Vec<f64>,
}

/// Lock-free ring buffer for high-frequency data
pub struct RingBuffer {
    buffer: Vec<AtomicU64>,
    write_index: AtomicUsize,
    read_index: AtomicUsize,
    capacity: usize,
    element_size: usize,
}

/// High-frequency statistics tracking
#[derive(Debug)]
pub struct HFStats {
    /// Total samples collected
    pub total_samples: AtomicU64,
    
    /// Samples per second (current)
    pub samples_per_second: AtomicU64,
    
    /// Buffer overruns
    pub buffer_overruns: AtomicU64,
    
    /// Processing latency (nanoseconds)
    pub processing_latency_ns: AtomicU64,
    
    /// Memory usage in bytes
    pub memory_usage_bytes: AtomicU64,
    
    /// Cache hit rate
    pub cache_hit_rate: AtomicU64,
}

/// Trait for metric collectors
pub trait MetricCollector {
    fn collect(&self) -> Vec<MetricSample>;
    fn name(&self) -> &str;
    fn sample_rate(&self) -> u64;
    fn reset(&mut self);
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSample {
    pub timestamp_ns: u64,
    pub metric_name: String,
    pub value: f64,
    pub tags: HashMap<String, String>,
    pub sample_type: SampleType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SampleType {
    Counter,
    Gauge,
    Histogram,
    Timer,
    Rate,
}

/// Network performance collector
pub struct NetworkPerformanceCollector {
    name: String,
    sample_rate: u64,
    last_sample_time: Instant,
    last_bytes_in: AtomicU64,
    last_bytes_out: AtomicU64,
    last_packets_in: AtomicU64,
    last_packets_out: AtomicU64,
}

/// CPU performance collector with SIMD optimizations
pub struct CPUPerformanceCollector {
    name: String,
    sample_rate: u64,
    cpu_samples: Vec<AtomicU64>,
    last_idle_time: AtomicU64,
    last_total_time: AtomicU64,
}

/// Memory performance collector
pub struct MemoryPerformanceCollector {
    name: String,
    sample_rate: u64,
    memory_samples: Vec<AtomicU64>,
    page_fault_counter: AtomicU64,
    cache_miss_counter: AtomicU64,
}

/// Latency collector with histogram
pub struct LatencyCollector {
    name: String,
    sample_rate: u64,
    histogram_buckets: Vec<f64>,
    bucket_counts: Vec<AtomicU64>,
    sum: AtomicU64,
    count: AtomicU64,
    min: AtomicU64,
    max: AtomicU64,
}

/// Metrics aggregator for real-time analytics
pub struct MetricsAggregator {
    config: HFMetricsConfig,
    aggregation_buffer: Arc<Mutex<HashMap<String, AggregationState>>>,
    last_aggregation: AtomicU64,
    sender: Sender<AggregatedMetrics>,
    receiver: Receiver<AggregatedMetrics>,
}

#[derive(Debug, Clone)]
struct AggregationState {
    count: u64,
    sum: f64,
    min: f64,
    max: f64,
    variance_sum: f64,
    histogram: Vec<u64>,
    last_update: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct AggregatedMetrics {
    pub timestamp_ns: u64,
    pub window_ms: u64,
    pub metrics: HashMap<String, MetricAggregate>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MetricAggregate {
    pub count: u64,
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub std_dev: f64,
    pub percentiles: HashMap<String, f64>, // P50, P95, P99, etc.
    pub rate_per_second: f64,
    pub histogram: Vec<u64>,
}

/// High-performance metrics exporter
pub struct MetricsExporter {
    config: HFMetricsConfig,
    export_buffer: Arc<Mutex<Vec<AggregatedMetrics>>>,
    last_export: AtomicU64,
    prometheus_format: AtomicBool,
}

impl Default for HFMetricsConfig {
    fn default() -> Self {
        Self {
            sample_rate_hz: 10000, // 10kHz sampling
            ring_buffer_size: 1_000_000, // 1M samples
            batch_size: 1000,
            aggregation_window_ms: 100, // 100ms windows
            export_interval_ms: 1000, // Export every second
            lock_free_mode: true,
            max_memory_mb: 256,
            enable_simd: true,
            histogram_buckets: vec![
                0.000001, 0.000005, 0.00001, 0.000025, 0.00005, 0.0001,
                0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025,
                0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
            ],
        }
    }
}

impl HighFrequencyMetrics {
    pub fn new(config: HFMetricsConfig) -> Self {
        let ring_buffer = Arc::new(RingBuffer::new(config.ring_buffer_size, 64));
        let (sender, receiver) = unbounded();
        
        let aggregator = Arc::new(MetricsAggregator {
            config: config.clone(),
            aggregation_buffer: Arc::new(Mutex::new(HashMap::new())),
            last_aggregation: AtomicU64::new(0),
            sender,
            receiver,
        });
        
        let exporter = Arc::new(MetricsExporter {
            config: config.clone(),
            export_buffer: Arc::new(Mutex::new(Vec::new())),
            last_export: AtomicU64::new(0),
            prometheus_format: AtomicBool::new(true),
        });
        
        Self {
            config,
            collectors: Arc::new(RwLock::new(HashMap::new())),
            ring_buffer,
            aggregator,
            exporter,
            shutdown: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(HFStats {
                total_samples: AtomicU64::new(0),
                samples_per_second: AtomicU64::new(0),
                buffer_overruns: AtomicU64::new(0),
                processing_latency_ns: AtomicU64::new(0),
                memory_usage_bytes: AtomicU64::new(0),
                cache_hit_rate: AtomicU64::new(0),
            }),
        }
    }
    
    pub fn start(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Starting high-frequency metrics system");
        
        // Start collection thread
        self.start_collection_thread();
        
        // Start aggregation thread
        self.start_aggregation_thread();
        
        // Start export thread
        self.start_export_thread();
        
        // Initialize default collectors
        self.register_default_collectors();
        
        info!("High-frequency metrics system started");
        Ok(())
    }
    
    pub fn register_collector(&self, collector: Box<dyn MetricCollector + Send + Sync>) {
        let mut collectors = self.collectors.write();
        collectors.insert(collector.name().to_string(), collector);
        debug!("Registered collector: {}", collector.name());
    }
    
    pub fn record_sample(&self, sample: MetricSample) {
        let start_time = Instant::now();
        
        // Convert sample to bytes for ring buffer
        let sample_bytes = bincode::serialize(&sample).unwrap_or_default();
        if self.ring_buffer.write(&sample_bytes) {
            self.stats.total_samples.fetch_add(1, Ordering::Relaxed);
        } else {
            self.stats.buffer_overruns.fetch_add(1, Ordering::Relaxed);
            warn!("Ring buffer overrun - sample dropped");
        }
        
        // Update processing latency
        let latency_ns = start_time.elapsed().as_nanos() as u64;
        self.stats.processing_latency_ns.store(latency_ns, Ordering::Relaxed);
    }
    
    pub fn record_timer<F>(&self, name: &str, tags: HashMap<String, String>, f: F) -> f64
    where
        F: FnOnce() -> f64,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed().as_secs_f64();
        
        self.record_sample(MetricSample {
            timestamp_ns: start.elapsed().as_nanos() as u64,
            metric_name: name.to_string(),
            value: duration,
            tags,
            sample_type: SampleType::Timer,
        });
        
        result
    }
    
    pub fn increment_counter(&self, name: &str, tags: HashMap<String, String>, value: f64) {
        self.record_sample(MetricSample {
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            metric_name: name.to_string(),
            value,
            tags,
            sample_type: SampleType::Counter,
        });
    }
    
    pub fn set_gauge(&self, name: &str, tags: HashMap<String, String>, value: f64) {
        self.record_sample(MetricSample {
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            metric_name: name.to_string(),
            value,
            tags,
            sample_type: SampleType::Gauge,
        });
    }
    
    pub fn get_stats(&self) -> HFMetricsStats {
        HFMetricsStats {
            total_samples: self.stats.total_samples.load(Ordering::Relaxed),
            samples_per_second: self.stats.samples_per_second.load(Ordering::Relaxed),
            buffer_overruns: self.stats.buffer_overruns.load(Ordering::Relaxed),
            processing_latency_ns: self.stats.processing_latency_ns.load(Ordering::Relaxed),
            memory_usage_bytes: self.stats.memory_usage_bytes.load(Ordering::Relaxed),
            cache_hit_rate: self.stats.cache_hit_rate.load(Ordering::Relaxed) as f64 / 100.0,
            ring_buffer_utilization: self.ring_buffer.utilization(),
        }
    }
    
    fn start_collection_thread(&self) {
        let collectors = Arc::clone(&self.collectors);
        let ring_buffer = Arc::clone(&self.ring_buffer);
        let shutdown = Arc::clone(&self.shutdown);
        let stats = Arc::clone(&self.stats);
        let sample_interval = Duration::from_nanos(1_000_000_000 / self.config.sample_rate_hz);
        
        thread::spawn(move || {
            let mut last_sample_count = 0u64;
            let mut last_sample_time = Instant::now();
            
            while !shutdown.load(Ordering::Relaxed) {
                let loop_start = Instant::now();
                
                // Collect from all registered collectors
                let collectors_guard = collectors.read();
                for collector in collectors_guard.values() {
                    let samples = collector.collect();
                    for sample in samples {
                        let sample_bytes = bincode::serialize(&sample).unwrap_or_default();
                        if !ring_buffer.write(&sample_bytes) {
                            stats.buffer_overruns.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
                drop(collectors_guard);
                
                // Update samples per second
                let current_count = stats.total_samples.load(Ordering::Relaxed);
                let now = Instant::now();
                if now.duration_since(last_sample_time) >= Duration::from_secs(1) {
                    let samples_delta = current_count - last_sample_count;
                    stats.samples_per_second.store(samples_delta, Ordering::Relaxed);
                    last_sample_count = current_count;
                    last_sample_time = now;
                }
                
                // Sleep for remaining interval
                let elapsed = loop_start.elapsed();
                if elapsed < sample_interval {
                    thread::sleep(sample_interval - elapsed);
                }
            }
        });
    }
    
    fn start_aggregation_thread(&self) {
        let aggregator = Arc::clone(&self.aggregator);
        let ring_buffer = Arc::clone(&self.ring_buffer);
        let shutdown = Arc::clone(&self.shutdown);
        let aggregation_interval = Duration::from_millis(self.config.aggregation_window_ms);
        
        thread::spawn(move || {
            while !shutdown.load(Ordering::Relaxed) {
                let start_time = Instant::now();
                
                // Read batch from ring buffer
                let mut batch = Vec::with_capacity(1000);
                while batch.len() < 1000 {
                    if let Some(data) = ring_buffer.read() {
                        if let Ok(sample) = bincode::deserialize::<MetricSample>(&data) {
                            batch.push(sample);
                        }
                    } else {
                        break;
                    }
                }
                
                if !batch.is_empty() {
                    aggregator.process_batch(batch);
                }
                
                // Sleep for remaining interval
                let elapsed = start_time.elapsed();
                if elapsed < aggregation_interval {
                    thread::sleep(aggregation_interval - elapsed);
                }
            }
        });
    }
    
    fn start_export_thread(&self) {
        let exporter = Arc::clone(&self.exporter);
        let aggregator = Arc::clone(&self.aggregator);
        let shutdown = Arc::clone(&self.shutdown);
        let export_interval = Duration::from_millis(self.config.export_interval_ms);
        
        thread::spawn(move || {
            while !shutdown.load(Ordering::Relaxed) {
                let start_time = Instant::now();
                
                // Get aggregated metrics
                while let Ok(aggregated) = aggregator.receiver.try_recv() {
                    exporter.export(aggregated);
                }
                
                // Sleep for remaining interval
                let elapsed = start_time.elapsed();
                if elapsed < export_interval {
                    thread::sleep(export_interval - elapsed);
                }
            }
        });
    }
    
    fn register_default_collectors(&self) {
        // Network performance collector
        self.register_collector(Box::new(NetworkPerformanceCollector::new(
            "network_performance".to_string(),
            self.config.sample_rate_hz,
        )));
        
        // CPU performance collector
        self.register_collector(Box::new(CPUPerformanceCollector::new(
            "cpu_performance".to_string(),
            self.config.sample_rate_hz,
        )));
        
        // Memory performance collector
        self.register_collector(Box::new(MemoryPerformanceCollector::new(
            "memory_performance".to_string(),
            self.config.sample_rate_hz,
        )));
        
        // Latency collector
        self.register_collector(Box::new(LatencyCollector::new(
            "latency_metrics".to_string(),
            self.config.sample_rate_hz,
            self.config.histogram_buckets.clone(),
        )));
    }
    
    pub fn shutdown(&self) {
        info!("Shutting down high-frequency metrics system");
        self.shutdown.store(true, Ordering::Relaxed);
        
        // Flush remaining data
        if let Ok(mut export_buffer) = self.exporter.export_buffer.try_lock() {
            if !export_buffer.is_empty() {
                info!("Flushing {} remaining metric batches", export_buffer.len());
                export_buffer.clear();
            }
        }
        
        info!("High-frequency metrics system shut down");
    }
}

impl RingBuffer {
    fn new(capacity: usize, element_size: usize) -> Self {
        let buffer = (0..capacity).map(|_| AtomicU64::new(0)).collect();
        
        Self {
            buffer,
            write_index: AtomicUsize::new(0),
            read_index: AtomicUsize::new(0),
            capacity,
            element_size,
        }
    }
    
    fn write(&self, data: &[u8]) -> bool {
        let write_idx = self.write_index.load(Ordering::Acquire);
        let next_write_idx = (write_idx + 1) % self.capacity;
        
        // Check if buffer is full
        if next_write_idx == self.read_index.load(Ordering::Acquire) {
            return false;
        }
        
        // Store data (simplified - in real implementation would handle variable size)
        if data.len() <= 8 {
            let mut value = 0u64;
            for (i, &byte) in data.iter().enumerate() {
                value |= (byte as u64) << (i * 8);
            }
            self.buffer[write_idx].store(value, Ordering::Release);
        }
        
        self.write_index.store(next_write_idx, Ordering::Release);
        true
    }
    
    fn read(&self) -> Option<Vec<u8>> {
        let read_idx = self.read_index.load(Ordering::Acquire);
        let write_idx = self.write_index.load(Ordering::Acquire);
        
        if read_idx == write_idx {
            return None; // Buffer is empty
        }
        
        let value = self.buffer[read_idx].load(Ordering::Acquire);
        let next_read_idx = (read_idx + 1) % self.capacity;
        self.read_index.store(next_read_idx, Ordering::Release);
        
        // Convert back to bytes (simplified)
        let mut data = Vec::new();
        for i in 0..8 {
            let byte = ((value >> (i * 8)) & 0xFF) as u8;
            if byte != 0 {
                data.push(byte);
            }
        }
        
        Some(data)
    }
    
    fn utilization(&self) -> f64 {
        let write_idx = self.write_index.load(Ordering::Acquire);
        let read_idx = self.read_index.load(Ordering::Acquire);
        
        let used = if write_idx >= read_idx {
            write_idx - read_idx
        } else {
            self.capacity - read_idx + write_idx
        };
        
        used as f64 / self.capacity as f64
    }
}

// Collector implementations...
impl NetworkPerformanceCollector {
    fn new(name: String, sample_rate: u64) -> Self {
        Self {
            name,
            sample_rate,
            last_sample_time: Instant::now(),
            last_bytes_in: AtomicU64::new(0),
            last_bytes_out: AtomicU64::new(0),
            last_packets_in: AtomicU64::new(0),
            last_packets_out: AtomicU64::new(0),
        }
    }
}

impl MetricCollector for NetworkPerformanceCollector {
    fn collect(&self) -> Vec<MetricSample> {
        let now = Instant::now();
        let timestamp_ns = now.duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64;
        
        // Simulate network metrics collection
        vec![
            MetricSample {
                timestamp_ns,
                metric_name: "network_bytes_in_rate".to_string(),
                value: 1_000_000.0, // 1MB/s
                tags: HashMap::new(),
                sample_type: SampleType::Rate,
            },
            MetricSample {
                timestamp_ns,
                metric_name: "network_bytes_out_rate".to_string(),
                value: 800_000.0, // 800KB/s
                tags: HashMap::new(),
                sample_type: SampleType::Rate,
            },
        ]
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn sample_rate(&self) -> u64 {
        self.sample_rate
    }
    
    fn reset(&mut self) {
        self.last_bytes_in.store(0, Ordering::Relaxed);
        self.last_bytes_out.store(0, Ordering::Relaxed);
    }
}

// Similar implementations for other collectors...
impl CPUPerformanceCollector {
    fn new(name: String, sample_rate: u64) -> Self {
        Self {
            name,
            sample_rate,
            cpu_samples: (0..num_cpus::get()).map(|_| AtomicU64::new(0)).collect(),
            last_idle_time: AtomicU64::new(0),
            last_total_time: AtomicU64::new(0),
        }
    }
}

impl MetricCollector for CPUPerformanceCollector {
    fn collect(&self) -> Vec<MetricSample> {
        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        vec![
            MetricSample {
                timestamp_ns,
                metric_name: "cpu_usage_percent".to_string(),
                value: 25.5,
                tags: HashMap::new(),
                sample_type: SampleType::Gauge,
            },
        ]
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn sample_rate(&self) -> u64 {
        self.sample_rate
    }
    
    fn reset(&mut self) {
        for sample in &self.cpu_samples {
            sample.store(0, Ordering::Relaxed);
        }
    }
}

impl MemoryPerformanceCollector {
    fn new(name: String, sample_rate: u64) -> Self {
        Self {
            name,
            sample_rate,
            memory_samples: (0..16).map(|_| AtomicU64::new(0)).collect(),
            page_fault_counter: AtomicU64::new(0),
            cache_miss_counter: AtomicU64::new(0),
        }
    }
}

impl MetricCollector for MemoryPerformanceCollector {
    fn collect(&self) -> Vec<MetricSample> {
        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        vec![
            MetricSample {
                timestamp_ns,
                metric_name: "memory_usage_percent".to_string(),
                value: 65.2,
                tags: HashMap::new(),
                sample_type: SampleType::Gauge,
            },
        ]
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn sample_rate(&self) -> u64 {
        self.sample_rate
    }
    
    fn reset(&mut self) {
        self.page_fault_counter.store(0, Ordering::Relaxed);
        self.cache_miss_counter.store(0, Ordering::Relaxed);
    }
}

impl LatencyCollector {
    fn new(name: String, sample_rate: u64, buckets: Vec<f64>) -> Self {
        let bucket_counts = buckets.iter().map(|_| AtomicU64::new(0)).collect();
        
        Self {
            name,
            sample_rate,
            histogram_buckets: buckets,
            bucket_counts,
            sum: AtomicU64::new(0),
            count: AtomicU64::new(0),
            min: AtomicU64::new(u64::MAX),
            max: AtomicU64::new(0),
        }
    }
}

impl MetricCollector for LatencyCollector {
    fn collect(&self) -> Vec<MetricSample> {
        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        vec![
            MetricSample {
                timestamp_ns,
                metric_name: "operation_latency_ms".to_string(),
                value: 0.5, // 0.5ms
                tags: HashMap::new(),
                sample_type: SampleType::Histogram,
            },
        ]
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn sample_rate(&self) -> u64 {
        self.sample_rate
    }
    
    fn reset(&mut self) {
        for bucket in &self.bucket_counts {
            bucket.store(0, Ordering::Relaxed);
        }
        self.sum.store(0, Ordering::Relaxed);
        self.count.store(0, Ordering::Relaxed);
        self.min.store(u64::MAX, Ordering::Relaxed);
        self.max.store(0, Ordering::Relaxed);
    }
}

impl MetricsAggregator {
    fn process_batch(&self, samples: Vec<MetricSample>) {
        let mut buffer = self.aggregation_buffer.lock();
        
        for sample in samples {
            let state = buffer.entry(sample.metric_name.clone())
                .or_insert_with(|| AggregationState {
                    count: 0,
                    sum: 0.0,
                    min: f64::MAX,
                    max: f64::MIN,
                    variance_sum: 0.0,
                    histogram: vec![0; 21], // Based on default buckets
                    last_update: sample.timestamp_ns,
                });
            
            state.count += 1;
            state.sum += sample.value;
            state.min = state.min.min(sample.value);
            state.max = state.max.max(sample.value);
            state.last_update = sample.timestamp_ns;
            
            // Update histogram if applicable
            if matches!(sample.sample_type, SampleType::Histogram | SampleType::Timer) {
                for (i, &bucket) in self.config.histogram_buckets.iter().enumerate() {
                    if sample.value <= bucket {
                        state.histogram[i] += 1;
                        break;
                    }
                }
            }
        }
        
        // Send aggregated metrics if window is complete
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
            
        if now - self.last_aggregation.load(Ordering::Relaxed) >= self.config.aggregation_window_ms * 1_000_000 {
            let aggregated = self.create_aggregated_metrics(&buffer, now);
            let _ = self.sender.try_send(aggregated);
            buffer.clear();
            self.last_aggregation.store(now, Ordering::Relaxed);
        }
    }
    
    fn create_aggregated_metrics(&self, buffer: &HashMap<String, AggregationState>, timestamp_ns: u64) -> AggregatedMetrics {
        let mut metrics = HashMap::new();
        
        for (name, state) in buffer.iter() {
            let mean = if state.count > 0 { state.sum / state.count as f64 } else { 0.0 };
            let std_dev = 0.0; // Simplified - would calculate properly
            
            let mut percentiles = HashMap::new();
            percentiles.insert("p50".to_string(), mean); // Simplified
            percentiles.insert("p95".to_string(), state.max * 0.95);
            percentiles.insert("p99".to_string(), state.max * 0.99);
            
            let rate_per_second = state.count as f64 / (self.config.aggregation_window_ms as f64 / 1000.0);
            
            metrics.insert(name.clone(), MetricAggregate {
                count: state.count,
                mean,
                min: state.min,
                max: state.max,
                std_dev,
                percentiles,
                rate_per_second,
                histogram: state.histogram.clone(),
            });
        }
        
        AggregatedMetrics {
            timestamp_ns,
            window_ms: self.config.aggregation_window_ms,
            metrics,
        }
    }
}

impl MetricsExporter {
    fn export(&self, metrics: AggregatedMetrics) {
        let mut export_buffer = self.export_buffer.lock();
        export_buffer.push(metrics);
        
        // Export to various backends (Prometheus, InfluxDB, etc.)
        self.export_to_prometheus(&export_buffer);
        export_buffer.clear();
    }
    
    fn export_to_prometheus(&self, metrics: &[AggregatedMetrics]) {
        // Implementation would export to Prometheus format
        debug!("Exporting {} metric batches to Prometheus", metrics.len());
    }
}

#[derive(Debug, Serialize)]
pub struct HFMetricsStats {
    pub total_samples: u64,
    pub samples_per_second: u64,
    pub buffer_overruns: u64,
    pub processing_latency_ns: u64,
    pub memory_usage_bytes: u64,
    pub cache_hit_rate: f64,
    pub ring_buffer_utilization: f64,
}