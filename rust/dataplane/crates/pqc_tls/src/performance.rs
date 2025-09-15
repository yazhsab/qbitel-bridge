//! Performance optimizations and memory-efficient operations for PQC
//! 
//! This module provides:
//! - Memory pool management for reducing allocations
//! - SIMD-optimized polynomial arithmetic
//! - Batch processing capabilities
//! - Cache-friendly data structures
//! - Zero-copy operations where possible
//! - Memory profiling and monitoring
//! - Performance benchmarking and optimization hints

use crate::errors::TlsError;
use crate::kyber::{KyberPublicKey, KyberPrivateKey, KyberCiphertext, KyberSharedSecret};
use bytes::{Bytes, BytesMut};
use metrics::{histogram, counter, gauge};
use rayon::prelude::*;
use std::alloc::{alloc, dealloc, Layout};
use std::collections::{HashMap, VecDeque};
use std::mem::{size_of, align_of, ManuallyDrop};
use std::ptr::{NonNull, copy_nonoverlapping};
use std::sync::{Arc, atomic::{AtomicU64, AtomicUsize, Ordering}};
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Mutex};
use tracing::{info, warn, debug, span, Level};
use wide::*;

/// Memory pool for efficient allocation of cryptographic buffers
pub struct MemoryPool {
    pools: HashMap<usize, Pool>,
    stats: PoolStatistics,
    max_pool_size: usize,
    alignment: usize,
}

struct Pool {
    free_blocks: VecDeque<NonNull<u8>>,
    block_size: usize,
    allocated_count: AtomicUsize,
    total_allocated: AtomicU64,
}

#[derive(Debug, Default)]
pub struct PoolStatistics {
    pub total_allocations: AtomicU64,
    pub total_deallocations: AtomicU64,
    pub current_allocated_bytes: AtomicU64,
    pub peak_allocated_bytes: AtomicU64,
    pub pool_hits: AtomicU64,
    pub pool_misses: AtomicU64,
}

impl MemoryPool {
    /// Create a new memory pool with specified configuration
    pub fn new(max_pool_size: usize, alignment: usize) -> Self {
        let mut pools = HashMap::new();
        
        // Pre-create pools for common PQC buffer sizes
        let common_sizes = vec![
            32,    // Shared secrets
            768,   // Kyber-768 private keys
            1184,  // Kyber-768 public keys
            1088,  // Kyber-768 ciphertexts
            2304,  // Dilithium-2 signatures
            4096,  // General purpose large buffers
            8192,  // Very large buffers
        ];

        for size in common_sizes {
            pools.insert(size, Pool {
                free_blocks: VecDeque::with_capacity(max_pool_size),
                block_size: size,
                allocated_count: AtomicUsize::new(0),
                total_allocated: AtomicU64::new(0),
            });
        }

        Self {
            pools,
            stats: PoolStatistics::default(),
            max_pool_size,
            alignment,
        }
    }

    /// Allocate a buffer from the pool
    pub fn allocate(&mut self, size: usize) -> Result<PooledBuffer, TlsError> {
        self.stats.total_allocations.fetch_add(1, Ordering::Relaxed);

        // Find the appropriate pool (next power of 2 or exact match)
        let pool_size = self.find_pool_size(size);
        
        if let Some(pool) = self.pools.get_mut(&pool_size) {
            if let Some(ptr) = pool.free_blocks.pop_front() {
                self.stats.pool_hits.fetch_add(1, Ordering::Relaxed);
                pool.allocated_count.fetch_add(1, Ordering::Relaxed);
                
                let allocated = pool_size as u64;
                self.stats.current_allocated_bytes.fetch_add(allocated, Ordering::Relaxed);
                
                // Update peak if necessary
                let current = self.stats.current_allocated_bytes.load(Ordering::Relaxed);
                let mut peak = self.stats.peak_allocated_bytes.load(Ordering::Relaxed);
                while current > peak {
                    match self.stats.peak_allocated_bytes.compare_exchange_weak(
                        peak, current, Ordering::Relaxed, Ordering::Relaxed
                    ) {
                        Ok(_) => break,
                        Err(new_peak) => peak = new_peak,
                    }
                }
                
                return Ok(PooledBuffer {
                    ptr,
                    size: pool_size,
                    actual_size: size,
                    pool_ref: self as *mut MemoryPool,
                });
            }
        }

        // Pool miss - allocate new buffer
        self.stats.pool_misses.fetch_add(1, Ordering::Relaxed);
        self.allocate_new(size, pool_size)
    }

    /// Find the appropriate pool size for the requested size
    fn find_pool_size(&self, size: usize) -> usize {
        // Find exact match first
        if self.pools.contains_key(&size) {
            return size;
        }

        // Find next larger pool
        self.pools.keys()
            .filter(|&&pool_size| pool_size >= size)
            .min()
            .copied()
            .unwrap_or_else(|| {
                // No suitable pool exists, use next power of 2
                size.next_power_of_two()
            })
    }

    /// Allocate a new buffer when pool is empty or doesn't exist
    fn allocate_new(&mut self, actual_size: usize, pool_size: usize) -> Result<PooledBuffer, TlsError> {
        let layout = Layout::from_size_align(pool_size, self.alignment)
            .map_err(|e| TlsError::Io(format!("invalid memory layout: {}", e)))?;

        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(TlsError::Io("memory allocation failed".into()));
        }

        let non_null = NonNull::new(ptr)
            .ok_or_else(|| TlsError::Io("null pointer from allocator".into()))?;

        // Create pool if it doesn't exist
        if !self.pools.contains_key(&pool_size) {
            self.pools.insert(pool_size, Pool {
                free_blocks: VecDeque::with_capacity(self.max_pool_size),
                block_size: pool_size,
                allocated_count: AtomicUsize::new(0),
                total_allocated: AtomicU64::new(0),
            });
        }

        if let Some(pool) = self.pools.get(&pool_size) {
            pool.allocated_count.fetch_add(1, Ordering::Relaxed);
            pool.total_allocated.fetch_add(pool_size as u64, Ordering::Relaxed);
        }

        let allocated = pool_size as u64;
        self.stats.current_allocated_bytes.fetch_add(allocated, Ordering::Relaxed);

        Ok(PooledBuffer {
            ptr: non_null,
            size: pool_size,
            actual_size,
            pool_ref: self as *mut MemoryPool,
        })
    }

    /// Return a buffer to the pool
    fn deallocate(&mut self, buffer: PooledBuffer) {
        self.stats.total_deallocations.fetch_add(1, Ordering::Relaxed);
        self.stats.current_allocated_bytes.fetch_sub(buffer.size as u64, Ordering::Relaxed);

        if let Some(pool) = self.pools.get_mut(&buffer.size) {
            pool.allocated_count.fetch_sub(1, Ordering::Relaxed);
            
            if pool.free_blocks.len() < self.max_pool_size {
                // Zero out the memory for security
                unsafe {
                    std::ptr::write_bytes(buffer.ptr.as_ptr(), 0, buffer.size);
                }
                
                pool.free_blocks.push_back(buffer.ptr);
            } else {
                // Pool is full, deallocate directly
                let layout = Layout::from_size_align(buffer.size, self.alignment).unwrap();
                unsafe {
                    std::ptr::write_bytes(buffer.ptr.as_ptr(), 0, buffer.size);
                    dealloc(buffer.ptr.as_ptr(), layout);
                }
            }
        } else {
            // No pool exists, deallocate directly
            let layout = Layout::from_size_align(buffer.size, self.alignment).unwrap();
            unsafe {
                std::ptr::write_bytes(buffer.ptr.as_ptr(), 0, buffer.size);
                dealloc(buffer.ptr.as_ptr(), layout);
            }
        }

        // Forget the buffer to prevent double-free
        std::mem::forget(buffer);
    }

    /// Get pool statistics
    pub fn statistics(&self) -> &PoolStatistics {
        &self.stats
    }

    /// Clear all pools (useful for testing or cleanup)
    pub fn clear(&mut self) {
        for pool in self.pools.values_mut() {
            while let Some(ptr) = pool.free_blocks.pop_front() {
                let layout = Layout::from_size_align(pool.block_size, self.alignment).unwrap();
                unsafe {
                    std::ptr::write_bytes(ptr.as_ptr(), 0, pool.block_size);
                    dealloc(ptr.as_ptr(), layout);
                }
            }
        }
    }
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        self.clear();
    }
}

/// A buffer allocated from a memory pool
pub struct PooledBuffer {
    ptr: NonNull<u8>,
    size: usize,
    actual_size: usize,
    pool_ref: *mut MemoryPool,
}

impl PooledBuffer {
    /// Get a mutable slice to the buffer data
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.actual_size)
        }
    }

    /// Get an immutable slice to the buffer data
    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self.ptr.as_ptr(), self.actual_size)
        }
    }

    /// Get the actual usable size
    pub fn len(&self) -> usize {
        self.actual_size
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.actual_size == 0
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        // Safety: pool_ref is valid during the lifetime of this buffer
        unsafe {
            let pool = &mut *self.pool_ref;
            let buffer = PooledBuffer {
                ptr: self.ptr,
                size: self.size,
                actual_size: self.actual_size,
                pool_ref: self.pool_ref,
            };
            pool.deallocate(buffer);
        }
    }
}

/// Batch processing operations for improved performance
pub struct BatchProcessor {
    memory_pool: Arc<Mutex<MemoryPool>>,
    batch_size: usize,
    thread_pool_size: usize,
}

impl BatchProcessor {
    pub fn new(batch_size: usize, thread_pool_size: usize) -> Self {
        Self {
            memory_pool: Arc::new(Mutex::new(MemoryPool::new(1024, 32))),
            batch_size,
            thread_pool_size,
        }
    }

    /// Process multiple encapsulations in parallel
    pub async fn batch_encapsulate(
        &self,
        public_keys: Vec<&KyberPublicKey>,
    ) -> Result<Vec<(KyberCiphertext, KyberSharedSecret)>, TlsError> {
        let _span = span!(Level::INFO, "batch_encapsulate", count = public_keys.len()).entered();

        if public_keys.is_empty() {
            return Ok(Vec::new());
        }

        // Split into batches for parallel processing
        let batches: Vec<_> = public_keys
            .chunks(self.batch_size)
            .collect();

        let results = batches
            .into_par_iter()
            .map(|batch| {
                let kem = crate::kyber::KyberKEM::new();
                let mut batch_results = Vec::new();

                for public_key in batch {
                    match tokio::runtime::Handle::current()
                        .block_on(kem.encapsulate(public_key)) {
                        Ok(result) => batch_results.push((result.ciphertext, result.shared_secret)),
                        Err(e) => return Err(e),
                    }
                }

                Ok(batch_results)
            })
            .collect::<Result<Vec<_>, TlsError>>()?;

        // Flatten results
        let mut all_results = Vec::new();
        for batch_result in results {
            all_results.extend(batch_result);
        }

        Ok(all_results)
    }

    /// Process multiple decapsulations in parallel
    pub async fn batch_decapsulate(
        &self,
        operations: Vec<(&KyberPrivateKey, &KyberCiphertext)>,
    ) -> Result<Vec<KyberSharedSecret>, TlsError> {
        let _span = span!(Level::INFO, "batch_decapsulate", count = operations.len()).entered();

        if operations.is_empty() {
            return Ok(Vec::new());
        }

        // Split into batches for parallel processing
        let batches: Vec<_> = operations
            .chunks(self.batch_size)
            .collect();

        let results = batches
            .into_par_iter()
            .map(|batch| {
                let kem = crate::kyber::KyberKEM::new();
                let mut batch_results = Vec::new();

                for (private_key, ciphertext) in batch {
                    match tokio::runtime::Handle::current()
                        .block_on(kem.decapsulate(private_key, ciphertext)) {
                        Ok(shared_secret) => batch_results.push(shared_secret),
                        Err(e) => return Err(e),
                    }
                }

                Ok(batch_results)
            })
            .collect::<Result<Vec<_>, TlsError>>()?;

        // Flatten results
        let mut all_results = Vec::new();
        for batch_result in results {
            all_results.extend(batch_result);
        }

        Ok(all_results)
    }
}

/// SIMD-optimized operations for polynomial arithmetic
pub struct SimdOperations;

impl SimdOperations {
    /// Add two polynomials using SIMD operations
    #[cfg(target_arch = "x86_64")]
    pub fn poly_add_simd(a: &[i16], b: &[i16], result: &mut [i16]) -> Result<(), TlsError> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(TlsError::Policy("polynomial dimension mismatch".into()));
        }

        if is_x86_feature_detected!("avx2") {
            Self::poly_add_avx2(a, b, result)
        } else if is_x86_feature_detected!("sse2") {
            Self::poly_add_sse2(a, b, result)
        } else {
            Self::poly_add_scalar(a, b, result)
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn poly_add_simd(a: &[i16], b: &[i16], result: &mut [i16]) -> Result<(), TlsError> {
        Self::poly_add_scalar(a, b, result)
    }

    /// AVX2-optimized polynomial addition
    #[cfg(target_arch = "x86_64")]
    fn poly_add_avx2(a: &[i16], b: &[i16], result: &mut [i16]) -> Result<(), TlsError> {
        let chunk_size = 16; // 16 i16 elements per AVX2 register
        let chunks = a.len() / chunk_size;

        for i in 0..chunks {
            let start = i * chunk_size;
            let end = start + chunk_size;

            // Use wide crate for portable SIMD
            let a_chunk = i16x16::new([
                a[start], a[start+1], a[start+2], a[start+3],
                a[start+4], a[start+5], a[start+6], a[start+7],
                a[start+8], a[start+9], a[start+10], a[start+11],
                a[start+12], a[start+13], a[start+14], a[start+15],
            ]);

            let b_chunk = i16x16::new([
                b[start], b[start+1], b[start+2], b[start+3],
                b[start+4], b[start+5], b[start+6], b[start+7],
                b[start+8], b[start+9], b[start+10], b[start+11],
                b[start+12], b[start+13], b[start+14], b[start+15],
            ]);

            let sum = a_chunk + b_chunk;
            let sum_array = sum.to_array();

            result[start..end].copy_from_slice(&sum_array);
        }

        // Handle remaining elements
        for i in chunks * chunk_size..a.len() {
            result[i] = a[i].wrapping_add(b[i]);
        }

        Ok(())
    }

    /// SSE2-optimized polynomial addition
    #[cfg(target_arch = "x86_64")]
    fn poly_add_sse2(a: &[i16], b: &[i16], result: &mut [i16]) -> Result<(), TlsError> {
        let chunk_size = 8; // 8 i16 elements per SSE2 register
        let chunks = a.len() / chunk_size;

        for i in 0..chunks {
            let start = i * chunk_size;
            let end = start + chunk_size;

            let a_chunk = i16x8::new([
                a[start], a[start+1], a[start+2], a[start+3],
                a[start+4], a[start+5], a[start+6], a[start+7],
            ]);

            let b_chunk = i16x8::new([
                b[start], b[start+1], b[start+2], b[start+3],
                b[start+4], b[start+5], b[start+6], b[start+7],
            ]);

            let sum = a_chunk + b_chunk;
            result[start..end].copy_from_slice(&sum.to_array());
        }

        // Handle remaining elements
        for i in chunks * chunk_size..a.len() {
            result[i] = a[i].wrapping_add(b[i]);
        }

        Ok(())
    }

    /// Scalar polynomial addition
    fn poly_add_scalar(a: &[i16], b: &[i16], result: &mut [i16]) -> Result<(), TlsError> {
        for i in 0..a.len() {
            result[i] = a[i].wrapping_add(b[i]);
        }
        Ok(())
    }

    /// Multiply two polynomials using NTT with SIMD optimization
    pub fn poly_mul_ntt_simd(a: &[i16], b: &[i16], result: &mut [i16]) -> Result<(), TlsError> {
        if a.len() != b.len() || a.len() != result.len() {
            return Err(TlsError::Policy("polynomial dimension mismatch".into()));
        }

        // This is a simplified version - real implementation would use optimized NTT
        let n = a.len();
        
        // Forward NTT on both inputs (parallelized)
        let mut a_ntt = vec![0i16; n];
        let mut b_ntt = vec![0i16; n];
        
        rayon::join(
            || Self::forward_ntt(a, &mut a_ntt),
            || Self::forward_ntt(b, &mut b_ntt),
        );

        // Pointwise multiplication with SIMD
        Self::pointwise_mul_simd(&a_ntt, &b_ntt, result)?;

        // Inverse NTT
        Self::inverse_ntt(result)?;

        Ok(())
    }

    /// Forward Number Theoretic Transform
    fn forward_ntt(input: &[i16], output: &mut [i16]) {
        // Placeholder for actual NTT implementation
        output.copy_from_slice(input);
    }

    /// Inverse Number Theoretic Transform
    fn inverse_ntt(data: &mut [i16]) -> Result<(), TlsError> {
        // Placeholder for actual inverse NTT implementation
        Ok(())
    }

    /// SIMD pointwise multiplication
    fn pointwise_mul_simd(a: &[i16], b: &[i16], result: &mut [i16]) -> Result<(), TlsError> {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return Self::pointwise_mul_avx2(a, b, result);
            }
        }

        // Fallback to scalar
        for i in 0..a.len() {
            result[i] = (a[i] as i32 * b[i] as i32) as i16;
        }
        
        Ok(())
    }

    /// AVX2 pointwise multiplication
    #[cfg(target_arch = "x86_64")]
    fn pointwise_mul_avx2(a: &[i16], b: &[i16], result: &mut [i16]) -> Result<(), TlsError> {
        let chunk_size = 8; // Process 8 i16 elements at a time for i32 result
        let chunks = a.len() / chunk_size;

        for i in 0..chunks {
            let start = i * chunk_size;
            
            let a_chunk = i16x8::new([
                a[start], a[start+1], a[start+2], a[start+3],
                a[start+4], a[start+5], a[start+6], a[start+7],
            ]);

            let b_chunk = i16x8::new([
                b[start], b[start+1], b[start+2], b[start+3],
                b[start+4], b[start+5], b[start+6], b[start+7],
            ]);

            // Convert to i32 for multiplication to avoid overflow
            let a_lo = a_chunk.cast::<i32>();
            let b_lo = b_chunk.cast::<i32>();
            let product = a_lo * b_lo;
            
            // Convert back to i16 (with potential truncation)
            let product_i16 = product.cast::<i16>();
            result[start..start + chunk_size].copy_from_slice(&product_i16.to_array());
        }

        // Handle remaining elements
        for i in chunks * chunk_size..a.len() {
            result[i] = (a[i] as i32 * b[i] as i32) as i16;
        }

        Ok(())
    }
}

/// Performance monitoring and profiling
pub struct PerformanceMonitor {
    operation_times: Arc<RwLock<HashMap<String, Vec<Duration>>>>,
    memory_usage: Arc<AtomicU64>,
    operation_counts: Arc<RwLock<HashMap<String, u64>>>,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            operation_times: Arc::new(RwLock::new(HashMap::new())),
            memory_usage: Arc::new(AtomicU64::new(0)),
            operation_counts: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Record the timing of an operation
    pub async fn record_operation(&self, operation: &str, duration: Duration) {
        let mut times = self.operation_times.write().await;
        times.entry(operation.to_string()).or_insert_with(Vec::new).push(duration);

        let mut counts = self.operation_counts.write().await;
        *counts.entry(operation.to_string()).or_insert(0) += 1;

        // Update metrics
        histogram!("pqc_operation_duration_seconds", "operation" => operation)
            .record(duration.as_secs_f64());
        counter!("pqc_operations_total", "operation" => operation).increment(1);
    }

    /// Record memory usage
    pub fn record_memory_usage(&self, bytes: u64) {
        self.memory_usage.store(bytes, Ordering::Relaxed);
        gauge!("pqc_memory_usage_bytes").set(bytes as f64);
    }

    /// Get performance statistics
    pub async fn get_statistics(&self) -> PerformanceStatistics {
        let times = self.operation_times.read().await;
        let counts = self.operation_counts.read().await;
        let memory = self.memory_usage.load(Ordering::Relaxed);

        let mut operation_stats = HashMap::new();
        for (operation, durations) in times.iter() {
            let count = *counts.get(operation).unwrap_or(&0);
            let total_time: Duration = durations.iter().sum();
            let avg_time = if count > 0 {
                total_time / count as u32
            } else {
                Duration::ZERO
            };
            
            let min_time = durations.iter().min().copied().unwrap_or(Duration::ZERO);
            let max_time = durations.iter().max().copied().unwrap_or(Duration::ZERO);

            operation_stats.insert(operation.clone(), OperationStatistics {
                count,
                total_time,
                average_time: avg_time,
                min_time,
                max_time,
            });
        }

        PerformanceStatistics {
            operation_stats,
            current_memory_usage: memory,
            total_operations: counts.values().sum(),
        }
    }

    /// Clear all statistics
    pub async fn clear_statistics(&self) {
        let mut times = self.operation_times.write().await;
        let mut counts = self.operation_counts.write().await;
        times.clear();
        counts.clear();
        self.memory_usage.store(0, Ordering::Relaxed);
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for individual operations
#[derive(Debug, Clone)]
pub struct OperationStatistics {
    pub count: u64,
    pub total_time: Duration,
    pub average_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
}

/// Overall performance statistics
#[derive(Debug)]
pub struct PerformanceStatistics {
    pub operation_stats: HashMap<String, OperationStatistics>,
    pub current_memory_usage: u64,
    pub total_operations: u64,
}

/// Cache-friendly data structures for PQC operations
pub struct OptimizedDataStructures;

impl OptimizedDataStructures {
    /// Create a cache-aligned buffer for cryptographic operations
    pub fn create_aligned_buffer(size: usize, alignment: usize) -> Result<AlignedBuffer, TlsError> {
        let layout = Layout::from_size_align(size, alignment)
            .map_err(|e| TlsError::Io(format!("invalid memory layout: {}", e)))?;

        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(TlsError::Io("memory allocation failed".into()));
        }

        unsafe {
            // Zero out the buffer
            std::ptr::write_bytes(ptr, 0, size);
        }

        Ok(AlignedBuffer {
            ptr: NonNull::new(ptr).unwrap(),
            size,
            layout,
        })
    }

    /// Prefetch data to improve cache performance
    pub fn prefetch_data(data: &[u8]) {
        #[cfg(target_arch = "x86_64")]
        {
            // Use prefetch instructions on x86_64
            for chunk in data.chunks(64) { // Cache line size
                unsafe {
                    std::arch::x86_64::_mm_prefetch(
                        chunk.as_ptr() as *const i8,
                        std::arch::x86_64::_MM_HINT_T0
                    );
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            // No-op on other architectures
            let _ = data;
        }
    }

    /// Optimize memory layout for better cache utilization
    pub fn optimize_layout<T>(data: &mut [T]) {
        // This is a placeholder for actual layout optimization
        // Real implementation would consider:
        // - Data access patterns
        // - Cache line alignment
        // - NUMA topology
        // - Hardware prefetching behavior
        let _ = data;
    }
}

/// Cache-aligned buffer for improved performance
pub struct AlignedBuffer {
    ptr: NonNull<u8>,
    size: usize,
    layout: Layout,
}

impl AlignedBuffer {
    /// Get a mutable slice to the buffer
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size)
        }
    }

    /// Get an immutable slice to the buffer
    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self.ptr.as_ptr(), self.size)
        }
    }

    /// Get the buffer size
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe {
            // Zero out memory before deallocation
            std::ptr::write_bytes(self.ptr.as_ptr(), 0, self.size);
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

/// Performance optimization hints and recommendations
pub struct OptimizationHints;

impl OptimizationHints {
    /// Get system-specific optimization recommendations
    pub fn get_system_recommendations() -> SystemRecommendations {
        let cpu_count = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let has_simd = cfg!(target_arch = "x86_64") && (
            is_x86_feature_detected!("avx2") ||
            is_x86_feature_detected!("avx512f")
        );

        SystemRecommendations {
            optimal_thread_count: cpu_count,
            recommended_batch_size: if cpu_count >= 8 { 32 } else { 16 },
            use_simd_operations: has_simd,
            memory_pool_size: 1024 * cpu_count,
            cache_line_size: 64, // Most common cache line size
            use_hardware_acceleration: has_simd,
        }
    }

    /// Benchmark different configurations to find optimal settings
    pub async fn benchmark_configurations() -> Result<OptimalConfiguration, TlsError> {
        let mut best_config = OptimalConfiguration::default();
        let mut best_throughput = 0.0;

        // Test different batch sizes
        for batch_size in [8, 16, 32, 64, 128] {
            let throughput = Self::benchmark_batch_size(batch_size).await?;
            if throughput > best_throughput {
                best_throughput = throughput;
                best_config.batch_size = batch_size;
            }
        }

        // Test memory pool configurations
        for pool_size in [512, 1024, 2048, 4096] {
            let throughput = Self::benchmark_memory_pool(pool_size).await?;
            if throughput > best_throughput {
                best_config.memory_pool_size = pool_size;
            }
        }

        Ok(best_config)
    }

    /// Benchmark a specific batch size
    async fn benchmark_batch_size(batch_size: usize) -> Result<f64, TlsError> {
        let start = Instant::now();
        let iterations = 100;

        let processor = BatchProcessor::new(batch_size, 4);
        
        // Create test public keys
        let kem = crate::kyber::KyberKEM::new();
        let mut public_keys = Vec::new();
        
        for _ in 0..batch_size {
            let keypair = kem.generate_keypair().await?;
            public_keys.push(keypair.public_key);
        }

        let public_key_refs: Vec<_> = public_keys.iter().collect();

        for _ in 0..iterations {
            let _ = processor.batch_encapsulate(public_key_refs.clone()).await?;
        }

        let elapsed = start.elapsed();
        let operations_per_sec = (iterations * batch_size) as f64 / elapsed.as_secs_f64();
        
        Ok(operations_per_sec)
    }

    /// Benchmark memory pool configuration
    async fn benchmark_memory_pool(pool_size: usize) -> Result<f64, TlsError> {
        let start = Instant::now();
        let iterations = 1000;

        let mut pool = MemoryPool::new(pool_size, 32);

        for _ in 0..iterations {
            let mut buffers = Vec::new();
            
            // Allocate multiple buffers
            for size in [32, 768, 1184, 1088] {
                if let Ok(buffer) = pool.allocate(size) {
                    buffers.push(buffer);
                }
            }
            
            // Buffers are automatically deallocated when dropped
        }

        let elapsed = start.elapsed();
        let allocations_per_sec = (iterations * 4) as f64 / elapsed.as_secs_f64();
        
        Ok(allocations_per_sec)
    }
}

/// System-specific optimization recommendations
#[derive(Debug)]
pub struct SystemRecommendations {
    pub optimal_thread_count: usize,
    pub recommended_batch_size: usize,
    pub use_simd_operations: bool,
    pub memory_pool_size: usize,
    pub cache_line_size: usize,
    pub use_hardware_acceleration: bool,
}

/// Optimal configuration determined by benchmarking
#[derive(Debug)]
pub struct OptimalConfiguration {
    pub batch_size: usize,
    pub memory_pool_size: usize,
    pub thread_count: usize,
    pub use_simd: bool,
}

impl Default for OptimalConfiguration {
    fn default() -> Self {
        Self {
            batch_size: 32,
            memory_pool_size: 1024,
            thread_count: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
            use_simd: cfg!(target_arch = "x86_64"),
        }
    }
}

/// Global performance manager for coordinating optimizations
pub struct GlobalPerformanceManager {
    monitor: Arc<PerformanceMonitor>,
    memory_pool: Arc<Mutex<MemoryPool>>,
    batch_processor: Arc<BatchProcessor>,
    recommendations: SystemRecommendations,
}

impl GlobalPerformanceManager {
    pub fn new() -> Self {
        let recommendations = OptimizationHints::get_system_recommendations();
        
        Self {
            monitor: Arc::new(PerformanceMonitor::new()),
            memory_pool: Arc::new(Mutex::new(MemoryPool::new(
                recommendations.memory_pool_size,
                recommendations.cache_line_size,
            ))),
            batch_processor: Arc::new(BatchProcessor::new(
                recommendations.recommended_batch_size,
                recommendations.optimal_thread_count,
            )),
            recommendations,
        }
    }

    pub fn monitor(&self) -> Arc<PerformanceMonitor> {
        Arc::clone(&self.monitor)
    }

    pub fn memory_pool(&self) -> Arc<Mutex<MemoryPool>> {
        Arc::clone(&self.memory_pool)
    }

    pub fn batch_processor(&self) -> Arc<BatchProcessor> {
        Arc::clone(&self.batch_processor)
    }

    pub fn recommendations(&self) -> &SystemRecommendations {
        &self.recommendations
    }

    /// Initialize performance optimizations
    pub async fn initialize(&self) -> Result<(), TlsError> {
        info!(
            thread_count = self.recommendations.optimal_thread_count,
            batch_size = self.recommendations.recommended_batch_size,
            simd = self.recommendations.use_simd_operations,
            "initialized performance optimizations"
        );

        // Warm up the memory pool
        {
            let mut pool = self.memory_pool.lock().await;
            for size in [32, 768, 1184, 1088] {
                for _ in 0..10 {
                    if let Ok(buffer) = pool.allocate(size) {
                        drop(buffer); // Immediately return to pool
                    }
                }
            }
        }

        Ok(())
    }
}

impl Default for GlobalPerformanceManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[test]
    fn test_memory_pool() {
        let mut pool = MemoryPool::new(10, 32);
        
        // Test allocation
        let buffer = pool.allocate(1024).unwrap();
        assert_eq!(buffer.len(), 1024);
        
        // Test statistics
        let stats = pool.statistics();
        assert_eq!(stats.total_allocations.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_batch_processor() {
        let processor = BatchProcessor::new(4, 2);
        
        // Create test data
        let kem = crate::kyber::KyberKEM::new();
        let keypair1 = kem.generate_keypair().await.unwrap();
        let keypair2 = kem.generate_keypair().await.unwrap();
        
        let public_keys = vec![&keypair1.public_key, &keypair2.public_key];
        
        // Test batch encapsulation
        let results = processor.batch_encapsulate(public_keys).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_simd_operations() {
        let a = vec![1i16; 256];
        let b = vec![2i16; 256];
        let mut result = vec![0i16; 256];
        
        SimdOperations::poly_add_simd(&a, &b, &mut result).unwrap();
        
        // Check results
        for &val in result.iter() {
            assert_eq!(val, 3);
        }
    }

    #[tokio::test]
    async fn test_performance_monitor() {
        let monitor = PerformanceMonitor::new();
        
        // Record some operations
        monitor.record_operation("test_op", Duration::from_millis(10)).await;
        monitor.record_operation("test_op", Duration::from_millis(20)).await;
        
        let stats = monitor.get_statistics().await;
        let test_stats = stats.operation_stats.get("test_op").unwrap();
        
        assert_eq!(test_stats.count, 2);
        assert_eq!(test_stats.total_time, Duration::from_millis(30));
    }

    #[test]
    fn test_aligned_buffer() {
        let mut buffer = OptimizedDataStructures::create_aligned_buffer(1024, 64).unwrap();
        assert_eq!(buffer.len(), 1024);
        
        // Test that we can write to the buffer
        let slice = buffer.as_mut_slice();
        slice[0] = 42;
        assert_eq!(buffer.as_slice()[0], 42);
    }

    #[test]
    fn test_system_recommendations() {
        let recommendations = OptimizationHints::get_system_recommendations();
        assert!(recommendations.optimal_thread_count > 0);
        assert!(recommendations.recommended_batch_size > 0);
        assert_eq!(recommendations.cache_line_size, 64);
    }
}