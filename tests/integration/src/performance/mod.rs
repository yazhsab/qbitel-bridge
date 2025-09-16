//! Performance and benchmark tests for CRONOS AI components

use crate::{TestConfig, TestResults, IntegrationTest, utils};
use std::time::{Duration, Instant};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::Semaphore;
use reqwest::Client;
use serde_json::{json, Value};
use tracing::{info, warn, error};

pub mod dataplane_performance;
pub mod aiengine_performance;
pub mod policy_performance;
pub mod memory_usage;

/// Performance test suite for all CRONOS AI components
pub struct PerformanceTestSuite {
    config: TestConfig,
    client: Client,
    test_duration: Duration,
}

impl PerformanceTestSuite {
    pub fn new(config: TestConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .build()
            .expect("Failed to create HTTP client");
            
        let test_duration = Duration::from_secs(config.performance_test_duration_seconds);
        
        Self {
            config,
            client,
            test_duration,
        }
    }
    
    /// Run all performance benchmarks
    pub async fn run_all_benchmarks(&mut self) -> TestResults {
        let mut results = TestResults::default();
        let start = Instant::now();
        
        info!("Starting comprehensive performance benchmarks");
        
        // Benchmark 1: Packet Processing Throughput
        match self.benchmark_packet_processing_throughput().await {
            Ok(throughput) => {
                results.passed += 1;
                info!("✅ Packet processing throughput: {:.0} packets/sec", throughput);
            }
            Err(e) => {
                results.add_failure("packet_processing_throughput", &e.to_string());
                error!("❌ Packet processing throughput benchmark failed: {}", e);
            }
        }
        results.total += 1;
        
        // Benchmark 2: AI Inference Latency
        match self.benchmark_ai_inference_latency().await {
            Ok(latency) => {
                results.passed += 1;
                info!("✅ AI inference latency: {:.2}ms average", latency);
            }
            Err(e) => {
                results.add_failure("ai_inference_latency", &e.to_string());
                error!("❌ AI inference latency benchmark failed: {}", e);
            }
        }
        results.total += 1;
        
        // Benchmark 3: Policy Evaluation Performance
        match self.benchmark_policy_evaluation().await {
            Ok(eval_rate) => {
                results.passed += 1;
                info!("✅ Policy evaluation rate: {:.0} evaluations/sec", eval_rate);
            }
            Err(e) => {
                results.add_failure("policy_evaluation_performance", &e.to_string());
                error!("❌ Policy evaluation benchmark failed: {}", e);
            }
        }
        results.total += 1;
        
        // Benchmark 4: Configuration Access Performance
        match self.benchmark_configuration_access().await {
            Ok(access_rate) => {
                results.passed += 1;
                info!("✅ Configuration access rate: {:.0} ops/sec", access_rate);
            }
            Err(e) => {
                results.add_failure("configuration_access_performance", &e.to_string());
                error!("❌ Configuration access benchmark failed: {}", e);
            }
        }
        results.total += 1;
        
        // Benchmark 5: High-Frequency Metrics Performance
        match self.benchmark_metrics_collection().await {
            Ok(metrics_rate) => {
                results.passed += 1;
                info!("✅ Metrics collection rate: {:.0} samples/sec", metrics_rate);
            }
            Err(e) => {
                results.add_failure("metrics_collection_performance", &e.to_string());
                error!("❌ Metrics collection benchmark failed: {}", e);
            }
        }
        results.total += 1;
        
        // Benchmark 6: Memory Usage Under Load
        match self.benchmark_memory_usage().await {
            Ok(peak_memory_mb) => {
                results.passed += 1;
                info!("✅ Peak memory usage under load: {:.0}MB", peak_memory_mb);
            }
            Err(e) => {
                results.add_failure("memory_usage_benchmark", &e.to_string());
                error!("❌ Memory usage benchmark failed: {}", e);
            }
        }
        results.total += 1;
        
        // Benchmark 7: Concurrent Connection Handling
        match self.benchmark_concurrent_connections().await {
            Ok(max_connections) => {
                results.passed += 1;
                info!("✅ Maximum concurrent connections: {}", max_connections);
            }
            Err(e) => {
                results.add_failure("concurrent_connections_benchmark", &e.to_string());
                error!("❌ Concurrent connections benchmark failed: {}", e);
            }
        }
        results.total += 1;
        
        results.duration_ms = start.elapsed().as_millis() as u64;
        
        info!(
            "Performance benchmarks completed: {}",
            results.summary()
        );
        
        results
    }
    
    /// Benchmark packet processing throughput
    async fn benchmark_packet_processing_throughput(&self) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        info!("Benchmarking packet processing throughput...");
        
        let packets_per_batch = 1000;
        let total_packets = Arc::new(AtomicU64::new(0));
        let semaphore = Arc::new(Semaphore::new(10)); // Limit concurrent requests
        
        let start_time = Instant::now();
        let mut handles = Vec::new();
        
        // Generate test load for the specified duration
        while start_time.elapsed() < self.test_duration {
            let permit = semaphore.clone().acquire_owned().await?;
            let client = self.client.clone();
            let endpoint = self.config.dataplane_endpoint.clone();
            let total_packets = total_packets.clone();
            
            let handle = tokio::spawn(async move {
                let _permit = permit;
                
                // Generate batch of test packets
                let test_packets = utils::generate_test_packets(packets_per_batch);
                let payload = json!({
                    "packets": test_packets.iter().map(|p| base64::encode(p)).collect::<Vec<_>>(),
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                    "batch_id": uuid::Uuid::new_v4().to_string()
                });
                
                match client
                    .post(&format!("{}/api/v1/packets/process", endpoint))
                    .json(&payload)
                    .send()
                    .await
                {
                    Ok(response) if response.status().is_success() => {
                        if let Ok(result) = response.json::<Value>().await {
                            if let Some(processed) = result["processed_count"].as_u64() {
                                total_packets.fetch_add(processed, Ordering::Relaxed);
                            }
                        }
                    }
                    Ok(response) => {
                        warn!("Packet processing request failed with status: {}", response.status());
                    }
                    Err(e) => {
                        warn!("Packet processing request error: {}", e);
                    }
                }
            });
            
            handles.push(handle);
            
            // Small delay to avoid overwhelming the system
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        
        // Wait for all requests to complete
        for handle in handles {
            let _ = handle.await;
        }
        
        let total_processed = total_packets.load(Ordering::Relaxed);
        let elapsed = start_time.elapsed();
        let throughput = total_processed as f64 / elapsed.as_secs_f64();
        
        // Verify minimum throughput requirement
        let min_throughput = 10000.0; // 10K packets/sec minimum
        if throughput < min_throughput {
            return Err(format!(
                "Throughput below minimum requirement: {:.0} < {:.0} packets/sec",
                throughput, min_throughput
            ).into());
        }
        
        Ok(throughput)
    }
    
    /// Benchmark AI inference latency
    async fn benchmark_ai_inference_latency(&self) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        info!("Benchmarking AI inference latency...");
        
        let num_requests = 1000;
        let mut latencies = Vec::with_capacity(num_requests);
        
        // Test protocol classification latency
        for _ in 0..num_requests {
            let test_data = json!({
                "flow_features": {
                    "packet_sizes": [64, 128, 256, 512, 1024],
                    "inter_arrival_times": [0.001, 0.002, 0.0015, 0.003, 0.0025],
                    "flags": ["SYN", "ACK", "PSH"],
                    "payload_entropy": fastrand::f64(),
                    "flow_duration": fastrand::f64() * 10.0
                },
                "model": "protocol_classifier"
            });
            
            let start = Instant::now();
            
            let response = self.client
                .post(&format!("{}/api/v1/inference/classify", self.config.aiengine_endpoint))
                .json(&test_data)
                .send()
                .await?;
            
            if response.status().is_success() {
                let _result: Value = response.json().await?;
                latencies.push(start.elapsed().as_secs_f64() * 1000.0); // Convert to milliseconds
            }
        }
        
        if latencies.is_empty() {
            return Err("No successful inference requests".into());
        }
        
        // Calculate statistics
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let p95_latency = latencies[(latencies.len() * 95 / 100).min(latencies.len() - 1)];
        let p99_latency = latencies[(latencies.len() * 99 / 100).min(latencies.len() - 1)];
        
        info!(
            "AI inference latency - Avg: {:.2}ms, P95: {:.2}ms, P99: {:.2}ms",
            avg_latency, p95_latency, p99_latency
        );
        
        // Verify latency requirements
        let max_avg_latency = 50.0; // 50ms average
        let max_p95_latency = 100.0; // 100ms P95
        
        if avg_latency > max_avg_latency {
            return Err(format!(
                "Average latency too high: {:.2}ms > {:.2}ms",
                avg_latency, max_avg_latency
            ).into());
        }
        
        if p95_latency > max_p95_latency {
            return Err(format!(
                "P95 latency too high: {:.2}ms > {:.2}ms",
                p95_latency, max_p95_latency
            ).into());
        }
        
        Ok(avg_latency)
    }
    
    /// Benchmark policy evaluation performance
    async fn benchmark_policy_evaluation(&self) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        info!("Benchmarking policy evaluation performance...");
        
        let total_evaluations = Arc::new(AtomicU64::new(0));
        let semaphore = Arc::new(Semaphore::new(20)); // Higher concurrency for policy evaluation
        let start_time = Instant::now();
        let mut handles = Vec::new();
        
        while start_time.elapsed() < self.test_duration {
            let permit = semaphore.clone().acquire_owned().await?;
            let client = self.client.clone();
            let endpoint = self.config.policy_engine_endpoint.clone();
            let total_evaluations = total_evaluations.clone();
            
            let handle = tokio::spawn(async move {
                let _permit = permit;
                
                // Generate random policy evaluation request
                let policy_data = json!({
                    "resource": {
                        "type": "network_connection",
                        "source_ip": format!("10.0.{}.{}", fastrand::u8(..), fastrand::u8(..)),
                        "dest_ip": format!("192.168.{}.{}", fastrand::u8(..), fastrand::u8(..)),
                        "dest_port": fastrand::u16(1024..65535),
                        "protocol": if fastrand::bool() { "TCP" } else { "UDP" }
                    },
                    "context": {
                        "user": format!("user_{}", fastrand::u32(..)),
                        "time": chrono::Utc::now().to_rfc3339(),
                        "classification": ["web_traffic", "database", "api_call", "file_transfer"][fastrand::usize(..4)]
                    }
                });
                
                match client
                    .post(&format!("{}/api/v1/policy/evaluate", endpoint))
                    .json(&policy_data)
                    .send()
                    .await
                {
                    Ok(response) if response.status().is_success() => {
                        total_evaluations.fetch_add(1, Ordering::Relaxed);
                    }
                    Ok(response) => {
                        warn!("Policy evaluation failed with status: {}", response.status());
                    }
                    Err(e) => {
                        warn!("Policy evaluation error: {}", e);
                    }
                }
            });
            
            handles.push(handle);
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        
        // Wait for completion
        for handle in handles {
            let _ = handle.await;
        }
        
        let total_evals = total_evaluations.load(Ordering::Relaxed);
        let elapsed = start_time.elapsed();
        let eval_rate = total_evals as f64 / elapsed.as_secs_f64();
        
        // Verify minimum evaluation rate
        let min_eval_rate = 1000.0; // 1000 evaluations/sec minimum
        if eval_rate < min_eval_rate {
            return Err(format!(
                "Policy evaluation rate below minimum: {:.0} < {:.0} evals/sec",
                eval_rate, min_eval_rate
            ).into());
        }
        
        Ok(eval_rate)
    }
    
    /// Benchmark configuration access performance
    async fn benchmark_configuration_access(&self) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        info!("Benchmarking configuration access performance...");
        
        let total_operations = Arc::new(AtomicU64::new(0));
        let start_time = Instant::now();
        let mut handles = Vec::new();
        
        // Mix of read and write operations
        let config_keys = [
            "dataplane.packet_processing.batch_size",
            "aiengine.model_management.cache_size",
            "policy.evaluation.timeout_ms",
            "monitoring.metrics.collection_interval",
            "security.tls.cipher_suites"
        ];
        
        while start_time.elapsed() < self.test_duration {
            for &key in &config_keys {
                let client = self.client.clone();
                let endpoint = self.config.controlplane_endpoint.clone();
                let total_operations = total_operations.clone();
                let key = key.to_string();
                
                // 80% reads, 20% writes
                let is_write = fastrand::f32() < 0.2;
                
                let handle = tokio::spawn(async move {
                    let result = if is_write {
                        // Write operation
                        let update_data = json!({
                            "value": fastrand::u64(100..10000),
                            "changed_by": "performance_test",
                            "reason": "Performance testing"
                        });
                        
                        client
                            .put(&format!("{}/api/v1/config/{}", endpoint, key))
                            .json(&update_data)
                            .send()
                            .await
                    } else {
                        // Read operation
                        client
                            .get(&format!("{}/api/v1/config/{}", endpoint, key))
                            .send()
                            .await
                    };
                    
                    match result {
                        Ok(response) if response.status().is_success() => {
                            total_operations.fetch_add(1, Ordering::Relaxed);
                        }
                        _ => {}
                    }
                });
                
                handles.push(handle);
            }
            
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        
        // Wait for completion
        for handle in handles {
            let _ = handle.await;
        }
        
        let total_ops = total_operations.load(Ordering::Relaxed);
        let elapsed = start_time.elapsed();
        let ops_rate = total_ops as f64 / elapsed.as_secs_f64();
        
        Ok(ops_rate)
    }
    
    /// Benchmark metrics collection performance
    async fn benchmark_metrics_collection(&self) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        info!("Benchmarking metrics collection performance...");
        
        let total_samples = Arc::new(AtomicU64::new(0));
        let start_time = Instant::now();
        let mut handles = Vec::new();
        
        // Test high-frequency metrics endpoint
        while start_time.elapsed() < self.test_duration {
            let client = self.client.clone();
            let endpoint = self.config.dataplane_endpoint.clone();
            let total_samples = total_samples.clone();
            
            let handle = tokio::spawn(async move {
                match client
                    .get(&format!("{}/api/v1/metrics/high-frequency", endpoint))
                    .send()
                    .await
                {
                    Ok(response) if response.status().is_success() => {
                        if let Ok(metrics) = response.json::<Value>().await {
                            if let Some(samples) = metrics["latest_samples"].as_array() {
                                total_samples.fetch_add(samples.len() as u64, Ordering::Relaxed);
                            }
                        }
                    }
                    _ => {}
                }
            });
            
            handles.push(handle);
            tokio::time::sleep(Duration::from_millis(100)).await; // 10Hz sampling
        }
        
        for handle in handles {
            let _ = handle.await;
        }
        
        let samples = total_samples.load(Ordering::Relaxed);
        let elapsed = start_time.elapsed();
        let sample_rate = samples as f64 / elapsed.as_secs_f64();
        
        Ok(sample_rate)
    }
    
    /// Benchmark memory usage under load
    async fn benchmark_memory_usage(&self) -> Result<f64, Box<dyn std::error::Error + Send + Sync>> {
        info!("Benchmarking memory usage under load...");
        
        let mut peak_memory_mb = 0.0;
        let start_time = Instant::now();
        
        // Generate sustained load while monitoring memory
        let load_task = {
            let client = self.client.clone();
            let endpoint = self.config.dataplane_endpoint.clone();
            
            tokio::spawn(async move {
                while start_time.elapsed() < Duration::from_secs(30) {
                    let test_packets = utils::generate_test_packets(500);
                    let payload = json!({
                        "packets": test_packets.iter().map(|p| base64::encode(p)).collect::<Vec<_>>()
                    });
                    
                    let _ = client
                        .post(&format!("{}/api/v1/packets/process", endpoint))
                        .json(&payload)
                        .send()
                        .await;
                    
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
            })
        };
        
        // Monitor memory usage
        let memory_monitor = {
            let client = self.client.clone();
            let metrics_endpoint = self.config.metrics_endpoint.clone();
            
            tokio::spawn(async move {
                let mut max_memory = 0.0;
                
                while start_time.elapsed() < Duration::from_secs(30) {
                    if let Ok(response) = client.get(&metrics_endpoint).send().await {
                        if let Ok(metrics_text) = response.text().await {
                            let metrics = utils::parse_prometheus_metrics(&metrics_text);
                            
                            // Look for memory usage metrics
                            for (metric_name, value) in metrics {
                                if metric_name.contains("memory_usage_bytes") {
                                    let memory_mb = value / 1024.0 / 1024.0;
                                    max_memory = max_memory.max(memory_mb);
                                }
                            }
                        }
                    }
                    
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
                
                max_memory
            })
        };
        
        // Wait for both tasks
        let _ = load_task.await;
        peak_memory_mb = memory_monitor.await.unwrap_or(0.0);
        
        if peak_memory_mb == 0.0 {
            return Err("Unable to measure memory usage".into());
        }
        
        // Verify memory usage is within acceptable bounds
        let max_memory_mb = 2048.0; // 2GB maximum
        if peak_memory_mb > max_memory_mb {
            return Err(format!(
                "Memory usage too high: {:.0}MB > {:.0}MB",
                peak_memory_mb, max_memory_mb
            ).into());
        }
        
        Ok(peak_memory_mb)
    }
    
    /// Benchmark concurrent connection handling
    async fn benchmark_concurrent_connections(&self) -> Result<u32, Box<dyn std::error::Error + Send + Sync>> {
        info!("Benchmarking concurrent connection handling...");
        
        let mut successful_connections = 0;
        let max_concurrent = 1000;
        
        // Create many concurrent connections
        let mut handles = Vec::new();
        
        for i in 0..max_concurrent {
            let client = Client::builder()
                .timeout(Duration::from_secs(10))
                .build()?;
            let endpoint = self.config.dataplane_endpoint.clone();
            
            let handle = tokio::spawn(async move {
                // Hold connection open briefly
                match client.get(&format!("{}/health", endpoint)).send().await {
                    Ok(response) if response.status().is_success() => {
                        tokio::time::sleep(Duration::from_millis(100)).await;
                        true
                    }
                    _ => false
                }
            });
            
            handles.push(handle);
            
            // Small delay between connection attempts
            if i % 50 == 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
        
        // Wait for all connections
        for handle in handles {
            if let Ok(success) = handle.await {
                if success {
                    successful_connections += 1;
                }
            }
        }
        
        Ok(successful_connections)
    }
}

#[async_trait::async_trait]
impl IntegrationTest for PerformanceTestSuite {
    async fn setup(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Setting up performance test environment");
        
        // Ensure all services are ready and warmed up
        let warmup_requests = 10;
        
        for _ in 0..warmup_requests {
            let test_packet = utils::generate_test_packets(1);
            let payload = json!({
                "packets": [base64::encode(&test_packet[0])]
            });
            
            let _ = self.client
                .post(&format!("{}/api/v1/packets/process", self.config.dataplane_endpoint))
                .json(&payload)
                .send()
                .await;
            
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        
        info!("Performance test environment ready");
        Ok(())
    }
    
    async fn run_tests(&mut self) -> TestResults {
        self.run_all_benchmarks().await
    }
    
    async fn cleanup(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Cleaning up performance test environment");
        
        // Allow system to settle after performance tests
        tokio::time::sleep(Duration::from_secs(5)).await;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_performance_suite_creation() {
        let config = TestConfig::default();
        let suite = PerformanceTestSuite::new(config);
        assert!(suite.test_duration.as_secs() > 0);
    }
}