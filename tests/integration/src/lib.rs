//! CRONOS AI Integration Tests
//! 
//! Comprehensive integration test suite covering all CRONOS AI components
//! including dataplane, control plane, AI engine, policy engine, and
//! Kubernetes orchestration.

pub mod common;
pub mod dataplane;
pub mod controlplane;
pub mod aiengine;
pub mod policy;
pub mod config;
pub mod k8s_operator;
pub mod performance;
pub mod network;
pub mod security;
pub mod e2e;

use std::sync::Once;
use tracing::info;
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

static INIT: Once = Once::new();

/// Initialize test logging and tracing
pub fn init_test_logging() {
    INIT.call_once(|| {
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("info"));
            
        tracing_subscriber::registry()
            .with(fmt::layer().with_test_writer())
            .with(filter)
            .init();
            
        info!("Test logging initialized");
    });
}

/// Test configuration
#[derive(Debug, Clone)]
pub struct TestConfig {
    pub dataplane_endpoint: String,
    pub controlplane_endpoint: String,
    pub aiengine_endpoint: String,
    pub policy_engine_endpoint: String,
    pub metrics_endpoint: String,
    pub k8s_namespace: String,
    pub test_data_dir: String,
    pub timeout_seconds: u64,
    pub performance_test_duration_seconds: u64,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            dataplane_endpoint: std::env::var("DATAPLANE_ENDPOINT")
                .unwrap_or_else(|_| "http://localhost:9090".to_string()),
            controlplane_endpoint: std::env::var("CONTROLPLANE_ENDPOINT")
                .unwrap_or_else(|_| "http://localhost:8080".to_string()),
            aiengine_endpoint: std::env::var("AIENGINE_ENDPOINT")
                .unwrap_or_else(|_| "http://localhost:8000".to_string()),
            policy_engine_endpoint: std::env::var("POLICY_ENGINE_ENDPOINT")
                .unwrap_or_else(|_| "http://localhost:8001".to_string()),
            metrics_endpoint: std::env::var("METRICS_ENDPOINT")
                .unwrap_or_else(|_| "http://localhost:9090/metrics".to_string()),
            k8s_namespace: std::env::var("TEST_NAMESPACE")
                .unwrap_or_else(|_| "cronos-ai-test".to_string()),
            test_data_dir: std::env::var("TEST_DATA_DIR")
                .unwrap_or_else(|_| "./test_data".to_string()),
            timeout_seconds: std::env::var("TEST_TIMEOUT_SECONDS")
                .unwrap_or_else(|_| "30".to_string())
                .parse()
                .unwrap_or(30),
            performance_test_duration_seconds: std::env::var("PERFORMANCE_TEST_DURATION")
                .unwrap_or_else(|_| "60".to_string())
                .parse()
                .unwrap_or(60),
        }
    }
}

/// Test results aggregation
#[derive(Debug, Default)]
pub struct TestResults {
    pub total: u32,
    pub passed: u32,
    pub failed: u32,
    pub skipped: u32,
    pub duration_ms: u64,
    pub failures: Vec<String>,
}

impl TestResults {
    pub fn success_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.passed as f64) / (self.total as f64) * 100.0
        }
    }
    
    pub fn is_success(&self) -> bool {
        self.failed == 0
    }
    
    pub fn add_failure(&mut self, test_name: &str, error: &str) {
        self.failed += 1;
        self.failures.push(format!("{}: {}", test_name, error));
    }
    
    pub fn merge(&mut self, other: TestResults) {
        self.total += other.total;
        self.passed += other.passed;
        self.failed += other.failed;
        self.skipped += other.skipped;
        self.duration_ms += other.duration_ms;
        self.failures.extend(other.failures);
    }
    
    pub fn summary(&self) -> String {
        format!(
            "Tests: {} total, {} passed ({:.1}%), {} failed, {} skipped. Duration: {}ms",
            self.total,
            self.passed,
            self.success_rate(),
            self.failed,
            self.skipped,
            self.duration_ms
        )
    }
}

/// Integration test trait for all test suites
#[async_trait::async_trait]
pub trait IntegrationTest {
    async fn setup(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    async fn run_tests(&mut self) -> TestResults;
    async fn cleanup(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}

/// Macro for creating integration tests
#[macro_export]
macro_rules! integration_test {
    ($name:ident, $test_fn:expr) => {
        #[tokio::test]
        async fn $name() {
            $crate::init_test_logging();
            
            let start = std::time::Instant::now();
            let result = $test_fn.await;
            let duration = start.elapsed();
            
            match result {
                Ok(_) => {
                    tracing::info!("Test {} passed in {:?}", stringify!($name), duration);
                }
                Err(e) => {
                    tracing::error!("Test {} failed in {:?}: {}", stringify!($name), duration, e);
                    panic!("Test failed: {}", e);
                }
            }
        }
    };
}

/// Macro for creating performance benchmark tests
#[macro_export]
macro_rules! performance_test {
    ($name:ident, $setup:expr, $test_fn:expr, $expected_ops_per_sec:expr) => {
        #[tokio::test]
        async fn $name() {
            $crate::init_test_logging();
            
            // Setup
            $setup.await.expect("Setup failed");
            
            let start = std::time::Instant::now();
            let operations = $test_fn.await.expect("Performance test failed");
            let duration = start.elapsed();
            
            let ops_per_sec = operations as f64 / duration.as_secs_f64();
            
            tracing::info!(
                "Performance test {}: {} operations in {:?} ({:.0} ops/sec)",
                stringify!($name),
                operations,
                duration,
                ops_per_sec
            );
            
            assert!(
                ops_per_sec >= $expected_ops_per_sec,
                "Performance below threshold: {:.0} ops/sec < {} ops/sec",
                ops_per_sec,
                $expected_ops_per_sec
            );
        }
    };
}

/// Utility functions for tests
pub mod utils {
    use super::*;
    use std::time::Duration;
    use tokio::time::timeout;
    
    /// Wait for a condition to be true with timeout
    pub async fn wait_for_condition<F, Fut>(
        condition: F,
        timeout_duration: Duration,
        check_interval: Duration,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = bool>,
    {
        let start = std::time::Instant::now();
        
        while start.elapsed() < timeout_duration {
            if condition().await {
                return Ok(());
            }
            tokio::time::sleep(check_interval).await;
        }
        
        Err("Condition timeout".into())
    }
    
    /// Send HTTP request with timeout
    pub async fn http_request_with_timeout(
        client: &reqwest::Client,
        request: reqwest::RequestBuilder,
        timeout_duration: Duration,
    ) -> Result<reqwest::Response, Box<dyn std::error::Error + Send + Sync>> {
        let response = timeout(timeout_duration, request.send()).await??;
        Ok(response)
    }
    
    /// Generate test data
    pub fn generate_test_packets(count: usize) -> Vec<Vec<u8>> {
        (0..count)
            .map(|i| {
                // Simple test packet with sequence number
                let mut packet = vec![0u8; 64];
                packet[0..4].copy_from_slice(&(i as u32).to_be_bytes());
                packet
            })
            .collect()
    }
    
    /// Create temporary test directory
    pub async fn create_test_dir() -> Result<tempfile::TempDir, std::io::Error> {
        tempfile::TempDir::new()
    }
    
    /// Load test configuration from file
    pub fn load_test_config() -> TestConfig {
        if let Ok(config_path) = std::env::var("TEST_CONFIG_PATH") {
            if let Ok(content) = std::fs::read_to_string(config_path) {
                if let Ok(config) = serde_yaml::from_str::<TestConfig>(&content) {
                    return config;
                }
            }
        }
        TestConfig::default()
    }
    
    /// Retry operation with exponential backoff
    pub async fn retry_with_backoff<F, Fut, T, E>(
        operation: F,
        max_attempts: u32,
        initial_delay: Duration,
    ) -> Result<T, E>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T, E>>,
        E: std::fmt::Debug,
    {
        let mut delay = initial_delay;
        
        for attempt in 1..=max_attempts {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    if attempt == max_attempts {
                        return Err(e);
                    }
                    tracing::warn!("Attempt {} failed: {:?}, retrying in {:?}", attempt, e, delay);
                    tokio::time::sleep(delay).await;
                    delay *= 2;
                }
            }
        }
        
        unreachable!()
    }
    
    /// Check if port is available
    pub async fn is_port_available(port: u16) -> bool {
        use tokio::net::TcpListener;
        TcpListener::bind(format!("127.0.0.1:{}", port))
            .await
            .is_ok()
    }
    
    /// Find available port starting from a base port
    pub async fn find_available_port(base_port: u16) -> u16 {
        for port in base_port..base_port + 1000 {
            if is_port_available(port).await {
                return port;
            }
        }
        panic!("No available ports found");
    }
    
    /// Parse Prometheus metrics
    pub fn parse_prometheus_metrics(content: &str) -> std::collections::HashMap<String, f64> {
        let mut metrics = std::collections::HashMap::new();
        
        for line in content.lines() {
            if line.starts_with('#') || line.is_empty() {
                continue;
            }
            
            if let Some(space_pos) = line.find(' ') {
                let metric_name = &line[..space_pos];
                if let Ok(value) = line[space_pos + 1..].parse::<f64>() {
                    metrics.insert(metric_name.to_string(), value);
                }
            }
        }
        
        metrics
    }
    
    /// Generate random test data
    pub fn generate_random_data(size: usize) -> Vec<u8> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut data = Vec::with_capacity(size);
        let mut hasher = DefaultHasher::new();
        
        for i in 0..size {
            i.hash(&mut hasher);
            data.push((hasher.finish() & 0xFF) as u8);
        }
        
        data
    }
    
    /// Compare floating point numbers with tolerance
    pub fn approx_equal(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() < tolerance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_results_success_rate() {
        let mut results = TestResults::default();
        results.total = 10;
        results.passed = 8;
        results.failed = 2;
        
        assert_eq!(results.success_rate(), 80.0);
        assert!(!results.is_success());
    }
    
    #[test]
    fn test_results_merge() {
        let mut results1 = TestResults {
            total: 5,
            passed: 4,
            failed: 1,
            skipped: 0,
            duration_ms: 1000,
            failures: vec!["test1: error".to_string()],
        };
        
        let results2 = TestResults {
            total: 3,
            passed: 2,
            failed: 1,
            skipped: 0,
            duration_ms: 500,
            failures: vec!["test2: error".to_string()],
        };
        
        results1.merge(results2);
        
        assert_eq!(results1.total, 8);
        assert_eq!(results1.passed, 6);
        assert_eq!(results1.failed, 2);
        assert_eq!(results1.duration_ms, 1500);
        assert_eq!(results1.failures.len(), 2);
    }
    
    #[tokio::test]
    async fn test_wait_for_condition() {
        let mut counter = 0;
        let condition = || {
            counter += 1;
            async move { counter >= 3 }
        };
        
        let result = utils::wait_for_condition(
            condition,
            Duration::from_secs(1),
            Duration::from_millis(10),
        ).await;
        
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_generate_test_packets() {
        let packets = utils::generate_test_packets(5);
        assert_eq!(packets.len(), 5);
        
        for (i, packet) in packets.iter().enumerate() {
            assert_eq!(packet.len(), 64);
            let seq = u32::from_be_bytes([packet[0], packet[1], packet[2], packet[3]]);
            assert_eq!(seq, i as u32);
        }
    }
    
    #[test]
    fn test_parse_prometheus_metrics() {
        let content = r#"
# HELP cronos_ai_packets_total Total packets processed
# TYPE cronos_ai_packets_total counter
cronos_ai_packets_total{component="dataplane"} 12345
cronos_ai_cpu_usage 0.75
"#;
        
        let metrics = utils::parse_prometheus_metrics(content);
        assert_eq!(metrics.len(), 2);
        assert_eq!(metrics.get("cronos_ai_packets_total{component=\"dataplane\"}"), Some(&12345.0));
        assert_eq!(metrics.get("cronos_ai_cpu_usage"), Some(&0.75));
    }
}