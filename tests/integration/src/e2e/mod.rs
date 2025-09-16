//! End-to-end integration tests for CRONOS AI system

use crate::{TestConfig, TestResults, IntegrationTest, utils};
use std::time::{Duration, Instant};
use tokio::time::timeout;
use reqwest::Client;
use serde_json::{json, Value};
use tracing::{info, warn, error};

pub mod dataplane_tests;
pub mod controlplane_tests;
pub mod aiengine_tests;
pub mod policy_tests;
pub mod performance_tests;

/// End-to-end test suite orchestrator
pub struct E2ETestSuite {
    config: TestConfig,
    client: Client,
    test_namespace: String,
}

impl E2ETestSuite {
    pub fn new(config: TestConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_seconds))
            .build()
            .expect("Failed to create HTTP client");
            
        Self {
            test_namespace: config.k8s_namespace.clone(),
            config,
            client,
        }
    }
    
    /// Run complete end-to-end test scenario
    pub async fn run_complete_scenario(&mut self) -> TestResults {
        let mut results = TestResults::default();
        let start = Instant::now();
        
        info!("Starting complete end-to-end test scenario");
        
        // Test 1: System Health Check
        match self.test_system_health().await {
            Ok(_) => {
                results.passed += 1;
                info!("✅ System health check passed");
            }
            Err(e) => {
                results.add_failure("system_health_check", &e.to_string());
                error!("❌ System health check failed: {}", e);
            }
        }
        results.total += 1;
        
        // Test 2: Packet Processing Pipeline
        match self.test_packet_processing_pipeline().await {
            Ok(_) => {
                results.passed += 1;
                info!("✅ Packet processing pipeline test passed");
            }
            Err(e) => {
                results.add_failure("packet_processing_pipeline", &e.to_string());
                error!("❌ Packet processing pipeline test failed: {}", e);
            }
        }
        results.total += 1;
        
        // Test 3: AI Model Inference
        match self.test_ai_model_inference().await {
            Ok(_) => {
                results.passed += 1;
                info!("✅ AI model inference test passed");
            }
            Err(e) => {
                results.add_failure("ai_model_inference", &e.to_string());
                error!("❌ AI model inference test failed: {}", e);
            }
        }
        results.total += 1;
        
        // Test 4: Policy Enforcement
        match self.test_policy_enforcement().await {
            Ok(_) => {
                results.passed += 1;
                info!("✅ Policy enforcement test passed");
            }
            Err(e) => {
                results.add_failure("policy_enforcement", &e.to_string());
                error!("❌ Policy enforcement test failed: {}", e);
            }
        }
        results.total += 1;
        
        // Test 5: Configuration Management
        match self.test_configuration_management().await {
            Ok(_) => {
                results.passed += 1;
                info!("✅ Configuration management test passed");
            }
            Err(e) => {
                results.add_failure("configuration_management", &e.to_string());
                error!("❌ Configuration management test failed: {}", e);
            }
        }
        results.total += 1;
        
        // Test 6: High-Frequency Metrics
        match self.test_high_frequency_metrics().await {
            Ok(_) => {
                results.passed += 1;
                info!("✅ High-frequency metrics test passed");
            }
            Err(e) => {
                results.add_failure("high_frequency_metrics", &e.to_string());
                error!("❌ High-frequency metrics test failed: {}", e);
            }
        }
        results.total += 1;
        
        // Test 7: Failover and Recovery
        match self.test_failover_recovery().await {
            Ok(_) => {
                results.passed += 1;
                info!("✅ Failover and recovery test passed");
            }
            Err(e) => {
                results.add_failure("failover_recovery", &e.to_string());
                error!("❌ Failover and recovery test failed: {}", e);
            }
        }
        results.total += 1;
        
        // Test 8: Load Testing
        match self.test_load_handling().await {
            Ok(_) => {
                results.passed += 1;
                info!("✅ Load testing passed");
            }
            Err(e) => {
                results.add_failure("load_testing", &e.to_string());
                error!("❌ Load testing failed: {}", e);
            }
        }
        results.total += 1;
        
        results.duration_ms = start.elapsed().as_millis() as u64;
        
        info!(
            "End-to-end test scenario completed: {}",
            results.summary()
        );
        
        results
    }
    
    /// Test system health across all components
    async fn test_system_health(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Testing system health...");
        
        // Check dataplane health
        let response = self.client
            .get(&format!("{}/health", self.config.dataplane_endpoint))
            .send()
            .await?;
        assert_eq!(response.status(), 200, "Dataplane health check failed");
        
        // Check control plane health
        let response = self.client
            .get(&format!("{}/health", self.config.controlplane_endpoint))
            .send()
            .await?;
        assert_eq!(response.status(), 200, "Control plane health check failed");
        
        // Check AI engine health
        let response = self.client
            .get(&format!("{}/health", self.config.aiengine_endpoint))
            .send()
            .await?;
        assert_eq!(response.status(), 200, "AI engine health check failed");
        
        // Check policy engine health
        let response = self.client
            .get(&format!("{}/health", self.config.policy_engine_endpoint))
            .send()
            .await?;
        assert_eq!(response.status(), 200, "Policy engine health check failed");
        
        Ok(())
    }
    
    /// Test complete packet processing pipeline
    async fn test_packet_processing_pipeline(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Testing packet processing pipeline...");
        
        // Generate test packets
        let test_packets = utils::generate_test_packets(100);
        
        // Submit packets to dataplane
        let payload = json!({
            "packets": test_packets.iter().map(|p| base64::encode(p)).collect::<Vec<_>>(),
            "timestamp": chrono::Utc::now().to_rfc3339()
        });
        
        let response = self.client
            .post(&format!("{}/api/v1/packets/process", self.config.dataplane_endpoint))
            .json(&payload)
            .send()
            .await?;
        
        assert_eq!(response.status(), 200, "Packet processing request failed");
        
        let result: Value = response.json().await?;
        let processed_count = result["processed_count"].as_u64().unwrap_or(0);
        
        assert_eq!(processed_count, 100, "Not all packets were processed");
        
        // Verify packets were analyzed by DPI engine
        tokio::time::sleep(Duration::from_millis(500)).await;
        
        let response = self.client
            .get(&format!("{}/api/v1/dpi/stats", self.config.dataplane_endpoint))
            .send()
            .await?;
        
        let stats: Value = response.json().await?;
        let analyzed_count = stats["analyzed_packets"].as_u64().unwrap_or(0);
        
        assert!(analyzed_count > 0, "No packets were analyzed by DPI engine");
        
        Ok(())
    }
    
    /// Test AI model inference functionality
    async fn test_ai_model_inference(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Testing AI model inference...");
        
        // Test protocol classification
        let test_data = json!({
            "flow_features": {
                "packet_sizes": [64, 128, 256, 512],
                "inter_arrival_times": [0.001, 0.002, 0.0015, 0.003],
                "flags": ["SYN", "ACK"],
                "payload_entropy": 0.75
            },
            "model": "protocol_classifier"
        });
        
        let response = self.client
            .post(&format!("{}/api/v1/inference/classify", self.config.aiengine_endpoint))
            .json(&test_data)
            .send()
            .await?;
        
        assert_eq!(response.status(), 200, "AI inference request failed");
        
        let result: Value = response.json().await?;
        assert!(result["classification"].is_string(), "No classification result");
        assert!(result["confidence"].is_number(), "No confidence score");
        
        let confidence = result["confidence"].as_f64().unwrap();
        assert!(confidence >= 0.0 && confidence <= 1.0, "Invalid confidence score");
        
        // Test anomaly detection
        let anomaly_data = json!({
            "metrics": {
                "packet_rate": 50000,
                "byte_rate": 64000000,
                "connection_rate": 1000,
                "error_rate": 0.01
            },
            "model": "anomaly_detector"
        });
        
        let response = self.client
            .post(&format!("{}/api/v1/inference/anomaly", self.config.aiengine_endpoint))
            .json(&anomaly_data)
            .send()
            .await?;
        
        assert_eq!(response.status(), 200, "Anomaly detection request failed");
        
        let result: Value = response.json().await?;
        assert!(result["is_anomaly"].is_boolean(), "No anomaly detection result");
        assert!(result["anomaly_score"].is_number(), "No anomaly score");
        
        Ok(())
    }
    
    /// Test policy enforcement across the system
    async fn test_policy_enforcement(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Testing policy enforcement...");
        
        // Test security policy
        let policy_data = json!({
            "resource": {
                "type": "network_connection",
                "source_ip": "10.0.0.100",
                "dest_ip": "192.168.1.50",
                "dest_port": 443,
                "protocol": "TCP"
            },
            "context": {
                "user": "test_user",
                "time": chrono::Utc::now().to_rfc3339(),
                "classification": "web_traffic"
            }
        });
        
        let response = self.client
            .post(&format!("{}/api/v1/policy/evaluate", self.config.policy_engine_endpoint))
            .json(&policy_data)
            .send()
            .await?;
        
        assert_eq!(response.status(), 200, "Policy evaluation request failed");
        
        let result: Value = response.json().await?;
        assert!(result["decision"].is_string(), "No policy decision");
        assert!(result["policies_applied"].is_array(), "No policies applied info");
        
        let decision = result["decision"].as_str().unwrap();
        assert!(
            decision == "allow" || decision == "deny" || decision == "monitor",
            "Invalid policy decision: {}", decision
        );
        
        // Test compliance check
        let compliance_data = json!({
            "resource": {
                "type": "data_processing",
                "data_classification": "PII",
                "encryption_enabled": true,
                "audit_logging": true
            }
        });
        
        let response = self.client
            .post(&format!("{}/api/v1/policy/compliance", self.config.policy_engine_endpoint))
            .json(&compliance_data)
            .send()
            .await?;
        
        assert_eq!(response.status(), 200, "Compliance check request failed");
        
        let result: Value = response.json().await?;
        assert!(result["compliant"].is_boolean(), "No compliance result");
        assert!(result["violations"].is_array(), "No violations array");
        
        Ok(())
    }
    
    /// Test configuration management system
    async fn test_configuration_management(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Testing configuration management...");
        
        // Test configuration retrieval
        let response = self.client
            .get(&format!("{}/api/v1/config/dataplane.packet_processing.batch_size", self.config.controlplane_endpoint))
            .send()
            .await?;
        
        assert_eq!(response.status(), 200, "Configuration retrieval failed");
        
        let config: Value = response.json().await?;
        assert!(config["value"].is_number() || config["value"].is_string(), "No config value");
        
        // Test configuration update
        let update_data = json!({
            "value": 128,
            "changed_by": "integration_test",
            "reason": "Testing configuration update"
        });
        
        let response = self.client
            .put(&format!("{}/api/v1/config/dataplane.packet_processing.test_setting", self.config.controlplane_endpoint))
            .json(&update_data)
            .send()
            .await?;
        
        assert_eq!(response.status(), 200, "Configuration update failed");
        
        // Verify the update took effect
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        let response = self.client
            .get(&format!("{}/api/v1/config/dataplane.packet_processing.test_setting", self.config.controlplane_endpoint))
            .send()
            .await?;
        
        let updated_config: Value = response.json().await?;
        assert_eq!(updated_config["value"].as_u64(), Some(128), "Configuration update not reflected");
        
        Ok(())
    }
    
    /// Test high-frequency metrics collection
    async fn test_high_frequency_metrics(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Testing high-frequency metrics...");
        
        // Get metrics from dataplane
        let response = self.client
            .get(&format!("{}/metrics", self.config.dataplane_endpoint))
            .send()
            .await?;
        
        assert_eq!(response.status(), 200, "Metrics retrieval failed");
        
        let metrics_text = response.text().await?;
        let metrics = utils::parse_prometheus_metrics(&metrics_text);
        
        // Verify key metrics are present
        let required_metrics = [
            "cronos_ai_packets_processed_total",
            "cronos_ai_packet_processing_duration_seconds",
            "cronos_ai_dpi_classifications_total",
            "cronos_ai_memory_usage_bytes",
            "cronos_ai_cpu_usage_percent"
        ];
        
        for metric in &required_metrics {
            assert!(
                metrics.keys().any(|k| k.contains(metric)),
                "Required metric '{}' not found", metric
            );
        }
        
        // Test high-frequency metrics endpoint
        let response = self.client
            .get(&format!("{}/api/v1/metrics/high-frequency", self.config.dataplane_endpoint))
            .send()
            .await?;
        
        assert_eq!(response.status(), 200, "High-frequency metrics retrieval failed");
        
        let hf_metrics: Value = response.json().await?;
        assert!(hf_metrics["sampling_rate_hz"].is_number(), "No sampling rate in metrics");
        assert!(hf_metrics["buffer_utilization"].is_number(), "No buffer utilization in metrics");
        assert!(hf_metrics["latest_samples"].is_array(), "No latest samples in metrics");
        
        Ok(())
    }
    
    /// Test system failover and recovery
    async fn test_failover_recovery(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Testing failover and recovery...");
        
        // Simulate load to establish baseline
        let baseline_response = self.client
            .get(&format!("{}/api/v1/stats", self.config.controlplane_endpoint))
            .send()
            .await?;
        
        assert_eq!(baseline_response.status(), 200, "Baseline stats retrieval failed");
        
        // Test graceful degradation under simulated failure
        // (In a real test, this might involve network partitions or pod termination)
        let stress_data = json!({
            "test_type": "failover_simulation",
            "duration_seconds": 10,
            "failure_scenario": "network_partition"
        });
        
        let response = self.client
            .post(&format!("{}/api/v1/test/failover", self.config.controlplane_endpoint))
            .json(&stress_data)
            .timeout(Duration::from_secs(30))
            .send()
            .await?;
        
        // System should handle the test gracefully
        assert!(
            response.status().is_success() || response.status() == 503,
            "Failover test returned unexpected status: {}", response.status()
        );
        
        // Wait for recovery
        tokio::time::sleep(Duration::from_secs(5)).await;
        
        // Verify system recovered
        let recovery_response = self.client
            .get(&format!("{}/health", self.config.controlplane_endpoint))
            .send()
            .await?;
        
        assert_eq!(recovery_response.status(), 200, "System did not recover after failover test");
        
        Ok(())
    }
    
    /// Test system behavior under load
    async fn test_load_handling(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Testing load handling...");
        
        let concurrent_requests = 50;
        let mut handles = Vec::new();
        
        // Launch concurrent requests
        for i in 0..concurrent_requests {
            let client = self.client.clone();
            let endpoint = self.config.dataplane_endpoint.clone();
            
            let handle = tokio::spawn(async move {
                let test_packet = utils::generate_test_packets(1);
                let payload = json!({
                    "packets": [base64::encode(&test_packet[0])],
                    "request_id": format!("load_test_{}", i)
                });
                
                client
                    .post(&format!("{}/api/v1/packets/process", endpoint))
                    .json(&payload)
                    .send()
                    .await
            });
            
            handles.push(handle);
        }
        
        // Wait for all requests to complete
        let mut successful_requests = 0;
        let mut failed_requests = 0;
        
        for handle in handles {
            match handle.await {
                Ok(Ok(response)) => {
                    if response.status().is_success() {
                        successful_requests += 1;
                    } else {
                        failed_requests += 1;
                    }
                }
                _ => {
                    failed_requests += 1;
                }
            }
        }
        
        info!(
            "Load test results: {} successful, {} failed out of {} total",
            successful_requests, failed_requests, concurrent_requests
        );
        
        // At least 80% should succeed under load
        let success_rate = successful_requests as f64 / concurrent_requests as f64;
        assert!(
            success_rate >= 0.8,
            "Load test success rate too low: {:.2}%", success_rate * 100.0
        );
        
        Ok(())
    }
}

#[async_trait::async_trait]
impl IntegrationTest for E2ETestSuite {
    async fn setup(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Setting up end-to-end test environment");
        
        // Wait for all services to be ready
        let services = [
            (&self.config.dataplane_endpoint, "DataPlane"),
            (&self.config.controlplane_endpoint, "ControlPlane"), 
            (&self.config.aiengine_endpoint, "AI Engine"),
            (&self.config.policy_engine_endpoint, "Policy Engine"),
        ];
        
        for (endpoint, name) in services {
            info!("Waiting for {} to be ready...", name);
            
            utils::wait_for_condition(
                || {
                    let client = self.client.clone();
                    let endpoint = endpoint.clone();
                    async move {
                        match client.get(&format!("{}/health", endpoint)).send().await {
                            Ok(response) => response.status() == 200,
                            Err(_) => false,
                        }
                    }
                },
                Duration::from_secs(60),
                Duration::from_secs(2),
            ).await.map_err(|_| format!("{} failed to become ready", name))?;
            
            info!("✅ {} is ready", name);
        }
        
        Ok(())
    }
    
    async fn run_tests(&mut self) -> TestResults {
        self.run_complete_scenario().await
    }
    
    async fn cleanup(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Cleaning up end-to-end test environment");
        
        // Reset test configurations
        let reset_data = json!({
            "action": "reset_test_config",
            "test_run_id": chrono::Utc::now().timestamp()
        });
        
        let _ = self.client
            .post(&format!("{}/api/v1/test/cleanup", self.config.controlplane_endpoint))
            .json(&reset_data)
            .send()
            .await;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_e2e_suite_creation() {
        let config = TestConfig::default();
        let suite = E2ETestSuite::new(config);
        assert_eq!(suite.test_namespace, "cronos-ai-test");
    }
}