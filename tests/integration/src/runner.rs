//! Integration test runner for CRONOS AI
//! 
//! Orchestrates and runs all integration test suites with proper setup,
//! execution, and teardown procedures.

use cronos_ai_integration_tests::{
    init_test_logging, TestConfig, TestResults, IntegrationTest,
    e2e::E2ETestSuite,
    performance::PerformanceTestSuite,
};

use std::time::Instant;
use std::process;
use tracing::{info, warn, error};
use clap::{Parser, Subcommand};
use serde_json;

#[derive(Parser)]
#[command(name = "integration-test-runner")]
#[command(about = "CRONOS AI Integration Test Runner")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    /// Test configuration file path
    #[arg(short, long, default_value = "./test-config.yaml")]
    config: String,
    
    /// Test environment namespace
    #[arg(short, long, default_value = "cronos-ai-test")]
    namespace: String,
    
    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
    
    /// Output format (text, json)
    #[arg(short, long, default_value = "text")]
    output: String,
    
    /// Fail fast on first test failure
    #[arg(long)]
    fail_fast: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Run all integration tests
    All {
        /// Skip performance tests (faster execution)
        #[arg(long)]
        skip_performance: bool,
    },
    /// Run end-to-end tests only
    E2e,
    /// Run performance benchmarks only
    Performance {
        /// Performance test duration in seconds
        #[arg(short, long, default_value = "60")]
        duration: u64,
    },
    /// Run specific test suite
    Suite {
        /// Test suite name
        suite: String,
    },
    /// List available test suites
    List,
    /// Validate test environment
    Validate,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    
    // Initialize logging
    init_test_logging();
    
    if cli.verbose {
        std::env::set_var("RUST_LOG", "debug");
    }
    
    info!("CRONOS AI Integration Test Runner starting");
    
    // Load test configuration
    let mut config = if std::path::Path::new(&cli.config).exists() {
        load_config_from_file(&cli.config)?
    } else {
        TestConfig::default()
    };
    
    // Override namespace if specified
    config.k8s_namespace = cli.namespace;
    
    let start_time = Instant::now();
    let mut overall_results = TestResults::default();
    
    match cli.command {
        Commands::All { skip_performance } => {
            info!("Running all integration tests");
            
            // Run E2E tests
            let e2e_results = run_e2e_tests(&config, cli.fail_fast).await?;
            overall_results.merge(e2e_results);
            
            // Run performance tests unless skipped
            if !skip_performance {
                let perf_results = run_performance_tests(&config, cli.fail_fast).await?;
                overall_results.merge(perf_results);
            }
        }
        
        Commands::E2e => {
            info!("Running end-to-end tests");
            let results = run_e2e_tests(&config, cli.fail_fast).await?;
            overall_results.merge(results);
        }
        
        Commands::Performance { duration } => {
            info!("Running performance benchmarks");
            config.performance_test_duration_seconds = duration;
            let results = run_performance_tests(&config, cli.fail_fast).await?;
            overall_results.merge(results);
        }
        
        Commands::Suite { suite } => {
            info!("Running test suite: {}", suite);
            let results = run_specific_suite(&config, &suite, cli.fail_fast).await?;
            overall_results.merge(results);
        }
        
        Commands::List => {
            list_available_suites();
            return Ok(());
        }
        
        Commands::Validate => {
            info!("Validating test environment");
            validate_test_environment(&config).await?;
            info!("✅ Test environment validation passed");
            return Ok(());
        }
    }
    
    let total_duration = start_time.elapsed();
    overall_results.duration_ms = total_duration.as_millis() as u64;
    
    // Output results
    match cli.output.as_str() {
        "json" => {
            let json_output = serde_json::json!({
                "summary": overall_results.summary(),
                "total": overall_results.total,
                "passed": overall_results.passed,
                "failed": overall_results.failed,
                "skipped": overall_results.skipped,
                "success_rate": overall_results.success_rate(),
                "duration_ms": overall_results.duration_ms,
                "failures": overall_results.failures
            });
            println!("{}", serde_json::to_string_pretty(&json_output)?);
        }
        _ => {
            // Text output
            info!("=== Test Results ===");
            info!("{}", overall_results.summary());
            
            if !overall_results.failures.is_empty() {
                warn!("Failures:");
                for failure in &overall_results.failures {
                    warn!("  - {}", failure);
                }
            }
        }
    }
    
    // Exit with appropriate code
    if overall_results.is_success() {
        info!("🎉 All tests passed!");
        Ok(())
    } else {
        error!("❌ Some tests failed");
        process::exit(1);
    }
}

/// Run end-to-end tests
async fn run_e2e_tests(config: &TestConfig, fail_fast: bool) -> Result<TestResults, Box<dyn std::error::Error>> {
    info!("Setting up E2E test suite");
    
    let mut suite = E2ETestSuite::new(config.clone());
    
    // Setup
    if let Err(e) = suite.setup().await {
        error!("E2E test setup failed: {}", e);
        if fail_fast {
            return Err(e);
        }
        let mut results = TestResults::default();
        results.total = 1;
        results.add_failure("e2e_setup", &e.to_string());
        return Ok(results);
    }
    
    // Run tests
    let results = suite.run_tests().await;
    
    // Cleanup
    if let Err(e) = suite.cleanup().await {
        warn!("E2E test cleanup failed: {}", e);
    }
    
    info!("E2E tests completed: {}", results.summary());
    Ok(results)
}

/// Run performance tests
async fn run_performance_tests(config: &TestConfig, fail_fast: bool) -> Result<TestResults, Box<dyn std::error::Error>> {
    info!("Setting up performance test suite");
    
    let mut suite = PerformanceTestSuite::new(config.clone());
    
    // Setup
    if let Err(e) = suite.setup().await {
        error!("Performance test setup failed: {}", e);
        if fail_fast {
            return Err(e);
        }
        let mut results = TestResults::default();
        results.total = 1;
        results.add_failure("performance_setup", &e.to_string());
        return Ok(results);
    }
    
    // Run benchmarks
    let results = suite.run_tests().await;
    
    // Cleanup
    if let Err(e) = suite.cleanup().await {
        warn!("Performance test cleanup failed: {}", e);
    }
    
    info!("Performance tests completed: {}", results.summary());
    Ok(results)
}

/// Run specific test suite
async fn run_specific_suite(
    config: &TestConfig, 
    suite_name: &str, 
    fail_fast: bool
) -> Result<TestResults, Box<dyn std::error::Error>> {
    match suite_name {
        "e2e" => run_e2e_tests(config, fail_fast).await,
        "performance" => run_performance_tests(config, fail_fast).await,
        _ => {
            error!("Unknown test suite: {}", suite_name);
            Err(format!("Unknown test suite: {}", suite_name).into())
        }
    }
}

/// List available test suites
fn list_available_suites() {
    println!("Available test suites:");
    println!("  e2e          - End-to-end integration tests");
    println!("  performance  - Performance benchmarks");
    println!();
    println!("Use 'all' to run all suites");
}

/// Validate test environment
async fn validate_test_environment(config: &TestConfig) -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()?;
    
    // Check service endpoints
    let services = [
        (&config.dataplane_endpoint, "DataPlane"),
        (&config.controlplane_endpoint, "ControlPlane"),
        (&config.aiengine_endpoint, "AI Engine"),
        (&config.policy_engine_endpoint, "Policy Engine"),
    ];
    
    for (endpoint, name) in &services {
        info!("Validating {} at {}", name, endpoint);
        
        match client.get(&format!("{}/health", endpoint)).send().await {
            Ok(response) if response.status() == 200 => {
                info!("✅ {} is healthy", name);
            }
            Ok(response) => {
                warn!("⚠️  {} returned status {}", name, response.status());
            }
            Err(e) => {
                error!("❌ {} is not accessible: {}", name, e);
                return Err(format!("{} validation failed: {}", name, e).into());
            }
        }
    }
    
    // Check Kubernetes connectivity (if configured)
    if let Ok(client) = kube::Client::try_default().await {
        use kube::Api;
        use k8s_openapi::api::core::v1::Namespace;
        
        let namespaces: Api<Namespace> = Api::all(client);
        match namespaces.get(&config.k8s_namespace).await {
            Ok(_) => {
                info!("✅ Kubernetes namespace '{}' exists", config.k8s_namespace);
            }
            Err(_) => {
                warn!("⚠️  Kubernetes namespace '{}' not found", config.k8s_namespace);
            }
        }
    } else {
        warn!("⚠️  Kubernetes client not available (running outside cluster?)");
    }
    
    // Validate test data directory
    if std::path::Path::new(&config.test_data_dir).exists() {
        info!("✅ Test data directory exists: {}", config.test_data_dir);
    } else {
        warn!("⚠️  Test data directory not found: {}", config.test_data_dir);
    }
    
    Ok(())
}

/// Load configuration from file
fn load_config_from_file(path: &str) -> Result<TestConfig, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    
    // Try YAML first, then JSON
    if path.ends_with(".yaml") || path.ends_with(".yml") {
        Ok(serde_yaml::from_str(&content)?)
    } else if path.ends_with(".json") {
        Ok(serde_json::from_str(&content)?)
    } else {
        // Try both formats
        serde_yaml::from_str(&content)
            .or_else(|_| serde_json::from_str(&content))
            .map_err(|e| format!("Failed to parse config file: {}", e).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;
    
    #[test]
    fn test_load_yaml_config() {
        let yaml_content = r#"
dataplane_endpoint: "http://localhost:9090"
controlplane_endpoint: "http://localhost:8080"
aiengine_endpoint: "http://localhost:8000"
policy_engine_endpoint: "http://localhost:8001"
metrics_endpoint: "http://localhost:9090/metrics"
k8s_namespace: "test-namespace"
test_data_dir: "./test_data"
timeout_seconds: 30
performance_test_duration_seconds: 60
"#;
        
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(yaml_content.as_bytes()).unwrap();
        
        let config = load_config_from_file(temp_file.path().to_str().unwrap()).unwrap();
        assert_eq!(config.k8s_namespace, "test-namespace");
        assert_eq!(config.timeout_seconds, 30);
    }
    
    #[test]
    fn test_load_json_config() {
        let json_content = r#"
{
    "dataplane_endpoint": "http://localhost:9090",
    "controlplane_endpoint": "http://localhost:8080",
    "aiengine_endpoint": "http://localhost:8000",
    "policy_engine_endpoint": "http://localhost:8001",
    "metrics_endpoint": "http://localhost:9090/metrics",
    "k8s_namespace": "test-namespace",
    "test_data_dir": "./test_data",
    "timeout_seconds": 30,
    "performance_test_duration_seconds": 60
}
"#;
        
        let mut temp_file = NamedTempFile::with_suffix(".json").unwrap();
        temp_file.write_all(json_content.as_bytes()).unwrap();
        
        let config = load_config_from_file(temp_file.path().to_str().unwrap()).unwrap();
        assert_eq!(config.k8s_namespace, "test-namespace");
        assert_eq!(config.timeout_seconds, 30);
    }
}