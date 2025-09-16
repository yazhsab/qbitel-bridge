//! Production example demonstrating complete PQC-TLS deployment
//! 
//! This example shows how to:
//! - Load production configuration
//! - Initialize the PQC-TLS manager with all components
//! - Set up monitoring and health checks
//! - Handle key lifecycle and rotation
//! - Implement proper error handling and logging

use pqc_tls::{
    PqcTlsManager, PqcTlsConfig, ProductionConfig,
    KyberKEM, QuantumSafeRotationManager, KeyLifecycleManager,
    RotationTrigger, QuantumThreatLevel, Environment,
};
use std::sync::Arc;
use tokio::time::{interval, Duration};
use tracing::{info, warn, error, Level};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize structured logging
    init_logging()?;

    info!("Starting PQC-TLS production deployment");

    // Load production configuration
    let config = load_production_config().await?;
    info!("Loaded production configuration for environment: {:?}", config.environment);

    // Initialize PQC-TLS manager
    let manager = initialize_pqc_manager(&config).await?;
    info!("PQC-TLS manager initialized successfully");

    // Display system capabilities
    display_system_info(&manager);

    // Start monitoring and health checks
    let health_monitor = start_health_monitoring(Arc::clone(&manager.lifecycle_manager)).await?;

    // Start quantum threat monitoring
    let threat_monitor = start_threat_monitoring(Arc::clone(&manager.rotation_manager)).await?;

    // Generate initial keys for the system
    let initial_keys = generate_initial_keys(&manager).await?;
    info!("Generated {} initial keys", initial_keys.len());

    // Demonstrate key rotation
    demonstrate_key_rotation(&manager, &initial_keys[0]).await?;

    // Run performance benchmarks
    run_performance_benchmarks(&manager).await?;

    // Start the main service loop
    run_main_service_loop(&manager).await?;

    // Cleanup
    info!("Shutting down PQC-TLS service");
    health_monitor.abort();
    threat_monitor.abort();

    Ok(())
}

fn init_logging() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .with(tracing_subscriber::fmt::layer().json())
        .init();

    Ok(())
}

async fn load_production_config() -> Result<ProductionConfig, Box<dyn std::error::Error>> {
    // Try to load from file first
    match ProductionConfig::from_file("examples/production_config.yaml") {
        Ok(mut config) => {
            // Merge environment variables
            config.merge_env_vars()?;
            Ok(config)
        }
        Err(_) => {
            warn!("Could not load config file, using default production config");
            let mut config = ProductionConfig::default_production();
            config.merge_env_vars()?;
            Ok(config)
        }
    }
}

async fn initialize_pqc_manager(
    config: &ProductionConfig,
) -> Result<PqcTlsManager, Box<dyn std::error::Error>> {
    let pqc_config = PqcTlsConfig {
        node_id: config.node.id.clone(),
        use_hardware_acceleration: config.pqc.hardware_acceleration,
        hsm_config: None, // Would be configured based on config.hsm
        lifecycle_config: config.pqc.lifecycle.clone(),
        rotation_config: config.pqc.rotation.clone(),
    };

    let manager = PqcTlsManager::new(pqc_config).await?;
    Ok(manager)
}

fn display_system_info(manager: &PqcTlsManager) {
    let info = manager.system_info();
    
    info!("=== PQC-TLS System Information ===");
    info!("Kyber Available: {}", info.kyber_available);
    info!("Hardware Acceleration: {}", info.hardware_acceleration);
    info!("HSM Available: {}", info.hsm_available);
    info!("FIPS Mode: {}", info.fips_mode);
    info!("CPU Cores: {}", info.hardware_capabilities.cpu_cores);
    info!("AVX-512 Support: {}", info.hardware_capabilities.has_avx512);
    info!("AVX2 Support: {}", info.hardware_capabilities.has_avx2);
    info!("AES-NI Support: {}", info.hardware_capabilities.has_aes_ni);
    info!("===================================");
}

async fn start_health_monitoring(
    lifecycle_manager: Arc<KeyLifecycleManager>,
) -> Result<tokio::task::JoinHandle<()>, Box<dyn std::error::Error>> {
    let handle = tokio::spawn(async move {
        let mut interval = interval(Duration::from_secs(30));
        
        loop {
            interval.tick().await;
            
            match perform_health_checks(&lifecycle_manager).await {
                Ok(healthy) => {
                    if healthy {
                        info!("Health check passed");
                    } else {
                        warn!("Health check failed");
                    }
                }
                Err(e) => {
                    error!("Health check error: {}", e);
                }
            }
        }
    });

    Ok(handle)
}

async fn perform_health_checks(
    lifecycle_manager: &KeyLifecycleManager,
) -> Result<bool, Box<dyn std::error::Error>> {
    // Check key lifecycle manager status
    let stats = lifecycle_manager.get_statistics().await;
    
    // Basic health checks
    let memory_ok = check_memory_usage().await?;
    let cpu_ok = check_cpu_usage().await?;
    let keys_ok = stats.active_keys > 0;
    
    let healthy = memory_ok && cpu_ok && keys_ok;
    
    if healthy {
        info!(
            "Health check: Memory OK: {}, CPU OK: {}, Active Keys: {}",
            memory_ok, cpu_ok, stats.active_keys
        );
    }
    
    Ok(healthy)
}

async fn check_memory_usage() -> Result<bool, Box<dyn std::error::Error>> {
    // In a real implementation, this would check actual memory usage
    // For now, we'll simulate a memory check
    Ok(true)
}

async fn check_cpu_usage() -> Result<bool, Box<dyn std::error::Error>> {
    // In a real implementation, this would check actual CPU usage
    // For now, we'll simulate a CPU check
    Ok(true)
}

async fn start_threat_monitoring(
    rotation_manager: Arc<QuantumSafeRotationManager>,
) -> Result<tokio::task::JoinHandle<()>, Box<dyn std::error::Error>> {
    let handle = tokio::spawn(async move {
        let mut interval = interval(Duration::from_secs(300)); // Every 5 minutes
        
        loop {
            interval.tick().await;
            
            match assess_quantum_threat().await {
                Ok(threat_level) => {
                    info!("Current quantum threat level: {:?}", threat_level);
                    
                    if threat_level >= QuantumThreatLevel::High {
                        warn!("Elevated quantum threat level detected: {:?}", threat_level);
                        
                        if threat_level >= QuantumThreatLevel::Critical {
                            // Trigger emergency rotations
                            info!("Triggering emergency key rotations due to critical threat level");
                            // In a real implementation, we would trigger emergency rotations here
                        }
                    }
                }
                Err(e) => {
                    error!("Threat assessment error: {}", e);
                }
            }
        }
    });

    Ok(handle)
}

async fn assess_quantum_threat() -> Result<QuantumThreatLevel, Box<dyn std::error::Error>> {
    // In a real implementation, this would:
    // 1. Check threat intelligence feeds
    // 2. Analyze quantum computing developments
    // 3. Monitor cryptanalytic advances
    // 4. Assess infrastructure vulnerabilities
    
    // For demonstration, we'll return a low threat level
    Ok(QuantumThreatLevel::Low)
}

async fn generate_initial_keys(
    manager: &PqcTlsManager,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut keys = Vec::new();
    
    info!("Generating initial key set...");
    
    // Generate primary service keys
    for i in 0..3 {
        let key_name = format!("service-key-{}", i);
        let key_id = manager.lifecycle_manager.generate_key(
            Some(key_name),
            None, // Use default policy
        ).await?;
        
        keys.push(key_id);
        info!("Generated service key: {}", keys[i]);
    }
    
    info!("Initial key generation completed");
    Ok(keys)
}

async fn demonstrate_key_rotation(
    manager: &PqcTlsManager,
    key_id: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Demonstrating key rotation for key: {}", key_id);
    
    // Perform a manual rotation
    let new_key_id = manager.rotation_manager.rotate_key(
        key_id,
        RotationTrigger::ManualRequest,
    ).await?;
    
    info!("Key rotation completed: {} -> {}", key_id, new_key_id);
    
    // Get rotation statistics
    let stats = manager.rotation_manager.get_rotation_statistics().await;
    info!(
        "Rotation statistics - Total: {}, Successful: {}, Failed: {}, Active: {}",
        stats.total_rotations,
        stats.successful_rotations,
        stats.failed_rotations,
        stats.active_rotations
    );
    
    Ok(())
}

async fn run_performance_benchmarks(
    manager: &PqcTlsManager,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Running performance benchmarks...");
    
    // Use the benchmarking module
    #[cfg(feature = "benchmarks")]
    {
        let analysis = pqc_tls::benchmarks::BenchmarkAnalysis::run_analysis().await?;
        let report = analysis.generate_report();
        info!("Performance Benchmark Results:\n{}", report);
    }
    
    #[cfg(not(feature = "benchmarks"))]
    {
        info!("Benchmarks not enabled, skipping performance analysis");
    }
    
    Ok(())
}

async fn run_main_service_loop(
    manager: &PqcTlsManager,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Starting main service loop...");
    
    // In a real production service, this would handle:
    // - Incoming TLS connections
    // - Key operations
    // - Certificate management
    // - Service discovery
    // - Load balancing
    
    let mut interval = interval(Duration::from_secs(60));
    let mut loop_count = 0;
    const MAX_LOOPS: u32 = 5; // For demonstration, run for 5 minutes
    
    while loop_count < MAX_LOOPS {
        interval.tick().await;
        loop_count += 1;
        
        // Simulate service operations
        let stats = manager.lifecycle_manager.get_statistics().await;
        
        info!(
            "Service loop {} - Active keys: {}, Total operations: {}",
            loop_count, stats.active_keys, stats.total_operations
        );
        
        // Simulate some key usage
        let keys = manager.lifecycle_manager.list_keys().await;
        if let Some(key) = keys.first() {
            manager.lifecycle_manager.record_usage(
                &key.key_id,
                100, // operations
                1024, // data volume
            ).await?;
        }
        
        // Check if any keys need rotation based on usage
        // This would normally be handled automatically by the rotation manager
    }
    
    info!("Main service loop completed");
    Ok(())
}

// Example of integrating with a web service
#[cfg(feature = "web-service")]
async fn start_web_service(
    manager: Arc<PqcTlsManager>,
) -> Result<(), Box<dyn std::error::Error>> {
    use warp::Filter;
    
    // Health check endpoint
    let health = warp::path("health")
        .map(|| warp::reply::json(&serde_json::json!({"status": "healthy"})));
    
    // System info endpoint
    let manager_clone = Arc::clone(&manager);
    let info = warp::path("info")
        .map(move || {
            let sys_info = manager_clone.system_info();
            warp::reply::json(&sys_info)
        });
    
    // Statistics endpoint
    let manager_clone = Arc::clone(&manager);
    let stats = warp::path("stats")
        .and_then(move || {
            let manager = Arc::clone(&manager_clone);
            async move {
                let stats = manager.lifecycle_manager.get_statistics().await;
                Ok::<_, warp::Rejection>(warp::reply::json(&stats))
            }
        });
    
    let routes = health.or(info).or(stats);
    
    info!("Starting web service on port 8080");
    warp::serve(routes)
        .run(([0, 0, 0, 0], 8080))
        .await;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_production_config_loading() {
        let config = load_production_config().await;
        assert!(config.is_ok());
        
        let config = config.unwrap();
        assert_eq!(config.environment, Environment::Production);
        assert!(config.security.fips_mode);
        assert!(config.monitoring.metrics_enabled);
    }
    
    #[tokio::test]
    async fn test_pqc_manager_initialization() {
        let config = ProductionConfig::default_production();
        let manager = initialize_pqc_manager(&config).await;
        assert!(manager.is_ok());
        
        let manager = manager.unwrap();
        let info = manager.system_info();
        assert!(info.kyber_available);
    }
    
    #[tokio::test]
    async fn test_health_checks() {
        let config = ProductionConfig::default_production();
        let manager = initialize_pqc_manager(&config).await.unwrap();
        
        let healthy = perform_health_checks(&manager.lifecycle_manager).await;
        // Initial health check might fail since no keys are generated yet
        assert!(healthy.is_ok());
    }
}