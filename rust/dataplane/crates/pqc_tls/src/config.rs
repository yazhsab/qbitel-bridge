//! Production-ready configuration and deployment for PQC-TLS
//! 
//! This module provides comprehensive configuration management for production
//! deployments with environment-specific configurations, security policies,
//! high availability, monitoring, and compliance settings.

use crate::errors::TlsError;
use crate::hsm::{HsmSlotConfig, HsmCredentials, HsmSessionConfig};
use crate::lifecycle::LifecycleManagerConfig;
use crate::rotation::{RotationPolicyConfig, QuantumThreatLevel};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Duration;
use tracing::{info, warn};

/// Complete production configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductionConfig {
    /// Environment type
    pub environment: Environment,
    /// Node configuration
    pub node: NodeConfig,
    /// Security policies
    pub security: SecurityConfig,
    /// PQC-specific settings
    pub pqc: PqcConfig,
    /// HSM configuration
    pub hsm: Option<HsmDeploymentConfig>,
    /// Monitoring and observability
    pub monitoring: MonitoringConfig,
    /// Performance tuning
    pub performance: PerformanceConfig,
    /// Compliance settings
    pub compliance: ComplianceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Environment {
    Development,
    Staging,
    Production,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    pub id: String,
    pub role: NodeRole,
    pub datacenter: String,
    pub region: String,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeRole {
    Primary,
    Secondary,
    Worker,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub fips_mode: bool,
    pub min_tls_version: TlsVersion,
    pub allowed_cipher_suites: Vec<String>,
    pub certificate_policy: CertificatePolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TlsVersion {
    #[serde(rename = "1.2")]
    V12,
    #[serde(rename = "1.3")]
    V13,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificatePolicy {
    pub require_pqc: bool,
    pub allow_hybrid: bool,
    pub validation_level: ValidationLevel,
    pub ocsp_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationLevel {
    Basic,
    Standard,
    Strict,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqcConfig {
    pub hardware_acceleration: bool,
    pub threat_level: QuantumThreatLevel,
    pub hybrid_mode: HybridModeConfig,
    pub algorithm_preferences: AlgorithmPreferences,
    pub lifecycle: LifecycleManagerConfig,
    pub rotation: RotationPolicyConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridModeConfig {
    pub enabled: bool,
    pub classical_algorithms: Vec<String>,
    pub pqc_algorithms: Vec<String>,
    pub combination_strategy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmPreferences {
    pub kem_algorithms: Vec<String>,
    pub signature_algorithms: Vec<String>,
    pub algorithm_params: HashMap<String, AlgorithmParams>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmParams {
    pub security_level: u8,
    pub performance_preference: PerformancePreference,
    pub custom_params: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformancePreference {
    Speed,
    Balanced,
    Security,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HsmDeploymentConfig {
    pub vendor: String,
    pub model: String,
    pub slots: Vec<HsmSlotConfig>,
    pub session_config: HsmSessionConfig,
    pub ha_config: HsmHaConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HsmHaConfig {
    pub clustering_enabled: bool,
    pub cluster_nodes: Vec<String>,
    pub failover_timeout: Duration,
    pub load_balancing: HsmLoadBalancing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HsmLoadBalancing {
    RoundRobin,
    LeastLoaded,
    HealthBased,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub metrics_enabled: bool,
    pub metrics_interval: Duration,
    pub prometheus: Option<PrometheusConfig>,
    pub health_checks: HealthCheckConfig,
    pub alerting: AlertingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrometheusConfig {
    pub endpoint: String,
    pub scrape_interval: Duration,
    pub labels: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    pub interval: Duration,
    pub timeout: Duration,
    pub checks: Vec<HealthCheck>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthCheck {
    KeyLifecycle,
    HsmConnectivity,
    MemoryUsage,
    CpuUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingConfig {
    pub enabled: bool,
    pub channels: Vec<AlertChannel>,
    pub rules: Vec<AlertRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertChannel {
    pub name: String,
    pub channel_type: AlertChannelType,
    pub config: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertChannelType {
    Email,
    Slack,
    Webhook,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub name: String,
    pub condition: String,
    pub threshold: f64,
    pub duration: Duration,
    pub severity: AlertSeverity,
    pub channels: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub thread_pools: ThreadPoolConfig,
    pub memory: MemoryConfig,
    pub caching: CachingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadPoolConfig {
    pub worker_threads: Option<usize>,
    pub max_blocking_threads: Option<usize>,
    pub thread_stack_size: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub pool_enabled: bool,
    pub pool_sizes: HashMap<String, usize>,
    pub alignment: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachingConfig {
    pub enabled: bool,
    pub max_entries: usize,
    pub max_memory: u64,
    pub default_ttl: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceConfig {
    pub fips_140_2: Fips140Config,
    pub common_criteria: CommonCriteriaConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fips140Config {
    pub enabled: bool,
    pub security_level: u8,
    pub approved_algorithms_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommonCriteriaConfig {
    pub enabled: bool,
    pub eal_level: u8,
    pub security_target: String,
}

impl ProductionConfig {
    /// Load configuration from file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, TlsError> {
        let content = fs::read_to_string(path)
            .map_err(|e| TlsError::Io(format!("failed to read config file: {}", e)))?;

        let config: ProductionConfig = serde_yaml::from_str(&content)
            .map_err(|e| TlsError::Io(format!("failed to parse YAML config: {}", e)))?;

        config.validate()?;
        Ok(config)
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), TlsError> {
        match self.environment {
            Environment::Production => {
                self.validate_production_requirements()?;
            }
            Environment::Staging => {
                self.validate_staging_requirements()?;
            }
            Environment::Development => {
                // Development environment has relaxed requirements
            }
        }

        info!("configuration validation completed successfully");
        Ok(())
    }

    fn validate_production_requirements(&self) -> Result<(), TlsError> {
        if !self.security.fips_mode && self.compliance.fips_140_2.enabled {
            return Err(TlsError::Policy(
                "FIPS mode required in production with FIPS 140-2 compliance".into()
            ));
        }

        if !self.monitoring.metrics_enabled {
            return Err(TlsError::Policy(
                "metrics must be enabled in production environment".into()
            ));
        }

        Ok(())
    }

    fn validate_staging_requirements(&self) -> Result<(), TlsError> {
        if !self.monitoring.metrics_enabled {
            warn!("metrics not enabled in staging environment");
        }
        Ok(())
    }

    /// Create default production configuration
    pub fn default_production() -> Self {
        Self {
            environment: Environment::Production,
            node: NodeConfig {
                id: uuid::Uuid::new_v4().to_string(),
                role: NodeRole::Primary,
                datacenter: "us-east-1a".to_string(),
                region: "us-east-1".to_string(),
                metadata: HashMap::new(),
            },
            security: SecurityConfig {
                fips_mode: true,
                min_tls_version: TlsVersion::V13,
                allowed_cipher_suites: vec![
                    "TLS_AES_256_GCM_SHA384".to_string(),
                    "TLS_AES_128_GCM_SHA256".to_string(),
                ],
                certificate_policy: CertificatePolicy {
                    require_pqc: true,
                    allow_hybrid: true,
                    validation_level: ValidationLevel::Strict,
                    ocsp_required: true,
                },
            },
            pqc: PqcConfig {
                hardware_acceleration: true,
                threat_level: QuantumThreatLevel::Low,
                hybrid_mode: HybridModeConfig {
                    enabled: true,
                    classical_algorithms: vec!["x25519".to_string()],
                    pqc_algorithms: vec!["kyber768".to_string()],
                    combination_strategy: "concatenation".to_string(),
                },
                algorithm_preferences: AlgorithmPreferences {
                    kem_algorithms: vec!["kyber768".to_string()],
                    signature_algorithms: vec!["dilithium3".to_string()],
                    algorithm_params: HashMap::new(),
                },
                lifecycle: LifecycleManagerConfig::default(),
                rotation: RotationPolicyConfig::default(),
            },
            hsm: Some(HsmDeploymentConfig {
                vendor: "SafeNet".to_string(),
                model: "Luna HSM".to_string(),
                slots: vec![],
                session_config: HsmSessionConfig::default(),
                ha_config: HsmHaConfig {
                    clustering_enabled: true,
                    cluster_nodes: vec![],
                    failover_timeout: Duration::from_secs(30),
                    load_balancing: HsmLoadBalancing::HealthBased,
                },
            }),
            monitoring: MonitoringConfig {
                metrics_enabled: true,
                metrics_interval: Duration::from_secs(30),
                prometheus: Some(PrometheusConfig {
                    endpoint: "http://prometheus:9090".to_string(),
                    scrape_interval: Duration::from_secs(15),
                    labels: HashMap::new(),
                }),
                health_checks: HealthCheckConfig {
                    interval: Duration::from_secs(30),
                    timeout: Duration::from_secs(10),
                    checks: vec![
                        HealthCheck::KeyLifecycle,
                        HealthCheck::HsmConnectivity,
                        HealthCheck::MemoryUsage,
                        HealthCheck::CpuUsage,
                    ],
                },
                alerting: AlertingConfig {
                    enabled: true,
                    channels: vec![],
                    rules: vec![],
                },
            },
            performance: PerformanceConfig {
                thread_pools: ThreadPoolConfig {
                    worker_threads: None, // Use defaults
                    max_blocking_threads: Some(512),
                    thread_stack_size: None,
                },
                memory: MemoryConfig {
                    pool_enabled: true,
                    pool_sizes: [
                        ("kyber_keys".to_string(), 1024),
                        ("general".to_string(), 2048),
                    ].iter().cloned().collect(),
                    alignment: 32,
                },
                caching: CachingConfig {
                    enabled: true,
                    max_entries: 10000,
                    max_memory: 1024 * 1024 * 1024, // 1GB
                    default_ttl: Duration::from_secs(3600),
                },
            },
            compliance: ComplianceConfig {
                fips_140_2: Fips140Config {
                    enabled: true,
                    security_level: 2,
                    approved_algorithms_only: true,
                },
                common_criteria: CommonCriteriaConfig {
                    enabled: true,
                    eal_level: 4,
                    security_target: "PQC-TLS v1.0".to_string(),
                },
            },
        }
    }

    /// Create default development configuration
    pub fn default_development() -> Self {
        Self {
            environment: Environment::Development,
            security: SecurityConfig {
                fips_mode: false,
                min_tls_version: TlsVersion::V12,
                allowed_cipher_suites: vec![
                    "TLS_AES_256_GCM_SHA384".to_string(),
                ],
                certificate_policy: CertificatePolicy {
                    require_pqc: false,
                    allow_hybrid: true,
                    validation_level: ValidationLevel::Basic,
                    ocsp_required: false,
                },
            },
            hsm: None,
            monitoring: MonitoringConfig {
                metrics_enabled: false,
                metrics_interval: Duration::from_secs(60),
                prometheus: None,
                health_checks: HealthCheckConfig {
                    interval: Duration::from_secs(60),
                    timeout: Duration::from_secs(5),
                    checks: vec![HealthCheck::MemoryUsage],
                },
                alerting: AlertingConfig {
                    enabled: false,
                    channels: vec![],
                    rules: vec![],
                },
            },
            compliance: ComplianceConfig {
                fips_140_2: Fips140Config {
                    enabled: false,
                    security_level: 1,
                    approved_algorithms_only: false,
                },
                common_criteria: CommonCriteriaConfig {
                    enabled: false,
                    eal_level: 1,
                    security_target: "Development".to_string(),
                },
            },
            ..Self::default_production()
        }
    }

    /// Merge with environment variables
    pub fn merge_env_vars(&mut self) -> Result<(), TlsError> {
        use std::env;

        if let Ok(node_id) = env::var("PQC_NODE_ID") {
            self.node.id = node_id;
        }

        if let Ok(fips_mode) = env::var("PQC_FIPS_MODE") {
            self.security.fips_mode = fips_mode.parse().unwrap_or(false);
        }

        info!("merged environment variables into configuration");
        Ok(())
    }
}

impl Default for ProductionConfig {
    fn default() -> Self {
        Self::default_development()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_production_config_validation() {
        let config = ProductionConfig::default_production();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_development_config_validation() {
        let config = ProductionConfig::default_development();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_environment_is_production() {
        assert!(Environment::Production.is_production());
        assert!(!Environment::Development.is_production());
    }
}

impl Environment {
    pub fn is_production(&self) -> bool {
        matches!(self, Environment::Production)
    }

    pub fn is_development(&self) -> bool {
        matches!(self, Environment::Development)
    }
}