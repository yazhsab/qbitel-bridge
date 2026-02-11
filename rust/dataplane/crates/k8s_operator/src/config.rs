//! Configuration for the QBITEL Bridge Kubernetes operator

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;

/// Operator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorConfig {
    /// Operator settings
    pub operator: OperatorSettings,
    
    /// Metrics server configuration
    pub metrics: MetricsConfig,
    
    /// Health check configuration
    pub health: HealthConfig,
    
    /// Controller-specific configurations
    pub controllers: ControllersConfig,
    
    /// Kubernetes client configuration
    pub kubernetes: KubernetesConfig,
    
    /// Observability configuration
    pub observability: ObservabilityConfig,
    
    /// Security configuration
    pub security: SecurityConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorSettings {
    /// Operator name
    pub name: String,
    
    /// Operator version
    pub version: String,
    
    /// Default reconcile interval in seconds
    pub reconcile_interval_seconds: u64,
    
    /// Number of controller workers per resource type
    pub worker_count: usize,
    
    /// Enable leader election
    pub leader_election_enabled: bool,
    
    /// Leader election namespace
    pub leader_election_namespace: String,
    
    /// Leader election lock name
    pub leader_election_lock_name: String,
    
    /// Graceful shutdown timeout in seconds
    pub shutdown_timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable metrics collection
    pub enabled: bool,
    
    /// Metrics server port
    pub port: u16,
    
    /// Metrics server bind address
    pub bind_address: String,
    
    /// Metrics collection interval in seconds
    pub collection_interval_seconds: u64,
    
    /// Custom metrics configuration
    pub custom_metrics: Vec<CustomMetricConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomMetricConfig {
    /// Metric name
    pub name: String,
    
    /// Metric type (counter, gauge, histogram)
    pub metric_type: String,
    
    /// Metric description
    pub description: String,
    
    /// Metric labels
    pub labels: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthConfig {
    /// Health check port
    pub port: u16,
    
    /// Health check bind address
    pub bind_address: String,
    
    /// Health check interval in seconds
    pub check_interval_seconds: u64,
    
    /// Health check timeout in seconds
    pub timeout_seconds: u64,
    
    /// Readiness probe configuration
    pub readiness: ReadinessConfig,
    
    /// Liveness probe configuration
    pub liveness: LivenessConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadinessConfig {
    /// Initial delay in seconds
    pub initial_delay_seconds: u64,
    
    /// Period in seconds
    pub period_seconds: u64,
    
    /// Timeout in seconds
    pub timeout_seconds: u64,
    
    /// Failure threshold
    pub failure_threshold: u32,
    
    /// Success threshold
    pub success_threshold: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LivenessConfig {
    /// Initial delay in seconds
    pub initial_delay_seconds: u64,
    
    /// Period in seconds
    pub period_seconds: u64,
    
    /// Timeout in seconds
    pub timeout_seconds: u64,
    
    /// Failure threshold
    pub failure_threshold: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControllersConfig {
    /// DataPlane controller configuration
    pub dataplane: ControllerConfig,
    
    /// ControlPlane controller configuration
    pub controlplane: ControllerConfig,
    
    /// AI Engine controller configuration
    pub aiengine: ControllerConfig,
    
    /// Policy Engine controller configuration
    pub policy_engine: ControllerConfig,
    
    /// Service Mesh controller configuration
    pub servicemesh: ControllerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControllerConfig {
    /// Enable this controller
    pub enabled: bool,
    
    /// Reconcile interval in seconds
    pub reconcile_interval_seconds: u64,
    
    /// Error requeue interval in seconds
    pub error_requeue_interval_seconds: u64,
    
    /// Maximum concurrent reconciles
    pub max_concurrent_reconciles: usize,
    
    /// Resource-specific configuration
    pub resource: ResourceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    /// Default resource requests
    pub default_requests: ResourceRequirements,
    
    /// Default resource limits
    pub default_limits: ResourceRequirements,
    
    /// Default image pull policy
    pub default_image_pull_policy: String,
    
    /// Default security context settings
    pub security_context: SecurityContextConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// CPU requirement
    pub cpu: String,
    
    /// Memory requirement
    pub memory: String,
    
    /// Storage requirement (if applicable)
    pub storage: Option<String>,
    
    /// GPU requirement (if applicable)
    pub gpu: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityContextConfig {
    /// Run as non-root user
    pub run_as_non_root: bool,
    
    /// User ID
    pub run_as_user: Option<u32>,
    
    /// Group ID
    pub run_as_group: Option<u32>,
    
    /// FS Group ID
    pub fs_group: Option<u32>,
    
    /// Read-only root filesystem
    pub read_only_root_filesystem: bool,
    
    /// Allow privilege escalation
    pub allow_privilege_escalation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KubernetesConfig {
    /// Kubeconfig path (optional, defaults to in-cluster config)
    pub kubeconfig_path: Option<String>,
    
    /// API server timeout in seconds
    pub api_timeout_seconds: u64,
    
    /// API request rate limiting
    pub rate_limiting: RateLimitConfig,
    
    /// Retry configuration
    pub retry: RetryConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Enable rate limiting
    pub enabled: bool,
    
    /// Requests per second
    pub requests_per_second: f64,
    
    /// Burst size
    pub burst_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum retry attempts
    pub max_attempts: u32,
    
    /// Initial backoff duration in milliseconds
    pub initial_backoff_ms: u64,
    
    /// Maximum backoff duration in milliseconds
    pub max_backoff_ms: u64,
    
    /// Backoff multiplier
    pub backoff_multiplier: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    /// Tracing configuration
    pub tracing: TracingConfig,
    
    /// Logging configuration
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingConfig {
    /// Enable tracing
    pub enabled: bool,
    
    /// Tracing endpoint
    pub endpoint: Option<String>,
    
    /// Sampling rate (0.0 to 1.0)
    pub sampling_rate: f64,
    
    /// Service name for tracing
    pub service_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error)
    pub level: String,
    
    /// Log format (json, text)
    pub format: String,
    
    /// Enable structured logging
    pub structured: bool,
    
    /// Log file path (optional, logs to stdout if not set)
    pub file_path: Option<String>,
    
    /// Maximum log file size in MB
    pub max_size_mb: u64,
    
    /// Maximum number of log files to retain
    pub max_files: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// TLS configuration
    pub tls: TlsConfig,
    
    /// Authentication configuration
    pub authentication: AuthenticationConfig,
    
    /// RBAC configuration
    pub rbac: RbacConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    /// Enable TLS
    pub enabled: bool,
    
    /// TLS certificate path
    pub cert_path: Option<String>,
    
    /// TLS private key path
    pub key_path: Option<String>,
    
    /// TLS CA certificate path
    pub ca_path: Option<String>,
    
    /// Minimum TLS version
    pub min_version: String,
    
    /// Cipher suites
    pub cipher_suites: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationConfig {
    /// Enable authentication
    pub enabled: bool,
    
    /// Authentication method (serviceaccount, token, cert)
    pub method: String,
    
    /// Token file path (for token auth)
    pub token_file: Option<String>,
    
    /// Service account token path (for serviceaccount auth)
    pub service_account_token_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RbacConfig {
    /// Enable RBAC
    pub enabled: bool,
    
    /// Service account name
    pub service_account_name: String,
    
    /// Cluster role name
    pub cluster_role_name: String,
    
    /// Additional permissions
    pub additional_permissions: Vec<Permission>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Permission {
    /// API groups
    pub api_groups: Vec<String>,
    
    /// Resources
    pub resources: Vec<String>,
    
    /// Verbs
    pub verbs: Vec<String>,
}

impl Default for OperatorConfig {
    fn default() -> Self {
        Self {
            operator: OperatorSettings::default(),
            metrics: MetricsConfig::default(),
            health: HealthConfig::default(),
            controllers: ControllersConfig::default(),
            kubernetes: KubernetesConfig::default(),
            observability: ObservabilityConfig::default(),
            security: SecurityConfig::default(),
        }
    }
}

impl Default for OperatorSettings {
    fn default() -> Self {
        Self {
            name: "qbitel-bridge-operator".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            reconcile_interval_seconds: 300,
            worker_count: 4,
            leader_election_enabled: true,
            leader_election_namespace: "qbitel-bridge-system".to_string(),
            leader_election_lock_name: "qbitel-bridge-operator-leader".to_string(),
            shutdown_timeout_seconds: 30,
        }
    }
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            port: 8080,
            bind_address: "0.0.0.0".to_string(),
            collection_interval_seconds: 30,
            custom_metrics: vec![],
        }
    }
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            port: 8081,
            bind_address: "0.0.0.0".to_string(),
            check_interval_seconds: 30,
            timeout_seconds: 10,
            readiness: ReadinessConfig::default(),
            liveness: LivenessConfig::default(),
        }
    }
}

impl Default for ReadinessConfig {
    fn default() -> Self {
        Self {
            initial_delay_seconds: 10,
            period_seconds: 10,
            timeout_seconds: 5,
            failure_threshold: 3,
            success_threshold: 1,
        }
    }
}

impl Default for LivenessConfig {
    fn default() -> Self {
        Self {
            initial_delay_seconds: 30,
            period_seconds: 30,
            timeout_seconds: 10,
            failure_threshold: 3,
        }
    }
}

impl Default for ControllersConfig {
    fn default() -> Self {
        Self {
            dataplane: ControllerConfig::default(),
            controlplane: ControllerConfig::default(),
            aiengine: ControllerConfig::default(),
            policy_engine: ControllerConfig::default(),
            servicemesh: ControllerConfig::default(),
        }
    }
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            reconcile_interval_seconds: 300,
            error_requeue_interval_seconds: 30,
            max_concurrent_reconciles: 2,
            resource: ResourceConfig::default(),
        }
    }
}

impl Default for ResourceConfig {
    fn default() -> Self {
        Self {
            default_requests: ResourceRequirements {
                cpu: "100m".to_string(),
                memory: "128Mi".to_string(),
                storage: None,
                gpu: None,
            },
            default_limits: ResourceRequirements {
                cpu: "1000m".to_string(),
                memory: "1Gi".to_string(),
                storage: None,
                gpu: None,
            },
            default_image_pull_policy: "Always".to_string(),
            security_context: SecurityContextConfig::default(),
        }
    }
}

impl Default for SecurityContextConfig {
    fn default() -> Self {
        Self {
            run_as_non_root: true,
            run_as_user: Some(10001),
            run_as_group: Some(10001),
            fs_group: Some(10001),
            read_only_root_filesystem: true,
            allow_privilege_escalation: false,
        }
    }
}

impl Default for KubernetesConfig {
    fn default() -> Self {
        Self {
            kubeconfig_path: None,
            api_timeout_seconds: 30,
            rate_limiting: RateLimitConfig::default(),
            retry: RetryConfig::default(),
        }
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            requests_per_second: 10.0,
            burst_size: 20,
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_backoff_ms: 1000,
            max_backoff_ms: 30000,
            backoff_multiplier: 2.0,
        }
    }
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            tracing: TracingConfig::default(),
            logging: LoggingConfig::default(),
        }
    }
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            endpoint: None,
            sampling_rate: 0.1,
            service_name: "qbitel-bridge-operator".to_string(),
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: "text".to_string(),
            structured: true,
            file_path: None,
            max_size_mb: 100,
            max_files: 5,
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            tls: TlsConfig::default(),
            authentication: AuthenticationConfig::default(),
            rbac: RbacConfig::default(),
        }
    }
}

impl Default for TlsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            cert_path: None,
            key_path: None,
            ca_path: None,
            min_version: "1.2".to_string(),
            cipher_suites: vec![],
        }
    }
}

impl Default for AuthenticationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            method: "serviceaccount".to_string(),
            token_file: None,
            service_account_token_path: Some("/var/run/secrets/kubernetes.io/serviceaccount/token".to_string()),
        }
    }
}

impl Default for RbacConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            service_account_name: "qbitel-bridge-operator".to_string(),
            cluster_role_name: "qbitel-bridge-operator".to_string(),
            additional_permissions: vec![],
        }
    }
}

impl OperatorConfig {
    /// Load configuration from file
    pub fn load() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let config_path = std::env::var("CONFIG_PATH").unwrap_or_else(|_| "/etc/config/operator.yaml".to_string());
        Self::load_from_file(&config_path)
    }
    
    /// Load configuration from specific file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let content = std::fs::read_to_string(path)?;
        
        // Try YAML first, then JSON
        if let Ok(config) = serde_yaml::from_str::<Self>(&content) {
            return Ok(config);
        }
        
        if let Ok(config) = serde_json::from_str::<Self>(&content) {
            return Ok(config);
        }
        
        Err("Failed to parse configuration file as YAML or JSON".into())
    }
    
    /// Load configuration from environment variables
    pub fn load_from_env() -> Self {
        let mut config = Self::default();
        
        // Override with environment variables
        if let Ok(name) = std::env::var("OPERATOR_NAME") {
            config.operator.name = name;
        }
        
        if let Ok(interval) = std::env::var("RECONCILE_INTERVAL_SECONDS") {
            if let Ok(val) = interval.parse() {
                config.operator.reconcile_interval_seconds = val;
            }
        }
        
        if let Ok(workers) = std::env::var("WORKER_COUNT") {
            if let Ok(val) = workers.parse() {
                config.operator.worker_count = val;
            }
        }
        
        if let Ok(port) = std::env::var("METRICS_PORT") {
            if let Ok(val) = port.parse() {
                config.metrics.port = val;
            }
        }
        
        if let Ok(port) = std::env::var("HEALTH_PORT") {
            if let Ok(val) = port.parse() {
                config.health.port = val;
            }
        }
        
        if let Ok(level) = std::env::var("LOG_LEVEL") {
            config.observability.logging.level = level;
        }
        
        if let Ok(format) = std::env::var("LOG_FORMAT") {
            config.observability.logging.format = format;
        }
        
        config
    }
    
    /// Get reconcile interval as Duration
    pub fn reconcile_interval(&self) -> Duration {
        Duration::from_secs(self.operator.reconcile_interval_seconds)
    }
    
    /// Get metrics bind address
    pub fn metrics_bind_address(&self) -> String {
        format!("{}:{}", self.metrics.bind_address, self.metrics.port)
    }
    
    /// Get health bind address
    pub fn health_bind_address(&self) -> String {
        format!("{}:{}", self.health.bind_address, self.health.port)
    }
    
    /// Check if controller is enabled
    pub fn is_controller_enabled(&self, controller: &str) -> bool {
        match controller {
            "dataplane" => self.controllers.dataplane.enabled,
            "controlplane" => self.controllers.controlplane.enabled,
            "aiengine" => self.controllers.aiengine.enabled,
            "policy-engine" => self.controllers.policy_engine.enabled,
            "servicemesh" => self.controllers.servicemesh.enabled,
            _ => false,
        }
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.operator.name.is_empty() {
            return Err("Operator name cannot be empty".to_string());
        }
        
        if self.operator.reconcile_interval_seconds == 0 {
            return Err("Reconcile interval must be greater than 0".to_string());
        }
        
        if self.operator.worker_count == 0 {
            return Err("Worker count must be greater than 0".to_string());
        }
        
        if self.metrics.port == 0 {
            return Err("Metrics port must be greater than 0".to_string());
        }
        
        if self.health.port == 0 {
            return Err("Health port must be greater than 0".to_string());
        }
        
        if self.metrics.port == self.health.port {
            return Err("Metrics port and health port cannot be the same".to_string());
        }
        
        // Validate log level
        match self.observability.logging.level.as_str() {
            "trace" | "debug" | "info" | "warn" | "error" => {},
            _ => return Err("Invalid log level. Must be one of: trace, debug, info, warn, error".to_string()),
        }
        
        // Validate log format
        match self.observability.logging.format.as_str() {
            "json" | "text" => {},
            _ => return Err("Invalid log format. Must be either 'json' or 'text'".to_string()),
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;
    
    #[test]
    fn test_default_config() {
        let config = OperatorConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.operator.name, "qbitel-bridge-operator");
        assert_eq!(config.metrics.port, 8080);
        assert_eq!(config.health.port, 8081);
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = OperatorConfig::default();
        
        // Test empty name
        config.operator.name = "".to_string();
        assert!(config.validate().is_err());
        
        // Reset and test zero interval
        config = OperatorConfig::default();
        config.operator.reconcile_interval_seconds = 0;
        assert!(config.validate().is_err());
        
        // Reset and test same ports
        config = OperatorConfig::default();
        config.health.port = config.metrics.port;
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_load_from_yaml() {
        let yaml_content = r#"
operator:
  name: "test-operator"
  version: "1.0.0"
  reconcile_interval_seconds: 60
  worker_count: 2
  leader_election_enabled: true
  leader_election_namespace: "test"
  leader_election_lock_name: "test-lock"
  shutdown_timeout_seconds: 30

metrics:
  enabled: true
  port: 9090
  bind_address: "0.0.0.0"
  collection_interval_seconds: 30
  custom_metrics: []

health:
  port: 8080
  bind_address: "0.0.0.0"
  check_interval_seconds: 30
  timeout_seconds: 10
  readiness:
    initial_delay_seconds: 10
    period_seconds: 10
    timeout_seconds: 5
    failure_threshold: 3
    success_threshold: 1
  liveness:
    initial_delay_seconds: 30
    period_seconds: 30
    timeout_seconds: 10
    failure_threshold: 3
"#;
        
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(yaml_content.as_bytes()).unwrap();
        
        let config = OperatorConfig::load_from_file(temp_file.path()).unwrap();
        assert_eq!(config.operator.name, "test-operator");
        assert_eq!(config.operator.reconcile_interval_seconds, 60);
        assert_eq!(config.metrics.port, 9090);
        assert_eq!(config.health.port, 8080);
    }
    
    #[test]
    fn test_controller_enabled() {
        let config = OperatorConfig::default();
        assert!(config.is_controller_enabled("dataplane"));
        assert!(config.is_controller_enabled("controlplane"));
        assert!(!config.is_controller_enabled("nonexistent"));
    }
}