//! Custom Resource Definitions for QBITEL Bridge services

use kube::CustomResource;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// DataPlane service configuration
#[derive(CustomResource, Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[kube(
    group = "qbitel.ai",
    version = "v1",
    kind = "DataPlaneService",
    plural = "dataplanes",
    derive = "Default",
    status = "DataPlaneStatus",
    shortname = "dp"
)]
#[serde(rename_all = "camelCase")]
pub struct DataPlaneSpec {
    /// Number of replicas
    pub replicas: u32,
    
    /// Container image to use
    pub image: String,
    
    /// Image pull policy
    pub image_pull_policy: Option<String>,
    
    /// Resource requirements
    pub resources: Option<ResourceRequirements>,
    
    /// Node selector
    pub node_selector: Option<HashMap<String, String>>,
    
    /// Tolerations
    pub tolerations: Option<Vec<Toleration>>,
    
    /// Affinity rules
    pub affinity: Option<Affinity>,
    
    /// DPDK configuration
    pub dpdk_config: DpdkConfig,
    
    /// DPI configuration
    pub dpi_config: DpiConfig,
    
    /// High-frequency metrics configuration
    pub metrics_config: Option<MetricsConfig>,
    
    /// Network interfaces to bind
    pub network_interfaces: Vec<NetworkInterface>,
    
    /// Security configuration
    pub security: Option<SecurityConfig>,
}

/// DataPlane status
#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct DataPlaneStatus {
    /// Current phase
    pub phase: String,
    
    /// Ready replicas
    pub ready_replicas: u32,
    
    /// Available replicas
    pub available_replicas: u32,
    
    /// Conditions
    pub conditions: Option<Vec<Condition>>,
    
    /// Performance metrics
    pub performance_metrics: Option<PerformanceMetrics>,
    
    /// Last updated timestamp
    pub last_updated: Option<String>,
}

/// ControlPlane service configuration
#[derive(CustomResource, Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[kube(
    group = "qbitel.ai",
    version = "v1",
    kind = "ControlPlaneService",
    plural = "controlplanes",
    derive = "Default",
    status = "ControlPlaneStatus",
    shortname = "cp"
)]
#[serde(rename_all = "camelCase")]
pub struct ControlPlaneSpec {
    /// Number of replicas
    pub replicas: u32,
    
    /// Container image
    pub image: String,
    
    /// Resource requirements
    pub resources: Option<ResourceRequirements>,
    
    /// Configuration service settings
    pub config_service: ConfigServiceSpec,
    
    /// Policy engine settings
    pub policy_engine: PolicyEngineSpec,
    
    /// Management API configuration
    pub management_api: ManagementApiSpec,
    
    /// Database configuration
    pub database: Option<DatabaseConfig>,
    
    /// High availability settings
    pub high_availability: Option<HighAvailabilityConfig>,
}

/// ControlPlane status
#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ControlPlaneStatus {
    /// Current phase
    pub phase: String,
    
    /// Ready replicas
    pub ready_replicas: u32,
    
    /// Leader instance
    pub leader: Option<String>,
    
    /// Conditions
    pub conditions: Option<Vec<Condition>>,
    
    /// Last updated timestamp
    pub last_updated: Option<String>,
}

/// AI Engine service configuration
#[derive(CustomResource, Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[kube(
    group = "qbitel.ai",
    version = "v1",
    kind = "AIEngineService",
    plural = "aiengines",
    derive = "Default",
    status = "AIEngineStatus",
    shortname = "ai"
)]
#[serde(rename_all = "camelCase")]
pub struct AIEngineSpec {
    /// Number of replicas
    pub replicas: u32,
    
    /// Container image
    pub image: String,
    
    /// Resource requirements (GPU support)
    pub resources: Option<ResourceRequirements>,
    
    /// GPU requirements
    pub gpu_requirements: Option<GpuRequirements>,
    
    /// Model management configuration
    pub model_management: ModelManagementSpec,
    
    /// Training pipeline configuration
    pub training_pipeline: TrainingPipelineSpec,
    
    /// Inference configuration
    pub inference: InferenceSpec,
    
    /// MLflow integration
    pub mlflow: Option<MlflowConfig>,
    
    /// Auto-scaling configuration
    pub auto_scaling: Option<AutoScalingConfig>,
}

/// AI Engine status
#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct AIEngineStatus {
    /// Current phase
    pub phase: String,
    
    /// Ready replicas
    pub ready_replicas: u32,
    
    /// Active models
    pub active_models: Option<Vec<String>>,
    
    /// Training jobs
    pub training_jobs: Option<Vec<TrainingJobStatus>>,
    
    /// Inference metrics
    pub inference_metrics: Option<InferenceMetrics>,
    
    /// Conditions
    pub conditions: Option<Vec<Condition>>,
    
    /// Last updated timestamp
    pub last_updated: Option<String>,
}

/// Policy Engine service configuration
#[derive(CustomResource, Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[kube(
    group = "qbitel.ai",
    version = "v1",
    kind = "PolicyEngineService",
    plural = "policyengines",
    derive = "Default",
    status = "PolicyEngineStatus",
    shortname = "pe"
)]
#[serde(rename_all = "camelCase")]
pub struct PolicyEngineSpec {
    /// Number of replicas
    pub replicas: u32,
    
    /// Container image
    pub image: String,
    
    /// Resource requirements
    pub resources: Option<ResourceRequirements>,
    
    /// Policy repositories
    pub policy_repositories: Vec<PolicyRepository>,
    
    /// Rule engines configuration
    pub rule_engines: RuleEnginesConfig,
    
    /// Compliance monitoring
    pub compliance_monitoring: Option<ComplianceMonitoringConfig>,
    
    /// Notification settings
    pub notifications: Option<NotificationConfig>,
}

/// Policy Engine status
#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PolicyEngineStatus {
    /// Current phase
    pub phase: String,
    
    /// Ready replicas
    pub ready_replicas: u32,
    
    /// Active policies
    pub active_policies: Option<Vec<String>>,
    
    /// Violations count
    pub violations_count: Option<u32>,
    
    /// Compliance score
    pub compliance_score: Option<f64>,
    
    /// Conditions
    pub conditions: Option<Vec<Condition>>,
    
    /// Last updated timestamp
    pub last_updated: Option<String>,
}

/// Service Mesh configuration
#[derive(CustomResource, Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[kube(
    group = "qbitel.ai",
    version = "v1",
    kind = "ServiceMeshConfig",
    plural = "servicemeshconfigs",
    derive = "Default",
    status = "ServiceMeshStatus",
    shortname = "sm"
)]
#[serde(rename_all = "camelCase")]
pub struct ServiceMeshSpec {
    /// Service mesh provider (istio, linkerd, consul-connect)
    pub provider: String,
    
    /// Mutual TLS configuration
    pub mtls: Option<MutualTlsConfig>,
    
    /// Traffic management
    pub traffic_management: Option<TrafficManagementConfig>,
    
    /// Observability configuration
    pub observability: Option<ObservabilityConfig>,
    
    /// Security policies
    pub security_policies: Option<SecurityPoliciesConfig>,
    
    /// Circuit breaker settings
    pub circuit_breaker: Option<CircuitBreakerConfig>,
    
    /// Rate limiting
    pub rate_limiting: Option<RateLimitingConfig>,
}

/// Service Mesh status
#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ServiceMeshStatus {
    /// Current phase
    pub phase: String,
    
    /// Mesh version
    pub mesh_version: Option<String>,
    
    /// Connected services
    pub connected_services: Option<Vec<String>>,
    
    /// TLS status
    pub tls_status: Option<String>,
    
    /// Conditions
    pub conditions: Option<Vec<Condition>>,
    
    /// Last updated timestamp
    pub last_updated: Option<String>,
}

// Supporting structures

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ResourceRequirements {
    pub limits: Option<HashMap<String, String>>,
    pub requests: Option<HashMap<String, String>>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct Toleration {
    pub key: Option<String>,
    pub operator: Option<String>,
    pub value: Option<String>,
    pub effect: Option<String>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct Affinity {
    pub node_affinity: Option<NodeAffinity>,
    pub pod_affinity: Option<PodAffinity>,
    pub pod_anti_affinity: Option<PodAntiAffinity>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct NodeAffinity {
    pub required_during_scheduling_ignored_during_execution: Option<NodeSelector>,
    pub preferred_during_scheduling_ignored_during_execution: Option<Vec<PreferredSchedulingTerm>>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct NodeSelector {
    pub node_selector_terms: Vec<NodeSelectorTerm>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct NodeSelectorTerm {
    pub match_expressions: Option<Vec<NodeSelectorRequirement>>,
    pub match_fields: Option<Vec<NodeSelectorRequirement>>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct NodeSelectorRequirement {
    pub key: String,
    pub operator: String,
    pub values: Option<Vec<String>>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PreferredSchedulingTerm {
    pub weight: i32,
    pub preference: NodeSelectorTerm,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PodAffinity {
    pub required_during_scheduling_ignored_during_execution: Option<Vec<PodAffinityTerm>>,
    pub preferred_during_scheduling_ignored_during_execution: Option<Vec<WeightedPodAffinityTerm>>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PodAntiAffinity {
    pub required_during_scheduling_ignored_during_execution: Option<Vec<PodAffinityTerm>>,
    pub preferred_during_scheduling_ignored_during_execution: Option<Vec<WeightedPodAffinityTerm>>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PodAffinityTerm {
    pub label_selector: Option<LabelSelector>,
    pub topology_key: String,
    pub namespace_selector: Option<LabelSelector>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct WeightedPodAffinityTerm {
    pub weight: i32,
    pub pod_affinity_term: PodAffinityTerm,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct LabelSelector {
    pub match_labels: Option<HashMap<String, String>>,
    pub match_expressions: Option<Vec<LabelSelectorRequirement>>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct LabelSelectorRequirement {
    pub key: String,
    pub operator: String,
    pub values: Option<Vec<String>>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct DpdkConfig {
    pub enabled: bool,
    pub memory_channels: Option<u32>,
    pub cores: Option<Vec<u32>>,
    pub huge_pages: Option<String>,
    pub pci_allowlist: Option<Vec<String>>,
    pub driver: Option<String>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct DpiConfig {
    pub enabled: bool,
    pub ml_models: Option<Vec<String>>,
    pub pattern_databases: Option<Vec<String>>,
    pub classification_engines: Option<Vec<String>>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MetricsConfig {
    pub enabled: bool,
    pub sampling_rate_hz: Option<u32>,
    pub buffer_size: Option<u32>,
    pub export_interval_ms: Option<u32>,
    pub collectors: Option<Vec<String>>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct NetworkInterface {
    pub name: String,
    pub pci_address: Option<String>,
    pub driver: Option<String>,
    pub queues: Option<u32>,
    pub mtu: Option<u32>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SecurityConfig {
    pub encryption_enabled: bool,
    pub tls_version: Option<String>,
    pub certificate_authority: Option<String>,
    pub quantum_safe: Option<bool>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct Condition {
    pub condition_type: String,
    pub status: String,
    pub reason: Option<String>,
    pub message: Option<String>,
    pub last_transition_time: Option<String>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PerformanceMetrics {
    pub packets_per_second: Option<u64>,
    pub bytes_per_second: Option<u64>,
    pub cpu_utilization: Option<f64>,
    pub memory_utilization: Option<f64>,
    pub latency_microseconds: Option<u64>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ConfigServiceSpec {
    pub storage_backend: String,
    pub encryption_enabled: bool,
    pub backup_enabled: bool,
    pub sync_interval_seconds: Option<u32>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PolicyEngineSpec {
    pub rule_engines: Vec<String>,
    pub policy_repositories: Vec<String>,
    pub evaluation_interval_seconds: Option<u32>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ManagementApiSpec {
    pub port: u32,
    pub tls_enabled: bool,
    pub authentication: String,
    pub rate_limiting: Option<RateLimitingConfig>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct DatabaseConfig {
    pub type_: String,
    pub host: String,
    pub port: u32,
    pub database: String,
    pub credentials_secret: String,
    pub connection_pool_size: Option<u32>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct HighAvailabilityConfig {
    pub enabled: bool,
    pub election_timeout_ms: Option<u32>,
    pub heartbeat_interval_ms: Option<u32>,
    pub backup_count: Option<u32>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct GpuRequirements {
    pub count: u32,
    pub memory: String,
    pub type_: Option<String>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ModelManagementSpec {
    pub registry_url: String,
    pub model_store: String,
    pub versioning_enabled: bool,
    pub auto_deployment: Option<bool>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TrainingPipelineSpec {
    pub enabled: bool,
    pub job_queue: String,
    pub resource_limits: Option<ResourceRequirements>,
    pub data_sources: Vec<String>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct InferenceSpec {
    pub batch_size: Option<u32>,
    pub timeout_ms: Option<u32>,
    pub auto_scaling: Option<AutoScalingConfig>,
    pub model_serving: Vec<ModelServingSpec>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ModelServingSpec {
    pub model_name: String,
    pub model_version: String,
    pub replicas: u32,
    pub resources: Option<ResourceRequirements>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MlflowConfig {
    pub tracking_uri: String,
    pub experiment_name: String,
    pub artifact_store: String,
    pub model_registry: Option<String>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct AutoScalingConfig {
    pub enabled: bool,
    pub min_replicas: u32,
    pub max_replicas: u32,
    pub target_cpu_utilization: Option<u32>,
    pub target_memory_utilization: Option<u32>,
    pub scale_up_policy: Option<ScalingPolicy>,
    pub scale_down_policy: Option<ScalingPolicy>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ScalingPolicy {
    pub period_seconds: u32,
    pub stabilization_window_seconds: u32,
    pub policies: Vec<HPAScalingRule>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct HPAScalingRule {
    pub type_: String,
    pub value: u32,
    pub period_seconds: u32,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TrainingJobStatus {
    pub job_id: String,
    pub status: String,
    pub model_name: String,
    pub progress: Option<f64>,
    pub started_at: Option<String>,
    pub completed_at: Option<String>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct InferenceMetrics {
    pub requests_per_second: Option<f64>,
    pub average_latency_ms: Option<f64>,
    pub error_rate: Option<f64>,
    pub active_models: Option<u32>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PolicyRepository {
    pub name: String,
    pub url: String,
    pub branch: Option<String>,
    pub credentials_secret: Option<String>,
    pub sync_interval_seconds: Option<u32>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct RuleEnginesConfig {
    pub python_engine: Option<PythonEngineConfig>,
    pub rego_engine: Option<RegoEngineConfig>,
    pub jsonpath_engine: Option<JsonPathEngineConfig>,
    pub regex_engine: Option<RegexEngineConfig>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PythonEngineConfig {
    pub enabled: bool,
    pub sandbox_mode: Option<bool>,
    pub timeout_seconds: Option<u32>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct RegoEngineConfig {
    pub enabled: bool,
    pub data_sources: Option<Vec<String>>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct JsonPathEngineConfig {
    pub enabled: bool,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct RegexEngineConfig {
    pub enabled: bool,
    pub compilation_cache_size: Option<u32>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ComplianceMonitoringConfig {
    pub enabled: bool,
    pub frameworks: Vec<String>,
    pub reporting_interval_hours: Option<u32>,
    pub alert_threshold: Option<f64>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct NotificationConfig {
    pub channels: Vec<NotificationChannel>,
    pub rate_limiting: Option<RateLimitingConfig>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct NotificationChannel {
    pub type_: String,
    pub endpoint: String,
    pub credentials_secret: Option<String>,
    pub severity_filter: Option<Vec<String>>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MutualTlsConfig {
    pub enabled: bool,
    pub mode: String, // STRICT, PERMISSIVE, DISABLE
    pub certificate_authority: Option<String>,
    pub auto_rotation: Option<bool>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TrafficManagementConfig {
    pub load_balancing: Option<LoadBalancingConfig>,
    pub timeout_config: Option<TimeoutConfig>,
    pub retry_policy: Option<RetryPolicy>,
    pub fault_injection: Option<FaultInjectionConfig>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct LoadBalancingConfig {
    pub algorithm: String, // ROUND_ROBIN, LEAST_CONN, RANDOM, RING_HASH
    pub consistent_hash: Option<ConsistentHashConfig>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ConsistentHashConfig {
    pub hash_based_on: String, // header, cookie, source_ip
    pub hash_key: Option<String>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TimeoutConfig {
    pub request_timeout_seconds: Option<u32>,
    pub idle_timeout_seconds: Option<u32>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub per_try_timeout_seconds: Option<u32>,
    pub retry_on: Vec<String>,
    pub backoff: Option<BackoffConfig>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct BackoffConfig {
    pub base_interval_seconds: f64,
    pub max_interval_seconds: f64,
    pub multiplier: Option<f64>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct FaultInjectionConfig {
    pub delay: Option<DelayConfig>,
    pub abort: Option<AbortConfig>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct DelayConfig {
    pub percentage: f64,
    pub fixed_delay_seconds: f64,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct AbortConfig {
    pub percentage: f64,
    pub http_status: u32,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ObservabilityConfig {
    pub tracing: Option<TracingConfig>,
    pub metrics: Option<ServiceMeshMetricsConfig>,
    pub access_logging: Option<AccessLoggingConfig>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TracingConfig {
    pub enabled: bool,
    pub sampling_rate: Option<f64>,
    pub jaeger_endpoint: Option<String>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ServiceMeshMetricsConfig {
    pub enabled: bool,
    pub prometheus_endpoint: Option<String>,
    pub custom_metrics: Option<Vec<String>>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct AccessLoggingConfig {
    pub enabled: bool,
    pub format: Option<String>,
    pub destination: Option<String>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SecurityPoliciesConfig {
    pub authorization_policies: Option<Vec<AuthorizationPolicy>>,
    pub peer_authentication: Option<PeerAuthenticationConfig>,
    pub request_authentication: Option<RequestAuthenticationConfig>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct AuthorizationPolicy {
    pub name: String,
    pub selector: Option<HashMap<String, String>>,
    pub rules: Vec<AuthorizationRule>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct AuthorizationRule {
    pub from: Option<Vec<AuthorizationSource>>,
    pub to: Option<Vec<AuthorizationOperation>>,
    pub when: Option<Vec<AuthorizationCondition>>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct AuthorizationSource {
    pub principals: Option<Vec<String>>,
    pub namespaces: Option<Vec<String>>,
    pub ip_blocks: Option<Vec<String>>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct AuthorizationOperation {
    pub methods: Option<Vec<String>>,
    pub paths: Option<Vec<String>>,
    pub ports: Option<Vec<String>>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct AuthorizationCondition {
    pub key: String,
    pub values: Vec<String>,
    pub not_values: Option<Vec<String>>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PeerAuthenticationConfig {
    pub mtls_mode: String, // STRICT, PERMISSIVE, DISABLE
    pub port_level_mtls: Option<HashMap<String, String>>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct RequestAuthenticationConfig {
    pub jwt_rules: Option<Vec<JwtRule>>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct JwtRule {
    pub issuer: String,
    pub jwks_uri: Option<String>,
    pub audiences: Option<Vec<String>>,
    pub from_headers: Option<Vec<JwtHeader>>,
    pub from_params: Option<Vec<String>>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct JwtHeader {
    pub name: String,
    pub prefix: Option<String>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct CircuitBreakerConfig {
    pub enabled: bool,
    pub max_connections: Option<u32>,
    pub max_pending_requests: Option<u32>,
    pub max_requests_per_connection: Option<u32>,
    pub consecutive_errors: Option<u32>,
    pub interval_seconds: Option<u32>,
    pub base_ejection_time_seconds: Option<u32>,
}

#[derive(Deserialize, Serialize, Clone, Debug, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct RateLimitingConfig {
    pub enabled: bool,
    pub requests_per_second: Option<u32>,
    pub requests_per_minute: Option<u32>,
    pub burst_size: Option<u32>,
    pub rate_limit_backend: Option<String>, // redis, memcached, local
}