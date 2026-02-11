//! Controller modules for QBITEL Bridge services

pub mod dataplane;
pub mod controlplane;
pub mod aiengine;
pub mod policy;
pub mod servicemesh;

use crate::OperatorContext;
use anyhow::Result;
use kube_runtime::controller::Action;
use std::sync::Arc;
use std::time::Duration;
use tracing::{error, info, warn};

/// Standard reconcile interval
pub const RECONCILE_INTERVAL: Duration = Duration::from_secs(300); // 5 minutes

/// Fast reconcile interval for critical resources
pub const FAST_RECONCILE_INTERVAL: Duration = Duration::from_secs(60); // 1 minute

/// Error requeue interval
pub const ERROR_REQUEUE_INTERVAL: Duration = Duration::from_secs(30);

/// Max error requeue interval
pub const MAX_ERROR_REQUEUE_INTERVAL: Duration = Duration::from_secs(600); // 10 minutes

/// Common error policy for controllers
pub fn default_error_policy(
    _obj: Arc<impl std::fmt::Debug>,
    error: &crate::error::OperatorError,
    _ctx: Arc<OperatorContext>,
) -> Action {
    error!("Controller error: {}", error);
    
    match error {
        crate::error::OperatorError::TemporaryFailure(_) => {
            warn!("Temporary failure, retrying in {:?}", ERROR_REQUEUE_INTERVAL);
            Action::requeue(ERROR_REQUEUE_INTERVAL)
        }
        crate::error::OperatorError::PermanentFailure(_) => {
            error!("Permanent failure, not retrying");
            Action::await_change()
        }
        _ => {
            warn!("Unknown error, retrying in {:?}", MAX_ERROR_REQUEUE_INTERVAL);
            Action::requeue(MAX_ERROR_REQUEUE_INTERVAL)
        }
    }
}

/// Generate common labels for QBITEL Bridge resources
pub fn common_labels() -> std::collections::BTreeMap<String, String> {
    let mut labels = std::collections::BTreeMap::new();
    labels.insert("app.kubernetes.io/name".to_string(), "qbitel-bridge".to_string());
    labels.insert("app.kubernetes.io/managed-by".to_string(), "qbitel-bridge-operator".to_string());
    labels.insert("app.kubernetes.io/version".to_string(), env!("CARGO_PKG_VERSION").to_string());
    labels
}

/// Generate resource-specific labels
pub fn resource_labels(component: &str, instance: &str) -> std::collections::BTreeMap<String, String> {
    let mut labels = common_labels();
    labels.insert("app.kubernetes.io/component".to_string(), component.to_string());
    labels.insert("app.kubernetes.io/instance".to_string(), instance.to_string());
    labels
}

/// Generate selector labels for matching resources
pub fn selector_labels(component: &str, instance: &str) -> std::collections::BTreeMap<String, String> {
    let mut labels = std::collections::BTreeMap::new();
    labels.insert("app.kubernetes.io/name".to_string(), "qbitel-bridge".to_string());
    labels.insert("app.kubernetes.io/component".to_string(), component.to_string());
    labels.insert("app.kubernetes.io/instance".to_string(), instance.to_string());
    labels
}

/// Common annotations for QBITEL Bridge resources
pub fn common_annotations() -> std::collections::BTreeMap<String, String> {
    let mut annotations = std::collections::BTreeMap::new();
    annotations.insert("qbitel.ai/operator-version".to_string(), env!("CARGO_PKG_VERSION").to_string());
    annotations.insert("qbitel.ai/managed-by-operator".to_string(), "true".to_string());
    annotations
}

/// Check if resource needs update based on generation
pub fn needs_update(current_generation: i64, observed_generation: Option<i64>) -> bool {
    match observed_generation {
        Some(observed) => current_generation > observed,
        None => true,
    }
}

/// Create default resource requirements
pub fn default_resources() -> k8s_openapi::api::core::v1::ResourceRequirements {
    use k8s_openapi::apimachinery::pkg::api::resource::Quantity;
    use std::collections::BTreeMap;
    
    let mut requests = BTreeMap::new();
    requests.insert("cpu".to_string(), Quantity("100m".to_string()));
    requests.insert("memory".to_string(), Quantity("128Mi".to_string()));
    
    let mut limits = BTreeMap::new();
    limits.insert("cpu".to_string(), Quantity("1000m".to_string()));
    limits.insert("memory".to_string(), Quantity("1Gi".to_string()));
    
    k8s_openapi::api::core::v1::ResourceRequirements {
        limits: Some(limits),
        requests: Some(requests),
        ..Default::default()
    }
}

/// Create security context for containers
pub fn default_security_context() -> k8s_openapi::api::core::v1::SecurityContext {
    k8s_openapi::api::core::v1::SecurityContext {
        run_as_non_root: Some(true),
        run_as_user: Some(10001),
        run_as_group: Some(10001),
        read_only_root_filesystem: Some(true),
        allow_privilege_escalation: Some(false),
        capabilities: Some(k8s_openapi::api::core::v1::Capabilities {
            drop: Some(vec!["ALL".to_string()]),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Create privileged security context for dataplane containers
pub fn privileged_security_context() -> k8s_openapi::api::core::v1::SecurityContext {
    k8s_openapi::api::core::v1::SecurityContext {
        privileged: Some(true),
        run_as_user: Some(0),
        capabilities: Some(k8s_openapi::api::core::v1::Capabilities {
            add: Some(vec![
                "SYS_ADMIN".to_string(),
                "SYS_RESOURCE".to_string(),
                "NET_ADMIN".to_string(),
                "NET_RAW".to_string(),
                "IPC_LOCK".to_string(),
            ]),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Create pod security context
pub fn default_pod_security_context() -> k8s_openapi::api::core::v1::PodSecurityContext {
    k8s_openapi::api::core::v1::PodSecurityContext {
        run_as_non_root: Some(true),
        run_as_user: Some(10001),
        run_as_group: Some(10001),
        fs_group: Some(10001),
        ..Default::default()
    }
}

/// Create privileged pod security context for dataplane
pub fn privileged_pod_security_context() -> k8s_openapi::api::core::v1::PodSecurityContext {
    k8s_openapi::api::core::v1::PodSecurityContext {
        run_as_user: Some(0),
        fs_group: Some(0),
        ..Default::default()
    }
}

/// Generate deployment strategy
pub fn default_deployment_strategy() -> k8s_openapi::api::apps::v1::DeploymentStrategy {
    k8s_openapi::api::apps::v1::DeploymentStrategy {
        type_: Some("RollingUpdate".to_string()),
        rolling_update: Some(k8s_openapi::api::apps::v1::RollingUpdateDeployment {
            max_surge: Some(k8s_openapi::apimachinery::pkg::util::intstr::IntOrString::String("25%".to_string())),
            max_unavailable: Some(k8s_openapi::apimachinery::pkg::util::intstr::IntOrString::String("25%".to_string())),
        }),
    }
}

/// Generate recreate deployment strategy for stateful components
pub fn recreate_deployment_strategy() -> k8s_openapi::api::apps::v1::DeploymentStrategy {
    k8s_openapi::api::apps::v1::DeploymentStrategy {
        type_: Some("Recreate".to_string()),
        rolling_update: None,
    }
}

/// Create readiness probe
pub fn default_readiness_probe(path: &str, port: i32) -> k8s_openapi::api::core::v1::Probe {
    k8s_openapi::api::core::v1::Probe {
        http_get: Some(k8s_openapi::api::core::v1::HTTPGetAction {
            path: Some(path.to_string()),
            port: k8s_openapi::apimachinery::pkg::util::intstr::IntOrString::Int(port),
            scheme: Some("HTTP".to_string()),
            ..Default::default()
        }),
        initial_delay_seconds: Some(10),
        period_seconds: Some(10),
        timeout_seconds: Some(5),
        failure_threshold: Some(3),
        success_threshold: Some(1),
        ..Default::default()
    }
}

/// Create liveness probe
pub fn default_liveness_probe(path: &str, port: i32) -> k8s_openapi::api::core::v1::Probe {
    k8s_openapi::api::core::v1::Probe {
        http_get: Some(k8s_openapi::api::core::v1::HTTPGetAction {
            path: Some(path.to_string()),
            port: k8s_openapi::apimachinery::pkg::util::intstr::IntOrString::Int(port),
            scheme: Some("HTTP".to_string()),
            ..Default::default()
        }),
        initial_delay_seconds: Some(30),
        period_seconds: Some(30),
        timeout_seconds: Some(10),
        failure_threshold: Some(3),
        success_threshold: Some(1),
        ..Default::default()
    }
}

/// Create startup probe for slow-starting services
pub fn default_startup_probe(path: &str, port: i32) -> k8s_openapi::api::core::v1::Probe {
    k8s_openapi::api::core::v1::Probe {
        http_get: Some(k8s_openapi::api::core::v1::HTTPGetAction {
            path: Some(path.to_string()),
            port: k8s_openapi::apimachinery::pkg::util::intstr::IntOrString::Int(port),
            scheme: Some("HTTP".to_string()),
            ..Default::default()
        }),
        initial_delay_seconds: Some(10),
        period_seconds: Some(10),
        timeout_seconds: Some(5),
        failure_threshold: Some(30), // Allow up to 5 minutes for startup
        success_threshold: Some(1),
        ..Default::default()
    }
}

/// Create environment variables from config map
pub fn env_from_config_map(config_map_name: &str) -> Vec<k8s_openapi::api::core::v1::EnvVar> {
    vec![
        k8s_openapi::api::core::v1::EnvVar {
            name: "CONFIG_MAP_NAME".to_string(),
            value: Some(config_map_name.to_string()),
            ..Default::default()
        }
    ]
}

/// Create environment variables from secret
pub fn env_from_secret(secret_name: &str, key: &str, env_name: &str) -> k8s_openapi::api::core::v1::EnvVar {
    k8s_openapi::api::core::v1::EnvVar {
        name: env_name.to_string(),
        value_from: Some(k8s_openapi::api::core::v1::EnvVarSource {
            secret_key_ref: Some(k8s_openapi::api::core::v1::SecretKeySelector {
                key: key.to_string(),
                name: Some(secret_name.to_string()),
                optional: Some(false),
            }),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Create volume mount for config map
pub fn config_map_volume_mount(name: &str, mount_path: &str, config_map_name: &str) -> (
    k8s_openapi::api::core::v1::Volume,
    k8s_openapi::api::core::v1::VolumeMount,
) {
    let volume = k8s_openapi::api::core::v1::Volume {
        name: name.to_string(),
        config_map: Some(k8s_openapi::api::core::v1::ConfigMapVolumeSource {
            name: Some(config_map_name.to_string()),
            default_mode: Some(0o644),
            ..Default::default()
        }),
        ..Default::default()
    };
    
    let volume_mount = k8s_openapi::api::core::v1::VolumeMount {
        name: name.to_string(),
        mount_path: mount_path.to_string(),
        read_only: Some(true),
        ..Default::default()
    };
    
    (volume, volume_mount)
}

/// Create volume mount for secret
pub fn secret_volume_mount(name: &str, mount_path: &str, secret_name: &str) -> (
    k8s_openapi::api::core::v1::Volume,
    k8s_openapi::api::core::v1::VolumeMount,
) {
    let volume = k8s_openapi::api::core::v1::Volume {
        name: name.to_string(),
        secret: Some(k8s_openapi::api::core::v1::SecretVolumeSource {
            secret_name: Some(secret_name.to_string()),
            default_mode: Some(0o600),
            ..Default::default()
        }),
        ..Default::default()
    };
    
    let volume_mount = k8s_openapi::api::core::v1::VolumeMount {
        name: name.to_string(),
        mount_path: mount_path.to_string(),
        read_only: Some(true),
        ..Default::default()
    };
    
    (volume, volume_mount)
}

/// Create empty dir volume mount
pub fn empty_dir_volume_mount(name: &str, mount_path: &str) -> (
    k8s_openapi::api::core::v1::Volume,
    k8s_openapi::api::core::v1::VolumeMount,
) {
    let volume = k8s_openapi::api::core::v1::Volume {
        name: name.to_string(),
        empty_dir: Some(k8s_openapi::api::core::v1::EmptyDirVolumeSource {
            medium: None,
            size_limit: None,
        }),
        ..Default::default()
    };
    
    let volume_mount = k8s_openapi::api::core::v1::VolumeMount {
        name: name.to_string(),
        mount_path: mount_path.to_string(),
        read_only: Some(false),
        ..Default::default()
    };
    
    (volume, volume_mount)
}

/// Create hugepages volume mount for DPDK
pub fn hugepages_volume_mount(name: &str, mount_path: &str, size: &str) -> (
    k8s_openapi::api::core::v1::Volume,
    k8s_openapi::api::core::v1::VolumeMount,
) {
    let volume = k8s_openapi::api::core::v1::Volume {
        name: name.to_string(),
        empty_dir: Some(k8s_openapi::api::core::v1::EmptyDirVolumeSource {
            medium: Some("HugePages".to_string()),
            size_limit: Some(k8s_openapi::apimachinery::pkg::api::resource::Quantity(size.to_string())),
        }),
        ..Default::default()
    };
    
    let volume_mount = k8s_openapi::api::core::v1::VolumeMount {
        name: name.to_string(),
        mount_path: mount_path.to_string(),
        read_only: Some(false),
        ..Default::default()
    };
    
    (volume, volume_mount)
}

/// Create service for a component
pub fn create_service(
    name: &str,
    namespace: &str,
    labels: std::collections::BTreeMap<String, String>,
    selector: std::collections::BTreeMap<String, String>,
    ports: Vec<k8s_openapi::api::core::v1::ServicePort>,
) -> k8s_openapi::api::core::v1::Service {
    k8s_openapi::api::core::v1::Service {
        metadata: k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta {
            name: Some(name.to_string()),
            namespace: Some(namespace.to_string()),
            labels: Some(labels),
            annotations: Some(common_annotations()),
            ..Default::default()
        },
        spec: Some(k8s_openapi::api::core::v1::ServiceSpec {
            selector: Some(selector),
            ports: Some(ports),
            type_: Some("ClusterIP".to_string()),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Create config map
pub fn create_config_map(
    name: &str,
    namespace: &str,
    labels: std::collections::BTreeMap<String, String>,
    data: std::collections::BTreeMap<String, String>,
) -> k8s_openapi::api::core::v1::ConfigMap {
    k8s_openapi::api::core::v1::ConfigMap {
        metadata: k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta {
            name: Some(name.to_string()),
            namespace: Some(namespace.to_string()),
            labels: Some(labels),
            annotations: Some(common_annotations()),
            ..Default::default()
        },
        data: Some(data),
        ..Default::default()
    }
}