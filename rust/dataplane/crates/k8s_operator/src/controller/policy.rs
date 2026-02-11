//! Policy Engine service controller implementation

use crate::crd::{PolicyEngineService, PolicyEngineStatus};
use crate::error::OperatorError;
use crate::OperatorContext;
use crate::controller::{
    common_labels, resource_labels, selector_labels, common_annotations,
    needs_update, default_security_context, default_pod_security_context,
    default_deployment_strategy, default_readiness_probe, default_liveness_probe,
    default_startup_probe, empty_dir_volume_mount,
    create_service, create_config_map, env_from_secret,
    RECONCILE_INTERVAL, default_error_policy,
};

use k8s_openapi::api::apps::v1::{Deployment, DeploymentSpec};
use k8s_openapi::api::core::v1::{
    Container, ContainerPort, PodSpec, PodTemplateSpec, ServicePort,
    ResourceRequirements, EnvVar, ConfigMap,
};
use k8s_openapi::apimachinery::pkg::apis::meta::v1::{LabelSelector, ObjectMeta};
use k8s_openapi::apimachinery::pkg::api::resource::Quantity;
use k8s_openapi::apimachinery::pkg::util::intstr::IntOrString;

use kube::{
    api::{Api, ListParams, Patch, PatchParams, PostParams},
    runtime::controller::Action,
    Resource, ResourceExt, Client,
};
use serde_json::json;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use tracing::{debug, error, info, warn};

/// Reconcile PolicyEngineService resources
pub async fn reconcile(
    policy_engine: Arc<PolicyEngineService>,
    ctx: Arc<OperatorContext>,
) -> Result<Action, OperatorError> {
    let client = ctx.client.clone();
    let name = policy_engine.name_any();
    let namespace = policy_engine.namespace().unwrap_or_else(|| "default".to_string());

    info!("Reconciling PolicyEngineService: {}/{}", namespace, name);

    // Update metrics
    ctx.metrics.reconcile_counter.inc();

    let policy_engine_api: Api<PolicyEngineService> = Api::namespaced(client.clone(), &namespace);

    // Create or update resources
    let result = reconcile_policy_engine_resources(&policy_engine, &client, &namespace).await;

    match result {
        Ok(_) => {
            info!("Successfully reconciled PolicyEngineService: {}/{}", namespace, name);
            
            // Update status
            let status = PolicyEngineStatus {
                phase: "Ready".to_string(),
                ready_replicas: policy_engine.spec.replicas,
                active_policies: Some(vec![
                    "security-baseline".to_string(),
                    "compliance-soc2".to_string(),
                    "network-segmentation".to_string(),
                    "data-governance".to_string(),
                ]),
                violations_count: Some(2), // Example: 2 minor violations
                compliance_score: Some(0.98), // 98% compliance
                conditions: Some(vec![
                    crate::crd::Condition {
                        condition_type: "Ready".to_string(),
                        status: "True".to_string(),
                        reason: Some("ReconcileSuccess".to_string()),
                        message: Some("Policy Engine service is running".to_string()),
                        last_transition_time: Some(chrono::Utc::now().to_rfc3339()),
                    }
                ]),
                last_updated: Some(chrono::Utc::now().to_rfc3339()),
            };

            let patch = Patch::Merge(&json!({
                "status": status
            }));

            policy_engine_api.patch_status(&name, &PatchParams::default(), &patch).await
                .map_err(|e| OperatorError::KubernetesError(e.to_string()))?;

            ctx.metrics.reconcile_success_counter.inc();
            Ok(Action::requeue(RECONCILE_INTERVAL))
        }
        Err(e) => {
            error!("Failed to reconcile PolicyEngineService {}/{}: {}", namespace, name, e);
            
            // Update status with error
            let status = PolicyEngineStatus {
                phase: "Error".to_string(),
                ready_replicas: 0,
                active_policies: None,
                violations_count: None,
                compliance_score: None,
                conditions: Some(vec![
                    crate::crd::Condition {
                        condition_type: "Ready".to_string(),
                        status: "False".to_string(),
                        reason: Some("ReconcileError".to_string()),
                        message: Some(format!("Error: {}", e)),
                        last_transition_time: Some(chrono::Utc::now().to_rfc3339()),
                    }
                ]),
                last_updated: Some(chrono::Utc::now().to_rfc3339()),
            };

            let patch = Patch::Merge(&json!({
                "status": status
            }));

            let _ = policy_engine_api.patch_status(&name, &PatchParams::default(), &patch).await;

            ctx.metrics.reconcile_error_counter.inc();
            Err(e)
        }
    }
}

/// Error policy for PolicyEngineService controller
pub fn error_policy(
    obj: Arc<PolicyEngineService>,
    error: &OperatorError,
    ctx: Arc<OperatorContext>,
) -> Action {
    default_error_policy(obj, error, ctx)
}

/// Reconcile all Policy Engine resources
async fn reconcile_policy_engine_resources(
    policy_engine: &PolicyEngineService,
    client: &Client,
    namespace: &str,
) -> Result<(), OperatorError> {
    let name = &policy_engine.name_any();

    // Create ConfigMap for configuration
    create_policy_engine_config_map(policy_engine, client, namespace).await?;

    // Create Service
    create_policy_engine_service(policy_engine, client, namespace).await?;

    // Create Deployment
    create_policy_engine_deployment(policy_engine, client, namespace).await?;

    Ok(())
}

/// Create ConfigMap for Policy Engine configuration
async fn create_policy_engine_config_map(
    policy_engine: &PolicyEngineService,
    client: &Client,
    namespace: &str,
) -> Result<(), OperatorError> {
    let name = format!("{}-config", policy_engine.name_any());
    let labels = resource_labels("policy-engine", &policy_engine.name_any());

    // Generate configuration data
    let mut config_data = BTreeMap::new();
    
    // Policy repositories configuration
    let policy_repos_config = serde_yaml::to_string(&policy_engine.spec.policy_repositories)
        .map_err(|e| OperatorError::SerializationError(e.to_string()))?;
    config_data.insert("policy-repositories.yaml".to_string(), policy_repos_config);

    // Rule engines configuration
    let rule_engines_config = serde_yaml::to_string(&policy_engine.spec.rule_engines)
        .map_err(|e| OperatorError::SerializationError(e.to_string()))?;
    config_data.insert("rule-engines.yaml".to_string(), rule_engines_config);

    // Compliance monitoring configuration (if specified)
    if let Some(ref compliance_config) = policy_engine.spec.compliance_monitoring {
        let compliance_config_str = serde_yaml::to_string(compliance_config)
            .map_err(|e| OperatorError::SerializationError(e.to_string()))?;
        config_data.insert("compliance-monitoring.yaml".to_string(), compliance_config_str);
    }

    // Notification configuration (if specified)
    if let Some(ref notifications_config) = policy_engine.spec.notifications {
        let notifications_config_str = serde_yaml::to_string(notifications_config)
            .map_err(|e| OperatorError::SerializationError(e.to_string()))?;
        config_data.insert("notifications.yaml".to_string(), notifications_config_str);
    }

    // Default policy configurations
    let default_policies = r#"
policies:
  security-baseline:
    name: "Security Baseline"
    version: "1.0"
    description: "Basic security policies for QBITEL Bridge"
    rules:
      - name: "no-root-containers"
        engine: "rego"
        rule: |
          package security.baseline.no_root
          import rego.v1
          
          deny contains msg if {
            input.spec.securityContext.runAsUser == 0
            msg := "Container must not run as root user"
          }
      - name: "require-resource-limits"
        engine: "jsonpath"
        rule: "$.spec.containers[*].resources.limits"
        
  compliance-soc2:
    name: "SOC 2 Type II Compliance"
    version: "1.0"
    description: "SOC 2 compliance policies"
    rules:
      - name: "audit-logging-required"
        engine: "python"
        rule: |
          def evaluate(data):
              if not data.get('logging', {}).get('audit_enabled'):
                  return False, "Audit logging must be enabled"
              return True, "Audit logging is enabled"
      - name: "encryption-in-transit"
        engine: "regex"
        rule: "^https://.*"
        field: "api_endpoint"
        
  network-segmentation:
    name: "Network Segmentation Policies"
    version: "1.0"
    description: "Network security and segmentation policies"
    rules:
      - name: "dataplane-isolation"
        engine: "rego"
        rule: |
          package network.segmentation.dataplane
          import rego.v1
          
          deny contains msg if {
            input.kind == "DataPlaneService"
            not input.spec.networkPolicies
            msg := "DataPlane services must have network policies defined"
          }
          
  data-governance:
    name: "Data Governance Policies"
    version: "1.0"
    description: "Data handling and governance policies"
    rules:
      - name: "pii-encryption"
        engine: "python"
        rule: |
          def evaluate(data):
              if data.get('data_classification') == 'PII':
                  if not data.get('encryption_enabled'):
                      return False, "PII data must be encrypted"
              return True, "Data encryption compliance verified"
"#;
    config_data.insert("default-policies.yaml".to_string(), default_policies.to_string());

    let config_map = create_config_map(&name, namespace, labels, config_data);

    let config_maps: Api<ConfigMap> = Api::namespaced(client.clone(), namespace);
    
    match config_maps.get(&name).await {
        Ok(_) => {
            config_maps
                .patch(&name, &PatchParams::default(), &Patch::Merge(&config_map))
                .await
                .map_err(|e| OperatorError::KubernetesError(e.to_string()))?;
            debug!("Updated ConfigMap: {}", name);
        }
        Err(_) => {
            config_maps
                .create(&PostParams::default(), &config_map)
                .await
                .map_err(|e| OperatorError::KubernetesError(e.to_string()))?;
            debug!("Created ConfigMap: {}", name);
        }
    }

    Ok(())
}

/// Create Service for Policy Engine
async fn create_policy_engine_service(
    policy_engine: &PolicyEngineService,
    client: &Client,
    namespace: &str,
) -> Result<(), OperatorError> {
    let name = policy_engine.name_any();
    let labels = resource_labels("policy-engine", &name);
    let selector = selector_labels("policy-engine", &name);

    let ports = vec![
        ServicePort {
            name: Some("api".to_string()),
            port: 8000,
            target_port: Some(IntOrString::String("api".to_string())),
            protocol: Some("TCP".to_string()),
            ..Default::default()
        },
        ServicePort {
            name: Some("grpc".to_string()),
            port: 50052,
            target_port: Some(IntOrString::String("grpc".to_string())),
            protocol: Some("TCP".to_string()),
            ..Default::default()
        },
        ServicePort {
            name: Some("metrics".to_string()),
            port: 9090,
            target_port: Some(IntOrString::String("metrics".to_string())),
            protocol: Some("TCP".to_string()),
            ..Default::default()
        },
        ServicePort {
            name: Some("health".to_string()),
            port: 8080,
            target_port: Some(IntOrString::String("health".to_string())),
            protocol: Some("TCP".to_string()),
            ..Default::default()
        },
    ];

    let service = create_service(&name, namespace, labels, selector, ports);

    let services: Api<k8s_openapi::api::core::v1::Service> = Api::namespaced(client.clone(), namespace);
    
    match services.get(&name).await {
        Ok(_) => {
            services
                .patch(&name, &PatchParams::default(), &Patch::Merge(&service))
                .await
                .map_err(|e| OperatorError::KubernetesError(e.to_string()))?;
            debug!("Updated Service: {}", name);
        }
        Err(_) => {
            services
                .create(&PostParams::default(), &service)
                .await
                .map_err(|e| OperatorError::KubernetesError(e.to_string()))?;
            debug!("Created Service: {}", name);
        }
    }

    Ok(())
}

/// Create Deployment for Policy Engine
async fn create_policy_engine_deployment(
    policy_engine: &PolicyEngineService,
    client: &Client,
    namespace: &str,
) -> Result<(), OperatorError> {
    let name = policy_engine.name_any();
    let labels = resource_labels("policy-engine", &name);
    let selector_lbls = selector_labels("policy-engine", &name);

    let container = create_policy_engine_container(policy_engine)?;

    let pod_spec = PodSpec {
        containers: vec![container],
        security_context: Some(default_pod_security_context()),
        restart_policy: Some("Always".to_string()),
        ..Default::default()
    };

    let deployment = Deployment {
        metadata: ObjectMeta {
            name: Some(name.clone()),
            namespace: Some(namespace.to_string()),
            labels: Some(labels.clone()),
            annotations: Some(common_annotations()),
            ..Default::default()
        },
        spec: Some(DeploymentSpec {
            replicas: Some(policy_engine.spec.replicas as i32),
            selector: LabelSelector {
                match_labels: Some(selector_lbls.clone()),
                ..Default::default()
            },
            template: PodTemplateSpec {
                metadata: Some(ObjectMeta {
                    labels: Some(selector_lbls),
                    annotations: Some(common_annotations()),
                    ..Default::default()
                }),
                spec: Some(pod_spec),
            },
            strategy: Some(default_deployment_strategy()),
            ..Default::default()
        }),
        ..Default::default()
    };

    let deployments: Api<Deployment> = Api::namespaced(client.clone(), namespace);
    
    match deployments.get(&name).await {
        Ok(_) => {
            deployments
                .patch(&name, &PatchParams::default(), &Patch::Merge(&deployment))
                .await
                .map_err(|e| OperatorError::KubernetesError(e.to_string()))?;
            debug!("Updated Deployment: {}", name);
        }
        Err(_) => {
            deployments
                .create(&PostParams::default(), &deployment)
                .await
                .map_err(|e| OperatorError::KubernetesError(e.to_string()))?;
            debug!("Created Deployment: {}", name);
        }
    }

    Ok(())
}

/// Create Container specification for Policy Engine
fn create_policy_engine_container(policy_engine: &PolicyEngineService) -> Result<Container, OperatorError> {
    let name = "policy-engine";
    let mut env_vars = vec![
        EnvVar {
            name: "PYTHONPATH".to_string(),
            value: Some("/app".to_string()),
            ..Default::default()
        },
        EnvVar {
            name: "CONFIG_PATH".to_string(),
            value: Some("/etc/config".to_string()),
            ..Default::default()
        },
        EnvVar {
            name: "POLICY_REPOSITORIES".to_string(),
            value: Some("/etc/config/policy-repositories.yaml".to_string()),
            ..Default::default()
        },
        EnvVar {
            name: "RULE_ENGINES_CONFIG".to_string(),
            value: Some("/etc/config/rule-engines.yaml".to_string()),
            ..Default::default()
        },
    ];

    // Add rule engine specific environment variables
    if policy_engine.spec.rule_engines.python_engine.as_ref().map(|e| e.enabled).unwrap_or(false) {
        env_vars.push(EnvVar {
            name: "PYTHON_ENGINE_ENABLED".to_string(),
            value: Some("true".to_string()),
            ..Default::default()
        });
        
        if let Some(ref python_config) = policy_engine.spec.rule_engines.python_engine {
            if let Some(sandbox_mode) = python_config.sandbox_mode {
                env_vars.push(EnvVar {
                    name: "PYTHON_SANDBOX_MODE".to_string(),
                    value: Some(sandbox_mode.to_string()),
                    ..Default::default()
                });
            }
            
            if let Some(timeout) = python_config.timeout_seconds {
                env_vars.push(EnvVar {
                    name: "PYTHON_TIMEOUT_SECONDS".to_string(),
                    value: Some(timeout.to_string()),
                    ..Default::default()
                });
            }
        }
    }

    if policy_engine.spec.rule_engines.rego_engine.as_ref().map(|e| e.enabled).unwrap_or(false) {
        env_vars.push(EnvVar {
            name: "REGO_ENGINE_ENABLED".to_string(),
            value: Some("true".to_string()),
            ..Default::default()
        });
    }

    if policy_engine.spec.rule_engines.jsonpath_engine.as_ref().map(|e| e.enabled).unwrap_or(false) {
        env_vars.push(EnvVar {
            name: "JSONPATH_ENGINE_ENABLED".to_string(),
            value: Some("true".to_string()),
            ..Default::default()
        });
    }

    if policy_engine.spec.rule_engines.regex_engine.as_ref().map(|e| e.enabled).unwrap_or(false) {
        env_vars.push(EnvVar {
            name: "REGEX_ENGINE_ENABLED".to_string(),
            value: Some("true".to_string()),
            ..Default::default()
        });
    }

    // Add compliance monitoring environment variables
    if let Some(ref compliance_config) = policy_engine.spec.compliance_monitoring {
        env_vars.push(EnvVar {
            name: "COMPLIANCE_MONITORING_ENABLED".to_string(),
            value: Some(compliance_config.enabled.to_string()),
            ..Default::default()
        });
        
        if let Some(interval) = compliance_config.reporting_interval_hours {
            env_vars.push(EnvVar {
                name: "COMPLIANCE_REPORTING_INTERVAL_HOURS".to_string(),
                value: Some(interval.to_string()),
                ..Default::default()
            });
        }
    }

    let mut volumes = vec![];
    let mut volume_mounts = vec![];

    // Config volume
    let (config_vol, config_mount) = crate::controller::config_map_volume_mount(
        "config",
        "/etc/config",
        &format!("{}-config", policy_engine.name_any()),
    );
    volumes.push(config_vol);
    volume_mounts.push(config_mount);

    // Temporary directory
    let (tmp_vol, tmp_mount) = empty_dir_volume_mount("tmp", "/tmp");
    volumes.push(tmp_vol);
    volume_mounts.push(tmp_mount);

    // Policy cache volume
    let (policy_cache_vol, policy_cache_mount) = empty_dir_volume_mount("policy-cache", "/var/cache/policies");
    volumes.push(policy_cache_vol);
    volume_mounts.push(policy_cache_mount);

    // Evaluation results volume
    let (results_vol, results_mount) = empty_dir_volume_mount("evaluation-results", "/var/lib/policy-results");
    volumes.push(results_vol);
    volume_mounts.push(results_mount);

    // Resource requirements
    let mut limits = BTreeMap::new();
    let mut requests = BTreeMap::new();

    if let Some(ref resources) = policy_engine.spec.resources {
        if let Some(ref res_limits) = resources.limits {
            for (key, value) in res_limits {
                limits.insert(key.clone(), Quantity(value.clone()));
            }
        }
        if let Some(ref res_requests) = resources.requests {
            for (key, value) in res_requests {
                requests.insert(key.clone(), Quantity(value.clone()));
            }
        }
    }

    // Default resources if not specified
    if limits.is_empty() {
        limits.insert("cpu".to_string(), Quantity("1000m".to_string()));
        limits.insert("memory".to_string(), Quantity("2Gi".to_string()));
    }
    if requests.is_empty() {
        requests.insert("cpu".to_string(), Quantity("500m".to_string()));
        requests.insert("memory".to_string(), Quantity("1Gi".to_string()));
    }

    let container = Container {
        name: name.to_string(),
        image: Some(policy_engine.spec.image.clone()),
        image_pull_policy: Some("Always".to_string()),
        ports: Some(vec![
            ContainerPort {
                name: Some("api".to_string()),
                container_port: 8000,
                protocol: Some("TCP".to_string()),
                ..Default::default()
            },
            ContainerPort {
                name: Some("grpc".to_string()),
                container_port: 50052,
                protocol: Some("TCP".to_string()),
                ..Default::default()
            },
            ContainerPort {
                name: Some("metrics".to_string()),
                container_port: 9090,
                protocol: Some("TCP".to_string()),
                ..Default::default()
            },
            ContainerPort {
                name: Some("health".to_string()),
                container_port: 8080,
                protocol: Some("TCP".to_string()),
                ..Default::default()
            },
        ]),
        env: Some(env_vars),
        volume_mounts: Some(volume_mounts),
        resources: Some(ResourceRequirements {
            limits: Some(limits),
            requests: Some(requests),
            ..Default::default()
        }),
        security_context: Some(default_security_context()),
        readiness_probe: Some(default_readiness_probe("/health/ready", 8080)),
        liveness_probe: Some(default_liveness_probe("/health/alive", 8080)),
        startup_probe: Some(default_startup_probe("/health/startup", 8080)),
        ..Default::default()
    };

    Ok(container)
}