//! DataPlane service controller implementation

use crate::crd::{DataPlaneService, DataPlaneStatus};
use crate::error::OperatorError;
use crate::OperatorContext;
use crate::controller::{
    common_labels, resource_labels, selector_labels, common_annotations,
    needs_update, privileged_security_context, privileged_pod_security_context,
    recreate_deployment_strategy, default_readiness_probe, default_liveness_probe,
    default_startup_probe, hugepages_volume_mount, empty_dir_volume_mount,
    create_service, create_config_map, env_from_secret,
    RECONCILE_INTERVAL, default_error_policy,
};

use k8s_openapi::api::apps::v1::{Deployment, DeploymentSpec, DaemonSet, DaemonSetSpec};
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

/// Reconcile DataPlaneService resources
pub async fn reconcile(
    dataplane: Arc<DataPlaneService>,
    ctx: Arc<OperatorContext>,
) -> Result<Action, OperatorError> {
    let client = ctx.client.clone();
    let name = dataplane.name_any();
    let namespace = dataplane.namespace().unwrap_or_else(|| "default".to_string());

    info!("Reconciling DataPlaneService: {}/{}", namespace, name);

    // Update metrics
    ctx.metrics.reconcile_counter.inc();

    let dataplane_api: Api<DataPlaneService> = Api::namespaced(client.clone(), &namespace);

    // Create or update resources
    let result = reconcile_dataplane_resources(&dataplane, &client, &namespace).await;

    match result {
        Ok(_) => {
            info!("Successfully reconciled DataPlaneService: {}/{}", namespace, name);
            
            // Update status
            let status = DataPlaneStatus {
                phase: "Ready".to_string(),
                ready_replicas: dataplane.spec.replicas,
                available_replicas: dataplane.spec.replicas,
                conditions: Some(vec![
                    crate::crd::Condition {
                        condition_type: "Ready".to_string(),
                        status: "True".to_string(),
                        reason: Some("ReconcileSuccess".to_string()),
                        message: Some("DataPlane service is running".to_string()),
                        last_transition_time: Some(chrono::Utc::now().to_rfc3339()),
                    }
                ]),
                performance_metrics: None, // Will be populated by monitoring
                last_updated: Some(chrono::Utc::now().to_rfc3339()),
            };

            let patch = Patch::Merge(&json!({
                "status": status
            }));

            dataplane_api.patch_status(&name, &PatchParams::default(), &patch).await
                .map_err(|e| OperatorError::KubernetesError(e.to_string()))?;

            ctx.metrics.reconcile_success_counter.inc();
            Ok(Action::requeue(RECONCILE_INTERVAL))
        }
        Err(e) => {
            error!("Failed to reconcile DataPlaneService {}/{}: {}", namespace, name, e);
            
            // Update status with error
            let status = DataPlaneStatus {
                phase: "Error".to_string(),
                ready_replicas: 0,
                available_replicas: 0,
                conditions: Some(vec![
                    crate::crd::Condition {
                        condition_type: "Ready".to_string(),
                        status: "False".to_string(),
                        reason: Some("ReconcileError".to_string()),
                        message: Some(format!("Error: {}", e)),
                        last_transition_time: Some(chrono::Utc::now().to_rfc3339()),
                    }
                ]),
                performance_metrics: None,
                last_updated: Some(chrono::Utc::now().to_rfc3339()),
            };

            let patch = Patch::Merge(&json!({
                "status": status
            }));

            let _ = dataplane_api.patch_status(&name, &PatchParams::default(), &patch).await;

            ctx.metrics.reconcile_error_counter.inc();
            Err(e)
        }
    }
}

/// Error policy for DataPlaneService controller
pub fn error_policy(
    obj: Arc<DataPlaneService>,
    error: &OperatorError,
    ctx: Arc<OperatorContext>,
) -> Action {
    default_error_policy(obj, error, ctx)
}

/// Reconcile all DataPlane resources
async fn reconcile_dataplane_resources(
    dataplane: &DataPlaneService,
    client: &Client,
    namespace: &str,
) -> Result<(), OperatorError> {
    let name = &dataplane.name_any();

    // Create ConfigMap for configuration
    create_dataplane_config_map(dataplane, client, namespace).await?;

    // Create Service
    create_dataplane_service(dataplane, client, namespace).await?;

    // Create DaemonSet (preferred for dataplane) or Deployment
    if should_use_daemonset(dataplane) {
        create_dataplane_daemonset(dataplane, client, namespace).await?;
    } else {
        create_dataplane_deployment(dataplane, client, namespace).await?;
    }

    Ok(())
}

/// Determine if DaemonSet should be used based on configuration
fn should_use_daemonset(dataplane: &DataPlaneService) -> bool {
    // Use DaemonSet if:
    // 1. DPDK is enabled (usually requires dedicated hardware)
    // 2. Specific network interfaces are configured
    // 3. Node-specific configuration is required
    dataplane.spec.dpdk_config.enabled || 
    !dataplane.spec.network_interfaces.is_empty() ||
    dataplane.spec.node_selector.is_some()
}

/// Create ConfigMap for DataPlane configuration
async fn create_dataplane_config_map(
    dataplane: &DataPlaneService,
    client: &Client,
    namespace: &str,
) -> Result<(), OperatorError> {
    let name = format!("{}-config", dataplane.name_any());
    let labels = resource_labels("dataplane", &dataplane.name_any());

    // Generate configuration data
    let mut config_data = BTreeMap::new();
    
    // DPDK configuration
    if dataplane.spec.dpdk_config.enabled {
        let dpdk_config = serde_yaml::to_string(&dataplane.spec.dpdk_config)
            .map_err(|e| OperatorError::SerializationError(e.to_string()))?;
        config_data.insert("dpdk.yaml".to_string(), dpdk_config);
    }

    // DPI configuration
    if dataplane.spec.dpi_config.enabled {
        let dpi_config = serde_yaml::to_string(&dataplane.spec.dpi_config)
            .map_err(|e| OperatorError::SerializationError(e.to_string()))?;
        config_data.insert("dpi.yaml".to_string(), dpi_config);
    }

    // High-frequency metrics configuration
    if let Some(ref metrics_config) = dataplane.spec.metrics_config {
        let metrics_config_str = serde_yaml::to_string(metrics_config)
            .map_err(|e| OperatorError::SerializationError(e.to_string()))?;
        config_data.insert("metrics.yaml".to_string(), metrics_config_str);
    }

    // Network interfaces configuration
    if !dataplane.spec.network_interfaces.is_empty() {
        let interfaces_config = serde_yaml::to_string(&dataplane.spec.network_interfaces)
            .map_err(|e| OperatorError::SerializationError(e.to_string()))?;
        config_data.insert("interfaces.yaml".to_string(), interfaces_config);
    }

    // Security configuration
    if let Some(ref security_config) = dataplane.spec.security {
        let security_config_str = serde_yaml::to_string(security_config)
            .map_err(|e| OperatorError::SerializationError(e.to_string()))?;
        config_data.insert("security.yaml".to_string(), security_config_str);
    }

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

/// Create Service for DataPlane
async fn create_dataplane_service(
    dataplane: &DataPlaneService,
    client: &Client,
    namespace: &str,
) -> Result<(), OperatorError> {
    let name = dataplane.name_any();
    let labels = resource_labels("dataplane", &name);
    let selector = selector_labels("dataplane", &name);

    let ports = vec![
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

/// Create DaemonSet for DataPlane
async fn create_dataplane_daemonset(
    dataplane: &DataPlaneService,
    client: &Client,
    namespace: &str,
) -> Result<(), OperatorError> {
    let name = dataplane.name_any();
    let labels = resource_labels("dataplane", &name);
    let selector_lbls = selector_labels("dataplane", &name);

    let container = create_dataplane_container(dataplane)?;

    let pod_spec = PodSpec {
        containers: vec![container],
        security_context: Some(privileged_pod_security_context()),
        node_selector: dataplane.spec.node_selector.clone(),
        tolerations: dataplane.spec.tolerations.clone().map(|t| 
            t.into_iter().map(|tol| k8s_openapi::api::core::v1::Toleration {
                key: tol.key,
                operator: tol.operator,
                value: tol.value,
                effect: tol.effect,
                ..Default::default()
            }).collect()
        ),
        affinity: convert_affinity(&dataplane.spec.affinity),
        host_network: Some(true), // For DPDK applications
        dns_policy: Some("ClusterFirstWithHostNet".to_string()),
        restart_policy: Some("Always".to_string()),
        ..Default::default()
    };

    let daemonset = DaemonSet {
        metadata: ObjectMeta {
            name: Some(name.clone()),
            namespace: Some(namespace.to_string()),
            labels: Some(labels.clone()),
            annotations: Some(common_annotations()),
            ..Default::default()
        },
        spec: Some(DaemonSetSpec {
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
            update_strategy: Some(k8s_openapi::api::apps::v1::DaemonSetUpdateStrategy {
                type_: Some("RollingUpdate".to_string()),
                rolling_update: Some(k8s_openapi::api::apps::v1::RollingUpdateDaemonSet {
                    max_surge: Some(IntOrString::String("10%".to_string())),
                    max_unavailable: Some(IntOrString::String("10%".to_string())),
                }),
            }),
            ..Default::default()
        }),
        ..Default::default()
    };

    let daemonsets: Api<DaemonSet> = Api::namespaced(client.clone(), namespace);
    
    match daemonsets.get(&name).await {
        Ok(_) => {
            daemonsets
                .patch(&name, &PatchParams::default(), &Patch::Merge(&daemonset))
                .await
                .map_err(|e| OperatorError::KubernetesError(e.to_string()))?;
            debug!("Updated DaemonSet: {}", name);
        }
        Err(_) => {
            daemonsets
                .create(&PostParams::default(), &daemonset)
                .await
                .map_err(|e| OperatorError::KubernetesError(e.to_string()))?;
            debug!("Created DaemonSet: {}", name);
        }
    }

    Ok(())
}

/// Create Deployment for DataPlane
async fn create_dataplane_deployment(
    dataplane: &DataPlaneService,
    client: &Client,
    namespace: &str,
) -> Result<(), OperatorError> {
    let name = dataplane.name_any();
    let labels = resource_labels("dataplane", &name);
    let selector_lbls = selector_labels("dataplane", &name);

    let container = create_dataplane_container(dataplane)?;

    let pod_spec = PodSpec {
        containers: vec![container],
        security_context: Some(privileged_pod_security_context()),
        node_selector: dataplane.spec.node_selector.clone(),
        tolerations: dataplane.spec.tolerations.clone().map(|t| 
            t.into_iter().map(|tol| k8s_openapi::api::core::v1::Toleration {
                key: tol.key,
                operator: tol.operator,
                value: tol.value,
                effect: tol.effect,
                ..Default::default()
            }).collect()
        ),
        affinity: convert_affinity(&dataplane.spec.affinity),
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
            replicas: Some(dataplane.spec.replicas as i32),
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
            strategy: Some(recreate_deployment_strategy()), // Recreate for stateful dataplane
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

/// Create Container specification for DataPlane
fn create_dataplane_container(dataplane: &DataPlaneService) -> Result<Container, OperatorError> {
    let name = "dataplane";
    let mut env_vars = vec![
        EnvVar {
            name: "RUST_LOG".to_string(),
            value: Some("info".to_string()),
            ..Default::default()
        },
        EnvVar {
            name: "CONFIG_PATH".to_string(),
            value: Some("/etc/config".to_string()),
            ..Default::default()
        },
    ];

    // Add environment variables based on configuration
    if dataplane.spec.dpdk_config.enabled {
        env_vars.push(EnvVar {
            name: "DPDK_ENABLED".to_string(),
            value: Some("true".to_string()),
            ..Default::default()
        });
    }

    if dataplane.spec.dpi_config.enabled {
        env_vars.push(EnvVar {
            name: "DPI_ENABLED".to_string(),
            value: Some("true".to_string()),
            ..Default::default()
        });
    }

    let mut volumes = vec![];
    let mut volume_mounts = vec![];

    // Config volume
    let (config_vol, config_mount) = crate::controller::config_map_volume_mount(
        "config",
        "/etc/config",
        &format!("{}-config", dataplane.name_any()),
    );
    volumes.push(config_vol);
    volume_mounts.push(config_mount);

    // Temporary directory
    let (tmp_vol, tmp_mount) = empty_dir_volume_mount("tmp", "/tmp");
    volumes.push(tmp_vol);
    volume_mounts.push(tmp_mount);

    // Hugepages volume for DPDK
    if dataplane.spec.dpdk_config.enabled {
        let huge_pages_size = dataplane.spec.dpdk_config.huge_pages
            .as_ref()
            .unwrap_or(&"1Gi".to_string())
            .clone();
        let (hugepages_vol, hugepages_mount) = hugepages_volume_mount(
            "hugepages",
            "/dev/hugepages",
            &huge_pages_size,
        );
        volumes.push(hugepages_vol);
        volume_mounts.push(hugepages_mount);
    }

    // Resource requirements
    let mut limits = BTreeMap::new();
    let mut requests = BTreeMap::new();

    if let Some(ref resources) = dataplane.spec.resources {
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
        limits.insert("cpu".to_string(), Quantity("2000m".to_string()));
        limits.insert("memory".to_string(), Quantity("4Gi".to_string()));
    }
    if requests.is_empty() {
        requests.insert("cpu".to_string(), Quantity("1000m".to_string()));
        requests.insert("memory".to_string(), Quantity("2Gi".to_string()));
    }

    // Add hugepages resources if DPDK is enabled
    if dataplane.spec.dpdk_config.enabled {
        let huge_pages_size = dataplane.spec.dpdk_config.huge_pages
            .as_ref()
            .unwrap_or(&"1Gi".to_string());
        limits.insert("hugepages-2Mi".to_string(), Quantity(huge_pages_size.clone()));
        requests.insert("hugepages-2Mi".to_string(), Quantity(huge_pages_size.clone()));
    }

    let container = Container {
        name: name.to_string(),
        image: Some(dataplane.spec.image.clone()),
        image_pull_policy: dataplane.spec.image_pull_policy.clone(),
        ports: Some(vec![
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
        security_context: Some(privileged_security_context()),
        readiness_probe: Some(default_readiness_probe("/health/ready", 8080)),
        liveness_probe: Some(default_liveness_probe("/health/alive", 8080)),
        startup_probe: Some(default_startup_probe("/health/startup", 8080)),
        ..Default::default()
    };

    Ok(container)
}

/// Convert custom Affinity to Kubernetes Affinity
fn convert_affinity(affinity: &Option<crate::crd::Affinity>) -> Option<k8s_openapi::api::core::v1::Affinity> {
    // This is a simplified conversion - in production, you'd want more comprehensive mapping
    affinity.as_ref().map(|_| k8s_openapi::api::core::v1::Affinity {
        node_affinity: None,
        pod_affinity: None,
        pod_anti_affinity: None,
    })
}