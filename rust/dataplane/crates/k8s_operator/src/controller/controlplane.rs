//! ControlPlane service controller implementation

use crate::crd::{ControlPlaneService, ControlPlaneStatus};
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

use k8s_openapi::api::apps::v1::{Deployment, DeploymentSpec, StatefulSet, StatefulSetSpec};
use k8s_openapi::api::core::v1::{
    Container, ContainerPort, PodSpec, PodTemplateSpec, ServicePort,
    ResourceRequirements, EnvVar, ConfigMap, PersistentVolumeClaimTemplate, PersistentVolumeClaimSpec,
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

/// Reconcile ControlPlaneService resources
pub async fn reconcile(
    controlplane: Arc<ControlPlaneService>,
    ctx: Arc<OperatorContext>,
) -> Result<Action, OperatorError> {
    let client = ctx.client.clone();
    let name = controlplane.name_any();
    let namespace = controlplane.namespace().unwrap_or_else(|| "default".to_string());

    info!("Reconciling ControlPlaneService: {}/{}", namespace, name);

    // Update metrics
    ctx.metrics.reconcile_counter.inc();

    let controlplane_api: Api<ControlPlaneService> = Api::namespaced(client.clone(), &namespace);

    // Create or update resources
    let result = reconcile_controlplane_resources(&controlplane, &client, &namespace).await;

    match result {
        Ok(_) => {
            info!("Successfully reconciled ControlPlaneService: {}/{}", namespace, name);
            
            // Update status
            let status = ControlPlaneStatus {
                phase: "Ready".to_string(),
                ready_replicas: controlplane.spec.replicas,
                leader: Some(format!("{}-0", name)), // Assume first replica is leader
                conditions: Some(vec![
                    crate::crd::Condition {
                        condition_type: "Ready".to_string(),
                        status: "True".to_string(),
                        reason: Some("ReconcileSuccess".to_string()),
                        message: Some("ControlPlane service is running".to_string()),
                        last_transition_time: Some(chrono::Utc::now().to_rfc3339()),
                    }
                ]),
                last_updated: Some(chrono::Utc::now().to_rfc3339()),
            };

            let patch = Patch::Merge(&json!({
                "status": status
            }));

            controlplane_api.patch_status(&name, &PatchParams::default(), &patch).await
                .map_err(|e| OperatorError::KubernetesError(e.to_string()))?;

            ctx.metrics.reconcile_success_counter.inc();
            Ok(Action::requeue(RECONCILE_INTERVAL))
        }
        Err(e) => {
            error!("Failed to reconcile ControlPlaneService {}/{}: {}", namespace, name, e);
            
            // Update status with error
            let status = ControlPlaneStatus {
                phase: "Error".to_string(),
                ready_replicas: 0,
                leader: None,
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

            let _ = controlplane_api.patch_status(&name, &PatchParams::default(), &patch).await;

            ctx.metrics.reconcile_error_counter.inc();
            Err(e)
        }
    }
}

/// Error policy for ControlPlaneService controller
pub fn error_policy(
    obj: Arc<ControlPlaneService>,
    error: &OperatorError,
    ctx: Arc<OperatorContext>,
) -> Action {
    default_error_policy(obj, error, ctx)
}

/// Reconcile all ControlPlane resources
async fn reconcile_controlplane_resources(
    controlplane: &ControlPlaneService,
    client: &Client,
    namespace: &str,
) -> Result<(), OperatorError> {
    let name = &controlplane.name_any();

    // Create ConfigMap for configuration
    create_controlplane_config_map(controlplane, client, namespace).await?;

    // Create Service
    create_controlplane_service(controlplane, client, namespace).await?;

    // Create headless service for StatefulSet
    create_controlplane_headless_service(controlplane, client, namespace).await?;

    // Create StatefulSet for high availability or Deployment for simple setup
    if should_use_statefulset(controlplane) {
        create_controlplane_statefulset(controlplane, client, namespace).await?;
    } else {
        create_controlplane_deployment(controlplane, client, namespace).await?;
    }

    Ok(())
}

/// Determine if StatefulSet should be used for high availability
fn should_use_statefulset(controlplane: &ControlPlaneService) -> bool {
    controlplane.spec.high_availability.as_ref()
        .map(|ha| ha.enabled)
        .unwrap_or(false) || 
    controlplane.spec.replicas > 1
}

/// Create ConfigMap for ControlPlane configuration
async fn create_controlplane_config_map(
    controlplane: &ControlPlaneService,
    client: &Client,
    namespace: &str,
) -> Result<(), OperatorError> {
    let name = format!("{}-config", controlplane.name_any());
    let labels = resource_labels("controlplane", &controlplane.name_any());

    // Generate configuration data
    let mut config_data = BTreeMap::new();
    
    // Configuration service settings
    let config_service_config = serde_yaml::to_string(&controlplane.spec.config_service)
        .map_err(|e| OperatorError::SerializationError(e.to_string()))?;
    config_data.insert("config-service.yaml".to_string(), config_service_config);

    // Policy engine settings
    let policy_engine_config = serde_yaml::to_string(&controlplane.spec.policy_engine)
        .map_err(|e| OperatorError::SerializationError(e.to_string()))?;
    config_data.insert("policy-engine.yaml".to_string(), policy_engine_config);

    // Management API configuration
    let management_api_config = serde_yaml::to_string(&controlplane.spec.management_api)
        .map_err(|e| OperatorError::SerializationError(e.to_string()))?;
    config_data.insert("management-api.yaml".to_string(), management_api_config);

    // Database configuration (if specified)
    if let Some(ref db_config) = controlplane.spec.database {
        let db_config_str = serde_yaml::to_string(db_config)
            .map_err(|e| OperatorError::SerializationError(e.to_string()))?;
        config_data.insert("database.yaml".to_string(), db_config_str);
    }

    // High availability configuration (if specified)
    if let Some(ref ha_config) = controlplane.spec.high_availability {
        let ha_config_str = serde_yaml::to_string(ha_config)
            .map_err(|e| OperatorError::SerializationError(e.to_string()))?;
        config_data.insert("high-availability.yaml".to_string(), ha_config_str);
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

/// Create Service for ControlPlane
async fn create_controlplane_service(
    controlplane: &ControlPlaneService,
    client: &Client,
    namespace: &str,
) -> Result<(), OperatorError> {
    let name = controlplane.name_any();
    let labels = resource_labels("controlplane", &name);
    let selector = selector_labels("controlplane", &name);

    let mut ports = vec![
        ServicePort {
            name: Some("api".to_string()),
            port: controlplane.spec.management_api.port as i32,
            target_port: Some(IntOrString::String("api".to_string())),
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

    // Add gRPC port if needed
    ports.push(ServicePort {
        name: Some("grpc".to_string()),
        port: 50051,
        target_port: Some(IntOrString::String("grpc".to_string())),
        protocol: Some("TCP".to_string()),
        ..Default::default()
    });

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

/// Create headless Service for StatefulSet
async fn create_controlplane_headless_service(
    controlplane: &ControlPlaneService,
    client: &Client,
    namespace: &str,
) -> Result<(), OperatorError> {
    let name = format!("{}-headless", controlplane.name_any());
    let labels = resource_labels("controlplane", &controlplane.name_any());
    let selector = selector_labels("controlplane", &controlplane.name_any());

    let ports = vec![
        ServicePort {
            name: Some("peer".to_string()),
            port: 9443,
            target_port: Some(IntOrString::String("peer".to_string())),
            protocol: Some("TCP".to_string()),
            ..Default::default()
        },
    ];

    let mut service = create_service(&name, namespace, labels, selector, ports);
    
    // Make it headless
    if let Some(ref mut spec) = service.spec {
        spec.cluster_ip = Some("None".to_string());
    }

    let services: Api<k8s_openapi::api::core::v1::Service> = Api::namespaced(client.clone(), namespace);
    
    match services.get(&name).await {
        Ok(_) => {
            services
                .patch(&name, &PatchParams::default(), &Patch::Merge(&service))
                .await
                .map_err(|e| OperatorError::KubernetesError(e.to_string()))?;
            debug!("Updated headless Service: {}", name);
        }
        Err(_) => {
            services
                .create(&PostParams::default(), &service)
                .await
                .map_err(|e| OperatorError::KubernetesError(e.to_string()))?;
            debug!("Created headless Service: {}", name);
        }
    }

    Ok(())
}

/// Create StatefulSet for ControlPlane with high availability
async fn create_controlplane_statefulset(
    controlplane: &ControlPlaneService,
    client: &Client,
    namespace: &str,
) -> Result<(), OperatorError> {
    let name = controlplane.name_any();
    let labels = resource_labels("controlplane", &name);
    let selector_lbls = selector_labels("controlplane", &name);

    let container = create_controlplane_container(controlplane, true)?;

    let pod_spec = PodSpec {
        containers: vec![container],
        security_context: Some(default_pod_security_context()),
        restart_policy: Some("Always".to_string()),
        ..Default::default()
    };

    // PVC template for persistent storage
    let pvc_template = PersistentVolumeClaimTemplate {
        metadata: ObjectMeta {
            name: Some("data".to_string()),
            labels: Some(labels.clone()),
            ..Default::default()
        },
        spec: PersistentVolumeClaimSpec {
            access_modes: Some(vec!["ReadWriteOnce".to_string()]),
            resources: Some(k8s_openapi::api::core::v1::ResourceRequirements {
                requests: Some({
                    let mut requests = BTreeMap::new();
                    requests.insert("storage".to_string(), Quantity("10Gi".to_string()));
                    requests
                }),
                ..Default::default()
            }),
            storage_class_name: Some("standard".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };

    let statefulset = StatefulSet {
        metadata: ObjectMeta {
            name: Some(name.clone()),
            namespace: Some(namespace.to_string()),
            labels: Some(labels.clone()),
            annotations: Some(common_annotations()),
            ..Default::default()
        },
        spec: Some(StatefulSetSpec {
            replicas: Some(controlplane.spec.replicas as i32),
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
            service_name: format!("{}-headless", name),
            volume_claim_templates: Some(vec![pvc_template]),
            update_strategy: Some(k8s_openapi::api::apps::v1::StatefulSetUpdateStrategy {
                type_: Some("RollingUpdate".to_string()),
                rolling_update: Some(k8s_openapi::api::apps::v1::RollingUpdateStatefulSetStrategy {
                    max_unavailable: Some(IntOrString::Int(1)),
                    partition: Some(0),
                }),
            }),
            ..Default::default()
        }),
        ..Default::default()
    };

    let statefulsets: Api<StatefulSet> = Api::namespaced(client.clone(), namespace);
    
    match statefulsets.get(&name).await {
        Ok(_) => {
            statefulsets
                .patch(&name, &PatchParams::default(), &Patch::Merge(&statefulset))
                .await
                .map_err(|e| OperatorError::KubernetesError(e.to_string()))?;
            debug!("Updated StatefulSet: {}", name);
        }
        Err(_) => {
            statefulsets
                .create(&PostParams::default(), &statefulset)
                .await
                .map_err(|e| OperatorError::KubernetesError(e.to_string()))?;
            debug!("Created StatefulSet: {}", name);
        }
    }

    Ok(())
}

/// Create Deployment for ControlPlane (simple setup)
async fn create_controlplane_deployment(
    controlplane: &ControlPlaneService,
    client: &Client,
    namespace: &str,
) -> Result<(), OperatorError> {
    let name = controlplane.name_any();
    let labels = resource_labels("controlplane", &name);
    let selector_lbls = selector_labels("controlplane", &name);

    let container = create_controlplane_container(controlplane, false)?;

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
            replicas: Some(controlplane.spec.replicas as i32),
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

/// Create Container specification for ControlPlane
fn create_controlplane_container(controlplane: &ControlPlaneService, is_stateful: bool) -> Result<Container, OperatorError> {
    let name = "controlplane";
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
        EnvVar {
            name: "API_PORT".to_string(),
            value: Some(controlplane.spec.management_api.port.to_string()),
            ..Default::default()
        },
        EnvVar {
            name: "TLS_ENABLED".to_string(),
            value: Some(controlplane.spec.management_api.tls_enabled.to_string()),
            ..Default::default()
        },
    ];

    // Add database connection environment variables
    if let Some(ref db_config) = controlplane.spec.database {
        env_vars.extend_from_slice(&[
            env_from_secret(&db_config.credentials_secret, "username", "DB_USERNAME"),
            env_from_secret(&db_config.credentials_secret, "password", "DB_PASSWORD"),
            EnvVar {
                name: "DB_HOST".to_string(),
                value: Some(db_config.host.clone()),
                ..Default::default()
            },
            EnvVar {
                name: "DB_PORT".to_string(),
                value: Some(db_config.port.to_string()),
                ..Default::default()
            },
            EnvVar {
                name: "DB_NAME".to_string(),
                value: Some(db_config.database.clone()),
                ..Default::default()
            },
        ]);
    }

    // Add HA-specific environment variables
    if is_stateful {
        env_vars.extend_from_slice(&[
            EnvVar {
                name: "CLUSTER_MODE".to_string(),
                value: Some("true".to_string()),
                ..Default::default()
            },
            EnvVar {
                name: "POD_NAME".to_string(),
                value_from: Some(k8s_openapi::api::core::v1::EnvVarSource {
                    field_ref: Some(k8s_openapi::api::core::v1::ObjectFieldSelector {
                        field_path: "metadata.name".to_string(),
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            },
            EnvVar {
                name: "POD_NAMESPACE".to_string(),
                value_from: Some(k8s_openapi::api::core::v1::EnvVarSource {
                    field_ref: Some(k8s_openapi::api::core::v1::ObjectFieldSelector {
                        field_path: "metadata.namespace".to_string(),
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            },
        ]);
    }

    let mut volumes = vec![];
    let mut volume_mounts = vec![];

    // Config volume
    let (config_vol, config_mount) = crate::controller::config_map_volume_mount(
        "config",
        "/etc/config",
        &format!("{}-config", controlplane.name_any()),
    );
    volumes.push(config_vol);
    volume_mounts.push(config_mount);

    // Temporary directory
    let (tmp_vol, tmp_mount) = empty_dir_volume_mount("tmp", "/tmp");
    volumes.push(tmp_vol);
    volume_mounts.push(tmp_mount);

    // Data volume for StatefulSet
    if is_stateful {
        volume_mounts.push(k8s_openapi::api::core::v1::VolumeMount {
            name: "data".to_string(),
            mount_path: "/var/lib/cronos-ai".to_string(),
            read_only: Some(false),
            ..Default::default()
        });
    }

    // Resource requirements
    let mut limits = BTreeMap::new();
    let mut requests = BTreeMap::new();

    if let Some(ref resources) = controlplane.spec.resources {
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
        image: Some(controlplane.spec.image.clone()),
        image_pull_policy: Some("Always".to_string()),
        ports: Some(vec![
            ContainerPort {
                name: Some("api".to_string()),
                container_port: controlplane.spec.management_api.port as i32,
                protocol: Some("TCP".to_string()),
                ..Default::default()
            },
            ContainerPort {
                name: Some("grpc".to_string()),
                container_port: 50051,
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
            ContainerPort {
                name: Some("peer".to_string()),
                container_port: 9443,
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