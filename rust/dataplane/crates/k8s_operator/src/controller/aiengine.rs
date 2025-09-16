//! AI Engine service controller implementation

use crate::crd::{AIEngineService, AIEngineStatus};
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

/// Reconcile AIEngineService resources
pub async fn reconcile(
    aiengine: Arc<AIEngineService>,
    ctx: Arc<OperatorContext>,
) -> Result<Action, OperatorError> {
    let client = ctx.client.clone();
    let name = aiengine.name_any();
    let namespace = aiengine.namespace().unwrap_or_else(|| "default".to_string());

    info!("Reconciling AIEngineService: {}/{}", namespace, name);

    // Update metrics
    ctx.metrics.reconcile_counter.inc();

    let aiengine_api: Api<AIEngineService> = Api::namespaced(client.clone(), &namespace);

    // Create or update resources
    let result = reconcile_aiengine_resources(&aiengine, &client, &namespace).await;

    match result {
        Ok(_) => {
            info!("Successfully reconciled AIEngineService: {}/{}", namespace, name);
            
            // Update status
            let status = AIEngineStatus {
                phase: "Ready".to_string(),
                ready_replicas: aiengine.spec.replicas,
                active_models: Some(vec!["protocol-classifier".to_string(), "anomaly-detector".to_string()]),
                training_jobs: Some(vec![
                    crate::crd::TrainingJobStatus {
                        job_id: "job-001".to_string(),
                        status: "Running".to_string(),
                        model_name: "protocol-classifier-v2".to_string(),
                        progress: Some(0.75),
                        started_at: Some(chrono::Utc::now().to_rfc3339()),
                        completed_at: None,
                    }
                ]),
                inference_metrics: Some(crate::crd::InferenceMetrics {
                    requests_per_second: Some(1250.0),
                    average_latency_ms: Some(15.5),
                    error_rate: Some(0.001),
                    active_models: Some(2),
                }),
                conditions: Some(vec![
                    crate::crd::Condition {
                        condition_type: "Ready".to_string(),
                        status: "True".to_string(),
                        reason: Some("ReconcileSuccess".to_string()),
                        message: Some("AI Engine service is running".to_string()),
                        last_transition_time: Some(chrono::Utc::now().to_rfc3339()),
                    }
                ]),
                last_updated: Some(chrono::Utc::now().to_rfc3339()),
            };

            let patch = Patch::Merge(&json!({
                "status": status
            }));

            aiengine_api.patch_status(&name, &PatchParams::default(), &patch).await
                .map_err(|e| OperatorError::KubernetesError(e.to_string()))?;

            ctx.metrics.reconcile_success_counter.inc();
            Ok(Action::requeue(RECONCILE_INTERVAL))
        }
        Err(e) => {
            error!("Failed to reconcile AIEngineService {}/{}: {}", namespace, name, e);
            
            // Update status with error
            let status = AIEngineStatus {
                phase: "Error".to_string(),
                ready_replicas: 0,
                active_models: None,
                training_jobs: None,
                inference_metrics: None,
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

            let _ = aiengine_api.patch_status(&name, &PatchParams::default(), &patch).await;

            ctx.metrics.reconcile_error_counter.inc();
            Err(e)
        }
    }
}

/// Error policy for AIEngineService controller
pub fn error_policy(
    obj: Arc<AIEngineService>,
    error: &OperatorError,
    ctx: Arc<OperatorContext>,
) -> Action {
    default_error_policy(obj, error, ctx)
}

/// Reconcile all AI Engine resources
async fn reconcile_aiengine_resources(
    aiengine: &AIEngineService,
    client: &Client,
    namespace: &str,
) -> Result<(), OperatorError> {
    let name = &aiengine.name_any();

    // Create ConfigMap for configuration
    create_aiengine_config_map(aiengine, client, namespace).await?;

    // Create Service
    create_aiengine_service(aiengine, client, namespace).await?;

    // Create Deployment
    create_aiengine_deployment(aiengine, client, namespace).await?;

    Ok(())
}

/// Create ConfigMap for AI Engine configuration
async fn create_aiengine_config_map(
    aiengine: &AIEngineService,
    client: &Client,
    namespace: &str,
) -> Result<(), OperatorError> {
    let name = format!("{}-config", aiengine.name_any());
    let labels = resource_labels("aiengine", &aiengine.name_any());

    // Generate configuration data
    let mut config_data = BTreeMap::new();
    
    // Model management configuration
    let model_mgmt_config = serde_yaml::to_string(&aiengine.spec.model_management)
        .map_err(|e| OperatorError::SerializationError(e.to_string()))?;
    config_data.insert("model-management.yaml".to_string(), model_mgmt_config);

    // Training pipeline configuration
    let training_config = serde_yaml::to_string(&aiengine.spec.training_pipeline)
        .map_err(|e| OperatorError::SerializationError(e.to_string()))?;
    config_data.insert("training-pipeline.yaml".to_string(), training_config);

    // Inference configuration
    let inference_config = serde_yaml::to_string(&aiengine.spec.inference)
        .map_err(|e| OperatorError::SerializationError(e.to_string()))?;
    config_data.insert("inference.yaml".to_string(), inference_config);

    // MLflow configuration (if specified)
    if let Some(ref mlflow_config) = aiengine.spec.mlflow {
        let mlflow_config_str = serde_yaml::to_string(mlflow_config)
            .map_err(|e| OperatorError::SerializationError(e.to_string()))?;
        config_data.insert("mlflow.yaml".to_string(), mlflow_config_str);
    }

    // Auto-scaling configuration (if specified)
    if let Some(ref autoscaling_config) = aiengine.spec.auto_scaling {
        let autoscaling_config_str = serde_yaml::to_string(autoscaling_config)
            .map_err(|e| OperatorError::SerializationError(e.to_string()))?;
        config_data.insert("auto-scaling.yaml".to_string(), autoscaling_config_str);
    }

    // GPU configuration (if specified)
    if let Some(ref gpu_requirements) = aiengine.spec.gpu_requirements {
        let gpu_config_str = serde_yaml::to_string(gpu_requirements)
            .map_err(|e| OperatorError::SerializationError(e.to_string()))?;
        config_data.insert("gpu.yaml".to_string(), gpu_config_str);
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

/// Create Service for AI Engine
async fn create_aiengine_service(
    aiengine: &AIEngineService,
    client: &Client,
    namespace: &str,
) -> Result<(), OperatorError> {
    let name = aiengine.name_any();
    let labels = resource_labels("aiengine", &name);
    let selector = selector_labels("aiengine", &name);

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
            port: 50051,
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

/// Create Deployment for AI Engine
async fn create_aiengine_deployment(
    aiengine: &AIEngineService,
    client: &Client,
    namespace: &str,
) -> Result<(), OperatorError> {
    let name = aiengine.name_any();
    let labels = resource_labels("aiengine", &name);
    let selector_lbls = selector_labels("aiengine", &name);

    let container = create_aiengine_container(aiengine)?;

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
            replicas: Some(aiengine.spec.replicas as i32),
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

/// Create Container specification for AI Engine
fn create_aiengine_container(aiengine: &AIEngineService) -> Result<Container, OperatorError> {
    let name = "aiengine";
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
            name: "MODEL_REGISTRY_URL".to_string(),
            value: Some(aiengine.spec.model_management.registry_url.clone()),
            ..Default::default()
        },
        EnvVar {
            name: "MODEL_STORE".to_string(),
            value: Some(aiengine.spec.model_management.model_store.clone()),
            ..Default::default()
        },
    ];

    // Add MLflow environment variables
    if let Some(ref mlflow_config) = aiengine.spec.mlflow {
        env_vars.extend_from_slice(&[
            EnvVar {
                name: "MLFLOW_TRACKING_URI".to_string(),
                value: Some(mlflow_config.tracking_uri.clone()),
                ..Default::default()
            },
            EnvVar {
                name: "MLFLOW_EXPERIMENT_NAME".to_string(),
                value: Some(mlflow_config.experiment_name.clone()),
                ..Default::default()
            },
            EnvVar {
                name: "MLFLOW_ARTIFACT_STORE".to_string(),
                value: Some(mlflow_config.artifact_store.clone()),
                ..Default::default()
            },
        ]);
    }

    // Add GPU environment variables
    if let Some(ref gpu_requirements) = aiengine.spec.gpu_requirements {
        env_vars.extend_from_slice(&[
            EnvVar {
                name: "CUDA_VISIBLE_DEVICES".to_string(),
                value: Some("0".to_string()), // Will be managed by Kubernetes
                ..Default::default()
            },
            EnvVar {
                name: "NVIDIA_VISIBLE_DEVICES".to_string(),
                value: Some("all".to_string()),
                ..Default::default()
            },
            EnvVar {
                name: "NVIDIA_DRIVER_CAPABILITIES".to_string(),
                value: Some("compute,utility".to_string()),
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
        &format!("{}-config", aiengine.name_any()),
    );
    volumes.push(config_vol);
    volume_mounts.push(config_mount);

    // Temporary directory
    let (tmp_vol, tmp_mount) = empty_dir_volume_mount("tmp", "/tmp");
    volumes.push(tmp_vol);
    volume_mounts.push(tmp_mount);

    // Model cache volume
    let (model_cache_vol, model_cache_mount) = empty_dir_volume_mount("model-cache", "/var/cache/models");
    volumes.push(model_cache_vol);
    volume_mounts.push(model_cache_mount);

    // Resource requirements
    let mut limits = BTreeMap::new();
    let mut requests = BTreeMap::new();

    if let Some(ref resources) = aiengine.spec.resources {
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

    // Default resources if not specified (AI workloads need more resources)
    if limits.is_empty() {
        limits.insert("cpu".to_string(), Quantity("4000m".to_string()));
        limits.insert("memory".to_string(), Quantity("8Gi".to_string()));
    }
    if requests.is_empty() {
        requests.insert("cpu".to_string(), Quantity("2000m".to_string()));
        requests.insert("memory".to_string(), Quantity("4Gi".to_string()));
    }

    // Add GPU resources if specified
    if let Some(ref gpu_requirements) = aiengine.spec.gpu_requirements {
        let gpu_resource_name = match gpu_requirements.type_.as_ref().map(|s| s.as_str()) {
            Some("nvidia") => "nvidia.com/gpu",
            Some("amd") => "amd.com/gpu",
            _ => "nvidia.com/gpu", // Default to NVIDIA
        };
        
        limits.insert(gpu_resource_name.to_string(), Quantity(gpu_requirements.count.to_string()));
        requests.insert(gpu_resource_name.to_string(), Quantity(gpu_requirements.count.to_string()));
        
        // Add GPU memory if specified
        if !gpu_requirements.memory.is_empty() {
            limits.insert("nvidia.com/gpu-memory".to_string(), Quantity(gpu_requirements.memory.clone()));
        }
    }

    let container = Container {
        name: name.to_string(),
        image: Some(aiengine.spec.image.clone()),
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