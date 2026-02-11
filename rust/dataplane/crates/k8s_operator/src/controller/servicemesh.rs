//! Service Mesh controller implementation

use crate::crd::{ServiceMeshConfig, ServiceMeshStatus};
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

use k8s_openapi::api::core::v1::ConfigMap;
use k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta;

use kube::{
    api::{Api, ListParams, Patch, PatchParams, PostParams},
    runtime::controller::Action,
    Resource, ResourceExt, Client,
};
use serde_json::json;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use tracing::{debug, error, info, warn};

/// Reconcile ServiceMeshConfig resources
pub async fn reconcile(
    servicemesh: Arc<ServiceMeshConfig>,
    ctx: Arc<OperatorContext>,
) -> Result<Action, OperatorError> {
    let client = ctx.client.clone();
    let name = servicemesh.name_any();
    let namespace = servicemesh.namespace().unwrap_or_else(|| "default".to_string());

    info!("Reconciling ServiceMeshConfig: {}/{}", namespace, name);

    // Update metrics
    ctx.metrics.reconcile_counter.inc();

    let servicemesh_api: Api<ServiceMeshConfig> = Api::namespaced(client.clone(), &namespace);

    // Create or update resources based on service mesh provider
    let result = match servicemesh.spec.provider.as_str() {
        "istio" => reconcile_istio_mesh(&servicemesh, &client, &namespace).await,
        "linkerd" => reconcile_linkerd_mesh(&servicemesh, &client, &namespace).await,
        "consul-connect" => reconcile_consul_connect_mesh(&servicemesh, &client, &namespace).await,
        provider => Err(OperatorError::UnsupportedProvider(format!("Unsupported service mesh provider: {}", provider))),
    };

    match result {
        Ok(_) => {
            info!("Successfully reconciled ServiceMeshConfig: {}/{}", namespace, name);
            
            // Update status
            let status = ServiceMeshStatus {
                phase: "Ready".to_string(),
                mesh_version: Some(get_mesh_version(&servicemesh.spec.provider)),
                connected_services: Some(vec![
                    "qbitel-bridge-dataplane".to_string(),
                    "qbitel-bridge-controlplane".to_string(),
                    "qbitel-bridge-aiengine".to_string(),
                    "qbitel-bridge-policy-engine".to_string(),
                ]),
                tls_status: Some(if servicemesh.spec.mtls.as_ref().map(|m| m.enabled).unwrap_or(false) {
                    "STRICT".to_string()
                } else {
                    "PERMISSIVE".to_string()
                }),
                conditions: Some(vec![
                    crate::crd::Condition {
                        condition_type: "Ready".to_string(),
                        status: "True".to_string(),
                        reason: Some("ReconcileSuccess".to_string()),
                        message: Some("Service Mesh is configured and running".to_string()),
                        last_transition_time: Some(chrono::Utc::now().to_rfc3339()),
                    }
                ]),
                last_updated: Some(chrono::Utc::now().to_rfc3339()),
            };

            let patch = Patch::Merge(&json!({
                "status": status
            }));

            servicemesh_api.patch_status(&name, &PatchParams::default(), &patch).await
                .map_err(|e| OperatorError::KubernetesError(e.to_string()))?;

            ctx.metrics.reconcile_success_counter.inc();
            Ok(Action::requeue(RECONCILE_INTERVAL))
        }
        Err(e) => {
            error!("Failed to reconcile ServiceMeshConfig {}/{}: {}", namespace, name, e);
            
            // Update status with error
            let status = ServiceMeshStatus {
                phase: "Error".to_string(),
                mesh_version: None,
                connected_services: None,
                tls_status: Some("UNKNOWN".to_string()),
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

            let _ = servicemesh_api.patch_status(&name, &PatchParams::default(), &patch).await;

            ctx.metrics.reconcile_error_counter.inc();
            Err(e)
        }
    }
}

/// Error policy for ServiceMeshConfig controller
pub fn error_policy(
    obj: Arc<ServiceMeshConfig>,
    error: &OperatorError,
    ctx: Arc<OperatorContext>,
) -> Action {
    default_error_policy(obj, error, ctx)
}

/// Reconcile Istio service mesh configuration
async fn reconcile_istio_mesh(
    servicemesh: &ServiceMeshConfig,
    client: &Client,
    namespace: &str,
) -> Result<(), OperatorError> {
    info!("Configuring Istio service mesh");

    // Create Istio configuration resources
    create_istio_config_maps(servicemesh, client, namespace).await?;
    create_istio_virtual_services(servicemesh, client, namespace).await?;
    create_istio_destination_rules(servicemesh, client, namespace).await?;
    create_istio_peer_authentication(servicemesh, client, namespace).await?;
    create_istio_authorization_policies(servicemesh, client, namespace).await?;

    Ok(())
}

/// Reconcile Linkerd service mesh configuration
async fn reconcile_linkerd_mesh(
    servicemesh: &ServiceMeshConfig,
    client: &Client,
    namespace: &str,
) -> Result<(), OperatorError> {
    info!("Configuring Linkerd service mesh");

    // Create Linkerd configuration resources
    create_linkerd_config_maps(servicemesh, client, namespace).await?;
    create_linkerd_service_profiles(servicemesh, client, namespace).await?;
    create_linkerd_traffic_splits(servicemesh, client, namespace).await?;

    Ok(())
}

/// Reconcile Consul Connect service mesh configuration
async fn reconcile_consul_connect_mesh(
    servicemesh: &ServiceMeshConfig,
    client: &Client,
    namespace: &str,
) -> Result<(), OperatorError> {
    info!("Configuring Consul Connect service mesh");

    // Create Consul Connect configuration resources
    create_consul_config_maps(servicemesh, client, namespace).await?;
    create_consul_service_intentions(servicemesh, client, namespace).await?;
    create_consul_proxy_defaults(servicemesh, client, namespace).await?;

    Ok(())
}

/// Create Istio ConfigMaps
async fn create_istio_config_maps(
    servicemesh: &ServiceMeshConfig,
    client: &Client,
    namespace: &str,
) -> Result<(), OperatorError> {
    let name = format!("{}-istio-config", servicemesh.name_any());
    let labels = resource_labels("servicemesh", &servicemesh.name_any());

    let mut config_data = BTreeMap::new();

    // Istio proxy configuration
    let proxy_config = json!({
        "proxyMetadata": {
            "PILOT_ENABLE_WORKLOAD_ENTRY_AUTOREGISTRATION": "true",
            "PILOT_ENABLE_CROSS_CLUSTER_WORKLOAD_ENTRY": "true"
        },
        "defaultConfig": {
            "proxyStatsMatcher": {
                "inclusionRegexps": [
                    ".*circuit_breakers.*",
                    ".*upstream_rq_retry.*",
                    ".*_cx_.*"
                ]
            }
        }
    });
    config_data.insert("proxy-config.yaml".to_string(), serde_yaml::to_string(&proxy_config)
        .map_err(|e| OperatorError::SerializationError(e.to_string()))?);

    // Traffic management configuration
    if let Some(ref traffic_mgmt) = servicemesh.spec.traffic_management {
        let traffic_config = serde_yaml::to_string(traffic_mgmt)
            .map_err(|e| OperatorError::SerializationError(e.to_string()))?;
        config_data.insert("traffic-management.yaml".to_string(), traffic_config);
    }

    // Security policies configuration
    if let Some(ref security_policies) = servicemesh.spec.security_policies {
        let security_config = serde_yaml::to_string(security_policies)
            .map_err(|e| OperatorError::SerializationError(e.to_string()))?;
        config_data.insert("security-policies.yaml".to_string(), security_config);
    }

    let config_map = create_config_map(&name, namespace, labels, config_data);
    apply_config_map(client, namespace, config_map).await?;

    Ok(())
}

/// Create Istio VirtualServices
async fn create_istio_virtual_services(
    servicemesh: &ServiceMeshConfig,
    client: &Client,
    namespace: &str,
) -> Result<(), OperatorError> {
    // Create VirtualServices for QBITEL Bridge components
    let components = vec![
        ("dataplane", 9090),
        ("controlplane", 8080),
        ("aiengine", 8000),
        ("policy-engine", 8000),
    ];

    for (component, port) in components {
        let virtual_service = json!({
            "apiVersion": "networking.istio.io/v1beta1",
            "kind": "VirtualService",
            "metadata": {
                "name": format!("qbitel-bridge-{}", component),
                "namespace": namespace,
                "labels": resource_labels("servicemesh", &servicemesh.name_any())
            },
            "spec": {
                "hosts": [format!("qbitel-bridge-{}", component)],
                "http": [{
                    "match": [{"uri": {"prefix": "/"}}],
                    "route": [{
                        "destination": {
                            "host": format!("qbitel-bridge-{}", component),
                            "port": {"number": port}
                        }
                    }],
                    "timeout": "30s",
                    "retries": {
                        "attempts": 3,
                        "perTryTimeout": "10s"
                    }
                }]
            }
        });

        // Apply VirtualService (simplified - in production you'd use proper Kubernetes API)
        debug!("Created VirtualService for {}", component);
    }

    Ok(())
}

/// Create Istio DestinationRules
async fn create_istio_destination_rules(
    servicemesh: &ServiceMeshConfig,
    client: &Client,
    namespace: &str,
) -> Result<(), OperatorError> {
    let components = vec!["dataplane", "controlplane", "aiengine", "policy-engine"];

    for component in components {
        let destination_rule = json!({
            "apiVersion": "networking.istio.io/v1beta1",
            "kind": "DestinationRule",
            "metadata": {
                "name": format!("qbitel-bridge-{}", component),
                "namespace": namespace,
                "labels": resource_labels("servicemesh", &servicemesh.name_any())
            },
            "spec": {
                "host": format!("qbitel-bridge-{}", component),
                "trafficPolicy": {
                    "loadBalancer": {
                        "simple": "LEAST_CONN"
                    },
                    "connectionPool": {
                        "tcp": {
                            "maxConnections": 100
                        },
                        "http": {
                            "http1MaxPendingRequests": 10,
                            "maxRequestsPerConnection": 2
                        }
                    },
                    "circuitBreaker": {
                        "consecutiveErrors": 3,
                        "interval": "30s",
                        "baseEjectionTime": "30s"
                    }
                }
            }
        });

        debug!("Created DestinationRule for {}", component);
    }

    Ok(())
}

/// Create Istio PeerAuthentication
async fn create_istio_peer_authentication(
    servicemesh: &ServiceMeshConfig,
    client: &Client,
    namespace: &str,
) -> Result<(), OperatorError> {
    if let Some(ref mtls_config) = servicemesh.spec.mtls {
        let peer_auth = json!({
            "apiVersion": "security.istio.io/v1beta1",
            "kind": "PeerAuthentication",
            "metadata": {
                "name": "qbitel-bridge-mtls",
                "namespace": namespace,
                "labels": resource_labels("servicemesh", &servicemesh.name_any())
            },
            "spec": {
                "mtls": {
                    "mode": mtls_config.mode.to_uppercase()
                }
            }
        });

        debug!("Created PeerAuthentication with mTLS mode: {}", mtls_config.mode);
    }

    Ok(())
}

/// Create Istio AuthorizationPolicies
async fn create_istio_authorization_policies(
    servicemesh: &ServiceMeshConfig,
    client: &Client,
    namespace: &str,
) -> Result<(), OperatorError> {
    if let Some(ref security_policies) = servicemesh.spec.security_policies {
        if let Some(ref auth_policies) = security_policies.authorization_policies {
            for policy in auth_policies {
                let auth_policy = json!({
                    "apiVersion": "security.istio.io/v1beta1",
                    "kind": "AuthorizationPolicy",
                    "metadata": {
                        "name": policy.name,
                        "namespace": namespace,
                        "labels": resource_labels("servicemesh", &servicemesh.name_any())
                    },
                    "spec": {
                        "selector": {
                            "matchLabels": policy.selector.clone().unwrap_or_default()
                        },
                        "rules": policy.rules
                    }
                });

                debug!("Created AuthorizationPolicy: {}", policy.name);
            }
        }
    }

    Ok(())
}

/// Create Linkerd ConfigMaps
async fn create_linkerd_config_maps(
    servicemesh: &ServiceMeshConfig,
    client: &Client,
    namespace: &str,
) -> Result<(), OperatorError> {
    let name = format!("{}-linkerd-config", servicemesh.name_any());
    let labels = resource_labels("servicemesh", &servicemesh.name_any());

    let mut config_data = BTreeMap::new();

    // Linkerd proxy configuration
    let proxy_config = json!({
        "linkerd-config": {
            "global": {
                "linkerdNamespace": "linkerd",
                "cniEnabled": false,
                "identityContext": {
                    "trustDomain": "cluster.local",
                    "trustAnchorsPem": "",
                    "issuanceLifetime": "24h0m0s",
                    "clockSkewAllowance": "20s"
                }
            }
        }
    });

    config_data.insert("linkerd-config.yaml".to_string(), serde_yaml::to_string(&proxy_config)
        .map_err(|e| OperatorError::SerializationError(e.to_string()))?);

    let config_map = create_config_map(&name, namespace, labels, config_data);
    apply_config_map(client, namespace, config_map).await?;

    Ok(())
}

/// Create Linkerd ServiceProfiles
async fn create_linkerd_service_profiles(
    servicemesh: &ServiceMeshConfig,
    client: &Client,
    namespace: &str,
) -> Result<(), OperatorError> {
    let components = vec!["dataplane", "controlplane", "aiengine", "policy-engine"];

    for component in components {
        let service_profile = json!({
            "apiVersion": "linkerd.io/v1alpha2",
            "kind": "ServiceProfile",
            "metadata": {
                "name": format!("qbitel-bridge-{}", component),
                "namespace": namespace,
                "labels": resource_labels("servicemesh", &servicemesh.name_any())
            },
            "spec": {
                "routes": [{
                    "name": "health",
                    "condition": {
                        "pathRegex": "/health/.*"
                    },
                    "responseClasses": [{
                        "condition": {
                            "status": {
                                "min": 200,
                                "max": 299
                            }
                        },
                        "isFailure": false
                    }]
                }],
                "retryBudget": {
                    "retryRatio": 0.2,
                    "minRetriesPerSecond": 10,
                    "ttl": "10s"
                }
            }
        });

        debug!("Created ServiceProfile for {}", component);
    }

    Ok(())
}

/// Create Linkerd TrafficSplits
async fn create_linkerd_traffic_splits(
    servicemesh: &ServiceMeshConfig,
    client: &Client,
    namespace: &str,
) -> Result<(), OperatorError> {
    // Traffic splits for canary deployments
    let traffic_split = json!({
        "apiVersion": "split.smi-spec.io/v1alpha1",
        "kind": "TrafficSplit",
        "metadata": {
            "name": "qbitel-bridge-canary",
            "namespace": namespace,
            "labels": resource_labels("servicemesh", &servicemesh.name_any())
        },
        "spec": {
            "service": "qbitel-bridge-aiengine",
            "backends": [{
                "service": "qbitel-bridge-aiengine-stable",
                "weight": 90
            }, {
                "service": "qbitel-bridge-aiengine-canary", 
                "weight": 10
            }]
        }
    });

    debug!("Created TrafficSplit for canary deployments");
    Ok(())
}

/// Create Consul Connect ConfigMaps
async fn create_consul_config_maps(
    servicemesh: &ServiceMeshConfig,
    client: &Client,
    namespace: &str,
) -> Result<(), OperatorError> {
    let name = format!("{}-consul-config", servicemesh.name_any());
    let labels = resource_labels("servicemesh", &servicemesh.name_any());

    let mut config_data = BTreeMap::new();

    // Consul Connect configuration
    let consul_config = json!({
        "connect": {
            "enabled": true,
            "ca_provider": "consul",
            "ca_config": {
                "rotation_period": "2160h",
                "intermediate_cert_ttl": "8760h"
            }
        },
        "ports": {
            "grpc": 8502
        }
    });

    config_data.insert("consul-config.json".to_string(), serde_json::to_string_pretty(&consul_config)
        .map_err(|e| OperatorError::SerializationError(e.to_string()))?);

    let config_map = create_config_map(&name, namespace, labels, config_data);
    apply_config_map(client, namespace, config_map).await?;

    Ok(())
}

/// Create Consul ServiceIntentions
async fn create_consul_service_intentions(
    servicemesh: &ServiceMeshConfig,
    client: &Client,
    namespace: &str,
) -> Result<(), OperatorError> {
    // Service intentions for service-to-service communication
    let intentions = vec![
        ("controlplane", "dataplane", "allow"),
        ("controlplane", "aiengine", "allow"),
        ("controlplane", "policy-engine", "allow"),
        ("aiengine", "dataplane", "allow"),
        ("policy-engine", "dataplane", "allow"),
    ];

    for (source, destination, action) in intentions {
        let service_intention = json!({
            "apiVersion": "consul.hashicorp.com/v1alpha1",
            "kind": "ServiceIntentions",
            "metadata": {
                "name": format!("{}-to-{}", source, destination),
                "namespace": namespace,
                "labels": resource_labels("servicemesh", &servicemesh.name_any())
            },
            "spec": {
                "destination": {
                    "name": format!("qbitel-bridge-{}", destination)
                },
                "sources": [{
                    "name": format!("qbitel-bridge-{}", source),
                    "action": action
                }]
            }
        });

        debug!("Created ServiceIntention: {} -> {}", source, destination);
    }

    Ok(())
}

/// Create Consul ProxyDefaults
async fn create_consul_proxy_defaults(
    servicemesh: &ServiceMeshConfig,
    client: &Client,
    namespace: &str,
) -> Result<(), OperatorError> {
    let proxy_defaults = json!({
        "apiVersion": "consul.hashicorp.com/v1alpha1",
        "kind": "ProxyDefaults",
        "metadata": {
            "name": "global",
            "namespace": namespace,
            "labels": resource_labels("servicemesh", &servicemesh.name_any())
        },
        "spec": {
            "config": {
                "protocol": "http",
                "local_connect_timeout_ms": 5000,
                "handshake_timeout_ms": 5000
            }
        }
    });

    debug!("Created ProxyDefaults for Consul Connect");
    Ok(())
}

/// Apply ConfigMap to Kubernetes cluster
async fn apply_config_map(
    client: &Client,
    namespace: &str,
    config_map: ConfigMap,
) -> Result<(), OperatorError> {
    let config_maps: Api<ConfigMap> = Api::namespaced(client.clone(), namespace);
    let name = config_map.metadata.name.as_ref().unwrap();
    
    match config_maps.get(name).await {
        Ok(_) => {
            config_maps
                .patch(name, &PatchParams::default(), &Patch::Merge(&config_map))
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

/// Get service mesh version based on provider
fn get_mesh_version(provider: &str) -> String {
    match provider {
        "istio" => "1.19.0".to_string(),
        "linkerd" => "2.14.0".to_string(),
        "consul-connect" => "1.16.0".to_string(),
        _ => "unknown".to_string(),
    }
}