//! QBITEL Bridge Kubernetes Operator
//! 
//! Enterprise-grade Kubernetes operator for managing QBITEL Bridge services,
//! including dataplane, control plane, AI engine, and service mesh integration.

pub mod crd;
pub mod controller;
pub mod reconciler;
pub mod service_mesh;
pub mod health;
pub mod metrics;
pub mod config;
pub mod error;

use anyhow::Result;
use kube::{Api, Client, CustomResourceExt};
use kube_runtime::{
    controller::{Action, Controller},
    finalizer::{finalizer, Event},
    watcher::Config as WatcherConfig,
};
use std::sync::Arc;
use std::time::Duration;
use tokio::time;
use tracing::{info, warn, error};

use crate::crd::*;
use crate::controller::*;
use crate::error::OperatorError;

/// Main operator context containing shared resources
#[derive(Clone)]
pub struct OperatorContext {
    pub client: Client,
    pub metrics: Arc<metrics::OperatorMetrics>,
    pub config: Arc<config::OperatorConfig>,
}

/// Initialize and run the QBITEL Bridge operator
pub async fn run_operator() -> Result<()> {
    info!("Starting QBITEL Bridge Kubernetes Operator");

    // Initialize Kubernetes client
    let client = Client::try_default().await?;
    
    // Initialize metrics
    let metrics = Arc::new(metrics::OperatorMetrics::new()?);
    
    // Load configuration
    let config = Arc::new(config::OperatorConfig::load()?);
    
    // Create operator context
    let context = OperatorContext {
        client: client.clone(),
        metrics: metrics.clone(),
        config: config.clone(),
    };

    // Install CRDs if required
    install_crds(&client).await?;

    // Start metrics server
    let metrics_server = metrics::start_metrics_server(metrics.clone(), config.metrics_port)?;
    
    // Start controllers concurrently
    let controllers = tokio::try_join!(
        run_dataplane_controller(context.clone()),
        run_controlplane_controller(context.clone()),
        run_aiengine_controller(context.clone()),
        run_policy_controller(context.clone()),
        run_service_mesh_controller(context.clone()),
    );

    match controllers {
        Ok(_) => {
            info!("All controllers started successfully");
        }
        Err(e) => {
            error!("Failed to start controllers: {}", e);
            return Err(e.into());
        }
    }

    // Graceful shutdown handling
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            info!("Received SIGINT, shutting down gracefully");
        }
        _ = metrics_server => {
            warn!("Metrics server terminated unexpectedly");
        }
    }

    Ok(())
}

/// Install Custom Resource Definitions
async fn install_crds(client: &Client) -> Result<()> {
    info!("Installing Custom Resource Definitions");

    // Install CRDs for all QBITEL Bridge resources
    let crds = vec![
        DataPlaneService::crd(),
        ControlPlaneService::crd(),
        AIEngineService::crd(),
        PolicyEngineService::crd(),
        ServiceMeshConfig::crd(),
    ];

    for crd in crds {
        let crd_api: Api<k8s_openapi::apiextensions_apiserver::pkg::apis::apiextensions::v1::CustomResourceDefinition> = 
            Api::all(client.clone());
        
        match crd_api.create(&Default::default(), &crd).await {
            Ok(_) => info!("Created CRD: {}", crd.metadata.name.as_ref().unwrap()),
            Err(kube::Error::Api(kube::core::ErrorResponse { code: 409, .. })) => {
                info!("CRD already exists: {}", crd.metadata.name.as_ref().unwrap());
            }
            Err(e) => {
                error!("Failed to create CRD {}: {}", crd.metadata.name.as_ref().unwrap(), e);
                return Err(e.into());
            }
        }
    }

    // Wait for CRDs to be established
    time::sleep(Duration::from_secs(5)).await;
    
    Ok(())
}

/// Run DataPlane service controller
async fn run_dataplane_controller(context: OperatorContext) -> Result<()> {
    info!("Starting DataPlane controller");
    
    let client = context.client.clone();
    let api: Api<DataPlaneService> = Api::all(client.clone());

    Controller::new(api, WatcherConfig::default())
        .shutdown_on_signal()
        .run(
            controller::dataplane::reconcile,
            controller::dataplane::error_policy,
            Arc::new(context),
        )
        .for_each(|res| async move {
            match res {
                Ok(o) => info!("DataPlane reconciled: {:?}", o),
                Err(e) => error!("DataPlane reconcile failed: {}", e),
            }
        })
        .await;

    Ok(())
}

/// Run ControlPlane service controller
async fn run_controlplane_controller(context: OperatorContext) -> Result<()> {
    info!("Starting ControlPlane controller");
    
    let client = context.client.clone();
    let api: Api<ControlPlaneService> = Api::all(client.clone());

    Controller::new(api, WatcherConfig::default())
        .shutdown_on_signal()
        .run(
            controller::controlplane::reconcile,
            controller::controlplane::error_policy,
            Arc::new(context),
        )
        .for_each(|res| async move {
            match res {
                Ok(o) => info!("ControlPlane reconciled: {:?}", o),
                Err(e) => error!("ControlPlane reconcile failed: {}", e),
            }
        })
        .await;

    Ok(())
}

/// Run AI Engine controller
async fn run_aiengine_controller(context: OperatorContext) -> Result<()> {
    info!("Starting AI Engine controller");
    
    let client = context.client.clone();
    let api: Api<AIEngineService> = Api::all(client.clone());

    Controller::new(api, WatcherConfig::default())
        .shutdown_on_signal()
        .run(
            controller::aiengine::reconcile,
            controller::aiengine::error_policy,
            Arc::new(context),
        )
        .for_each(|res| async move {
            match res {
                Ok(o) => info!("AI Engine reconciled: {:?}", o),
                Err(e) => error!("AI Engine reconcile failed: {}", e),
            }
        })
        .await;

    Ok(())
}

/// Run Policy Engine controller
async fn run_policy_controller(context: OperatorContext) -> Result<()> {
    info!("Starting Policy Engine controller");
    
    let client = context.client.clone();
    let api: Api<PolicyEngineService> = Api::all(client.clone());

    Controller::new(api, WatcherConfig::default())
        .shutdown_on_signal()
        .run(
            controller::policy::reconcile,
            controller::policy::error_policy,
            Arc::new(context),
        )
        .for_each(|res| async move {
            match res {
                Ok(o) => info!("Policy Engine reconciled: {:?}", o),
                Err(e) => error!("Policy Engine reconcile failed: {}", e),
            }
        })
        .await;

    Ok(())
}

/// Run Service Mesh controller
async fn run_service_mesh_controller(context: OperatorContext) -> Result<()> {
    info!("Starting Service Mesh controller");
    
    let client = context.client.clone();
    let api: Api<ServiceMeshConfig> = Api::all(client.clone());

    Controller::new(api, WatcherConfig::default())
        .shutdown_on_signal()
        .run(
            controller::servicemesh::reconcile,
            controller::servicemesh::error_policy,
            Arc::new(context),
        )
        .for_each(|res| async move {
            match res {
                Ok(o) => info!("Service Mesh reconciled: {:?}", o),
                Err(e) => error!("Service Mesh reconcile failed: {}", e),
            }
        })
        .await;

    Ok(())
}