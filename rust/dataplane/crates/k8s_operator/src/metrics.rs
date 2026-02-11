//! Metrics collection and export for the QBITEL Bridge operator

use prometheus::{
    Counter, CounterVec, Gauge, GaugeVec, Histogram, HistogramVec, IntCounter, IntCounterVec,
    IntGauge, IntGaugeVec, Opts, Registry, TextEncoder, Encoder,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::time::{Duration, Instant};
use tracing::{debug, error, info, warn};
use warp::Filter;

/// Operator metrics collection
pub struct OperatorMetrics {
    pub registry: Registry,
    
    // Reconciliation metrics
    pub reconcile_counter: IntCounterVec,
    pub reconcile_success_counter: IntCounterVec,
    pub reconcile_error_counter: IntCounterVec,
    pub reconcile_duration_histogram: HistogramVec,
    
    // Resource metrics
    pub resources_total: IntGaugeVec,
    pub resources_ready: IntGaugeVec,
    pub resources_error: IntGaugeVec,
    
    // Controller health metrics
    pub controller_up: IntGaugeVec,
    pub controller_last_success: GaugeVec,
    
    // Kubernetes API metrics
    pub k8s_api_requests_total: IntCounterVec,
    pub k8s_api_request_duration: HistogramVec,
    pub k8s_api_errors_total: IntCounterVec,
    
    // Custom resource metrics
    pub dataplane_instances: IntGauge,
    pub controlplane_instances: IntGauge,
    pub aiengine_instances: IntGauge,
    pub policy_engine_instances: IntGauge,
    pub servicemesh_configs: IntGauge,
    
    // Performance metrics
    pub operator_memory_usage: Gauge,
    pub operator_cpu_usage: Gauge,
    pub goroutines_total: IntGauge,
    
    // Error tracking
    pub errors_by_type: IntCounterVec,
    pub warnings_total: IntCounter,
}

impl OperatorMetrics {
    /// Create new metrics instance
    pub fn new() -> Result<Self, prometheus::Error> {
        let registry = Registry::new();
        
        // Reconciliation metrics
        let reconcile_counter = IntCounterVec::new(
            Opts::new("qbitel_bridge_operator_reconcile_total", "Total number of reconciliations"),
            &["controller", "resource_type"],
        )?;
        
        let reconcile_success_counter = IntCounterVec::new(
            Opts::new("qbitel_bridge_operator_reconcile_success_total", "Total number of successful reconciliations"),
            &["controller", "resource_type"],
        )?;
        
        let reconcile_error_counter = IntCounterVec::new(
            Opts::new("qbitel_bridge_operator_reconcile_errors_total", "Total number of reconciliation errors"),
            &["controller", "resource_type", "error_type"],
        )?;
        
        let reconcile_duration_histogram = HistogramVec::new(
            prometheus::HistogramOpts::new("qbitel_bridge_operator_reconcile_duration_seconds", "Reconciliation duration"),
            &["controller", "resource_type"],
        )?;
        
        // Resource metrics
        let resources_total = IntGaugeVec::new(
            Opts::new("qbitel_bridge_operator_resources_total", "Total number of resources"),
            &["resource_type", "namespace"],
        )?;
        
        let resources_ready = IntGaugeVec::new(
            Opts::new("qbitel_bridge_operator_resources_ready", "Number of ready resources"),
            &["resource_type", "namespace"],
        )?;
        
        let resources_error = IntGaugeVec::new(
            Opts::new("qbitel_bridge_operator_resources_error", "Number of resources in error state"),
            &["resource_type", "namespace"],
        )?;
        
        // Controller health metrics
        let controller_up = IntGaugeVec::new(
            Opts::new("qbitel_bridge_operator_controller_up", "Controller health status (1 = up, 0 = down)"),
            &["controller"],
        )?;
        
        let controller_last_success = GaugeVec::new(
            Opts::new("qbitel_bridge_operator_controller_last_success_timestamp", "Timestamp of last successful reconciliation"),
            &["controller"],
        )?;
        
        // Kubernetes API metrics
        let k8s_api_requests_total = IntCounterVec::new(
            Opts::new("qbitel_bridge_operator_k8s_requests_total", "Total Kubernetes API requests"),
            &["method", "code"],
        )?;
        
        let k8s_api_request_duration = HistogramVec::new(
            prometheus::HistogramOpts::new("qbitel_bridge_operator_k8s_request_duration_seconds", "Kubernetes API request duration"),
            &["method"],
        )?;
        
        let k8s_api_errors_total = IntCounterVec::new(
            Opts::new("qbitel_bridge_operator_k8s_errors_total", "Total Kubernetes API errors"),
            &["method", "code"],
        )?;
        
        // Custom resource metrics
        let dataplane_instances = IntGauge::new("qbitel_bridge_dataplane_instances", "Number of DataPlane instances")?;
        let controlplane_instances = IntGauge::new("qbitel_bridge_controlplane_instances", "Number of ControlPlane instances")?;
        let aiengine_instances = IntGauge::new("qbitel_bridge_aiengine_instances", "Number of AIEngine instances")?;
        let policy_engine_instances = IntGauge::new("qbitel_bridge_policy_engine_instances", "Number of PolicyEngine instances")?;
        let servicemesh_configs = IntGauge::new("qbitel_bridge_servicemesh_configs", "Number of ServiceMesh configurations")?;
        
        // Performance metrics
        let operator_memory_usage = Gauge::new("qbitel_bridge_operator_memory_usage_bytes", "Operator memory usage in bytes")?;
        let operator_cpu_usage = Gauge::new("qbitel_bridge_operator_cpu_usage_percent", "Operator CPU usage percentage")?;
        let goroutines_total = IntGauge::new("qbitel_bridge_operator_goroutines", "Number of goroutines")?;
        
        // Error tracking
        let errors_by_type = IntCounterVec::new(
            Opts::new("qbitel_bridge_operator_errors_by_type_total", "Total errors by type"),
            &["error_type", "severity"],
        )?;
        
        let warnings_total = IntCounter::new("qbitel_bridge_operator_warnings_total", "Total warnings")?;
        
        // Register all metrics
        registry.register(Box::new(reconcile_counter.clone()))?;
        registry.register(Box::new(reconcile_success_counter.clone()))?;
        registry.register(Box::new(reconcile_error_counter.clone()))?;
        registry.register(Box::new(reconcile_duration_histogram.clone()))?;
        
        registry.register(Box::new(resources_total.clone()))?;
        registry.register(Box::new(resources_ready.clone()))?;
        registry.register(Box::new(resources_error.clone()))?;
        
        registry.register(Box::new(controller_up.clone()))?;
        registry.register(Box::new(controller_last_success.clone()))?;
        
        registry.register(Box::new(k8s_api_requests_total.clone()))?;
        registry.register(Box::new(k8s_api_request_duration.clone()))?;
        registry.register(Box::new(k8s_api_errors_total.clone()))?;
        
        registry.register(Box::new(dataplane_instances.clone()))?;
        registry.register(Box::new(controlplane_instances.clone()))?;
        registry.register(Box::new(aiengine_instances.clone()))?;
        registry.register(Box::new(policy_engine_instances.clone()))?;
        registry.register(Box::new(servicemesh_configs.clone()))?;
        
        registry.register(Box::new(operator_memory_usage.clone()))?;
        registry.register(Box::new(operator_cpu_usage.clone()))?;
        registry.register(Box::new(goroutines_total.clone()))?;
        
        registry.register(Box::new(errors_by_type.clone()))?;
        registry.register(Box::new(warnings_total.clone()))?;
        
        Ok(Self {
            registry,
            reconcile_counter,
            reconcile_success_counter,
            reconcile_error_counter,
            reconcile_duration_histogram,
            resources_total,
            resources_ready,
            resources_error,
            controller_up,
            controller_last_success,
            k8s_api_requests_total,
            k8s_api_request_duration,
            k8s_api_errors_total,
            dataplane_instances,
            controlplane_instances,
            aiengine_instances,
            policy_engine_instances,
            servicemesh_configs,
            operator_memory_usage,
            operator_cpu_usage,
            goroutines_total,
            errors_by_type,
            warnings_total,
        })
    }
    
    /// Record reconciliation attempt
    pub fn record_reconciliation(&self, controller: &str, resource_type: &str) {
        self.reconcile_counter
            .with_label_values(&[controller, resource_type])
            .inc();
    }
    
    /// Record successful reconciliation
    pub fn record_reconciliation_success(&self, controller: &str, resource_type: &str, duration: Duration) {
        self.reconcile_success_counter
            .with_label_values(&[controller, resource_type])
            .inc();
            
        self.reconcile_duration_histogram
            .with_label_values(&[controller, resource_type])
            .observe(duration.as_secs_f64());
            
        self.controller_last_success
            .with_label_values(&[controller])
            .set(chrono::Utc::now().timestamp() as f64);
            
        self.controller_up
            .with_label_values(&[controller])
            .set(1);
    }
    
    /// Record reconciliation error
    pub fn record_reconciliation_error(&self, controller: &str, resource_type: &str, error_type: &str) {
        self.reconcile_error_counter
            .with_label_values(&[controller, resource_type, error_type])
            .inc();
            
        self.controller_up
            .with_label_values(&[controller])
            .set(0);
    }
    
    /// Record Kubernetes API request
    pub fn record_k8s_request(&self, method: &str, status_code: u16, duration: Duration) {
        self.k8s_api_requests_total
            .with_label_values(&[method, &status_code.to_string()])
            .inc();
            
        self.k8s_api_request_duration
            .with_label_values(&[method])
            .observe(duration.as_secs_f64());
        
        if status_code >= 400 {
            self.k8s_api_errors_total
                .with_label_values(&[method, &status_code.to_string()])
                .inc();
        }
    }
    
    /// Update resource counts
    pub fn update_resource_counts(&self, resource_type: &str, namespace: &str, total: i64, ready: i64, error: i64) {
        self.resources_total
            .with_label_values(&[resource_type, namespace])
            .set(total);
            
        self.resources_ready
            .with_label_values(&[resource_type, namespace])
            .set(ready);
            
        self.resources_error
            .with_label_values(&[resource_type, namespace])
            .set(error);
    }
    
    /// Update custom resource instance counts
    pub fn update_instance_counts(&self, dataplane: i64, controlplane: i64, aiengine: i64, policy_engine: i64, servicemesh: i64) {
        self.dataplane_instances.set(dataplane);
        self.controlplane_instances.set(controlplane);
        self.aiengine_instances.set(aiengine);
        self.policy_engine_instances.set(policy_engine);
        self.servicemesh_configs.set(servicemesh);
    }
    
    /// Record error by type
    pub fn record_error(&self, error: &crate::error::OperatorError) {
        self.errors_by_type
            .with_label_values(&[error.category(), error.severity().as_str()])
            .inc();
    }
    
    /// Record warning
    pub fn record_warning(&self) {
        self.warnings_total.inc();
    }
    
    /// Update performance metrics
    pub fn update_performance_metrics(&self, memory_bytes: f64, cpu_percent: f64) {
        self.operator_memory_usage.set(memory_bytes);
        self.operator_cpu_usage.set(cpu_percent);
    }
    
    /// Export metrics in Prometheus format
    pub fn export_metrics(&self) -> Result<String, prometheus::Error> {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        encoder.encode_to_string(&metric_families)
    }
}

/// Start metrics server
pub async fn start_metrics_server(
    metrics: Arc<OperatorMetrics>,
    port: u16,
) -> Result<tokio::task::JoinHandle<()>, Box<dyn std::error::Error + Send + Sync>> {
    let metrics_route = warp::path("metrics")
        .and(warp::get())
        .and(warp::any().map(move || metrics.clone()))
        .and_then(handle_metrics);
        
    let health_route = warp::path("health")
        .and(warp::get())
        .map(|| {
            warp::reply::with_status("OK", warp::http::StatusCode::OK)
        });
        
    let ready_route = warp::path("ready")
        .and(warp::get())
        .map(|| {
            warp::reply::with_status("Ready", warp::http::StatusCode::OK)
        });
    
    let routes = metrics_route.or(health_route).or(ready_route);
    
    let addr = ([0, 0, 0, 0], port);
    info!("Starting metrics server on port {}", port);
    
    let server = warp::serve(routes).run(addr);
    let handle = tokio::spawn(server);
    
    Ok(handle)
}

/// Handle metrics endpoint
async fn handle_metrics(
    metrics: Arc<OperatorMetrics>,
) -> Result<impl warp::Reply, warp::Rejection> {
    match metrics.export_metrics() {
        Ok(metrics_output) => {
            Ok(warp::reply::with_header(
                metrics_output,
                "content-type",
                "text/plain; version=0.0.4; charset=utf-8",
            ))
        }
        Err(e) => {
            error!("Failed to export metrics: {}", e);
            Err(warp::reject::custom(MetricsError))
        }
    }
}

/// Custom rejection for metrics errors
#[derive(Debug)]
struct MetricsError;

impl warp::reject::Reject for MetricsError {}

/// Periodic metrics collection task
pub async fn start_metrics_collector(metrics: Arc<OperatorMetrics>) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(30));
        
        loop {
            interval.tick().await;
            collect_system_metrics(&metrics).await;
        }
    })
}

/// Collect system-level metrics
async fn collect_system_metrics(metrics: &OperatorMetrics) {
    // Memory usage
    if let Ok(memory_info) = sys_info::mem_info() {
        let used_memory = (memory_info.total - memory_info.free - memory_info.cached - memory_info.buffers) * 1024; // Convert to bytes
        metrics.operator_memory_usage.set(used_memory as f64);
    }
    
    // CPU usage (simplified - in production you'd want more accurate measurements)
    if let Ok(load_avg) = sys_info::loadavg() {
        metrics.operator_cpu_usage.set(load_avg.one as f64);
    }
    
    // Thread count approximation
    let thread_count = std::thread::available_parallelism()
        .map(|n| n.get() as i64)
        .unwrap_or(1);
    metrics.goroutines_total.set(thread_count);
}

/// Timer for measuring operation duration
pub struct OperationTimer {
    start: Instant,
    histogram: HistogramVec,
    labels: Vec<String>,
}

impl OperationTimer {
    /// Start a new timer
    pub fn new(histogram: HistogramVec, labels: Vec<String>) -> Self {
        Self {
            start: Instant::now(),
            histogram,
            labels,
        }
    }
    
    /// Record the elapsed time
    pub fn record(self) {
        let duration = self.start.elapsed();
        let label_refs: Vec<&str> = self.labels.iter().map(|s| s.as_str()).collect();
        self.histogram
            .with_label_values(&label_refs)
            .observe(duration.as_secs_f64());
    }
}

/// Convenience macro for timing operations
#[macro_export]
macro_rules! time_operation {
    ($metrics:expr, $histogram:ident, $labels:expr, $operation:expr) => {{
        let timer = $crate::metrics::OperationTimer::new(
            $metrics.$histogram.clone(),
            $labels.into_iter().map(|s| s.to_string()).collect(),
        );
        let result = $operation;
        timer.record();
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    
    #[tokio::test]
    async fn test_metrics_creation() {
        let metrics = OperatorMetrics::new().unwrap();
        assert!(metrics.export_metrics().is_ok());
    }
    
    #[tokio::test]
    async fn test_reconciliation_metrics() {
        let metrics = OperatorMetrics::new().unwrap();
        
        metrics.record_reconciliation("dataplane", "DataPlaneService");
        metrics.record_reconciliation_success("dataplane", "DataPlaneService", Duration::from_millis(100));
        
        let output = metrics.export_metrics().unwrap();
        assert!(output.contains("qbitel_bridge_operator_reconcile_total"));
        assert!(output.contains("qbitel_bridge_operator_reconcile_success_total"));
    }
    
    #[tokio::test]
    async fn test_error_recording() {
        let metrics = OperatorMetrics::new().unwrap();
        
        let error = crate::error::OperatorError::KubernetesError("test".to_string());
        metrics.record_error(&error);
        
        let output = metrics.export_metrics().unwrap();
        assert!(output.contains("qbitel_bridge_operator_errors_by_type_total"));
    }
}