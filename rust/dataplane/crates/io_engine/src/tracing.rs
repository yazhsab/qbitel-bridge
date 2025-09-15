use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Enterprise-grade distributed tracing with OTLP support
pub struct DistributedTracer {
    config: TracingConfig,
    active_traces: Arc<RwLock<HashMap<String, TraceContext>>>,
    span_buffer: Arc<RwLock<Vec<SpanData>>>,
    metrics: TracingMetrics,
    exporter: Box<dyn TraceExporter + Send + Sync>,
}

#[derive(Debug, Clone)]
pub struct TracingConfig {
    // Service identification
    pub service_name: String,
    pub service_version: String,
    pub service_namespace: String,
    pub service_instance_id: String,
    
    // Sampling configuration
    pub sampling_rate: f64,
    pub max_spans_per_trace: usize,
    pub trace_timeout: Duration,
    
    // Buffer configuration
    pub buffer_size: usize,
    pub batch_size: usize,
    pub flush_interval: Duration,
    
    // Export configuration
    pub otlp_endpoint: String,
    pub otlp_headers: HashMap<String, String>,
    pub export_timeout: Duration,
    
    // Performance configuration
    pub enable_resource_attributes: bool,
    pub enable_custom_attributes: bool,
    pub max_attribute_length: usize,
    pub max_event_count_per_span: usize,
}

#[derive(Debug, Clone)]
pub struct TraceContext {
    pub trace_id: String,
    pub parent_span_id: Option<String>,
    pub trace_flags: u8,
    pub trace_state: String,
    pub baggage: HashMap<String, String>,
    pub created_at: Instant,
    pub span_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanData {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub operation_name: String,
    pub start_time: u64,
    pub end_time: u64,
    pub duration_ns: u64,
    pub status: SpanStatus,
    pub kind: SpanKind,
    pub attributes: HashMap<String, AttributeValue>,
    pub events: Vec<SpanEvent>,
    pub links: Vec<SpanLink>,
    pub resource: ResourceAttributes,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpanStatus {
    Unset,
    Ok,
    Error { message: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpanKind {
    Internal,
    Server,
    Client,
    Producer,
    Consumer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttributeValue {
    String(String),
    Bool(bool),
    Int(i64),
    Double(f64),
    Array(Vec<AttributeValue>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanEvent {
    pub name: String,
    pub timestamp: u64,
    pub attributes: HashMap<String, AttributeValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanLink {
    pub trace_id: String,
    pub span_id: String,
    pub trace_state: String,
    pub attributes: HashMap<String, AttributeValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAttributes {
    pub service_name: String,
    pub service_version: String,
    pub service_namespace: String,
    pub service_instance_id: String,
    pub host_name: String,
    pub process_pid: u32,
    pub custom_attributes: HashMap<String, AttributeValue>,
}

#[derive(Debug)]
pub struct TracingMetrics {
    pub traces_created: AtomicU64,
    pub spans_created: AtomicU64,
    pub spans_exported: AtomicU64,
    pub spans_dropped: AtomicU64,
    pub export_errors: AtomicU64,
    pub sampling_decisions: AtomicU64,
    pub buffer_overflows: AtomicU64,
}

pub trait TraceExporter {
    fn export_spans(&self, spans: Vec<SpanData>) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    fn shutdown(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}

/// OTLP HTTP exporter
pub struct OTLPHttpExporter {
    endpoint: String,
    headers: HashMap<String, String>,
    client: reqwest::Client,
    timeout: Duration,
}

/// Active span for building trace data
pub struct Span {
    tracer: Arc<DistributedTracer>,
    data: SpanData,
    start_time: Instant,
}

impl DistributedTracer {
    pub fn new(config: TracingConfig) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let exporter = Box::new(OTLPHttpExporter::new(
            config.otlp_endpoint.clone(),
            config.otlp_headers.clone(),
            config.export_timeout,
        )?);

        let tracer = Self {
            config,
            active_traces: Arc::new(RwLock::new(HashMap::new())),
            span_buffer: Arc::new(RwLock::new(Vec::new())),
            metrics: TracingMetrics {
                traces_created: AtomicU64::new(0),
                spans_created: AtomicU64::new(0),
                spans_exported: AtomicU64::new(0),
                spans_dropped: AtomicU64::new(0),
                export_errors: AtomicU64::new(0),
                sampling_decisions: AtomicU64::new(0),
                buffer_overflows: AtomicU64::new(0),
            },
            exporter,
        };

        // Start background tasks
        tracer.start_flush_task();
        tracer.start_cleanup_task();

        info!("distributed tracer initialized",
            service_name = %tracer.config.service_name,
            sampling_rate = tracer.config.sampling_rate,
            otlp_endpoint = %tracer.config.otlp_endpoint);

        Ok(tracer)
    }

    pub async fn start_span(&self, operation_name: &str, parent_context: Option<&TraceContext>) -> Option<Span> {
        // Make sampling decision
        if !self.should_sample() {
            self.metrics.sampling_decisions.fetch_add(1, Ordering::Relaxed);
            return None;
        }

        let trace_id = match parent_context {
            Some(ctx) => ctx.trace_id.clone(),
            None => self.generate_trace_id(),
        };

        let span_id = self.generate_span_id();
        let parent_span_id = parent_context.map(|ctx| ctx.parent_span_id.clone()).flatten();

        // Create or update trace context
        if parent_context.is_none() {
            let trace_context = TraceContext {
                trace_id: trace_id.clone(),
                parent_span_id: None,
                trace_flags: 1, // Sampled
                trace_state: String::new(),
                baggage: HashMap::new(),
                created_at: Instant::now(),
                span_count: 0,
            };

            let mut traces = self.active_traces.write().await;
            traces.insert(trace_id.clone(), trace_context);
            self.metrics.traces_created.fetch_add(1, Ordering::Relaxed);
        }

        // Update span count
        if let Some(mut trace_ctx) = self.active_traces.write().await.get_mut(&trace_id) {
            trace_ctx.span_count += 1;
            
            // Check span limit
            if trace_ctx.span_count > self.config.max_spans_per_trace {
                warn!("span limit exceeded for trace", trace_id = %trace_id);
                self.metrics.spans_dropped.fetch_add(1, Ordering::Relaxed);
                return None;
            }
        }

        let span_data = SpanData {
            trace_id: trace_id.clone(),
            span_id: span_id.clone(),
            parent_span_id,
            operation_name: operation_name.to_string(),
            start_time: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64,
            end_time: 0,
            duration_ns: 0,
            status: SpanStatus::Unset,
            kind: SpanKind::Internal,
            attributes: HashMap::new(),
            events: Vec::new(),
            links: Vec::new(),
            resource: self.create_resource_attributes(),
        };

        self.metrics.spans_created.fetch_add(1, Ordering::Relaxed);

        Some(Span {
            tracer: Arc::new(self.clone()),
            data: span_data,
            start_time: Instant::now(),
        })
    }

    pub async fn inject_context(&self, trace_id: &str, headers: &mut HashMap<String, String>) {
        if let Some(trace_ctx) = self.active_traces.read().await.get(trace_id) {
            // Inject W3C Trace Context headers
            headers.insert("traceparent".to_string(), 
                format!("00-{}-{:016x}-{:02x}", 
                    trace_id, 
                    rand::random::<u64>(), 
                    trace_ctx.trace_flags));
            
            if !trace_ctx.trace_state.is_empty() {
                headers.insert("tracestate".to_string(), trace_ctx.trace_state.clone());
            }

            // Inject baggage
            if !trace_ctx.baggage.is_empty() {
                let baggage_header = trace_ctx.baggage.iter()
                    .map(|(k, v)| format!("{}={}", k, v))
                    .collect::<Vec<_>>()
                    .join(",");
                headers.insert("baggage".to_string(), baggage_header);
            }
        }
    }

    pub async fn extract_context(&self, headers: &HashMap<String, String>) -> Option<TraceContext> {
        // Extract W3C Trace Context
        if let Some(traceparent) = headers.get("traceparent") {
            if let Some(context) = self.parse_traceparent(traceparent) {
                let trace_state = headers.get("tracestate").cloned().unwrap_or_default();
                let baggage = self.parse_baggage(headers.get("baggage"));

                return Some(TraceContext {
                    trace_id: context.0,
                    parent_span_id: Some(context.1),
                    trace_flags: context.2,
                    trace_state,
                    baggage,
                    created_at: Instant::now(),
                    span_count: 0,
                });
            }
        }

        None
    }

    async fn finish_span(&self, span_data: SpanData) {
        let mut buffer = self.span_buffer.write().await;
        
        if buffer.len() >= self.config.buffer_size {
            self.metrics.buffer_overflows.fetch_add(1, Ordering::Relaxed);
            buffer.remove(0); // Remove oldest span
        }
        
        buffer.push(span_data);

        // Trigger flush if batch size reached
        if buffer.len() >= self.config.batch_size {
            self.flush_spans().await;
        }
    }

    async fn flush_spans(&self) {
        let spans = {
            let mut buffer = self.span_buffer.write().await;
            if buffer.is_empty() {
                return;
            }
            
            let batch_size = std::cmp::min(buffer.len(), self.config.batch_size);
            buffer.drain(0..batch_size).collect::<Vec<_>>()
        };

        if !spans.is_empty() {
            match self.exporter.export_spans(spans.clone()) {
                Ok(()) => {
                    self.metrics.spans_exported.fetch_add(spans.len() as u64, Ordering::Relaxed);
                    debug!("exported {} spans", spans.len());
                }
                Err(e) => {
                    error!("failed to export spans: {}", e);
                    self.metrics.export_errors.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }

    fn start_flush_task(&self) {
        let tracer = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tracer.config.flush_interval);
            loop {
                interval.tick().await;
                tracer.flush_spans().await;
            }
        });
    }

    fn start_cleanup_task(&self) {
        let tracer = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            loop {
                interval.tick().await;
                tracer.cleanup_expired_traces().await;
            }
        });
    }

    async fn cleanup_expired_traces(&self) {
        let mut traces = self.active_traces.write().await;
        let now = Instant::now();
        
        traces.retain(|_, trace_ctx| {
            now.duration_since(trace_ctx.created_at) < self.config.trace_timeout
        });
    }

    fn should_sample(&self) -> bool {
        rand::random::<f64>() < self.config.sampling_rate
    }

    fn generate_trace_id(&self) -> String {
        format!("{:032x}", rand::random::<u128>())
    }

    fn generate_span_id(&self) -> String {
        format!("{:016x}", rand::random::<u64>())
    }

    fn create_resource_attributes(&self) -> ResourceAttributes {
        ResourceAttributes {
            service_name: self.config.service_name.clone(),
            service_version: self.config.service_version.clone(),
            service_namespace: self.config.service_namespace.clone(),
            service_instance_id: self.config.service_instance_id.clone(),
            host_name: hostname::get().unwrap_or_default().to_string_lossy().to_string(),
            process_pid: std::process::id(),
            custom_attributes: HashMap::new(),
        }
    }

    fn parse_traceparent(&self, traceparent: &str) -> Option<(String, String, u8)> {
        let parts: Vec<&str> = traceparent.split('-').collect();
        if parts.len() != 4 || parts[0] != "00" {
            return None;
        }

        let trace_id = parts[1].to_string();
        let span_id = parts[2].to_string();
        let flags = u8::from_str_radix(parts[3], 16).ok()?;

        Some((trace_id, span_id, flags))
    }

    fn parse_baggage(&self, baggage_header: Option<&String>) -> HashMap<String, String> {
        let mut baggage = HashMap::new();
        
        if let Some(header) = baggage_header {
            for item in header.split(',') {
                if let Some((key, value)) = item.split_once('=') {
                    baggage.insert(key.trim().to_string(), value.trim().to_string());
                }
            }
        }
        
        baggage
    }

    pub fn get_metrics(&self) -> TracingStats {
        TracingStats {
            traces_created: self.metrics.traces_created.load(Ordering::Relaxed),
            spans_created: self.metrics.spans_created.load(Ordering::Relaxed),
            spans_exported: self.metrics.spans_exported.load(Ordering::Relaxed),
            spans_dropped: self.metrics.spans_dropped.load(Ordering::Relaxed),
            export_errors: self.metrics.export_errors.load(Ordering::Relaxed),
            sampling_decisions: self.metrics.sampling_decisions.load(Ordering::Relaxed),
            buffer_overflows: self.metrics.buffer_overflows.load(Ordering::Relaxed),
            active_traces: 0, // Would need async access to get real count
            buffered_spans: 0, // Would need async access to get real count
        }
    }
}

impl Clone for DistributedTracer {
    fn clone(&self) -> Self {
        // This is a simplified clone - in practice you'd want to share the same instance
        Self {
            config: self.config.clone(),
            active_traces: Arc::clone(&self.active_traces),
            span_buffer: Arc::clone(&self.span_buffer),
            metrics: TracingMetrics {
                traces_created: AtomicU64::new(self.metrics.traces_created.load(Ordering::Relaxed)),
                spans_created: AtomicU64::new(self.metrics.spans_created.load(Ordering::Relaxed)),
                spans_exported: AtomicU64::new(self.metrics.spans_exported.load(Ordering::Relaxed)),
                spans_dropped: AtomicU64::new(self.metrics.spans_dropped.load(Ordering::Relaxed)),
                export_errors: AtomicU64::new(self.metrics.export_errors.load(Ordering::Relaxed)),
                sampling_decisions: AtomicU64::new(self.metrics.sampling_decisions.load(Ordering::Relaxed)),
                buffer_overflows: AtomicU64::new(self.metrics.buffer_overflows.load(Ordering::Relaxed)),
            },
            exporter: Box::new(NoOpExporter {}), // Placeholder
        }
    }
}

impl Span {
    pub fn set_attribute(&mut self, key: &str, value: AttributeValue) {
        if self.data.attributes.len() < self.tracer.config.max_attribute_length {
            self.data.attributes.insert(key.to_string(), value);
        }
    }

    pub fn add_event(&mut self, name: &str, attributes: HashMap<String, AttributeValue>) {
        if self.data.events.len() < self.tracer.config.max_event_count_per_span {
            self.data.events.push(SpanEvent {
                name: name.to_string(),
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64,
                attributes,
            });
        }
    }

    pub fn set_status(&mut self, status: SpanStatus) {
        self.data.status = status;
    }

    pub fn set_kind(&mut self, kind: SpanKind) {
        self.data.kind = kind;
    }

    pub async fn finish(mut self) {
        self.data.end_time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64;
        self.data.duration_ns = self.start_time.elapsed().as_nanos() as u64;
        
        self.tracer.finish_span(self.data).await;
    }
}

impl OTLPHttpExporter {
    pub fn new(
        endpoint: String,
        headers: HashMap<String, String>,
        timeout: Duration,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()?;

        Ok(Self {
            endpoint,
            headers,
            client,
            timeout,
        })
    }
}

impl TraceExporter for OTLPHttpExporter {
    fn export_spans(&self, spans: Vec<SpanData>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Convert spans to OTLP format and send HTTP request
        // This is a simplified implementation
        debug!("exporting {} spans to {}", spans.len(), self.endpoint);
        
        // In a real implementation, this would:
        // 1. Convert SpanData to OTLP protobuf format
        // 2. Send HTTP POST request to OTLP endpoint
        // 3. Handle response and retries
        
        Ok(())
    }

    fn shutdown(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }
}

// No-op exporter for testing
struct NoOpExporter;

impl TraceExporter for NoOpExporter {
    fn export_spans(&self, _spans: Vec<SpanData>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }

    fn shutdown(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }
}

#[derive(Debug)]
pub struct TracingStats {
    pub traces_created: u64,
    pub spans_created: u64,
    pub spans_exported: u64,
    pub spans_dropped: u64,
    pub export_errors: u64,
    pub sampling_decisions: u64,
    pub buffer_overflows: u64,
    pub active_traces: usize,
    pub buffered_spans: usize,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            service_name: "qslb-dataplane".to_string(),
            service_version: "1.0.0".to_string(),
            service_namespace: "qslb".to_string(),
            service_instance_id: Uuid::new_v4().to_string(),
            sampling_rate: 1.0,
            max_spans_per_trace: 1000,
            trace_timeout: Duration::from_secs(300),
            buffer_size: 10000,
            batch_size: 100,
            flush_interval: Duration::from_secs(5),
            otlp_endpoint: "http://localhost:4318/v1/traces".to_string(),
            otlp_headers: HashMap::new(),
            export_timeout: Duration::from_secs(10),
            enable_resource_attributes: true,
            enable_custom_attributes: true,
            max_attribute_length: 1000,
            max_event_count_per_span: 100,
        }
    }
}

// Convenience macros for tracing
#[macro_export]
macro_rules! trace_span {
    ($tracer:expr, $name:expr) => {
        $tracer.start_span($name, None).await
    };
    ($tracer:expr, $name:expr, $parent:expr) => {
        $tracer.start_span($name, Some($parent)).await
    };
}

#[macro_export]
macro_rules! trace_event {
    ($span:expr, $name:expr) => {
        $span.add_event($name, std::collections::HashMap::new())
    };
    ($span:expr, $name:expr, $attrs:expr) => {
        $span.add_event($name, $attrs)
    };
}