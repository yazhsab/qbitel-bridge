use once_cell::sync::Lazy;
use prometheus::{register_histogram, register_int_counter, register_int_gauge, Encoder, Histogram, IntCounter, IntGauge, TextEncoder};

static QBITEL_CONNECTIONS: Lazy<IntGauge> = Lazy::new(|| register_int_gauge!("qbitel_connections", "Current number of active connections").unwrap());
static QBITEL_BYTES_IN: Lazy<IntCounter> = Lazy::new(|| register_int_counter!("qbitel_bytes_in", "Total bytes received from clients").unwrap());
static QBITEL_BYTES_OUT: Lazy<IntCounter> = Lazy::new(|| register_int_counter!("qbitel_bytes_out", "Total bytes sent upstream").unwrap());
static QBITEL_HS_LATENCY: Lazy<Histogram> = Lazy::new(|| register_histogram!("qbitel_handshake_latency_ms", "TCP connect/handshake latency in ms", prometheus::exponential_buckets(0.5, 1.7, 20).unwrap()).unwrap());

pub fn inc_connections() { QBITEL_CONNECTIONS.inc(); }
pub fn dec_connections() { QBITEL_CONNECTIONS.dec(); }
pub fn inc_bytes_in(n: u64) { QBITEL_BYTES_IN.inc_by(n); }
pub fn inc_bytes_out(n: u64) { QBITEL_BYTES_OUT.inc_by(n); }
pub fn observe_handshake_latency(d: std::time::Duration) { QBITEL_HS_LATENCY.observe(d.as_secs_f64() * 1000.0); }

pub fn gather_text() -> String {
    let metric_families = prometheus::gather();
    let mut buf = Vec::new();
    TextEncoder::new().encode(&metric_families, &mut buf).ok();
    String::from_utf8(buf).unwrap_or_default()
}

