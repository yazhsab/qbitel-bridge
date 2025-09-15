use adapter_sdk::L7Adapter;
use bytes::Bytes;
use io_engine::{Bridge, UpstreamSelector};
use io_engine::ha::ActiveStandby;
use io_engine::metrics as qmetrics;
use std::sync::Arc;
use tokio::signal;
use tracing_subscriber::EnvFilter;
use std::env;
use std::net::SocketAddr;
use hyper::{Body, Request, Response, Server};
use hyper::service::{make_service_fn, service_fn};

struct EchoAdapter;
#[async_trait::async_trait]
impl L7Adapter for EchoAdapter {
    async fn to_upstream(&self, input: Bytes) -> Result<Bytes, adapter_sdk::AdapterError> { Ok(input) }
    async fn to_client(&self, input: Bytes) -> Result<Bytes, adapter_sdk::AdapterError> { Ok(input) }
    fn name(&self) -> &'static str { "echo" }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter(EnvFilter::from_default_env()).init();

    let listen_addr = env::var("BRIDGE_LISTEN").unwrap_or_else(|_| "127.0.0.1:9000".into());
    let upstream_primary = env::var("UPSTREAM_PRIMARY").unwrap_or_else(|_| "127.0.0.1:9001".into());
    let upstream_standby = env::var("UPSTREAM_STANDBY").ok();
    let metrics_addr: SocketAddr = env::var("METRICS_ADDR").unwrap_or_else(|_| "127.0.0.1:9090".into()).parse().unwrap();

    let upstream = match upstream_standby {
        Some(standby) => {
            let ha = ActiveStandby::new(upstream_primary, standby, std::time::Duration::from_millis(500), std::time::Duration::from_millis(300));
            UpstreamSelector::HA(ha)
        }
        None => UpstreamSelector::Static(upstream_primary),
    };

    let bridge = Bridge {
        listen_addr,
        upstream,
        adapter: Arc::new(EchoAdapter),
        buf_size: 64 * 1024,
        chan_capacity: 64,
        io_timeout: std::time::Duration::from_secs(5),
    };

    // Metrics endpoint
    tokio::spawn(async move {
        let make_svc = make_service_fn(|_conn| async {
            Ok::<_, hyper::Error>(service_fn(|req: Request<Body>| async move {
                if req.uri().path() == "/metrics" {
                    let body = qmetrics::gather_text();
                    Ok::<_, hyper::Error>(Response::new(Body::from(body)))
                } else {
                    Ok::<_, hyper::Error>(Response::builder().status(404).body(Body::from("not found")).unwrap())
                }
            }))
        });
        let server = Server::bind(&metrics_addr).serve(make_svc);
        if let Err(e) = server.await { eprintln!("metrics server error: {}", e); }
    });

    tokio::spawn(async move {
        let _ = bridge.run().await;
    });

    signal::ctrl_c().await?;
    Ok(())
}
