pub mod metrics;
pub mod ha;
pub mod performance;
pub mod high_frequency_metrics;
pub mod tracing;

pub use performance::{PerformanceEngine, PerformanceConfig, PerformanceStats};
pub use high_frequency_metrics::{HighFrequencyMetrics, HFMetricsConfig, HFMetricsStats};
pub use tracing::{DistributedTracer, TracingConfig, Span, AttributeValue};

use adapter_sdk::L7Adapter;
use bytes::{Bytes, BytesMut};
use ha::ActiveStandby;
use metrics as qmetrics;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio::time::{timeout, Duration, Instant};
use tracing::{error, info, instrument, warn};

pub enum UpstreamSelector {
    Static(String),
    HA(Arc<ActiveStandby>),
}

pub struct Bridge<A: L7Adapter + 'static> {
    pub listen_addr: String,
    pub upstream: UpstreamSelector,
    pub adapter: Arc<A>,
    pub buf_size: usize,
    pub chan_capacity: usize,
    pub io_timeout: Duration,
}

impl<A: L7Adapter + 'static> Bridge<A> {
    fn upstream_addr(&self) -> String {
        match &self.upstream {
            UpstreamSelector::Static(s) => s.clone(),
            UpstreamSelector::HA(h) => h.current().to_string(),
        }
    }

    #[instrument(skip(self))]
    pub async fn run(self) -> anyhow::Result<()> {
        let listener = TcpListener::bind(&self.listen_addr).await?;
        info!(addr=%self.listen_addr, "bridge listening");

        loop {
            let (mut inbound, peer) = listener.accept().await?;
            let upstream_sel = match &self.upstream {
                UpstreamSelector::Static(s) => UpstreamSelector::Static(s.clone()),
                UpstreamSelector::HA(h) => UpstreamSelector::HA(Arc::clone(h)),
            };
            let adapter = Arc::clone(&self.adapter);
            let adapter_name = adapter.name();
            let buf_size = self.buf_size;
            let chan_capacity = self.chan_capacity;
            let io_timeout = self.io_timeout;

            tokio::spawn(async move {
                let upstream = match upstream_sel {
                    UpstreamSelector::Static(s) => s,
                    UpstreamSelector::HA(h) => h.current().to_string(),
                };
                let hs_start = Instant::now();
                match timeout(io_timeout, TcpStream::connect(&upstream)).await {
                    Ok(Ok(mut out)) => {
                        let dur = hs_start.elapsed();
                        qmetrics::observe_handshake_latency(dur);
                        info!(%peer, adapter = adapter_name, ms=%dur.as_millis(), "accepted and connected upstream");

                        let (mut ri, mut wi) = inbound.split();
                        let (mut ro, mut wo) = out.split();

                        // bounded channels for backpressure
                        let (tx_up, mut rx_up) = mpsc::channel::<Bytes>(chan_capacity);
                        let (tx_down, mut rx_down) = mpsc::channel::<Bytes>(chan_capacity);

                        // reader: client -> adapter -> tx_up
                        let reader = async {
                            let mut buf = BytesMut::with_capacity(buf_size);
                            loop {
                                buf.resize(buf_size, 0);
                                let n = match timeout(io_timeout, ri.read(&mut buf)).await {
                                    Ok(Ok(n)) => n,
                                    Ok(Err(e)) => return Err(e),
                                    Err(_) => return Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "read timeout")),
                                };
                                if n == 0 { break; }
                                let frame = Bytes::copy_from_slice(&buf[..n]);
                                qmetrics::inc_bytes_in(n as u64);
                                let up_bytes = adapter
                                    .to_upstream(frame)
                                    .await
                                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
                                if tx_up.send(up_bytes).await.is_err() { break; }
                            }
                            Ok::<_, std::io::Error>(())
                        };

                        // writer: rx_up -> upstream
                        let writer_up = async {
                            while let Some(b) = rx_up.recv().await {
                                qmetrics::inc_bytes_out(b.len() as u64);
                                timeout(io_timeout, wo.write_all(&b)).await??;
                            }
                            Ok::<_, std::io::Error>(())
                        };

                        // reader: upstream -> adapter -> tx_down
                        let upstream_reader = async {
                            let mut buf = BytesMut::with_capacity(buf_size);
                            loop {
                                buf.resize(buf_size, 0);
                                let n = match timeout(io_timeout, ro.read(&mut buf)).await {
                                    Ok(Ok(n)) => n,
                                    Ok(Err(e)) => return Err(e),
                                    Err(_) => return Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "read timeout")),
                                };
                                if n == 0 { break; }
                                let frame = Bytes::copy_from_slice(&buf[..n]);
                                let cl_bytes = adapter
                                    .to_client(frame)
                                    .await
                                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
                                if tx_down.send(cl_bytes).await.is_err() { break; }
                            }
                            Ok::<_, std::io::Error>(())
                        };

                        // writer: rx_down -> client
                        let writer_down = async {
                            while let Some(b) = rx_down.recv().await {
                                timeout(io_timeout, wi.write_all(&b)).await??;
                            }
                            Ok::<_, std::io::Error>(())
                        };

                        if let Err(e) = tokio::try_join!(reader, writer_up, upstream_reader, writer_down) {
                            warn!(error=%e, "pipe terminated");
                        }
                        qmetrics::dec_connections();
                    }
                    Ok(Err(e)) => error!(error=%e, "connect upstream failed"),
                    Err(_) => error!("connect upstream timed out"),
                }
            });
            qmetrics::inc_connections();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    struct EchoAdapter;
    #[async_trait::async_trait]
    impl L7Adapter for EchoAdapter {
        async fn to_upstream(&self, input: Bytes) -> Result<Bytes, adapter_sdk::AdapterError> { Ok(input) }
        async fn to_client(&self, input: Bytes) -> Result<Bytes, adapter_sdk::AdapterError> { Ok(input) }
        fn name(&self) -> &'static str { "echo" }
    }

    #[tokio::test]
    async fn echo_round_trip() {
        // upstream echo server (slow)
        let upstream_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let upstream_addr = upstream_listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let (mut s, _) = upstream_listener.accept().await.unwrap();
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 1024];
                    loop {
                        let n = s.read(&mut buf).await.unwrap();
                        if n == 0 { break; }
                        tokio::time::sleep(Duration::from_millis(1)).await; // backpressure simulation
                        s.write_all(&buf[..n]).await.unwrap();
                    }
                });
            }
        });

        let bridge_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let listen_addr = bridge_listener.local_addr().unwrap();
        let adapter = Arc::new(EchoAdapter);
        let bridge = Bridge {
            listen_addr: listen_addr.to_string(),
            upstream: UpstreamSelector::Static(upstream_addr.to_string()),
            adapter,
            buf_size: 16 * 1024,
            chan_capacity: 8,
            io_timeout: Duration::from_secs(2),
        };

        tokio::spawn(async move {
            // move ownership of listener to run loop by rebinding to the address
            let _ = bridge.run().await;
        });

        // client side
        let mut client = TcpStream::connect(listen_addr).await.unwrap();
        let payload = vec![1u8; 4 * 1024];
        client.write_all(&payload).await.unwrap();
        let mut got = vec![0u8; payload.len()];
        client.read_exact(&mut got).await.unwrap();
        assert_eq!(payload, got);
    }
}
