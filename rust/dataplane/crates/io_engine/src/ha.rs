use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::net::TcpStream;
use tokio::time::{timeout, Duration, Instant};
use tracing::{info, warn};

pub mod cluster;

pub use cluster::{HACluster, ClusterConfig, ClusterEvent, ClusterStatus, NodeRole, NodeState};

pub struct ActiveStandby {
    primary: String,
    standby: String,
    is_primary_healthy: AtomicBool,
    failover_instant: tokio::sync::RwLock<Option<Instant>>,
    check_interval: Duration,
    timeout: Duration,
}

impl ActiveStandby {
    pub fn new(primary: String, standby: String, check_interval: Duration, timeout: Duration) -> Arc<Self> {
        let me = Arc::new(Self {
            primary,
            standby,
            is_primary_healthy: AtomicBool::new(true),
            failover_instant: tokio::sync::RwLock::new(None),
            check_interval,
            timeout,
        });
        let me2 = Arc::clone(&me);
        tokio::spawn(async move { me2.monitor().await; });
        me
    }

    pub fn current(&self) -> &str {
        if self.is_primary_healthy.load(Ordering::Relaxed) { &self.primary } else { &self.standby }
    }

    pub async fn get_failover_time(&self) -> Option<Instant> {
        *self.failover_instant.read().await
    }

    pub fn is_healthy(&self) -> bool {
        self.is_primary_healthy.load(Ordering::Relaxed)
    }

    async fn monitor(self: Arc<Self>) {
        loop {
            let healthy = timeout(self.timeout, TcpStream::connect(&self.primary)).await.is_ok();
            let was_healthy = self.is_primary_healthy.swap(healthy, Ordering::Relaxed);
            if was_healthy && !healthy {
                let mut g = self.failover_instant.write().await;
                *g = Some(Instant::now());
                warn!("primary unhealthy; failing over to standby");
            }
            if !was_healthy && healthy {
                let mut g = self.failover_instant.write().await;
                *g = None;
                info!("primary healthy again; failing back");
            }
            tokio::time::sleep(self.check_interval).await;
        }
    }
}

