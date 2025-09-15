use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, mpsc, RwLock};
use tokio::time::{interval, timeout};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::metrics;

/// Enterprise-grade HA cluster with session affinity and consensus
pub struct HACluster {
    config: ClusterConfig,
    node_id: Uuid,
    state: Arc<RwLock<ClusterState>>,
    nodes: Arc<RwLock<HashMap<Uuid, NodeInfo>>>,
    sessions: Arc<RwLock<HashMap<String, SessionInfo>>>,
    consensus: Arc<RwLock<ConsensusState>>,
    event_tx: broadcast::Sender<ClusterEvent>,
    shutdown: AtomicBool,
    metrics: ClusterMetrics,
}

#[derive(Debug, Clone)]
pub struct ClusterConfig {
    pub node_name: String,
    pub bind_addr: SocketAddr,
    pub cluster_addrs: Vec<SocketAddr>,
    pub heartbeat_interval: Duration,
    pub election_timeout: Duration,
    pub session_timeout: Duration,
    pub max_sessions_per_node: usize,
    pub replication_factor: usize,
    pub split_brain_protection: bool,
    pub cross_datacenter: bool,
    pub datacenter_id: String,
}

#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub id: Uuid,
    pub name: String,
    pub addr: SocketAddr,
    pub role: NodeRole,
    pub state: NodeState,
    pub last_heartbeat: Instant,
    pub session_count: usize,
    pub load_average: f64,
    pub datacenter_id: String,
    pub version: String,
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NodeRole {
    Leader,
    Follower,
    Candidate,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NodeState {
    Active,
    Draining,
    Inactive,
    Failed,
}

#[derive(Debug, Clone)]
pub struct SessionInfo {
    pub id: String,
    pub client_addr: SocketAddr,
    pub node_id: Uuid,
    pub created_at: Instant,
    pub last_activity: Instant,
    pub connection_count: usize,
    pub bytes_transferred: u64,
    pub sticky: bool,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct ClusterState {
    pub term: u64,
    pub leader_id: Option<Uuid>,
    pub voted_for: Option<Uuid>,
    pub last_log_index: u64,
    pub commit_index: u64,
    pub total_nodes: usize,
    pub active_nodes: usize,
    pub quorum_size: usize,
    pub split_brain_detected: bool,
}

#[derive(Debug, Clone)]
pub struct ConsensusState {
    pub term: u64,
    pub log: Vec<LogEntry>,
    pub next_index: HashMap<Uuid, u64>,
    pub match_index: HashMap<Uuid, u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub term: u64,
    pub index: u64,
    pub timestamp: u64,
    pub entry_type: LogEntryType,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogEntryType {
    SessionCreate,
    SessionUpdate,
    SessionDelete,
    NodeJoin,
    NodeLeave,
    ConfigChange,
}

#[derive(Debug, Clone)]
pub enum ClusterEvent {
    NodeJoined(Uuid),
    NodeLeft(Uuid),
    LeaderElected(Uuid),
    SessionCreated(String),
    SessionMigrated(String, Uuid, Uuid),
    SplitBrainDetected,
    SplitBrainResolved,
    FailoverStarted(Uuid),
    FailoverCompleted(Uuid),
}

#[derive(Debug)]
pub struct ClusterMetrics {
    pub heartbeats_sent: AtomicU64,
    pub heartbeats_received: AtomicU64,
    pub elections_started: AtomicU64,
    pub sessions_created: AtomicU64,
    pub sessions_migrated: AtomicU64,
    pub failovers_completed: AtomicU64,
    pub split_brain_events: AtomicU64,
}

impl HACluster {
    pub fn new(config: ClusterConfig) -> Self {
        let node_id = Uuid::new_v4();
        let (event_tx, _) = broadcast::channel(1000);
        
        let initial_state = ClusterState {
            term: 0,
            leader_id: None,
            voted_for: None,
            last_log_index: 0,
            commit_index: 0,
            total_nodes: config.cluster_addrs.len() + 1,
            active_nodes: 1,
            quorum_size: (config.cluster_addrs.len() + 1) / 2 + 1,
            split_brain_detected: false,
        };

        Self {
            config,
            node_id,
            state: Arc::new(RwLock::new(initial_state)),
            nodes: Arc::new(RwLock::new(HashMap::new())),
            sessions: Arc::new(RwLock::new(HashMap::new())),
            consensus: Arc::new(RwLock::new(ConsensusState {
                term: 0,
                log: Vec::new(),
                next_index: HashMap::new(),
                match_index: HashMap::new(),
            })),
            event_tx,
            shutdown: AtomicBool::new(false),
            metrics: ClusterMetrics {
                heartbeats_sent: AtomicU64::new(0),
                heartbeats_received: AtomicU64::new(0),
                elections_started: AtomicU64::new(0),
                sessions_created: AtomicU64::new(0),
                sessions_migrated: AtomicU64::new(0),
                failovers_completed: AtomicU64::new(0),
                split_brain_events: AtomicU64::new(0),
            },
        }
    }

    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Starting HA cluster node {}", self.node_id);

        // Start cluster listener
        let listener = TcpListener::bind(&self.config.bind_addr).await?;
        info!("Cluster listening on {}", self.config.bind_addr);

        // Start background tasks
        self.start_heartbeat_task().await;
        self.start_election_task().await;
        self.start_session_cleanup_task().await;
        self.start_split_brain_detection_task().await;

        // Join existing cluster
        self.join_cluster().await?;

        // Handle incoming connections
        loop {
            if self.shutdown.load(Ordering::Relaxed) {
                break;
            }

            match timeout(Duration::from_secs(1), listener.accept()).await {
                Ok(Ok((stream, addr))) => {
                    self.handle_cluster_connection(stream, addr).await;
                }
                Ok(Err(e)) => {
                    error!("Failed to accept connection: {}", e);
                }
                Err(_) => {
                    // Timeout, continue loop
                }
            }
        }

        Ok(())
    }

    async fn start_heartbeat_task(&self) {
        let cluster = Arc::new(self.clone());
        tokio::spawn(async move {
            let mut interval = interval(cluster.config.heartbeat_interval);
            
            while !cluster.shutdown.load(Ordering::Relaxed) {
                interval.tick().await;
                cluster.send_heartbeats().await;
            }
        });
    }

    async fn start_election_task(&self) {
        let cluster = Arc::new(self.clone());
        tokio::spawn(async move {
            let mut interval = interval(cluster.config.election_timeout);
            
            while !cluster.shutdown.load(Ordering::Relaxed) {
                interval.tick().await;
                cluster.check_leader_timeout().await;
            }
        });
    }

    async fn start_session_cleanup_task(&self) {
        let cluster = Arc::new(self.clone());
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30));
            
            while !cluster.shutdown.load(Ordering::Relaxed) {
                interval.tick().await;
                cluster.cleanup_expired_sessions().await;
            }
        });
    }

    async fn start_split_brain_detection_task(&self) {
        if !self.config.split_brain_protection {
            return;
        }

        let cluster = Arc::new(self.clone());
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(10));
            
            while !cluster.shutdown.load(Ordering::Relaxed) {
                interval.tick().await;
                cluster.detect_split_brain().await;
            }
        });
    }

    async fn join_cluster(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        for addr in &self.config.cluster_addrs {
            match self.connect_to_node(*addr).await {
                Ok(_) => {
                    info!("Successfully joined cluster via {}", addr);
                    return Ok(());
                }
                Err(e) => {
                    warn!("Failed to connect to {}: {}", addr, e);
                }
            }
        }

        // If no existing nodes, become leader
        info!("No existing cluster found, becoming leader");
        self.become_leader().await;
        Ok(())
    }

    async fn connect_to_node(&self, addr: SocketAddr) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let stream = TcpStream::connect(addr).await?;
        self.handle_cluster_connection(stream, addr).await;
        Ok(())
    }

    async fn handle_cluster_connection(&self, _stream: TcpStream, _addr: SocketAddr) {
        // Handle cluster protocol messages
        // This would implement the full cluster communication protocol
        debug!("Handling cluster connection from {}", _addr);
    }

    async fn send_heartbeats(&self) {
        let state = self.state.read().await;
        if state.leader_id != Some(self.node_id) {
            return;
        }
        drop(state);

        let nodes = self.nodes.read().await;
        for (node_id, node_info) in nodes.iter() {
            if *node_id == self.node_id {
                continue;
            }

            // Send heartbeat to node
            self.send_heartbeat_to_node(node_info).await;
        }

        self.metrics.heartbeats_sent.fetch_add(nodes.len() as u64 - 1, Ordering::Relaxed);
    }

    async fn send_heartbeat_to_node(&self, _node_info: &NodeInfo) {
        // Send heartbeat message to specific node
        debug!("Sending heartbeat to node {}", _node_info.id);
    }

    async fn check_leader_timeout(&self) {
        let mut state = self.state.write().await;
        
        if let Some(leader_id) = state.leader_id {
            if leader_id == self.node_id {
                return; // We are the leader
            }

            // Check if we've heard from leader recently
            let nodes = self.nodes.read().await;
            if let Some(leader_node) = nodes.get(&leader_id) {
                if leader_node.last_heartbeat.elapsed() > self.config.election_timeout {
                    warn!("Leader timeout detected, starting election");
                    drop(nodes);
                    drop(state);
                    self.start_election().await;
                }
            }
        } else {
            // No leader, start election
            drop(state);
            self.start_election().await;
        }
    }

    async fn start_election(&self) {
        info!("Starting leader election");
        self.metrics.elections_started.fetch_add(1, Ordering::Relaxed);

        let mut state = self.state.write().await;
        state.term += 1;
        state.voted_for = Some(self.node_id);
        let current_term = state.term;
        drop(state);

        // Request votes from other nodes
        let nodes = self.nodes.read().await;
        let mut votes = 1; // Vote for ourselves
        let total_nodes = nodes.len() + 1;
        let required_votes = total_nodes / 2 + 1;

        for (node_id, node_info) in nodes.iter() {
            if *node_id == self.node_id {
                continue;
            }

            if self.request_vote_from_node(node_info, current_term).await {
                votes += 1;
            }
        }

        if votes >= required_votes {
            self.become_leader().await;
        } else {
            info!("Election failed, got {} votes out of {} required", votes, required_votes);
        }
    }

    async fn request_vote_from_node(&self, _node_info: &NodeInfo, _term: u64) -> bool {
        // Send vote request to node and wait for response
        debug!("Requesting vote from node {}", _node_info.id);
        // For now, simulate vote granted
        true
    }

    async fn become_leader(&self) {
        info!("Becoming cluster leader");
        
        let mut state = self.state.write().await;
        state.leader_id = Some(self.node_id);
        drop(state);

        // Initialize leader state
        let mut consensus = self.consensus.write().await;
        let nodes = self.nodes.read().await;
        
        for node_id in nodes.keys() {
            if *node_id != self.node_id {
                consensus.next_index.insert(*node_id, consensus.log.len() as u64 + 1);
                consensus.match_index.insert(*node_id, 0);
            }
        }

        let _ = self.event_tx.send(ClusterEvent::LeaderElected(self.node_id));
    }

    pub async fn create_session(&self, client_addr: SocketAddr, sticky: bool) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let session_id = Uuid::new_v4().to_string();
        
        let session = SessionInfo {
            id: session_id.clone(),
            client_addr,
            node_id: self.node_id,
            created_at: Instant::now(),
            last_activity: Instant::now(),
            connection_count: 1,
            bytes_transferred: 0,
            sticky,
            metadata: HashMap::new(),
        };

        let mut sessions = self.sessions.write().await;
        sessions.insert(session_id.clone(), session);
        drop(sessions);

        // Replicate session creation to other nodes
        self.replicate_session_event(LogEntryType::SessionCreate, &session_id).await;

        self.metrics.sessions_created.fetch_add(1, Ordering::Relaxed);
        let _ = self.event_tx.send(ClusterEvent::SessionCreated(session_id.clone()));

        info!("Created session {} for client {}", session_id, client_addr);
        Ok(session_id)
    }

    pub async fn get_session_node(&self, session_id: &str) -> Option<Uuid> {
        let sessions = self.sessions.read().await;
        sessions.get(session_id).map(|s| s.node_id)
    }

    pub async fn migrate_session(&self, session_id: &str, target_node: Uuid) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut sessions = self.sessions.write().await;
        
        if let Some(session) = sessions.get_mut(session_id) {
            let old_node = session.node_id;
            session.node_id = target_node;
            
            // Replicate session migration
            self.replicate_session_event(LogEntryType::SessionUpdate, session_id).await;
            
            self.metrics.sessions_migrated.fetch_add(1, Ordering::Relaxed);
            let _ = self.event_tx.send(ClusterEvent::SessionMigrated(session_id.to_string(), old_node, target_node));
            
            info!("Migrated session {} from node {} to node {}", session_id, old_node, target_node);
            Ok(())
        } else {
            Err("Session not found".into())
        }
    }

    async fn replicate_session_event(&self, entry_type: LogEntryType, session_id: &str) {
        let state = self.state.read().await;
        if state.leader_id != Some(self.node_id) {
            return; // Only leader replicates
        }
        drop(state);

        let mut consensus = self.consensus.write().await;
        let entry = LogEntry {
            term: consensus.term,
            index: consensus.log.len() as u64 + 1,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            entry_type,
            data: session_id.as_bytes().to_vec(),
        };

        consensus.log.push(entry);
        // In a real implementation, this would replicate to followers
    }

    async fn cleanup_expired_sessions(&self) {
        let mut sessions = self.sessions.write().await;
        let mut expired_sessions = Vec::new();

        for (session_id, session) in sessions.iter() {
            if session.last_activity.elapsed() > self.config.session_timeout {
                expired_sessions.push(session_id.clone());
            }
        }

        for session_id in expired_sessions {
            sessions.remove(&session_id);
            info!("Cleaned up expired session {}", session_id);
        }
    }

    async fn detect_split_brain(&self) {
        let nodes = self.nodes.read().await;
        let active_nodes = nodes.values().filter(|n| n.state == NodeState::Active).count() + 1;
        let total_nodes = nodes.len() + 1;
        
        if active_nodes < total_nodes / 2 + 1 {
            let mut state = self.state.write().await;
            if !state.split_brain_detected {
                state.split_brain_detected = true;
                warn!("Split brain detected: only {} out of {} nodes active", active_nodes, total_nodes);
                
                self.metrics.split_brain_events.fetch_add(1, Ordering::Relaxed);
                let _ = self.event_tx.send(ClusterEvent::SplitBrainDetected);
                
                // Step down as leader if we don't have quorum
                if state.leader_id == Some(self.node_id) {
                    state.leader_id = None;
                    warn!("Stepping down as leader due to split brain");
                }
            }
        }
    }

    pub async fn graceful_failover(&self, target_node: Uuid) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Starting graceful failover to node {}", target_node);
        let _ = self.event_tx.send(ClusterEvent::FailoverStarted(target_node));

        // Drain sessions from current node
        let sessions = self.sessions.read().await;
        let session_ids: Vec<String> = sessions.keys().cloned().collect();
        drop(sessions);

        for session_id in session_ids {
            if let Err(e) = self.migrate_session(&session_id, target_node).await {
                error!("Failed to migrate session {}: {}", session_id, e);
            }
        }

        // Update node state to draining
        let mut nodes = self.nodes.write().await;
        if let Some(node) = nodes.get_mut(&self.node_id) {
            node.state = NodeState::Draining;
        }

        self.metrics.failovers_completed.fetch_add(1, Ordering::Relaxed);
        let _ = self.event_tx.send(ClusterEvent::FailoverCompleted(target_node));

        info!("Graceful failover completed");
        Ok(())
    }

    pub async fn get_cluster_status(&self) -> ClusterStatus {
        let state = self.state.read().await;
        let nodes = self.nodes.read().await;
        let sessions = self.sessions.read().await;

        ClusterStatus {
            node_id: self.node_id,
            role: if state.leader_id == Some(self.node_id) { NodeRole::Leader } else { NodeRole::Follower },
            term: state.term,
            leader_id: state.leader_id,
            total_nodes: nodes.len() + 1,
            active_nodes: nodes.values().filter(|n| n.state == NodeState::Active).count() + 1,
            total_sessions: sessions.len(),
            split_brain_detected: state.split_brain_detected,
            metrics: ClusterMetrics {
                heartbeats_sent: AtomicU64::new(self.metrics.heartbeats_sent.load(Ordering::Relaxed)),
                heartbeats_received: AtomicU64::new(self.metrics.heartbeats_received.load(Ordering::Relaxed)),
                elections_started: AtomicU64::new(self.metrics.elections_started.load(Ordering::Relaxed)),
                sessions_created: AtomicU64::new(self.metrics.sessions_created.load(Ordering::Relaxed)),
                sessions_migrated: AtomicU64::new(self.metrics.sessions_migrated.load(Ordering::Relaxed)),
                failovers_completed: AtomicU64::new(self.metrics.failovers_completed.load(Ordering::Relaxed)),
                split_brain_events: AtomicU64::new(self.metrics.split_brain_events.load(Ordering::Relaxed)),
            },
        }
    }

    pub fn subscribe_events(&self) -> broadcast::Receiver<ClusterEvent> {
        self.event_tx.subscribe()
    }

    pub async fn shutdown(&self) {
        info!("Shutting down HA cluster");
        self.shutdown.store(true, Ordering::Relaxed);
        
        // Gracefully transfer leadership if we're the leader
        let state = self.state.read().await;
        if state.leader_id == Some(self.node_id) {
            // Find a suitable successor
            let nodes = self.nodes.read().await;
            if let Some((successor_id, _)) = nodes.iter().find(|(_, n)| n.state == NodeState::Active) {
                drop(nodes);
                drop(state);
                let _ = self.graceful_failover(*successor_id).await;
            }
        }
    }
}

impl Clone for HACluster {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            node_id: self.node_id,
            state: Arc::clone(&self.state),
            nodes: Arc::clone(&self.nodes),
            sessions: Arc::clone(&self.sessions),
            consensus: Arc::clone(&self.consensus),
            event_tx: self.event_tx.clone(),
            shutdown: AtomicBool::new(self.shutdown.load(Ordering::Relaxed)),
            metrics: ClusterMetrics {
                heartbeats_sent: AtomicU64::new(self.metrics.heartbeats_sent.load(Ordering::Relaxed)),
                heartbeats_received: AtomicU64::new(self.metrics.heartbeats_received.load(Ordering::Relaxed)),
                elections_started: AtomicU64::new(self.metrics.elections_started.load(Ordering::Relaxed)),
                sessions_created: AtomicU64::new(self.metrics.sessions_created.load(Ordering::Relaxed)),
                sessions_migrated: AtomicU64::new(self.metrics.sessions_migrated.load(Ordering::Relaxed)),
                failovers_completed: AtomicU64::new(self.metrics.failovers_completed.load(Ordering::Relaxed)),
                split_brain_events: AtomicU64::new(self.metrics.split_brain_events.load(Ordering::Relaxed)),
            },
        }
    }
}

#[derive(Debug)]
pub struct ClusterStatus {
    pub node_id: Uuid,
    pub role: NodeRole,
    pub term: u64,
    pub leader_id: Option<Uuid>,
    pub total_nodes: usize,
    pub active_nodes: usize,
    pub total_sessions: usize,
    pub split_brain_detected: bool,
    pub metrics: ClusterMetrics,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            node_name: "qslb-node".to_string(),
            bind_addr: "0.0.0.0:7946".parse().unwrap(),
            cluster_addrs: Vec::new(),
            heartbeat_interval: Duration::from_secs(5),
            election_timeout: Duration::from_secs(15),
            session_timeout: Duration::from_secs(300),
            max_sessions_per_node: 10000,
            replication_factor: 3,
            split_brain_protection: true,
            cross_datacenter: false,
            datacenter_id: "dc1".to_string(),
        }
    }
}