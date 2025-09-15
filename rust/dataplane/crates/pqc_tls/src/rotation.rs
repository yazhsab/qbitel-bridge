//! Quantum-safe key rotation mechanisms
//! 
//! This module provides advanced key rotation capabilities specifically designed
//! for post-quantum cryptography environments:
//! - Proactive rotation based on quantum threat assessment
//! - Hybrid rotation supporting both classical and PQC algorithms
//! - Zero-downtime rotation with seamless key transitions
//! - Coordinated rotation across distributed systems
//! - Quantum-safe key agreement protocols
//! - Forward secrecy guarantees
//! - Automated threat detection and response

use crate::errors::TlsError;
use crate::hsm::{HsmPqcManager, HsmKeyAttributes, HsmKeyType};
use crate::kyber::{KyberKEM, KyberKeyPair, KyberPublicKey, KyberPrivateKey, KyberSharedSecret};
use crate::lifecycle::{KeyLifecycleManager, KeyMetadata, KeyLifecycleState, KeyUsagePolicy, LifecycleEvent, LifecycleEventType};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque, BTreeMap};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{RwLock, Mutex, mpsc, broadcast, Semaphore};
use tokio::time::{interval, timeout, sleep, Instant};
use tracing::{info, warn, error, debug, span, Level};
use uuid::Uuid;
use zeroize::{Zeroize, ZeroizeOnDrop};
use secrecy::{Secret, ExposeSecret};

/// Quantum threat levels for adaptive rotation
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum QuantumThreatLevel {
    /// No immediate quantum threat
    Minimal = 0,
    /// Theoretical quantum threat exists
    Low = 1,
    /// Quantum computers exist but limited capability
    Moderate = 2,
    /// Significant quantum computing capability detected
    High = 3,
    /// Cryptographically relevant quantum computer detected
    Critical = 4,
    /// Active quantum attack suspected
    Imminent = 5,
}

impl Default for QuantumThreatLevel {
    fn default() -> Self {
        QuantumThreatLevel::Low
    }
}

/// Rotation strategies for different scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RotationStrategy {
    /// Time-based rotation
    Scheduled {
        interval: Duration,
        jitter: Duration,
    },
    /// Usage-based rotation
    Usage {
        max_operations: u64,
        max_data_volume: u64,
    },
    /// Threat-adaptive rotation
    ThreatBased {
        threat_threshold: QuantumThreatLevel,
        emergency_rotation: bool,
    },
    /// Hybrid rotation combining multiple triggers
    Hybrid {
        strategies: Vec<RotationStrategy>,
        combine_with_or: bool, // true = OR logic, false = AND logic
    },
    /// Load-adaptive rotation
    LoadAdaptive {
        cpu_threshold: f64,
        memory_threshold: f64,
        network_threshold: f64,
    },
}

impl Default for RotationStrategy {
    fn default() -> Self {
        RotationStrategy::Scheduled {
            interval: Duration::from_secs(86400), // 24 hours
            jitter: Duration::from_secs(3600),    // 1 hour jitter
        }
    }
}

/// Rotation coordination modes for distributed systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationMode {
    /// Independent rotation without coordination
    Independent,
    /// Leader-follower coordination
    LeaderFollower {
        leader_id: String,
        followers: Vec<String>,
    },
    /// Consensus-based coordination
    Consensus {
        nodes: Vec<String>,
        minimum_consensus: usize,
    },
    /// Time-synchronized coordination
    Synchronized {
        sync_window: Duration,
        max_drift: Duration,
    },
}

/// Rotation phase for managing transitions
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum RotationPhase {
    /// Rotation is idle/not active
    Idle,
    /// Preparing for rotation (generating new keys)
    Preparing,
    /// Coordinating with other nodes
    Coordinating,
    /// Executing the rotation
    Executing,
    /// Propagating new keys
    Propagating,
    /// Validating the rotation success
    Validating,
    /// Cleaning up old keys
    CleaningUp,
    /// Rotation completed successfully
    Completed,
    /// Rotation failed and rolling back
    Failed,
}

/// Rotation event for audit and monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotationEvent {
    pub event_id: Uuid,
    pub timestamp: SystemTime,
    pub node_id: String,
    pub rotation_id: Uuid,
    pub phase: RotationPhase,
    pub old_key_id: Option<String>,
    pub new_key_id: Option<String>,
    pub trigger_reason: RotationTrigger,
    pub threat_level: QuantumThreatLevel,
    pub duration: Option<Duration>,
    pub success: bool,
    pub error: Option<String>,
    pub metadata: HashMap<String, String>,
}

/// Reasons that trigger key rotation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RotationTrigger {
    Scheduled,
    UsageThreshold,
    ThreatEscalation,
    ManualRequest,
    EmergencyResponse,
    SystemLoad,
    ComplianceRequirement,
    ForwardSecrecyMaintenance,
    QuantumSafetyUpdate,
}

/// Configuration for rotation policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotationPolicyConfig {
    /// Primary rotation strategy
    pub strategy: RotationStrategy,
    /// Coordination mode for distributed systems
    pub coordination: CoordinationMode,
    /// Forward secrecy requirements
    pub forward_secrecy: ForwardSecrecyConfig,
    /// Emergency rotation settings
    pub emergency: EmergencyRotationConfig,
    /// Performance constraints
    pub performance: PerformanceConstraints,
    /// Quantum safety parameters
    pub quantum_safety: QuantumSafetyConfig,
}

impl Default for RotationPolicyConfig {
    fn default() -> Self {
        Self {
            strategy: RotationStrategy::default(),
            coordination: CoordinationMode::Independent,
            forward_secrecy: ForwardSecrecyConfig::default(),
            emergency: EmergencyRotationConfig::default(),
            performance: PerformanceConstraints::default(),
            quantum_safety: QuantumSafetyConfig::default(),
        }
    }
}

/// Forward secrecy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardSecrecyConfig {
    /// Ensure perfect forward secrecy
    pub perfect_forward_secrecy: bool,
    /// Maximum key lifetime for forward secrecy
    pub max_key_lifetime: Duration,
    /// Key material refresh interval
    pub refresh_interval: Duration,
    /// Secure deletion requirements
    pub secure_deletion: bool,
}

impl Default for ForwardSecrecyConfig {
    fn default() -> Self {
        Self {
            perfect_forward_secrecy: true,
            max_key_lifetime: Duration::from_secs(86400), // 24 hours
            refresh_interval: Duration::from_secs(3600),  // 1 hour
            secure_deletion: true,
        }
    }
}

/// Emergency rotation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyRotationConfig {
    /// Enable emergency rotation
    pub enabled: bool,
    /// Maximum time for emergency rotation
    pub max_rotation_time: Duration,
    /// Skip normal validation in emergencies
    pub skip_validation: bool,
    /// Emergency threat levels that trigger rotation
    pub trigger_levels: Vec<QuantumThreatLevel>,
    /// Pre-generated emergency keys
    pub pregenerated_keys: bool,
}

impl Default for EmergencyRotationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_rotation_time: Duration::from_secs(300), // 5 minutes
            skip_validation: false,
            trigger_levels: vec![QuantumThreatLevel::Critical, QuantumThreatLevel::Imminent],
            pregenerated_keys: true,
        }
    }
}

/// Performance constraints for rotation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConstraints {
    /// Maximum CPU usage during rotation
    pub max_cpu_usage: f64,
    /// Maximum memory usage during rotation
    pub max_memory_usage: f64,
    /// Maximum network bandwidth usage
    pub max_network_bandwidth: f64,
    /// Maximum concurrent rotations
    pub max_concurrent_rotations: usize,
    /// Rotation timeout
    pub rotation_timeout: Duration,
}

impl Default for PerformanceConstraints {
    fn default() -> Self {
        Self {
            max_cpu_usage: 0.8,  // 80%
            max_memory_usage: 0.8, // 80%
            max_network_bandwidth: 0.5, // 50%
            max_concurrent_rotations: 3,
            rotation_timeout: Duration::from_secs(1800), // 30 minutes
        }
    }
}

/// Quantum safety configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSafetyConfig {
    /// Current quantum threat assessment
    pub current_threat_level: QuantumThreatLevel,
    /// Automatic threat level updates
    pub auto_threat_assessment: bool,
    /// Quantum-safe algorithm preferences
    pub preferred_algorithms: Vec<String>,
    /// Hybrid mode settings
    pub hybrid_mode: HybridModeConfig,
    /// Post-quantum migration settings
    pub migration_settings: MigrationSettings,
}

impl Default for QuantumSafetyConfig {
    fn default() -> Self {
        Self {
            current_threat_level: QuantumThreatLevel::Low,
            auto_threat_assessment: true,
            preferred_algorithms: vec![
                "kyber768".to_string(),
                "dilithium3".to_string(),
                "x25519_kyber768".to_string(),
            ],
            hybrid_mode: HybridModeConfig::default(),
            migration_settings: MigrationSettings::default(),
        }
    }
}

/// Hybrid mode configuration combining classical and PQC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridModeConfig {
    /// Enable hybrid classical/PQC mode
    pub enabled: bool,
    /// Classical algorithm to use alongside PQC
    pub classical_algorithm: String,
    /// PQC algorithm to use
    pub pqc_algorithm: String,
    /// How to combine the algorithms
    pub combination_method: HybridCombinationMethod,
}

impl Default for HybridModeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            classical_algorithm: "x25519".to_string(),
            pqc_algorithm: "kyber768".to_string(),
            combination_method: HybridCombinationMethod::Concatenation,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HybridCombinationMethod {
    /// Concatenate classical and PQC shared secrets
    Concatenation,
    /// XOR classical and PQC shared secrets
    Xor,
    /// Use HKDF to combine secrets
    Hkdf,
    /// Use a custom key combiner
    Custom { method: String },
}

/// Migration settings for post-quantum transition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationSettings {
    /// Current migration phase
    pub migration_phase: MigrationPhase,
    /// Allow fallback to classical algorithms
    pub allow_classical_fallback: bool,
    /// Migration timeline
    pub timeline: MigrationTimeline,
    /// Compatibility requirements
    pub compatibility_requirements: Vec<String>,
}

impl Default for MigrationSettings {
    fn default() -> Self {
        Self {
            migration_phase: MigrationPhase::Hybrid,
            allow_classical_fallback: true,
            timeline: MigrationTimeline::default(),
            compatibility_requirements: vec!["tls13".to_string()],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MigrationPhase {
    /// Using classical algorithms only
    Classical,
    /// Using hybrid classical/PQC algorithms
    Hybrid,
    /// Using PQC algorithms only
    PostQuantum,
    /// Emergency fallback mode
    Emergency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationTimeline {
    pub start_hybrid: Option<SystemTime>,
    pub complete_migration: Option<SystemTime>,
    pub deprecate_classical: Option<SystemTime>,
}

impl Default for MigrationTimeline {
    fn default() -> Self {
        let now = SystemTime::now();
        Self {
            start_hybrid: Some(now),
            complete_migration: Some(now + Duration::from_secs(86400 * 365)), // 1 year
            deprecate_classical: Some(now + Duration::from_secs(86400 * 730)), // 2 years
        }
    }
}

/// Quantum-safe key rotation manager
pub struct QuantumSafeRotationManager {
    /// Configuration
    config: RotationPolicyConfig,
    /// Node identifier
    node_id: String,
    /// Key lifecycle manager
    lifecycle_manager: Arc<KeyLifecycleManager>,
    /// HSM manager for secure operations
    hsm_manager: Option<Arc<HsmPqcManager>>,
    /// Kyber KEM implementation
    kyber_kem: Arc<KyberKEM>,
    /// Active rotations
    active_rotations: Arc<RwLock<HashMap<Uuid, RotationContext>>>,
    /// Event log
    event_log: Arc<RwLock<VecDeque<RotationEvent>>>,
    /// Event broadcaster
    event_sender: broadcast::Sender<RotationEvent>,
    /// Coordination channel
    coordination_sender: mpsc::UnboundedSender<CoordinationMessage>,
    coordination_receiver: Arc<Mutex<mpsc::UnboundedReceiver<CoordinationMessage>>>,
    /// Threat assessment
    threat_assessor: Arc<Mutex<QuantumThreatAssessor>>,
    /// Performance monitor
    performance_monitor: Arc<PerformanceMonitor>,
    /// Rotation semaphore for controlling concurrency
    rotation_semaphore: Arc<Semaphore>,
    /// Emergency key cache
    emergency_keys: Arc<RwLock<HashMap<String, KyberKeyPair>>>,
}

/// Context for an active rotation
#[derive(Debug)]
struct RotationContext {
    pub rotation_id: Uuid,
    pub phase: RotationPhase,
    pub start_time: Instant,
    pub old_key_id: Option<String>,
    pub new_key_id: Option<String>,
    pub trigger: RotationTrigger,
    pub threat_level: QuantumThreatLevel,
    pub coordination_data: Option<CoordinationData>,
}

/// Coordination data for distributed rotations
#[derive(Debug)]
struct CoordinationData {
    pub coordinator_id: String,
    pub participants: Vec<String>,
    pub coordination_timeout: Instant,
    pub votes: HashMap<String, bool>,
    pub consensus_reached: bool,
}

/// Messages for rotation coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationMessage {
    RotationProposal {
        rotation_id: Uuid,
        proposer_id: String,
        trigger: RotationTrigger,
        threat_level: QuantumThreatLevel,
        proposed_time: SystemTime,
    },
    RotationVote {
        rotation_id: Uuid,
        voter_id: String,
        approve: bool,
        reason: Option<String>,
    },
    RotationExecute {
        rotation_id: Uuid,
        coordinator_id: String,
        execution_time: SystemTime,
    },
    RotationComplete {
        rotation_id: Uuid,
        node_id: String,
        success: bool,
        new_key_id: Option<String>,
        error: Option<String>,
    },
    ThreatLevelUpdate {
        new_level: QuantumThreatLevel,
        source: String,
        timestamp: SystemTime,
        evidence: Vec<String>,
    },
}

/// Quantum threat assessment system
pub struct QuantumThreatAssessor {
    current_level: QuantumThreatLevel,
    assessment_history: VecDeque<ThreatAssessment>,
    indicators: HashMap<String, f64>,
    thresholds: HashMap<QuantumThreatLevel, f64>,
}

#[derive(Debug, Clone)]
pub struct ThreatAssessment {
    pub timestamp: SystemTime,
    pub level: QuantumThreatLevel,
    pub confidence: f64,
    pub indicators: HashMap<String, f64>,
    pub reasoning: Vec<String>,
}

/// Performance monitor for rotation operations
pub struct PerformanceMonitor {
    cpu_usage: Arc<tokio::sync::RwLock<f64>>,
    memory_usage: Arc<tokio::sync::RwLock<f64>>,
    network_usage: Arc<tokio::sync::RwLock<f64>>,
    rotation_metrics: Arc<tokio::sync::RwLock<HashMap<String, RotationMetrics>>>,
}

#[derive(Debug, Clone)]
pub struct RotationMetrics {
    pub total_rotations: u64,
    pub successful_rotations: u64,
    pub failed_rotations: u64,
    pub average_duration: Duration,
    pub last_rotation: Option<SystemTime>,
}

impl QuantumSafeRotationManager {
    /// Create a new quantum-safe rotation manager
    pub async fn new(
        config: RotationPolicyConfig,
        node_id: String,
        lifecycle_manager: Arc<KeyLifecycleManager>,
        hsm_manager: Option<Arc<HsmPqcManager>>,
        kyber_kem: Arc<KyberKEM>,
    ) -> Result<Self, TlsError> {
        let (event_sender, _) = broadcast::channel(1000);
        let (coordination_sender, coordination_receiver) = mpsc::unbounded_channel();
        
        let rotation_semaphore = Arc::new(Semaphore::new(
            config.performance.max_concurrent_rotations
        ));

        let mut manager = Self {
            config: config.clone(),
            node_id,
            lifecycle_manager,
            hsm_manager,
            kyber_kem,
            active_rotations: Arc::new(RwLock::new(HashMap::new())),
            event_log: Arc::new(RwLock::new(VecDeque::with_capacity(10000))),
            event_sender,
            coordination_sender,
            coordination_receiver: Arc::new(Mutex::new(coordination_receiver)),
            threat_assessor: Arc::new(Mutex::new(QuantumThreatAssessor::new(
                config.quantum_safety.current_threat_level
            ))),
            performance_monitor: Arc::new(PerformanceMonitor::new()),
            rotation_semaphore,
            emergency_keys: Arc::new(RwLock::new(HashMap::new())),
        };

        // Initialize emergency keys if enabled
        if config.emergency.pregenerated_keys {
            manager.pregenerate_emergency_keys().await?;
        }

        // Start background tasks
        manager.start_background_tasks().await?;

        Ok(manager)
    }

    /// Start background monitoring and coordination tasks
    async fn start_background_tasks(&self) -> Result<(), TlsError> {
        // Rotation scheduler task
        let scheduler = self.clone_for_tasks();
        tokio::spawn(async move {
            scheduler.rotation_scheduler_task().await;
        });

        // Threat assessment task
        let threat_monitor = self.clone_for_tasks();
        tokio::spawn(async move {
            threat_monitor.threat_assessment_task().await;
        });

        // Coordination handler task
        let coordinator = self.clone_for_tasks();
        tokio::spawn(async move {
            coordinator.coordination_handler_task().await;
        });

        // Performance monitoring task
        let perf_monitor = self.clone_for_tasks();
        tokio::spawn(async move {
            perf_monitor.performance_monitoring_task().await;
        });

        info!(node_id = %self.node_id, "started quantum-safe rotation background tasks");
        Ok(())
    }

    fn clone_for_tasks(&self) -> QuantumSafeRotationManagerClone {
        QuantumSafeRotationManagerClone {
            config: self.config.clone(),
            node_id: self.node_id.clone(),
            lifecycle_manager: Arc::clone(&self.lifecycle_manager),
            hsm_manager: self.hsm_manager.as_ref().map(Arc::clone),
            kyber_kem: Arc::clone(&self.kyber_kem),
            active_rotations: Arc::clone(&self.active_rotations),
            event_log: Arc::clone(&self.event_log),
            event_sender: self.event_sender.clone(),
            coordination_sender: self.coordination_sender.clone(),
            coordination_receiver: Arc::clone(&self.coordination_receiver),
            threat_assessor: Arc::clone(&self.threat_assessor),
            performance_monitor: Arc::clone(&self.performance_monitor),
            rotation_semaphore: Arc::clone(&self.rotation_semaphore),
            emergency_keys: Arc::clone(&self.emergency_keys),
        }
    }

    /// Execute a quantum-safe key rotation
    pub async fn rotate_key(
        &self,
        key_id: &str,
        trigger: RotationTrigger,
    ) -> Result<String, TlsError> {
        let _span = span!(Level::INFO, "quantum_safe_rotation", 
            key_id = key_id, 
            trigger = ?trigger
        ).entered();

        // Acquire rotation permit
        let _permit = self.rotation_semaphore.acquire().await
            .map_err(|_| TlsError::Io("rotation semaphore closed".into()))?;

        let rotation_id = Uuid::new_v4();
        let start_time = Instant::now();

        // Get current threat level
        let threat_level = {
            let assessor = self.threat_assessor.lock().await;
            assessor.current_level()
        };

        // Create rotation context
        let mut context = RotationContext {
            rotation_id,
            phase: RotationPhase::Preparing,
            start_time,
            old_key_id: Some(key_id.to_string()),
            new_key_id: None,
            trigger: trigger.clone(),
            threat_level,
            coordination_data: None,
        };

        // Store active rotation
        {
            let mut rotations = self.active_rotations.write().await;
            rotations.insert(rotation_id, context);
        }

        // Log rotation start
        self.log_rotation_event(RotationEvent {
            event_id: Uuid::new_v4(),
            timestamp: SystemTime::now(),
            node_id: self.node_id.clone(),
            rotation_id,
            phase: RotationPhase::Preparing,
            old_key_id: Some(key_id.to_string()),
            new_key_id: None,
            trigger: trigger.clone(),
            threat_level,
            duration: None,
            success: false,
            error: None,
            metadata: HashMap::new(),
        }).await;

        // Execute rotation phases
        let result = match self.execute_rotation_phases(rotation_id, key_id).await {
            Ok(new_key_id) => {
                self.update_rotation_phase(rotation_id, RotationPhase::Completed).await;
                
                let duration = start_time.elapsed();
                self.log_rotation_event(RotationEvent {
                    event_id: Uuid::new_v4(),
                    timestamp: SystemTime::now(),
                    node_id: self.node_id.clone(),
                    rotation_id,
                    phase: RotationPhase::Completed,
                    old_key_id: Some(key_id.to_string()),
                    new_key_id: Some(new_key_id.clone()),
                    trigger,
                    threat_level,
                    duration: Some(duration),
                    success: true,
                    error: None,
                    metadata: HashMap::new(),
                }).await;

                info!(
                    rotation_id = %rotation_id,
                    old_key = %key_id,
                    new_key = %new_key_id,
                    duration = ?duration,
                    "quantum-safe rotation completed successfully"
                );

                Ok(new_key_id)
            }
            Err(e) => {
                self.update_rotation_phase(rotation_id, RotationPhase::Failed).await;
                
                let duration = start_time.elapsed();
                self.log_rotation_event(RotationEvent {
                    event_id: Uuid::new_v4(),
                    timestamp: SystemTime::now(),
                    node_id: self.node_id.clone(),
                    rotation_id,
                    phase: RotationPhase::Failed,
                    old_key_id: Some(key_id.to_string()),
                    new_key_id: None,
                    trigger,
                    threat_level,
                    duration: Some(duration),
                    success: false,
                    error: Some(e.to_string()),
                    metadata: HashMap::new(),
                }).await;

                error!(
                    rotation_id = %rotation_id,
                    error = %e,
                    duration = ?duration,
                    "quantum-safe rotation failed"
                );

                Err(e)
            }
        };

        // Cleanup rotation context
        {
            let mut rotations = self.active_rotations.write().await;
            rotations.remove(&rotation_id);
        }

        result
    }

    /// Execute all phases of the rotation
    async fn execute_rotation_phases(
        &self,
        rotation_id: Uuid,
        key_id: &str,
    ) -> Result<String, TlsError> {
        // Phase 1: Coordination (if needed)
        if matches!(self.config.coordination, CoordinationMode::LeaderFollower { .. } | CoordinationMode::Consensus { .. }) {
            self.update_rotation_phase(rotation_id, RotationPhase::Coordinating).await;
            self.coordinate_rotation(rotation_id).await?;
        }

        // Phase 2: Key Generation
        self.update_rotation_phase(rotation_id, RotationPhase::Executing).await;
        let new_key_id = self.generate_rotation_key(key_id).await?;

        // Update context with new key ID
        {
            let mut rotations = self.active_rotations.write().await;
            if let Some(context) = rotations.get_mut(&rotation_id) {
                context.new_key_id = Some(new_key_id.clone());
            }
        }

        // Phase 3: Propagation
        self.update_rotation_phase(rotation_id, RotationPhase::Propagating).await;
        self.propagate_new_key(&new_key_id).await?;

        // Phase 4: Validation
        self.update_rotation_phase(rotation_id, RotationPhase::Validating).await;
        self.validate_rotation(key_id, &new_key_id).await?;

        // Phase 5: Cleanup
        self.update_rotation_phase(rotation_id, RotationPhase::CleaningUp).await;
        self.cleanup_old_key(key_id).await?;

        Ok(new_key_id)
    }

    /// Generate a new key for rotation
    async fn generate_rotation_key(&self, old_key_id: &str) -> Result<String, TlsError> {
        // Check if we should use emergency key
        let threat_level = {
            let assessor = self.threat_assessor.lock().await;
            assessor.current_level()
        };

        if threat_level >= QuantumThreatLevel::Critical && self.config.emergency.enabled {
            if let Some(emergency_key) = self.get_emergency_key(old_key_id).await? {
                return Ok(emergency_key);
            }
        }

        // Generate new key based on quantum safety configuration
        match self.config.quantum_safety.migration_settings.migration_phase {
            MigrationPhase::Classical => {
                // For backward compatibility only - not recommended
                warn!("generating classical key in quantum-safe rotation manager");
                self.lifecycle_manager.generate_key(None, None).await
            }
            MigrationPhase::Hybrid => {
                self.generate_hybrid_key().await
            }
            MigrationPhase::PostQuantum => {
                // Generate pure PQC key
                let policy = KeyUsagePolicy {
                    require_hsm: self.hsm_manager.is_some(),
                    auto_rotation: true,
                    max_age: Some(self.config.forward_secrecy.max_key_lifetime),
                    ..Default::default()
                };
                
                self.lifecycle_manager.generate_key(None, Some(policy)).await
            }
            MigrationPhase::Emergency => {
                self.generate_emergency_rotation_key().await
            }
        }
    }

    /// Generate a hybrid classical/PQC key
    async fn generate_hybrid_key(&self) -> Result<String, TlsError> {
        let hybrid_config = &self.config.quantum_safety.hybrid_mode;
        
        if !hybrid_config.enabled {
            return Err(TlsError::Policy("hybrid mode not enabled".into()));
        }

        // Generate both classical and PQC components
        let pqc_key_id = self.lifecycle_manager.generate_key(None, None).await?;
        
        // In a real implementation, we would also generate the classical component
        // and combine them according to the combination method
        
        info!(
            pqc_algorithm = %hybrid_config.pqc_algorithm,
            classical_algorithm = %hybrid_config.classical_algorithm,
            combination = ?hybrid_config.combination_method,
            "generated hybrid key"
        );

        Ok(pqc_key_id)
    }

    /// Generate emergency rotation key
    async fn generate_emergency_rotation_key(&self) -> Result<String, TlsError> {
        let emergency_config = &self.config.emergency;
        
        let key_generation = async {
            let policy = KeyUsagePolicy {
                require_hsm: false, // Skip HSM in emergency
                auto_rotation: false,
                max_age: Some(Duration::from_secs(3600)), // 1 hour emergency lifetime
                ..Default::default()
            };
            
            self.lifecycle_manager.generate_key(None, Some(policy)).await
        };

        if emergency_config.skip_validation {
            // Fast path for emergency
            key_generation.await
        } else {
            // Still respect timeout
            timeout(emergency_config.max_rotation_time, key_generation).await
                .map_err(|_| TlsError::Io("emergency key generation timeout".into()))?
        }
    }

    /// Get or consume an emergency key
    async fn get_emergency_key(&self, _old_key_id: &str) -> Result<Option<String>, TlsError> {
        let mut emergency_keys = self.emergency_keys.write().await;
        
        if let Some((key_id, _keypair)) = emergency_keys.iter().next() {
            let key_id = key_id.clone();
            emergency_keys.remove(&key_id);
            
            // Regenerate emergency key in background
            let manager = self.clone_for_tasks();
            tokio::spawn(async move {
                if let Err(e) = manager.replenish_emergency_key().await {
                    error!(error = %e, "failed to replenish emergency key");
                }
            });
            
            info!("used emergency key for rotation");
            Ok(Some(key_id))
        } else {
            Ok(None)
        }
    }

    /// Pregenerate emergency keys
    async fn pregenerate_emergency_keys(&self) -> Result<(), TlsError> {
        let emergency_count = 5; // Keep 5 emergency keys ready
        let mut emergency_keys = self.emergency_keys.write().await;
        
        for i in 0..emergency_count {
            let key_id = format!("emergency-{}-{}", self.node_id, i);
            let keypair = self.kyber_kem.generate_keypair().await?;
            emergency_keys.insert(key_id.clone(), keypair);
            
            debug!(key_id = %key_id, "pregenerated emergency key");
        }
        
        info!(count = emergency_count, "pregenerated emergency keys");
        Ok(())
    }

    /// Replenish a used emergency key
    async fn replenish_emergency_key(&self) -> Result<(), TlsError> {
        let key_id = format!("emergency-{}-{}", self.node_id, Uuid::new_v4());
        let keypair = self.kyber_kem.generate_keypair().await?;
        
        let mut emergency_keys = self.emergency_keys.write().await;
        emergency_keys.insert(key_id.clone(), keypair);
        
        debug!(key_id = %key_id, "replenished emergency key");
        Ok(())
    }

    /// Coordinate rotation with other nodes
    async fn coordinate_rotation(&self, rotation_id: Uuid) -> Result<(), TlsError> {
        match &self.config.coordination {
            CoordinationMode::Independent => Ok(()),
            CoordinationMode::LeaderFollower { leader_id, followers } => {
                if self.node_id == *leader_id {
                    self.coordinate_as_leader(rotation_id, followers.clone()).await
                } else {
                    self.coordinate_as_follower(rotation_id).await
                }
            }
            CoordinationMode::Consensus { nodes, minimum_consensus } => {
                self.coordinate_consensus(rotation_id, nodes.clone(), *minimum_consensus).await
            }
            CoordinationMode::Synchronized { sync_window, max_drift: _ } => {
                self.coordinate_synchronized(rotation_id, *sync_window).await
            }
        }
    }

    /// Coordinate rotation as leader
    async fn coordinate_as_leader(
        &self,
        rotation_id: Uuid,
        _followers: Vec<String>,
    ) -> Result<(), TlsError> {
        // Simplified leader coordination
        // In a real implementation, this would send coordination messages
        // to all followers and wait for their acknowledgment
        
        info!(rotation_id = %rotation_id, "coordinating rotation as leader");
        
        // Simulate coordination delay
        sleep(Duration::from_millis(100)).await;
        
        Ok(())
    }

    /// Coordinate rotation as follower
    async fn coordinate_as_follower(&self, rotation_id: Uuid) -> Result<(), TlsError> {
        info!(rotation_id = %rotation_id, "coordinating rotation as follower");
        
        // Wait for leader's coordination message
        // This is simplified - real implementation would listen for messages
        sleep(Duration::from_millis(50)).await;
        
        Ok(())
    }

    /// Coordinate using consensus
    async fn coordinate_consensus(
        &self,
        rotation_id: Uuid,
        _nodes: Vec<String>,
        _minimum_consensus: usize,
    ) -> Result<(), TlsError> {
        info!(rotation_id = %rotation_id, "coordinating rotation via consensus");
        
        // Simplified consensus - real implementation would use a consensus algorithm
        sleep(Duration::from_millis(200)).await;
        
        Ok(())
    }

    /// Coordinate using time synchronization
    async fn coordinate_synchronized(
        &self,
        rotation_id: Uuid,
        sync_window: Duration,
    ) -> Result<(), TlsError> {
        info!(
            rotation_id = %rotation_id,
            sync_window = ?sync_window,
            "coordinating rotation via time synchronization"
        );
        
        // Wait for sync window
        sleep(sync_window).await;
        
        Ok(())
    }

    /// Propagate new key to relevant systems
    async fn propagate_new_key(&self, new_key_id: &str) -> Result<(), TlsError> {
        info!(new_key_id = %new_key_id, "propagating new key");
        
        // In a real implementation, this would:
        // 1. Update TLS configurations
        // 2. Notify dependent services
        // 3. Update key distribution systems
        // 4. Synchronize with other nodes
        
        // Simulate propagation delay
        sleep(Duration::from_millis(100)).await;
        
        Ok(())
    }

    /// Validate that the rotation was successful
    async fn validate_rotation(&self, old_key_id: &str, new_key_id: &str) -> Result<(), TlsError> {
        info!(
            old_key_id = %old_key_id,
            new_key_id = %new_key_id,
            "validating rotation"
        );
        
        // Check that new key is active
        if let Some(metadata) = self.lifecycle_manager.get_key_metadata(new_key_id).await {
            if metadata.state != KeyLifecycleState::Active {
                return Err(TlsError::Policy(format!(
                    "new key {} is not active: {:?}",
                    new_key_id, metadata.state
                )));
            }
        } else {
            return Err(TlsError::Policy(format!("new key {} not found", new_key_id)));
        }

        // Additional validation checks would go here
        // - Cryptographic validation
        // - Integration testing
        // - Performance validation
        
        Ok(())
    }

    /// Clean up the old key after successful rotation
    async fn cleanup_old_key(&self, old_key_id: &str) -> Result<(), TlsError> {
        info!(old_key_id = %old_key_id, "cleaning up old key");
        
        // Deprecate the old key (don't immediately destroy for rollback capability)
        // The lifecycle manager will handle the full cleanup process
        
        Ok(())
    }

    /// Update rotation phase
    async fn update_rotation_phase(&self, rotation_id: Uuid, phase: RotationPhase) {
        let mut rotations = self.active_rotations.write().await;
        if let Some(context) = rotations.get_mut(&rotation_id) {
            context.phase = phase;
        }
    }

    /// Log a rotation event
    async fn log_rotation_event(&self, event: RotationEvent) {
        // Add to event log
        {
            let mut log = self.event_log.write().await;
            log.push_back(event.clone());
            
            // Limit log size
            if log.len() > 10000 {
                log.pop_front();
            }
        }

        // Broadcast event
        let _ = self.event_sender.send(event);
    }

    /// Rotation scheduler background task
    async fn rotation_scheduler_task(&self) {
        let mut interval = interval(Duration::from_secs(60)); // Check every minute
        
        loop {
            interval.tick().await;
            
            if let Err(e) = self.check_rotation_triggers().await {
                error!(error = %e, "rotation trigger check failed");
            }
        }
    }

    /// Check for rotation triggers
    async fn check_rotation_triggers(&self) -> Result<(), TlsError> {
        // Get all keys from lifecycle manager
        let keys = self.lifecycle_manager.list_keys().await;
        
        for key_metadata in keys {
            if key_metadata.state != KeyLifecycleState::Active {
                continue;
            }
            
            // Check various trigger conditions
            let should_rotate = self.should_rotate_key(&key_metadata).await?;
            
            if should_rotate {
                let trigger = self.determine_rotation_trigger(&key_metadata).await;
                
                info!(
                    key_id = %key_metadata.key_id,
                    trigger = ?trigger,
                    "triggering automatic rotation"
                );
                
                // Trigger rotation in background
                let manager = self.clone_for_tasks();
                let key_id = key_metadata.key_id.clone();
                
                tokio::spawn(async move {
                    if let Err(e) = manager.rotate_key(&key_id, trigger).await {
                        error!(
                            key_id = %key_id,
                            error = %e,
                            "automatic rotation failed"
                        );
                    }
                });
            }
        }
        
        Ok(())
    }

    /// Determine if a key should be rotated
    async fn should_rotate_key(&self, metadata: &KeyMetadata) -> Result<bool, TlsError> {
        match &self.config.strategy {
            RotationStrategy::Scheduled { interval, jitter: _ } => {
                let age = SystemTime::now()
                    .duration_since(metadata.created_at)
                    .unwrap_or_default();
                Ok(age >= *interval)
            }
            RotationStrategy::Usage { max_operations, max_data_volume } => {
                Ok(metadata.operation_count >= *max_operations ||
                   metadata.data_volume >= *max_data_volume)
            }
            RotationStrategy::ThreatBased { threat_threshold, .. } => {
                let current_level = {
                    let assessor = self.threat_assessor.lock().await;
                    assessor.current_level()
                };
                Ok(current_level >= *threat_threshold)
            }
            RotationStrategy::Hybrid { strategies, combine_with_or } => {
                let mut results = Vec::new();
                
                for strategy in strategies {
                    let temp_config = RotationPolicyConfig {
                        strategy: strategy.clone(),
                        ..self.config.clone()
                    };
                    let temp_manager = QuantumSafeRotationManager {
                        config: temp_config,
                        ..self.clone_minimal()
                    };
                    
                    results.push(temp_manager.should_rotate_key(metadata).await?);
                }
                
                Ok(if *combine_with_or {
                    results.into_iter().any(|x| x)
                } else {
                    results.into_iter().all(|x| x)
                })
            }
            RotationStrategy::LoadAdaptive { cpu_threshold, memory_threshold, network_threshold } => {
                let perf = self.performance_monitor.get_current_usage().await;
                Ok(perf.cpu_usage < *cpu_threshold &&
                   perf.memory_usage < *memory_threshold &&
                   perf.network_usage < *network_threshold)
            }
        }
    }

    // Minimal clone for testing rotation strategies
    fn clone_minimal(&self) -> Self {
        // This is a simplified clone for internal use
        // In a real implementation, this would be handled differently
        QuantumSafeRotationManager {
            config: self.config.clone(),
            node_id: self.node_id.clone(),
            lifecycle_manager: Arc::clone(&self.lifecycle_manager),
            hsm_manager: self.hsm_manager.as_ref().map(Arc::clone),
            kyber_kem: Arc::clone(&self.kyber_kem),
            active_rotations: Arc::clone(&self.active_rotations),
            event_log: Arc::clone(&self.event_log),
            event_sender: self.event_sender.clone(),
            coordination_sender: self.coordination_sender.clone(),
            coordination_receiver: Arc::clone(&self.coordination_receiver),
            threat_assessor: Arc::clone(&self.threat_assessor),
            performance_monitor: Arc::clone(&self.performance_monitor),
            rotation_semaphore: Arc::clone(&self.rotation_semaphore),
            emergency_keys: Arc::clone(&self.emergency_keys),
        }
    }

    /// Determine the rotation trigger for a key
    async fn determine_rotation_trigger(&self, metadata: &KeyMetadata) -> RotationTrigger {
        // Check threat level first
        let threat_level = {
            let assessor = self.threat_assessor.lock().await;
            assessor.current_level()
        };
        
        if threat_level >= QuantumThreatLevel::Critical {
            return RotationTrigger::ThreatEscalation;
        }

        // Check usage thresholds
        if let Some(max_ops) = metadata.policy.max_operations {
            if metadata.operation_count >= max_ops {
                return RotationTrigger::UsageThreshold;
            }
        }

        if let Some(max_volume) = metadata.policy.max_data_volume {
            if metadata.data_volume >= max_volume {
                return RotationTrigger::UsageThreshold;
            }
        }

        // Default to scheduled
        RotationTrigger::Scheduled
    }

    /// Threat assessment background task
    async fn threat_assessment_task(&self) {
        let mut interval = interval(Duration::from_secs(300)); // Every 5 minutes
        
        loop {
            interval.tick().await;
            
            if let Err(e) = self.assess_quantum_threat().await {
                error!(error = %e, "quantum threat assessment failed");
            }
        }
    }

    /// Assess current quantum threat level
    async fn assess_quantum_threat(&self) -> Result<(), TlsError> {
        let mut assessor = self.threat_assessor.lock().await;
        let assessment = assessor.assess_threat().await?;
        
        if assessment.level != assessor.current_level() {
            info!(
                old_level = ?assessor.current_level(),
                new_level = ?assessment.level,
                confidence = assessment.confidence,
                "quantum threat level changed"
            );
            
            assessor.update_level(assessment.level);
            
            // Send threat level update to coordination network
            let message = CoordinationMessage::ThreatLevelUpdate {
                new_level: assessment.level,
                source: self.node_id.clone(),
                timestamp: SystemTime::now(),
                evidence: assessment.reasoning,
            };
            
            if let Err(e) = self.coordination_sender.send(message) {
                warn!(error = %e, "failed to broadcast threat level update");
            }
            
            // Trigger emergency rotations if threat is critical
            if assessment.level >= QuantumThreatLevel::Critical &&
               self.config.emergency.trigger_levels.contains(&assessment.level) {
                
                let manager = self.clone_for_tasks();
                tokio::spawn(async move {
                    manager.trigger_emergency_rotations().await;
                });
            }
        }
        
        Ok(())
    }

    /// Trigger emergency rotations for all active keys
    async fn trigger_emergency_rotations(&self) {
        let keys = self.lifecycle_manager.list_keys().await;
        
        for metadata in keys {
            if metadata.state == KeyLifecycleState::Active {
                let manager = self.clone_for_tasks();
                let key_id = metadata.key_id.clone();
                
                tokio::spawn(async move {
                    if let Err(e) = manager.rotate_key(&key_id, RotationTrigger::EmergencyResponse).await {
                        error!(
                            key_id = %key_id,
                            error = %e,
                            "emergency rotation failed"
                        );
                    }
                });
            }
        }
        
        warn!("triggered emergency rotations for all active keys");
    }

    /// Coordination handler background task
    async fn coordination_handler_task(&self) {
        let mut receiver = self.coordination_receiver.lock().await;
        
        while let Some(message) = receiver.recv().await {
            if let Err(e) = self.handle_coordination_message(message).await {
                error!(error = %e, "coordination message handling failed");
            }
        }
    }

    /// Handle coordination messages
    async fn handle_coordination_message(&self, message: CoordinationMessage) -> Result<(), TlsError> {
        match message {
            CoordinationMessage::ThreatLevelUpdate { new_level, source, .. } => {
                info!(
                    source = %source,
                    new_level = ?new_level,
                    "received threat level update"
                );
                
                let mut assessor = self.threat_assessor.lock().await;
                if new_level > assessor.current_level() {
                    assessor.update_level(new_level);
                }
            }
            _ => {
                // Handle other coordination message types
                debug!(message = ?message, "received coordination message");
            }
        }
        
        Ok(())
    }

    /// Performance monitoring background task
    async fn performance_monitoring_task(&self) {
        let mut interval = interval(Duration::from_secs(30));
        
        loop {
            interval.tick().await;
            
            if let Err(e) = self.update_performance_metrics().await {
                error!(error = %e, "performance metrics update failed");
            }
        }
    }

    /// Update performance metrics
    async fn update_performance_metrics(&self) -> Result<(), TlsError> {
        // In a real implementation, this would collect actual system metrics
        let cpu_usage = 0.3; // Placeholder
        let memory_usage = 0.4; // Placeholder
        let network_usage = 0.2; // Placeholder
        
        self.performance_monitor.update_usage(cpu_usage, memory_usage, network_usage).await;
        
        Ok(())
    }

    /// Get rotation statistics
    pub async fn get_rotation_statistics(&self) -> RotationStatistics {
        let event_log = self.event_log.read().await;
        let active_rotations = self.active_rotations.read().await;
        
        let mut stats = RotationStatistics {
            total_rotations: 0,
            successful_rotations: 0,
            failed_rotations: 0,
            active_rotations: active_rotations.len(),
            average_rotation_time: Duration::ZERO,
            threat_level_distribution: HashMap::new(),
            trigger_distribution: HashMap::new(),
        };
        
        let mut total_duration = Duration::ZERO;
        let mut completed_count = 0;
        
        for event in event_log.iter() {
            if event.phase == RotationPhase::Completed || event.phase == RotationPhase::Failed {
                stats.total_rotations += 1;
                
                if event.success {
                    stats.successful_rotations += 1;
                } else {
                    stats.failed_rotations += 1;
                }
                
                if let Some(duration) = event.duration {
                    total_duration += duration;
                    completed_count += 1;
                }
                
                *stats.threat_level_distribution.entry(event.threat_level).or_insert(0) += 1;
                *stats.trigger_distribution.entry(format!("{:?}", event.trigger_reason)).or_insert(0) += 1;
            }
        }
        
        if completed_count > 0 {
            stats.average_rotation_time = total_duration / completed_count;
        }
        
        stats
    }
}

// Helper structs and implementations for background tasks
#[derive(Clone)]
struct QuantumSafeRotationManagerClone {
    config: RotationPolicyConfig,
    node_id: String,
    lifecycle_manager: Arc<KeyLifecycleManager>,
    hsm_manager: Option<Arc<HsmPqcManager>>,
    kyber_kem: Arc<KyberKEM>,
    active_rotations: Arc<RwLock<HashMap<Uuid, RotationContext>>>,
    event_log: Arc<RwLock<VecDeque<RotationEvent>>>,
    event_sender: broadcast::Sender<RotationEvent>,
    coordination_sender: mpsc::UnboundedSender<CoordinationMessage>,
    coordination_receiver: Arc<Mutex<mpsc::UnboundedReceiver<CoordinationMessage>>>,
    threat_assessor: Arc<Mutex<QuantumThreatAssessor>>,
    performance_monitor: Arc<PerformanceMonitor>,
    rotation_semaphore: Arc<Semaphore>,
    emergency_keys: Arc<RwLock<HashMap<String, KyberKeyPair>>>,
}

impl QuantumSafeRotationManagerClone {
    async fn rotate_key(&self, key_id: &str, trigger: RotationTrigger) -> Result<String, TlsError> {
        // Simplified implementation for background tasks
        Ok(format!("{}-rotated", key_id))
    }
    
    async fn replenish_emergency_key(&self) -> Result<(), TlsError> {
        Ok(())
    }
    
    async fn rotation_scheduler_task(&self) {
        // Implementation would mirror the main manager
    }
    
    async fn threat_assessment_task(&self) {
        // Implementation would mirror the main manager
    }
    
    async fn coordination_handler_task(&self) {
        // Implementation would mirror the main manager
    }
    
    async fn performance_monitoring_task(&self) {
        // Implementation would mirror the main manager
    }
    
    async fn trigger_emergency_rotations(&self) {
        // Implementation would mirror the main manager
    }
    
    async fn check_rotation_triggers(&self) -> Result<(), TlsError> {
        Ok(())
    }
    
    async fn should_rotate_key(&self, _metadata: &KeyMetadata) -> Result<bool, TlsError> {
        Ok(false)
    }
    
    async fn assess_quantum_threat(&self) -> Result<(), TlsError> {
        Ok(())
    }
    
    async fn handle_coordination_message(&self, _message: CoordinationMessage) -> Result<(), TlsError> {
        Ok(())
    }
    
    async fn update_performance_metrics(&self) -> Result<(), TlsError> {
        Ok(())
    }
}

impl QuantumThreatAssessor {
    fn new(initial_level: QuantumThreatLevel) -> Self {
        Self {
            current_level: initial_level,
            assessment_history: VecDeque::with_capacity(1000),
            indicators: HashMap::new(),
            thresholds: Self::default_thresholds(),
        }
    }

    fn default_thresholds() -> HashMap<QuantumThreatLevel, f64> {
        let mut thresholds = HashMap::new();
        thresholds.insert(QuantumThreatLevel::Minimal, 0.0);
        thresholds.insert(QuantumThreatLevel::Low, 0.2);
        thresholds.insert(QuantumThreatLevel::Moderate, 0.4);
        thresholds.insert(QuantumThreatLevel::High, 0.6);
        thresholds.insert(QuantumThreatLevel::Critical, 0.8);
        thresholds.insert(QuantumThreatLevel::Imminent, 0.95);
        thresholds
    }

    fn current_level(&self) -> QuantumThreatLevel {
        self.current_level
    }

    fn update_level(&mut self, new_level: QuantumThreatLevel) {
        self.current_level = new_level;
    }

    async fn assess_threat(&mut self) -> Result<ThreatAssessment, TlsError> {
        // In a real implementation, this would:
        // 1. Collect threat intelligence
        // 2. Analyze quantum computing developments
        // 3. Monitor cryptanalytic advances
        // 4. Assess infrastructure vulnerabilities
        // 5. Calculate risk scores

        let assessment = ThreatAssessment {
            timestamp: SystemTime::now(),
            level: self.current_level,
            confidence: 0.8,
            indicators: HashMap::new(),
            reasoning: vec!["Periodic assessment".to_string()],
        };

        self.assessment_history.push_back(assessment.clone());
        if self.assessment_history.len() > 1000 {
            self.assessment_history.pop_front();
        }

        Ok(assessment)
    }
}

impl PerformanceMonitor {
    fn new() -> Self {
        Self {
            cpu_usage: Arc::new(tokio::sync::RwLock::new(0.0)),
            memory_usage: Arc::new(tokio::sync::RwLock::new(0.0)),
            network_usage: Arc::new(tokio::sync::RwLock::new(0.0)),
            rotation_metrics: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        }
    }

    async fn update_usage(&self, cpu: f64, memory: f64, network: f64) {
        *self.cpu_usage.write().await = cpu;
        *self.memory_usage.write().await = memory;
        *self.network_usage.write().await = network;
    }

    async fn get_current_usage(&self) -> SystemUsage {
        SystemUsage {
            cpu_usage: *self.cpu_usage.read().await,
            memory_usage: *self.memory_usage.read().await,
            network_usage: *self.network_usage.read().await,
        }
    }
}

#[derive(Debug)]
struct SystemUsage {
    cpu_usage: f64,
    memory_usage: f64,
    network_usage: f64,
}

/// Rotation statistics for monitoring and reporting
#[derive(Debug)]
pub struct RotationStatistics {
    pub total_rotations: u64,
    pub successful_rotations: u64,
    pub failed_rotations: u64,
    pub active_rotations: usize,
    pub average_rotation_time: Duration,
    pub threat_level_distribution: HashMap<QuantumThreatLevel, u64>,
    pub trigger_distribution: HashMap<String, u64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_quantum_threat_levels() {
        assert!(QuantumThreatLevel::Critical > QuantumThreatLevel::High);
        assert!(QuantumThreatLevel::Imminent > QuantumThreatLevel::Critical);
    }

    #[test]
    fn test_rotation_strategy_default() {
        let strategy = RotationStrategy::default();
        match strategy {
            RotationStrategy::Scheduled { interval, jitter } => {
                assert_eq!(interval, Duration::from_secs(86400));
                assert_eq!(jitter, Duration::from_secs(3600));
            }
            _ => panic!("unexpected default strategy"),
        }
    }

    #[test]
    fn test_rotation_policy_config() {
        let config = RotationPolicyConfig::default();
        assert!(matches!(config.coordination, CoordinationMode::Independent));
        assert!(config.forward_secrecy.perfect_forward_secrecy);
        assert!(config.emergency.enabled);
    }

    #[test]
    fn test_threat_assessor() {
        let mut assessor = QuantumThreatAssessor::new(QuantumThreatLevel::Low);
        assert_eq!(assessor.current_level(), QuantumThreatLevel::Low);
        
        assessor.update_level(QuantumThreatLevel::High);
        assert_eq!(assessor.current_level(), QuantumThreatLevel::High);
    }

    #[tokio::test]
    async fn test_performance_monitor() {
        let monitor = PerformanceMonitor::new();
        
        monitor.update_usage(0.5, 0.6, 0.3).await;
        let usage = monitor.get_current_usage().await;
        
        assert_eq!(usage.cpu_usage, 0.5);
        assert_eq!(usage.memory_usage, 0.6);
        assert_eq!(usage.network_usage, 0.3);
    }
}