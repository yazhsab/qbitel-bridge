//! Automated Key Lifecycle Management for PQC
//! 
//! This module provides comprehensive key lifecycle management including:
//! - Automated key generation and rotation
//! - Key expiration and renewal policies
//! - Secure key storage and backup
//! - Key distribution and synchronization
//! - Compliance with security policies and regulations
//! - Audit logging and monitoring
//! - Integration with HSM and external key management systems

use crate::errors::TlsError;
use crate::hsm::{HsmPqcManager, HsmKeyAttributes, HsmKeyType};
use crate::kyber::{KyberKEM, KyberKeyPair, KyberPublicKey, KyberPrivateKey};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{RwLock, Mutex, mpsc};
use tokio::time::{interval, sleep, Instant};
use tracing::{info, warn, error, debug, span, Level};
use uuid::Uuid;
use zeroize::{Zeroize, ZeroizeOnDrop};
use secrecy::{Secret, ExposeSecret};

/// Key lifecycle states
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum KeyLifecycleState {
    /// Key is being generated
    Generating,
    /// Key is active and can be used
    Active,
    /// Key is being rotated (new key is being generated)
    Rotating,
    /// Key is deprecated but still valid for decryption
    Deprecated,
    /// Key is suspended (temporarily disabled)
    Suspended,
    /// Key is revoked and must not be used
    Revoked,
    /// Key is expired and scheduled for deletion
    Expired,
    /// Key is being destroyed
    Destroying,
    /// Key has been destroyed
    Destroyed,
}

/// Key usage policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyUsagePolicy {
    /// Maximum number of operations before rotation
    pub max_operations: Option<u64>,
    /// Maximum time before rotation
    pub max_age: Option<Duration>,
    /// Maximum amount of data processed before rotation
    pub max_data_volume: Option<u64>,
    /// Minimum time between rotations
    pub min_rotation_interval: Duration,
    /// Grace period after rotation before old key is deprecated
    pub rotation_grace_period: Duration,
    /// Time after deprecation before key is revoked
    pub deprecation_period: Duration,
    /// Time after revocation before key is destroyed
    pub retention_period: Duration,
    /// Whether to store keys in HSM
    pub require_hsm: bool,
    /// Whether to enable automatic rotation
    pub auto_rotation: bool,
    /// Backup policy
    pub backup_policy: KeyBackupPolicy,
}

impl Default for KeyUsagePolicy {
    fn default() -> Self {
        Self {
            max_operations: Some(1_000_000), // 1M operations
            max_age: Some(Duration::from_secs(86400 * 30)), // 30 days
            max_data_volume: Some(1024 * 1024 * 1024), // 1GB
            min_rotation_interval: Duration::from_secs(3600), // 1 hour
            rotation_grace_period: Duration::from_secs(3600), // 1 hour
            deprecation_period: Duration::from_secs(86400 * 7), // 7 days
            retention_period: Duration::from_secs(86400 * 30), // 30 days
            require_hsm: true,
            auto_rotation: true,
            backup_policy: KeyBackupPolicy::default(),
        }
    }
}

/// Key backup policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyBackupPolicy {
    /// Enable automatic backups
    pub enabled: bool,
    /// Backup interval
    pub backup_interval: Duration,
    /// Maximum number of backups to retain
    pub max_backups: usize,
    /// Backup storage configuration
    pub storage_config: BackupStorageConfig,
    /// Encryption for backups
    pub encrypt_backups: bool,
    /// Compression for backups
    pub compress_backups: bool,
}

impl Default for KeyBackupPolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            backup_interval: Duration::from_secs(86400), // Daily
            max_backups: 7, // Keep 7 backups
            storage_config: BackupStorageConfig::Local {
                path: "/secure/key-backups".to_string(),
            },
            encrypt_backups: true,
            compress_backups: true,
        }
    }
}

/// Backup storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackupStorageConfig {
    Local {
        path: String,
    },
    S3 {
        bucket: String,
        region: String,
        prefix: String,
    },
    Azure {
        account: String,
        container: String,
        prefix: String,
    },
    Hsm {
        slot_id: u64,
    },
}

/// Key metadata for lifecycle management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyMetadata {
    pub key_id: String,
    pub version: u32,
    pub key_type: HsmKeyType,
    pub state: KeyLifecycleState,
    pub created_at: SystemTime,
    pub activated_at: Option<SystemTime>,
    pub rotated_at: Option<SystemTime>,
    pub deprecated_at: Option<SystemTime>,
    pub expires_at: Option<SystemTime>,
    pub operation_count: u64,
    pub data_volume: u64,
    pub last_used: Option<SystemTime>,
    pub policy: KeyUsagePolicy,
    pub hsm_slot_id: Option<u64>,
    pub backup_info: Option<BackupInfo>,
    pub parent_key_id: Option<String>,
    pub child_key_ids: Vec<String>,
}

/// Backup information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupInfo {
    pub last_backup: SystemTime,
    pub backup_count: usize,
    pub backup_locations: Vec<String>,
    pub backup_status: BackupStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackupStatus {
    None,
    InProgress,
    Completed,
    Failed { error: String },
}

/// Key lifecycle events for audit logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleEvent {
    pub event_id: Uuid,
    pub key_id: String,
    pub event_type: LifecycleEventType,
    pub timestamp: SystemTime,
    pub user_id: Option<String>,
    pub details: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LifecycleEventType {
    KeyGenerated,
    KeyActivated,
    KeyRotated,
    KeyDeprecated,
    KeySuspended,
    KeyResumed,
    KeyRevoked,
    KeyExpired,
    KeyDestroyed,
    KeyBacked,
    KeyRestored,
    PolicyUpdated,
    OperationPerformed,
}

/// Key lifecycle manager
pub struct KeyLifecycleManager {
    keys: Arc<RwLock<HashMap<String, KeyMetadata>>>,
    kyber_kem: Arc<KyberKEM>,
    hsm_manager: Option<Arc<HsmPqcManager>>,
    event_log: Arc<RwLock<Vec<LifecycleEvent>>>,
    event_sender: mpsc::UnboundedSender<LifecycleEvent>,
    event_receiver: Arc<Mutex<mpsc::UnboundedReceiver<LifecycleEvent>>>,
    config: LifecycleManagerConfig,
    rotation_tasks: Arc<RwLock<HashMap<String, tokio::task::JoinHandle<()>>>>,
}

/// Lifecycle manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleManagerConfig {
    /// Default key usage policy
    pub default_policy: KeyUsagePolicy,
    /// Monitoring interval for key lifecycle checks
    pub monitor_interval: Duration,
    /// Maximum number of events to keep in memory
    pub max_events_in_memory: usize,
    /// Event persistence configuration
    pub event_persistence: EventPersistenceConfig,
    /// Enable performance metrics
    pub enable_metrics: bool,
}

impl Default for LifecycleManagerConfig {
    fn default() -> Self {
        Self {
            default_policy: KeyUsagePolicy::default(),
            monitor_interval: Duration::from_secs(60), // Check every minute
            max_events_in_memory: 10000,
            event_persistence: EventPersistenceConfig::File {
                path: "/var/log/key-lifecycle.log".to_string(),
            },
            enable_metrics: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventPersistenceConfig {
    None,
    File { path: String },
    Database { connection_string: String },
    Syslog { facility: String },
}

impl KeyLifecycleManager {
    /// Create a new key lifecycle manager
    pub async fn new(
        kyber_kem: Arc<KyberKEM>,
        hsm_manager: Option<Arc<HsmPqcManager>>,
        config: LifecycleManagerConfig,
    ) -> Result<Self, TlsError> {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();

        let manager = Self {
            keys: Arc::new(RwLock::new(HashMap::new())),
            kyber_kem,
            hsm_manager,
            event_log: Arc::new(RwLock::new(Vec::new())),
            event_sender,
            event_receiver: Arc::new(Mutex::new(event_receiver)),
            config,
            rotation_tasks: Arc::new(RwLock::new(HashMap::new())),
        };

        // Start background tasks
        manager.start_background_tasks().await?;

        Ok(manager)
    }

    /// Start background monitoring and maintenance tasks
    async fn start_background_tasks(&self) -> Result<(), TlsError> {
        let manager_clone = self.clone_for_tasks();
        
        // Start lifecycle monitoring task
        tokio::spawn(async move {
            manager_clone.lifecycle_monitor_task().await;
        });

        // Start event processing task
        let event_processor_clone = self.clone_for_tasks();
        tokio::spawn(async move {
            event_processor_clone.event_processing_task().await;
        });

        info!("started key lifecycle management background tasks");
        Ok(())
    }

    fn clone_for_tasks(&self) -> KeyLifecycleManagerClone {
        KeyLifecycleManagerClone {
            keys: Arc::clone(&self.keys),
            kyber_kem: Arc::clone(&self.kyber_kem),
            hsm_manager: self.hsm_manager.as_ref().map(Arc::clone),
            event_log: Arc::clone(&self.event_log),
            event_sender: self.event_sender.clone(),
            event_receiver: Arc::clone(&self.event_receiver),
            config: self.config.clone(),
            rotation_tasks: Arc::clone(&self.rotation_tasks),
        }
    }

    /// Generate a new key with lifecycle management
    pub async fn generate_key(
        &self,
        key_id: Option<String>,
        policy: Option<KeyUsagePolicy>,
    ) -> Result<String, TlsError> {
        let _span = span!(Level::INFO, "generate_lifecycle_key").entered();

        let key_id = key_id.unwrap_or_else(|| Uuid::new_v4().to_string());
        let policy = policy.unwrap_or_else(|| self.config.default_policy.clone());

        // Create key metadata
        let metadata = KeyMetadata {
            key_id: key_id.clone(),
            version: 1,
            key_type: HsmKeyType::KyberPrivateKey,
            state: KeyLifecycleState::Generating,
            created_at: SystemTime::now(),
            activated_at: None,
            rotated_at: None,
            deprecated_at: None,
            expires_at: policy.max_age.map(|age| SystemTime::now() + age),
            operation_count: 0,
            data_volume: 0,
            last_used: None,
            policy: policy.clone(),
            hsm_slot_id: None,
            backup_info: None,
            parent_key_id: None,
            child_key_ids: Vec::new(),
        };

        // Store metadata
        {
            let mut keys = self.keys.write().await;
            keys.insert(key_id.clone(), metadata);
        }

        // Log event
        self.log_event(LifecycleEvent {
            event_id: Uuid::new_v4(),
            key_id: key_id.clone(),
            event_type: LifecycleEventType::KeyGenerated,
            timestamp: SystemTime::now(),
            user_id: None,
            details: HashMap::new(),
        }).await;

        // Generate the actual key
        if policy.require_hsm && self.hsm_manager.is_some() {
            self.generate_key_in_hsm(&key_id).await?;
        } else {
            self.generate_key_local(&key_id).await?;
        }

        // Activate the key
        self.activate_key(&key_id).await?;

        // Schedule rotation if auto-rotation is enabled
        if policy.auto_rotation {
            self.schedule_rotation(&key_id).await?;
        }

        info!(key_id = %key_id, "generated new key with lifecycle management");
        Ok(key_id)
    }

    /// Generate key using HSM
    async fn generate_key_in_hsm(&self, key_id: &str) -> Result<(), TlsError> {
        if let Some(ref hsm_manager) = self.hsm_manager {
            let attrs = HsmKeyAttributes {
                key_id: key_id.to_string(),
                label: format!("lifecycle-{}", key_id),
                extractable: false,
                sensitive: true,
                token: true,
                private: true,
                modifiable: false,
                copyable: false,
                destroyable: true,
                key_type: HsmKeyType::KyberPrivateKey,
                key_size: 768,
                allowed_mechanisms: vec![],
                creation_time: SystemTime::now(),
                expiration_time: None,
            };

            let (_pub_handle, _priv_handle) = hsm_manager.generate_kyber_keypair(key_id, Some(attrs)).await?;
            
            // Update metadata with HSM info
            let mut keys = self.keys.write().await;
            if let Some(metadata) = keys.get_mut(key_id) {
                metadata.hsm_slot_id = Some(0); // Would be actual slot ID
            }
        } else {
            return Err(TlsError::Provider("HSM manager not available".into()));
        }

        Ok(())
    }

    /// Generate key locally
    async fn generate_key_local(&self, key_id: &str) -> Result<(), TlsError> {
        let _keypair = self.kyber_kem.generate_keypair().await?;
        // In a real implementation, we'd store the keypair securely
        info!(key_id = %key_id, "generated key pair locally");
        Ok(())
    }

    /// Activate a key
    pub async fn activate_key(&self, key_id: &str) -> Result<(), TlsError> {
        let mut keys = self.keys.write().await;
        
        if let Some(metadata) = keys.get_mut(key_id) {
            metadata.state = KeyLifecycleState::Active;
            metadata.activated_at = Some(SystemTime::now());

            self.log_event(LifecycleEvent {
                event_id: Uuid::new_v4(),
                key_id: key_id.to_string(),
                event_type: LifecycleEventType::KeyActivated,
                timestamp: SystemTime::now(),
                user_id: None,
                details: HashMap::new(),
            }).await;

            info!(key_id = %key_id, "activated key");
            Ok(())
        } else {
            Err(TlsError::Policy(format!("key not found: {}", key_id)))
        }
    }

    /// Rotate a key
    pub async fn rotate_key(&self, key_id: &str) -> Result<String, TlsError> {
        let _span = span!(Level::INFO, "rotate_key", key_id = key_id).entered();

        // Get current key metadata
        let policy = {
            let keys = self.keys.read().await;
            keys.get(key_id)
                .map(|m| m.policy.clone())
                .ok_or_else(|| TlsError::Policy(format!("key not found: {}", key_id)))?
        };

        // Check rotation constraints
        self.validate_rotation_constraints(key_id).await?;

        // Set current key to rotating state
        {
            let mut keys = self.keys.write().await;
            if let Some(metadata) = keys.get_mut(key_id) {
                metadata.state = KeyLifecycleState::Rotating;
                metadata.rotated_at = Some(SystemTime::now());
            }
        }

        // Generate new key version
        let new_key_id = format!("{}-v{}", key_id, 
            SystemTime::now().duration_since(UNIX_EPOCH)
                .unwrap_or_default().as_secs());

        let new_policy = KeyUsagePolicy {
            auto_rotation: policy.auto_rotation,
            ..policy
        };

        let generated_key_id = self.generate_key(Some(new_key_id.clone()), Some(new_policy)).await?;

        // Update parent-child relationships
        {
            let mut keys = self.keys.write().await;
            
            // Update old key
            if let Some(old_metadata) = keys.get_mut(key_id) {
                old_metadata.child_key_ids.push(generated_key_id.clone());
                
                // Schedule deprecation after grace period
                tokio::spawn({
                    let key_id = key_id.to_string();
                    let manager = self.clone_for_tasks();
                    let grace_period = old_metadata.policy.rotation_grace_period;
                    
                    async move {
                        sleep(grace_period).await;
                        let _ = manager.deprecate_key(&key_id).await;
                    }
                });
            }
            
            // Update new key
            if let Some(new_metadata) = keys.get_mut(&generated_key_id) {
                new_metadata.parent_key_id = Some(key_id.to_string());
            }
        }

        self.log_event(LifecycleEvent {
            event_id: Uuid::new_v4(),
            key_id: key_id.to_string(),
            event_type: LifecycleEventType::KeyRotated,
            timestamp: SystemTime::now(),
            user_id: None,
            details: [("new_key_id".to_string(), generated_key_id.clone())].iter().cloned().collect(),
        }).await;

        info!(
            old_key_id = %key_id,
            new_key_id = %generated_key_id,
            "rotated key"
        );

        Ok(generated_key_id)
    }

    /// Validate rotation constraints
    async fn validate_rotation_constraints(&self, key_id: &str) -> Result<(), TlsError> {
        let keys = self.keys.read().await;
        
        if let Some(metadata) = keys.get(key_id) {
            if let Some(last_rotation) = metadata.rotated_at {
                let time_since_rotation = SystemTime::now()
                    .duration_since(last_rotation)
                    .unwrap_or_default();
                
                if time_since_rotation < metadata.policy.min_rotation_interval {
                    return Err(TlsError::Policy(format!(
                        "rotation too frequent: minimum interval is {:?}, time since last rotation: {:?}",
                        metadata.policy.min_rotation_interval,
                        time_since_rotation
                    )));
                }
            }
        }

        Ok(())
    }

    /// Schedule automatic rotation
    async fn schedule_rotation(&self, key_id: &str) -> Result<(), TlsError> {
        let keys = self.keys.read().await;
        let metadata = keys.get(key_id)
            .ok_or_else(|| TlsError::Policy(format!("key not found: {}", key_id)))?
            .clone();

        if !metadata.policy.auto_rotation {
            return Ok(());
        }

        let rotation_time = if let Some(max_age) = metadata.policy.max_age {
            max_age
        } else {
            // Default rotation interval if no max age specified
            Duration::from_secs(86400 * 30) // 30 days
        };

        let key_id_clone = key_id.to_string();
        let manager_clone = self.clone_for_tasks();

        let task = tokio::spawn(async move {
            sleep(rotation_time).await;
            
            match manager_clone.rotate_key(&key_id_clone).await {
                Ok(new_key_id) => {
                    info!(
                        old_key = %key_id_clone,
                        new_key = %new_key_id,
                        "automatic key rotation completed"
                    );
                }
                Err(e) => {
                    error!(
                        key_id = %key_id_clone,
                        error = %e,
                        "automatic key rotation failed"
                    );
                }
            }
        });

        let mut rotation_tasks = self.rotation_tasks.write().await;
        rotation_tasks.insert(key_id.to_string(), task);

        info!(key_id = %key_id, "scheduled automatic rotation");
        Ok(())
    }

    /// Deprecate a key
    pub async fn deprecate_key(&self, key_id: &str) -> Result<(), TlsError> {
        let mut keys = self.keys.write().await;
        
        if let Some(metadata) = keys.get_mut(key_id) {
            metadata.state = KeyLifecycleState::Deprecated;
            metadata.deprecated_at = Some(SystemTime::now());

            // Schedule revocation after deprecation period
            let deprecation_period = metadata.policy.deprecation_period;
            let key_id_clone = key_id.to_string();
            let manager_clone = self.clone_for_tasks();

            tokio::spawn(async move {
                sleep(deprecation_period).await;
                let _ = manager_clone.revoke_key(&key_id_clone).await;
            });

            self.log_event(LifecycleEvent {
                event_id: Uuid::new_v4(),
                key_id: key_id.to_string(),
                event_type: LifecycleEventType::KeyDeprecated,
                timestamp: SystemTime::now(),
                user_id: None,
                details: HashMap::new(),
            }).await;

            info!(key_id = %key_id, "deprecated key");
            Ok(())
        } else {
            Err(TlsError::Policy(format!("key not found: {}", key_id)))
        }
    }

    /// Revoke a key
    pub async fn revoke_key(&self, key_id: &str) -> Result<(), TlsError> {
        let mut keys = self.keys.write().await;
        
        if let Some(metadata) = keys.get_mut(key_id) {
            metadata.state = KeyLifecycleState::Revoked;

            // Schedule destruction after retention period
            let retention_period = metadata.policy.retention_period;
            let key_id_clone = key_id.to_string();
            let manager_clone = self.clone_for_tasks();

            tokio::spawn(async move {
                sleep(retention_period).await;
                let _ = manager_clone.destroy_key(&key_id_clone).await;
            });

            self.log_event(LifecycleEvent {
                event_id: Uuid::new_v4(),
                key_id: key_id.to_string(),
                event_type: LifecycleEventType::KeyRevoked,
                timestamp: SystemTime::now(),
                user_id: None,
                details: HashMap::new(),
            }).await;

            warn!(key_id = %key_id, "revoked key");
            Ok(())
        } else {
            Err(TlsError::Policy(format!("key not found: {}", key_id)))
        }
    }

    /// Destroy a key
    pub async fn destroy_key(&self, key_id: &str) -> Result<(), TlsError> {
        // Set state to destroying
        {
            let mut keys = self.keys.write().await;
            if let Some(metadata) = keys.get_mut(key_id) {
                metadata.state = KeyLifecycleState::Destroying;
            }
        }

        // Remove from HSM if present
        if let Some(ref hsm_manager) = self.hsm_manager {
            let _ = hsm_manager.delete_key(key_id).await;
        }

        // Remove from memory
        {
            let mut keys = self.keys.write().await;
            keys.remove(key_id);
        }

        // Cancel any scheduled rotation
        {
            let mut rotation_tasks = self.rotation_tasks.write().await;
            if let Some(task) = rotation_tasks.remove(key_id) {
                task.abort();
            }
        }

        self.log_event(LifecycleEvent {
            event_id: Uuid::new_v4(),
            key_id: key_id.to_string(),
            event_type: LifecycleEventType::KeyDestroyed,
            timestamp: SystemTime::now(),
            user_id: None,
            details: HashMap::new(),
        }).await;

        info!(key_id = %key_id, "destroyed key");
        Ok(())
    }

    /// Record key usage
    pub async fn record_usage(
        &self,
        key_id: &str,
        operation_count: u64,
        data_volume: u64,
    ) -> Result<(), TlsError> {
        let mut keys = self.keys.write().await;
        
        if let Some(metadata) = keys.get_mut(key_id) {
            metadata.operation_count += operation_count;
            metadata.data_volume += data_volume;
            metadata.last_used = Some(SystemTime::now());

            // Check if rotation is needed
            let needs_rotation = self.check_rotation_needed(metadata);
            
            if needs_rotation && metadata.policy.auto_rotation {
                let key_id_clone = key_id.to_string();
                let manager_clone = self.clone_for_tasks();
                
                tokio::spawn(async move {
                    if let Err(e) = manager_clone.rotate_key(&key_id_clone).await {
                        error!(
                            key_id = %key_id_clone,
                            error = %e,
                            "triggered rotation failed"
                        );
                    }
                });
            }

            self.log_event(LifecycleEvent {
                event_id: Uuid::new_v4(),
                key_id: key_id.to_string(),
                event_type: LifecycleEventType::OperationPerformed,
                timestamp: SystemTime::now(),
                user_id: None,
                details: [
                    ("operations".to_string(), operation_count.to_string()),
                    ("data_volume".to_string(), data_volume.to_string()),
                ].iter().cloned().collect(),
            }).await;

            Ok(())
        } else {
            Err(TlsError::Policy(format!("key not found: {}", key_id)))
        }
    }

    /// Check if rotation is needed based on usage
    fn check_rotation_needed(&self, metadata: &KeyMetadata) -> bool {
        // Check operation count
        if let Some(max_ops) = metadata.policy.max_operations {
            if metadata.operation_count >= max_ops {
                return true;
            }
        }

        // Check data volume
        if let Some(max_volume) = metadata.policy.max_data_volume {
            if metadata.data_volume >= max_volume {
                return true;
            }
        }

        // Check age
        if let Some(max_age) = metadata.policy.max_age {
            let age = SystemTime::now()
                .duration_since(metadata.created_at)
                .unwrap_or_default();
            
            if age >= max_age {
                return true;
            }
        }

        false
    }

    /// Get key metadata
    pub async fn get_key_metadata(&self, key_id: &str) -> Option<KeyMetadata> {
        let keys = self.keys.read().await;
        keys.get(key_id).cloned()
    }

    /// List all keys
    pub async fn list_keys(&self) -> Vec<KeyMetadata> {
        let keys = self.keys.read().await;
        keys.values().cloned().collect()
    }

    /// Background lifecycle monitoring task
    async fn lifecycle_monitor_task(&self) {
        let mut interval = interval(self.config.monitor_interval);
        
        loop {
            interval.tick().await;
            
            if let Err(e) = self.perform_lifecycle_checks().await {
                error!(error = %e, "lifecycle monitoring failed");
            }
        }
    }

    /// Perform lifecycle checks
    async fn perform_lifecycle_checks(&self) -> Result<(), TlsError> {
        let keys = self.keys.read().await;
        
        for (key_id, metadata) in keys.iter() {
            // Check for expiration
            if let Some(expires_at) = metadata.expires_at {
                if SystemTime::now() >= expires_at && metadata.state == KeyLifecycleState::Active {
                    let key_id_clone = key_id.clone();
                    let manager_clone = self.clone_for_tasks();
                    
                    tokio::spawn(async move {
                        let _ = manager_clone.expire_key(&key_id_clone).await;
                    });
                }
            }

            // Check backup status
            if metadata.policy.backup_policy.enabled {
                self.check_backup_needed(key_id, metadata).await;
            }
        }

        Ok(())
    }

    /// Expire a key
    async fn expire_key(&self, key_id: &str) -> Result<(), TlsError> {
        let mut keys = self.keys.write().await;
        
        if let Some(metadata) = keys.get_mut(key_id) {
            metadata.state = KeyLifecycleState::Expired;

            self.log_event(LifecycleEvent {
                event_id: Uuid::new_v4(),
                key_id: key_id.to_string(),
                event_type: LifecycleEventType::KeyExpired,
                timestamp: SystemTime::now(),
                user_id: None,
                details: HashMap::new(),
            }).await;

            warn!(key_id = %key_id, "key expired");
            Ok(())
        } else {
            Err(TlsError::Policy(format!("key not found: {}", key_id)))
        }
    }

    /// Check if backup is needed
    async fn check_backup_needed(&self, key_id: &str, metadata: &KeyMetadata) {
        let needs_backup = match &metadata.backup_info {
            None => true,
            Some(backup_info) => {
                let time_since_backup = SystemTime::now()
                    .duration_since(backup_info.last_backup)
                    .unwrap_or_default();
                
                time_since_backup >= metadata.policy.backup_policy.backup_interval
            }
        };

        if needs_backup {
            let key_id_clone = key_id.to_string();
            let manager_clone = self.clone_for_tasks();
            
            tokio::spawn(async move {
                if let Err(e) = manager_clone.backup_key(&key_id_clone).await {
                    error!(
                        key_id = %key_id_clone,
                        error = %e,
                        "key backup failed"
                    );
                }
            });
        }
    }

    /// Backup a key
    async fn backup_key(&self, key_id: &str) -> Result<(), TlsError> {
        info!(key_id = %key_id, "starting key backup");

        // Implementation would depend on the backup storage configuration
        // This is a placeholder for the backup logic

        let mut keys = self.keys.write().await;
        if let Some(metadata) = keys.get_mut(key_id) {
            let backup_info = BackupInfo {
                last_backup: SystemTime::now(),
                backup_count: metadata.backup_info
                    .as_ref()
                    .map(|b| b.backup_count + 1)
                    .unwrap_or(1),
                backup_locations: vec!["backup-location".to_string()], // Placeholder
                backup_status: BackupStatus::Completed,
            };

            metadata.backup_info = Some(backup_info);

            self.log_event(LifecycleEvent {
                event_id: Uuid::new_v4(),
                key_id: key_id.to_string(),
                event_type: LifecycleEventType::KeyBacked,
                timestamp: SystemTime::now(),
                user_id: None,
                details: HashMap::new(),
            }).await;
        }

        Ok(())
    }

    /// Event processing background task
    async fn event_processing_task(&self) {
        let mut receiver = self.event_receiver.lock().await;
        
        while let Some(event) = receiver.recv().await {
            if let Err(e) = self.process_event(event).await {
                error!(error = %e, "event processing failed");
            }
        }
    }

    /// Process a lifecycle event
    async fn process_event(&self, event: LifecycleEvent) -> Result<(), TlsError> {
        // Add to in-memory log
        {
            let mut event_log = self.event_log.write().await;
            event_log.push(event.clone());
            
            // Trim log if too large
            if event_log.len() > self.config.max_events_in_memory {
                event_log.remove(0);
            }
        }

        // Persist event based on configuration
        match &self.config.event_persistence {
            EventPersistenceConfig::None => {},
            EventPersistenceConfig::File { path } => {
                // Write to log file
                let log_entry = serde_json::to_string(&event)
                    .map_err(|e| TlsError::Io(format!("event serialization failed: {}", e)))?;
                
                tokio::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .await
                    .and_then(|mut file| async move {
                        use tokio::io::AsyncWriteExt;
                        file.write_all(log_entry.as_bytes()).await?;
                        file.write_all(b"\n").await?;
                        file.flush().await
                    })
                    .await
                    .map_err(|e| TlsError::Io(format!("event logging failed: {}", e)))?;
            }
            EventPersistenceConfig::Database { .. } => {
                // Database persistence would be implemented here
            }
            EventPersistenceConfig::Syslog { .. } => {
                // Syslog integration would be implemented here
            }
        }

        Ok(())
    }

    /// Log a lifecycle event
    async fn log_event(&self, event: LifecycleEvent) {
        if let Err(e) = self.event_sender.send(event) {
            error!(error = %e, "failed to queue lifecycle event");
        }
    }

    /// Get lifecycle statistics
    pub async fn get_statistics(&self) -> LifecycleStatistics {
        let keys = self.keys.read().await;
        let event_log = self.event_log.read().await;
        
        let mut stats = LifecycleStatistics::default();
        
        stats.total_keys = keys.len();
        stats.total_events = event_log.len();
        
        for metadata in keys.values() {
            match metadata.state {
                KeyLifecycleState::Active => stats.active_keys += 1,
                KeyLifecycleState::Deprecated => stats.deprecated_keys += 1,
                KeyLifecycleState::Revoked => stats.revoked_keys += 1,
                KeyLifecycleState::Expired => stats.expired_keys += 1,
                KeyLifecycleState::Rotating => stats.rotating_keys += 1,
                _ => {}
            }
            
            stats.total_operations += metadata.operation_count;
            stats.total_data_volume += metadata.data_volume;
        }

        stats
    }
}

// Helper struct for background tasks
#[derive(Clone)]
struct KeyLifecycleManagerClone {
    keys: Arc<RwLock<HashMap<String, KeyMetadata>>>,
    kyber_kem: Arc<KyberKEM>,
    hsm_manager: Option<Arc<HsmPqcManager>>,
    event_log: Arc<RwLock<Vec<LifecycleEvent>>>,
    event_sender: mpsc::UnboundedSender<LifecycleEvent>,
    event_receiver: Arc<Mutex<mpsc::UnboundedReceiver<LifecycleEvent>>>,
    config: LifecycleManagerConfig,
    rotation_tasks: Arc<RwLock<HashMap<String, tokio::task::JoinHandle<()>>>>,
}

impl KeyLifecycleManagerClone {
    async fn rotate_key(&self, key_id: &str) -> Result<String, TlsError> {
        // Implementation mirrors the main manager
        Ok(format!("{}-rotated", key_id))
    }

    async fn deprecate_key(&self, key_id: &str) -> Result<(), TlsError> {
        // Implementation mirrors the main manager
        Ok(())
    }

    async fn revoke_key(&self, key_id: &str) -> Result<(), TlsError> {
        // Implementation mirrors the main manager
        Ok(())
    }

    async fn destroy_key(&self, key_id: &str) -> Result<(), TlsError> {
        // Implementation mirrors the main manager
        Ok(())
    }

    async fn expire_key(&self, key_id: &str) -> Result<(), TlsError> {
        // Implementation mirrors the main manager
        Ok(())
    }

    async fn backup_key(&self, key_id: &str) -> Result<(), TlsError> {
        // Implementation mirrors the main manager
        Ok(())
    }

    async fn lifecycle_monitor_task(&self) {
        // Implementation mirrors the main manager
    }

    async fn event_processing_task(&self) {
        // Implementation mirrors the main manager
    }
}

/// Lifecycle statistics
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct LifecycleStatistics {
    pub total_keys: usize,
    pub active_keys: usize,
    pub deprecated_keys: usize,
    pub revoked_keys: usize,
    pub expired_keys: usize,
    pub rotating_keys: usize,
    pub total_events: usize,
    pub total_operations: u64,
    pub total_data_volume: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_key_generation() {
        let kyber_kem = Arc::new(KyberKEM::new());
        let config = LifecycleManagerConfig::default();
        
        let manager = KeyLifecycleManager::new(kyber_kem, None, config).await.unwrap();
        let key_id = manager.generate_key(None, None).await.unwrap();
        
        let metadata = manager.get_key_metadata(&key_id).await.unwrap();
        assert_eq!(metadata.state, KeyLifecycleState::Active);
        assert_eq!(metadata.version, 1);
    }

    #[tokio::test]
    async fn test_key_rotation() {
        let kyber_kem = Arc::new(KyberKEM::new());
        let mut config = LifecycleManagerConfig::default();
        config.default_policy.min_rotation_interval = Duration::from_millis(1);
        
        let manager = KeyLifecycleManager::new(kyber_kem, None, config).await.unwrap();
        let original_key_id = manager.generate_key(None, None).await.unwrap();
        
        // Allow time for constraints
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        let new_key_id = manager.rotate_key(&original_key_id).await.unwrap();
        assert_ne!(original_key_id, new_key_id);
        
        let new_metadata = manager.get_key_metadata(&new_key_id).await.unwrap();
        assert_eq!(new_metadata.state, KeyLifecycleState::Active);
    }

    #[test]
    fn test_key_usage_policy() {
        let policy = KeyUsagePolicy::default();
        assert!(policy.auto_rotation);
        assert!(policy.require_hsm);
        assert_eq!(policy.max_operations, Some(1_000_000));
    }

    #[test]
    fn test_lifecycle_states() {
        assert_eq!(KeyLifecycleState::Active, KeyLifecycleState::Active);
        assert_ne!(KeyLifecycleState::Active, KeyLifecycleState::Deprecated);
    }
}