pub mod errors;
pub mod provider;
pub mod certs;
pub mod client;
pub mod server;
pub mod validation;

// Core PQC modules (NIST Standards)
pub mod kyber;
pub mod mlkem;
pub mod falcon;
pub mod dilithium;
pub mod hybrid;

// Key management modules
pub mod hsm;
pub mod lifecycle;
pub mod performance;
pub mod rotation;
pub mod config;

// Domain-specific modules
#[cfg(feature = "healthcare")]
pub mod domains_healthcare;
#[cfg(feature = "automotive")]
pub mod domains_automotive;
#[cfg(feature = "aviation")]
pub mod domains_aviation;
#[cfg(feature = "industrial")]
pub mod domains_industrial;

#[cfg(test)]
mod tests;

#[cfg(feature = "benchmarks")]
pub mod benchmarks;

pub use client::{TlsClient, TlsClientConfig};
pub use server::{TlsServer, TlsServerConfig};
pub use errors::TlsError;
pub use validation::{CertificateValidator, ValidationPolicy, ValidationResult, CertificatePinning, OcspConfig};

// Re-export key PQC types
// Re-export Kyber types
pub use kyber::{KyberKEM, KyberKeyPair, KyberPublicKey, KyberPrivateKey, KyberSharedSecret, KyberCiphertext, EncapsulationResult};

// Re-export ML-KEM types (NIST FIPS 203)
pub use mlkem::{MlKemEngine, MlKemKeyPair, MlKemPublicKey, MlKemPrivateKey, MlKemCiphertext, MlKemSharedSecret, MlKemSecurityLevel, MlKemEncapsulationResult};

// Re-export Falcon types
pub use falcon::{FalconEngine, FalconKeyPair, FalconPublicKey, FalconPrivateKey, FalconSignature, FalconSecurityLevel, FalconBatchVerifier, FalconDomainConfig, FalconDomain};

// Re-export Dilithium types
pub use dilithium::{DilithiumEngine, DilithiumKeyPair, DilithiumPublicKey, DilithiumPrivateKey, DilithiumSignature, DilithiumSecurityLevel};

// Re-export Hybrid types
pub use hybrid::{HybridKemEngine, HybridKeyExchange, HybridKemResult, X25519MlKem768, P384MlKem1024};

// Re-export key management types
pub use hsm::{HsmPqcManager, HsmKeyAttributes, HsmKeyType, HsmPool, HsmSlotConfig, HsmCredentials};
pub use lifecycle::{KeyLifecycleManager, KeyMetadata, KeyLifecycleState, KeyUsagePolicy, LifecycleEvent, LifecycleEventType};
pub use performance::{MemoryPool, BatchProcessor, SimdOperations, PerformanceMonitor, GlobalPerformanceManager};
pub use rotation::{QuantumSafeRotationManager, RotationPolicyConfig, RotationStrategy, QuantumThreatLevel, RotationTrigger};
pub use config::{ProductionConfig, Environment, SecurityConfig, PqcConfig, MonitoringConfig, ComplianceConfig};

/// Production-ready PQC-TLS implementation with complete feature set
pub struct PqcTlsManager {
    /// Kyber KEM implementation
    pub kyber_kem: std::sync::Arc<KyberKEM>,
    /// HSM integration (optional)
    pub hsm_manager: Option<std::sync::Arc<HsmPqcManager>>,
    /// Key lifecycle management
    pub lifecycle_manager: std::sync::Arc<KeyLifecycleManager>,
    /// Performance optimizations
    pub performance_manager: std::sync::Arc<GlobalPerformanceManager>,
    /// Quantum-safe rotation
    pub rotation_manager: std::sync::Arc<QuantumSafeRotationManager>,
}

impl PqcTlsManager {
    /// Create a new production-ready PQC-TLS manager
    pub async fn new(config: PqcTlsConfig) -> Result<Self, TlsError> {
        let kyber_kem = std::sync::Arc::new(
            if config.use_hardware_acceleration {
                KyberKEM::new()
            } else {
                KyberKEM::new().without_hardware_acceleration()
            }
        );

        let performance_manager = std::sync::Arc::new(GlobalPerformanceManager::new());
        performance_manager.initialize().await?;

        let hsm_manager = if let Some(hsm_config) = config.hsm_config {
            let hsm_pool = std::sync::Arc::new(
                HsmPool::new(
                    hsm_config.slots,
                    hsm_config.credentials,
                    hsm_config.session_config,
                    hsm_config.max_sessions_per_slot,
                ).await?
            );
            Some(std::sync::Arc::new(HsmPqcManager::new(hsm_pool)))
        } else {
            None
        };

        let lifecycle_manager = std::sync::Arc::new(
            KeyLifecycleManager::new(
                std::sync::Arc::clone(&kyber_kem),
                hsm_manager.as_ref().map(std::sync::Arc::clone),
                config.lifecycle_config,
            ).await?
        );

        let rotation_manager = std::sync::Arc::new(
            QuantumSafeRotationManager::new(
                config.rotation_config,
                config.node_id,
                std::sync::Arc::clone(&lifecycle_manager),
                hsm_manager.as_ref().map(std::sync::Arc::clone),
                std::sync::Arc::clone(&kyber_kem),
            ).await?
        );

        Ok(Self {
            kyber_kem,
            hsm_manager,
            lifecycle_manager,
            performance_manager,
            rotation_manager,
        })
    }

    /// Get system information and capabilities
    pub fn system_info(&self) -> PqcSystemInfo {
        let hardware_info = self.kyber_kem.hardware_info();
        
        PqcSystemInfo {
            kyber_available: true,
            hardware_acceleration: self.kyber_kem.uses_hardware_acceleration(),
            hsm_available: self.hsm_manager.is_some(),
            fips_mode: self.kyber_kem.is_fips_mode(),
            hardware_capabilities: hardware_info.clone(),
        }
    }
}

/// Configuration for PQC-TLS manager
#[derive(Debug, Clone)]
pub struct PqcTlsConfig {
    /// Node identifier for distributed systems
    pub node_id: String,
    /// Enable hardware acceleration
    pub use_hardware_acceleration: bool,
    /// HSM configuration (optional)
    pub hsm_config: Option<HsmConfig>,
    /// Key lifecycle configuration
    pub lifecycle_config: lifecycle::LifecycleManagerConfig,
    /// Rotation policy configuration
    pub rotation_config: RotationPolicyConfig,
}

impl Default for PqcTlsConfig {
    fn default() -> Self {
        Self {
            node_id: uuid::Uuid::new_v4().to_string(),
            use_hardware_acceleration: true,
            hsm_config: None,
            lifecycle_config: lifecycle::LifecycleManagerConfig::default(),
            rotation_config: RotationPolicyConfig::default(),
        }
    }
}

/// HSM configuration
#[derive(Debug, Clone)]
pub struct HsmConfig {
    pub slots: Vec<HsmSlotConfig>,
    pub credentials: HsmCredentials,
    pub session_config: hsm::HsmSessionConfig,
    pub max_sessions_per_slot: usize,
}

/// System information and capabilities
#[derive(Debug, Clone)]
pub struct PqcSystemInfo {
    pub kyber_available: bool,
    pub hardware_acceleration: bool,
    pub hsm_available: bool,
    pub fips_mode: bool,
    pub hardware_capabilities: kyber::HardwareCapabilities,
}
