//! Falcon Digital Signature Algorithm - NIST PQC Round 3 Selection
//!
//! Falcon offers significantly smaller signatures than ML-DSA (Dilithium):
//! - Falcon-512: 666 bytes (vs ML-DSA-44: 2,420 bytes) - 3.6x smaller
//! - Falcon-1024: 1,280 bytes (vs ML-DSA-87: 4,595 bytes) - 3.6x smaller
//!
//! This makes Falcon ideal for bandwidth-constrained environments:
//! - Automotive V2X (IEEE 1609.2)
//! - Aviation (ACARS, ADS-B)
//! - IoT/Embedded systems
//!
//! Trade-off: Falcon requires more careful implementation (floating-point)
//! and has slower signing than Dilithium, but faster verification.

use crate::errors::TlsError;
use metrics::{counter, histogram};
use secrecy::{Secret, SecretVec, ExposeSecret};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tracing::{debug, info, span, Level};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Falcon Security Level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FalconSecurityLevel {
    /// Level 1: 128-bit classical security
    Falcon512,
    /// Level 5: 256-bit classical security
    Falcon1024,
}

impl FalconSecurityLevel {
    pub fn public_key_size(&self) -> usize {
        match self {
            FalconSecurityLevel::Falcon512 => 897,
            FalconSecurityLevel::Falcon1024 => 1793,
        }
    }

    pub fn private_key_size(&self) -> usize {
        match self {
            FalconSecurityLevel::Falcon512 => 1281,
            FalconSecurityLevel::Falcon1024 => 2305,
        }
    }

    /// Maximum signature size (actual size varies slightly)
    pub fn signature_size_max(&self) -> usize {
        match self {
            FalconSecurityLevel::Falcon512 => 690,  // Average ~666
            FalconSecurityLevel::Falcon1024 => 1330, // Average ~1280
        }
    }

    /// Typical/average signature size
    pub fn signature_size_typical(&self) -> usize {
        match self {
            FalconSecurityLevel::Falcon512 => 666,
            FalconSecurityLevel::Falcon1024 => 1280,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            FalconSecurityLevel::Falcon512 => "Falcon-512",
            FalconSecurityLevel::Falcon1024 => "Falcon-1024",
        }
    }

    /// NIST security level (1 or 5)
    pub fn nist_level(&self) -> u8 {
        match self {
            FalconSecurityLevel::Falcon512 => 1,
            FalconSecurityLevel::Falcon1024 => 5,
        }
    }

    /// Signature size comparison with ML-DSA
    pub fn size_advantage_vs_mldsa(&self) -> f64 {
        match self {
            FalconSecurityLevel::Falcon512 => 2420.0 / 666.0,  // ~3.6x smaller
            FalconSecurityLevel::Falcon1024 => 4595.0 / 1280.0, // ~3.6x smaller
        }
    }
}

impl Default for FalconSecurityLevel {
    fn default() -> Self {
        FalconSecurityLevel::Falcon512 // Preferred for bandwidth-constrained
    }
}

/// Falcon public key
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FalconPublicKey {
    level: FalconSecurityLevel,
    data: Vec<u8>,
}

impl FalconPublicKey {
    pub fn from_bytes(level: FalconSecurityLevel, bytes: &[u8]) -> Result<Self, TlsError> {
        let expected_size = level.public_key_size();
        if bytes.len() != expected_size {
            return Err(TlsError::Policy(format!(
                "{} invalid public key size: expected {}, got {}",
                level.name(),
                expected_size,
                bytes.len()
            )));
        }

        Ok(Self {
            level,
            data: bytes.to_vec(),
        })
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    pub fn level(&self) -> FalconSecurityLevel {
        self.level
    }
}

/// Falcon private key with secure memory handling
#[derive(Debug, ZeroizeOnDrop)]
pub struct FalconPrivateKey {
    level: FalconSecurityLevel,
    data: SecretVec<u8>,
}

impl FalconPrivateKey {
    pub fn from_bytes(level: FalconSecurityLevel, bytes: &[u8]) -> Result<Self, TlsError> {
        let expected_size = level.private_key_size();
        if bytes.len() != expected_size {
            return Err(TlsError::Policy(format!(
                "{} invalid private key size: expected {}, got {}",
                level.name(),
                expected_size,
                bytes.len()
            )));
        }

        Ok(Self {
            level,
            data: SecretVec::new(bytes.to_vec()),
        })
    }

    pub fn expose_secret(&self) -> &[u8] {
        self.data.expose_secret()
    }

    pub fn level(&self) -> FalconSecurityLevel {
        self.level
    }
}

/// Falcon key pair
#[derive(Debug)]
pub struct FalconKeyPair {
    pub public_key: FalconPublicKey,
    pub private_key: FalconPrivateKey,
    pub level: FalconSecurityLevel,
}

/// Falcon signature
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FalconSignature {
    level: FalconSecurityLevel,
    data: Vec<u8>,
}

impl FalconSignature {
    pub fn from_bytes(level: FalconSecurityLevel, bytes: &[u8]) -> Result<Self, TlsError> {
        // Falcon signatures have variable length
        let max_size = level.signature_size_max();
        if bytes.len() > max_size {
            return Err(TlsError::Policy(format!(
                "{} signature too large: max {}, got {}",
                level.name(),
                max_size,
                bytes.len()
            )));
        }

        Ok(Self {
            level,
            data: bytes.to_vec(),
        })
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    pub fn level(&self) -> FalconSecurityLevel {
        self.level
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }
}

/// Falcon metrics
struct FalconMetrics;

impl FalconMetrics {
    fn record_keygen(level: FalconSecurityLevel, duration: Duration) {
        histogram!("falcon_keygen_duration_seconds", "level" => level.name())
            .record(duration.as_secs_f64());
    }

    fn record_sign(level: FalconSecurityLevel, duration: Duration) {
        histogram!("falcon_sign_duration_seconds", "level" => level.name())
            .record(duration.as_secs_f64());
    }

    fn record_verify(level: FalconSecurityLevel, duration: Duration) {
        histogram!("falcon_verify_duration_seconds", "level" => level.name())
            .record(duration.as_secs_f64());
    }

    fn record_signature_size(level: FalconSecurityLevel, size: usize) {
        histogram!("falcon_signature_size_bytes", "level" => level.name())
            .record(size as f64);
    }

    fn increment_operations(level: FalconSecurityLevel, op: &str) {
        counter!("falcon_operations_total", "level" => level.name(), "operation" => op.to_string())
            .increment(1);
    }
}

/// Domain-specific Falcon configuration
#[derive(Debug, Clone)]
pub struct FalconDomainConfig {
    /// Target domain
    pub domain: FalconDomain,
    /// Security level
    pub level: FalconSecurityLevel,
    /// Enable signature compression (for aviation/V2X)
    pub enable_compression: bool,
    /// Maximum acceptable latency (for automotive real-time)
    pub max_latency_ms: Option<u32>,
}

/// Domain identifiers for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FalconDomain {
    /// General enterprise use
    Enterprise,
    /// Automotive V2X (IEEE 1609.2)
    AutomotiveV2X,
    /// Aviation (ACARS, ADS-B, LDACS)
    Aviation,
    /// Healthcare devices
    Healthcare,
    /// Industrial OT/ICS
    Industrial,
}

impl Default for FalconDomainConfig {
    fn default() -> Self {
        Self {
            domain: FalconDomain::Enterprise,
            level: FalconSecurityLevel::Falcon512,
            enable_compression: false,
            max_latency_ms: None,
        }
    }
}

impl FalconDomainConfig {
    /// Configuration for automotive V2X
    pub fn for_automotive_v2x() -> Self {
        Self {
            domain: FalconDomain::AutomotiveV2X,
            level: FalconSecurityLevel::Falcon512, // Smaller signatures
            enable_compression: true,
            max_latency_ms: Some(10), // <10ms requirement
        }
    }

    /// Configuration for aviation
    pub fn for_aviation() -> Self {
        Self {
            domain: FalconDomain::Aviation,
            level: FalconSecurityLevel::Falcon512,
            enable_compression: true, // Critical for 600bps channels
            max_latency_ms: None,
        }
    }

    /// Configuration for healthcare
    pub fn for_healthcare() -> Self {
        Self {
            domain: FalconDomain::Healthcare,
            level: FalconSecurityLevel::Falcon512, // Smaller for constrained devices
            enable_compression: false,
            max_latency_ms: None,
        }
    }
}

/// Production-ready Falcon signature engine
pub struct FalconEngine {
    default_level: FalconSecurityLevel,
    domain_config: Option<FalconDomainConfig>,
    use_hardware_acceleration: bool,
    fips_mode: bool,
}

impl FalconEngine {
    /// Create a new Falcon engine with the specified security level
    pub fn new(level: FalconSecurityLevel) -> Self {
        Self {
            default_level: level,
            domain_config: None,
            use_hardware_acceleration: true,
            fips_mode: false,
        }
    }

    /// Create engine with domain-specific configuration
    pub fn with_domain_config(config: FalconDomainConfig) -> Self {
        Self {
            default_level: config.level,
            domain_config: Some(config),
            use_hardware_acceleration: true,
            fips_mode: false,
        }
    }

    /// Create engine for bandwidth-constrained environments
    pub fn for_bandwidth_constrained() -> Self {
        Self::new(FalconSecurityLevel::Falcon512)
    }

    /// Create engine for maximum security
    pub fn for_maximum_security() -> Self {
        Self::new(FalconSecurityLevel::Falcon1024)
    }

    /// Enable FIPS compliance mode
    pub fn with_fips_mode(mut self) -> Self {
        self.fips_mode = true;
        self
    }

    /// Disable hardware acceleration
    pub fn without_hardware_acceleration(mut self) -> Self {
        self.use_hardware_acceleration = false;
        self
    }

    /// Generate a key pair at the default security level
    pub async fn generate_keypair(&self) -> Result<FalconKeyPair, TlsError> {
        self.generate_keypair_at_level(self.default_level).await
    }

    /// Generate a key pair at a specific security level
    pub async fn generate_keypair_at_level(
        &self,
        level: FalconSecurityLevel,
    ) -> Result<FalconKeyPair, TlsError> {
        let _span = span!(Level::INFO, "falcon_keygen", level = level.name()).entered();
        let start = Instant::now();

        let result = tokio::task::spawn_blocking(move || {
            Self::generate_keypair_impl(level)
        })
        .await
        .map_err(|e| TlsError::Io(format!("Falcon key generation failed: {}", e)))??;

        FalconMetrics::record_keygen(level, start.elapsed());
        FalconMetrics::increment_operations(level, "keygen");

        Ok(result)
    }

    fn generate_keypair_impl(level: FalconSecurityLevel) -> Result<FalconKeyPair, TlsError> {
        match level {
            FalconSecurityLevel::Falcon512 => {
                let (pk, sk) = pqcrypto_falcon::falcon512::keypair();
                Ok(FalconKeyPair {
                    public_key: FalconPublicKey::from_bytes(level, pk.as_bytes())?,
                    private_key: FalconPrivateKey::from_bytes(level, sk.as_bytes())?,
                    level,
                })
            }
            FalconSecurityLevel::Falcon1024 => {
                let (pk, sk) = pqcrypto_falcon::falcon1024::keypair();
                Ok(FalconKeyPair {
                    public_key: FalconPublicKey::from_bytes(level, pk.as_bytes())?,
                    private_key: FalconPrivateKey::from_bytes(level, sk.as_bytes())?,
                    level,
                })
            }
        }
    }

    /// Sign a message
    pub async fn sign(
        &self,
        private_key: &FalconPrivateKey,
        message: &[u8],
    ) -> Result<FalconSignature, TlsError> {
        let level = private_key.level();
        let _span = span!(Level::INFO, "falcon_sign", level = level.name()).entered();
        let start = Instant::now();

        // Check latency constraint if configured
        if let Some(ref config) = self.domain_config {
            if let Some(max_ms) = config.max_latency_ms {
                // Pre-check: we can't guarantee completion within deadline
                // but we can monitor and warn
                debug!("Falcon signing with {}ms latency budget", max_ms);
            }
        }

        let sk_bytes = private_key.expose_secret().to_vec();
        let msg = message.to_vec();

        let result = tokio::task::spawn_blocking(move || {
            Self::sign_impl(level, &sk_bytes, &msg)
        })
        .await
        .map_err(|e| TlsError::Io(format!("Falcon signing failed: {}", e)))??;

        let elapsed = start.elapsed();
        FalconMetrics::record_sign(level, elapsed);
        FalconMetrics::record_signature_size(level, result.size());
        FalconMetrics::increment_operations(level, "sign");

        // Check latency constraint
        if let Some(ref config) = self.domain_config {
            if let Some(max_ms) = config.max_latency_ms {
                if elapsed.as_millis() as u32 > max_ms {
                    tracing::warn!(
                        "Falcon signing exceeded latency budget: {}ms > {}ms",
                        elapsed.as_millis(),
                        max_ms
                    );
                }
            }
        }

        Ok(result)
    }

    fn sign_impl(
        level: FalconSecurityLevel,
        sk_bytes: &[u8],
        message: &[u8],
    ) -> Result<FalconSignature, TlsError> {
        match level {
            FalconSecurityLevel::Falcon512 => {
                let sk = pqcrypto_falcon::falcon512::SecretKey::from_bytes(sk_bytes)
                    .map_err(|_| TlsError::Policy("Invalid Falcon-512 private key".into()))?;
                let sig = pqcrypto_falcon::falcon512::detached_sign(message, &sk);
                FalconSignature::from_bytes(level, sig.as_bytes())
            }
            FalconSecurityLevel::Falcon1024 => {
                let sk = pqcrypto_falcon::falcon1024::SecretKey::from_bytes(sk_bytes)
                    .map_err(|_| TlsError::Policy("Invalid Falcon-1024 private key".into()))?;
                let sig = pqcrypto_falcon::falcon1024::detached_sign(message, &sk);
                FalconSignature::from_bytes(level, sig.as_bytes())
            }
        }
    }

    /// Verify a signature
    pub async fn verify(
        &self,
        public_key: &FalconPublicKey,
        message: &[u8],
        signature: &FalconSignature,
    ) -> Result<bool, TlsError> {
        let level = public_key.level();
        if level != signature.level() {
            return Err(TlsError::Policy(format!(
                "Security level mismatch: key is {}, signature is {}",
                level.name(),
                signature.level().name()
            )));
        }

        let _span = span!(Level::INFO, "falcon_verify", level = level.name()).entered();
        let start = Instant::now();

        let pk_bytes = public_key.data.clone();
        let sig_bytes = signature.data.clone();
        let msg = message.to_vec();

        let result = tokio::task::spawn_blocking(move || {
            Self::verify_impl(level, &pk_bytes, &msg, &sig_bytes)
        })
        .await
        .map_err(|e| TlsError::Io(format!("Falcon verification failed: {}", e)))??;

        FalconMetrics::record_verify(level, start.elapsed());
        FalconMetrics::increment_operations(level, "verify");

        Ok(result)
    }

    fn verify_impl(
        level: FalconSecurityLevel,
        pk_bytes: &[u8],
        message: &[u8],
        sig_bytes: &[u8],
    ) -> Result<bool, TlsError> {
        match level {
            FalconSecurityLevel::Falcon512 => {
                let pk = pqcrypto_falcon::falcon512::PublicKey::from_bytes(pk_bytes)
                    .map_err(|_| TlsError::Policy("Invalid Falcon-512 public key".into()))?;
                let sig = pqcrypto_falcon::falcon512::DetachedSignature::from_bytes(sig_bytes)
                    .map_err(|_| TlsError::Policy("Invalid Falcon-512 signature".into()))?;
                Ok(pqcrypto_falcon::falcon512::verify_detached_signature(&sig, message, &pk).is_ok())
            }
            FalconSecurityLevel::Falcon1024 => {
                let pk = pqcrypto_falcon::falcon1024::PublicKey::from_bytes(pk_bytes)
                    .map_err(|_| TlsError::Policy("Invalid Falcon-1024 public key".into()))?;
                let sig = pqcrypto_falcon::falcon1024::DetachedSignature::from_bytes(sig_bytes)
                    .map_err(|_| TlsError::Policy("Invalid Falcon-1024 signature".into()))?;
                Ok(pqcrypto_falcon::falcon1024::verify_detached_signature(&sig, message, &pk).is_ok())
            }
        }
    }

    /// Get the default security level
    pub fn default_level(&self) -> FalconSecurityLevel {
        self.default_level
    }

    /// Check if FIPS mode is enabled
    pub fn is_fips_mode(&self) -> bool {
        self.fips_mode
    }

    /// Get domain configuration if set
    pub fn domain_config(&self) -> Option<&FalconDomainConfig> {
        self.domain_config.as_ref()
    }
}

impl Default for FalconEngine {
    fn default() -> Self {
        Self::for_bandwidth_constrained()
    }
}

/// Batch verification for high-throughput scenarios (automotive V2X)
pub struct FalconBatchVerifier {
    level: FalconSecurityLevel,
    pending: Vec<(FalconPublicKey, Vec<u8>, FalconSignature)>,
    batch_size: usize,
}

impl FalconBatchVerifier {
    /// Create a new batch verifier
    pub fn new(level: FalconSecurityLevel, batch_size: usize) -> Self {
        Self {
            level,
            pending: Vec::with_capacity(batch_size),
            batch_size,
        }
    }

    /// Create for automotive V2X (optimized for 1000+ msg/sec)
    pub fn for_automotive_v2x() -> Self {
        Self::new(FalconSecurityLevel::Falcon512, 64)
    }

    /// Add a verification task to the batch
    pub fn add(
        &mut self,
        public_key: FalconPublicKey,
        message: Vec<u8>,
        signature: FalconSignature,
    ) -> Result<(), TlsError> {
        if public_key.level() != self.level || signature.level() != self.level {
            return Err(TlsError::Policy("Security level mismatch".into()));
        }
        self.pending.push((public_key, message, signature));
        Ok(())
    }

    /// Check if batch is ready to verify
    pub fn is_ready(&self) -> bool {
        self.pending.len() >= self.batch_size
    }

    /// Get current batch size
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Verify all pending signatures in parallel
    /// Returns results in the same order as they were added
    pub async fn verify_batch(&mut self) -> Result<Vec<bool>, TlsError> {
        use rayon::prelude::*;

        let _span = span!(Level::INFO, "falcon_batch_verify", count = self.pending.len()).entered();
        let start = Instant::now();

        let pending = std::mem::take(&mut self.pending);
        let level = self.level;

        let results = tokio::task::spawn_blocking(move || {
            pending
                .into_par_iter()
                .map(|(pk, msg, sig)| {
                    FalconEngine::verify_impl(level, pk.as_bytes(), &msg, sig.as_bytes())
                        .unwrap_or(Ok(false))
                        .unwrap_or(false)
                })
                .collect::<Vec<_>>()
        })
        .await
        .map_err(|e| TlsError::Io(format!("Batch verification failed: {}", e)))?;

        let elapsed = start.elapsed();
        let count = results.len();
        let verified = results.iter().filter(|&&v| v).count();

        info!(
            "Batch verified {} signatures in {}ms ({} valid, {} invalid)",
            count,
            elapsed.as_millis(),
            verified,
            count - verified
        );

        histogram!("falcon_batch_verify_duration_seconds", "level" => level.name())
            .record(elapsed.as_secs_f64());
        counter!("falcon_batch_verify_total", "level" => level.name())
            .increment(count as u64);

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_falcon_512_roundtrip() {
        let engine = FalconEngine::new(FalconSecurityLevel::Falcon512);
        let keypair = engine.generate_keypair().await.unwrap();

        assert_eq!(keypair.level, FalconSecurityLevel::Falcon512);
        assert_eq!(keypair.public_key.as_bytes().len(), 897);

        let message = b"Test message for Falcon signature";
        let signature = engine.sign(&keypair.private_key, message).await.unwrap();

        // Falcon-512 signatures are typically ~666 bytes
        assert!(signature.size() <= 690);

        let valid = engine.verify(&keypair.public_key, message, &signature).await.unwrap();
        assert!(valid);

        // Verify with wrong message fails
        let invalid = engine.verify(&keypair.public_key, b"Wrong message", &signature).await.unwrap();
        assert!(!invalid);
    }

    #[tokio::test]
    async fn test_falcon_1024_roundtrip() {
        let engine = FalconEngine::for_maximum_security();
        let keypair = engine.generate_keypair().await.unwrap();

        assert_eq!(keypair.level, FalconSecurityLevel::Falcon1024);
        assert_eq!(keypair.public_key.as_bytes().len(), 1793);

        let message = b"Test message for Falcon-1024 signature";
        let signature = engine.sign(&keypair.private_key, message).await.unwrap();

        // Falcon-1024 signatures are typically ~1280 bytes
        assert!(signature.size() <= 1330);

        let valid = engine.verify(&keypair.public_key, message, &signature).await.unwrap();
        assert!(valid);
    }

    #[tokio::test]
    async fn test_batch_verification() {
        let engine = FalconEngine::new(FalconSecurityLevel::Falcon512);
        let mut batch = FalconBatchVerifier::for_automotive_v2x();

        // Generate multiple signatures
        for i in 0..10 {
            let keypair = engine.generate_keypair().await.unwrap();
            let message = format!("Message {}", i).into_bytes();
            let signature = engine.sign(&keypair.private_key, &message).await.unwrap();

            batch.add(keypair.public_key, message, signature).unwrap();
        }

        assert_eq!(batch.len(), 10);

        let results = batch.verify_batch().await.unwrap();
        assert_eq!(results.len(), 10);
        assert!(results.iter().all(|&v| v));
    }

    #[tokio::test]
    async fn test_domain_config_automotive() {
        let config = FalconDomainConfig::for_automotive_v2x();
        let engine = FalconEngine::with_domain_config(config);

        let keypair = engine.generate_keypair().await.unwrap();
        let message = b"V2X Basic Safety Message";
        let signature = engine.sign(&keypair.private_key, message).await.unwrap();

        // Should use Falcon-512 for smaller signatures
        assert_eq!(keypair.level, FalconSecurityLevel::Falcon512);
        assert!(signature.size() < 700); // Much smaller than Dilithium's 2420 bytes
    }

    #[test]
    fn test_signature_size_advantage() {
        let level = FalconSecurityLevel::Falcon512;
        let advantage = level.size_advantage_vs_mldsa();
        assert!(advantage > 3.5); // ~3.6x smaller
    }

    #[test]
    fn test_security_level_properties() {
        let level = FalconSecurityLevel::Falcon512;
        assert_eq!(level.nist_level(), 1);
        assert_eq!(level.name(), "Falcon-512");
        assert_eq!(level.public_key_size(), 897);
        assert_eq!(level.signature_size_typical(), 666);
    }
}
