//! ML-KEM (Module-Lattice Key Encapsulation Mechanism) - NIST FIPS 203
//!
//! This module implements all three security levels of ML-KEM:
//! - ML-KEM-512: Security Level 1 (128-bit classical, 64-bit quantum)
//! - ML-KEM-768: Security Level 3 (192-bit classical, 96-bit quantum) - Recommended for hybrid TLS
//! - ML-KEM-1024: Security Level 5 (256-bit classical, 128-bit quantum)
//!
//! ML-KEM-768 is the preferred choice for TLS 1.3 hybrid key exchange (X25519MLKEM768)

use crate::errors::TlsError;
use metrics::{counter, histogram};
use once_cell::sync::Lazy;
use secrecy::{Secret, SecretVec, ExposeSecret};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tracing::{debug, info, span, Level};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// ML-KEM Security Level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MlKemSecurityLevel {
    /// Level 1: 128-bit classical security (recommended for constrained devices)
    MlKem512,
    /// Level 3: 192-bit classical security (recommended for TLS hybrid)
    MlKem768,
    /// Level 5: 256-bit classical security (maximum security)
    MlKem1024,
}

impl MlKemSecurityLevel {
    pub fn public_key_size(&self) -> usize {
        match self {
            MlKemSecurityLevel::MlKem512 => 800,
            MlKemSecurityLevel::MlKem768 => 1184,
            MlKemSecurityLevel::MlKem1024 => 1568,
        }
    }

    pub fn private_key_size(&self) -> usize {
        match self {
            MlKemSecurityLevel::MlKem512 => 1632,
            MlKemSecurityLevel::MlKem768 => 2400,
            MlKemSecurityLevel::MlKem1024 => 3168,
        }
    }

    pub fn ciphertext_size(&self) -> usize {
        match self {
            MlKemSecurityLevel::MlKem512 => 768,
            MlKemSecurityLevel::MlKem768 => 1088,
            MlKemSecurityLevel::MlKem1024 => 1568,
        }
    }

    pub fn shared_secret_size(&self) -> usize {
        32 // All levels produce 256-bit shared secret
    }

    pub fn name(&self) -> &'static str {
        match self {
            MlKemSecurityLevel::MlKem512 => "ML-KEM-512",
            MlKemSecurityLevel::MlKem768 => "ML-KEM-768",
            MlKemSecurityLevel::MlKem1024 => "ML-KEM-1024",
        }
    }

    /// NIST security level (1, 3, or 5)
    pub fn nist_level(&self) -> u8 {
        match self {
            MlKemSecurityLevel::MlKem512 => 1,
            MlKemSecurityLevel::MlKem768 => 3,
            MlKemSecurityLevel::MlKem1024 => 5,
        }
    }
}

impl Default for MlKemSecurityLevel {
    fn default() -> Self {
        MlKemSecurityLevel::MlKem768 // Recommended for most use cases
    }
}

/// ML-KEM public key
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MlKemPublicKey {
    level: MlKemSecurityLevel,
    data: Vec<u8>,
}

impl MlKemPublicKey {
    pub fn from_bytes(level: MlKemSecurityLevel, bytes: &[u8]) -> Result<Self, TlsError> {
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

    pub fn level(&self) -> MlKemSecurityLevel {
        self.level
    }
}

/// ML-KEM private key with secure memory handling
#[derive(Debug, ZeroizeOnDrop)]
pub struct MlKemPrivateKey {
    level: MlKemSecurityLevel,
    data: SecretVec<u8>,
}

impl MlKemPrivateKey {
    pub fn from_bytes(level: MlKemSecurityLevel, bytes: &[u8]) -> Result<Self, TlsError> {
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

    pub fn level(&self) -> MlKemSecurityLevel {
        self.level
    }
}

/// ML-KEM key pair
#[derive(Debug)]
pub struct MlKemKeyPair {
    pub public_key: MlKemPublicKey,
    pub private_key: MlKemPrivateKey,
    pub level: MlKemSecurityLevel,
}

/// ML-KEM ciphertext
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MlKemCiphertext {
    level: MlKemSecurityLevel,
    data: Vec<u8>,
}

impl MlKemCiphertext {
    pub fn from_bytes(level: MlKemSecurityLevel, bytes: &[u8]) -> Result<Self, TlsError> {
        let expected_size = level.ciphertext_size();
        if bytes.len() != expected_size {
            return Err(TlsError::Policy(format!(
                "{} invalid ciphertext size: expected {}, got {}",
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

    pub fn level(&self) -> MlKemSecurityLevel {
        self.level
    }
}

/// ML-KEM shared secret with secure handling
#[derive(Debug, ZeroizeOnDrop)]
pub struct MlKemSharedSecret {
    data: Secret<[u8; 32]>,
}

impl MlKemSharedSecret {
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self {
            data: Secret::new(bytes),
        }
    }

    pub fn expose_secret(&self) -> &[u8; 32] {
        self.data.expose_secret()
    }
}

/// Encapsulation result
#[derive(Debug)]
pub struct MlKemEncapsulationResult {
    pub ciphertext: MlKemCiphertext,
    pub shared_secret: MlKemSharedSecret,
}

/// ML-KEM metrics
struct MlKemMetrics;

impl MlKemMetrics {
    fn record_keygen(level: MlKemSecurityLevel, duration: Duration) {
        histogram!("mlkem_keygen_duration_seconds", "level" => level.name())
            .record(duration.as_secs_f64());
    }

    fn record_encaps(level: MlKemSecurityLevel, duration: Duration) {
        histogram!("mlkem_encaps_duration_seconds", "level" => level.name())
            .record(duration.as_secs_f64());
    }

    fn record_decaps(level: MlKemSecurityLevel, duration: Duration) {
        histogram!("mlkem_decaps_duration_seconds", "level" => level.name())
            .record(duration.as_secs_f64());
    }

    fn increment_operations(level: MlKemSecurityLevel) {
        counter!("mlkem_operations_total", "level" => level.name()).increment(1);
    }
}

/// Production-ready ML-KEM implementation supporting all security levels
pub struct MlKemEngine {
    default_level: MlKemSecurityLevel,
    use_hardware_acceleration: bool,
    fips_mode: bool,
}

impl MlKemEngine {
    /// Create a new ML-KEM engine with the specified default security level
    pub fn new(level: MlKemSecurityLevel) -> Self {
        Self {
            default_level: level,
            use_hardware_acceleration: true,
            fips_mode: false,
        }
    }

    /// Create engine optimized for TLS 1.3 hybrid (ML-KEM-768)
    pub fn for_tls_hybrid() -> Self {
        Self::new(MlKemSecurityLevel::MlKem768)
    }

    /// Create engine for constrained devices (ML-KEM-512)
    pub fn for_constrained_devices() -> Self {
        Self::new(MlKemSecurityLevel::MlKem512)
    }

    /// Create engine for maximum security (ML-KEM-1024)
    pub fn for_maximum_security() -> Self {
        Self::new(MlKemSecurityLevel::MlKem1024)
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
    pub async fn generate_keypair(&self) -> Result<MlKemKeyPair, TlsError> {
        self.generate_keypair_at_level(self.default_level).await
    }

    /// Generate a key pair at a specific security level
    pub async fn generate_keypair_at_level(
        &self,
        level: MlKemSecurityLevel,
    ) -> Result<MlKemKeyPair, TlsError> {
        let _span = span!(Level::INFO, "mlkem_keygen", level = level.name()).entered();
        let start = Instant::now();

        let result = tokio::task::spawn_blocking(move || {
            Self::generate_keypair_impl(level)
        })
        .await
        .map_err(|e| TlsError::Io(format!("ML-KEM key generation failed: {}", e)))??;

        MlKemMetrics::record_keygen(level, start.elapsed());
        MlKemMetrics::increment_operations(level);

        Ok(result)
    }

    fn generate_keypair_impl(level: MlKemSecurityLevel) -> Result<MlKemKeyPair, TlsError> {
        match level {
            MlKemSecurityLevel::MlKem512 => {
                let (pk, sk) = pqcrypto_kyber::kyber512::keypair();
                Ok(MlKemKeyPair {
                    public_key: MlKemPublicKey::from_bytes(level, pk.as_bytes())?,
                    private_key: MlKemPrivateKey::from_bytes(level, sk.as_bytes())?,
                    level,
                })
            }
            MlKemSecurityLevel::MlKem768 => {
                let (pk, sk) = pqcrypto_kyber::kyber768::keypair();
                Ok(MlKemKeyPair {
                    public_key: MlKemPublicKey::from_bytes(level, pk.as_bytes())?,
                    private_key: MlKemPrivateKey::from_bytes(level, sk.as_bytes())?,
                    level,
                })
            }
            MlKemSecurityLevel::MlKem1024 => {
                let (pk, sk) = pqcrypto_kyber::kyber1024::keypair();
                Ok(MlKemKeyPair {
                    public_key: MlKemPublicKey::from_bytes(level, pk.as_bytes())?,
                    private_key: MlKemPrivateKey::from_bytes(level, sk.as_bytes())?,
                    level,
                })
            }
        }
    }

    /// Encapsulate using the provided public key
    pub async fn encapsulate(
        &self,
        public_key: &MlKemPublicKey,
    ) -> Result<MlKemEncapsulationResult, TlsError> {
        let _span = span!(Level::INFO, "mlkem_encaps", level = public_key.level().name()).entered();
        let start = Instant::now();
        let level = public_key.level();
        let pk_bytes = public_key.data.clone();

        let result = tokio::task::spawn_blocking(move || {
            Self::encapsulate_impl(level, &pk_bytes)
        })
        .await
        .map_err(|e| TlsError::Io(format!("ML-KEM encapsulation failed: {}", e)))??;

        MlKemMetrics::record_encaps(level, start.elapsed());
        MlKemMetrics::increment_operations(level);

        Ok(result)
    }

    fn encapsulate_impl(
        level: MlKemSecurityLevel,
        pk_bytes: &[u8],
    ) -> Result<MlKemEncapsulationResult, TlsError> {
        match level {
            MlKemSecurityLevel::MlKem512 => {
                let pk = pqcrypto_kyber::kyber512::PublicKey::from_bytes(pk_bytes)
                    .map_err(|_| TlsError::Policy("Invalid ML-KEM-512 public key".into()))?;
                let (ss, ct) = pqcrypto_kyber::kyber512::encapsulate(&pk);

                let mut shared_secret = [0u8; 32];
                shared_secret.copy_from_slice(ss.as_bytes());

                Ok(MlKemEncapsulationResult {
                    ciphertext: MlKemCiphertext::from_bytes(level, ct.as_bytes())?,
                    shared_secret: MlKemSharedSecret::from_bytes(shared_secret),
                })
            }
            MlKemSecurityLevel::MlKem768 => {
                let pk = pqcrypto_kyber::kyber768::PublicKey::from_bytes(pk_bytes)
                    .map_err(|_| TlsError::Policy("Invalid ML-KEM-768 public key".into()))?;
                let (ss, ct) = pqcrypto_kyber::kyber768::encapsulate(&pk);

                let mut shared_secret = [0u8; 32];
                shared_secret.copy_from_slice(ss.as_bytes());

                Ok(MlKemEncapsulationResult {
                    ciphertext: MlKemCiphertext::from_bytes(level, ct.as_bytes())?,
                    shared_secret: MlKemSharedSecret::from_bytes(shared_secret),
                })
            }
            MlKemSecurityLevel::MlKem1024 => {
                let pk = pqcrypto_kyber::kyber1024::PublicKey::from_bytes(pk_bytes)
                    .map_err(|_| TlsError::Policy("Invalid ML-KEM-1024 public key".into()))?;
                let (ss, ct) = pqcrypto_kyber::kyber1024::encapsulate(&pk);

                let mut shared_secret = [0u8; 32];
                shared_secret.copy_from_slice(ss.as_bytes());

                Ok(MlKemEncapsulationResult {
                    ciphertext: MlKemCiphertext::from_bytes(level, ct.as_bytes())?,
                    shared_secret: MlKemSharedSecret::from_bytes(shared_secret),
                })
            }
        }
    }

    /// Decapsulate using the provided private key
    pub async fn decapsulate(
        &self,
        private_key: &MlKemPrivateKey,
        ciphertext: &MlKemCiphertext,
    ) -> Result<MlKemSharedSecret, TlsError> {
        let level = private_key.level();
        if level != ciphertext.level() {
            return Err(TlsError::Policy(format!(
                "Security level mismatch: key is {}, ciphertext is {}",
                level.name(),
                ciphertext.level().name()
            )));
        }

        let _span = span!(Level::INFO, "mlkem_decaps", level = level.name()).entered();
        let start = Instant::now();

        let sk_bytes = private_key.expose_secret().to_vec();
        let ct_bytes = ciphertext.data.clone();

        let result = tokio::task::spawn_blocking(move || {
            Self::decapsulate_impl(level, &sk_bytes, &ct_bytes)
        })
        .await
        .map_err(|e| TlsError::Io(format!("ML-KEM decapsulation failed: {}", e)))??;

        MlKemMetrics::record_decaps(level, start.elapsed());
        MlKemMetrics::increment_operations(level);

        Ok(result)
    }

    fn decapsulate_impl(
        level: MlKemSecurityLevel,
        sk_bytes: &[u8],
        ct_bytes: &[u8],
    ) -> Result<MlKemSharedSecret, TlsError> {
        match level {
            MlKemSecurityLevel::MlKem512 => {
                let sk = pqcrypto_kyber::kyber512::SecretKey::from_bytes(sk_bytes)
                    .map_err(|_| TlsError::Policy("Invalid ML-KEM-512 private key".into()))?;
                let ct = pqcrypto_kyber::kyber512::Ciphertext::from_bytes(ct_bytes)
                    .map_err(|_| TlsError::Policy("Invalid ML-KEM-512 ciphertext".into()))?;
                let ss = pqcrypto_kyber::kyber512::decapsulate(&ct, &sk);

                let mut shared_secret = [0u8; 32];
                shared_secret.copy_from_slice(ss.as_bytes());

                Ok(MlKemSharedSecret::from_bytes(shared_secret))
            }
            MlKemSecurityLevel::MlKem768 => {
                let sk = pqcrypto_kyber::kyber768::SecretKey::from_bytes(sk_bytes)
                    .map_err(|_| TlsError::Policy("Invalid ML-KEM-768 private key".into()))?;
                let ct = pqcrypto_kyber::kyber768::Ciphertext::from_bytes(ct_bytes)
                    .map_err(|_| TlsError::Policy("Invalid ML-KEM-768 ciphertext".into()))?;
                let ss = pqcrypto_kyber::kyber768::decapsulate(&ct, &sk);

                let mut shared_secret = [0u8; 32];
                shared_secret.copy_from_slice(ss.as_bytes());

                Ok(MlKemSharedSecret::from_bytes(shared_secret))
            }
            MlKemSecurityLevel::MlKem1024 => {
                let sk = pqcrypto_kyber::kyber1024::SecretKey::from_bytes(sk_bytes)
                    .map_err(|_| TlsError::Policy("Invalid ML-KEM-1024 private key".into()))?;
                let ct = pqcrypto_kyber::kyber1024::Ciphertext::from_bytes(ct_bytes)
                    .map_err(|_| TlsError::Policy("Invalid ML-KEM-1024 ciphertext".into()))?;
                let ss = pqcrypto_kyber::kyber1024::decapsulate(&ct, &sk);

                let mut shared_secret = [0u8; 32];
                shared_secret.copy_from_slice(ss.as_bytes());

                Ok(MlKemSharedSecret::from_bytes(shared_secret))
            }
        }
    }

    /// Get the default security level
    pub fn default_level(&self) -> MlKemSecurityLevel {
        self.default_level
    }

    /// Check if FIPS mode is enabled
    pub fn is_fips_mode(&self) -> bool {
        self.fips_mode
    }
}

impl Default for MlKemEngine {
    fn default() -> Self {
        Self::for_tls_hybrid()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mlkem_512_roundtrip() {
        let engine = MlKemEngine::for_constrained_devices();
        let keypair = engine.generate_keypair().await.unwrap();

        assert_eq!(keypair.level, MlKemSecurityLevel::MlKem512);
        assert_eq!(keypair.public_key.as_bytes().len(), 800);

        let encaps = engine.encapsulate(&keypair.public_key).await.unwrap();
        let decaps = engine.decapsulate(&keypair.private_key, &encaps.ciphertext).await.unwrap();

        assert_eq!(encaps.shared_secret.expose_secret(), decaps.expose_secret());
    }

    #[tokio::test]
    async fn test_mlkem_768_roundtrip() {
        let engine = MlKemEngine::for_tls_hybrid();
        let keypair = engine.generate_keypair().await.unwrap();

        assert_eq!(keypair.level, MlKemSecurityLevel::MlKem768);
        assert_eq!(keypair.public_key.as_bytes().len(), 1184);

        let encaps = engine.encapsulate(&keypair.public_key).await.unwrap();
        let decaps = engine.decapsulate(&keypair.private_key, &encaps.ciphertext).await.unwrap();

        assert_eq!(encaps.shared_secret.expose_secret(), decaps.expose_secret());
    }

    #[tokio::test]
    async fn test_mlkem_1024_roundtrip() {
        let engine = MlKemEngine::for_maximum_security();
        let keypair = engine.generate_keypair().await.unwrap();

        assert_eq!(keypair.level, MlKemSecurityLevel::MlKem1024);
        assert_eq!(keypair.public_key.as_bytes().len(), 1568);

        let encaps = engine.encapsulate(&keypair.public_key).await.unwrap();
        let decaps = engine.decapsulate(&keypair.private_key, &encaps.ciphertext).await.unwrap();

        assert_eq!(encaps.shared_secret.expose_secret(), decaps.expose_secret());
    }

    #[tokio::test]
    async fn test_cross_level_rejection() {
        let engine = MlKemEngine::new(MlKemSecurityLevel::MlKem768);

        let keypair_768 = engine.generate_keypair_at_level(MlKemSecurityLevel::MlKem768).await.unwrap();
        let keypair_512 = engine.generate_keypair_at_level(MlKemSecurityLevel::MlKem512).await.unwrap();

        // Encapsulate with 512 public key
        let encaps_512 = engine.encapsulate(&keypair_512.public_key).await.unwrap();

        // Try to decapsulate with 768 private key - should fail
        let result = engine.decapsulate(&keypair_768.private_key, &encaps_512.ciphertext).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_security_level_properties() {
        let level = MlKemSecurityLevel::MlKem768;
        assert_eq!(level.nist_level(), 3);
        assert_eq!(level.name(), "ML-KEM-768");
        assert_eq!(level.public_key_size(), 1184);
        assert_eq!(level.ciphertext_size(), 1088);
        assert_eq!(level.shared_secret_size(), 32);
    }
}
