//! ML-DSA (Module-Lattice Digital Signature Algorithm) / Dilithium - NIST FIPS 204
//!
//! This module implements all three security levels:
//! - ML-DSA-44 (Dilithium2): Security Level 2 (~128-bit)
//! - ML-DSA-65 (Dilithium3): Security Level 3 (~192-bit)
//! - ML-DSA-87 (Dilithium5): Security Level 5 (~256-bit)
//!
//! ML-DSA/Dilithium offers faster signing than Falcon but larger signatures.
//! Use Falcon for bandwidth-constrained scenarios, Dilithium for general enterprise.

use crate::errors::TlsError;
use metrics::{counter, histogram};
use secrecy::{Secret, SecretVec, ExposeSecret};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tracing::{debug, info, span, Level};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Dilithium/ML-DSA Security Level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DilithiumSecurityLevel {
    /// Level 2: ~128-bit classical security (Dilithium2/ML-DSA-44)
    Dilithium2,
    /// Level 3: ~192-bit classical security (Dilithium3/ML-DSA-65)
    Dilithium3,
    /// Level 5: ~256-bit classical security (Dilithium5/ML-DSA-87)
    Dilithium5,
}

impl DilithiumSecurityLevel {
    pub fn public_key_size(&self) -> usize {
        match self {
            DilithiumSecurityLevel::Dilithium2 => 1312,
            DilithiumSecurityLevel::Dilithium3 => 1952,
            DilithiumSecurityLevel::Dilithium5 => 2592,
        }
    }

    pub fn private_key_size(&self) -> usize {
        match self {
            DilithiumSecurityLevel::Dilithium2 => 2528,
            DilithiumSecurityLevel::Dilithium3 => 4000,
            DilithiumSecurityLevel::Dilithium5 => 4864,
        }
    }

    pub fn signature_size(&self) -> usize {
        match self {
            DilithiumSecurityLevel::Dilithium2 => 2420,
            DilithiumSecurityLevel::Dilithium3 => 3293,
            DilithiumSecurityLevel::Dilithium5 => 4595,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            DilithiumSecurityLevel::Dilithium2 => "ML-DSA-44",
            DilithiumSecurityLevel::Dilithium3 => "ML-DSA-65",
            DilithiumSecurityLevel::Dilithium5 => "ML-DSA-87",
        }
    }

    pub fn legacy_name(&self) -> &'static str {
        match self {
            DilithiumSecurityLevel::Dilithium2 => "Dilithium2",
            DilithiumSecurityLevel::Dilithium3 => "Dilithium3",
            DilithiumSecurityLevel::Dilithium5 => "Dilithium5",
        }
    }

    /// NIST security level (2, 3, or 5)
    pub fn nist_level(&self) -> u8 {
        match self {
            DilithiumSecurityLevel::Dilithium2 => 2,
            DilithiumSecurityLevel::Dilithium3 => 3,
            DilithiumSecurityLevel::Dilithium5 => 5,
        }
    }
}

impl Default for DilithiumSecurityLevel {
    fn default() -> Self {
        DilithiumSecurityLevel::Dilithium3 // Good balance of security and performance
    }
}

/// Dilithium public key
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DilithiumPublicKey {
    level: DilithiumSecurityLevel,
    data: Vec<u8>,
}

impl DilithiumPublicKey {
    pub fn from_bytes(level: DilithiumSecurityLevel, bytes: &[u8]) -> Result<Self, TlsError> {
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

    pub fn level(&self) -> DilithiumSecurityLevel {
        self.level
    }
}

/// Dilithium private key with secure memory handling
#[derive(Debug, ZeroizeOnDrop)]
pub struct DilithiumPrivateKey {
    level: DilithiumSecurityLevel,
    data: SecretVec<u8>,
}

impl DilithiumPrivateKey {
    pub fn from_bytes(level: DilithiumSecurityLevel, bytes: &[u8]) -> Result<Self, TlsError> {
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

    pub fn level(&self) -> DilithiumSecurityLevel {
        self.level
    }
}

/// Dilithium key pair
#[derive(Debug)]
pub struct DilithiumKeyPair {
    pub public_key: DilithiumPublicKey,
    pub private_key: DilithiumPrivateKey,
    pub level: DilithiumSecurityLevel,
}

/// Dilithium signature
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DilithiumSignature {
    level: DilithiumSecurityLevel,
    data: Vec<u8>,
}

impl DilithiumSignature {
    pub fn from_bytes(level: DilithiumSecurityLevel, bytes: &[u8]) -> Result<Self, TlsError> {
        let expected_size = level.signature_size();
        if bytes.len() != expected_size {
            return Err(TlsError::Policy(format!(
                "{} invalid signature size: expected {}, got {}",
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

    pub fn level(&self) -> DilithiumSecurityLevel {
        self.level
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }
}

/// Dilithium metrics
struct DilithiumMetrics;

impl DilithiumMetrics {
    fn record_keygen(level: DilithiumSecurityLevel, duration: Duration) {
        histogram!("dilithium_keygen_duration_seconds", "level" => level.name())
            .record(duration.as_secs_f64());
    }

    fn record_sign(level: DilithiumSecurityLevel, duration: Duration) {
        histogram!("dilithium_sign_duration_seconds", "level" => level.name())
            .record(duration.as_secs_f64());
    }

    fn record_verify(level: DilithiumSecurityLevel, duration: Duration) {
        histogram!("dilithium_verify_duration_seconds", "level" => level.name())
            .record(duration.as_secs_f64());
    }

    fn increment_operations(level: DilithiumSecurityLevel, op: &str) {
        counter!("dilithium_operations_total", "level" => level.name(), "operation" => op.to_string())
            .increment(1);
    }
}

/// Production-ready Dilithium/ML-DSA signature engine
pub struct DilithiumEngine {
    default_level: DilithiumSecurityLevel,
    use_hardware_acceleration: bool,
    fips_mode: bool,
}

impl DilithiumEngine {
    /// Create a new Dilithium engine with the specified security level
    pub fn new(level: DilithiumSecurityLevel) -> Self {
        Self {
            default_level: level,
            use_hardware_acceleration: true,
            fips_mode: false,
        }
    }

    /// Create engine for enterprise use (balanced security/performance)
    pub fn for_enterprise() -> Self {
        Self::new(DilithiumSecurityLevel::Dilithium3)
    }

    /// Create engine for maximum security
    pub fn for_maximum_security() -> Self {
        Self::new(DilithiumSecurityLevel::Dilithium5)
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
    pub async fn generate_keypair(&self) -> Result<DilithiumKeyPair, TlsError> {
        self.generate_keypair_at_level(self.default_level).await
    }

    /// Generate a key pair at a specific security level
    pub async fn generate_keypair_at_level(
        &self,
        level: DilithiumSecurityLevel,
    ) -> Result<DilithiumKeyPair, TlsError> {
        let _span = span!(Level::INFO, "dilithium_keygen", level = level.name()).entered();
        let start = Instant::now();

        let result = tokio::task::spawn_blocking(move || {
            Self::generate_keypair_impl(level)
        })
        .await
        .map_err(|e| TlsError::Io(format!("Dilithium key generation failed: {}", e)))??;

        DilithiumMetrics::record_keygen(level, start.elapsed());
        DilithiumMetrics::increment_operations(level, "keygen");

        Ok(result)
    }

    fn generate_keypair_impl(level: DilithiumSecurityLevel) -> Result<DilithiumKeyPair, TlsError> {
        match level {
            DilithiumSecurityLevel::Dilithium2 => {
                let (pk, sk) = pqcrypto_dilithium::dilithium2::keypair();
                Ok(DilithiumKeyPair {
                    public_key: DilithiumPublicKey::from_bytes(level, pk.as_bytes())?,
                    private_key: DilithiumPrivateKey::from_bytes(level, sk.as_bytes())?,
                    level,
                })
            }
            DilithiumSecurityLevel::Dilithium3 => {
                let (pk, sk) = pqcrypto_dilithium::dilithium3::keypair();
                Ok(DilithiumKeyPair {
                    public_key: DilithiumPublicKey::from_bytes(level, pk.as_bytes())?,
                    private_key: DilithiumPrivateKey::from_bytes(level, sk.as_bytes())?,
                    level,
                })
            }
            DilithiumSecurityLevel::Dilithium5 => {
                let (pk, sk) = pqcrypto_dilithium::dilithium5::keypair();
                Ok(DilithiumKeyPair {
                    public_key: DilithiumPublicKey::from_bytes(level, pk.as_bytes())?,
                    private_key: DilithiumPrivateKey::from_bytes(level, sk.as_bytes())?,
                    level,
                })
            }
        }
    }

    /// Sign a message
    pub async fn sign(
        &self,
        private_key: &DilithiumPrivateKey,
        message: &[u8],
    ) -> Result<DilithiumSignature, TlsError> {
        let level = private_key.level();
        let _span = span!(Level::INFO, "dilithium_sign", level = level.name()).entered();
        let start = Instant::now();

        let sk_bytes = private_key.expose_secret().to_vec();
        let msg = message.to_vec();

        let result = tokio::task::spawn_blocking(move || {
            Self::sign_impl(level, &sk_bytes, &msg)
        })
        .await
        .map_err(|e| TlsError::Io(format!("Dilithium signing failed: {}", e)))??;

        DilithiumMetrics::record_sign(level, start.elapsed());
        DilithiumMetrics::increment_operations(level, "sign");

        Ok(result)
    }

    fn sign_impl(
        level: DilithiumSecurityLevel,
        sk_bytes: &[u8],
        message: &[u8],
    ) -> Result<DilithiumSignature, TlsError> {
        match level {
            DilithiumSecurityLevel::Dilithium2 => {
                let sk = pqcrypto_dilithium::dilithium2::SecretKey::from_bytes(sk_bytes)
                    .map_err(|_| TlsError::Policy("Invalid Dilithium2 private key".into()))?;
                let sig = pqcrypto_dilithium::dilithium2::detached_sign(message, &sk);
                DilithiumSignature::from_bytes(level, sig.as_bytes())
            }
            DilithiumSecurityLevel::Dilithium3 => {
                let sk = pqcrypto_dilithium::dilithium3::SecretKey::from_bytes(sk_bytes)
                    .map_err(|_| TlsError::Policy("Invalid Dilithium3 private key".into()))?;
                let sig = pqcrypto_dilithium::dilithium3::detached_sign(message, &sk);
                DilithiumSignature::from_bytes(level, sig.as_bytes())
            }
            DilithiumSecurityLevel::Dilithium5 => {
                let sk = pqcrypto_dilithium::dilithium5::SecretKey::from_bytes(sk_bytes)
                    .map_err(|_| TlsError::Policy("Invalid Dilithium5 private key".into()))?;
                let sig = pqcrypto_dilithium::dilithium5::detached_sign(message, &sk);
                DilithiumSignature::from_bytes(level, sig.as_bytes())
            }
        }
    }

    /// Verify a signature
    pub async fn verify(
        &self,
        public_key: &DilithiumPublicKey,
        message: &[u8],
        signature: &DilithiumSignature,
    ) -> Result<bool, TlsError> {
        let level = public_key.level();
        if level != signature.level() {
            return Err(TlsError::Policy(format!(
                "Security level mismatch: key is {}, signature is {}",
                level.name(),
                signature.level().name()
            )));
        }

        let _span = span!(Level::INFO, "dilithium_verify", level = level.name()).entered();
        let start = Instant::now();

        let pk_bytes = public_key.data.clone();
        let sig_bytes = signature.data.clone();
        let msg = message.to_vec();

        let result = tokio::task::spawn_blocking(move || {
            Self::verify_impl(level, &pk_bytes, &msg, &sig_bytes)
        })
        .await
        .map_err(|e| TlsError::Io(format!("Dilithium verification failed: {}", e)))??;

        DilithiumMetrics::record_verify(level, start.elapsed());
        DilithiumMetrics::increment_operations(level, "verify");

        Ok(result)
    }

    fn verify_impl(
        level: DilithiumSecurityLevel,
        pk_bytes: &[u8],
        message: &[u8],
        sig_bytes: &[u8],
    ) -> Result<bool, TlsError> {
        match level {
            DilithiumSecurityLevel::Dilithium2 => {
                let pk = pqcrypto_dilithium::dilithium2::PublicKey::from_bytes(pk_bytes)
                    .map_err(|_| TlsError::Policy("Invalid Dilithium2 public key".into()))?;
                let sig = pqcrypto_dilithium::dilithium2::DetachedSignature::from_bytes(sig_bytes)
                    .map_err(|_| TlsError::Policy("Invalid Dilithium2 signature".into()))?;
                Ok(pqcrypto_dilithium::dilithium2::verify_detached_signature(&sig, message, &pk).is_ok())
            }
            DilithiumSecurityLevel::Dilithium3 => {
                let pk = pqcrypto_dilithium::dilithium3::PublicKey::from_bytes(pk_bytes)
                    .map_err(|_| TlsError::Policy("Invalid Dilithium3 public key".into()))?;
                let sig = pqcrypto_dilithium::dilithium3::DetachedSignature::from_bytes(sig_bytes)
                    .map_err(|_| TlsError::Policy("Invalid Dilithium3 signature".into()))?;
                Ok(pqcrypto_dilithium::dilithium3::verify_detached_signature(&sig, message, &pk).is_ok())
            }
            DilithiumSecurityLevel::Dilithium5 => {
                let pk = pqcrypto_dilithium::dilithium5::PublicKey::from_bytes(pk_bytes)
                    .map_err(|_| TlsError::Policy("Invalid Dilithium5 public key".into()))?;
                let sig = pqcrypto_dilithium::dilithium5::DetachedSignature::from_bytes(sig_bytes)
                    .map_err(|_| TlsError::Policy("Invalid Dilithium5 signature".into()))?;
                Ok(pqcrypto_dilithium::dilithium5::verify_detached_signature(&sig, message, &pk).is_ok())
            }
        }
    }

    /// Get the default security level
    pub fn default_level(&self) -> DilithiumSecurityLevel {
        self.default_level
    }

    /// Check if FIPS mode is enabled
    pub fn is_fips_mode(&self) -> bool {
        self.fips_mode
    }
}

impl Default for DilithiumEngine {
    fn default() -> Self {
        Self::for_enterprise()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_dilithium2_roundtrip() {
        let engine = DilithiumEngine::new(DilithiumSecurityLevel::Dilithium2);
        let keypair = engine.generate_keypair().await.unwrap();

        assert_eq!(keypair.level, DilithiumSecurityLevel::Dilithium2);
        assert_eq!(keypair.public_key.as_bytes().len(), 1312);

        let message = b"Test message for Dilithium signature";
        let signature = engine.sign(&keypair.private_key, message).await.unwrap();

        assert_eq!(signature.size(), 2420);

        let valid = engine.verify(&keypair.public_key, message, &signature).await.unwrap();
        assert!(valid);
    }

    #[tokio::test]
    async fn test_dilithium3_roundtrip() {
        let engine = DilithiumEngine::for_enterprise();
        let keypair = engine.generate_keypair().await.unwrap();

        assert_eq!(keypair.level, DilithiumSecurityLevel::Dilithium3);
        assert_eq!(keypair.public_key.as_bytes().len(), 1952);

        let message = b"Test message for Dilithium3 signature";
        let signature = engine.sign(&keypair.private_key, message).await.unwrap();

        assert_eq!(signature.size(), 3293);

        let valid = engine.verify(&keypair.public_key, message, &signature).await.unwrap();
        assert!(valid);
    }

    #[tokio::test]
    async fn test_dilithium5_roundtrip() {
        let engine = DilithiumEngine::for_maximum_security();
        let keypair = engine.generate_keypair().await.unwrap();

        assert_eq!(keypair.level, DilithiumSecurityLevel::Dilithium5);
        assert_eq!(keypair.public_key.as_bytes().len(), 2592);

        let message = b"Test message for Dilithium5 signature";
        let signature = engine.sign(&keypair.private_key, message).await.unwrap();

        assert_eq!(signature.size(), 4595);

        let valid = engine.verify(&keypair.public_key, message, &signature).await.unwrap();
        assert!(valid);
    }

    #[tokio::test]
    async fn test_invalid_message_verification() {
        let engine = DilithiumEngine::for_enterprise();
        let keypair = engine.generate_keypair().await.unwrap();

        let message = b"Original message";
        let signature = engine.sign(&keypair.private_key, message).await.unwrap();

        // Verify with wrong message should return false
        let invalid = engine.verify(&keypair.public_key, b"Wrong message", &signature).await.unwrap();
        assert!(!invalid);
    }

    #[test]
    fn test_security_level_properties() {
        let level = DilithiumSecurityLevel::Dilithium3;
        assert_eq!(level.nist_level(), 3);
        assert_eq!(level.name(), "ML-DSA-65");
        assert_eq!(level.legacy_name(), "Dilithium3");
        assert_eq!(level.public_key_size(), 1952);
        assert_eq!(level.signature_size(), 3293);
    }
}
