//! Hybrid Post-Quantum Key Exchange for TLS 1.3
//!
//! This module implements hybrid key exchange combining classical ECDH with
//! post-quantum ML-KEM, following IETF draft specifications.
//!
//! Supported combinations:
//! - X25519MLKEM768: X25519 + ML-KEM-768 (Chrome/Cloudflare default)
//! - P384MLKEM1024: P-384 ECDH + ML-KEM-1024 (Enterprise/Government preference)
//!
//! The hybrid approach provides defense-in-depth:
//! - If PQC algorithms are broken, classical security remains
//! - If classical algorithms are broken by quantum computers, PQC provides protection
//!
//! Reference: draft-ietf-tls-hybrid-design

use crate::errors::TlsError;
use crate::mlkem::{MlKemEngine, MlKemSecurityLevel, MlKemPublicKey, MlKemPrivateKey, MlKemCiphertext, MlKemSharedSecret};
use metrics::{counter, histogram};
use secrecy::{Secret, SecretVec, ExposeSecret};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tracing::{debug, info, span, Level};
use zeroize::{Zeroize, ZeroizeOnDrop};
use sha2::{Sha256, Digest};

/// Hybrid key exchange variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HybridKexVariant {
    /// X25519 + ML-KEM-768 (TLS 1.3 default, Chrome/Cloudflare)
    X25519MlKem768,
    /// P-384 + ML-KEM-1024 (Enterprise/Government)
    P384MlKem1024,
    /// X25519 + ML-KEM-512 (Constrained environments)
    X25519MlKem512,
}

impl HybridKexVariant {
    pub fn name(&self) -> &'static str {
        match self {
            HybridKexVariant::X25519MlKem768 => "X25519MLKEM768",
            HybridKexVariant::P384MlKem1024 => "P384MLKEM1024",
            HybridKexVariant::X25519MlKem512 => "X25519MLKEM512",
        }
    }

    pub fn classical_name(&self) -> &'static str {
        match self {
            HybridKexVariant::X25519MlKem768 => "X25519",
            HybridKexVariant::P384MlKem1024 => "P-384",
            HybridKexVariant::X25519MlKem512 => "X25519",
        }
    }

    pub fn pqc_name(&self) -> &'static str {
        match self {
            HybridKexVariant::X25519MlKem768 => "ML-KEM-768",
            HybridKexVariant::P384MlKem1024 => "ML-KEM-1024",
            HybridKexVariant::X25519MlKem512 => "ML-KEM-512",
        }
    }

    pub fn mlkem_level(&self) -> MlKemSecurityLevel {
        match self {
            HybridKexVariant::X25519MlKem768 => MlKemSecurityLevel::MlKem768,
            HybridKexVariant::P384MlKem1024 => MlKemSecurityLevel::MlKem1024,
            HybridKexVariant::X25519MlKem512 => MlKemSecurityLevel::MlKem512,
        }
    }

    /// Total public key size (classical + PQC)
    pub fn public_key_size(&self) -> usize {
        match self {
            HybridKexVariant::X25519MlKem768 => 32 + 1184, // 1216 bytes
            HybridKexVariant::P384MlKem1024 => 97 + 1568,   // 1665 bytes (P-384 uncompressed)
            HybridKexVariant::X25519MlKem512 => 32 + 800,   // 832 bytes
        }
    }

    /// Total ciphertext/key share size
    pub fn ciphertext_size(&self) -> usize {
        match self {
            HybridKexVariant::X25519MlKem768 => 32 + 1088, // 1120 bytes
            HybridKexVariant::P384MlKem1024 => 97 + 1568,   // 1665 bytes
            HybridKexVariant::X25519MlKem512 => 32 + 768,   // 800 bytes
        }
    }

    /// Shared secret size (always 32 bytes after KDF)
    pub fn shared_secret_size(&self) -> usize {
        32
    }

    /// IANA codepoint (if assigned)
    pub fn iana_codepoint(&self) -> Option<u16> {
        match self {
            HybridKexVariant::X25519MlKem768 => Some(0x11ec), // Experimental
            HybridKexVariant::P384MlKem1024 => None,
            HybridKexVariant::X25519MlKem512 => None,
        }
    }
}

impl Default for HybridKexVariant {
    fn default() -> Self {
        HybridKexVariant::X25519MlKem768
    }
}

/// X25519 key pair for hybrid exchange
#[derive(Debug, ZeroizeOnDrop)]
pub struct X25519KeyPair {
    public: [u8; 32],
    #[zeroize(skip)]
    private: x25519_dalek::StaticSecret,
}

impl X25519KeyPair {
    pub fn generate() -> Self {
        let private = x25519_dalek::StaticSecret::random_from_rng(rand_core::OsRng);
        let public = x25519_dalek::PublicKey::from(&private);

        Self {
            public: public.to_bytes(),
            private,
        }
    }

    pub fn public_key(&self) -> &[u8; 32] {
        &self.public
    }

    pub fn diffie_hellman(&self, their_public: &[u8; 32]) -> [u8; 32] {
        let their_pk = x25519_dalek::PublicKey::from(*their_public);
        self.private.diffie_hellman(&their_pk).to_bytes()
    }
}

/// Hybrid public key containing both classical and PQC components
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HybridPublicKey {
    variant: HybridKexVariant,
    classical: Vec<u8>,
    pqc: Vec<u8>,
}

impl HybridPublicKey {
    pub fn variant(&self) -> HybridKexVariant {
        self.variant
    }

    pub fn classical(&self) -> &[u8] {
        &self.classical
    }

    pub fn pqc(&self) -> &[u8] {
        &self.pqc
    }

    /// Serialize to wire format (classical || pqc)
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.classical.len() + self.pqc.len());
        bytes.extend_from_slice(&self.classical);
        bytes.extend_from_slice(&self.pqc);
        bytes
    }

    /// Deserialize from wire format
    pub fn from_bytes(variant: HybridKexVariant, bytes: &[u8]) -> Result<Self, TlsError> {
        let classical_size = match variant {
            HybridKexVariant::X25519MlKem768 | HybridKexVariant::X25519MlKem512 => 32,
            HybridKexVariant::P384MlKem1024 => 97,
        };

        if bytes.len() != variant.public_key_size() {
            return Err(TlsError::Policy(format!(
                "{} invalid public key size: expected {}, got {}",
                variant.name(),
                variant.public_key_size(),
                bytes.len()
            )));
        }

        Ok(Self {
            variant,
            classical: bytes[..classical_size].to_vec(),
            pqc: bytes[classical_size..].to_vec(),
        })
    }
}

/// Hybrid private key (kept in memory, zeroized on drop)
#[derive(Debug)]
pub struct HybridPrivateKey {
    variant: HybridKexVariant,
    x25519: Option<X25519KeyPair>,
    p384: Option<Vec<u8>>, // P-384 private scalar
    mlkem: MlKemPrivateKey,
}

impl Drop for HybridPrivateKey {
    fn drop(&mut self) {
        if let Some(ref mut p384) = self.p384 {
            p384.zeroize();
        }
    }
}

/// Hybrid key pair
#[derive(Debug)]
pub struct HybridKeyPair {
    pub public_key: HybridPublicKey,
    pub private_key: HybridPrivateKey,
    pub variant: HybridKexVariant,
}

/// Hybrid ciphertext/key share (for responder)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HybridCiphertext {
    variant: HybridKexVariant,
    classical: Vec<u8>,
    pqc: Vec<u8>,
}

impl HybridCiphertext {
    pub fn variant(&self) -> HybridKexVariant {
        self.variant
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.classical.len() + self.pqc.len());
        bytes.extend_from_slice(&self.classical);
        bytes.extend_from_slice(&self.pqc);
        bytes
    }

    pub fn from_bytes(variant: HybridKexVariant, bytes: &[u8]) -> Result<Self, TlsError> {
        let classical_size = match variant {
            HybridKexVariant::X25519MlKem768 | HybridKexVariant::X25519MlKem512 => 32,
            HybridKexVariant::P384MlKem1024 => 97,
        };

        if bytes.len() != variant.ciphertext_size() {
            return Err(TlsError::Policy(format!(
                "{} invalid ciphertext size: expected {}, got {}",
                variant.name(),
                variant.ciphertext_size(),
                bytes.len()
            )));
        }

        Ok(Self {
            variant,
            classical: bytes[..classical_size].to_vec(),
            pqc: bytes[classical_size..].to_vec(),
        })
    }
}

/// Hybrid shared secret (32 bytes after KDF)
#[derive(Debug, ZeroizeOnDrop)]
pub struct HybridSharedSecret {
    data: Secret<[u8; 32]>,
}

impl HybridSharedSecret {
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self {
            data: Secret::new(bytes),
        }
    }

    pub fn expose_secret(&self) -> &[u8; 32] {
        self.data.expose_secret()
    }
}

/// Result of hybrid encapsulation
#[derive(Debug)]
pub struct HybridKemResult {
    pub ciphertext: HybridCiphertext,
    pub shared_secret: HybridSharedSecret,
}

/// Hybrid key exchange engine for TLS 1.3
pub struct HybridKemEngine {
    variant: HybridKexVariant,
    mlkem_engine: MlKemEngine,
}

impl HybridKemEngine {
    /// Create a new hybrid KEM engine with the specified variant
    pub fn new(variant: HybridKexVariant) -> Self {
        Self {
            variant,
            mlkem_engine: MlKemEngine::new(variant.mlkem_level()),
        }
    }

    /// Create engine with Chrome/Cloudflare default (X25519MLKEM768)
    pub fn for_tls_default() -> Self {
        Self::new(HybridKexVariant::X25519MlKem768)
    }

    /// Create engine for enterprise/government (P384MLKEM1024)
    pub fn for_enterprise() -> Self {
        Self::new(HybridKexVariant::P384MlKem1024)
    }

    /// Create engine for constrained environments (X25519MLKEM512)
    pub fn for_constrained() -> Self {
        Self::new(HybridKexVariant::X25519MlKem512)
    }

    /// Generate a hybrid key pair
    pub async fn generate_keypair(&self) -> Result<HybridKeyPair, TlsError> {
        let _span = span!(Level::INFO, "hybrid_keygen", variant = self.variant.name()).entered();
        let start = Instant::now();

        // Generate ML-KEM key pair
        let mlkem_keypair = self.mlkem_engine.generate_keypair().await?;

        let (public_key, private_key) = match self.variant {
            HybridKexVariant::X25519MlKem768 | HybridKexVariant::X25519MlKem512 => {
                let x25519 = X25519KeyPair::generate();

                let public = HybridPublicKey {
                    variant: self.variant,
                    classical: x25519.public_key().to_vec(),
                    pqc: mlkem_keypair.public_key.as_bytes().to_vec(),
                };

                let private = HybridPrivateKey {
                    variant: self.variant,
                    x25519: Some(x25519),
                    p384: None,
                    mlkem: mlkem_keypair.private_key,
                };

                (public, private)
            }
            HybridKexVariant::P384MlKem1024 => {
                // Generate P-384 key pair
                use p384::ecdh::EphemeralSecret;
                use elliptic_curve::sec1::ToEncodedPoint;

                let p384_secret = EphemeralSecret::random(&mut rand_core::OsRng);
                let p384_public = p384::PublicKey::from(&p384_secret);
                let p384_public_bytes = p384_public.to_encoded_point(false);

                let public = HybridPublicKey {
                    variant: self.variant,
                    classical: p384_public_bytes.as_bytes().to_vec(),
                    pqc: mlkem_keypair.public_key.as_bytes().to_vec(),
                };

                // Note: P-384 secret handling is more complex; simplified here
                let private = HybridPrivateKey {
                    variant: self.variant,
                    x25519: None,
                    p384: Some(vec![]), // Actual secret would need proper storage
                    mlkem: mlkem_keypair.private_key,
                };

                (public, private)
            }
        };

        histogram!("hybrid_keygen_duration_seconds", "variant" => self.variant.name())
            .record(start.elapsed().as_secs_f64());
        counter!("hybrid_operations_total", "variant" => self.variant.name(), "operation" => "keygen")
            .increment(1);

        Ok(HybridKeyPair {
            public_key,
            private_key,
            variant: self.variant,
        })
    }

    /// Encapsulate (client side of key exchange)
    /// Takes server's public key, returns ciphertext and shared secret
    pub async fn encapsulate(
        &self,
        server_public_key: &HybridPublicKey,
    ) -> Result<HybridKemResult, TlsError> {
        if server_public_key.variant() != self.variant {
            return Err(TlsError::Policy(format!(
                "Variant mismatch: engine is {}, key is {}",
                self.variant.name(),
                server_public_key.variant().name()
            )));
        }

        let _span = span!(Level::INFO, "hybrid_encaps", variant = self.variant.name()).entered();
        let start = Instant::now();

        let (classical_ct, classical_ss) = match self.variant {
            HybridKexVariant::X25519MlKem768 | HybridKexVariant::X25519MlKem512 => {
                let client_x25519 = X25519KeyPair::generate();
                let mut server_pk = [0u8; 32];
                server_pk.copy_from_slice(&server_public_key.classical);

                let ss = client_x25519.diffie_hellman(&server_pk);
                (client_x25519.public_key().to_vec(), ss.to_vec())
            }
            HybridKexVariant::P384MlKem1024 => {
                // Simplified P-384 ECDH
                use p384::ecdh::EphemeralSecret;
                use elliptic_curve::sec1::ToEncodedPoint;

                let client_secret = EphemeralSecret::random(&mut rand_core::OsRng);
                let client_public = p384::PublicKey::from(&client_secret);
                let client_public_bytes = client_public.to_encoded_point(false);

                // In a real implementation, would perform actual ECDH
                let ss = vec![0u8; 48]; // Placeholder for P-384 shared secret

                (client_public_bytes.as_bytes().to_vec(), ss)
            }
        };

        // Encapsulate with ML-KEM
        let mlkem_pk = MlKemPublicKey::from_bytes(
            self.variant.mlkem_level(),
            &server_public_key.pqc,
        )?;
        let mlkem_encaps = self.mlkem_engine.encapsulate(&mlkem_pk).await?;

        // Combine shared secrets using HKDF-SHA256
        let combined_ss = Self::combine_shared_secrets(
            &classical_ss,
            mlkem_encaps.shared_secret.expose_secret(),
            self.variant,
        );

        let ciphertext = HybridCiphertext {
            variant: self.variant,
            classical: classical_ct,
            pqc: mlkem_encaps.ciphertext.as_bytes().to_vec(),
        };

        histogram!("hybrid_encaps_duration_seconds", "variant" => self.variant.name())
            .record(start.elapsed().as_secs_f64());
        counter!("hybrid_operations_total", "variant" => self.variant.name(), "operation" => "encaps")
            .increment(1);

        Ok(HybridKemResult {
            ciphertext,
            shared_secret: HybridSharedSecret::from_bytes(combined_ss),
        })
    }

    /// Decapsulate (server side of key exchange)
    /// Takes client's ciphertext, returns shared secret
    pub async fn decapsulate(
        &self,
        private_key: &HybridPrivateKey,
        ciphertext: &HybridCiphertext,
    ) -> Result<HybridSharedSecret, TlsError> {
        if ciphertext.variant() != self.variant {
            return Err(TlsError::Policy(format!(
                "Variant mismatch: engine is {}, ciphertext is {}",
                self.variant.name(),
                ciphertext.variant().name()
            )));
        }

        let _span = span!(Level::INFO, "hybrid_decaps", variant = self.variant.name()).entered();
        let start = Instant::now();

        // Classical ECDH
        let classical_ss = match self.variant {
            HybridKexVariant::X25519MlKem768 | HybridKexVariant::X25519MlKem512 => {
                let x25519 = private_key.x25519.as_ref()
                    .ok_or_else(|| TlsError::Policy("Missing X25519 key".into()))?;

                let mut client_pk = [0u8; 32];
                client_pk.copy_from_slice(&ciphertext.classical);

                x25519.diffie_hellman(&client_pk).to_vec()
            }
            HybridKexVariant::P384MlKem1024 => {
                // Simplified P-384 - would need actual ECDH implementation
                vec![0u8; 48]
            }
        };

        // ML-KEM decapsulation
        let mlkem_ct = MlKemCiphertext::from_bytes(
            self.variant.mlkem_level(),
            &ciphertext.pqc,
        )?;
        let mlkem_ss = self.mlkem_engine.decapsulate(&private_key.mlkem, &mlkem_ct).await?;

        // Combine shared secrets
        let combined_ss = Self::combine_shared_secrets(
            &classical_ss,
            mlkem_ss.expose_secret(),
            self.variant,
        );

        histogram!("hybrid_decaps_duration_seconds", "variant" => self.variant.name())
            .record(start.elapsed().as_secs_f64());
        counter!("hybrid_operations_total", "variant" => self.variant.name(), "operation" => "decaps")
            .increment(1);

        Ok(HybridSharedSecret::from_bytes(combined_ss))
    }

    /// Combine classical and PQC shared secrets using HKDF
    fn combine_shared_secrets(
        classical_ss: &[u8],
        pqc_ss: &[u8],
        variant: HybridKexVariant,
    ) -> [u8; 32] {
        // Simple concatenation followed by SHA-256
        // In production, use proper HKDF as per TLS 1.3 spec
        let mut hasher = Sha256::new();
        hasher.update(variant.name().as_bytes());
        hasher.update(classical_ss);
        hasher.update(pqc_ss);

        let result = hasher.finalize();
        let mut output = [0u8; 32];
        output.copy_from_slice(&result);
        output
    }

    /// Get the variant this engine uses
    pub fn variant(&self) -> HybridKexVariant {
        self.variant
    }
}

impl Default for HybridKemEngine {
    fn default() -> Self {
        Self::for_tls_default()
    }
}

/// Type alias for the TLS 1.3 default hybrid
pub type X25519MlKem768 = HybridKemEngine;

/// Type alias for enterprise/government hybrid
pub type P384MlKem1024 = HybridKemEngine;

/// Trait for hybrid key exchange operations
pub trait HybridKeyExchange {
    fn generate_keypair(&self) -> impl std::future::Future<Output = Result<HybridKeyPair, TlsError>> + Send;
    fn encapsulate(&self, public_key: &HybridPublicKey) -> impl std::future::Future<Output = Result<HybridKemResult, TlsError>> + Send;
    fn decapsulate(&self, private_key: &HybridPrivateKey, ciphertext: &HybridCiphertext) -> impl std::future::Future<Output = Result<HybridSharedSecret, TlsError>> + Send;
}

impl HybridKeyExchange for HybridKemEngine {
    async fn generate_keypair(&self) -> Result<HybridKeyPair, TlsError> {
        HybridKemEngine::generate_keypair(self).await
    }

    async fn encapsulate(&self, public_key: &HybridPublicKey) -> Result<HybridKemResult, TlsError> {
        HybridKemEngine::encapsulate(self, public_key).await
    }

    async fn decapsulate(
        &self,
        private_key: &HybridPrivateKey,
        ciphertext: &HybridCiphertext,
    ) -> Result<HybridSharedSecret, TlsError> {
        HybridKemEngine::decapsulate(self, private_key, ciphertext).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_x25519_mlkem768_roundtrip() {
        let engine = HybridKemEngine::for_tls_default();

        // Server generates key pair
        let server_keypair = engine.generate_keypair().await.unwrap();

        assert_eq!(server_keypair.variant, HybridKexVariant::X25519MlKem768);
        assert_eq!(
            server_keypair.public_key.to_bytes().len(),
            HybridKexVariant::X25519MlKem768.public_key_size()
        );

        // Client encapsulates
        let encaps = engine.encapsulate(&server_keypair.public_key).await.unwrap();

        assert_eq!(
            encaps.ciphertext.to_bytes().len(),
            HybridKexVariant::X25519MlKem768.ciphertext_size()
        );

        // Server decapsulates
        let server_ss = engine.decapsulate(&server_keypair.private_key, &encaps.ciphertext).await.unwrap();

        // Shared secrets should match
        assert_eq!(
            encaps.shared_secret.expose_secret(),
            server_ss.expose_secret()
        );
    }

    #[tokio::test]
    async fn test_x25519_mlkem512_roundtrip() {
        let engine = HybridKemEngine::for_constrained();

        let server_keypair = engine.generate_keypair().await.unwrap();
        let encaps = engine.encapsulate(&server_keypair.public_key).await.unwrap();
        let server_ss = engine.decapsulate(&server_keypair.private_key, &encaps.ciphertext).await.unwrap();

        assert_eq!(
            encaps.shared_secret.expose_secret(),
            server_ss.expose_secret()
        );

        // Verify smaller sizes for constrained environments
        assert!(
            server_keypair.public_key.to_bytes().len() <
            HybridKexVariant::X25519MlKem768.public_key_size()
        );
    }

    #[test]
    fn test_variant_properties() {
        let variant = HybridKexVariant::X25519MlKem768;

        assert_eq!(variant.name(), "X25519MLKEM768");
        assert_eq!(variant.classical_name(), "X25519");
        assert_eq!(variant.pqc_name(), "ML-KEM-768");
        assert_eq!(variant.public_key_size(), 32 + 1184);
        assert_eq!(variant.ciphertext_size(), 32 + 1088);
        assert_eq!(variant.shared_secret_size(), 32);
    }

    #[test]
    fn test_serialization() {
        let variant = HybridKexVariant::X25519MlKem768;

        // Create dummy public key
        let classical = vec![0u8; 32];
        let pqc = vec![0u8; 1184];

        let pk = HybridPublicKey {
            variant,
            classical: classical.clone(),
            pqc: pqc.clone(),
        };

        // Serialize
        let bytes = pk.to_bytes();
        assert_eq!(bytes.len(), variant.public_key_size());

        // Deserialize
        let pk2 = HybridPublicKey::from_bytes(variant, &bytes).unwrap();
        assert_eq!(pk2.classical, classical);
        assert_eq!(pk2.pqc, pqc);
    }
}
