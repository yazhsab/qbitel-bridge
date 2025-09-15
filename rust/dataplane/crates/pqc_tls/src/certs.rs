use crate::errors::TlsError;
use openssl::pkey::PKey;
use openssl::x509::X509;
use tracing::{info, warn};

pub struct CertKeyPair {
    pub cert: X509,
    pub key: PKey<openssl::pkey::Private>,
}

#[derive(Clone, Debug)]
pub enum CertPolicy {
    DevAllowClassical,
    RequireHybrid,
    PQCOnly,
}

impl CertKeyPair {
    pub fn from_pem(cert_pem: &[u8], key_pem: &[u8], policy: &CertPolicy) -> Result<Self, TlsError> {
        let cert = X509::from_pem(cert_pem)?;
        let key = PKey::private_key_from_pem(key_pem)?;
        enforce_cert_policy(&cert, policy)?;
        info!("loaded certificate and key (PEM)");
        Ok(Self { cert, key })
    }
}

fn enforce_cert_policy(cert: &X509, policy: &CertPolicy) -> Result<(), TlsError> {
    match policy {
        CertPolicy::DevAllowClassical => {
            // No strong enforcement; used in dev and tests.
            let sig = cert
                .signature_algorithm()
                .object()
                .nid()
                .short_name()
                .unwrap_or("unknown");
            warn!(algorithm=%sig, "dev mode: allowing non-Dilithium signature");
            Ok(())
        }
        CertPolicy::RequireHybrid | CertPolicy::PQCOnly => {
            // Without oqs provider types, we cannot introspect Dilithium algorithm id using openssl crate enums.
            // Enforce at TLS policy level instead; here we only reject obviously classical RSA/EC when identified.
            let alg = cert.signature_algorithm().object().nid().short_name().unwrap_or("unknown");
            // Known classical alg snippets
            if alg.contains("RSA") || alg.contains("ECDSA") || alg.contains("ED25519") || alg.contains("ED448") {
                return Err(TlsError::Policy(format!("non-PQC certificate signature detected ({alg})")));
            }
            Ok(())
        }
    }
}
