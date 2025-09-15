use crate::errors::TlsError;
use crate::provider::{is_oqs_loaded, load_oqs_provider, verify_kyber768_usage, verify_dilithium_signature};
use crate::validation::{CertificateValidator, ValidationPolicy, ValidationResult};
use openssl::ssl::{SslConnector, SslMethod, SslOptions, SslVerifyMode};
use tokio::net::TcpStream;
use tokio_openssl::SslStream as TokioSslStream;
use tracing::{info, warn, error};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct TlsClientConfig {
    pub server_name: String,
    pub require_hybrid: bool,
    pub alpn_protocols: Option<Vec<Vec<u8>>>,
    pub insecure_skip_verify: bool,
    /// Fail-closed policy: reject connections that don't meet PQC requirements
    pub fail_closed: bool,
    /// Certificate validator for enhanced validation
    pub validator: Option<Arc<CertificateValidator>>,
    /// Minimum TLS version (default: TLS 1.3 for PQC)
    pub min_tls_version: Option<openssl::ssl::SslVersion>,
}

impl Default for TlsClientConfig {
    fn default() -> Self {
        Self {
            server_name: "localhost".to_string(),
            require_hybrid: false,
            alpn_protocols: None,
            insecure_skip_verify: false,
            fail_closed: false,
            validator: None,
            min_tls_version: Some(openssl::ssl::SslVersion::TLS1_3),
        }
    }
}

pub struct TlsClient;

impl TlsClient {
    pub fn new() -> Self { TlsClient }

    pub async fn connect(&self, addr: &str, cfg: &TlsClientConfig) -> Result<TokioSslStream<TcpStream>, TlsError> {
        // Load OQS provider if available
        let oqs = load_oqs_provider();
        
        // Fail-closed validation: if PQC is required but provider not available, fail
        if cfg.require_hybrid && !oqs {
            return Err(TlsError::Provider("hybrid required but oqsprovider not available".into()));
        }
        
        // Additional fail-closed check for configuration
        if cfg.fail_closed && cfg.require_hybrid && !oqs {
            return Err(TlsError::Policy("fail-closed policy requires OQS provider for hybrid mode".into()));
        }

        let mut builder = SslConnector::builder(SslMethod::tls())?;
        
        // Set minimum TLS version
        if let Some(min_version) = cfg.min_tls_version {
            builder.set_min_proto_version(Some(min_version))?;
        }
        
        if cfg.require_hybrid {
            // Force TLS 1.3 for hybrid mode
            builder.set_min_proto_version(Some(openssl::ssl::SslVersion::TLS1_3))?;
            builder.set_max_proto_version(Some(openssl::ssl::SslVersion::TLS1_3))?;
            builder.set_options(SslOptions::NO_TLSV1 | SslOptions::NO_TLSV1_1 | SslOptions::NO_TLSV1_2);
            
            // Restrict to PQC-safe cipher suites
            builder.set_ciphersuites("TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256")?;
        }
        
        // Configure key exchange groups
        let groups = if cfg.require_hybrid {
            // Only allow hybrid key exchange
            "x25519_kyber768"
        } else {
            // Allow both classical and hybrid
            "x25519:x25519_kyber768:secp256r1:secp384r1"
        };
        
        if let Err(e) = builder.set_groups_list(groups) {
            if cfg.require_hybrid || cfg.fail_closed {
                return Err(TlsError::Policy(format!("failed to set required groups: {e}")));
            }
            warn!(error=%e, "failed to set groups list; proceeding (dev/classical)");
        }

        // Configure ALPN
        if let Some(alpns) = &cfg.alpn_protocols {
            let mut buf = Vec::new();
            for p in alpns {
                buf.push(p.len() as u8);
                buf.extend_from_slice(p);
            }
            builder.set_alpn_protos(&buf)?;
        }

        // Configure certificate verification
        if cfg.insecure_skip_verify {
            if cfg.fail_closed {
                return Err(TlsError::Policy("fail-closed policy prohibits insecure verification".into()));
            }
            builder.set_verify(SslVerifyMode::NONE);
        } else {
            builder.set_verify(SslVerifyMode::PEER);
            
            // Use custom validator if provided
            if let Some(validator) = &cfg.validator {
                // TODO: Set custom verification callback using validator
                // This would require storing the validator in a way accessible to the callback
            }
        }

        let connector = builder.build();
        info!(
            oqs_loaded = oqs,
            groups = ?groups,
            fail_closed = cfg.fail_closed,
            require_hybrid = cfg.require_hybrid,
            "client TLS configured"
        );

        // Connect to server
        let tcp = TcpStream::connect(addr).await?;
        let ssl = connector.configure()?.into_ssl(&cfg.server_name)?;
        let mut stream = TokioSslStream::new(ssl, tcp)?;
        
        // Perform TLS handshake with timeout
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            Pin::new(&mut stream).connect().await
        })
        .await
        .map_err(|_| TlsError::Io("handshake timeout".into()))??;

        // Post-handshake validation for fail-closed policy
        if cfg.fail_closed || cfg.require_hybrid {
            self.validate_connection(&stream, cfg).await?;
        }

        Ok(stream)
    }
    
    /// Validate connection meets security requirements (fail-closed policy)
    async fn validate_connection(&self, stream: &TokioSslStream<TcpStream>, cfg: &TlsClientConfig) -> Result<(), TlsError> {
        let ssl = stream.ssl();
        
        // Check cipher suite
        if let Some(cipher) = ssl.current_cipher() {
            let cipher_name = cipher.name();
            info!(cipher = cipher_name, "negotiated cipher suite");
            
            if cfg.require_hybrid {
                // Ensure we're using a PQC-safe cipher suite
                if !cipher_name.contains("AES_256_GCM_SHA384") &&
                   !cipher_name.contains("CHACHA20_POLY1305_SHA256") {
                    return Err(TlsError::Policy(format!(
                        "cipher suite {} not allowed in hybrid mode", cipher_name
                    )));
                }
            }
        }
        
        // Check key exchange group
        if cfg.require_hybrid {
            let uses_kyber = verify_kyber768_usage(ssl)?;
            if !uses_kyber {
                return Err(TlsError::Policy(
                    "hybrid mode requires Kyber-768 key exchange".into()
                ));
            }
        }
        
        // Check protocol version
        let version = ssl.version_str();
        info!(version = version, "negotiated TLS version");
        
        if cfg.fail_closed && version != "TLSv1.3" {
            return Err(TlsError::Policy(format!(
                "TLS version {} not allowed in fail-closed mode", version
            )));
        }
        
        // Additional validation with certificate validator if provided
        if let Some(validator) = &cfg.validator {
            if let Some(peer_cert_chain) = ssl.peer_cert_chain() {
                let validation_result = validator.validate_chain(peer_cert_chain, Some(ssl))?;
                
                if !validation_result.is_valid() {
                    return Err(TlsError::Policy(
                        "certificate validation failed in fail-closed mode".into()
                    ));
                }
                
                if cfg.require_hybrid && !validation_result.meets_pqc_requirements() {
                    return Err(TlsError::Policy(
                        "connection does not meet PQC requirements".into()
                    ));
                }
            }
        }
        
        info!("connection validation passed");
        Ok(())
    }
}

use core::pin::Pin;
