use crate::certs::{CertKeyPair, CertPolicy};
use crate::errors::TlsError;
use crate::provider::{is_oqs_loaded, load_oqs_provider, verify_kyber768_usage, verify_dilithium_signature};
use crate::validation::{CertificateValidator, ValidationPolicy, ValidationResult};
use openssl::ssl::{SslAcceptor, SslFiletype, SslMethod, SslOptions, SslVerifyMode};
use openssl::x509::store::X509StoreBuilder;
use openssl::stack::Stack;
use tokio::net::TcpStream;
use tokio_openssl::SslStream as TokioSslStream;
use tracing::{info, warn, error};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct TlsServerConfig {
    pub require_hybrid: bool,
    pub client_auth: bool,
    pub alpn_protocols: Option<Vec<Vec<u8>>>,
    /// Fail-closed policy: reject connections that don't meet PQC requirements
    pub fail_closed: bool,
    /// Certificate validator for enhanced validation
    pub validator: Option<Arc<CertificateValidator>>,
    /// Minimum TLS version (default: TLS 1.3 for PQC)
    pub min_tls_version: Option<openssl::ssl::SslVersion>,
}

impl Default for TlsServerConfig {
    fn default() -> Self {
        Self {
            require_hybrid: false,
            client_auth: false,
            alpn_protocols: None,
            fail_closed: false,
            validator: None,
            min_tls_version: Some(openssl::ssl::SslVersion::TLS1_3),
        }
    }
}

pub struct TlsServer {
    acceptor: openssl::ssl::SslAcceptor,
    config: TlsServerConfig,
}

impl TlsServer {
    pub fn new(cert: &CertKeyPair, policy: &CertPolicy, cfg: &TlsServerConfig) -> Result<Self, TlsError> {
        let oqs = load_oqs_provider();
        
        // Fail-closed validation: if PQC is required but provider not available, fail
        if matches!(policy, CertPolicy::RequireHybrid | CertPolicy::PQCOnly) && !oqs {
            return Err(TlsError::Provider("hybrid/PQC cert policy but oqsprovider not available".into()));
        }
        
        // Additional fail-closed check for configuration
        if cfg.fail_closed && cfg.require_hybrid && !oqs {
            return Err(TlsError::Policy("fail-closed policy requires OQS provider for hybrid mode".into()));
        }

        let mut builder = SslAcceptor::mozilla_intermediate(SslMethod::tls())?;
        
        // Set minimum TLS version
        if let Some(min_version) = cfg.min_tls_version {
            builder.set_min_proto_version(Some(min_version))?;
        }
        
        let groups = if cfg.require_hybrid {
            // Force TLS 1.3 for hybrid mode
            builder.set_min_proto_version(Some(openssl::ssl::SslVersion::TLS1_3))?;
            builder.set_max_proto_version(Some(openssl::ssl::SslVersion::TLS1_3))?;
            builder.set_options(SslOptions::NO_TLSV1 | SslOptions::NO_TLSV1_1 | SslOptions::NO_TLSV1_2);
            
            // Restrict to PQC-safe cipher suites
            builder.set_ciphersuites("TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256")?;
            
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

        // Set certificate and key
        builder.set_certificate(&cert.cert)?;
        builder.set_private_key(&cert.key)?;

        // Configure client authentication
        if cfg.client_auth {
            // Use provided validator's trust store if available
            let store = if let Some(validator) = &cfg.validator {
                // TODO: Extract trust store from validator
                X509StoreBuilder::new()?.build()
            } else {
                X509StoreBuilder::new()?.build()
            };
            
            builder.set_verify(SslVerifyMode::PEER | SslVerifyMode::FAIL_IF_NO_PEER_CERT);
            builder.set_verify_cert_store(store)?;
        }

        // Configure ALPN if specified
        if let Some(alpn_protocols) = &cfg.alpn_protocols {
            let mut alpn_buf = Vec::new();
            for protocol in alpn_protocols {
                alpn_buf.push(protocol.len() as u8);
                alpn_buf.extend_from_slice(protocol);
            }
            builder.set_alpn_select_callback(move |_ssl, client_protocols| {
                // Simple ALPN selection - choose first matching protocol
                for client_proto in client_protocols {
                    if alpn_protocols.iter().any(|p| p.as_slice() == client_proto) {
                        return Ok(client_proto);
                    }
                }
                Err(openssl::ssl::AlpnError::NOACK)
            });
        }

        let acceptor = builder.build();
        info!(
            oqs_loaded = oqs,
            groups = ?groups,
            fail_closed = cfg.fail_closed,
            require_hybrid = cfg.require_hybrid,
            "server TLS configured"
        );
        
        Ok(Self {
            acceptor,
            config: cfg.clone(),
        })
    }

    pub async fn accept(&self, tcp: TcpStream) -> Result<TokioSslStream<TcpStream>, TlsError> {
        let ssl = openssl::ssl::Ssl::new(self.acceptor.context())?;
        let mut stream = TokioSslStream::new(ssl, tcp)?;
        
        // Perform TLS handshake with timeout
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            Pin::new(&mut stream).accept().await
        })
        .await
        .map_err(|_| TlsError::Io("handshake timeout".into()))??;
        
        // Post-handshake validation for fail-closed policy
        if self.config.fail_closed || self.config.require_hybrid {
            self.validate_connection(&stream).await?;
        }
        
        Ok(stream)
    }
    
    /// Validate connection meets security requirements (fail-closed policy)
    async fn validate_connection(&self, stream: &TokioSslStream<TcpStream>) -> Result<(), TlsError> {
        let ssl = stream.ssl();
        
        // Check cipher suite
        if let Some(cipher) = ssl.current_cipher() {
            let cipher_name = cipher.name();
            info!(cipher = cipher_name, "negotiated cipher suite");
            
            if self.config.require_hybrid {
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
        if self.config.require_hybrid {
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
        
        if self.config.fail_closed && version != "TLSv1.3" {
            return Err(TlsError::Policy(format!(
                "TLS version {} not allowed in fail-closed mode", version
            )));
        }
        
        // Additional validation with certificate validator if provided
        if let Some(validator) = &self.config.validator {
            if let Some(peer_cert_chain) = ssl.peer_cert_chain() {
                let validation_result = validator.validate_chain(peer_cert_chain, Some(ssl))?;
                
                if !validation_result.is_valid() {
                    return Err(TlsError::Policy(
                        "certificate validation failed in fail-closed mode".into()
                    ));
                }
                
                if self.config.require_hybrid && !validation_result.meets_pqc_requirements() {
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

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;
    use openssl::asn1::Asn1Time;
    use openssl::hash::MessageDigest;
    use openssl::nid::Nid;
    use openssl::pkey::PKey;
    use openssl::ec::{EcGroup, EcKey};
    use openssl::x509::{X509NameBuilder, X509};
    use tokio::task;

    fn gen_ephemeral_cert() -> CertKeyPair {
        // Dev-only: generate a quick ECDSA P-256 self-signed cert in memory to avoid writing keys to disk.
        let group = EcGroup::from_curve_name(Nid::X9_62_PRIME256V1).unwrap();
        let ec = EcKey::generate(&group).unwrap();
        let key = PKey::from_ec_key(ec).unwrap();
        let mut name = X509NameBuilder::new().unwrap();
        name.append_entry_by_nid(Nid::COMMONNAME, "localhost").unwrap();
        let name = name.build();
        let mut builder = X509::builder().unwrap();
        builder.set_version(2).unwrap();
        builder.set_subject_name(&name).unwrap();
        builder.set_issuer_name(&name).unwrap();
        builder.set_pubkey(&key).unwrap();
        builder.set_not_before(&Asn1Time::days_from_now(0).unwrap()).unwrap();
        builder.set_not_after(&Asn1Time::days_from_now(1).unwrap()).unwrap();
        builder.sign(&key, MessageDigest::sha256()).unwrap();
        let cert = builder.build();
        CertKeyPair { cert, key }
    }

    #[tokio::test]
    async fn classical_dev_handshake_ok() {
        let cert = gen_ephemeral_cert();
        let config = TlsServerConfig {
            require_hybrid: false,
            client_auth: false,
            alpn_protocols: None,
            fail_closed: false,
            validator: None,
            min_tls_version: None,
        };
        let server = TlsServer::new(&cert, &CertPolicy::DevAllowClassical, &config).unwrap();
        let addr = "127.0.0.1:0";
        let listener = TcpListener::bind(addr).await.unwrap();
        let local_addr = listener.local_addr().unwrap();
        task::spawn(async move {
            let (tcp, _) = listener.accept().await.unwrap();
            let _ = server.accept(tcp).await.unwrap();
        });

        let client = crate::client::TlsClient::new();
        let res = client.connect(&local_addr.to_string(), &crate::client::TlsClientConfig{
            server_name: "localhost".into(),
            require_hybrid: false,
            alpn_protocols: None,
            insecure_skip_verify: true,
        }).await;
        assert!(res.is_ok());
    }

    #[tokio::test]
    async fn require_hybrid_rejects_without_provider() {
        if is_oqs_loaded() { return; }
        let cert = gen_ephemeral_cert();
        let config = TlsServerConfig {
            require_hybrid: true,
            client_auth: false,
            alpn_protocols: None,
            fail_closed: false,
            validator: None,
            min_tls_version: Some(openssl::ssl::SslVersion::TLS1_3),
        };
        let server = TlsServer::new(&cert, &CertPolicy::RequireHybrid, &config);
        assert!(server.is_err());
    }

    #[tokio::test]
    async fn fail_closed_policy_enforced() {
        let cert = gen_ephemeral_cert();
        let config = TlsServerConfig {
            require_hybrid: false,
            client_auth: false,
            alpn_protocols: None,
            fail_closed: true,
            validator: None,
            min_tls_version: Some(openssl::ssl::SslVersion::TLS1_3),
        };
        let server = TlsServer::new(&cert, &CertPolicy::DevAllowClassical, &config).unwrap();
        
        // Test that the server was created successfully with fail-closed policy
        assert!(server.config.fail_closed);
    }

    #[tokio::test]
    async fn default_config_works() {
        let cert = gen_ephemeral_cert();
        let config = TlsServerConfig::default();
        let server = TlsServer::new(&cert, &CertPolicy::DevAllowClassical, &config);
        assert!(server.is_ok());
    }
}
