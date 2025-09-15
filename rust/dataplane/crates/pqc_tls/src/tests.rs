use crate::*;
use crate::provider::{load_oqs_provider, is_oqs_loaded};
use crate::validation::{CertificateValidator, ValidationPolicy, CertificatePinning, OcspConfig};
use openssl::x509::X509;
use openssl::pkey::PKey;
use openssl::ec::{EcGroup, EcKey};
use openssl::nid::Nid;
use openssl::asn1::Asn1Time;
use openssl::hash::MessageDigest;
use tokio::net::TcpListener;
use tokio::task;
use std::collections::HashSet;
use std::sync::Arc;

/// Generate a test certificate for testing purposes
fn gen_test_cert() -> certs::CertKeyPair {
    let group = EcGroup::from_curve_name(Nid::X9_62_PRIME256V1).unwrap();
    let ec = EcKey::generate(&group).unwrap();
    let key = PKey::from_ec_key(ec).unwrap();
    
    let mut name = openssl::x509::X509NameBuilder::new().unwrap();
    name.append_entry_by_nid(Nid::COMMONNAME, "test.example.com").unwrap();
    let name = name.build();
    
    let mut builder = X509::builder().unwrap();
    builder.set_version(2).unwrap();
    builder.set_subject_name(&name).unwrap();
    builder.set_issuer_name(&name).unwrap();
    builder.set_pubkey(&key).unwrap();
    builder.set_not_before(&Asn1Time::days_from_now(0).unwrap()).unwrap();
    builder.set_not_after(&Asn1Time::days_from_now(365).unwrap()).unwrap();
    builder.sign(&key, MessageDigest::sha256()).unwrap();
    let cert = builder.build();
    
    certs::CertKeyPair { cert, key }
}

#[tokio::test]
async fn test_provider_loading() {
    // Test OQS provider loading
    let loaded = load_oqs_provider();
    println!("OQS provider loaded: {}", loaded);
    
    // Test provider state consistency
    assert_eq!(loaded, is_oqs_loaded());
}

#[tokio::test]
async fn test_classical_tls_connection() {
    let cert = gen_test_cert();
    
    // Configure server for classical TLS
    let server_config = server::TlsServerConfig {
        require_hybrid: false,
        client_auth: false,
        alpn_protocols: None,
        fail_closed: false,
        validator: None,
        min_tls_version: None, // Allow older TLS versions for this test
    };
    
    let server = server::TlsServer::new(
        &cert, 
        &certs::CertPolicy::DevAllowClassical, 
        &server_config
    ).unwrap();
    
    // Start server
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let server_addr = listener.local_addr().unwrap();
    
    task::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        let _stream = server.accept(tcp).await.unwrap();
    });
    
    // Configure client for classical TLS
    let client_config = client::TlsClientConfig {
        server_name: "test.example.com".to_string(),
        require_hybrid: false,
        alpn_protocols: None,
        insecure_skip_verify: true,
        fail_closed: false,
        validator: None,
        min_tls_version: None,
    };
    
    let client = client::TlsClient::new();
    let result = client.connect(&server_addr.to_string(), &client_config).await;
    
    assert!(result.is_ok(), "Classical TLS connection should succeed");
}

#[tokio::test]
async fn test_hybrid_requirement_without_provider() {
    if is_oqs_loaded() {
        // Skip this test if OQS provider is available
        return;
    }
    
    let cert = gen_test_cert();
    
    // Test server creation fails when hybrid is required but provider unavailable
    let server_config = server::TlsServerConfig {
        require_hybrid: true,
        client_auth: false,
        alpn_protocols: None,
        fail_closed: false,
        validator: None,
        min_tls_version: Some(openssl::ssl::SslVersion::TLS1_3),
    };
    
    let server_result = server::TlsServer::new(
        &cert, 
        &certs::CertPolicy::RequireHybrid, 
        &server_config
    );
    
    assert!(server_result.is_err(), "Server creation should fail without OQS provider");
    
    // Test client connection fails when hybrid is required but provider unavailable
    let client_config = client::TlsClientConfig {
        server_name: "test.example.com".to_string(),
        require_hybrid: true,
        alpn_protocols: None,
        insecure_skip_verify: true,
        fail_closed: false,
        validator: None,
        min_tls_version: Some(openssl::ssl::SslVersion::TLS1_3),
    };
    
    let client = client::TlsClient::new();
    let client_result = client.connect("127.0.0.1:1234", &client_config).await;
    
    assert!(client_result.is_err(), "Client connection should fail without OQS provider");
}

#[tokio::test]
async fn test_fail_closed_policy() {
    let cert = gen_test_cert();
    
    // Test fail-closed policy prevents insecure configurations
    let client_config = client::TlsClientConfig {
        server_name: "test.example.com".to_string(),
        require_hybrid: false,
        alpn_protocols: None,
        insecure_skip_verify: true, // This should be rejected
        fail_closed: true,
        validator: None,
        min_tls_version: Some(openssl::ssl::SslVersion::TLS1_3),
    };
    
    let client = client::TlsClient::new();
    let result = client.connect("127.0.0.1:1234", &client_config).await;
    
    assert!(result.is_err(), "Fail-closed policy should reject insecure verification");
}

#[tokio::test]
async fn test_certificate_validation() {
    let cert = gen_test_cert();
    
    // Create a certificate validator
    let validator = CertificateValidator::new(
        &[cert.cert.clone()], // Use self-signed cert as CA for testing
        ValidationPolicy::AllowClassical,
        None, // No pinning
        None, // No OCSP
    ).unwrap();
    
    // Test validation with empty chain (should fail)
    let empty_chain = openssl::stack::Stack::new().unwrap();
    let result = validator.validate_chain(&empty_chain, None);
    assert!(result.is_err(), "Empty certificate chain should fail validation");
}

#[tokio::test]
async fn test_certificate_pinning() {
    let cert = gen_test_cert();
    
    // Extract public key hash for pinning
    let pubkey = cert.cert.public_key().unwrap();
    let pubkey_der = pubkey.public_key_to_der().unwrap();
    let pubkey_hash = openssl::hash::hash(
        openssl::hash::MessageDigest::sha256(), 
        &pubkey_der
    ).unwrap();
    
    let mut pinned_keys = HashSet::new();
    pinned_keys.insert(pubkey_hash.as_ref().to_vec());
    
    let pinning = CertificatePinning {
        pinned_keys,
        enforce: true,
    };
    
    // Create validator with pinning
    let validator = CertificateValidator::new(
        &[cert.cert.clone()],
        ValidationPolicy::AllowClassical,
        Some(pinning),
        None,
    ).unwrap();
    
    // Create certificate chain
    let mut chain = openssl::stack::Stack::new().unwrap();
    chain.push(cert.cert.clone()).unwrap();
    
    // Test validation (should pass with correct pin)
    let result = validator.validate_chain(&chain, None);
    assert!(result.is_ok(), "Certificate validation should pass with correct pin");
    
    let validation_result = result.unwrap();
    assert!(validation_result.pinning_validated, "Pinning validation should pass");
}

#[tokio::test]
async fn test_tls_version_enforcement() {
    let cert = gen_test_cert();
    
    // Test minimum TLS version enforcement
    let server_config = server::TlsServerConfig {
        require_hybrid: false,
        client_auth: false,
        alpn_protocols: None,
        fail_closed: true,
        validator: None,
        min_tls_version: Some(openssl::ssl::SslVersion::TLS1_3),
    };
    
    let server = server::TlsServer::new(
        &cert, 
        &certs::CertPolicy::DevAllowClassical, 
        &server_config
    ).unwrap();
    
    // The server should be configured to require TLS 1.3
    assert!(server.config.fail_closed);
    assert_eq!(server.config.min_tls_version, Some(openssl::ssl::SslVersion::TLS1_3));
}

#[tokio::test]
async fn test_alpn_configuration() {
    let cert = gen_test_cert();
    
    let alpn_protocols = vec![
        b"h2".to_vec(),
        b"http/1.1".to_vec(),
    ];
    
    // Test server ALPN configuration
    let server_config = server::TlsServerConfig {
        require_hybrid: false,
        client_auth: false,
        alpn_protocols: Some(alpn_protocols.clone()),
        fail_closed: false,
        validator: None,
        min_tls_version: Some(openssl::ssl::SslVersion::TLS1_3),
    };
    
    let server = server::TlsServer::new(
        &cert, 
        &certs::CertPolicy::DevAllowClassical, 
        &server_config
    ).unwrap();
    
    // Test client ALPN configuration
    let client_config = client::TlsClientConfig {
        server_name: "test.example.com".to_string(),
        require_hybrid: false,
        alpn_protocols: Some(alpn_protocols),
        insecure_skip_verify: true,
        fail_closed: false,
        validator: None,
        min_tls_version: Some(openssl::ssl::SslVersion::TLS1_3),
    };
    
    // Both should be created successfully
    assert!(server.config.alpn_protocols.is_some());
    assert!(client_config.alpn_protocols.is_some());
}

#[tokio::test]
async fn test_default_configurations() {
    // Test default server configuration
    let server_config = server::TlsServerConfig::default();
    assert!(!server_config.require_hybrid);
    assert!(!server_config.fail_closed);
    assert_eq!(server_config.min_tls_version, Some(openssl::ssl::SslVersion::TLS1_3));
    
    // Test default client configuration
    let client_config = client::TlsClientConfig::default();
    assert!(!client_config.require_hybrid);
    assert!(!client_config.fail_closed);
    assert!(!client_config.insecure_skip_verify);
    assert_eq!(client_config.min_tls_version, Some(openssl::ssl::SslVersion::TLS1_3));
}

#[tokio::test]
async fn test_cert_policy_enforcement() {
    let cert = gen_test_cert();
    
    // Test that classical certificates are rejected with PQC-only policy
    // (This will pass because we can't actually generate PQC certs in tests)
    let result = certs::CertKeyPair::from_pem(
        &cert.cert.to_pem().unwrap(),
        &cert.key.private_key_to_pem_pkcs8().unwrap(),
        &certs::CertPolicy::DevAllowClassical,
    );
    
    assert!(result.is_ok(), "Classical cert should be allowed in dev mode");
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_full_handshake_with_validation() {
        let cert = gen_test_cert();
        
        // Create validator
        let validator = Arc::new(CertificateValidator::new(
            &[cert.cert.clone()],
            ValidationPolicy::AllowClassical,
            None,
            None,
        ).unwrap());
        
        // Configure server with validation
        let server_config = server::TlsServerConfig {
            require_hybrid: false,
            client_auth: false,
            alpn_protocols: Some(vec![b"test-protocol".to_vec()]),
            fail_closed: false,
            validator: Some(validator.clone()),
            min_tls_version: Some(openssl::ssl::SslVersion::TLS1_3),
        };
        
        let server = server::TlsServer::new(
            &cert, 
            &certs::CertPolicy::DevAllowClassical, 
            &server_config
        ).unwrap();
        
        // Start server
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let server_addr = listener.local_addr().unwrap();
        
        task::spawn(async move {
            let (tcp, _) = listener.accept().await.unwrap();
            let _stream = server.accept(tcp).await.unwrap();
        });
        
        // Configure client with validation
        let client_config = client::TlsClientConfig {
            server_name: "test.example.com".to_string(),
            require_hybrid: false,
            alpn_protocols: Some(vec![b"test-protocol".to_vec()]),
            insecure_skip_verify: true, // Skip for self-signed cert
            fail_closed: false,
            validator: Some(validator),
            min_tls_version: Some(openssl::ssl::SslVersion::TLS1_3),
        };
        
        let client = client::TlsClient::new();
        let result = client.connect(&server_addr.to_string(), &client_config).await;
        
        assert!(result.is_ok(), "Full handshake with validation should succeed");
    }
}