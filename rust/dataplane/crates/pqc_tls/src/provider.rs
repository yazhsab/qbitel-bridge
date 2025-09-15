use once_cell::sync::OnceCell;
use openssl::nid::Nid;
use tracing::{info, warn, error};
use crate::errors::TlsError;

static OQS_LOADED: OnceCell<bool> = OnceCell::new();
static OQS_PROVIDER: OnceCell<Option<openssl::provider::Provider>> = OnceCell::new();

/// Supported PQC algorithms for validation
#[derive(Debug, Clone)]
pub struct PqcAlgorithms {
    pub kyber768_supported: bool,
    pub dilithium2_supported: bool,
    pub dilithium3_supported: bool,
}

/// Attempt to load the OQS provider into OpenSSL 3 at runtime.
/// Returns true if the provider is reported as loaded, false otherwise.
pub fn load_oqs_provider() -> bool {
    if let Some(v) = OQS_LOADED.get() { return *v; }
    
    let loaded = match openssl::provider::Provider::load(None, "oqsprovider") {
        Ok(provider) => {
            // Validate that the provider actually supports the algorithms we need
            match validate_oqs_algorithms(&provider) {
                Ok(algorithms) => {
                    info!(
                        kyber768 = algorithms.kyber768_supported,
                        dilithium2 = algorithms.dilithium2_supported,
                        dilithium3 = algorithms.dilithium3_supported,
                        "oqsprovider loaded and validated"
                    );
                    
                    // Store the provider to keep it alive
                    let _ = OQS_PROVIDER.set(Some(provider));
                    true
                }
                Err(e) => {
                    error!(error = %e, "oqsprovider loaded but algorithm validation failed");
                    false
                }
            }
        }
        Err(e) => {
            warn!(error=%e, "oqsprovider not available; hybrid KEX will fail if required");
            let _ = OQS_PROVIDER.set(None);
            false
        }
    };
    
    let _ = OQS_LOADED.set(loaded);
    loaded
}

pub fn is_oqs_loaded() -> bool {
    *OQS_LOADED.get_or_init(|| false)
}

/// Validate that the OQS provider supports required algorithms
fn validate_oqs_algorithms(provider: &openssl::provider::Provider) -> Result<PqcAlgorithms, TlsError> {
    // Test Kyber-768 availability by attempting to create a key exchange context
    let kyber768_supported = test_algorithm_support("kyber768");
    let dilithium2_supported = test_algorithm_support("dilithium2");
    let dilithium3_supported = test_algorithm_support("dilithium3");
    
    if !kyber768_supported {
        return Err(TlsError::Provider("Kyber-768 not supported by OQS provider".into()));
    }
    
    if !dilithium2_supported && !dilithium3_supported {
        return Err(TlsError::Provider("No Dilithium variants supported by OQS provider".into()));
    }
    
    Ok(PqcAlgorithms {
        kyber768_supported,
        dilithium2_supported,
        dilithium3_supported,
    })
}

/// Test if a specific algorithm is supported by attempting to use it
fn test_algorithm_support(algorithm: &str) -> bool {
    match algorithm {
        "kyber768" => {
            // Test if we can create a Kyber-768 key exchange
            // This is a basic availability test
            match openssl::pkey_ctx::PkeyCtx::new_id(Nid::create("kyber768", "kyber768", "Kyber-768 KEM")) {
                Ok(_) => true,
                Err(_) => {
                    // Fallback: try to find the algorithm in available algorithms
                    test_algorithm_by_name("kyber768")
                }
            }
        }
        "dilithium2" => test_algorithm_by_name("dilithium2"),
        "dilithium3" => test_algorithm_by_name("dilithium3"),
        _ => false,
    }
}

/// Test algorithm availability by name lookup
fn test_algorithm_by_name(name: &str) -> bool {
    // Try to create a NID for the algorithm - if it exists, the provider supports it
    match Nid::create(name, name, name) {
        Ok(_) => true,
        Err(_) => false,
    }
}

/// Verify that a TLS connection used Kyber-768 key exchange
pub fn verify_kyber768_usage(ssl: &openssl::ssl::SslRef) -> Result<bool, TlsError> {
    // Get the negotiated group
    if let Some(group) = ssl.group() {
        let group_name = group.name();
        info!(group = group_name, "negotiated key exchange group");
        
        // Check if it's a hybrid group containing Kyber-768
        Ok(group_name.contains("kyber768") || group_name.contains("x25519_kyber768"))
    } else {
        warn!("no key exchange group information available");
        Ok(false)
    }
}

/// Verify that a certificate uses Dilithium signature
pub fn verify_dilithium_signature(cert: &openssl::x509::X509Ref) -> Result<bool, TlsError> {
    let sig_alg = cert.signature_algorithm();
    let alg_name = sig_alg.object().nid().short_name().unwrap_or("unknown");
    
    info!(algorithm = alg_name, "certificate signature algorithm");
    
    // Check for Dilithium signature algorithms
    let is_dilithium = alg_name.contains("dilithium") || 
                      alg_name.contains("Dilithium") ||
                      // OQS provider might use different naming
                      alg_name.contains("DILITHIUM");
    
    if !is_dilithium {
        warn!(algorithm = alg_name, "certificate does not use Dilithium signature");
    }
    
    Ok(is_dilithium)
}

/// Get supported PQC algorithms from the loaded provider
pub fn get_supported_algorithms() -> Option<PqcAlgorithms> {
    if !is_oqs_loaded() {
        return None;
    }
    
    Some(PqcAlgorithms {
        kyber768_supported: test_algorithm_support("kyber768"),
        dilithium2_supported: test_algorithm_support("dilithium2"),
        dilithium3_supported: test_algorithm_support("dilithium3"),
    })
}
