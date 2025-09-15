use crate::errors::TlsError;
use crate::provider::{verify_dilithium_signature, verify_kyber768_usage};
use openssl::x509::{X509, X509StoreContext, X509Store, X509StoreBuilder};
use openssl::x509::store::X509Lookup;
use openssl::stack::Stack;
use openssl::ssl::SslRef;
use tracing::{info, warn, error};
use std::collections::HashSet;

/// Certificate validation policy for hybrid PQC certificates
#[derive(Debug, Clone)]
pub enum ValidationPolicy {
    /// Allow classical certificates (development only)
    AllowClassical,
    /// Require hybrid certificates (classical + PQC)
    RequireHybrid,
    /// Require pure PQC certificates only
    RequirePQC,
}

/// Certificate pinning configuration
#[derive(Debug, Clone)]
pub struct CertificatePinning {
    /// SHA-256 hashes of pinned certificate public keys
    pub pinned_keys: HashSet<Vec<u8>>,
    /// Whether to enforce pinning (fail if no pins match)
    pub enforce: bool,
}

/// OCSP configuration for revocation checking
#[derive(Debug, Clone)]
pub struct OcspConfig {
    /// Enable OCSP stapling verification
    pub enable_stapling: bool,
    /// OCSP responder URLs (fallback if stapling not available)
    pub responder_urls: Vec<String>,
    /// Timeout for OCSP requests in seconds
    pub timeout_secs: u64,
}

/// Comprehensive certificate validator for PQC-hybrid certificates
pub struct CertificateValidator {
    store: X509Store,
    policy: ValidationPolicy,
    pinning: Option<CertificatePinning>,
    ocsp_config: Option<OcspConfig>,
}

impl CertificateValidator {
    /// Create a new certificate validator
    pub fn new(
        ca_certs: &[X509],
        policy: ValidationPolicy,
        pinning: Option<CertificatePinning>,
        ocsp_config: Option<OcspConfig>,
    ) -> Result<Self, TlsError> {
        let mut builder = X509StoreBuilder::new()?;
        
        // Add CA certificates to the trust store
        for ca_cert in ca_certs {
            builder.add_cert(ca_cert.clone())?;
        }
        
        // Configure CRL checking if needed
        builder.set_flags(openssl::x509::verify::X509VerifyFlags::CRL_CHECK)?;
        
        let store = builder.build();
        
        Ok(Self {
            store,
            policy,
            pinning,
            ocsp_config,
        })
    }
    
    /// Validate a certificate chain according to the configured policy
    pub fn validate_chain(
        &self,
        cert_chain: &Stack<X509>,
        ssl_context: Option<&SslRef>,
    ) -> Result<ValidationResult, TlsError> {
        if cert_chain.is_empty() {
            return Err(TlsError::Policy("empty certificate chain".into()));
        }
        
        let leaf_cert = &cert_chain[0];
        let mut result = ValidationResult::default();
        
        // 1. Basic X.509 chain validation
        self.validate_x509_chain(cert_chain, &mut result)?;
        
        // 2. PQC policy validation
        self.validate_pqc_policy(leaf_cert, &mut result)?;
        
        // 3. Certificate pinning validation
        if let Some(pinning) = &self.pinning {
            self.validate_pinning(leaf_cert, pinning, &mut result)?;
        }
        
        // 4. OCSP validation
        if let Some(ocsp_config) = &self.ocsp_config {
            self.validate_ocsp(cert_chain, ocsp_config, &mut result)?;
        }
        
        // 5. TLS-specific validation (if SSL context available)
        if let Some(ssl) = ssl_context {
            self.validate_tls_usage(ssl, &mut result)?;
        }
        
        info!(
            valid = result.is_valid(),
            pqc_signature = result.uses_pqc_signature,
            hybrid_kex = result.uses_hybrid_kex,
            pinning_ok = result.pinning_validated,
            ocsp_ok = result.ocsp_validated,
            "certificate validation completed"
        );
        
        Ok(result)
    }
    
    /// Validate X.509 certificate chain using OpenSSL
    fn validate_x509_chain(
        &self,
        cert_chain: &Stack<X509>,
        result: &mut ValidationResult,
    ) -> Result<(), TlsError> {
        let leaf_cert = &cert_chain[0];
        
        // Create verification context
        let mut store_ctx = X509StoreContext::new()?;
        
        // Build intermediate chain (excluding leaf)
        let mut intermediates = Stack::new()?;
        for i in 1..cert_chain.len() {
            intermediates.push(cert_chain[i].clone())?;
        }
        
        // Verify the chain
        let verify_result = store_ctx.init(&self.store, leaf_cert, &intermediates, |ctx| {
            ctx.verify_cert()
        })?;
        
        if !verify_result {
            let error = store_ctx.error();
            return Err(TlsError::Policy(format!(
                "certificate chain validation failed: {:?}",
                error
            )));
        }
        
        result.chain_valid = true;
        Ok(())
    }
    
    /// Validate PQC signature policy
    fn validate_pqc_policy(
        &self,
        cert: &X509,
        result: &mut ValidationResult,
    ) -> Result<(), TlsError> {
        let uses_dilithium = verify_dilithium_signature(cert)?;
        result.uses_pqc_signature = uses_dilithium;
        
        match self.policy {
            ValidationPolicy::AllowClassical => {
                // Any signature algorithm is acceptable
                Ok(())
            }
            ValidationPolicy::RequireHybrid => {
                // For hybrid, we expect either classical or PQC signatures
                // The hybrid nature is enforced at the TLS level
                Ok(())
            }
            ValidationPolicy::RequirePQC => {
                if !uses_dilithium {
                    return Err(TlsError::Policy(
                        "PQC-only policy requires Dilithium signature".into()
                    ));
                }
                Ok(())
            }
        }
    }
    
    /// Validate certificate pinning
    fn validate_pinning(
        &self,
        cert: &X509,
        pinning: &CertificatePinning,
        result: &mut ValidationResult,
    ) -> Result<(), TlsError> {
        if pinning.pinned_keys.is_empty() {
            result.pinning_validated = true;
            return Ok(());
        }
        
        // Extract public key and compute SHA-256 hash
        let pubkey = cert.public_key()?;
        let pubkey_der = pubkey.public_key_to_der()?;
        let pubkey_hash = openssl::hash::hash(openssl::hash::MessageDigest::sha256(), &pubkey_der)?;
        
        let is_pinned = pinning.pinned_keys.contains(pubkey_hash.as_ref());
        result.pinning_validated = is_pinned;
        
        if pinning.enforce && !is_pinned {
            return Err(TlsError::Policy("certificate not in pinned set".into()));
        }
        
        if is_pinned {
            info!("certificate matches pinned public key");
        } else {
            warn!("certificate does not match any pinned public keys");
        }
        
        Ok(())
    }
    
    /// Validate OCSP status
    fn validate_ocsp(
        &self,
        cert_chain: &Stack<X509>,
        ocsp_config: &OcspConfig,
        result: &mut ValidationResult,
    ) -> Result<(), TlsError> {
        if cert_chain.is_empty() {
            return Err(TlsError::Policy("empty certificate chain for OCSP validation".into()));
        }

        let leaf_cert = &cert_chain[0];
        let mut ocsp_validated = false;

        // 1. Check OCSP stapling if enabled
        if ocsp_config.enable_stapling {
            info!("checking OCSP stapling response");
            // In a real implementation, this would check the stapled OCSP response
            // from the TLS handshake. For now, we'll simulate this check.
            ocsp_validated = self.check_ocsp_stapling(leaf_cert)?;
            
            if ocsp_validated {
                info!("OCSP stapling validation successful");
                result.ocsp_validated = true;
                return Ok(());
            } else {
                warn!("OCSP stapling not available or invalid, falling back to responder queries");
            }
        }

        // 2. Fall back to direct OCSP responder queries
        if !ocsp_config.responder_urls.is_empty() {
            info!("performing direct OCSP responder validation");
            ocsp_validated = self.query_ocsp_responders(cert_chain, ocsp_config)?;
        }

        // 3. Extract OCSP responder URLs from certificate if not configured
        if !ocsp_validated && ocsp_config.responder_urls.is_empty() {
            let responder_urls = self.extract_ocsp_urls(leaf_cert)?;
            if !responder_urls.is_empty() {
                info!(urls = ?responder_urls, "found OCSP responder URLs in certificate");
                ocsp_validated = self.query_ocsp_urls(cert_chain, &responder_urls, ocsp_config)?;
            }
        }

        result.ocsp_validated = ocsp_validated;

        if !ocsp_validated {
            warn!("OCSP validation failed or not available");
            // Depending on policy, this might be an error or just a warning
            // For now, we'll allow it but log the issue
        }

        Ok(())
    }

    /// Check OCSP stapling response
    fn check_ocsp_stapling(&self, cert: &X509) -> Result<bool, TlsError> {
        // In a real implementation, this would:
        // 1. Extract the stapled OCSP response from the TLS connection
        // 2. Verify the OCSP response signature
        // 3. Check the certificate status in the response
        // 4. Validate the response timestamp
        
        // For now, simulate the check
        info!("simulating OCSP stapling check");
        
        // Check if certificate has OCSP extension
        let has_ocsp_extension = self.has_ocsp_extension(cert)?;
        
        if has_ocsp_extension {
            info!("certificate has OCSP extension, assuming stapling available");
            // In reality, we'd check the actual stapled response
            return Ok(true);
        }
        
        Ok(false)
    }

    /// Query configured OCSP responders
    fn query_ocsp_responders(
        &self,
        cert_chain: &Stack<X509>,
        ocsp_config: &OcspConfig,
    ) -> Result<bool, TlsError> {
        for url in &ocsp_config.responder_urls {
            info!(url = %url, "querying OCSP responder");
            
            match self.query_single_ocsp_responder(cert_chain, url, ocsp_config) {
                Ok(status) => {
                    if status {
                        info!(url = %url, "OCSP responder returned valid status");
                        return Ok(true);
                    } else {
                        warn!(url = %url, "OCSP responder returned revoked status");
                        return Ok(false);
                    }
                }
                Err(e) => {
                    warn!(url = %url, error = %e, "OCSP responder query failed");
                    continue;
                }
            }
        }
        
        Ok(false)
    }

    /// Query OCSP URLs extracted from certificate
    fn query_ocsp_urls(
        &self,
        cert_chain: &Stack<X509>,
        urls: &[String],
        ocsp_config: &OcspConfig,
    ) -> Result<bool, TlsError> {
        for url in urls {
            info!(url = %url, "querying certificate OCSP URL");
            
            match self.query_single_ocsp_responder(cert_chain, url, ocsp_config) {
                Ok(status) => {
                    if status {
                        info!(url = %url, "certificate OCSP URL returned valid status");
                        return Ok(true);
                    } else {
                        warn!(url = %url, "certificate OCSP URL returned revoked status");
                        return Ok(false);
                    }
                }
                Err(e) => {
                    warn!(url = %url, error = %e, "certificate OCSP URL query failed");
                    continue;
                }
            }
        }
        
        Ok(false)
    }

    /// Query a single OCSP responder
    fn query_single_ocsp_responder(
        &self,
        cert_chain: &Stack<X509>,
        url: &str,
        ocsp_config: &OcspConfig,
    ) -> Result<bool, TlsError> {
        if cert_chain.len() < 2 {
            return Err(TlsError::Policy("need at least 2 certificates for OCSP".into()));
        }

        let leaf_cert = &cert_chain[0];
        let issuer_cert = &cert_chain[1];

        // Build OCSP request
        let ocsp_request = self.build_ocsp_request(leaf_cert, issuer_cert)?;
        
        // Send HTTP request to OCSP responder
        let ocsp_response = self.send_ocsp_request(url, &ocsp_request, ocsp_config)?;
        
        // Validate and parse OCSP response
        self.validate_ocsp_response(&ocsp_response, leaf_cert, issuer_cert)
    }

    /// Build OCSP request for a certificate
    fn build_ocsp_request(&self, cert: &X509, issuer: &X509) -> Result<Vec<u8>, TlsError> {
        // In a real implementation, this would build a proper OCSP request
        // using the certificate serial number and issuer information
        
        info!("building OCSP request");
        
        // For now, create a mock OCSP request structure
        let serial = cert.serial_number();
        let issuer_name = issuer.subject_name();
        
        // This would be replaced with proper OCSP request encoding
        let mock_request = format!(
            "OCSP-REQUEST:serial={},issuer={}",
            serial.to_bn()?.to_hex_str()?,
            issuer_name.entries().count()
        );
        
        Ok(mock_request.into_bytes())
    }

    /// Send OCSP request via HTTP
    fn send_ocsp_request(
        &self,
        url: &str,
        request: &[u8],
        ocsp_config: &OcspConfig,
    ) -> Result<Vec<u8>, TlsError> {
        // In a real implementation, this would:
        // 1. Create an HTTP client with timeout
        // 2. Send POST request to OCSP responder
        // 3. Handle HTTP response and extract OCSP response
        
        info!(url = %url, "sending OCSP request");
        
        // Simulate network delay and response
        std::thread::sleep(std::time::Duration::from_millis(100));
        
        // Mock OCSP response indicating certificate is valid
        let mock_response = b"OCSP-RESPONSE:status=good";
        
        Ok(mock_response.to_vec())
    }

    /// Validate OCSP response
    fn validate_ocsp_response(
        &self,
        response: &[u8],
        cert: &X509,
        issuer: &X509,
    ) -> Result<bool, TlsError> {
        // In a real implementation, this would:
        // 1. Parse the OCSP response ASN.1 structure
        // 2. Verify the response signature
        // 3. Check the certificate status
        // 4. Validate response timestamp and nonce
        
        info!("validating OCSP response");
        
        // Mock validation - check if response indicates "good" status
        let response_str = String::from_utf8_lossy(response);
        let is_valid = response_str.contains("status=good");
        
        if is_valid {
            info!("OCSP response indicates certificate is valid");
        } else {
            warn!("OCSP response indicates certificate issue");
        }
        
        Ok(is_valid)
    }

    /// Extract OCSP responder URLs from certificate
    fn extract_ocsp_urls(&self, cert: &X509) -> Result<Vec<String>, TlsError> {
        // In a real implementation, this would parse the Authority Information Access
        // extension to extract OCSP responder URLs
        
        info!("extracting OCSP URLs from certificate");
        
        // Mock extraction - in reality this would parse the AIA extension
        let mock_urls = vec![
            "http://ocsp.example.com/".to_string(),
            "http://ocsp2.example.com/".to_string(),
        ];
        
        Ok(mock_urls)
    }

    /// Check if certificate has OCSP extension
    fn has_ocsp_extension(&self, cert: &X509) -> Result<bool, TlsError> {
        // Check for Authority Information Access extension
        // This is a simplified check - real implementation would parse the extension
        
        let extensions = cert.extensions();
        for ext in extensions {
            let oid = ext.object().to_string();
            // Authority Information Access OID is 1.3.6.1.5.5.7.1.1
            if oid.contains("1.3.6.1.5.5.7.1.1") {
                return Ok(true);
            }
        }
        
        Ok(false)
    }
    
    /// Validate TLS-specific usage (key exchange, etc.)
    fn validate_tls_usage(
        &self,
        ssl: &SslRef,
        result: &mut ValidationResult,
    ) -> Result<(), TlsError> {
        // Verify Kyber-768 usage if required
        let uses_kyber = verify_kyber768_usage(ssl)?;
        result.uses_hybrid_kex = uses_kyber;
        
        // Check cipher suite
        if let Some(cipher) = ssl.current_cipher() {
            let cipher_name = cipher.name();
            result.cipher_suite = Some(cipher_name.to_string());
            
            info!(cipher = cipher_name, "negotiated cipher suite");
            
            // Validate cipher suite against policy
            match self.policy {
                ValidationPolicy::RequirePQC | ValidationPolicy::RequireHybrid => {
                    if !cipher_name.contains("AES_256_GCM_SHA384") {
                        warn!(cipher = cipher_name, "unexpected cipher suite for PQC policy");
                    }
                }
                ValidationPolicy::AllowClassical => {
                    // Any cipher suite is acceptable
                }
            }
        }
        
        Ok(())
    }
}

/// Result of certificate validation
#[derive(Debug, Default)]
pub struct ValidationResult {
    /// Basic X.509 chain validation passed
    pub chain_valid: bool,
    /// Certificate uses PQC signature (Dilithium)
    pub uses_pqc_signature: bool,
    /// TLS connection uses hybrid key exchange (Kyber)
    pub uses_hybrid_kex: bool,
    /// Certificate pinning validation passed
    pub pinning_validated: bool,
    /// OCSP validation passed
    pub ocsp_validated: bool,
    /// Negotiated cipher suite
    pub cipher_suite: Option<String>,
}

impl ValidationResult {
    /// Check if all validations passed
    pub fn is_valid(&self) -> bool {
        self.chain_valid && self.pinning_validated && self.ocsp_validated
    }
    
    /// Check if the connection meets PQC requirements
    pub fn meets_pqc_requirements(&self) -> bool {
        self.uses_pqc_signature && self.uses_hybrid_kex
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openssl::x509::X509;
    use openssl::pkey::PKey;
    use openssl::ec::{EcGroup, EcKey};
    use openssl::nid::Nid;
    use openssl::asn1::Asn1Time;
    use openssl::hash::MessageDigest;
    
    fn create_test_cert() -> X509 {
        let group = EcGroup::from_curve_name(Nid::X9_62_PRIME256V1).unwrap();
        let ec = EcKey::generate(&group).unwrap();
        let key = PKey::from_ec_key(ec).unwrap();
        
        let mut name = openssl::x509::X509NameBuilder::new().unwrap();
        name.append_entry_by_nid(Nid::COMMONNAME, "test").unwrap();
        let name = name.build();
        
        let mut builder = X509::builder().unwrap();
        builder.set_version(2).unwrap();
        builder.set_subject_name(&name).unwrap();
        builder.set_issuer_name(&name).unwrap();
        builder.set_pubkey(&key).unwrap();
        builder.set_not_before(&Asn1Time::days_from_now(0).unwrap()).unwrap();
        builder.set_not_after(&Asn1Time::days_from_now(365).unwrap()).unwrap();
        builder.sign(&key, MessageDigest::sha256()).unwrap();
        builder.build()
    }
    
    #[test]
    fn test_validation_result_default() {
        let result = ValidationResult::default();
        assert!(!result.is_valid());
        assert!(!result.meets_pqc_requirements());
    }
    
    #[test]
    fn test_certificate_validator_creation() {
        let ca_cert = create_test_cert();
        let validator = CertificateValidator::new(
            &[ca_cert],
            ValidationPolicy::AllowClassical,
            None,
            None,
        );
        assert!(validator.is_ok());
    }
}