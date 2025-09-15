use crate::parser::{Iso8583Message, Iso8583Parser};
use aes::Aes256;
use aes::cipher::{BlockEncrypt, BlockDecrypt, KeyInit, generic_array::GenericArray};
use sha2::{Sha256, Digest};
use rand::{Rng, thread_rng};
use std::collections::HashMap;
use thiserror::Error;
use tracing::{debug, warn, error};

#[derive(Debug, Error)]
pub enum SecurityError {
    #[error("encryption error: {0}")]
    Encryption(String),
    #[error("decryption error: {0}")]
    Decryption(String),
    #[error("key management error: {0}")]
    KeyManagement(String),
    #[error("MAC validation error: {0}")]
    MacValidation(String),
    #[error("field masking error: {0}")]
    FieldMasking(String),
}

/// PCI DSS compliance levels for field masking
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PciComplianceLevel {
    /// No masking (internal processing only)
    None,
    /// Partial masking (show first 6 and last 4 digits of PAN)
    Partial,
    /// Full masking (replace with asterisks)
    Full,
    /// Tokenization (replace with non-sensitive token)
    Tokenized,
}

/// Field masking configuration
#[derive(Debug, Clone)]
pub struct MaskingConfig {
    /// Default masking level
    pub default_level: PciComplianceLevel,
    /// Field-specific masking levels
    pub field_levels: HashMap<u8, PciComplianceLevel>,
    /// Masking character (default: '*')
    pub mask_char: char,
}

impl Default for MaskingConfig {
    fn default() -> Self {
        let mut field_levels = HashMap::new();
        
        // Configure sensitive fields according to PCI DSS requirements
        field_levels.insert(2, PciComplianceLevel::Partial);   // PAN
        field_levels.insert(14, PciComplianceLevel::Full);     // Expiration date
        field_levels.insert(34, PciComplianceLevel::Partial);  // PAN extended
        field_levels.insert(35, PciComplianceLevel::Full);     // Track 2 data
        field_levels.insert(36, PciComplianceLevel::Full);     // Track 3 data
        field_levels.insert(45, PciComplianceLevel::Full);     // Track 1 data
        field_levels.insert(52, PciComplianceLevel::Full);     // PIN data
        field_levels.insert(53, PciComplianceLevel::Full);     // Security control info
        field_levels.insert(64, PciComplianceLevel::Full);     // MAC
        field_levels.insert(96, PciComplianceLevel::Full);     // Message security code
        field_levels.insert(102, PciComplianceLevel::Partial); // Account ID 1
        field_levels.insert(103, PciComplianceLevel::Partial); // Account ID 2
        field_levels.insert(128, PciComplianceLevel::Full);    // MAC
        
        Self {
            default_level: PciComplianceLevel::None,
            field_levels,
            mask_char: '*',
        }
    }
}

/// Key management for cryptographic operations
#[derive(Debug)]
pub struct KeyManager {
    /// Data encryption keys (DEK)
    data_keys: HashMap<String, [u8; 32]>,
    /// Key encryption keys (KEK)
    key_keys: HashMap<String, [u8; 32]>,
    /// MAC keys
    mac_keys: HashMap<String, [u8; 32]>,
}

impl KeyManager {
    /// Create a new key manager
    pub fn new() -> Self {
        Self {
            data_keys: HashMap::new(),
            key_keys: HashMap::new(),
            mac_keys: HashMap::new(),
        }
    }
    
    /// Generate a new random key
    pub fn generate_key() -> [u8; 32] {
        let mut key = [0u8; 32];
        thread_rng().fill(&mut key);
        key
    }
    
    /// Add a data encryption key
    pub fn add_data_key(&mut self, key_id: String, key: [u8; 32]) {
        self.data_keys.insert(key_id, key);
    }
    
    /// Add a MAC key
    pub fn add_mac_key(&mut self, key_id: String, key: [u8; 32]) {
        self.mac_keys.insert(key_id, key);
    }
    
    /// Get a data encryption key
    pub fn get_data_key(&self, key_id: &str) -> Option<&[u8; 32]> {
        self.data_keys.get(key_id)
    }
    
    /// Get a MAC key
    pub fn get_mac_key(&self, key_id: &str) -> Option<&[u8; 32]> {
        self.mac_keys.get(key_id)
    }
    
    /// Derive key from password and salt
    pub fn derive_key(password: &str, salt: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(password.as_bytes());
        hasher.update(salt);
        let result = hasher.finalize();
        
        let mut key = [0u8; 32];
        key.copy_from_slice(&result);
        key
    }
}

/// Security processor for ISO-8583 messages
pub struct SecurityProcessor {
    key_manager: KeyManager,
    masking_config: MaskingConfig,
}

impl SecurityProcessor {
    /// Create a new security processor
    pub fn new(key_manager: KeyManager, masking_config: MaskingConfig) -> Self {
        Self {
            key_manager,
            masking_config,
        }
    }
    
    /// Mask sensitive fields in an ISO-8583 message for PCI compliance
    pub fn mask_sensitive_fields(
        &self,
        message: &mut Iso8583Message,
        parser: &Iso8583Parser,
    ) -> Result<(), SecurityError> {
        for (field_num, field_value) in message.fields.iter_mut() {
            let masking_level = self.masking_config.field_levels
                .get(field_num)
                .copied()
                .unwrap_or(self.masking_config.default_level);
            
            if masking_level != PciComplianceLevel::None {
                let is_sensitive = parser.is_sensitive_field(*field_num);
                if is_sensitive {
                    *field_value = self.apply_masking(field_value, *field_num, masking_level)?;
                    debug!(field = field_num, level = ?masking_level, "masked sensitive field");
                }
            }
        }
        
        Ok(())
    }
    
    /// Apply masking to a field value
    fn apply_masking(
        &self,
        value: &str,
        field_num: u8,
        level: PciComplianceLevel,
    ) -> Result<String, SecurityError> {
        match level {
            PciComplianceLevel::None => Ok(value.to_string()),
            PciComplianceLevel::Partial => {
                match field_num {
                    2 | 34 | 102 | 103 => {
                        // PAN masking: show first 6 and last 4 digits
                        self.mask_pan(value)
                    }
                    _ => {
                        // Generic partial masking: show first and last 2 characters
                        self.mask_partial(value)
                    }
                }
            }
            PciComplianceLevel::Full => {
                Ok(self.mask_char.to_string().repeat(value.len().min(16)))
            }
            PciComplianceLevel::Tokenized => {
                // Generate a deterministic token based on the value
                self.tokenize_value(value, field_num)
            }
        }
    }
    
    /// Mask PAN according to PCI DSS requirements
    fn mask_pan(&self, pan: &str) -> Result<String, SecurityError> {
        let digits_only: String = pan.chars().filter(|c| c.is_ascii_digit()).collect();
        
        if digits_only.len() < 10 {
            return Err(SecurityError::FieldMasking(
                "PAN too short for proper masking".to_string()
            ));
        }
        
        let first_six = &digits_only[..6];
        let last_four = &digits_only[digits_only.len() - 4..];
        let middle_len = digits_only.len() - 10;
        
        Ok(format!("{}{}{}",
            first_six,
            self.mask_char.to_string().repeat(middle_len),
            last_four
        ))
    }
    
    /// Apply partial masking (show first and last 2 characters)
    fn mask_partial(&self, value: &str) -> Result<String, SecurityError> {
        if value.len() <= 4 {
            return Ok(self.mask_char.to_string().repeat(value.len()));
        }
        
        let first_two = &value[..2];
        let last_two = &value[value.len() - 2..];
        let middle_len = value.len() - 4;
        
        Ok(format!("{}{}{}",
            first_two,
            self.mask_char.to_string().repeat(middle_len),
            last_two
        ))
    }
    
    /// Generate a deterministic token for a value
    fn tokenize_value(&self, value: &str, field_num: u8) -> Result<String, SecurityError> {
        let mut hasher = Sha256::new();
        hasher.update(value.as_bytes());
        hasher.update(&[field_num]);
        
        // Add a secret salt if available
        if let Some(key) = self.key_manager.get_data_key("tokenization") {
            hasher.update(key);
        }
        
        let hash = hasher.finalize();
        let token = format!("TOK{}", hex::encode(&hash[..8]));
        
        Ok(token)
    }
    
    /// Encrypt sensitive field data
    pub fn encrypt_field(
        &self,
        field_value: &str,
        key_id: &str,
    ) -> Result<String, SecurityError> {
        let key = self.key_manager.get_data_key(key_id)
            .ok_or_else(|| SecurityError::KeyManagement(format!("key not found: {}", key_id)))?;
        
        let cipher = Aes256::new(GenericArray::from_slice(key));
        
        // Pad the data to block size (16 bytes)
        let mut data = field_value.as_bytes().to_vec();
        let padding_len = 16 - (data.len() % 16);
        data.extend(vec![padding_len as u8; padding_len]);
        
        // Encrypt in blocks
        let mut encrypted = Vec::new();
        for chunk in data.chunks(16) {
            let mut block = GenericArray::clone_from_slice(chunk);
            cipher.encrypt_block(&mut block);
            encrypted.extend_from_slice(&block);
        }
        
        Ok(hex::encode(encrypted))
    }
    
    /// Decrypt sensitive field data
    pub fn decrypt_field(
        &self,
        encrypted_value: &str,
        key_id: &str,
    ) -> Result<String, SecurityError> {
        let key = self.key_manager.get_data_key(key_id)
            .ok_or_else(|| SecurityError::KeyManagement(format!("key not found: {}", key_id)))?;
        
        let cipher = Aes256::new(GenericArray::from_slice(key));
        
        let encrypted_data = hex::decode(encrypted_value)
            .map_err(|e| SecurityError::Decryption(format!("invalid hex: {}", e)))?;
        
        if encrypted_data.len() % 16 != 0 {
            return Err(SecurityError::Decryption("invalid block size".to_string()));
        }
        
        // Decrypt in blocks
        let mut decrypted = Vec::new();
        for chunk in encrypted_data.chunks(16) {
            let mut block = GenericArray::clone_from_slice(chunk);
            cipher.decrypt_block(&mut block);
            decrypted.extend_from_slice(&block);
        }
        
        // Remove padding
        if let Some(&padding_len) = decrypted.last() {
            if padding_len as usize <= decrypted.len() {
                decrypted.truncate(decrypted.len() - padding_len as usize);
            }
        }
        
        String::from_utf8(decrypted)
            .map_err(|e| SecurityError::Decryption(format!("invalid UTF-8: {}", e)))
    }
    
    /// Generate MAC for message authentication
    pub fn generate_mac(
        &self,
        message_data: &[u8],
        key_id: &str,
    ) -> Result<String, SecurityError> {
        let key = self.key_manager.get_mac_key(key_id)
            .ok_or_else(|| SecurityError::KeyManagement(format!("MAC key not found: {}", key_id)))?;
        
        let mut hasher = Sha256::new();
        hasher.update(key);
        hasher.update(message_data);
        let mac = hasher.finalize();
        
        Ok(hex::encode(&mac[..8])) // Use first 8 bytes as MAC
    }
    
    /// Validate MAC
    pub fn validate_mac(
        &self,
        message_data: &[u8],
        expected_mac: &str,
        key_id: &str,
    ) -> Result<bool, SecurityError> {
        let calculated_mac = self.generate_mac(message_data, key_id)?;
        Ok(calculated_mac.eq_ignore_ascii_case(expected_mac))
    }
    
    /// Create a secure copy of a message with sensitive fields encrypted
    pub fn secure_message(
        &self,
        message: &Iso8583Message,
        parser: &Iso8583Parser,
        key_id: &str,
    ) -> Result<Iso8583Message, SecurityError> {
        let mut secure_msg = message.clone();
        
        for (field_num, field_value) in secure_msg.fields.iter_mut() {
            if parser.is_sensitive_field(*field_num) {
                *field_value = self.encrypt_field(field_value, key_id)?;
                debug!(field = field_num, "encrypted sensitive field");
            }
        }
        
        Ok(secure_msg)
    }
    
    /// Audit log entry for security operations
    pub fn audit_log(&self, operation: &str, field_num: Option<u8>, success: bool) {
        if success {
            debug!(
                operation = operation,
                field = field_num,
                "security operation completed"
            );
        } else {
            warn!(
                operation = operation,
                field = field_num,
                "security operation failed"
            );
        }
    }
}

/// Security policy for ISO-8583 processing
#[derive(Debug, Clone)]
pub struct SecurityPolicy {
    /// Require encryption for sensitive fields
    pub require_encryption: bool,
    /// Require MAC validation
    pub require_mac: bool,
    /// PCI compliance level
    pub pci_level: PciComplianceLevel,
    /// Allowed message types
    pub allowed_mtis: Vec<String>,
    /// Maximum message size
    pub max_message_size: usize,
}

impl Default for SecurityPolicy {
    fn default() -> Self {
        Self {
            require_encryption: true,
            require_mac: true,
            pci_level: PciComplianceLevel::Partial,
            allowed_mtis: vec![
                "0100".to_string(), // Authorization request
                "0110".to_string(), // Authorization response
                "0200".to_string(), // Financial request
                "0210".to_string(), // Financial response
                "0400".to_string(), // Reversal request
                "0410".to_string(), // Reversal response
                "0800".to_string(), // Network management request
                "0810".to_string(), // Network management response
            ],
            max_message_size: 8192, // 8KB max message size
        }
    }
}

impl SecurityPolicy {
    /// Validate message against security policy
    pub fn validate_message(&self, message: &Iso8583Message) -> Result<(), SecurityError> {
        // Check MTI
        if !self.allowed_mtis.contains(&message.mti) {
            return Err(SecurityError::MacValidation(
                format!("MTI {} not allowed by policy", message.mti)
            ));
        }
        
        // Additional policy validations can be added here
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pan_masking() {
        let mut key_manager = KeyManager::new();
        let masking_config = MaskingConfig::default();
        let processor = SecurityProcessor::new(key_manager, masking_config);
        
        let pan = "1234567890123456";
        let masked = processor.mask_pan(pan).unwrap();
        assert_eq!(masked, "123456******3456");
    }
    
    #[test]
    fn test_partial_masking() {
        let mut key_manager = KeyManager::new();
        let masking_config = MaskingConfig::default();
        let processor = SecurityProcessor::new(key_manager, masking_config);
        
        let value = "ABCDEFGH";
        let masked = processor.mask_partial(value).unwrap();
        assert_eq!(masked, "AB****GH");
    }
    
    #[test]
    fn test_tokenization() {
        let mut key_manager = KeyManager::new();
        let masking_config = MaskingConfig::default();
        let processor = SecurityProcessor::new(key_manager, masking_config);
        
        let value = "1234567890123456";
        let token = processor.tokenize_value(value, 2).unwrap();
        assert!(token.starts_with("TOK"));
        
        // Same value should produce same token
        let token2 = processor.tokenize_value(value, 2).unwrap();
        assert_eq!(token, token2);
    }
    
    #[test]
    fn test_key_generation() {
        let key1 = KeyManager::generate_key();
        let key2 = KeyManager::generate_key();
        assert_ne!(key1, key2); // Keys should be different
    }
    
    #[test]
    fn test_encryption_decryption() {
        let mut key_manager = KeyManager::new();
        let key = KeyManager::generate_key();
        key_manager.add_data_key("test".to_string(), key);
        
        let masking_config = MaskingConfig::default();
        let processor = SecurityProcessor::new(key_manager, masking_config);
        
        let plaintext = "1234567890123456";
        let encrypted = processor.encrypt_field(plaintext, "test").unwrap();
        let decrypted = processor.decrypt_field(&encrypted, "test").unwrap();
        
        assert_eq!(plaintext, decrypted);
    }
}