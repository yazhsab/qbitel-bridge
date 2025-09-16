//! Protocol Analysis and Parsing
//!
//! This module provides protocol-specific analysis and parsing capabilities
//! for various network protocols and applications.

use std::collections::HashMap;
use std::sync::Arc;

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, error, info, warn, instrument};

use crate::{DpiError, PacketData, ProtocolType, ApplicationType};
use crate::features::PacketFeatures;

/// Protocol analysis errors
#[derive(Error, Debug)]
pub enum ProtocolError {
    #[error("Protocol parsing failed: {0}")]
    ParsingFailed(String),
    
    #[error("Unsupported protocol: {0}")]
    UnsupportedProtocol(String),
    
    #[error("Invalid protocol data: {0}")]
    InvalidData(String),
    
    #[error("Protocol signature not found: {0}")]
    SignatureNotFound(String),
}

type Result<T> = std::result::Result<T, ProtocolError>;

/// Protocol signature for identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolSignature {
    pub protocol: ProtocolType,
    pub application: ApplicationType,
    pub version: Option<String>,
    pub confidence: f32,
    pub signature_type: SignatureType,
    pub metadata: HashMap<String, String>,
    pub extracted_fields: HashMap<String, String>,
}

/// Types of protocol signatures
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignatureType {
    Header,
    Handshake,
    Content,
    Behavioral,
    Statistical,
}

/// Protocol analyzer trait
pub trait ProtocolAnalyzer: Send + Sync {
    fn can_analyze(&self, packet_data: &PacketData, features: &PacketFeatures) -> bool;
    fn analyze(&self, packet_data: &PacketData, features: &PacketFeatures) -> Result<ProtocolSignature>;
    fn get_protocol_type(&self) -> ProtocolType;
    fn get_supported_applications(&self) -> Vec<ApplicationType>;
}

/// HTTP protocol analyzer
pub struct HttpAnalyzer;

impl ProtocolAnalyzer for HttpAnalyzer {
    fn can_analyze(&self, packet_data: &PacketData, features: &PacketFeatures) -> bool {
        packet_data.dst_port == 80 || packet_data.src_port == 80 || features.has_http_headers
    }
    
    fn analyze(&self, packet_data: &PacketData, features: &PacketFeatures) -> Result<ProtocolSignature> {
        if packet_data.payload_offset >= packet_data.data.len() {
            return Err(ProtocolError::InvalidData("No payload data".to_string()));
        }
        
        let payload = &packet_data.data[packet_data.payload_offset..];
        let payload_str = String::from_utf8_lossy(payload);
        
        let mut metadata = HashMap::new();
        let mut extracted_fields = HashMap::new();
        
        // Extract HTTP method
        if let Some(method) = self.extract_http_method(&payload_str) {
            extracted_fields.insert("method".to_string(), method.clone());
            metadata.insert("http_method".to_string(), method);
        }
        
        // Extract User-Agent
        if let Some(user_agent) = self.extract_user_agent(&payload_str) {
            extracted_fields.insert("user_agent".to_string(), user_agent.clone());
            metadata.insert("user_agent".to_string(), user_agent);
        }
        
        // Extract Host header
        if let Some(host) = self.extract_host_header(&payload_str) {
            extracted_fields.insert("host".to_string(), host.clone());
            metadata.insert("host".to_string(), host);
        }
        
        let confidence = if features.has_http_headers { 0.95 } else { 0.7 };
        
        Ok(ProtocolSignature {
            protocol: ProtocolType::HTTP,
            application: self.determine_application(&extracted_fields),
            version: self.extract_http_version(&payload_str),
            confidence,
            signature_type: SignatureType::Header,
            metadata,
            extracted_fields,
        })
    }
    
    fn get_protocol_type(&self) -> ProtocolType {
        ProtocolType::HTTP
    }
    
    fn get_supported_applications(&self) -> Vec<ApplicationType> {
        vec![ApplicationType::WebBrowsing]
    }
}

impl HttpAnalyzer {
    fn extract_http_method(&self, payload: &str) -> Option<String> {
        if payload.starts_with("GET ") {
            Some("GET".to_string())
        } else if payload.starts_with("POST ") {
            Some("POST".to_string())
        } else if payload.starts_with("PUT ") {
            Some("PUT".to_string())
        } else if payload.starts_with("DELETE ") {
            Some("DELETE".to_string())
        } else {
            None
        }
    }
    
    fn extract_user_agent(&self, payload: &str) -> Option<String> {
        self.extract_header_value(payload, "User-Agent")
    }
    
    fn extract_host_header(&self, payload: &str) -> Option<String> {
        self.extract_header_value(payload, "Host")
    }
    
    fn extract_header_value(&self, payload: &str, header_name: &str) -> Option<String> {
        let header_pattern = format!("{}:", header_name);
        if let Some(start) = payload.find(&header_pattern) {
            let line_start = start + header_pattern.len();
            if let Some(line_end) = payload[line_start..].find('\n') {
                let value = payload[line_start..line_start + line_end].trim();
                return Some(value.to_string());
            }
        }
        None
    }
    
    fn extract_http_version(&self, payload: &str) -> Option<String> {
        if payload.contains("HTTP/1.1") {
            Some("1.1".to_string())
        } else if payload.contains("HTTP/1.0") {
            Some("1.0".to_string())
        } else if payload.contains("HTTP/2") {
            Some("2.0".to_string())
        } else {
            None
        }
    }
    
    fn determine_application(&self, _fields: &HashMap<String, String>) -> ApplicationType {
        ApplicationType::WebBrowsing
    }
}

/// HTTPS/TLS analyzer
pub struct HttpsAnalyzer;

impl ProtocolAnalyzer for HttpsAnalyzer {
    fn can_analyze(&self, packet_data: &PacketData, features: &PacketFeatures) -> bool {
        packet_data.dst_port == 443 || packet_data.src_port == 443 || features.has_tls_handshake
    }
    
    fn analyze(&self, packet_data: &PacketData, features: &PacketFeatures) -> Result<ProtocolSignature> {
        if packet_data.payload_offset >= packet_data.data.len() {
            return Err(ProtocolError::InvalidData("No payload data".to_string()));
        }
        
        let payload = &packet_data.data[packet_data.payload_offset..];
        let mut metadata = HashMap::new();
        let mut extracted_fields = HashMap::new();
        
        // Extract TLS version
        if let Some(version) = self.extract_tls_version(payload) {
            extracted_fields.insert("tls_version".to_string(), version.clone());
            metadata.insert("tls_version".to_string(), version);
        }
        
        // Extract cipher suite if available
        if let Some(cipher) = self.extract_cipher_suite(payload) {
            extracted_fields.insert("cipher_suite".to_string(), cipher.clone());
            metadata.insert("cipher_suite".to_string(), cipher);
        }
        
        let confidence = if features.has_tls_handshake { 0.95 } else { 0.8 };
        
        Ok(ProtocolSignature {
            protocol: ProtocolType::HTTPS,
            application: ApplicationType::WebBrowsing,
            version: extracted_fields.get("tls_version").cloned(),
            confidence,
            signature_type: SignatureType::Handshake,
            metadata,
            extracted_fields,
        })
    }
    
    fn get_protocol_type(&self) -> ProtocolType {
        ProtocolType::HTTPS
    }
    
    fn get_supported_applications(&self) -> Vec<ApplicationType> {
        vec![ApplicationType::WebBrowsing]
    }
}

impl HttpsAnalyzer {
    fn extract_tls_version(&self, payload: &[u8]) -> Option<String> {
        if payload.len() < 3 {
            return None;
        }
        
        // Check TLS record header
        if payload[0] == 0x16 && payload[1] == 0x03 {
            match payload[2] {
                0x01 => Some("1.0".to_string()),
                0x02 => Some("1.1".to_string()),
                0x03 => Some("1.2".to_string()),
                0x04 => Some("1.3".to_string()),
                _ => None,
            }
        } else {
            None
        }
    }
    
    fn extract_cipher_suite(&self, _payload: &[u8]) -> Option<String> {
        // Simplified - would need full TLS parsing
        None
    }
}

/// DNS analyzer
pub struct DnsAnalyzer;

impl ProtocolAnalyzer for DnsAnalyzer {
    fn can_analyze(&self, packet_data: &PacketData, features: &PacketFeatures) -> bool {
        packet_data.dst_port == 53 || packet_data.src_port == 53 || features.has_dns_query
    }
    
    fn analyze(&self, packet_data: &PacketData, _features: &PacketFeatures) -> Result<ProtocolSignature> {
        if packet_data.payload_offset >= packet_data.data.len() {
            return Err(ProtocolError::InvalidData("No payload data".to_string()));
        }
        
        let payload = &packet_data.data[packet_data.payload_offset..];
        let mut metadata = HashMap::new();
        let mut extracted_fields = HashMap::new();
        
        if payload.len() >= 12 {
            // Parse DNS header
            let transaction_id = u16::from_be_bytes([payload[0], payload[1]]);
            let flags = u16::from_be_bytes([payload[2], payload[3]]);
            let is_response = (flags & 0x8000) != 0;
            let opcode = (flags >> 11) & 0x0F;
            let rcode = flags & 0x0F;
            
            extracted_fields.insert("transaction_id".to_string(), transaction_id.to_string());
            extracted_fields.insert("is_response".to_string(), is_response.to_string());
            extracted_fields.insert("opcode".to_string(), opcode.to_string());
            extracted_fields.insert("rcode".to_string(), rcode.to_string());
            
            metadata.insert("dns_type".to_string(), 
                          if is_response { "response".to_string() } else { "query".to_string() });
        }
        
        Ok(ProtocolSignature {
            protocol: ProtocolType::DNS,
            application: ApplicationType::Unknown,
            version: None,
            confidence: 0.9,
            signature_type: SignatureType::Header,
            metadata,
            extracted_fields,
        })
    }
    
    fn get_protocol_type(&self) -> ProtocolType {
        ProtocolType::DNS
    }
    
    fn get_supported_applications(&self) -> Vec<ApplicationType> {
        vec![ApplicationType::Unknown]
    }
}

/// SSH analyzer
pub struct SshAnalyzer;

impl ProtocolAnalyzer for SshAnalyzer {
    fn can_analyze(&self, packet_data: &PacketData, _features: &PacketFeatures) -> bool {
        packet_data.dst_port == 22 || packet_data.src_port == 22
    }
    
    fn analyze(&self, packet_data: &PacketData, _features: &PacketFeatures) -> Result<ProtocolSignature> {
        if packet_data.payload_offset >= packet_data.data.len() {
            return Err(ProtocolError::InvalidData("No payload data".to_string()));
        }
        
        let payload = &packet_data.data[packet_data.payload_offset..];
        let payload_str = String::from_utf8_lossy(payload);
        
        let mut metadata = HashMap::new();
        let mut extracted_fields = HashMap::new();
        
        // Look for SSH version string
        if payload_str.starts_with("SSH-") {
            if let Some(version_end) = payload_str.find('\n') {
                let version_line = &payload_str[..version_end];
                extracted_fields.insert("version_string".to_string(), version_line.to_string());
                
                // Extract SSH version
                if let Some(version) = self.extract_ssh_version(version_line) {
                    extracted_fields.insert("ssh_version".to_string(), version.clone());
                    metadata.insert("ssh_version".to_string(), version);
                }
            }
        }
        
        let confidence = if extracted_fields.contains_key("version_string") { 0.95 } else { 0.7 };
        
        Ok(ProtocolSignature {
            protocol: ProtocolType::SSH,
            application: ApplicationType::RemoteAccess,
            version: extracted_fields.get("ssh_version").cloned(),
            confidence,
            signature_type: SignatureType::Handshake,
            metadata,
            extracted_fields,
        })
    }
    
    fn get_protocol_type(&self) -> ProtocolType {
        ProtocolType::SSH
    }
    
    fn get_supported_applications(&self) -> Vec<ApplicationType> {
        vec![ApplicationType::RemoteAccess]
    }
}

impl SshAnalyzer {
    fn extract_ssh_version(&self, version_string: &str) -> Option<String> {
        if version_string.starts_with("SSH-2.0") {
            Some("2.0".to_string())
        } else if version_string.starts_with("SSH-1.99") {
            Some("1.99".to_string())
        } else if version_string.starts_with("SSH-1.5") {
            Some("1.5".to_string())
        } else {
            None
        }
    }
}

/// Protocol parser registry
pub struct ProtocolParser {
    analyzers: HashMap<ProtocolType, Box<dyn ProtocolAnalyzer + Send + Sync>>,
}

impl ProtocolParser {
    /// Create a new protocol parser with default analyzers
    pub fn new() -> Self {
        let mut analyzers: HashMap<ProtocolType, Box<dyn ProtocolAnalyzer + Send + Sync>> = HashMap::new();
        
        analyzers.insert(ProtocolType::HTTP, Box::new(HttpAnalyzer));
        analyzers.insert(ProtocolType::HTTPS, Box::new(HttpsAnalyzer));
        analyzers.insert(ProtocolType::DNS, Box::new(DnsAnalyzer));
        analyzers.insert(ProtocolType::SSH, Box::new(SshAnalyzer));
        
        Self { analyzers }
    }
    
    /// Parse packet and extract protocol signature
    #[instrument(skip(self, packet_data, features))]
    pub fn parse_protocol(&self, packet_data: &PacketData, features: &PacketFeatures) -> Option<ProtocolSignature> {
        // Try each analyzer to see which one can handle this packet
        for analyzer in self.analyzers.values() {
            if analyzer.can_analyze(packet_data, features) {
                match analyzer.analyze(packet_data, features) {
                    Ok(signature) => {
                        debug!("Protocol {} analyzed with confidence {:.2}", 
                               format!("{:?}", signature.protocol), signature.confidence);
                        return Some(signature);
                    },
                    Err(e) => {
                        warn!("Protocol analysis failed: {}", e);
                    }
                }
            }
        }
        
        None
    }
    
    /// Add a custom protocol analyzer
    pub fn add_analyzer(&mut self, protocol: ProtocolType, analyzer: Box<dyn ProtocolAnalyzer + Send + Sync>) {
        self.analyzers.insert(protocol, analyzer);
        info!("Added custom analyzer for protocol {:?}", protocol);
    }
    
    /// Get supported protocols
    pub fn get_supported_protocols(&self) -> Vec<ProtocolType> {
        self.analyzers.keys().cloned().collect()
    }
    
    /// Check if protocol is supported
    pub fn is_protocol_supported(&self, protocol: &ProtocolType) -> bool {
        self.analyzers.contains_key(protocol)
    }
}

impl Default for ProtocolParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};
    use bytes::Bytes;
    use std::time::Instant;
    
    #[test]
    fn test_http_analyzer() {
        let analyzer = HttpAnalyzer;
        
        let packet_data = PacketData {
            packet_id: 1,
            timestamp: Instant::now(),
            data: Bytes::from("GET /index.html HTTP/1.1\r\nHost: example.com\r\nUser-Agent: Mozilla/5.0\r\n\r\n"),
            flow_id: None,
            src_ip: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100)),
            dst_ip: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
            src_port: 12345,
            dst_port: 80,
            protocol: 6,
            payload_offset: 0,
            metadata: HashMap::new(),
        };
        
        let features = PacketFeatures {
            has_http_headers: true,
            dst_port: 80,
            ..Default::default()
        };
        
        assert!(analyzer.can_analyze(&packet_data, &features));
        
        let result = analyzer.analyze(&packet_data, &features).unwrap();
        assert_eq!(result.protocol, ProtocolType::HTTP);
        assert!(result.confidence > 0.8);
        assert!(result.extracted_fields.contains_key("method"));
        assert_eq!(result.extracted_fields.get("method").unwrap(), "GET");
    }
    
    #[test]
    fn test_https_analyzer() {
        let analyzer = HttpsAnalyzer;
        
        // TLS handshake packet (simplified)
        let tls_data = vec![0x16, 0x03, 0x03, 0x00, 0x30]; // TLS 1.2 handshake
        let packet_data = PacketData {
            packet_id: 1,
            timestamp: Instant::now(),
            data: Bytes::from(tls_data),
            flow_id: None,
            src_ip: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100)),
            dst_ip: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
            src_port: 12345,
            dst_port: 443,
            protocol: 6,
            payload_offset: 0,
            metadata: HashMap::new(),
        };
        
        let features = PacketFeatures {
            has_tls_handshake: true,
            dst_port: 443,
            ..Default::default()
        };
        
        assert!(analyzer.can_analyze(&packet_data, &features));
        
        let result = analyzer.analyze(&packet_data, &features).unwrap();
        assert_eq!(result.protocol, ProtocolType::HTTPS);
        assert!(result.confidence > 0.8);
    }
    
    #[test]
    fn test_dns_analyzer() {
        let analyzer = DnsAnalyzer;
        
        // Simple DNS query header
        let dns_data = vec![
            0x12, 0x34, // Transaction ID
            0x01, 0x00, // Flags: standard query
            0x00, 0x01, // Questions: 1
            0x00, 0x00, // Answer RRs: 0
            0x00, 0x00, // Authority RRs: 0
            0x00, 0x00, // Additional RRs: 0
        ];
        
        let packet_data = PacketData {
            packet_id: 1,
            timestamp: Instant::now(),
            data: Bytes::from(dns_data),
            flow_id: None,
            src_ip: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100)),
            dst_ip: IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8)),
            src_port: 12345,
            dst_port: 53,
            protocol: 17,
            payload_offset: 0,
            metadata: HashMap::new(),
        };
        
        let features = PacketFeatures {
            has_dns_query: true,
            dst_port: 53,
            ..Default::default()
        };
        
        assert!(analyzer.can_analyze(&packet_data, &features));
        
        let result = analyzer.analyze(&packet_data, &features).unwrap();
        assert_eq!(result.protocol, ProtocolType::DNS);
        assert!(result.confidence > 0.8);
        assert!(result.extracted_fields.contains_key("transaction_id"));
        assert_eq!(result.extracted_fields.get("transaction_id").unwrap(), "4660"); // 0x1234
    }
    
    #[test]
    fn test_protocol_parser() {
        let parser = ProtocolParser::new();
        
        assert!(parser.is_protocol_supported(&ProtocolType::HTTP));
        assert!(parser.is_protocol_supported(&ProtocolType::HTTPS));
        assert!(parser.is_protocol_supported(&ProtocolType::DNS));
        assert!(parser.is_protocol_supported(&ProtocolType::SSH));
        
        let supported_protocols = parser.get_supported_protocols();
        assert!(supported_protocols.len() >= 4);
    }
}