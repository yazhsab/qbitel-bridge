use adapter_sdk::{L7Adapter, AdapterError};
use async_trait::async_trait;
use bytes::{Bytes, BytesMut, BufMut};
use std::collections::HashMap;
use tracing::{info, warn, error};

/// HL7 MLLP (Minimum Lower Layer Protocol) Adapter
///
/// This adapter handles HL7 v2.x messages wrapped in MLLP framing.
/// MLLP uses specific start and end bytes to frame HL7 messages:
/// - Start Block: 0x0B (VT - Vertical Tab)
/// - End Block: 0x1C (FS - File Separator)
/// - Carriage Return: 0x0D (CR)
pub struct Hl7MllpAdapter {
    config: Hl7Config,
}

#[derive(Debug, Clone)]
pub struct Hl7Config {
    /// Whether to validate HL7 message structure
    pub validate_structure: bool,
    /// Whether to normalize field separators
    pub normalize_separators: bool,
    /// Custom field mappings for transformation
    pub field_mappings: HashMap<String, String>,
    /// Maximum message size in bytes
    pub max_message_size: usize,
}

impl Default for Hl7Config {
    fn default() -> Self {
        Self {
            validate_structure: true,
            normalize_separators: true,
            field_mappings: HashMap::new(),
            max_message_size: 1024 * 1024, // 1MB
        }
    }
}

impl Hl7MllpAdapter {
    pub fn new(config: Hl7Config) -> Self {
        Self { config }
    }
}

impl Default for Hl7MllpAdapter {
    fn default() -> Self {
        Self::new(Hl7Config::default())
    }
}

#[async_trait]
impl L7Adapter for Hl7MllpAdapter {
    /// Transform client HL7 message to upstream format
    async fn to_upstream(&self, input: Bytes) -> Result<Bytes, AdapterError> {
        info!("processing HL7 message to upstream, size: {} bytes", input.len());
        
        if input.len() > self.config.max_message_size {
            return Err(AdapterError::InvalidInput(
                format!("message size {} exceeds maximum {}", input.len(), self.config.max_message_size)
            ));
        }

        // 1. Extract HL7 message from MLLP framing
        let hl7_message = self.extract_from_mllp(&input)?;
        
        // 2. Parse and validate HL7 message structure
        let mut parsed_message = if self.config.validate_structure {
            self.parse_hl7_message(&hl7_message)?
        } else {
            Hl7Message::from_raw(hl7_message)
        };
        
        // 3. Apply field transformations
        self.apply_field_mappings(&mut parsed_message)?;
        
        // 4. Normalize message format
        if self.config.normalize_separators {
            self.normalize_message(&mut parsed_message)?;
        }
        
        // 5. Re-wrap in MLLP framing for upstream
        let output = self.wrap_in_mllp(&parsed_message.to_bytes())?;
        
        info!("HL7 message processed for upstream, output size: {} bytes", output.len());
        Ok(output)
    }

    /// Transform upstream HL7 response back to client format
    async fn to_client(&self, input: Bytes) -> Result<Bytes, AdapterError> {
        info!("processing HL7 response to client, size: {} bytes", input.len());
        
        if input.len() > self.config.max_message_size {
            return Err(AdapterError::InvalidInput(
                format!("response size {} exceeds maximum {}", input.len(), self.config.max_message_size)
            ));
        }

        // 1. Extract HL7 message from MLLP framing
        let hl7_message = self.extract_from_mllp(&input)?;
        
        // 2. Parse HL7 response
        let mut parsed_response = if self.config.validate_structure {
            self.parse_hl7_message(&hl7_message)?
        } else {
            Hl7Message::from_raw(hl7_message)
        };
        
        // 3. Apply reverse field mappings
        self.apply_reverse_field_mappings(&mut parsed_response)?;
        
        // 4. Re-wrap in MLLP framing for client
        let output = self.wrap_in_mllp(&parsed_response.to_bytes())?;
        
        info!("HL7 response processed for client, output size: {} bytes", output.len());
        Ok(output)
    }

    fn name(&self) -> &'static str {
        "hl7_mllp"
    }
}

impl Hl7MllpAdapter {
    /// Extract HL7 message from MLLP framing
    fn extract_from_mllp(&self, input: &Bytes) -> Result<Bytes, AdapterError> {
        if input.len() < 3 {
            return Err(AdapterError::InvalidInput("MLLP message too short".to_string()));
        }

        // Check for MLLP start block (0x0B)
        if input[0] != 0x0B {
            return Err(AdapterError::InvalidInput("Missing MLLP start block".to_string()));
        }

        // Find end block (0x1C) and carriage return (0x0D)
        let mut end_pos = None;
        for i in 1..input.len()-1 {
            if input[i] == 0x1C && input[i+1] == 0x0D {
                end_pos = Some(i);
                break;
            }
        }

        let end_pos = end_pos.ok_or_else(|| {
            AdapterError::InvalidInput("Missing MLLP end block".to_string())
        })?;

        // Extract HL7 message (between start and end blocks)
        let hl7_data = input.slice(1..end_pos);
        
        info!("extracted HL7 message from MLLP, size: {} bytes", hl7_data.len());
        Ok(hl7_data)
    }

    /// Wrap HL7 message in MLLP framing
    fn wrap_in_mllp(&self, hl7_message: &Bytes) -> Result<Bytes, AdapterError> {
        let mut output = BytesMut::with_capacity(hl7_message.len() + 3);
        
        // Add MLLP start block
        output.put_u8(0x0B);
        
        // Add HL7 message
        output.put(hl7_message.clone());
        
        // Add MLLP end block and carriage return
        output.put_u8(0x1C);
        output.put_u8(0x0D);
        
        Ok(output.freeze())
    }

    /// Parse HL7 message structure
    fn parse_hl7_message(&self, data: &Bytes) -> Result<Hl7Message, AdapterError> {
        let message_str = String::from_utf8_lossy(data);
        
        // Split into segments (typically separated by \r)
        let segments: Vec<&str> = message_str.split('\r').collect();
        
        if segments.is_empty() {
            return Err(AdapterError::InvalidInput("Empty HL7 message".to_string()));
        }

        // Parse MSH (Message Header) segment
        let msh_segment = segments[0];
        if !msh_segment.starts_with("MSH") {
            return Err(AdapterError::InvalidInput("HL7 message must start with MSH segment".to_string()));
        }

        // Extract field separator (usually |)
        let field_separator = if msh_segment.len() > 3 {
            msh_segment.chars().nth(3).unwrap_or('|')
        } else {
            '|'
        };

        let mut parsed_segments = Vec::new();
        
        for segment_str in segments {
            if segment_str.trim().is_empty() {
                continue;
            }
            
            let segment = self.parse_segment(segment_str, field_separator)?;
            parsed_segments.push(segment);
        }

        Ok(Hl7Message {
            segments: parsed_segments,
            field_separator,
        })
    }

    /// Parse individual HL7 segment
    fn parse_segment(&self, segment_str: &str, field_separator: char) -> Result<Hl7Segment, AdapterError> {
        let fields: Vec<&str> = segment_str.split(field_separator).collect();
        
        if fields.is_empty() {
            return Err(AdapterError::InvalidInput("Empty segment".to_string()));
        }

        let segment_type = fields[0].to_string();
        let segment_fields: Vec<String> = fields.iter().skip(1).map(|f| f.to_string()).collect();

        Ok(Hl7Segment {
            segment_type,
            fields: segment_fields,
        })
    }

    /// Apply field mappings for transformation
    fn apply_field_mappings(&self, message: &mut Hl7Message) -> Result<(), AdapterError> {
        if self.config.field_mappings.is_empty() {
            return Ok(());
        }

        info!("applying {} field mappings", self.config.field_mappings.len());

        for segment in &mut message.segments {
            for (from_path, to_value) in &self.config.field_mappings {
                if let Some(field_index) = self.parse_field_path(&segment.segment_type, from_path) {
                    if field_index < segment.fields.len() {
                        segment.fields[field_index] = to_value.clone();
                        info!("mapped field {} to {}", from_path, to_value);
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply reverse field mappings
    fn apply_reverse_field_mappings(&self, message: &mut Hl7Message) -> Result<(), AdapterError> {
        if self.config.field_mappings.is_empty() {
            return Ok(());
        }

        info!("applying reverse field mappings");

        // Create reverse mapping
        let reverse_mappings: HashMap<String, String> = self.config.field_mappings
            .iter()
            .map(|(k, v)| (v.clone(), k.clone()))
            .collect();

        for segment in &mut message.segments {
            for (from_value, to_path) in &reverse_mappings {
                if let Some(field_index) = self.parse_field_path(&segment.segment_type, to_path) {
                    if field_index < segment.fields.len() && segment.fields[field_index] == *from_value {
                        // In a real implementation, this would restore the original value
                        info!("reverse mapped field {} from {}", to_path, from_value);
                    }
                }
            }
        }

        Ok(())
    }

    /// Parse field path (e.g., "MSH.3" -> segment MSH, field 3)
    fn parse_field_path(&self, segment_type: &str, path: &str) -> Option<usize> {
        let parts: Vec<&str> = path.split('.').collect();
        if parts.len() == 2 && parts[0] == segment_type {
            parts[1].parse::<usize>().ok().map(|i| i.saturating_sub(1)) // Convert to 0-based index
        } else {
            None
        }
    }

    /// Normalize message format
    fn normalize_message(&self, message: &mut Hl7Message) -> Result<(), AdapterError> {
        info!("normalizing HL7 message format");
        
        // Ensure consistent field separator
        message.field_separator = '|';
        
        // Normalize segment formatting
        for segment in &mut message.segments {
            // Trim whitespace from fields
            for field in &mut segment.fields {
                *field = field.trim().to_string();
            }
        }

        Ok(())
    }
}

/// Represents a parsed HL7 message
#[derive(Debug, Clone)]
pub struct Hl7Message {
    pub segments: Vec<Hl7Segment>,
    pub field_separator: char,
}

impl Hl7Message {
    pub fn from_raw(data: Bytes) -> Self {
        Self {
            segments: vec![Hl7Segment {
                segment_type: "RAW".to_string(),
                fields: vec![String::from_utf8_lossy(&data).to_string()],
            }],
            field_separator: '|',
        }
    }

    pub fn to_bytes(&self) -> Bytes {
        let mut output = String::new();
        
        for (i, segment) in self.segments.iter().enumerate() {
            if i > 0 {
                output.push('\r');
            }
            
            output.push_str(&segment.segment_type);
            
            for field in &segment.fields {
                output.push(self.field_separator);
                output.push_str(field);
            }
        }
        
        Bytes::from(output)
    }
}

/// Represents an HL7 segment
#[derive(Debug, Clone)]
pub struct Hl7Segment {
    pub segment_type: String,
    pub fields: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mllp_framing() {
        let adapter = Hl7MllpAdapter::default();
        let hl7_msg = Bytes::from("MSH|^~\\&|SYSTEM|SENDER|RECEIVER|DEST|20230101120000||ADT^A01|12345|P|2.5");
        
        // Test wrapping
        let wrapped = adapter.wrap_in_mllp(&hl7_msg).unwrap();
        assert_eq!(wrapped[0], 0x0B); // Start block
        assert_eq!(wrapped[wrapped.len()-2], 0x1C); // End block
        assert_eq!(wrapped[wrapped.len()-1], 0x0D); // Carriage return
        
        // Test extraction
        let extracted = adapter.extract_from_mllp(&wrapped).unwrap();
        assert_eq!(extracted, hl7_msg);
    }

    #[tokio::test]
    async fn test_hl7_parsing() {
        let adapter = Hl7MllpAdapter::default();
        let hl7_data = Bytes::from("MSH|^~\\&|SYSTEM|SENDER\rPID|1||12345^^^MRN||DOE^JOHN");
        
        let parsed = adapter.parse_hl7_message(&hl7_data).unwrap();
        assert_eq!(parsed.segments.len(), 2);
        assert_eq!(parsed.segments[0].segment_type, "MSH");
        assert_eq!(parsed.segments[1].segment_type, "PID");
    }

    #[tokio::test]
    async fn test_field_mappings() {
        let mut config = Hl7Config::default();
        config.field_mappings.insert("MSH.3".to_string(), "MAPPED_SYSTEM".to_string());
        
        let adapter = Hl7MllpAdapter::new(config);
        let input = Bytes::from("\x0BMSH|^~\\&|SYSTEM|SENDER\x1C\x0D");
        
        let result = adapter.to_upstream(input).await.unwrap();
        let extracted = adapter.extract_from_mllp(&result).unwrap();
        let result_str = String::from_utf8_lossy(&extracted);
        
        assert!(result_str.contains("MAPPED_SYSTEM"));
    }
}
