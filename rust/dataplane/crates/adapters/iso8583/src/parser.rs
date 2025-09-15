use bytes::{Bytes, BytesMut, Buf, BufMut};
use std::collections::HashMap;
use thiserror::Error;
use tracing::{debug, warn, error};

#[derive(Debug, Error)]
pub enum Iso8583Error {
    #[error("invalid message format: {0}")]
    InvalidFormat(String),
    #[error("unsupported encoding: {0}")]
    UnsupportedEncoding(String),
    #[error("field parsing error: {0}")]
    FieldError(String),
    #[error("bitmap parsing error: {0}")]
    BitmapError(String),
    #[error("insufficient data: expected {expected}, got {actual}")]
    InsufficientData { expected: usize, actual: usize },
}

/// ISO-8583 message encoding types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MessageEncoding {
    Ascii,
    Ebcdic,
    Binary,
}

/// ISO-8583 field data types
#[derive(Debug, Clone, PartialEq)]
pub enum FieldType {
    /// Fixed length numeric
    Numeric(usize),
    /// Fixed length alphanumeric
    Alpha(usize),
    /// Variable length with 2-digit length prefix
    Llvar,
    /// Variable length with 3-digit length prefix
    Lllvar,
    /// Binary data
    Binary(usize),
}

/// ISO-8583 field definition
#[derive(Debug, Clone)]
pub struct FieldDefinition {
    pub field_type: FieldType,
    pub description: String,
    pub sensitive: bool, // For PCI compliance masking
}

/// ISO-8583 message structure
#[derive(Debug, Clone)]
pub struct Iso8583Message {
    pub mti: String,
    pub bitmap: Vec<bool>,
    pub fields: HashMap<u8, String>,
    pub encoding: MessageEncoding,
}

/// ISO-8583 message parser
pub struct Iso8583Parser {
    field_definitions: HashMap<u8, FieldDefinition>,
    encoding: MessageEncoding,
}

impl Iso8583Parser {
    /// Create a new parser with standard field definitions
    pub fn new(encoding: MessageEncoding) -> Self {
        let mut parser = Self {
            field_definitions: HashMap::new(),
            encoding,
        };
        parser.init_standard_fields();
        parser
    }
    
    /// Initialize standard ISO-8583 field definitions
    fn init_standard_fields(&mut self) {
        // Primary fields (1-64)
        self.add_field(1, FieldType::Binary(8), "Secondary bitmap", false);
        self.add_field(2, FieldType::Llvar, "Primary account number (PAN)", true);
        self.add_field(3, FieldType::Numeric(6), "Processing code", false);
        self.add_field(4, FieldType::Numeric(12), "Amount, transaction", false);
        self.add_field(5, FieldType::Numeric(12), "Amount, settlement", false);
        self.add_field(6, FieldType::Numeric(12), "Amount, cardholder billing", false);
        self.add_field(7, FieldType::Numeric(10), "Transmission date & time", false);
        self.add_field(8, FieldType::Numeric(8), "Amount, cardholder billing fee", false);
        self.add_field(9, FieldType::Numeric(8), "Conversion rate, settlement", false);
        self.add_field(10, FieldType::Numeric(8), "Conversion rate, cardholder billing", false);
        self.add_field(11, FieldType::Numeric(6), "System trace audit number", false);
        self.add_field(12, FieldType::Numeric(6), "Time, local transaction", false);
        self.add_field(13, FieldType::Numeric(4), "Date, local transaction", false);
        self.add_field(14, FieldType::Numeric(4), "Date, expiration", true);
        self.add_field(15, FieldType::Numeric(4), "Date, settlement", false);
        self.add_field(16, FieldType::Numeric(4), "Date, conversion", false);
        self.add_field(17, FieldType::Numeric(4), "Date, capture", false);
        self.add_field(18, FieldType::Numeric(4), "Merchant type", false);
        self.add_field(19, FieldType::Numeric(3), "Acquiring institution country code", false);
        self.add_field(20, FieldType::Numeric(3), "PAN extended, country code", false);
        self.add_field(21, FieldType::Numeric(3), "Forwarding institution country code", false);
        self.add_field(22, FieldType::Numeric(3), "Point of service entry mode", false);
        self.add_field(23, FieldType::Numeric(3), "Application PAN sequence number", false);
        self.add_field(24, FieldType::Numeric(3), "Network International identifier", false);
        self.add_field(25, FieldType::Numeric(2), "Point of service condition code", false);
        self.add_field(26, FieldType::Numeric(2), "Point of service capture code", false);
        self.add_field(27, FieldType::Numeric(1), "Authorization identification response length", false);
        self.add_field(28, FieldType::Alpha(9), "Amount, transaction fee", false);
        self.add_field(29, FieldType::Alpha(9), "Amount, settlement fee", false);
        self.add_field(30, FieldType::Alpha(9), "Amount, transaction processing fee", false);
        self.add_field(31, FieldType::Alpha(9), "Amount, settlement processing fee", false);
        self.add_field(32, FieldType::Llvar, "Acquiring institution identification code", false);
        self.add_field(33, FieldType::Llvar, "Forwarding institution identification code", false);
        self.add_field(34, FieldType::Llvar, "Primary account number, extended", true);
        self.add_field(35, FieldType::Llvar, "Track 2 data", true);
        self.add_field(36, FieldType::Lllvar, "Track 3 data", true);
        self.add_field(37, FieldType::Alpha(12), "Retrieval reference number", false);
        self.add_field(38, FieldType::Alpha(6), "Authorization identification response", false);
        self.add_field(39, FieldType::Alpha(2), "Response code", false);
        self.add_field(40, FieldType::Alpha(3), "Service restriction code", false);
        self.add_field(41, FieldType::Alpha(8), "Card acceptor terminal identification", false);
        self.add_field(42, FieldType::Alpha(15), "Card acceptor identification code", false);
        self.add_field(43, FieldType::Alpha(40), "Card acceptor name/location", false);
        self.add_field(44, FieldType::Llvar, "Additional response data", false);
        self.add_field(45, FieldType::Llvar, "Track 1 data", true);
        self.add_field(46, FieldType::Lllvar, "Additional data - ISO", false);
        self.add_field(47, FieldType::Lllvar, "Additional data - national", false);
        self.add_field(48, FieldType::Lllvar, "Additional data - private", false);
        self.add_field(49, FieldType::Alpha(3), "Currency code, transaction", false);
        self.add_field(50, FieldType::Alpha(3), "Currency code, settlement", false);
        self.add_field(51, FieldType::Alpha(3), "Currency code, cardholder billing", false);
        self.add_field(52, FieldType::Binary(8), "Personal identification number data", true);
        self.add_field(53, FieldType::Numeric(16), "Security related control information", true);
        self.add_field(54, FieldType::Lllvar, "Additional amounts", false);
        self.add_field(55, FieldType::Lllvar, "Reserved ISO", false);
        self.add_field(56, FieldType::Lllvar, "Reserved ISO", false);
        self.add_field(57, FieldType::Lllvar, "Reserved national", false);
        self.add_field(58, FieldType::Lllvar, "Reserved national", false);
        self.add_field(59, FieldType::Lllvar, "Reserved national", false);
        self.add_field(60, FieldType::Lllvar, "Reserved private", false);
        self.add_field(61, FieldType::Lllvar, "Reserved private", false);
        self.add_field(62, FieldType::Lllvar, "Reserved private", false);
        self.add_field(63, FieldType::Lllvar, "Reserved private", false);
        self.add_field(64, FieldType::Binary(8), "Message authentication code", true);
        
        // Secondary bitmap fields (65-128) - commonly used ones
        self.add_field(65, FieldType::Binary(1), "Bitmap, extended", false);
        self.add_field(66, FieldType::Numeric(1), "Settlement code", false);
        self.add_field(67, FieldType::Numeric(2), "Extended payment code", false);
        self.add_field(68, FieldType::Numeric(3), "Receiving institution country code", false);
        self.add_field(69, FieldType::Numeric(3), "Settlement institution country code", false);
        self.add_field(70, FieldType::Numeric(3), "Network management information code", false);
        self.add_field(90, FieldType::Numeric(42), "Original data elements", false);
        self.add_field(95, FieldType::Alpha(42), "Replacement amounts", false);
        self.add_field(96, FieldType::Binary(8), "Message security code", true);
        self.add_field(97, FieldType::Alpha(17), "Amount, net settlement", false);
        self.add_field(98, FieldType::Alpha(25), "Payee", false);
        self.add_field(100, FieldType::Llvar, "Receiving institution identification code", false);
        self.add_field(101, FieldType::Llvar, "File name", false);
        self.add_field(102, FieldType::Llvar, "Account identification 1", true);
        self.add_field(103, FieldType::Llvar, "Account identification 2", true);
        self.add_field(120, FieldType::Lllvar, "Reserved private", false);
        self.add_field(121, FieldType::Lllvar, "Reserved private", false);
        self.add_field(122, FieldType::Lllvar, "Reserved private", false);
        self.add_field(123, FieldType::Lllvar, "Reserved private", false);
        self.add_field(124, FieldType::Lllvar, "Reserved private", false);
        self.add_field(125, FieldType::Lllvar, "Reserved private", false);
        self.add_field(126, FieldType::Lllvar, "Reserved private", false);
        self.add_field(127, FieldType::Lllvar, "Reserved private", false);
        self.add_field(128, FieldType::Binary(8), "Message authentication code", true);
    }
    
    /// Add a field definition
    fn add_field(&mut self, field_num: u8, field_type: FieldType, description: &str, sensitive: bool) {
        self.field_definitions.insert(field_num, FieldDefinition {
            field_type,
            description: description.to_string(),
            sensitive,
        });
    }
    
    /// Parse an ISO-8583 message from bytes
    pub fn parse(&self, data: &[u8]) -> Result<Iso8583Message, Iso8583Error> {
        if data.len() < 4 {
            return Err(Iso8583Error::InsufficientData {
                expected: 4,
                actual: data.len(),
            });
        }
        
        let mut cursor = 0;
        
        // Parse MTI (Message Type Indicator) - first 4 bytes
        let mti = self.parse_mti(&data[cursor..cursor + 4])?;
        cursor += 4;
        
        debug!(mti = %mti, "parsed MTI");
        
        // Parse primary bitmap - next 8 bytes
        if data.len() < cursor + 8 {
            return Err(Iso8583Error::InsufficientData {
                expected: cursor + 8,
                actual: data.len(),
            });
        }
        
        let primary_bitmap = &data[cursor..cursor + 8];
        cursor += 8;
        
        let mut bitmap = self.parse_bitmap(primary_bitmap)?;
        
        // Check if secondary bitmap is present (field 1)
        let has_secondary_bitmap = bitmap[0]; // Field 1
        if has_secondary_bitmap {
            if data.len() < cursor + 8 {
                return Err(Iso8583Error::InsufficientData {
                    expected: cursor + 8,
                    actual: data.len(),
                });
            }
            
            let secondary_bitmap = &data[cursor..cursor + 8];
            cursor += 8;
            
            let secondary_bits = self.parse_bitmap(secondary_bitmap)?;
            bitmap.extend(secondary_bits);
        }
        
        debug!(bitmap_len = bitmap.len(), "parsed bitmap");
        
        // Parse fields based on bitmap
        let mut fields = HashMap::new();
        
        for (field_num, &present) in bitmap.iter().enumerate() {
            if !present || field_num == 0 { // Skip field 0 (not used) and absent fields
                continue;
            }
            
            let field_num = (field_num + 1) as u8; // Convert to 1-based indexing
            
            if field_num == 1 && has_secondary_bitmap {
                // Field 1 is the secondary bitmap, already processed
                continue;
            }
            
            if let Some(field_def) = self.field_definitions.get(&field_num) {
                let (field_value, consumed) = self.parse_field(&data[cursor..], &field_def.field_type)?;
                fields.insert(field_num, field_value);
                cursor += consumed;
                
                debug!(field = field_num, value_len = consumed, "parsed field");
            } else {
                warn!(field = field_num, "unknown field in bitmap");
            }
        }
        
        Ok(Iso8583Message {
            mti,
            bitmap,
            fields,
            encoding: self.encoding,
        })
    }
    
    /// Parse MTI based on encoding
    fn parse_mti(&self, data: &[u8]) -> Result<String, Iso8583Error> {
        match self.encoding {
            MessageEncoding::Ascii => {
                String::from_utf8(data.to_vec())
                    .map_err(|e| Iso8583Error::InvalidFormat(format!("invalid ASCII MTI: {}", e)))
            }
            MessageEncoding::Ebcdic => {
                // Convert EBCDIC to ASCII
                let ascii_data: Vec<u8> = data.iter().map(|&b| self.ebcdic_to_ascii(b)).collect();
                String::from_utf8(ascii_data)
                    .map_err(|e| Iso8583Error::InvalidFormat(format!("invalid EBCDIC MTI: {}", e)))
            }
            MessageEncoding::Binary => {
                // Binary MTI is typically BCD encoded
                let mut mti = String::new();
                for &byte in data {
                    mti.push_str(&format!("{:02x}", byte));
                }
                Ok(mti)
            }
        }
    }
    
    /// Parse bitmap from 8 bytes
    fn parse_bitmap(&self, data: &[u8]) -> Result<Vec<bool>, Iso8583Error> {
        if data.len() != 8 {
            return Err(Iso8583Error::BitmapError(format!(
                "bitmap must be 8 bytes, got {}", data.len()
            )));
        }
        
        let mut bitmap = Vec::with_capacity(64);
        
        for &byte in data {
            for bit in (0..8).rev() {
                bitmap.push((byte >> bit) & 1 == 1);
            }
        }
        
        Ok(bitmap)
    }
    
    /// Parse a single field based on its type
    fn parse_field(&self, data: &[u8], field_type: &FieldType) -> Result<(String, usize), Iso8583Error> {
        match field_type {
            FieldType::Numeric(len) => {
                if data.len() < *len {
                    return Err(Iso8583Error::InsufficientData {
                        expected: *len,
                        actual: data.len(),
                    });
                }
                let value = self.decode_string(&data[..*len])?;
                Ok((value, *len))
            }
            FieldType::Alpha(len) => {
                if data.len() < *len {
                    return Err(Iso8583Error::InsufficientData {
                        expected: *len,
                        actual: data.len(),
                    });
                }
                let value = self.decode_string(&data[..*len])?;
                Ok((value, *len))
            }
            FieldType::Llvar => {
                if data.len() < 2 {
                    return Err(Iso8583Error::InsufficientData {
                        expected: 2,
                        actual: data.len(),
                    });
                }
                let len_str = self.decode_string(&data[..2])?;
                let len: usize = len_str.parse()
                    .map_err(|e| Iso8583Error::FieldError(format!("invalid LLVAR length: {}", e)))?;
                
                if data.len() < 2 + len {
                    return Err(Iso8583Error::InsufficientData {
                        expected: 2 + len,
                        actual: data.len(),
                    });
                }
                
                let value = self.decode_string(&data[2..2 + len])?;
                Ok((value, 2 + len))
            }
            FieldType::Lllvar => {
                if data.len() < 3 {
                    return Err(Iso8583Error::InsufficientData {
                        expected: 3,
                        actual: data.len(),
                    });
                }
                let len_str = self.decode_string(&data[..3])?;
                let len: usize = len_str.parse()
                    .map_err(|e| Iso8583Error::FieldError(format!("invalid LLLVAR length: {}", e)))?;
                
                if data.len() < 3 + len {
                    return Err(Iso8583Error::InsufficientData {
                        expected: 3 + len,
                        actual: data.len(),
                    });
                }
                
                let value = self.decode_string(&data[3..3 + len])?;
                Ok((value, 3 + len))
            }
            FieldType::Binary(len) => {
                if data.len() < *len {
                    return Err(Iso8583Error::InsufficientData {
                        expected: *len,
                        actual: data.len(),
                    });
                }
                let value = hex::encode(&data[..*len]);
                Ok((value, *len))
            }
        }
    }
    
    /// Decode string based on encoding
    fn decode_string(&self, data: &[u8]) -> Result<String, Iso8583Error> {
        match self.encoding {
            MessageEncoding::Ascii => {
                String::from_utf8(data.to_vec())
                    .map_err(|e| Iso8583Error::InvalidFormat(format!("invalid ASCII: {}", e)))
            }
            MessageEncoding::Ebcdic => {
                let ascii_data: Vec<u8> = data.iter().map(|&b| self.ebcdic_to_ascii(b)).collect();
                String::from_utf8(ascii_data)
                    .map_err(|e| Iso8583Error::InvalidFormat(format!("invalid EBCDIC: {}", e)))
            }
            MessageEncoding::Binary => {
                // For binary encoding, return hex representation
                Ok(hex::encode(data))
            }
        }
    }
    
    /// Convert EBCDIC byte to ASCII
    fn ebcdic_to_ascii(&self, ebcdic: u8) -> u8 {
        // Simplified EBCDIC to ASCII conversion table
        // This is a basic implementation - production code should use a complete table
        match ebcdic {
            0x40 => b' ',  // Space
            0x4B => b'.',  // Period
            0x4C => b'<',  // Less than
            0x4D => b'(',  // Left parenthesis
            0x4E => b'+',  // Plus
            0x4F => b'|',  // Vertical bar
            0x50 => b'&',  // Ampersand
            0x5A => b'!',  // Exclamation
            0x5B => b'$',  // Dollar
            0x5C => b'*',  // Asterisk
            0x5D => b')',  // Right parenthesis
            0x5E => b';',  // Semicolon
            0x60 => b'-',  // Hyphen
            0x61 => b'/',  // Slash
            0x6B => b',',  // Comma
            0x6C => b'%',  // Percent
            0x6D => b'_',  // Underscore
            0x6E => b'>',  // Greater than
            0x6F => b'?',  // Question mark
            0x7A => b':',  // Colon
            0x7B => b'#',  // Hash
            0x7C => b'@',  // At sign
            0x7D => b'\'', // Apostrophe
            0x7E => b'=',  // Equal
            0x7F => b'"',  // Quote
            0x81..=0x89 => b'a' + (ebcdic - 0x81), // a-i
            0x91..=0x99 => b'j' + (ebcdic - 0x91), // j-r
            0xA2..=0xA9 => b's' + (ebcdic - 0xA2), // s-z
            0xC1..=0xC9 => b'A' + (ebcdic - 0xC1), // A-I
            0xD1..=0xD9 => b'J' + (ebcdic - 0xD1), // J-R
            0xE2..=0xE9 => b'S' + (ebcdic - 0xE2), // S-Z
            0xF0..=0xF9 => b'0' + (ebcdic - 0xF0), // 0-9
            _ => b'?', // Unknown character
        }
    }
    
    /// Get field definition
    pub fn get_field_definition(&self, field_num: u8) -> Option<&FieldDefinition> {
        self.field_definitions.get(&field_num)
    }
    
    /// Check if a field is sensitive (for PCI compliance)
    pub fn is_sensitive_field(&self, field_num: u8) -> bool {
        self.field_definitions.get(&field_num)
            .map(|def| def.sensitive)
            .unwrap_or(false)
    }
}

impl Iso8583Message {
    /// Get the message type indicator
    pub fn get_mti(&self) -> &str {
        &self.mti
    }
    
    /// Get a field value
    pub fn get_field(&self, field_num: u8) -> Option<&String> {
        self.fields.get(&field_num)
    }
    
    /// Get processing code (field 3)
    pub fn get_processing_code(&self) -> Option<&String> {
        self.get_field(3)
    }
    
    /// Get primary account number (field 2)
    pub fn get_pan(&self) -> Option<&String> {
        self.get_field(2)
    }
    
    /// Get transaction amount (field 4)
    pub fn get_amount(&self) -> Option<&String> {
        self.get_field(4)
    }
    
    /// Get system trace audit number (field 11)
    pub fn get_stan(&self) -> Option<&String> {
        self.get_field(11)
    }
    
    /// Get response code (field 39)
    pub fn get_response_code(&self) -> Option<&String> {
        self.get_field(39)
    }
    
    /// Check if this is a request message
    pub fn is_request(&self) -> bool {
        self.mti.chars().nth(2) == Some('0') || self.mti.chars().nth(2) == Some('2')
    }
    
    /// Check if this is a response message
    pub fn is_response(&self) -> bool {
        self.mti.chars().nth(2) == Some('1') || self.mti.chars().nth(2) == Some('3')
    }
    
    /// Get message class (authorization, financial, etc.)
    pub fn get_message_class(&self) -> Option<char> {
        self.mti.chars().nth(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bitmap_parsing() {
        let parser = Iso8583Parser::new(MessageEncoding::Binary);
        
        // Test bitmap with first bit set (field 1)
        let bitmap_data = [0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let bitmap = parser.parse_bitmap(&bitmap_data).unwrap();
        
        assert_eq!(bitmap.len(), 64);
        assert!(bitmap[0]); // Field 1 should be set
        assert!(!bitmap[1]); // Field 2 should not be set
    }
    
    #[test]
    fn test_mti_parsing() {
        let ascii_parser = Iso8583Parser::new(MessageEncoding::Ascii);
        let mti = ascii_parser.parse_mti(b"0200").unwrap();
        assert_eq!(mti, "0200");
        
        let binary_parser = Iso8583Parser::new(MessageEncoding::Binary);
        let mti = binary_parser.parse_mti(&[0x02, 0x00, 0x00, 0x00]).unwrap();
        assert_eq!(mti, "02000000");
    }
    
    #[test]
    fn test_field_definitions() {
        let parser = Iso8583Parser::new(MessageEncoding::Ascii);
        
        // Test that sensitive fields are marked correctly
        assert!(parser.is_sensitive_field(2)); // PAN
        assert!(parser.is_sensitive_field(35)); // Track 2
        assert!(!parser.is_sensitive_field(3)); // Processing code
    }
}