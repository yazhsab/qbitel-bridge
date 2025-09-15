use bytes::{Bytes, BytesMut, Buf, BufMut};
use std::collections::HashMap;
use thiserror::Error;
use tracing::{debug, warn, error, info};

#[derive(Debug, Error)]
pub enum DataStreamError {
    #[error("invalid order: {0}")]
    InvalidOrder(String),
    #[error("invalid attribute: {0}")]
    InvalidAttribute(String),
    #[error("buffer overflow: position {position} exceeds screen size")]
    BufferOverflow { position: u16 },
    #[error("invalid field: {0}")]
    InvalidField(String),
    #[error("screen scraping error: {0}")]
    ScreenScraping(String),
    #[error("insufficient data: expected {expected}, got {actual}")]
    InsufficientData { expected: usize, actual: usize },
}

/// 3270 screen buffer position
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BufferAddress {
    pub row: u16,
    pub col: u16,
}

impl BufferAddress {
    /// Convert to linear buffer address
    pub fn to_linear(&self, cols: u16) -> u16 {
        self.row * cols + self.col
    }
    
    /// Convert from linear buffer address
    pub fn from_linear(addr: u16, cols: u16) -> Self {
        Self {
            row: addr / cols,
            col: addr % cols,
        }
    }
    
    /// Parse from 3270 address encoding
    pub fn from_3270_address(addr: &[u8]) -> Result<u16, DataStreamError> {
        if addr.len() != 2 {
            return Err(DataStreamError::InvalidOrder(
                "3270 address must be 2 bytes".to_string()
            ));
        }
        
        // 3270 addresses are encoded in a special format
        let addr1 = Self::decode_address_byte(addr[0])?;
        let addr2 = Self::decode_address_byte(addr[1])?;
        
        Ok((addr1 << 6) | addr2)
    }
    
    /// Decode a single address byte
    fn decode_address_byte(byte: u8) -> Result<u16, DataStreamError> {
        match byte {
            0x40..=0x4F => Ok((byte - 0x40) as u16),
            0x50..=0x5F => Ok((byte - 0x50 + 10) as u16),
            0x60..=0x6F => Ok((byte - 0x60 + 20) as u16),
            0x70..=0x7F => Ok((byte - 0x70 + 30) as u16),
            0xC1..=0xC9 => Ok((byte - 0xC1 + 40) as u16),
            0xD1..=0xD9 => Ok((byte - 0xD1 + 49) as u16),
            0xE2..=0xE9 => Ok((byte - 0xE2 + 58) as u16),
            0xF0..=0xF9 => Ok((byte - 0xF0 + 66) as u16),
            _ => Err(DataStreamError::InvalidOrder(
                format!("invalid address byte: 0x{:02X}", byte)
            )),
        }
    }
}

/// 3270 field attribute
#[derive(Debug, Clone, PartialEq)]
pub struct FieldAttribute {
    /// Field is protected (read-only)
    pub protected: bool,
    /// Field is numeric only
    pub numeric: bool,
    /// Field display characteristics
    pub display: DisplayAttribute,
    /// Field is modified (MDT - Modified Data Tag)
    pub modified: bool,
}

impl FieldAttribute {
    /// Parse from attribute byte
    pub fn from_byte(byte: u8) -> Self {
        Self {
            protected: (byte & 0x20) != 0,
            numeric: (byte & 0x10) != 0,
            display: DisplayAttribute::from_bits((byte & 0x0C) >> 2),
            modified: (byte & 0x01) != 0,
        }
    }
    
    /// Convert to attribute byte
    pub fn to_byte(&self) -> u8 {
        let mut byte = 0u8;
        
        if self.protected {
            byte |= 0x20;
        }
        if self.numeric {
            byte |= 0x10;
        }
        byte |= (self.display.to_bits() << 2) & 0x0C;
        if self.modified {
            byte |= 0x01;
        }
        
        byte
    }
}

/// Display attribute for fields
#[derive(Debug, Clone, PartialEq)]
pub enum DisplayAttribute {
    /// Normal display
    Normal,
    /// Blink
    Blink,
    /// Reverse video
    Reverse,
    /// Underscore
    Underscore,
}

impl DisplayAttribute {
    /// Convert from bit pattern
    pub fn from_bits(bits: u8) -> Self {
        match bits {
            0 => DisplayAttribute::Normal,
            1 => DisplayAttribute::Blink,
            2 => DisplayAttribute::Reverse,
            3 => DisplayAttribute::Underscore,
            _ => DisplayAttribute::Normal,
        }
    }
    
    /// Convert to bit pattern
    pub fn to_bits(&self) -> u8 {
        match self {
            DisplayAttribute::Normal => 0,
            DisplayAttribute::Blink => 1,
            DisplayAttribute::Reverse => 2,
            DisplayAttribute::Underscore => 3,
        }
    }
}

/// 3270 screen field
#[derive(Debug, Clone)]
pub struct ScreenField {
    /// Field start position
    pub start_pos: u16,
    /// Field attribute
    pub attribute: FieldAttribute,
    /// Field data
    pub data: String,
    /// Field length
    pub length: u16,
}

impl ScreenField {
    /// Check if field contains a specific position
    pub fn contains_position(&self, pos: u16) -> bool {
        pos >= self.start_pos && pos < self.start_pos + self.length
    }
    
    /// Extract data from field
    pub fn extract_data(&self) -> &str {
        &self.data
    }
    
    /// Check if field is input field (unprotected)
    pub fn is_input_field(&self) -> bool {
        !self.attribute.protected
    }
    
    /// Check if field is modified
    pub fn is_modified(&self) -> bool {
        self.attribute.modified
    }
}

/// 3270 screen buffer
#[derive(Debug, Clone)]
pub struct ScreenBuffer {
    /// Screen dimensions
    pub rows: u16,
    pub cols: u16,
    /// Character buffer
    buffer: Vec<u8>,
    /// Attribute buffer
    attributes: Vec<u8>,
    /// Current cursor position
    cursor_pos: u16,
    /// Screen fields
    fields: Vec<ScreenField>,
}

impl ScreenBuffer {
    /// Create a new screen buffer
    pub fn new(rows: u16, cols: u16) -> Self {
        let size = (rows * cols) as usize;
        Self {
            rows,
            cols,
            buffer: vec![0x40; size], // Fill with spaces (EBCDIC 0x40)
            attributes: vec![0x00; size],
            cursor_pos: 0,
            fields: Vec::new(),
        }
    }
    
    /// Clear the screen
    pub fn clear(&mut self) {
        let size = (self.rows * self.cols) as usize;
        self.buffer.fill(0x40); // EBCDIC space
        self.attributes.fill(0x00);
        self.cursor_pos = 0;
        self.fields.clear();
    }
    
    /// Set cursor position
    pub fn set_cursor(&mut self, pos: u16) -> Result<(), DataStreamError> {
        if pos >= self.rows * self.cols {
            return Err(DataStreamError::BufferOverflow { position: pos });
        }
        self.cursor_pos = pos;
        Ok(())
    }
    
    /// Get cursor position
    pub fn cursor_position(&self) -> u16 {
        self.cursor_pos
    }
    
    /// Write data at current cursor position
    pub fn write_data(&mut self, data: &[u8]) -> Result<(), DataStreamError> {
        for &byte in data {
            if self.cursor_pos >= self.rows * self.cols {
                return Err(DataStreamError::BufferOverflow { 
                    position: self.cursor_pos 
                });
            }
            
            self.buffer[self.cursor_pos as usize] = byte;
            self.cursor_pos += 1;
        }
        
        Ok(())
    }
    
    /// Write attribute at current cursor position
    pub fn write_attribute(&mut self, attr: u8) -> Result<(), DataStreamError> {
        if self.cursor_pos >= self.rows * self.cols {
            return Err(DataStreamError::BufferOverflow { 
                position: self.cursor_pos 
            });
        }
        
        self.attributes[self.cursor_pos as usize] = attr;
        
        // Create a new field if this is a field attribute
        if attr != 0 {
            let field_attr = FieldAttribute::from_byte(attr);
            let field = ScreenField {
                start_pos: self.cursor_pos,
                attribute: field_attr,
                data: String::new(),
                length: 0, // Will be calculated later
            };
            self.fields.push(field);
        }
        
        self.cursor_pos += 1;
        Ok(())
    }
    
    /// Get character at position
    pub fn get_char(&self, pos: u16) -> Option<u8> {
        if pos < self.rows * self.cols {
            Some(self.buffer[pos as usize])
        } else {
            None
        }
    }
    
    /// Get attribute at position
    pub fn get_attribute(&self, pos: u16) -> Option<u8> {
        if pos < self.rows * self.cols {
            Some(self.attributes[pos as usize])
        } else {
            None
        }
    }
    
    /// Extract all fields from screen
    pub fn extract_fields(&mut self) -> Vec<ScreenField> {
        self.update_field_data();
        self.fields.clone()
    }
    
    /// Update field data based on current buffer content
    fn update_field_data(&mut self) {
        for field in &mut self.fields {
            let mut data = String::new();
            let mut length = 0u16;
            
            // Find the end of the field (next attribute or end of screen)
            let mut pos = field.start_pos + 1; // Skip the attribute byte
            while pos < self.rows * self.cols {
                if self.attributes[pos as usize] != 0 {
                    break; // Found next field
                }
                
                let ch = self.buffer[pos as usize];
                if ch != 0x40 { // Not a space
                    data.push(self.ebcdic_to_ascii(ch) as char);
                }
                length += 1;
                pos += 1;
            }
            
            field.data = data.trim_end().to_string();
            field.length = length;
        }
    }
    
    /// Convert EBCDIC to ASCII (simplified)
    fn ebcdic_to_ascii(&self, ebcdic: u8) -> u8 {
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
    
    /// Get screen content as text (for debugging)
    pub fn to_text(&self) -> String {
        let mut result = String::new();
        
        for row in 0..self.rows {
            for col in 0..self.cols {
                let pos = (row * self.cols + col) as usize;
                let ch = self.ebcdic_to_ascii(self.buffer[pos]);
                result.push(ch as char);
            }
            result.push('\n');
        }
        
        result
    }
}

/// 3270 data stream parser
pub struct DataStreamParser {
    /// Screen buffer
    screen: ScreenBuffer,
}

impl DataStreamParser {
    /// Create a new parser
    pub fn new(rows: u16, cols: u16) -> Self {
        Self {
            screen: ScreenBuffer::new(rows, cols),
        }
    }
    
    /// Parse 3270 data stream
    pub fn parse(&mut self, data: &[u8]) -> Result<(), DataStreamError> {
        let mut i = 0;
        
        while i < data.len() {
            match data[i] {
                // Orders (commands)
                0x11 => {
                    // Set Buffer Address (SBA)
                    if i + 2 >= data.len() {
                        return Err(DataStreamError::InsufficientData {
                            expected: 3,
                            actual: data.len() - i,
                        });
                    }
                    let addr = BufferAddress::from_3270_address(&data[i + 1..i + 3])?;
                    self.screen.set_cursor(addr)?;
                    i += 3;
                }
                0x1D => {
                    // Start Field (SF)
                    if i + 1 >= data.len() {
                        return Err(DataStreamError::InsufficientData {
                            expected: 2,
                            actual: data.len() - i,
                        });
                    }
                    let attr = data[i + 1];
                    self.screen.write_attribute(attr)?;
                    i += 2;
                }
                0x29 => {
                    // Start Field Extended (SFE)
                    if i + 2 >= data.len() {
                        return Err(DataStreamError::InsufficientData {
                            expected: 3,
                            actual: data.len() - i,
                        });
                    }
                    let count = data[i + 1];
                    if i + 2 + count as usize > data.len() {
                        return Err(DataStreamError::InsufficientData {
                            expected: 2 + count as usize,
                            actual: data.len() - i,
                        });
                    }
                    // Process extended attributes (simplified)
                    let attr = data[i + 2];
                    self.screen.write_attribute(attr)?;
                    i += 2 + count as usize;
                }
                0x2C => {
                    // Modify Field (MF)
                    if i + 2 >= data.len() {
                        return Err(DataStreamError::InsufficientData {
                            expected: 3,
                            actual: data.len() - i,
                        });
                    }
                    let count = data[i + 1];
                    // Skip modify field data for now
                    i += 2 + count as usize;
                }
                0x08 => {
                    // Graphic Escape (GE)
                    if i + 1 >= data.len() {
                        return Err(DataStreamError::InsufficientData {
                            expected: 2,
                            actual: data.len() - i,
                        });
                    }
                    // Skip graphic character
                    i += 2;
                }
                0x13 => {
                    // Insert Cursor (IC)
                    // Set cursor position to current position
                    i += 1;
                }
                0x05 => {
                    // Program Tab (PT)
                    // Move cursor to next unprotected field
                    i += 1;
                }
                0x0D => {
                    // New Line (NL)
                    let current_row = self.screen.cursor_pos / self.screen.cols;
                    let new_pos = (current_row + 1) * self.screen.cols;
                    if new_pos < self.screen.rows * self.screen.cols {
                        self.screen.set_cursor(new_pos)?;
                    }
                    i += 1;
                }
                0x15 => {
                    // End of Medium (EM)
                    i += 1;
                }
                // Data characters
                _ => {
                    // Regular data character
                    self.screen.write_data(&[data[i]])?;
                    i += 1;
                }
            }
        }
        
        Ok(())
    }
    
    /// Get screen buffer
    pub fn screen(&self) -> &ScreenBuffer {
        &self.screen
    }
    
    /// Get mutable screen buffer
    pub fn screen_mut(&mut self) -> &mut ScreenBuffer {
        &mut self.screen
    }
    
    /// Extract fields from screen
    pub fn extract_fields(&mut self) -> Vec<ScreenField> {
        self.screen.extract_fields()
    }
    
    /// Scrape specific field by position
    pub fn scrape_field_at(&mut self, row: u16, col: u16) -> Result<Option<String>, DataStreamError> {
        let pos = row * self.screen.cols + col;
        let fields = self.extract_fields();
        
        for field in fields {
            if field.contains_position(pos) && field.is_input_field() {
                return Ok(Some(field.data));
            }
        }
        
        Ok(None)
    }
    
    /// Scrape all input fields
    pub fn scrape_input_fields(&mut self) -> Result<HashMap<u16, String>, DataStreamError> {
        let fields = self.extract_fields();
        let mut result = HashMap::new();
        
        for field in fields {
            if field.is_input_field() && !field.data.is_empty() {
                result.insert(field.start_pos, field.data);
            }
        }
        
        Ok(result)
    }
    
    /// Find field containing specific text
    pub fn find_field_with_text(&mut self, text: &str) -> Result<Option<ScreenField>, DataStreamError> {
        let fields = self.extract_fields();
        
        for field in fields {
            if field.data.contains(text) {
                return Ok(Some(field));
            }
        }
        
        Ok(None)
    }
    
    /// Get screen content as text
    pub fn get_screen_text(&self) -> String {
        self.screen.to_text()
    }
    
    /// Clear screen
    pub fn clear_screen(&mut self) {
        self.screen.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_buffer_address_conversion() {
        let addr = BufferAddress { row: 5, col: 10 };
        let linear = addr.to_linear(80);
        assert_eq!(linear, 5 * 80 + 10);
        
        let back = BufferAddress::from_linear(linear, 80);
        assert_eq!(back, addr);
    }
    
    #[test]
    fn test_field_attribute_parsing() {
        let attr = FieldAttribute::from_byte(0x20); // Protected field
        assert!(attr.protected);
        assert!(!attr.numeric);
        assert_eq!(attr.display, DisplayAttribute::Normal);
        assert!(!attr.modified);
        
        let byte = attr.to_byte();
        assert_eq!(byte, 0x20);
    }
    
    #[test]
    fn test_screen_buffer_creation() {
        let screen = ScreenBuffer::new(24, 80);
        assert_eq!(screen.rows, 24);
        assert_eq!(screen.cols, 80);
        assert_eq!(screen.cursor_position(), 0);
    }
    
    #[test]
    fn test_data_stream_parsing() {
        let mut parser = DataStreamParser::new(24, 80);
        
        // Simple data stream: SBA to position 0, write "HELLO"
        let data = [
            0x11, 0x40, 0x40, // SBA to 0,0
            b'H', b'E', b'L', b'L', b'O'
        ];
        
        parser.parse(&data).unwrap();
        
        // Check that data was written
        assert_eq!(parser.screen().get_char(3), Some(b'H'));
        assert_eq!(parser.screen().get_char(4), Some(b'E'));
    }
    
    #[test]
    fn test_field_extraction() {
        let mut parser = DataStreamParser::new(24, 80);
        
        // Create a field: SBA, SF with attribute, data
        let data = [
            0x11, 0x40, 0x40, // SBA to 0,0
            0x1D, 0x00,       // SF with normal attribute
            b'T', b'E', b'S', b'T'
        ];
        
        parser.parse(&data).unwrap();
        let fields = parser.extract_fields();
        
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].data, "TEST");
    }
    
    #[test]
    fn test_screen_scraping() {
        let mut parser = DataStreamParser::new(24, 80);
        
        // Create input field
        let data = [
            0x11, 0x40, 0x40, // SBA to 0,0
            0x1D, 0x00,       // SF with unprotected attribute
            b'I', b'N', b'P', b'U', b'T'
        ];
        
        parser.parse(&data).unwrap();
        let input_fields = parser.scrape_input_fields().unwrap();
        
        assert!(!input_fields.is_empty());
    }
}