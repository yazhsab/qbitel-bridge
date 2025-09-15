use bytes::{Bytes, BytesMut, Buf, BufMut};
use std::collections::HashMap;
use thiserror::Error;
use tracing::{debug, warn, error, info};

#[derive(Debug, Error)]
pub enum Tn3270Error {
    #[error("protocol error: {0}")]
    Protocol(String),
    #[error("invalid data stream: {0}")]
    InvalidDataStream(String),
    #[error("session error: {0}")]
    Session(String),
    #[error("device negotiation failed: {0}")]
    DeviceNegotiation(String),
    #[error("insufficient data: expected {expected}, got {actual}")]
    InsufficientData { expected: usize, actual: usize },
}

/// TN3270E protocol state
#[derive(Debug, Clone, PartialEq)]
pub enum ProtocolState {
    /// Initial connection state
    Initial,
    /// Telnet negotiation in progress
    TelnetNegotiation,
    /// TN3270E negotiation in progress
    Tn3270Negotiation,
    /// Device type negotiation
    DeviceNegotiation,
    /// Session established and ready
    SessionEstablished,
    /// Session terminated
    Terminated,
    /// Error state
    Error(String),
}

/// TN3270E device types
#[derive(Debug, Clone, PartialEq)]
pub enum DeviceType {
    /// IBM 3278 terminal (24x80)
    Ibm3278_2,
    /// IBM 3278 terminal (32x80)
    Ibm3278_3,
    /// IBM 3278 terminal (43x80)
    Ibm3278_4,
    /// IBM 3278 terminal (27x132)
    Ibm3278_5,
    /// IBM 3279 color terminal
    Ibm3279_2,
    /// IBM 3279 color terminal
    Ibm3279_3,
    /// Generic 3270 terminal
    Generic3270,
}

impl DeviceType {
    /// Get screen dimensions (rows, columns)
    pub fn dimensions(&self) -> (u16, u16) {
        match self {
            DeviceType::Ibm3278_2 | DeviceType::Ibm3279_2 => (24, 80),
            DeviceType::Ibm3278_3 | DeviceType::Ibm3279_3 => (32, 80),
            DeviceType::Ibm3278_4 => (43, 80),
            DeviceType::Ibm3278_5 => (27, 132),
            DeviceType::Generic3270 => (24, 80),
        }
    }
    
    /// Get device name string
    pub fn name(&self) -> &'static str {
        match self {
            DeviceType::Ibm3278_2 => "IBM-3278-2",
            DeviceType::Ibm3278_3 => "IBM-3278-3",
            DeviceType::Ibm3278_4 => "IBM-3278-4",
            DeviceType::Ibm3278_5 => "IBM-3278-5",
            DeviceType::Ibm3279_2 => "IBM-3279-2",
            DeviceType::Ibm3279_3 => "IBM-3279-3",
            DeviceType::Generic3270 => "IBM-3270",
        }
    }
}

/// TN3270E functions
#[derive(Debug, Clone, PartialEq)]
pub enum Tn3270Function {
    /// Bind image
    BindImage,
    /// Data stream utilities
    DataStreamUtils,
    /// Response handling
    Responses,
    /// SCS (SNA Character String)
    Scs,
    /// System request
    SysReq,
}

/// TN3270E header data type
#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    /// 3270 data
    Data3270,
    /// SCS data
    ScsData,
    /// Response data
    Response,
    /// Bind image
    BindImage,
    /// Unbind
    Unbind,
    /// NVT data
    NvtData,
    /// Request
    Request,
    /// SSCP-LU data
    SscpLuData,
    /// Print job
    PrintJob,
}

impl DataType {
    /// Convert to byte value
    pub fn to_byte(&self) -> u8 {
        match self {
            DataType::Data3270 => 0x00,
            DataType::ScsData => 0x01,
            DataType::Response => 0x02,
            DataType::BindImage => 0x03,
            DataType::Unbind => 0x04,
            DataType::NvtData => 0x05,
            DataType::Request => 0x06,
            DataType::SscpLuData => 0x07,
            DataType::PrintJob => 0x08,
        }
    }
    
    /// Convert from byte value
    pub fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0x00 => Some(DataType::Data3270),
            0x01 => Some(DataType::ScsData),
            0x02 => Some(DataType::Response),
            0x03 => Some(DataType::BindImage),
            0x04 => Some(DataType::Unbind),
            0x05 => Some(DataType::NvtData),
            0x06 => Some(DataType::Request),
            0x07 => Some(DataType::SscpLuData),
            0x08 => Some(DataType::PrintJob),
            _ => None,
        }
    }
}

/// TN3270E header
#[derive(Debug, Clone)]
pub struct Tn3270Header {
    /// Data type
    pub data_type: DataType,
    /// Request flag
    pub request_flag: u8,
    /// Response flag
    pub response_flag: u8,
    /// Sequence number
    pub seq_number: Option<u16>,
}

impl Tn3270Header {
    /// Parse header from bytes
    pub fn parse(data: &[u8]) -> Result<(Self, usize), Tn3270Error> {
        if data.len() < 5 {
            return Err(Tn3270Error::InsufficientData {
                expected: 5,
                actual: data.len(),
            });
        }
        
        let data_type = DataType::from_byte(data[0])
            .ok_or_else(|| Tn3270Error::InvalidDataStream(format!("invalid data type: {}", data[0])))?;
        
        let request_flag = data[1];
        let response_flag = data[2];
        
        let seq_number = if data[3] != 0 || data[4] != 0 {
            Some(u16::from_be_bytes([data[3], data[4]]))
        } else {
            None
        };
        
        Ok((Self {
            data_type,
            request_flag,
            response_flag,
            seq_number,
        }, 5))
    }
    
    /// Serialize header to bytes
    pub fn serialize(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(5);
        result.push(self.data_type.to_byte());
        result.push(self.request_flag);
        result.push(self.response_flag);
        
        if let Some(seq) = self.seq_number {
            result.extend_from_slice(&seq.to_be_bytes());
        } else {
            result.extend_from_slice(&[0, 0]);
        }
        
        result
    }
}

/// TN3270E session configuration
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Device type
    pub device_type: DeviceType,
    /// Supported functions
    pub functions: Vec<Tn3270Function>,
    /// Device name
    pub device_name: Option<String>,
    /// Enable TN3270E mode
    pub tn3270e_mode: bool,
    /// Enable binary mode
    pub binary_mode: bool,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            device_type: DeviceType::Ibm3278_2,
            functions: vec![
                Tn3270Function::BindImage,
                Tn3270Function::DataStreamUtils,
                Tn3270Function::Responses,
            ],
            device_name: None,
            tn3270e_mode: true,
            binary_mode: true,
        }
    }
}

/// TN3270E protocol state machine
pub struct Tn3270StateMachine {
    /// Current protocol state
    state: ProtocolState,
    /// Session configuration
    config: SessionConfig,
    /// Negotiated device type
    negotiated_device: Option<DeviceType>,
    /// Negotiated functions
    negotiated_functions: Vec<Tn3270Function>,
    /// Session sequence number
    sequence_number: u16,
    /// Pending responses
    pending_responses: HashMap<u16, Vec<u8>>,
}

impl Tn3270StateMachine {
    /// Create a new state machine
    pub fn new(config: SessionConfig) -> Self {
        Self {
            state: ProtocolState::Initial,
            config,
            negotiated_device: None,
            negotiated_functions: Vec::new(),
            sequence_number: 1,
            pending_responses: HashMap::new(),
        }
    }
    
    /// Get current state
    pub fn state(&self) -> &ProtocolState {
        &self.state
    }
    
    /// Process incoming data and update state
    pub fn process_data(&mut self, data: &[u8]) -> Result<Vec<u8>, Tn3270Error> {
        match &self.state {
            ProtocolState::Initial => {
                self.handle_initial_data(data)
            }
            ProtocolState::TelnetNegotiation => {
                self.handle_telnet_negotiation(data)
            }
            ProtocolState::Tn3270Negotiation => {
                self.handle_tn3270_negotiation(data)
            }
            ProtocolState::DeviceNegotiation => {
                self.handle_device_negotiation(data)
            }
            ProtocolState::SessionEstablished => {
                self.handle_session_data(data)
            }
            ProtocolState::Terminated => {
                Err(Tn3270Error::Session("session terminated".to_string()))
            }
            ProtocolState::Error(msg) => {
                Err(Tn3270Error::Session(format!("session in error state: {}", msg)))
            }
        }
    }
    
    /// Handle initial connection data
    fn handle_initial_data(&mut self, data: &[u8]) -> Result<Vec<u8>, Tn3270Error> {
        // Look for Telnet IAC (Interpret As Command) sequences
        if data.contains(&0xFF) {
            self.state = ProtocolState::TelnetNegotiation;
            self.handle_telnet_negotiation(data)
        } else {
            // Assume direct 3270 data stream
            self.state = ProtocolState::SessionEstablished;
            self.negotiated_device = Some(self.config.device_type.clone());
            self.handle_session_data(data)
        }
    }
    
    /// Handle Telnet negotiation
    fn handle_telnet_negotiation(&mut self, data: &[u8]) -> Result<Vec<u8>, Tn3270Error> {
        let mut response = Vec::new();
        let mut i = 0;
        
        while i < data.len() {
            if data[i] == 0xFF && i + 2 < data.len() {
                // Telnet command sequence
                let command = data[i + 1];
                let option = data[i + 2];
                
                match command {
                    0xFD => {
                        // DO command - respond with WILL or WONT
                        match option {
                            0x18 => {
                                // Terminal Type
                                response.extend_from_slice(&[0xFF, 0xFB, 0x18]); // WILL TERMINAL-TYPE
                            }
                            0x28 => {
                                // TN3270E
                                if self.config.tn3270e_mode {
                                    response.extend_from_slice(&[0xFF, 0xFB, 0x28]); // WILL TN3270E
                                    self.state = ProtocolState::Tn3270Negotiation;
                                } else {
                                    response.extend_from_slice(&[0xFF, 0xFC, 0x28]); // WONT TN3270E
                                    self.state = ProtocolState::DeviceNegotiation;
                                }
                            }
                            0x00 => {
                                // Binary Transmission
                                if self.config.binary_mode {
                                    response.extend_from_slice(&[0xFF, 0xFB, 0x00]); // WILL BINARY
                                } else {
                                    response.extend_from_slice(&[0xFF, 0xFC, 0x00]); // WONT BINARY
                                }
                            }
                            _ => {
                                // Unknown option - respond with WONT
                                response.extend_from_slice(&[0xFF, 0xFC, option]);
                            }
                        }
                    }
                    0xFE => {
                        // DONT command - respond with WONT
                        response.extend_from_slice(&[0xFF, 0xFC, option]);
                    }
                    0xFB => {
                        // WILL command - respond with DO or DONT
                        match option {
                            0x00 => {
                                // Binary Transmission
                                response.extend_from_slice(&[0xFF, 0xFD, 0x00]); // DO BINARY
                            }
                            _ => {
                                // Unknown option - respond with DONT
                                response.extend_from_slice(&[0xFF, 0xFE, option]);
                            }
                        }
                    }
                    0xFC => {
                        // WONT command - acknowledge
                        response.extend_from_slice(&[0xFF, 0xFE, option]); // DONT
                    }
                    _ => {
                        debug!(command = command, option = option, "unknown telnet command");
                    }
                }
                
                i += 3;
            } else {
                i += 1;
            }
        }
        
        debug!(state = ?self.state, "telnet negotiation processed");
        Ok(response)
    }
    
    /// Handle TN3270E negotiation
    fn handle_tn3270_negotiation(&mut self, data: &[u8]) -> Result<Vec<u8>, Tn3270Error> {
        // TN3270E subnegotiation format:
        // IAC SB TN3270E <subnegotiation-data> IAC SE
        
        let mut response = Vec::new();
        
        // Look for subnegotiation sequences
        if let Some(start) = data.windows(3).position(|w| w == [0xFF, 0xFA, 0x28]) {
            if let Some(end) = data[start..].windows(2).position(|w| w == [0xFF, 0xF0]) {
                let sub_data = &data[start + 3..start + end];
                
                if !sub_data.is_empty() {
                    match sub_data[0] {
                        0x02 => {
                            // DEVICE-TYPE request
                            let device_name = self.config.device_type.name();
                            response.extend_from_slice(&[0xFF, 0xFA, 0x28, 0x02, 0x00]);
                            response.extend_from_slice(device_name.as_bytes());
                            response.extend_from_slice(&[0xFF, 0xF0]);
                            
                            self.negotiated_device = Some(self.config.device_type.clone());
                            self.state = ProtocolState::DeviceNegotiation;
                        }
                        0x03 => {
                            // FUNCTIONS request
                            response.extend_from_slice(&[0xFF, 0xFA, 0x28, 0x03, 0x00]);
                            
                            // Add supported functions
                            for function in &self.config.functions {
                                match function {
                                    Tn3270Function::BindImage => response.extend_from_slice(b"BIND_IMAGE "),
                                    Tn3270Function::DataStreamUtils => response.extend_from_slice(b"DATA_STREAM_UTILS "),
                                    Tn3270Function::Responses => response.extend_from_slice(b"RESPONSES "),
                                    Tn3270Function::Scs => response.extend_from_slice(b"SCS "),
                                    Tn3270Function::SysReq => response.extend_from_slice(b"SYSREQ "),
                                }
                            }
                            
                            response.extend_from_slice(&[0xFF, 0xF0]);
                            self.negotiated_functions = self.config.functions.clone();
                        }
                        _ => {
                            warn!(command = sub_data[0], "unknown TN3270E subnegotiation command");
                        }
                    }
                }
            }
        }
        
        // Check if negotiation is complete
        if self.negotiated_device.is_some() && !self.negotiated_functions.is_empty() {
            self.state = ProtocolState::SessionEstablished;
            info!(
                device = ?self.negotiated_device,
                functions = ?self.negotiated_functions,
                "TN3270E negotiation completed"
            );
        }
        
        Ok(response)
    }
    
    /// Handle device negotiation
    fn handle_device_negotiation(&mut self, data: &[u8]) -> Result<Vec<u8>, Tn3270Error> {
        // For basic TN3270 (non-E), device negotiation is simpler
        self.negotiated_device = Some(self.config.device_type.clone());
        self.state = ProtocolState::SessionEstablished;
        
        info!(device = ?self.negotiated_device, "device negotiation completed");
        
        // Echo back the data for now
        Ok(data.to_vec())
    }
    
    /// Handle session data
    fn handle_session_data(&mut self, data: &[u8]) -> Result<Vec<u8>, Tn3270Error> {
        if self.config.tn3270e_mode {
            self.handle_tn3270e_data(data)
        } else {
            self.handle_tn3270_data(data)
        }
    }
    
    /// Handle TN3270E data with headers
    fn handle_tn3270e_data(&mut self, data: &[u8]) -> Result<Vec<u8>, Tn3270Error> {
        if data.len() < 5 {
            return Ok(Vec::new()); // Not enough data for header
        }
        
        let (header, header_len) = Tn3270Header::parse(data)?;
        let payload = &data[header_len..];
        
        debug!(
            data_type = ?header.data_type,
            seq = ?header.seq_number,
            payload_len = payload.len(),
            "received TN3270E data"
        );
        
        match header.data_type {
            DataType::Data3270 => {
                // Process 3270 data stream
                let response_data = self.process_3270_data(payload)?;
                
                // Create response header
                let response_header = Tn3270Header {
                    data_type: DataType::Data3270,
                    request_flag: 0x00,
                    response_flag: 0x00,
                    seq_number: Some(self.sequence_number),
                };
                
                self.sequence_number = self.sequence_number.wrapping_add(1);
                
                let mut response = response_header.serialize();
                response.extend_from_slice(&response_data);
                
                Ok(response)
            }
            DataType::Response => {
                // Handle response data
                if let Some(seq) = header.seq_number {
                    self.pending_responses.remove(&seq);
                }
                Ok(Vec::new())
            }
            _ => {
                warn!(data_type = ?header.data_type, "unhandled TN3270E data type");
                Ok(Vec::new())
            }
        }
    }
    
    /// Handle basic TN3270 data
    fn handle_tn3270_data(&mut self, data: &[u8]) -> Result<Vec<u8>, Tn3270Error> {
        self.process_3270_data(data)
    }
    
    /// Process 3270 data stream
    fn process_3270_data(&mut self, data: &[u8]) -> Result<Vec<u8>, Tn3270Error> {
        if data.is_empty() {
            return Ok(Vec::new());
        }
        
        // Basic 3270 command processing
        match data[0] {
            0x01 => {
                // Write command
                debug!("processing Write command");
                self.process_write_command(&data[1..])
            }
            0x05 => {
                // Erase/Write command
                debug!("processing Erase/Write command");
                self.process_erase_write_command(&data[1..])
            }
            0x0D => {
                // Write Structured Field command
                debug!("processing Write Structured Field command");
                self.process_write_structured_field(&data[1..])
            }
            0x6F => {
                // Erase/Write Alternate command
                debug!("processing Erase/Write Alternate command");
                self.process_erase_write_alternate(&data[1..])
            }
            0x7E => {
                // Erase All Unprotected command
                debug!("processing Erase All Unprotected command");
                self.process_erase_all_unprotected(&data[1..])
            }
            _ => {
                warn!(command = data[0], "unknown 3270 command");
                Ok(Vec::new())
            }
        }
    }
    
    /// Process Write command
    fn process_write_command(&mut self, data: &[u8]) -> Result<Vec<u8>, Tn3270Error> {
        // Write command format: WCC [orders and data]
        if data.is_empty() {
            return Ok(Vec::new());
        }
        
        let wcc = data[0]; // Write Control Character
        let stream_data = &data[1..];
        
        debug!(wcc = wcc, data_len = stream_data.len(), "processing write command");
        
        // Process the data stream (simplified)
        // In a real implementation, this would parse orders, attributes, and data
        
        // Generate a simple acknowledgment
        Ok(vec![0x88]) // AID: No AID generated
    }
    
    /// Process Erase/Write command
    fn process_erase_write_command(&mut self, data: &[u8]) -> Result<Vec<u8>, Tn3270Error> {
        // Similar to Write but clears screen first
        debug!(data_len = data.len(), "processing erase/write command");
        
        // Generate acknowledgment
        Ok(vec![0x88]) // AID: No AID generated
    }
    
    /// Process Write Structured Field command
    fn process_write_structured_field(&mut self, data: &[u8]) -> Result<Vec<u8>, Tn3270Error> {
        debug!(data_len = data.len(), "processing write structured field command");
        
        // Structured fields have length and type headers
        // This is a simplified implementation
        
        Ok(vec![0x88]) // AID: No AID generated
    }
    
    /// Process Erase/Write Alternate command
    fn process_erase_write_alternate(&mut self, data: &[u8]) -> Result<Vec<u8>, Tn3270Error> {
        debug!(data_len = data.len(), "processing erase/write alternate command");
        Ok(vec![0x88])
    }
    
    /// Process Erase All Unprotected command
    fn process_erase_all_unprotected(&mut self, data: &[u8]) -> Result<Vec<u8>, Tn3270Error> {
        debug!(data_len = data.len(), "processing erase all unprotected command");
        Ok(vec![0x88])
    }
    
    /// Get negotiated device type
    pub fn negotiated_device(&self) -> Option<&DeviceType> {
        self.negotiated_device.as_ref()
    }
    
    /// Get negotiated functions
    pub fn negotiated_functions(&self) -> &[Tn3270Function] {
        &self.negotiated_functions
    }
    
    /// Terminate session
    pub fn terminate(&mut self) {
        self.state = ProtocolState::Terminated;
        info!("TN3270E session terminated");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_device_type_dimensions() {
        assert_eq!(DeviceType::Ibm3278_2.dimensions(), (24, 80));
        assert_eq!(DeviceType::Ibm3278_4.dimensions(), (43, 80));
        assert_eq!(DeviceType::Ibm3278_5.dimensions(), (27, 132));
    }
    
    #[test]
    fn test_data_type_conversion() {
        assert_eq!(DataType::Data3270.to_byte(), 0x00);
        assert_eq!(DataType::from_byte(0x00), Some(DataType::Data3270));
        assert_eq!(DataType::from_byte(0xFF), None);
    }
    
    #[test]
    fn test_tn3270_header_parsing() {
        let data = [0x00, 0x01, 0x02, 0x00, 0x10];
        let (header, len) = Tn3270Header::parse(&data).unwrap();
        
        assert_eq!(header.data_type, DataType::Data3270);
        assert_eq!(header.request_flag, 0x01);
        assert_eq!(header.response_flag, 0x02);
        assert_eq!(header.seq_number, Some(0x0010));
        assert_eq!(len, 5);
    }
    
    #[test]
    fn test_state_machine_creation() {
        let config = SessionConfig::default();
        let sm = Tn3270StateMachine::new(config);
        
        assert_eq!(sm.state(), &ProtocolState::Initial);
        assert_eq!(sm.sequence_number, 1);
    }
    
    #[test]
    fn test_telnet_negotiation() {
        let config = SessionConfig::default();
        let mut sm = Tn3270StateMachine::new(config);
        
        // Simulate DO TERMINAL-TYPE
        let data = [0xFF, 0xFD, 0x18];
        let response = sm.process_data(&data).unwrap();
        
        // Should respond with WILL TERMINAL-TYPE
        assert_eq!(response, vec![0xFF, 0xFB, 0x18]);
    }
}