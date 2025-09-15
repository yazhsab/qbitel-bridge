use adapter_sdk::{L7Adapter, AdapterError};
use async_trait::async_trait;
use bytes::Bytes;
use tracing::{debug, warn, error, info};
use std::sync::{Arc, Mutex};
use std::collections::HashSet;

pub mod protocol;
pub mod security;

use protocol::{ModbusFrameParser, ModbusTcpFrame, ModbusRtuFrame, FunctionCode, ExceptionCode};
use security::{ModbusSecurityValidator, SecurityPolicy, RegisterRange, SecurityStats};

/// Modbus adapter configuration
#[derive(Debug, Clone)]
pub struct ModbusConfig {
    /// Protocol variant (TCP or RTU)
    pub protocol: ModbusProtocol,
    /// Enable security validation
    pub enable_security: bool,
    /// Enable CRC validation for RTU frames
    pub validate_crc: bool,
    /// Maximum frame size
    pub max_frame_size: usize,
    /// Default unit ID for TCP frames
    pub default_unit_id: u8,
    /// Enable frame validation
    pub validate_frames: bool,
    /// Enable industrial safety controls
    pub safety_controls: bool,
}

impl Default for ModbusConfig {
    fn default() -> Self {
        Self {
            protocol: ModbusProtocol::Tcp,
            enable_security: true,
            validate_crc: true,
            max_frame_size: 260,
            default_unit_id: 1,
            validate_frames: true,
            safety_controls: true,
        }
    }
}

/// Modbus protocol variants
#[derive(Debug, Clone, PartialEq)]
pub enum ModbusProtocol {
    /// Modbus TCP/IP
    Tcp,
    /// Modbus RTU (serial)
    Rtu,
}

/// Modbus adapter with comprehensive frame parsing and security
pub struct ModbusAdapter {
    /// Adapter configuration
    config: ModbusConfig,
    /// Frame parser
    parser: ModbusFrameParser,
    /// Security validator
    security_validator: Option<Arc<Mutex<ModbusSecurityValidator>>>,
}

impl ModbusAdapter {
    /// Create a new Modbus adapter with default configuration
    pub fn new() -> Self {
        let config = ModbusConfig::default();
        let parser = ModbusFrameParser::new();
        
        // Create default security validator if security is enabled
        let security_validator = if config.enable_security {
            let policy = SecurityPolicy::default();
            Some(Arc::new(Mutex::new(ModbusSecurityValidator::new(policy))))
        } else {
            None
        };
        
        Self {
            config,
            parser,
            security_validator,
        }
    }
    
    /// Create a new Modbus adapter with custom configuration
    pub fn with_config(config: ModbusConfig) -> Self {
        let parser = ModbusFrameParser::new();
        
        let security_validator = if config.enable_security {
            let policy = SecurityPolicy::default();
            Some(Arc::new(Mutex::new(ModbusSecurityValidator::new(policy))))
        } else {
            None
        };
        
        Self {
            config,
            parser,
            security_validator,
        }
    }
    
    /// Set security policy
    pub fn with_security_policy(mut self, policy: SecurityPolicy) -> Self {
        if self.config.enable_security {
            self.security_validator = Some(Arc::new(Mutex::new(
                ModbusSecurityValidator::new(policy)
            )));
        }
        self
    }
    
    /// Process upstream data (client to server)
    async fn process_upstream(&self, input: Bytes) -> Result<Bytes, AdapterError> {
        // Parse the frame based on protocol
        let frame_result = match self.config.protocol {
            ModbusProtocol::Tcp => {
                self.parser.parse_tcp_frame(&input)
                    .map_err(|e| AdapterError::ProcessingError(format!("TCP frame parsing failed: {}", e)))
            }
            ModbusProtocol::Rtu => {
                self.parser.parse_rtu_frame(&input)
                    .map_err(|e| AdapterError::ProcessingError(format!("RTU frame parsing failed: {}", e)))
                    .map(|rtu_frame| {
                        // Convert RTU to TCP format for internal processing
                        ModbusTcpFrame {
                            transaction_id: 0,
                            protocol_id: 0,
                            length: (rtu_frame.pdu.data.len() + 2) as u16,
                            unit_id: rtu_frame.slave_address,
                            pdu: rtu_frame.pdu,
                        }
                    })
            }
        };
        
        let frame = frame_result?;
        
        debug!(
            protocol = ?self.config.protocol,
            function_code = ?frame.pdu.function_code,
            unit_id = frame.unit_id,
            data_len = frame.pdu.data.len(),
            "parsed Modbus frame"
        );
        
        // Validate frame if enabled
        if self.config.validate_frames {
            self.validate_frame(&frame)?;
        }
        
        // Apply security validation if enabled
        if let Some(security_validator) = &self.security_validator {
            let client_id = self.extract_client_id(&input);
            let mut validator = security_validator.lock()
                .map_err(|e| AdapterError::ProcessingError(format!("security validator lock failed: {}", e)))?;
            
            validator.validate_request(&client_id, &frame)
                .map_err(|e| AdapterError::SecurityError(format!("security validation failed: {}", e)))?;
        }
        
        // Apply safety controls if enabled
        if self.config.safety_controls {
            self.apply_safety_controls(&frame)?;
        }
        
        // Serialize frame back to bytes
        let output = match self.config.protocol {
            ModbusProtocol::Tcp => self.parser.serialize_tcp_frame(&frame),
            ModbusProtocol::Rtu => {
                // Convert back to RTU format
                let rtu_frame = ModbusRtuFrame {
                    slave_address: frame.unit_id,
                    pdu: frame.pdu,
                    crc: 0, // Will be calculated during serialization
                };
                self.parser.serialize_rtu_frame(&rtu_frame)
            }
        };
        
        info!(
            protocol = ?self.config.protocol,
            function_code = ?frame.pdu.function_code,
            "processed upstream Modbus frame"
        );
        
        Ok(Bytes::from(output))
    }
    
    /// Process downstream data (server to client)
    async fn process_downstream(&self, input: Bytes) -> Result<Bytes, AdapterError> {
        // Parse and validate response frame
        let frame_result = match self.config.protocol {
            ModbusProtocol::Tcp => {
                self.parser.parse_tcp_frame(&input)
                    .map_err(|e| AdapterError::ProcessingError(format!("TCP response parsing failed: {}", e)))
            }
            ModbusProtocol::Rtu => {
                self.parser.parse_rtu_frame(&input)
                    .map_err(|e| AdapterError::ProcessingError(format!("RTU response parsing failed: {}", e)))
                    .map(|rtu_frame| {
                        ModbusTcpFrame {
                            transaction_id: 0,
                            protocol_id: 0,
                            length: (rtu_frame.pdu.data.len() + 2) as u16,
                            unit_id: rtu_frame.slave_address,
                            pdu: rtu_frame.pdu,
                        }
                    })
            }
        };
        
        let frame = frame_result?;
        
        debug!(
            protocol = ?self.config.protocol,
            function_code = ?frame.pdu.function_code,
            unit_id = frame.unit_id,
            "parsed downstream Modbus frame"
        );
        
        // Validate response frame
        if self.config.validate_frames {
            self.validate_response_frame(&frame)?;
        }
        
        // Serialize frame back to bytes
        let output = match self.config.protocol {
            ModbusProtocol::Tcp => self.parser.serialize_tcp_frame(&frame),
            ModbusProtocol::Rtu => {
                let rtu_frame = ModbusRtuFrame {
                    slave_address: frame.unit_id,
                    pdu: frame.pdu,
                    crc: 0,
                };
                self.parser.serialize_rtu_frame(&rtu_frame)
            }
        };
        
        Ok(Bytes::from(output))
    }
    
    /// Validate Modbus frame
    fn validate_frame(&self, frame: &ModbusTcpFrame) -> Result<(), AdapterError> {
        // Check unit ID
        if frame.unit_id == 0 && self.config.protocol == ModbusProtocol::Tcp {
            warn!("unit ID 0 is reserved for broadcast");
        }
        
        // Check function code
        if !self.is_supported_function_code(frame.pdu.function_code) {
            return Err(AdapterError::ProcessingError(format!(
                "unsupported function code: {:?}", frame.pdu.function_code
            )));
        }
        
        // Check data length
        if frame.pdu.data.len() > self.config.max_frame_size - 8 {
            return Err(AdapterError::ProcessingError(format!(
                "frame data too large: {} bytes", frame.pdu.data.len()
            )));
        }
        
        Ok(())
    }
    
    /// Validate response frame
    fn validate_response_frame(&self, frame: &ModbusTcpFrame) -> Result<(), AdapterError> {
        // Check for exception responses
        let function_code_byte = frame.pdu.function_code.to_byte();
        if function_code_byte & 0x80 != 0 {
            if !frame.pdu.data.is_empty() {
                let exception_code = frame.pdu.data[0];
                debug!(
                    function_code = function_code_byte & 0x7F,
                    exception_code = exception_code,
                    "received Modbus exception response"
                );
            }
        }
        
        Ok(())
    }
    
    /// Check if function code is supported
    fn is_supported_function_code(&self, function_code: FunctionCode) -> bool {
        matches!(function_code,
            FunctionCode::ReadCoils |
            FunctionCode::ReadDiscreteInputs |
            FunctionCode::ReadHoldingRegisters |
            FunctionCode::ReadInputRegisters |
            FunctionCode::WriteSingleCoil |
            FunctionCode::WriteSingleRegister |
            FunctionCode::WriteMultipleCoils |
            FunctionCode::WriteMultipleRegisters
        )
    }
    
    /// Apply industrial safety controls
    fn apply_safety_controls(&self, frame: &ModbusTcpFrame) -> Result<(), AdapterError> {
        // Check for potentially dangerous write operations
        if frame.pdu.function_code.is_write_operation() {
            debug!(
                function_code = ?frame.pdu.function_code,
                unit_id = frame.unit_id,
                "applying safety controls to write operation"
            );
            
            // Additional safety checks could be implemented here
            // For example:
            // - Check for writes to safety-critical registers
            // - Validate write values are within safe ranges
            // - Implement interlocks for certain operations
        }
        
        Ok(())
    }
    
    /// Extract client identifier from request data
    fn extract_client_id(&self, _data: &[u8]) -> String {
        // In a real implementation, this would extract client information
        // from connection context, IP address, etc.
        "unknown_client".to_string()
    }
    
    /// Add authorized client to security validator
    pub fn add_authorized_client(&self, client_id: String) -> Result<(), AdapterError> {
        if let Some(security_validator) = &self.security_validator {
            let mut validator = security_validator.lock()
                .map_err(|e| AdapterError::ProcessingError(format!("security validator lock failed: {}", e)))?;
            validator.add_authorized_client(client_id);
            Ok(())
        } else {
            Err(AdapterError::ProcessingError("security not enabled".to_string()))
        }
    }
    
    /// Get security statistics
    pub fn get_security_stats(&self) -> Result<SecurityStats, AdapterError> {
        if let Some(security_validator) = &self.security_validator {
            let validator = security_validator.lock()
                .map_err(|e| AdapterError::ProcessingError(format!("security validator lock failed: {}", e)))?;
            Ok(validator.get_stats())
        } else {
            Err(AdapterError::ProcessingError("security not enabled".to_string()))
        }
    }
    
    /// Get adapter statistics
    pub fn get_stats(&self) -> ModbusStats {
        let security_stats = self.get_security_stats().ok();
        
        ModbusStats {
            protocol: self.config.protocol.clone(),
            security_enabled: self.config.enable_security,
            safety_controls_enabled: self.config.safety_controls,
            frame_validation_enabled: self.config.validate_frames,
            max_frame_size: self.config.max_frame_size,
            security_stats,
        }
    }
}

/// Modbus adapter statistics
#[derive(Debug, Clone)]
pub struct ModbusStats {
    pub protocol: ModbusProtocol,
    pub security_enabled: bool,
    pub safety_controls_enabled: bool,
    pub frame_validation_enabled: bool,
    pub max_frame_size: usize,
    pub security_stats: Option<SecurityStats>,
}

#[async_trait]
impl L7Adapter for ModbusAdapter {
    async fn to_upstream(&self, input: Bytes) -> Result<Bytes, AdapterError> {
        self.process_upstream(input).await
    }
    
    async fn to_client(&self, input: Bytes) -> Result<Bytes, AdapterError> {
        self.process_downstream(input).await
    }
    
    fn name(&self) -> &'static str { 
        "modbus" 
    }
}

impl Default for ModbusAdapter {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for Modbus adapter configuration
pub struct ModbusAdapterBuilder {
    config: ModbusConfig,
    security_policy: Option<SecurityPolicy>,
}

impl ModbusAdapterBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: ModbusConfig::default(),
            security_policy: None,
        }
    }
    
    /// Set protocol variant
    pub fn protocol(mut self, protocol: ModbusProtocol) -> Self {
        self.config.protocol = protocol;
        self
    }
    
    /// Enable/disable security validation
    pub fn security(mut self, enabled: bool) -> Self {
        self.config.enable_security = enabled;
        self
    }
    
    /// Enable/disable CRC validation
    pub fn crc_validation(mut self, enabled: bool) -> Self {
        self.config.validate_crc = enabled;
        self
    }
    
    /// Set maximum frame size
    pub fn max_frame_size(mut self, size: usize) -> Self {
        self.config.max_frame_size = size;
        self
    }
    
    /// Set default unit ID
    pub fn default_unit_id(mut self, unit_id: u8) -> Self {
        self.config.default_unit_id = unit_id;
        self
    }
    
    /// Enable/disable frame validation
    pub fn frame_validation(mut self, enabled: bool) -> Self {
        self.config.validate_frames = enabled;
        self
    }
    
    /// Enable/disable safety controls
    pub fn safety_controls(mut self, enabled: bool) -> Self {
        self.config.safety_controls = enabled;
        self
    }
    
    /// Set security policy
    pub fn security_policy(mut self, policy: SecurityPolicy) -> Self {
        self.security_policy = Some(policy);
        self
    }
    
    /// Build the adapter
    pub fn build(self) -> ModbusAdapter {
        let mut adapter = ModbusAdapter::with_config(self.config);
        
        if let Some(policy) = self.security_policy {
            adapter = adapter.with_security_policy(policy);
        }
        
        adapter
    }
}

impl Default for ModbusAdapterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use protocol::{ModbusPdu, FunctionCode};
    
    #[tokio::test]
    async fn test_adapter_creation() {
        let adapter = ModbusAdapter::new();
        assert_eq!(adapter.name(), "modbus");
    }
    
    #[tokio::test]
    async fn test_adapter_builder() {
        let adapter = ModbusAdapterBuilder::new()
            .protocol(ModbusProtocol::Rtu)
            .security(false)
            .max_frame_size(512)
            .safety_controls(true)
            .build();
        
        let stats = adapter.get_stats();
        assert_eq!(stats.protocol, ModbusProtocol::Rtu);
        assert!(!stats.security_enabled);
        assert!(stats.safety_controls_enabled);
        assert_eq!(stats.max_frame_size, 512);
    }
    
    #[tokio::test]
    async fn test_tcp_frame_processing() {
        let adapter = ModbusAdapter::new();
        
        // Create a valid Modbus TCP frame
        let frame_data = vec![
            0x00, 0x01, // Transaction ID
            0x00, 0x00, // Protocol ID
            0x00, 0x06, // Length
            0x01,       // Unit ID
            0x03,       // Function code (Read Holding Registers)
            0x00, 0x00, // Starting address
            0x00, 0x02, // Quantity
        ];
        
        let input = Bytes::from(frame_data);
        let result = adapter.to_upstream(input).await;
        
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_security_validation() {
        let mut policy = SecurityPolicy::default();
        policy.blocked_registers.push(RegisterRange::new(0, 99, "System registers".to_string()));
        
        let adapter = ModbusAdapterBuilder::new()
            .security(true)
            .security_policy(policy)
            .build();
        
        // Create a frame that tries to access blocked registers
        let frame_data = vec![
            0x00, 0x01, // Transaction ID
            0x00, 0x00, // Protocol ID
            0x00, 0x06, // Length
            0x01,       // Unit ID
            0x03,       // Function code
            0x00, 0x32, // Starting address (50)
            0x00, 0x0A, // Quantity (10) - overlaps with blocked range
        ];
        
        let input = Bytes::from(frame_data);
        let result = adapter.to_upstream(input).await;
        
        // Should be rejected by security validation
        assert!(result.is_err());
    }
    
    #[test]
    fn test_config_defaults() {
        let config = ModbusConfig::default();
        assert_eq!(config.protocol, ModbusProtocol::Tcp);
        assert!(config.enable_security);
        assert!(config.validate_crc);
        assert_eq!(config.max_frame_size, 260);
        assert_eq!(config.default_unit_id, 1);
    }
}
