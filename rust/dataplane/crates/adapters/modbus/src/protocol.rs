use bytes::{Bytes, BytesMut, Buf, BufMut};
use std::collections::HashMap;
use thiserror::Error;
use tracing::{debug, warn, error, info};

#[derive(Debug, Error)]
pub enum ModbusError {
    #[error("invalid frame: {0}")]
    InvalidFrame(String),
    #[error("invalid function code: {0}")]
    InvalidFunctionCode(u8),
    #[error("invalid data: {0}")]
    InvalidData(String),
    #[error("CRC error: expected {expected:04X}, got {actual:04X}")]
    CrcError { expected: u16, actual: u16 },
    #[error("insufficient data: expected {expected}, got {actual}")]
    InsufficientData { expected: usize, actual: usize },
    #[error("register access error: {0}")]
    RegisterAccess(String),
    #[error("security violation: {0}")]
    SecurityViolation(String),
}

/// Modbus function codes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FunctionCode {
    /// Read Coils (0x01)
    ReadCoils = 0x01,
    /// Read Discrete Inputs (0x02)
    ReadDiscreteInputs = 0x02,
    /// Read Holding Registers (0x03)
    ReadHoldingRegisters = 0x03,
    /// Read Input Registers (0x04)
    ReadInputRegisters = 0x04,
    /// Write Single Coil (0x05)
    WriteSingleCoil = 0x05,
    /// Write Single Register (0x06)
    WriteSingleRegister = 0x06,
    /// Write Multiple Coils (0x0F)
    WriteMultipleCoils = 0x0F,
    /// Write Multiple Registers (0x10)
    WriteMultipleRegisters = 0x10,
    /// Read/Write Multiple Registers (0x17)
    ReadWriteMultipleRegisters = 0x17,
    /// Mask Write Register (0x16)
    MaskWriteRegister = 0x16,
    /// Read FIFO Queue (0x18)
    ReadFifoQueue = 0x18,
    /// Read Device Identification (0x2B)
    ReadDeviceIdentification = 0x2B,
}

impl FunctionCode {
    /// Convert from byte value
    pub fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0x01 => Some(FunctionCode::ReadCoils),
            0x02 => Some(FunctionCode::ReadDiscreteInputs),
            0x03 => Some(FunctionCode::ReadHoldingRegisters),
            0x04 => Some(FunctionCode::ReadInputRegisters),
            0x05 => Some(FunctionCode::WriteSingleCoil),
            0x06 => Some(FunctionCode::WriteSingleRegister),
            0x0F => Some(FunctionCode::WriteMultipleCoils),
            0x10 => Some(FunctionCode::WriteMultipleRegisters),
            0x16 => Some(FunctionCode::MaskWriteRegister),
            0x17 => Some(FunctionCode::ReadWriteMultipleRegisters),
            0x18 => Some(FunctionCode::ReadFifoQueue),
            0x2B => Some(FunctionCode::ReadDeviceIdentification),
            _ => None,
        }
    }
    
    /// Convert to byte value
    pub fn to_byte(self) -> u8 {
        self as u8
    }
    
    /// Check if function code is a read operation
    pub fn is_read_operation(self) -> bool {
        matches!(self, 
            FunctionCode::ReadCoils |
            FunctionCode::ReadDiscreteInputs |
            FunctionCode::ReadHoldingRegisters |
            FunctionCode::ReadInputRegisters |
            FunctionCode::ReadFifoQueue |
            FunctionCode::ReadDeviceIdentification
        )
    }
    
    /// Check if function code is a write operation
    pub fn is_write_operation(self) -> bool {
        matches!(self,
            FunctionCode::WriteSingleCoil |
            FunctionCode::WriteSingleRegister |
            FunctionCode::WriteMultipleCoils |
            FunctionCode::WriteMultipleRegisters |
            FunctionCode::MaskWriteRegister |
            FunctionCode::ReadWriteMultipleRegisters
        )
    }
}

/// Modbus exception codes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExceptionCode {
    /// Illegal Function (0x01)
    IllegalFunction = 0x01,
    /// Illegal Data Address (0x02)
    IllegalDataAddress = 0x02,
    /// Illegal Data Value (0x03)
    IllegalDataValue = 0x03,
    /// Slave Device Failure (0x04)
    SlaveDeviceFailure = 0x04,
    /// Acknowledge (0x05)
    Acknowledge = 0x05,
    /// Slave Device Busy (0x06)
    SlaveDeviceBusy = 0x06,
    /// Memory Parity Error (0x08)
    MemoryParityError = 0x08,
    /// Gateway Path Unavailable (0x0A)
    GatewayPathUnavailable = 0x0A,
    /// Gateway Target Device Failed to Respond (0x0B)
    GatewayTargetDeviceFailedToRespond = 0x0B,
}

impl ExceptionCode {
    /// Convert from byte value
    pub fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0x01 => Some(ExceptionCode::IllegalFunction),
            0x02 => Some(ExceptionCode::IllegalDataAddress),
            0x03 => Some(ExceptionCode::IllegalDataValue),
            0x04 => Some(ExceptionCode::SlaveDeviceFailure),
            0x05 => Some(ExceptionCode::Acknowledge),
            0x06 => Some(ExceptionCode::SlaveDeviceBusy),
            0x08 => Some(ExceptionCode::MemoryParityError),
            0x0A => Some(ExceptionCode::GatewayPathUnavailable),
            0x0B => Some(ExceptionCode::GatewayTargetDeviceFailedToRespond),
            _ => None,
        }
    }
    
    /// Convert to byte value
    pub fn to_byte(self) -> u8 {
        self as u8
    }
}

/// Modbus TCP/IP Application Data Unit (ADU)
#[derive(Debug, Clone)]
pub struct ModbusTcpFrame {
    /// Transaction identifier
    pub transaction_id: u16,
    /// Protocol identifier (always 0 for Modbus)
    pub protocol_id: u16,
    /// Length field
    pub length: u16,
    /// Unit identifier (slave address)
    pub unit_id: u8,
    /// Protocol Data Unit
    pub pdu: ModbusPdu,
}

/// Modbus Protocol Data Unit (PDU)
#[derive(Debug, Clone)]
pub struct ModbusPdu {
    /// Function code
    pub function_code: FunctionCode,
    /// Data payload
    pub data: Vec<u8>,
}

/// Modbus RTU frame (for serial communication)
#[derive(Debug, Clone)]
pub struct ModbusRtuFrame {
    /// Slave address
    pub slave_address: u8,
    /// Protocol Data Unit
    pub pdu: ModbusPdu,
    /// CRC checksum
    pub crc: u16,
}

/// Modbus request data structures
#[derive(Debug, Clone)]
pub enum ModbusRequest {
    /// Read coils request
    ReadCoils {
        starting_address: u16,
        quantity: u16,
    },
    /// Read discrete inputs request
    ReadDiscreteInputs {
        starting_address: u16,
        quantity: u16,
    },
    /// Read holding registers request
    ReadHoldingRegisters {
        starting_address: u16,
        quantity: u16,
    },
    /// Read input registers request
    ReadInputRegisters {
        starting_address: u16,
        quantity: u16,
    },
    /// Write single coil request
    WriteSingleCoil {
        address: u16,
        value: bool,
    },
    /// Write single register request
    WriteSingleRegister {
        address: u16,
        value: u16,
    },
    /// Write multiple coils request
    WriteMultipleCoils {
        starting_address: u16,
        values: Vec<bool>,
    },
    /// Write multiple registers request
    WriteMultipleRegisters {
        starting_address: u16,
        values: Vec<u16>,
    },
}

/// Modbus response data structures
#[derive(Debug, Clone)]
pub enum ModbusResponse {
    /// Read coils response
    ReadCoils {
        values: Vec<bool>,
    },
    /// Read discrete inputs response
    ReadDiscreteInputs {
        values: Vec<bool>,
    },
    /// Read holding registers response
    ReadHoldingRegisters {
        values: Vec<u16>,
    },
    /// Read input registers response
    ReadInputRegisters {
        values: Vec<u16>,
    },
    /// Write single coil response
    WriteSingleCoil {
        address: u16,
        value: bool,
    },
    /// Write single register response
    WriteSingleRegister {
        address: u16,
        value: u16,
    },
    /// Write multiple coils response
    WriteMultipleCoils {
        starting_address: u16,
        quantity: u16,
    },
    /// Write multiple registers response
    WriteMultipleRegisters {
        starting_address: u16,
        quantity: u16,
    },
    /// Exception response
    Exception {
        function_code: u8,
        exception_code: ExceptionCode,
    },
}

/// Modbus frame parser and validator
pub struct ModbusFrameParser {
    /// Enable CRC validation for RTU frames
    validate_crc: bool,
    /// Maximum frame size
    max_frame_size: usize,
}

impl ModbusFrameParser {
    /// Create a new frame parser
    pub fn new() -> Self {
        Self {
            validate_crc: true,
            max_frame_size: 260, // Standard Modbus maximum
        }
    }
    
    /// Parse Modbus TCP frame
    pub fn parse_tcp_frame(&self, data: &[u8]) -> Result<ModbusTcpFrame, ModbusError> {
        if data.len() < 8 {
            return Err(ModbusError::InsufficientData {
                expected: 8,
                actual: data.len(),
            });
        }
        
        if data.len() > self.max_frame_size {
            return Err(ModbusError::InvalidFrame(format!(
                "frame too large: {} bytes", data.len()
            )));
        }
        
        let transaction_id = u16::from_be_bytes([data[0], data[1]]);
        let protocol_id = u16::from_be_bytes([data[2], data[3]]);
        let length = u16::from_be_bytes([data[4], data[5]]);
        let unit_id = data[6];
        
        // Validate protocol ID
        if protocol_id != 0 {
            return Err(ModbusError::InvalidFrame(format!(
                "invalid protocol ID: {}", protocol_id
            )));
        }
        
        // Validate length field
        if length as usize != data.len() - 6 {
            return Err(ModbusError::InvalidFrame(format!(
                "length mismatch: expected {}, got {}", length, data.len() - 6
            )));
        }
        
        // Parse PDU
        let pdu_data = &data[7..];
        let pdu = self.parse_pdu(pdu_data)?;
        
        Ok(ModbusTcpFrame {
            transaction_id,
            protocol_id,
            length,
            unit_id,
            pdu,
        })
    }
    
    /// Parse Modbus RTU frame
    pub fn parse_rtu_frame(&self, data: &[u8]) -> Result<ModbusRtuFrame, ModbusError> {
        if data.len() < 4 {
            return Err(ModbusError::InsufficientData {
                expected: 4,
                actual: data.len(),
            });
        }
        
        let slave_address = data[0];
        let crc_offset = data.len() - 2;
        let crc = u16::from_le_bytes([data[crc_offset], data[crc_offset + 1]]);
        
        // Validate CRC if enabled
        if self.validate_crc {
            let calculated_crc = self.calculate_crc(&data[..crc_offset]);
            if calculated_crc != crc {
                return Err(ModbusError::CrcError {
                    expected: calculated_crc,
                    actual: crc,
                });
            }
        }
        
        // Parse PDU
        let pdu_data = &data[1..crc_offset];
        let pdu = self.parse_pdu(pdu_data)?;
        
        Ok(ModbusRtuFrame {
            slave_address,
            pdu,
            crc,
        })
    }
    
    /// Parse Protocol Data Unit (PDU)
    fn parse_pdu(&self, data: &[u8]) -> Result<ModbusPdu, ModbusError> {
        if data.is_empty() {
            return Err(ModbusError::InsufficientData {
                expected: 1,
                actual: 0,
            });
        }
        
        let function_code_byte = data[0];
        
        // Check for exception response
        if function_code_byte & 0x80 != 0 {
            let original_function = function_code_byte & 0x7F;
            if data.len() < 2 {
                return Err(ModbusError::InsufficientData {
                    expected: 2,
                    actual: data.len(),
                });
            }
            
            let exception_code = ExceptionCode::from_byte(data[1])
                .ok_or_else(|| ModbusError::InvalidData(format!("invalid exception code: {}", data[1])))?;
            
            return Ok(ModbusPdu {
                function_code: FunctionCode::from_byte(original_function)
                    .ok_or_else(|| ModbusError::InvalidFunctionCode(original_function))?,
                data: vec![exception_code.to_byte()],
            });
        }
        
        let function_code = FunctionCode::from_byte(function_code_byte)
            .ok_or_else(|| ModbusError::InvalidFunctionCode(function_code_byte))?;
        
        let pdu_data = data[1..].to_vec();
        
        // Validate PDU data based on function code
        self.validate_pdu_data(function_code, &pdu_data)?;
        
        Ok(ModbusPdu {
            function_code,
            data: pdu_data,
        })
    }
    
    /// Validate PDU data for specific function codes
    fn validate_pdu_data(&self, function_code: FunctionCode, data: &[u8]) -> Result<(), ModbusError> {
        match function_code {
            FunctionCode::ReadCoils |
            FunctionCode::ReadDiscreteInputs |
            FunctionCode::ReadHoldingRegisters |
            FunctionCode::ReadInputRegisters => {
                if data.len() != 4 {
                    return Err(ModbusError::InvalidData(format!(
                        "read request requires 4 bytes, got {}", data.len()
                    )));
                }
                
                let quantity = u16::from_be_bytes([data[2], data[3]]);
                let max_quantity = match function_code {
                    FunctionCode::ReadCoils | FunctionCode::ReadDiscreteInputs => 2000,
                    _ => 125,
                };
                
                if quantity == 0 || quantity > max_quantity {
                    return Err(ModbusError::InvalidData(format!(
                        "invalid quantity: {} (max: {})", quantity, max_quantity
                    )));
                }
            }
            FunctionCode::WriteSingleCoil => {
                if data.len() != 4 {
                    return Err(ModbusError::InvalidData(format!(
                        "write single coil requires 4 bytes, got {}", data.len()
                    )));
                }
                
                let value = u16::from_be_bytes([data[2], data[3]]);
                if value != 0x0000 && value != 0xFF00 {
                    return Err(ModbusError::InvalidData(format!(
                        "invalid coil value: 0x{:04X}", value
                    )));
                }
            }
            FunctionCode::WriteSingleRegister => {
                if data.len() != 4 {
                    return Err(ModbusError::InvalidData(format!(
                        "write single register requires 4 bytes, got {}", data.len()
                    )));
                }
            }
            FunctionCode::WriteMultipleCoils => {
                if data.len() < 5 {
                    return Err(ModbusError::InvalidData(format!(
                        "write multiple coils requires at least 5 bytes, got {}", data.len()
                    )));
                }
                
                let quantity = u16::from_be_bytes([data[2], data[3]]);
                let byte_count = data[4] as usize;
                let expected_bytes = (quantity + 7) / 8;
                
                if byte_count != expected_bytes as usize {
                    return Err(ModbusError::InvalidData(format!(
                        "byte count mismatch: expected {}, got {}", expected_bytes, byte_count
                    )));
                }
                
                if data.len() != 5 + byte_count {
                    return Err(ModbusError::InvalidData(format!(
                        "data length mismatch: expected {}, got {}", 5 + byte_count, data.len()
                    )));
                }
            }
            FunctionCode::WriteMultipleRegisters => {
                if data.len() < 5 {
                    return Err(ModbusError::InvalidData(format!(
                        "write multiple registers requires at least 5 bytes, got {}", data.len()
                    )));
                }
                
                let quantity = u16::from_be_bytes([data[2], data[3]]);
                let byte_count = data[4] as usize;
                let expected_bytes = quantity * 2;
                
                if byte_count != expected_bytes as usize {
                    return Err(ModbusError::InvalidData(format!(
                        "byte count mismatch: expected {}, got {}", expected_bytes, byte_count
                    )));
                }
                
                if data.len() != 5 + byte_count {
                    return Err(ModbusError::InvalidData(format!(
                        "data length mismatch: expected {}, got {}", 5 + byte_count, data.len()
                    )));
                }
            }
            _ => {
                // Other function codes - basic validation
                debug!(function_code = ?function_code, data_len = data.len(), "validating PDU data");
            }
        }
        
        Ok(())
    }
    
    /// Calculate CRC-16 for RTU frames
    fn calculate_crc(&self, data: &[u8]) -> u16 {
        let mut crc: u16 = 0xFFFF;
        
        for &byte in data {
            crc ^= byte as u16;
            
            for _ in 0..8 {
                if crc & 0x0001 != 0 {
                    crc >>= 1;
                    crc ^= 0xA001;
                } else {
                    crc >>= 1;
                }
            }
        }
        
        crc
    }
    
    /// Serialize TCP frame to bytes
    pub fn serialize_tcp_frame(&self, frame: &ModbusTcpFrame) -> Vec<u8> {
        let mut result = Vec::new();
        
        result.extend_from_slice(&frame.transaction_id.to_be_bytes());
        result.extend_from_slice(&frame.protocol_id.to_be_bytes());
        result.extend_from_slice(&frame.length.to_be_bytes());
        result.push(frame.unit_id);
        result.push(frame.pdu.function_code.to_byte());
        result.extend_from_slice(&frame.pdu.data);
        
        result
    }
    
    /// Serialize RTU frame to bytes
    pub fn serialize_rtu_frame(&self, frame: &ModbusRtuFrame) -> Vec<u8> {
        let mut result = Vec::new();
        
        result.push(frame.slave_address);
        result.push(frame.pdu.function_code.to_byte());
        result.extend_from_slice(&frame.pdu.data);
        
        let crc = self.calculate_crc(&result);
        result.extend_from_slice(&crc.to_le_bytes());
        
        result
    }
    
    /// Parse request from PDU
    pub fn parse_request(&self, pdu: &ModbusPdu) -> Result<ModbusRequest, ModbusError> {
        match pdu.function_code {
            FunctionCode::ReadCoils => {
                if pdu.data.len() != 4 {
                    return Err(ModbusError::InvalidData("read coils requires 4 bytes".to_string()));
                }
                Ok(ModbusRequest::ReadCoils {
                    starting_address: u16::from_be_bytes([pdu.data[0], pdu.data[1]]),
                    quantity: u16::from_be_bytes([pdu.data[2], pdu.data[3]]),
                })
            }
            FunctionCode::ReadHoldingRegisters => {
                if pdu.data.len() != 4 {
                    return Err(ModbusError::InvalidData("read holding registers requires 4 bytes".to_string()));
                }
                Ok(ModbusRequest::ReadHoldingRegisters {
                    starting_address: u16::from_be_bytes([pdu.data[0], pdu.data[1]]),
                    quantity: u16::from_be_bytes([pdu.data[2], pdu.data[3]]),
                })
            }
            FunctionCode::WriteSingleRegister => {
                if pdu.data.len() != 4 {
                    return Err(ModbusError::InvalidData("write single register requires 4 bytes".to_string()));
                }
                Ok(ModbusRequest::WriteSingleRegister {
                    address: u16::from_be_bytes([pdu.data[0], pdu.data[1]]),
                    value: u16::from_be_bytes([pdu.data[2], pdu.data[3]]),
                })
            }
            _ => Err(ModbusError::InvalidFunctionCode(pdu.function_code.to_byte())),
        }
    }
    
    /// Create exception response
    pub fn create_exception_response(
        &self,
        function_code: FunctionCode,
        exception_code: ExceptionCode,
    ) -> ModbusPdu {
        ModbusPdu {
            function_code,
            data: vec![exception_code.to_byte()],
        }
    }
}

impl Default for ModbusFrameParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_function_code_conversion() {
        assert_eq!(FunctionCode::from_byte(0x03), Some(FunctionCode::ReadHoldingRegisters));
        assert_eq!(FunctionCode::ReadHoldingRegisters.to_byte(), 0x03);
        assert_eq!(FunctionCode::from_byte(0xFF), None);
    }
    
    #[test]
    fn test_tcp_frame_parsing() {
        let parser = ModbusFrameParser::new();
        
        // Read holding registers request
        let data = [
            0x00, 0x01, // Transaction ID
            0x00, 0x00, // Protocol ID
            0x00, 0x06, // Length
            0x01,       // Unit ID
            0x03,       // Function code
            0x00, 0x00, // Starting address
            0x00, 0x02, // Quantity
        ];
        
        let frame = parser.parse_tcp_frame(&data).unwrap();
        assert_eq!(frame.transaction_id, 1);
        assert_eq!(frame.protocol_id, 0);
        assert_eq!(frame.length, 6);
        assert_eq!(frame.unit_id, 1);
        assert_eq!(frame.pdu.function_code, FunctionCode::ReadHoldingRegisters);
    }
    
    #[test]
    fn test_crc_calculation() {
        let parser = ModbusFrameParser::new();
        let data = [0x01, 0x03, 0x00, 0x00, 0x00, 0x02];
        let crc = parser.calculate_crc(&data);
        
        // Expected CRC for this data
        assert_eq!(crc, 0xC40B);
    }
    
    #[test]
    fn test_rtu_frame_parsing() {
        let parser = ModbusFrameParser::new();
        
        // Read holding registers request with CRC
        let data = [
            0x01,       // Slave address
            0x03,       // Function code
            0x00, 0x00, // Starting address
            0x00, 0x02, // Quantity
            0x0B, 0xC4, // CRC (little-endian)
        ];
        
        let frame = parser.parse_rtu_frame(&data).unwrap();
        assert_eq!(frame.slave_address, 1);
        assert_eq!(frame.pdu.function_code, FunctionCode::ReadHoldingRegisters);
        assert_eq!(frame.crc, 0xC40B);
    }
    
    #[test]
    fn test_invalid_frame() {
        let parser = ModbusFrameParser::new();
        
        // Too short frame
        let data = [0x00, 0x01];
        let result = parser.parse_tcp_frame(&data);
        assert!(result.is_err());
        
        // Invalid protocol ID
        let data = [
            0x00, 0x01, // Transaction ID
            0x00, 0x01, // Invalid Protocol ID
            0x00, 0x06, // Length
            0x01,       // Unit ID
            0x03,       // Function code
            0x00, 0x00, // Starting address
            0x00, 0x02, // Quantity
        ];
        let result = parser.parse_tcp_frame(&data);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_request_parsing() {
        let parser = ModbusFrameParser::new();
        let pdu = ModbusPdu {
            function_code: FunctionCode::ReadHoldingRegisters,
            data: vec![0x00, 0x00, 0x00, 0x02],
        };
        
        let request = parser.parse_request(&pdu).unwrap();
        match request {
            ModbusRequest::ReadHoldingRegisters { starting_address, quantity } => {
                assert_eq!(starting_address, 0);
                assert_eq!(quantity, 2);
            }
            _ => panic!("unexpected request type"),
        }
    }
}