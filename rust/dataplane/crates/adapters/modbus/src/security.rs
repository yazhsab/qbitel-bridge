use crate::protocol::{ModbusTcpFrame, ModbusRequest, FunctionCode, ExceptionCode};
use std::collections::{HashMap, HashSet};
use thiserror::Error;
use tracing::{debug, warn, error, info};
use std::time::{Duration, Instant};

#[derive(Debug, Error)]
pub enum SecurityError {
    #[error("access denied: {0}")]
    AccessDenied(String),
    #[error("rate limit exceeded: {0}")]
    RateLimitExceeded(String),
    #[error("invalid operation: {0}")]
    InvalidOperation(String),
    #[error("safety violation: {0}")]
    SafetyViolation(String),
    #[error("authentication failed: {0}")]
    AuthenticationFailed(String),
}

/// Security policy for Modbus operations
#[derive(Debug, Clone)]
pub struct SecurityPolicy {
    /// Allow read operations
    pub allow_reads: bool,
    /// Allow write operations
    pub allow_writes: bool,
    /// Allowed function codes
    pub allowed_functions: HashSet<FunctionCode>,
    /// Blocked register ranges
    pub blocked_registers: Vec<RegisterRange>,
    /// Read-only register ranges
    pub readonly_registers: Vec<RegisterRange>,
    /// Critical register ranges (require special authorization)
    pub critical_registers: Vec<RegisterRange>,
    /// Rate limiting configuration
    pub rate_limits: RateLimitConfig,
    /// Enable audit logging
    pub audit_logging: bool,
}

impl Default for SecurityPolicy {
    fn default() -> Self {
        let mut allowed_functions = HashSet::new();
        allowed_functions.insert(FunctionCode::ReadHoldingRegisters);
        allowed_functions.insert(FunctionCode::ReadInputRegisters);
        allowed_functions.insert(FunctionCode::ReadCoils);
        allowed_functions.insert(FunctionCode::ReadDiscreteInputs);
        
        Self {
            allow_reads: true,
            allow_writes: false, // Default to read-only for safety
            allowed_functions,
            blocked_registers: Vec::new(),
            readonly_registers: Vec::new(),
            critical_registers: Vec::new(),
            rate_limits: RateLimitConfig::default(),
            audit_logging: true,
        }
    }
}

/// Register range definition
#[derive(Debug, Clone, PartialEq)]
pub struct RegisterRange {
    /// Starting register address
    pub start: u16,
    /// Ending register address (inclusive)
    pub end: u16,
    /// Description of the register range
    pub description: String,
}

impl RegisterRange {
    /// Create a new register range
    pub fn new(start: u16, end: u16, description: String) -> Self {
        Self { start, end, description }
    }
    
    /// Check if an address is within this range
    pub fn contains(&self, address: u16) -> bool {
        address >= self.start && address <= self.end
    }
    
    /// Check if a range overlaps with this range
    pub fn overlaps(&self, start: u16, count: u16) -> bool {
        let end = start.saturating_add(count.saturating_sub(1));
        !(end < self.start || start > self.end)
    }
}

/// Rate limiting configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum requests per second per client
    pub max_requests_per_second: u32,
    /// Maximum write operations per minute per client
    pub max_writes_per_minute: u32,
    /// Time window for rate limiting
    pub time_window: Duration,
    /// Enable rate limiting
    pub enabled: bool,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_requests_per_second: 100,
            max_writes_per_minute: 10,
            time_window: Duration::from_secs(60),
            enabled: true,
        }
    }
}

/// Client tracking for rate limiting
#[derive(Debug)]
struct ClientTracker {
    /// Request timestamps
    request_times: Vec<Instant>,
    /// Write operation timestamps
    write_times: Vec<Instant>,
    /// Last activity time
    last_activity: Instant,
}

impl ClientTracker {
    fn new() -> Self {
        Self {
            request_times: Vec::new(),
            write_times: Vec::new(),
            last_activity: Instant::now(),
        }
    }
    
    fn cleanup_old_entries(&mut self, window: Duration) {
        let cutoff = Instant::now() - window;
        self.request_times.retain(|&time| time > cutoff);
        self.write_times.retain(|&time| time > cutoff);
    }
    
    fn add_request(&mut self) {
        self.request_times.push(Instant::now());
        self.last_activity = Instant::now();
    }
    
    fn add_write(&mut self) {
        self.write_times.push(Instant::now());
        self.last_activity = Instant::now();
    }
    
    fn is_expired(&self, timeout: Duration) -> bool {
        self.last_activity.elapsed() > timeout
    }
}

/// Modbus security validator
pub struct ModbusSecurityValidator {
    /// Security policy
    policy: SecurityPolicy,
    /// Client tracking for rate limiting
    clients: HashMap<String, ClientTracker>,
    /// Authorized clients
    authorized_clients: HashSet<String>,
    /// Audit log entries
    audit_log: Vec<AuditEntry>,
}

/// Audit log entry
#[derive(Debug, Clone)]
pub struct AuditEntry {
    /// Timestamp
    pub timestamp: Instant,
    /// Client identifier
    pub client_id: String,
    /// Operation performed
    pub operation: String,
    /// Register address(es) accessed
    pub registers: Vec<u16>,
    /// Operation result
    pub result: AuditResult,
    /// Additional details
    pub details: String,
}

/// Audit result
#[derive(Debug, Clone, PartialEq)]
pub enum AuditResult {
    /// Operation allowed
    Allowed,
    /// Operation denied
    Denied,
    /// Operation blocked by security policy
    Blocked,
    /// Rate limit exceeded
    RateLimited,
}

impl ModbusSecurityValidator {
    /// Create a new security validator
    pub fn new(policy: SecurityPolicy) -> Self {
        Self {
            policy,
            clients: HashMap::new(),
            authorized_clients: HashSet::new(),
            audit_log: Vec::new(),
        }
    }
    
    /// Add an authorized client
    pub fn add_authorized_client(&mut self, client_id: String) {
        self.authorized_clients.insert(client_id);
    }
    
    /// Remove an authorized client
    pub fn remove_authorized_client(&mut self, client_id: &str) {
        self.authorized_clients.remove(client_id);
    }
    
    /// Validate a Modbus request
    pub fn validate_request(
        &mut self,
        client_id: &str,
        frame: &ModbusTcpFrame,
    ) -> Result<(), SecurityError> {
        // Clean up expired client entries
        self.cleanup_expired_clients();
        
        // Get or create client tracker
        let client_tracker = self.clients.entry(client_id.to_string())
            .or_insert_with(ClientTracker::new);
        
        // Clean up old rate limiting entries
        client_tracker.cleanup_old_entries(self.policy.rate_limits.time_window);
        
        // Check rate limits
        if self.policy.rate_limits.enabled {
            self.check_rate_limits(client_id, client_tracker)?;
        }
        
        // Validate function code
        if !self.policy.allowed_functions.contains(&frame.pdu.function_code) {
            self.log_audit_entry(
                client_id,
                format!("Function code {:?}", frame.pdu.function_code),
                Vec::new(),
                AuditResult::Blocked,
                "Function code not allowed".to_string(),
            );
            return Err(SecurityError::AccessDenied(format!(
                "function code {:?} not allowed", frame.pdu.function_code
            )));
        }
        
        // Parse and validate the request
        let request = self.parse_request_from_frame(frame)?;
        self.validate_register_access(client_id, &request)?;
        
        // Update client tracking
        client_tracker.add_request();
        if frame.pdu.function_code.is_write_operation() {
            client_tracker.add_write();
        }
        
        // Log successful validation
        let registers = self.extract_registers_from_request(&request);
        self.log_audit_entry(
            client_id,
            format!("Function code {:?}", frame.pdu.function_code),
            registers,
            AuditResult::Allowed,
            "Request validated successfully".to_string(),
        );
        
        info!(
            client = client_id,
            function_code = ?frame.pdu.function_code,
            unit_id = frame.unit_id,
            "Modbus request validated"
        );
        
        Ok(())
    }
    
    /// Check rate limits for a client
    fn check_rate_limits(
        &self,
        client_id: &str,
        client_tracker: &ClientTracker,
    ) -> Result<(), SecurityError> {
        // Check requests per second
        let recent_requests = client_tracker.request_times.iter()
            .filter(|&&time| time.elapsed() < Duration::from_secs(1))
            .count();
        
        if recent_requests >= self.policy.rate_limits.max_requests_per_second as usize {
            return Err(SecurityError::RateLimitExceeded(format!(
                "client {} exceeded {} requests per second", 
                client_id, 
                self.policy.rate_limits.max_requests_per_second
            )));
        }
        
        // Check writes per minute
        let recent_writes = client_tracker.write_times.iter()
            .filter(|&&time| time.elapsed() < Duration::from_secs(60))
            .count();
        
        if recent_writes >= self.policy.rate_limits.max_writes_per_minute as usize {
            return Err(SecurityError::RateLimitExceeded(format!(
                "client {} exceeded {} writes per minute", 
                client_id, 
                self.policy.rate_limits.max_writes_per_minute
            )));
        }
        
        Ok(())
    }
    
    /// Parse request from frame (simplified)
    fn parse_request_from_frame(&self, frame: &ModbusTcpFrame) -> Result<ModbusRequest, SecurityError> {
        match frame.pdu.function_code {
            FunctionCode::ReadHoldingRegisters => {
                if frame.pdu.data.len() >= 4 {
                    let start = u16::from_be_bytes([frame.pdu.data[0], frame.pdu.data[1]]);
                    let count = u16::from_be_bytes([frame.pdu.data[2], frame.pdu.data[3]]);
                    Ok(ModbusRequest::ReadHoldingRegisters {
                        starting_address: start,
                        quantity: count,
                    })
                } else {
                    Err(SecurityError::InvalidOperation("insufficient data for read holding registers".to_string()))
                }
            }
            FunctionCode::WriteSingleRegister => {
                if frame.pdu.data.len() >= 4 {
                    let address = u16::from_be_bytes([frame.pdu.data[0], frame.pdu.data[1]]);
                    let value = u16::from_be_bytes([frame.pdu.data[2], frame.pdu.data[3]]);
                    Ok(ModbusRequest::WriteSingleRegister { address, value })
                } else {
                    Err(SecurityError::InvalidOperation("insufficient data for write single register".to_string()))
                }
            }
            _ => Err(SecurityError::InvalidOperation(format!(
                "unsupported function code for security validation: {:?}", 
                frame.pdu.function_code
            ))),
        }
    }
    
    /// Validate register access based on security policy
    fn validate_register_access(
        &mut self,
        client_id: &str,
        request: &ModbusRequest,
    ) -> Result<(), SecurityError> {
        match request {
            ModbusRequest::ReadHoldingRegisters { starting_address, quantity } |
            ModbusRequest::ReadInputRegisters { starting_address, quantity } => {
                self.validate_read_access(client_id, *starting_address, *quantity)
            }
            ModbusRequest::WriteSingleRegister { address, .. } => {
                self.validate_write_access(client_id, *address, 1)
            }
            ModbusRequest::WriteMultipleRegisters { starting_address, values } => {
                self.validate_write_access(client_id, *starting_address, values.len() as u16)
            }
            _ => Ok(()), // Other request types not implemented
        }
    }
    
    /// Validate read access to registers
    fn validate_read_access(
        &mut self,
        client_id: &str,
        start_address: u16,
        count: u16,
    ) -> Result<(), SecurityError> {
        if !self.policy.allow_reads {
            return Err(SecurityError::AccessDenied("read operations not allowed".to_string()));
        }
        
        // Check blocked registers
        for blocked_range in &self.policy.blocked_registers {
            if blocked_range.overlaps(start_address, count) {
                self.log_audit_entry(
                    client_id,
                    "Read attempt".to_string(),
                    (start_address..start_address + count).collect(),
                    AuditResult::Blocked,
                    format!("Blocked range: {}", blocked_range.description),
                );
                return Err(SecurityError::AccessDenied(format!(
                    "access to blocked register range: {}", blocked_range.description
                )));
            }
        }
        
        // Check critical registers (may require special authorization)
        for critical_range in &self.policy.critical_registers {
            if critical_range.overlaps(start_address, count) {
                if !self.authorized_clients.contains(client_id) {
                    self.log_audit_entry(
                        client_id,
                        "Critical read attempt".to_string(),
                        (start_address..start_address + count).collect(),
                        AuditResult::Denied,
                        format!("Unauthorized access to critical range: {}", critical_range.description),
                    );
                    return Err(SecurityError::AccessDenied(format!(
                        "unauthorized access to critical register range: {}", critical_range.description
                    )));
                }
            }
        }
        
        Ok(())
    }
    
    /// Validate write access to registers
    fn validate_write_access(
        &mut self,
        client_id: &str,
        start_address: u16,
        count: u16,
    ) -> Result<(), SecurityError> {
        if !self.policy.allow_writes {
            return Err(SecurityError::AccessDenied("write operations not allowed".to_string()));
        }
        
        // Check blocked registers
        for blocked_range in &self.policy.blocked_registers {
            if blocked_range.overlaps(start_address, count) {
                self.log_audit_entry(
                    client_id,
                    "Write attempt".to_string(),
                    (start_address..start_address + count).collect(),
                    AuditResult::Blocked,
                    format!("Blocked range: {}", blocked_range.description),
                );
                return Err(SecurityError::AccessDenied(format!(
                    "access to blocked register range: {}", blocked_range.description
                )));
            }
        }
        
        // Check read-only registers
        for readonly_range in &self.policy.readonly_registers {
            if readonly_range.overlaps(start_address, count) {
                self.log_audit_entry(
                    client_id,
                    "Write attempt to read-only".to_string(),
                    (start_address..start_address + count).collect(),
                    AuditResult::Blocked,
                    format!("Read-only range: {}", readonly_range.description),
                );
                return Err(SecurityError::AccessDenied(format!(
                    "write to read-only register range: {}", readonly_range.description
                )));
            }
        }
        
        // Check critical registers
        for critical_range in &self.policy.critical_registers {
            if critical_range.overlaps(start_address, count) {
                if !self.authorized_clients.contains(client_id) {
                    self.log_audit_entry(
                        client_id,
                        "Critical write attempt".to_string(),
                        (start_address..start_address + count).collect(),
                        AuditResult::Denied,
                        format!("Unauthorized write to critical range: {}", critical_range.description),
                    );
                    return Err(SecurityError::AccessDenied(format!(
                        "unauthorized write to critical register range: {}", critical_range.description
                    )));
                }
            }
        }
        
        Ok(())
    }
    
    /// Extract register addresses from request
    fn extract_registers_from_request(&self, request: &ModbusRequest) -> Vec<u16> {
        match request {
            ModbusRequest::ReadHoldingRegisters { starting_address, quantity } |
            ModbusRequest::ReadInputRegisters { starting_address, quantity } => {
                (*starting_address..*starting_address + *quantity).collect()
            }
            ModbusRequest::WriteSingleRegister { address, .. } => {
                vec![*address]
            }
            ModbusRequest::WriteMultipleRegisters { starting_address, values } => {
                (*starting_address..*starting_address + values.len() as u16).collect()
            }
            _ => Vec::new(),
        }
    }
    
    /// Log audit entry
    fn log_audit_entry(
        &mut self,
        client_id: &str,
        operation: String,
        registers: Vec<u16>,
        result: AuditResult,
        details: String,
    ) {
        if self.policy.audit_logging {
            let entry = AuditEntry {
                timestamp: Instant::now(),
                client_id: client_id.to_string(),
                operation,
                registers,
                result: result.clone(),
                details: details.clone(),
            };
            
            self.audit_log.push(entry);
            
            // Log to tracing system
            match result {
                AuditResult::Allowed => {
                    info!(client = client_id, details = details, "Modbus operation allowed");
                }
                AuditResult::Denied | AuditResult::Blocked => {
                    warn!(client = client_id, details = details, "Modbus operation denied");
                }
                AuditResult::RateLimited => {
                    warn!(client = client_id, details = details, "Modbus operation rate limited");
                }
            }
        }
    }
    
    /// Clean up expired client entries
    fn cleanup_expired_clients(&mut self) {
        let timeout = Duration::from_secs(3600); // 1 hour
        self.clients.retain(|_, tracker| !tracker.is_expired(timeout));
    }
    
    /// Get audit log entries
    pub fn get_audit_log(&self) -> &[AuditEntry] {
        &self.audit_log
    }
    
    /// Clear audit log
    pub fn clear_audit_log(&mut self) {
        self.audit_log.clear();
    }
    
    /// Get security statistics
    pub fn get_stats(&self) -> SecurityStats {
        let total_clients = self.clients.len();
        let authorized_clients = self.authorized_clients.len();
        let audit_entries = self.audit_log.len();
        
        let (allowed, denied, blocked, rate_limited) = self.audit_log.iter()
            .fold((0, 0, 0, 0), |(a, d, b, r), entry| {
                match entry.result {
                    AuditResult::Allowed => (a + 1, d, b, r),
                    AuditResult::Denied => (a, d + 1, b, r),
                    AuditResult::Blocked => (a, d, b + 1, r),
                    AuditResult::RateLimited => (a, d, b, r + 1),
                }
            });
        
        SecurityStats {
            total_clients,
            authorized_clients,
            audit_entries,
            operations_allowed: allowed,
            operations_denied: denied,
            operations_blocked: blocked,
            operations_rate_limited: rate_limited,
        }
    }
}

/// Security statistics
#[derive(Debug, Clone)]
pub struct SecurityStats {
    pub total_clients: usize,
    pub authorized_clients: usize,
    pub audit_entries: usize,
    pub operations_allowed: usize,
    pub operations_denied: usize,
    pub operations_blocked: usize,
    pub operations_rate_limited: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{ModbusPdu, FunctionCode};
    
    #[test]
    fn test_register_range() {
        let range = RegisterRange::new(100, 199, "Test range".to_string());
        
        assert!(range.contains(150));
        assert!(!range.contains(50));
        assert!(!range.contains(250));
        
        assert!(range.overlaps(150, 10));
        assert!(!range.overlaps(50, 10));
        assert!(range.overlaps(95, 10)); // Overlaps at the start
    }
    
    #[test]
    fn test_security_policy_default() {
        let policy = SecurityPolicy::default();
        
        assert!(policy.allow_reads);
        assert!(!policy.allow_writes);
        assert!(policy.allowed_functions.contains(&FunctionCode::ReadHoldingRegisters));
        assert!(!policy.allowed_functions.contains(&FunctionCode::WriteSingleRegister));
    }
    
    #[test]
    fn test_security_validator() {
        let mut policy = SecurityPolicy::default();
        policy.blocked_registers.push(RegisterRange::new(0, 99, "System registers".to_string()));
        
        let mut validator = ModbusSecurityValidator::new(policy);
        
        // Create a test frame
        let frame = ModbusTcpFrame {
            transaction_id: 1,
            protocol_id: 0,
            length: 6,
            unit_id: 1,
            pdu: ModbusPdu {
                function_code: FunctionCode::ReadHoldingRegisters,
                data: vec![0x00, 0x32, 0x00, 0x0A], // Read from address 50, count 10
            },
        };
        
        // Should be blocked because it overlaps with blocked range
        let result = validator.validate_request("test_client", &frame);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_rate_limiting() {
        let mut policy = SecurityPolicy::default();
        policy.rate_limits.max_requests_per_second = 2;
        
        let mut validator = ModbusSecurityValidator::new(policy);
        
        let frame = ModbusTcpFrame {
            transaction_id: 1,
            protocol_id: 0,
            length: 6,
            unit_id: 1,
            pdu: ModbusPdu {
                function_code: FunctionCode::ReadHoldingRegisters,
                data: vec![0x01, 0x00, 0x00, 0x01], // Read from address 256, count 1
            },
        };
        
        // First two requests should succeed
        assert!(validator.validate_request("test_client", &frame).is_ok());
        assert!(validator.validate_request("test_client", &frame).is_ok());
        
        // Third request should be rate limited
        let result = validator.validate_request("test_client", &frame);
        assert!(result.is_err());
    }
}