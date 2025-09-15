use adapter_sdk::{L7Adapter, AdapterError};
use async_trait::async_trait;
use bytes::Bytes;
use tracing::{debug, warn, error, info};
use std::sync::{Arc, Mutex};

pub mod protocol;
pub mod datastream;

use protocol::{Tn3270StateMachine, SessionConfig, DeviceType, ProtocolState};
use datastream::{DataStreamParser, ScreenField};

/// TN3270E adapter configuration
#[derive(Debug, Clone)]
pub struct Tn3270Config {
    /// Device type to emulate
    pub device_type: DeviceType,
    /// Enable TN3270E mode (vs basic TN3270)
    pub tn3270e_mode: bool,
    /// Enable binary mode
    pub binary_mode: bool,
    /// Screen dimensions override
    pub screen_dimensions: Option<(u16, u16)>,
    /// Device name for negotiation
    pub device_name: Option<String>,
    /// Enable screen scraping
    pub enable_scraping: bool,
    /// Session timeout in seconds
    pub session_timeout: u64,
}

impl Default for Tn3270Config {
    fn default() -> Self {
        Self {
            device_type: DeviceType::Ibm3278_2,
            tn3270e_mode: true,
            binary_mode: true,
            screen_dimensions: None,
            device_name: None,
            enable_scraping: true,
            session_timeout: 300, // 5 minutes
        }
    }
}

/// TN3270E session state
#[derive(Debug)]
struct SessionState {
    /// Protocol state machine
    state_machine: Tn3270StateMachine,
    /// Data stream parser
    parser: DataStreamParser,
    /// Last activity timestamp
    last_activity: std::time::Instant,
}

impl SessionState {
    fn new(config: &Tn3270Config) -> Self {
        let session_config = SessionConfig {
            device_type: config.device_type.clone(),
            functions: vec![
                protocol::Tn3270Function::BindImage,
                protocol::Tn3270Function::DataStreamUtils,
                protocol::Tn3270Function::Responses,
            ],
            device_name: config.device_name.clone(),
            tn3270e_mode: config.tn3270e_mode,
            binary_mode: config.binary_mode,
        };
        
        let state_machine = Tn3270StateMachine::new(session_config);
        
        let (rows, cols) = config.screen_dimensions
            .unwrap_or_else(|| config.device_type.dimensions());
        let parser = DataStreamParser::new(rows, cols);
        
        Self {
            state_machine,
            parser,
            last_activity: std::time::Instant::now(),
        }
    }
    
    fn update_activity(&mut self) {
        self.last_activity = std::time::Instant::now();
    }
    
    fn is_expired(&self, timeout: std::time::Duration) -> bool {
        self.last_activity.elapsed() > timeout
    }
}

/// TN3270E adapter with protocol state machine and screen scraping
pub struct Tn3270EAdapter {
    /// Adapter configuration
    config: Tn3270Config,
    /// Session state (shared between connections)
    session: Arc<Mutex<Option<SessionState>>>,
}

impl Tn3270EAdapter {
    /// Create a new TN3270E adapter with default configuration
    pub fn new() -> Self {
        Self {
            config: Tn3270Config::default(),
            session: Arc::new(Mutex::new(None)),
        }
    }
    
    /// Create a new TN3270E adapter with custom configuration
    pub fn with_config(config: Tn3270Config) -> Self {
        Self {
            config,
            session: Arc::new(Mutex::new(None)),
        }
    }
    
    /// Process upstream data (client to mainframe)
    async fn process_upstream(&self, input: Bytes) -> Result<Bytes, AdapterError> {
        let mut session_guard = self.session.lock()
            .map_err(|e| AdapterError::ProcessingError(format!("session lock failed: {}", e)))?;
        
        // Initialize session if needed
        if session_guard.is_none() {
            *session_guard = Some(SessionState::new(&self.config));
            info!("initialized new TN3270E session");
        }
        
        let session = session_guard.as_mut().unwrap();
        session.update_activity();
        
        // Process data through protocol state machine
        let response_data = session.state_machine.process_data(&input)
            .map_err(|e| AdapterError::ProcessingError(format!("protocol processing failed: {}", e)))?;
        
        debug!(
            state = ?session.state_machine.state(),
            input_len = input.len(),
            output_len = response_data.len(),
            "processed upstream data"
        );
        
        // If we're in session established state, also parse the data stream
        if matches!(session.state_machine.state(), ProtocolState::SessionEstablished) {
            if let Err(e) = session.parser.parse(&input) {
                warn!(error = %e, "data stream parsing failed");
            }
        }
        
        Ok(Bytes::from(response_data))
    }
    
    /// Process downstream data (mainframe to client)
    async fn process_downstream(&self, input: Bytes) -> Result<Bytes, AdapterError> {
        let mut session_guard = self.session.lock()
            .map_err(|e| AdapterError::ProcessingError(format!("session lock failed: {}", e)))?;
        
        if let Some(session) = session_guard.as_mut() {
            session.update_activity();
            
            // Parse data stream for screen scraping
            if self.config.enable_scraping {
                if let Err(e) = session.parser.parse(&input) {
                    warn!(error = %e, "downstream data stream parsing failed");
                }
                
                // Extract fields for potential processing
                let fields = session.parser.extract_fields();
                if !fields.is_empty() {
                    debug!(field_count = fields.len(), "extracted screen fields");
                }
            }
            
            debug!(
                input_len = input.len(),
                "processed downstream data"
            );
        }
        
        // For downstream, typically pass through the data
        Ok(input)
    }
    
    /// Get current session state
    pub fn get_session_state(&self) -> Result<Option<ProtocolState>, AdapterError> {
        let session_guard = self.session.lock()
            .map_err(|e| AdapterError::ProcessingError(format!("session lock failed: {}", e)))?;
        
        Ok(session_guard.as_ref().map(|s| s.state_machine.state().clone()))
    }
    
    /// Extract screen fields (for screen scraping)
    pub fn extract_screen_fields(&self) -> Result<Vec<ScreenField>, AdapterError> {
        let mut session_guard = self.session.lock()
            .map_err(|e| AdapterError::ProcessingError(format!("session lock failed: {}", e)))?;
        
        if let Some(session) = session_guard.as_mut() {
            Ok(session.parser.extract_fields())
        } else {
            Ok(Vec::new())
        }
    }
    
    /// Get screen content as text
    pub fn get_screen_text(&self) -> Result<String, AdapterError> {
        let session_guard = self.session.lock()
            .map_err(|e| AdapterError::ProcessingError(format!("session lock failed: {}", e)))?;
        
        if let Some(session) = session_guard.as_ref() {
            Ok(session.parser.get_screen_text())
        } else {
            Ok(String::new())
        }
    }
    
    /// Scrape input fields from screen
    pub fn scrape_input_fields(&self) -> Result<std::collections::HashMap<u16, String>, AdapterError> {
        let mut session_guard = self.session.lock()
            .map_err(|e| AdapterError::ProcessingError(format!("session lock failed: {}", e)))?;
        
        if let Some(session) = session_guard.as_mut() {
            session.parser.scrape_input_fields()
                .map_err(|e| AdapterError::ProcessingError(format!("screen scraping failed: {}", e)))
        } else {
            Ok(std::collections::HashMap::new())
        }
    }
    
    /// Find field containing specific text
    pub fn find_field_with_text(&self, text: &str) -> Result<Option<ScreenField>, AdapterError> {
        let mut session_guard = self.session.lock()
            .map_err(|e| AdapterError::ProcessingError(format!("session lock failed: {}", e)))?;
        
        if let Some(session) = session_guard.as_mut() {
            session.parser.find_field_with_text(text)
                .map_err(|e| AdapterError::ProcessingError(format!("field search failed: {}", e)))
        } else {
            Ok(None)
        }
    }
    
    /// Terminate current session
    pub fn terminate_session(&self) -> Result<(), AdapterError> {
        let mut session_guard = self.session.lock()
            .map_err(|e| AdapterError::ProcessingError(format!("session lock failed: {}", e)))?;
        
        if let Some(session) = session_guard.as_mut() {
            session.state_machine.terminate();
            info!("terminated TN3270E session");
        }
        
        *session_guard = None;
        Ok(())
    }
    
    /// Clean up expired sessions
    pub fn cleanup_expired_sessions(&self) -> Result<(), AdapterError> {
        let mut session_guard = self.session.lock()
            .map_err(|e| AdapterError::ProcessingError(format!("session lock failed: {}", e)))?;
        
        if let Some(session) = session_guard.as_ref() {
            let timeout = std::time::Duration::from_secs(self.config.session_timeout);
            if session.is_expired(timeout) {
                info!("cleaning up expired TN3270E session");
                *session_guard = None;
            }
        }
        
        Ok(())
    }
    
    /// Get adapter statistics
    pub fn get_stats(&self) -> Result<Tn3270Stats, AdapterError> {
        let session_guard = self.session.lock()
            .map_err(|e| AdapterError::ProcessingError(format!("session lock failed: {}", e)))?;
        
        let (session_active, protocol_state, negotiated_device) = if let Some(session) = session_guard.as_ref() {
            (
                true,
                Some(session.state_machine.state().clone()),
                session.state_machine.negotiated_device().cloned(),
            )
        } else {
            (false, None, None)
        };
        
        Ok(Tn3270Stats {
            device_type: self.config.device_type.clone(),
            tn3270e_mode: self.config.tn3270e_mode,
            binary_mode: self.config.binary_mode,
            scraping_enabled: self.config.enable_scraping,
            session_timeout: self.config.session_timeout,
            session_active,
            protocol_state,
            negotiated_device,
        })
    }
}

/// TN3270E adapter statistics
#[derive(Debug, Clone)]
pub struct Tn3270Stats {
    pub device_type: DeviceType,
    pub tn3270e_mode: bool,
    pub binary_mode: bool,
    pub scraping_enabled: bool,
    pub session_timeout: u64,
    pub session_active: bool,
    pub protocol_state: Option<ProtocolState>,
    pub negotiated_device: Option<DeviceType>,
}

#[async_trait]
impl L7Adapter for Tn3270EAdapter {
    async fn to_upstream(&self, input: Bytes) -> Result<Bytes, AdapterError> {
        self.process_upstream(input).await
    }
    
    async fn to_client(&self, input: Bytes) -> Result<Bytes, AdapterError> {
        self.process_downstream(input).await
    }
    
    fn name(&self) -> &'static str { 
        "tn3270e" 
    }
}

impl Default for Tn3270EAdapter {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for TN3270E adapter configuration
pub struct Tn3270AdapterBuilder {
    config: Tn3270Config,
}

impl Tn3270AdapterBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: Tn3270Config::default(),
        }
    }
    
    /// Set device type
    pub fn device_type(mut self, device_type: DeviceType) -> Self {
        self.config.device_type = device_type;
        self
    }
    
    /// Enable/disable TN3270E mode
    pub fn tn3270e_mode(mut self, enabled: bool) -> Self {
        self.config.tn3270e_mode = enabled;
        self
    }
    
    /// Enable/disable binary mode
    pub fn binary_mode(mut self, enabled: bool) -> Self {
        self.config.binary_mode = enabled;
        self
    }
    
    /// Set screen dimensions
    pub fn screen_dimensions(mut self, rows: u16, cols: u16) -> Self {
        self.config.screen_dimensions = Some((rows, cols));
        self
    }
    
    /// Set device name
    pub fn device_name(mut self, name: String) -> Self {
        self.config.device_name = Some(name);
        self
    }
    
    /// Enable/disable screen scraping
    pub fn scraping(mut self, enabled: bool) -> Self {
        self.config.enable_scraping = enabled;
        self
    }
    
    /// Set session timeout
    pub fn session_timeout(mut self, timeout_secs: u64) -> Self {
        self.config.session_timeout = timeout_secs;
        self
    }
    
    /// Build the adapter
    pub fn build(self) -> Tn3270EAdapter {
        Tn3270EAdapter::with_config(self.config)
    }
}

impl Default for Tn3270AdapterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_adapter_creation() {
        let adapter = Tn3270EAdapter::new();
        assert_eq!(adapter.name(), "tn3270e");
    }
    
    #[tokio::test]
    async fn test_adapter_builder() {
        let adapter = Tn3270AdapterBuilder::new()
            .device_type(DeviceType::Ibm3278_4)
            .tn3270e_mode(false)
            .binary_mode(false)
            .screen_dimensions(43, 80)
            .scraping(true)
            .session_timeout(600)
            .build();
        
        let stats = adapter.get_stats().unwrap();
        assert_eq!(stats.device_type, DeviceType::Ibm3278_4);
        assert!(!stats.tn3270e_mode);
        assert!(!stats.binary_mode);
        assert!(stats.scraping_enabled);
        assert_eq!(stats.session_timeout, 600);
    }
    
    #[tokio::test]
    async fn test_session_initialization() {
        let adapter = Tn3270EAdapter::new();
        
        // Process some data to initialize session
        let test_data = Bytes::from(vec![0xFF, 0xFD, 0x18]); // DO TERMINAL-TYPE
        let result = adapter.to_upstream(test_data).await;
        
        assert!(result.is_ok());
        
        // Check that session is now active
        let state = adapter.get_session_state().unwrap();
        assert!(state.is_some());
    }
    
    #[tokio::test]
    async fn test_screen_scraping() {
        let adapter = Tn3270EAdapter::with_config(Tn3270Config {
            enable_scraping: true,
            ..Default::default()
        });
        
        // Initialize session first
        let init_data = Bytes::from(vec![0xFF, 0xFD, 0x18]);
        let _ = adapter.to_upstream(init_data).await;
        
        // Send some 3270 data stream
        let data_stream = Bytes::from(vec![
            0x11, 0x40, 0x40, // SBA to 0,0
            0x1D, 0x00,       // SF with normal attribute
            b'T', b'E', b'S', b'T'
        ]);
        let _ = adapter.to_client(data_stream).await;
        
        // Extract fields
        let fields = adapter.extract_screen_fields().unwrap();
        assert!(!fields.is_empty());
    }
    
    #[tokio::test]
    async fn test_session_termination() {
        let adapter = Tn3270EAdapter::new();
        
        // Initialize session
        let test_data = Bytes::from(vec![0xFF, 0xFD, 0x18]);
        let _ = adapter.to_upstream(test_data).await;
        
        // Verify session is active
        assert!(adapter.get_session_state().unwrap().is_some());
        
        // Terminate session
        adapter.terminate_session().unwrap();
        
        // Verify session is terminated
        assert!(adapter.get_session_state().unwrap().is_none());
    }
    
    #[test]
    fn test_config_defaults() {
        let config = Tn3270Config::default();
        assert_eq!(config.device_type, DeviceType::Ibm3278_2);
        assert!(config.tn3270e_mode);
        assert!(config.binary_mode);
        assert!(config.enable_scraping);
        assert_eq!(config.session_timeout, 300);
    }
}
