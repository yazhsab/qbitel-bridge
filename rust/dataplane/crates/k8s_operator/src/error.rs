//! Error types for the QBITEL Bridge Kubernetes operator

use thiserror::Error;

/// Operator error types
#[derive(Error, Debug)]
pub enum OperatorError {
    #[error("Kubernetes API error: {0}")]
    KubernetesError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    #[error("Temporary failure: {0}")]
    TemporaryFailure(String),
    
    #[error("Permanent failure: {0}")]
    PermanentFailure(String),
    
    #[error("Resource not found: {0}")]
    ResourceNotFound(String),
    
    #[error("Invalid resource specification: {0}")]
    InvalidResourceSpec(String),
    
    #[error("Unsupported provider: {0}")]
    UnsupportedProvider(String),
    
    #[error("Reconciliation failed: {0}")]
    ReconciliationFailed(String),
    
    #[error("Metrics error: {0}")]
    MetricsError(String),
    
    #[error("Health check failed: {0}")]
    HealthCheckFailed(String),
    
    #[error("Network error: {0}")]
    NetworkError(String),
    
    #[error("Authentication error: {0}")]
    AuthenticationError(String),
    
    #[error("Authorization error: {0}")]
    AuthorizationError(String),
    
    #[error("Validation error: {0}")]
    ValidationError(String),
    
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
    
    #[error("YAML error: {0}")]
    YamlError(#[from] serde_yaml::Error),
    
    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),
    
    #[error("Unknown error: {0}")]
    Unknown(String),
}

/// Result type alias for operator operations
pub type OperatorResult<T> = Result<T, OperatorError>;

impl OperatorError {
    /// Check if the error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            OperatorError::TemporaryFailure(_)
                | OperatorError::NetworkError(_)
                | OperatorError::KubernetesError(_)
                | OperatorError::HttpError(_)
        )
    }
    
    /// Check if the error should trigger a fast retry
    pub fn should_fast_retry(&self) -> bool {
        matches!(
            self,
            OperatorError::NetworkError(_) | OperatorError::HttpError(_)
        )
    }
    
    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            OperatorError::PermanentFailure(_)
            | OperatorError::AuthenticationError(_)
            | OperatorError::AuthorizationError(_) => ErrorSeverity::Critical,
            
            OperatorError::KubernetesError(_)
            | OperatorError::ReconciliationFailed(_)
            | OperatorError::InvalidResourceSpec(_) => ErrorSeverity::High,
            
            OperatorError::TemporaryFailure(_)
            | OperatorError::NetworkError(_)
            | OperatorError::MetricsError(_) => ErrorSeverity::Medium,
            
            OperatorError::ConfigurationError(_)
            | OperatorError::ValidationError(_)
            | OperatorError::SerializationError(_) => ErrorSeverity::Low,
            
            _ => ErrorSeverity::Medium,
        }
    }
    
    /// Get error category for metrics
    pub fn category(&self) -> &'static str {
        match self {
            OperatorError::KubernetesError(_) => "kubernetes",
            OperatorError::SerializationError(_) => "serialization",
            OperatorError::ConfigurationError(_) => "configuration",
            OperatorError::TemporaryFailure(_) => "temporary",
            OperatorError::PermanentFailure(_) => "permanent",
            OperatorError::ResourceNotFound(_) => "resource_not_found",
            OperatorError::InvalidResourceSpec(_) => "invalid_spec",
            OperatorError::UnsupportedProvider(_) => "unsupported_provider",
            OperatorError::ReconciliationFailed(_) => "reconciliation",
            OperatorError::MetricsError(_) => "metrics",
            OperatorError::HealthCheckFailed(_) => "health_check",
            OperatorError::NetworkError(_) => "network",
            OperatorError::AuthenticationError(_) => "authentication",
            OperatorError::AuthorizationError(_) => "authorization",
            OperatorError::ValidationError(_) => "validation",
            OperatorError::IoError(_) => "io",
            OperatorError::JsonError(_) => "json",
            OperatorError::YamlError(_) => "yaml",
            OperatorError::HttpError(_) => "http",
            OperatorError::Unknown(_) => "unknown",
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

impl ErrorSeverity {
    pub fn as_str(&self) -> &'static str {
        match self {
            ErrorSeverity::Low => "low",
            ErrorSeverity::Medium => "medium", 
            ErrorSeverity::High => "high",
            ErrorSeverity::Critical => "critical",
        }
    }
}

/// Convert Kubernetes errors to OperatorError
impl From<kube::Error> for OperatorError {
    fn from(err: kube::Error) -> Self {
        match err {
            kube::Error::Api(api_err) => {
                match api_err.code {
                    404 => OperatorError::ResourceNotFound(format!("Resource not found: {}", api_err.message)),
                    401 => OperatorError::AuthenticationError(format!("Authentication failed: {}", api_err.message)),
                    403 => OperatorError::AuthorizationError(format!("Authorization failed: {}", api_err.message)),
                    409 => OperatorError::TemporaryFailure(format!("Resource conflict: {}", api_err.message)),
                    422 => OperatorError::InvalidResourceSpec(format!("Invalid resource: {}", api_err.message)),
                    500..=599 => OperatorError::TemporaryFailure(format!("Server error: {}", api_err.message)),
                    _ => OperatorError::KubernetesError(format!("API error {}: {}", api_err.code, api_err.message)),
                }
            }
            kube::Error::HttpError(http_err) => OperatorError::NetworkError(format!("HTTP error: {}", http_err)),
            kube::Error::SerdeError(serde_err) => OperatorError::SerializationError(format!("Serialization error: {}", serde_err)),
            kube::Error::Discovery(discovery_err) => OperatorError::KubernetesError(format!("Discovery error: {}", discovery_err)),
            kube::Error::Config(config_err) => OperatorError::ConfigurationError(format!("Kubeconfig error: {}", config_err)),
            _ => OperatorError::KubernetesError(format!("Kubernetes error: {}", err)),
        }
    }
}

/// Helper macro for creating operator errors
#[macro_export]
macro_rules! operator_error {
    ($variant:ident, $msg:expr) => {
        $crate::error::OperatorError::$variant($msg.to_string())
    };
    ($variant:ident, $fmt:expr, $($args:tt)*) => {
        $crate::error::OperatorError::$variant(format!($fmt, $($args)*))
    };
}

/// Helper macro for creating retryable errors
#[macro_export]
macro_rules! retryable_error {
    ($msg:expr) => {
        $crate::error::OperatorError::TemporaryFailure($msg.to_string())
    };
    ($fmt:expr, $($args:tt)*) => {
        $crate::error::OperatorError::TemporaryFailure(format!($fmt, $($args)*))
    };
}

/// Helper macro for creating permanent errors
#[macro_export]
macro_rules! permanent_error {
    ($msg:expr) => {
        $crate::error::OperatorError::PermanentFailure($msg.to_string())
    };
    ($fmt:expr, $($args:tt)*) => {
        $crate::error::OperatorError::PermanentFailure(format!($fmt, $($args)*))
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_severity() {
        let err = OperatorError::PermanentFailure("test".to_string());
        assert_eq!(err.severity(), ErrorSeverity::Critical);
        
        let err = OperatorError::TemporaryFailure("test".to_string());
        assert_eq!(err.severity(), ErrorSeverity::Medium);
    }
    
    #[test]
    fn test_error_retryable() {
        let err = OperatorError::TemporaryFailure("test".to_string());
        assert!(err.is_retryable());
        
        let err = OperatorError::PermanentFailure("test".to_string());
        assert!(!err.is_retryable());
    }
    
    #[test]
    fn test_error_category() {
        let err = OperatorError::KubernetesError("test".to_string());
        assert_eq!(err.category(), "kubernetes");
        
        let err = OperatorError::NetworkError("test".to_string());
        assert_eq!(err.category(), "network");
    }
    
    #[test]
    fn test_error_macros() {
        let err = operator_error!(ConfigurationError, "test message");
        assert!(matches!(err, OperatorError::ConfigurationError(_)));
        
        let err = retryable_error!("temporary issue");
        assert!(matches!(err, OperatorError::TemporaryFailure(_)));
        assert!(err.is_retryable());
        
        let err = permanent_error!("permanent issue");
        assert!(matches!(err, OperatorError::PermanentFailure(_)));
        assert!(!err.is_retryable());
    }
}