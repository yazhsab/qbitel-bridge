use thiserror::Error;

#[derive(Debug, Error)]
pub enum TlsError {
    #[error("openssl error: {0}")]
    Openssl(String),
    #[error("io error: {0}")]
    Io(String),
    #[error("provider not available: {0}")]
    Provider(String),
    #[error("policy violation: {0}")]
    Policy(String),
}

impl From<std::io::Error> for TlsError {
    fn from(e: std::io::Error) -> Self { Self::Io(e.to_string()) }
}

impl From<openssl::error::ErrorStack> for TlsError {
    fn from(e: openssl::error::ErrorStack) -> Self { Self::Openssl(e.to_string()) }
}

impl From<openssl::ssl::Error> for TlsError {
    fn from(e: openssl::ssl::Error) -> Self { Self::Openssl(e.to_string()) }
}
