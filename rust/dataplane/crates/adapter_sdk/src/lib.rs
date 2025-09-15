use async_trait::async_trait;
use bytes::Bytes;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AdapterError {
    #[error("io error: {0}")]
    Io(String),
    #[error("protocol error: {0}")]
    Protocol(String),
}

#[async_trait]
pub trait L7Adapter: Send + Sync {
    /// Accept bytes from client, return bytes to send to upstream (normalized/encoded).
    async fn to_upstream(&self, input: Bytes) -> Result<Bytes, AdapterError>;
    /// Accept bytes from upstream, return bytes to send back to client (decoded/mapped).
    async fn to_client(&self, input: Bytes) -> Result<Bytes, AdapterError>;
    /// Optional name for metrics/logs.
    fn name(&self) -> &'static str;
}
