"""
Rust SDK Generator

Generates idiomatic Rust SDK with:
- Strongly typed structs with serde
- Async/await with tokio
- Error handling with thiserror
- Builder pattern for requests
- Comprehensive documentation
"""

import re
from pathlib import Path
from typing import Any, Dict, List

from sdk_generator.generator import (
    BaseLanguageGenerator,
    Language,
    OpenAPISchema,
    GeneratorConfig,
    TypeMapper,
)


class RustGenerator(BaseLanguageGenerator):
    """Rust SDK generator."""

    language = Language.RUST
    file_extension = ".rs"

    def generate(self) -> None:
        """Generate Rust SDK."""
        self._ensure_dir(self.output_dir / "src")

        # Generate crate files
        self._generate_cargo_toml()
        self._generate_lib_rs()
        self._generate_client()
        self._generate_types()
        self._generate_error()
        self._generate_operations()

        if self.config.generate_tests:
            self._generate_tests()

        if self.config.generate_examples:
            self._generate_examples()

    def _generate_cargo_toml(self) -> None:
        """Generate Cargo.toml."""
        content = f'''[package]
name = "{self.config.package_name.replace("_", "-")}"
version = "{self.config.package_version}"
edition = "2021"
authors = ["{self.config.author}"]
description = "{self.config.description}"
license = "{self.config.license}"
repository = "https://github.com/yazhsab/{self.config.package_name}-rust"
documentation = "https://docs.rs/{self.config.package_name.replace("_", "-")}"
keywords = ["qbitel", "ai", "security", "sdk"]
categories = ["api-bindings", "asynchronous"]

[dependencies]
reqwest = {{ version = "0.11", features = ["json", "rustls-tls"] }}
tokio = {{ version = "1", features = ["full"] }}
serde = {{ version = "1", features = ["derive"] }}
serde_json = "1"
thiserror = "1"
async-trait = "0.1"
chrono = {{ version = "0.4", features = ["serde"] }}
uuid = {{ version = "1", features = ["v4", "serde"] }}
url = "2"
tracing = "0.1"

[dev-dependencies]
tokio-test = "0.4"
mockito = "1"
pretty_assertions = "1"

[features]
default = ["rustls-tls"]
rustls-tls = ["reqwest/rustls-tls"]
native-tls = ["reqwest/native-tls"]
'''
        self._write_file(self.output_dir / "Cargo.toml", content)

    def _generate_lib_rs(self) -> None:
        """Generate lib.rs."""
        content = f'''//! # QBITEL Rust SDK
//!
//! {self.config.description}
//!
//! ## Quick Start
//!
//! ```rust
//! use {self._rust_crate_name()}::Client;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {{
//!     let client = Client::builder()
//!         .api_key("your-api-key")
//!         .build()?;
//!
//!     // Use the client
//!     Ok(())
//! }}
//! ```

#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

pub mod client;
pub mod error;
pub mod types;
pub mod services;

pub use client::{{Client, ClientBuilder}};
pub use error::{{Error, Result}};
pub use types::*;
'''
        self._write_file(self.output_dir / "src" / "lib.rs", content)

    def _generate_client(self) -> None:
        """Generate client module."""
        content = f'''//! HTTP client for QBITEL API.

use std::sync::Arc;
use std::time::Duration;

use reqwest::header::{{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE, USER_AGENT}};
use url::Url;

use crate::error::{{Error, Result}};
use crate::services::*;

/// Default API base URL
pub const DEFAULT_BASE_URL: &str = "{self.config.api_base_url}";

/// SDK version
pub const VERSION: &str = "{self.config.package_version}";

/// QBITEL API client
#[derive(Clone)]
pub struct Client {{
    inner: Arc<ClientInner>,
}}

struct ClientInner {{
    http_client: reqwest::Client,
    base_url: Url,
    api_key: String,
}}

impl Client {{
    /// Create a new client builder
    pub fn builder() -> ClientBuilder {{
        ClientBuilder::default()
    }}

    /// Get the Protocol Discovery service
    pub fn protocol_discovery(&self) -> ProtocolDiscoveryService {{
        ProtocolDiscoveryService::new(self.clone())
    }}

    /// Get the Legacy Whisperer service
    pub fn legacy_whisperer(&self) -> LegacyWhispererService {{
        LegacyWhispererService::new(self.clone())
    }}

    /// Get the AI Gateway service
    pub fn ai_gateway(&self) -> AIGatewayService {{
        AIGatewayService::new(self.clone())
    }}

    /// Get the Security service
    pub fn security(&self) -> SecurityService {{
        SecurityService::new(self.clone())
    }}

    /// Execute a GET request
    pub async fn get<T: serde::de::DeserializeOwned>(&self, path: &str) -> Result<T> {{
        self.request(reqwest::Method::GET, path, Option::<()>::None).await
    }}

    /// Execute a POST request
    pub async fn post<T, B>(&self, path: &str, body: Option<B>) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
        B: serde::Serialize,
    {{
        self.request(reqwest::Method::POST, path, body).await
    }}

    /// Execute a PUT request
    pub async fn put<T, B>(&self, path: &str, body: Option<B>) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
        B: serde::Serialize,
    {{
        self.request(reqwest::Method::PUT, path, body).await
    }}

    /// Execute a DELETE request
    pub async fn delete<T>(&self, path: &str) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
    {{
        self.request(reqwest::Method::DELETE, path, Option::<()>::None).await
    }}

    async fn request<T, B>(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<B>,
    ) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
        B: serde::Serialize,
    {{
        let url = self.inner.base_url.join(path).map_err(Error::url)?;

        let mut request = self.inner.http_client.request(method, url);

        if let Some(body) = body {{
            request = request.json(&body);
        }}

        let response = request.send().await.map_err(Error::request)?;

        let status = response.status();
        if !status.is_success() {{
            let error_body = response.text().await.unwrap_or_default();
            return Err(Error::api(status.as_u16(), error_body));
        }}

        response.json().await.map_err(Error::deserialize)
    }}
}}

/// Client builder
#[derive(Default)]
pub struct ClientBuilder {{
    base_url: Option<String>,
    api_key: Option<String>,
    timeout: Option<Duration>,
    user_agent: Option<String>,
}}

impl ClientBuilder {{
    /// Set the API key
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {{
        self.api_key = Some(api_key.into());
        self
    }}

    /// Set the base URL
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {{
        self.base_url = Some(base_url.into());
        self
    }}

    /// Set the request timeout
    pub fn timeout(mut self, timeout: Duration) -> Self {{
        self.timeout = Some(timeout);
        self
    }}

    /// Set a custom user agent
    pub fn user_agent(mut self, user_agent: impl Into<String>) -> Self {{
        self.user_agent = Some(user_agent.into());
        self
    }}

    /// Build the client
    pub fn build(self) -> Result<Client> {{
        let api_key = self.api_key.ok_or_else(|| Error::configuration("API key is required"))?;

        let base_url = self
            .base_url
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        let base_url = Url::parse(&base_url).map_err(Error::url)?;

        let user_agent = self
            .user_agent
            .unwrap_or_else(|| format!("qbitel-rust/{{}}", VERSION));

        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {{}}", api_key)).map_err(|_| {{
                Error::configuration("Invalid API key format")
            }})?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            USER_AGENT,
            HeaderValue::from_str(&user_agent).unwrap_or_else(|_| {{
                HeaderValue::from_static("qbitel-rust")
            }}),
        );

        let http_client = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(self.timeout.unwrap_or(Duration::from_secs(30)))
            .build()
            .map_err(Error::request)?;

        Ok(Client {{
            inner: Arc::new(ClientInner {{
                http_client,
                base_url,
                api_key,
            }}),
        }})
    }}
}}
'''
        self._write_file(self.output_dir / "src" / "client.rs", content)

    def _generate_error(self) -> None:
        """Generate error module."""
        content = '''//! Error types for the QBITEL SDK.

use thiserror::Error;

/// SDK error type
#[derive(Error, Debug)]
pub enum Error {
    /// HTTP request error
    #[error("Request error: {0}")]
    Request(#[from] reqwest::Error),

    /// URL parse error
    #[error("URL error: {0}")]
    Url(#[from] url::ParseError),

    /// JSON serialization error
    #[error("Serialization error: {0}")]
    Serialize(#[source] serde_json::Error),

    /// JSON deserialization error
    #[error("Deserialization error: {0}")]
    Deserialize(#[source] serde_json::Error),

    /// API error response
    #[error("API error {status}: {message}")]
    Api {
        /// HTTP status code
        status: u16,
        /// Error message
        message: String,
    },

    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),
}

impl Error {
    /// Create a request error
    pub fn request(err: reqwest::Error) -> Self {
        Self::Request(err)
    }

    /// Create a URL error
    pub fn url(err: url::ParseError) -> Self {
        Self::Url(err)
    }

    /// Create a serialization error
    pub fn serialize(err: serde_json::Error) -> Self {
        Self::Serialize(err)
    }

    /// Create a deserialization error
    pub fn deserialize(err: reqwest::Error) -> Self {
        Self::Request(err)
    }

    /// Create an API error
    pub fn api(status: u16, message: String) -> Self {
        Self::Api { status, message }
    }

    /// Create a configuration error
    pub fn configuration(message: impl Into<String>) -> Self {
        Self::Configuration(message.into())
    }

    /// Check if this is a not found error (404)
    pub fn is_not_found(&self) -> bool {
        matches!(self, Self::Api { status: 404, .. })
    }

    /// Check if this is an unauthorized error (401)
    pub fn is_unauthorized(&self) -> bool {
        matches!(self, Self::Api { status: 401, .. })
    }

    /// Check if this is a rate limit error (429)
    pub fn is_rate_limited(&self) -> bool {
        matches!(self, Self::Api { status: 429, .. })
    }

    /// Check if this is a server error (5xx)
    pub fn is_server_error(&self) -> bool {
        matches!(self, Self::Api { status, .. } if *status >= 500)
    }
}

/// Result type alias
pub type Result<T> = std::result::Result<T, Error>;
'''
        self._write_file(self.output_dir / "src" / "error.rs", content)

    def _generate_types(self) -> None:
        """Generate types module."""
        schemas = self.schema.components.get("schemas", {})

        content = '''//! Data types for the QBITEL API.

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

'''
        for name, schema in schemas.items():
            content += self._generate_rust_struct(name, schema)
            content += "\n"

        self._write_file(self.output_dir / "src" / "types.rs", content)

    def _generate_rust_struct(self, name: str, schema: Dict[str, Any]) -> str:
        """Generate a Rust struct."""
        struct_name = self._to_pascal_case(name)
        description = schema.get("description", f"{struct_name}")

        lines = [f"/// {description}"]
        lines.append("#[derive(Debug, Clone, Serialize, Deserialize)]")
        lines.append(f"pub struct {struct_name} {{")

        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        for prop_name, prop_schema in properties.items():
            field_name = self._to_snake_case(prop_name)
            field_type = self._schema_to_rust_type(prop_schema)

            # Make optional fields Option<T>
            if prop_name not in required:
                field_type = f"Option<{field_type}>"

            field_desc = prop_schema.get("description", "")

            if field_desc:
                lines.append(f"    /// {field_desc}")

            # Add serde rename if needed
            if field_name != prop_name:
                lines.append(f'    #[serde(rename = "{prop_name}")]')

            lines.append(f"    pub {field_name}: {field_type},")

        lines.append("}")
        return "\n".join(lines) + "\n"

    def _generate_operations(self) -> None:
        """Generate service modules."""
        services_dir = self.output_dir / "src" / "services"
        self._ensure_dir(services_dir)

        # Generate mod.rs
        mod_content = '''//! API service modules.

mod protocol_discovery;
mod legacy_whisperer;
mod ai_gateway;
mod security;

pub use protocol_discovery::ProtocolDiscoveryService;
pub use legacy_whisperer::LegacyWhispererService;
pub use ai_gateway::AIGatewayService;
pub use security::SecurityService;
'''
        self._write_file(services_dir / "mod.rs", mod_content)

        # Generate service files
        services = ["protocol_discovery", "legacy_whisperer", "ai_gateway", "security"]
        for service in services:
            self._generate_service_module(services_dir, service)

    def _generate_service_module(self, services_dir: Path, service_name: str) -> None:
        """Generate a service module."""
        struct_name = self._to_pascal_case(service_name) + "Service"

        content = f'''//! {self._to_pascal_case(service_name)} service.

use crate::{{Client, Result}};

/// {self._to_pascal_case(service_name)} service client
#[derive(Clone)]
pub struct {struct_name} {{
    client: Client,
}}

impl {struct_name} {{
    /// Create a new service instance
    pub(crate) fn new(client: Client) -> Self {{
        Self {{ client }}
    }}

    // Add service methods here
}}
'''
        self._write_file(services_dir / f"{service_name}.rs", content)

    def _generate_tests(self) -> None:
        """Generate test files."""
        tests_dir = self.output_dir / "tests"
        self._ensure_dir(tests_dir)

        content = '''use qbitel::Client;

#[test]
fn test_client_builder() {
    let result = Client::builder().api_key("test-key").build();
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_client_operations() {
    // Add integration tests here
}
'''
        self._write_file(tests_dir / "client_test.rs", content)

    def _generate_examples(self) -> None:
        """Generate example files."""
        examples_dir = self.output_dir / "examples"
        self._ensure_dir(examples_dir)

        content = f'''//! Basic usage example

use {self._rust_crate_name()}::Client;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {{
    let api_key = std::env::var("QBITEL_API_KEY")
        .expect("QBITEL_API_KEY environment variable required");

    let client = Client::builder()
        .api_key(api_key)
        .build()?;

    println!("QBITEL Rust SDK Example");
    println!("SDK Version: {{}}", {self._rust_crate_name()}::client::VERSION);

    // Add your API calls here
    let _ = client;

    Ok(())
}}
'''
        self._write_file(examples_dir / "basic.rs", content)

    def _rust_crate_name(self) -> str:
        """Get Rust crate name."""
        return self.config.package_name.replace("-", "_")

    def _schema_to_rust_type(self, schema: Dict[str, Any]) -> str:
        """Convert OpenAPI schema to Rust type."""
        return TypeMapper.map_schema_type(Language.RUST, schema)

    def _to_pascal_case(self, s: str) -> str:
        """Convert to PascalCase."""
        words = re.split(r"[-_\s]+", s)
        return "".join(word.capitalize() for word in words)

    def _to_snake_case(self, s: str) -> str:
        """Convert to snake_case."""
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
        s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
        return s.replace("-", "_").lower()
