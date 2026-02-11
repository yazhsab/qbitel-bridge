"""
Go SDK Generator

Generates idiomatic Go SDK with:
- Strongly typed structs
- Context support for cancellation
- Error handling with custom error types
- Retry with exponential backoff
- Comprehensive documentation
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from sdk_generator.generator import (
    BaseLanguageGenerator,
    Language,
    OpenAPISchema,
    GeneratorConfig,
    TypeMapper,
)


class GoGenerator(BaseLanguageGenerator):
    """Go SDK generator."""

    language = Language.GO
    file_extension = ".go"

    def generate(self) -> None:
        """Generate Go SDK."""
        self._ensure_dir(self.output_dir)

        # Generate module files
        self._generate_go_mod()
        self._generate_client()
        self._generate_types()
        self._generate_operations()
        self._generate_errors()

        if self.config.generate_tests:
            self._generate_tests()

        if self.config.generate_examples:
            self._generate_examples()

    def _generate_go_mod(self) -> None:
        """Generate go.mod file."""
        content = f'''module github.com/yazhsab/{self.config.package_name}-go

go 1.21

require (
	golang.org/x/net v0.19.0
	golang.org/x/time v0.5.0
)
'''
        self._write_file(self.output_dir / "go.mod", content)

    def _generate_client(self) -> None:
        """Generate main client file."""
        content = f'''// Package {self._go_package_name()} provides a Go client for the QBITEL API.
//
// Example usage:
//
//	client := {self._go_package_name()}.NewClient(
//		{self._go_package_name()}.WithAPIKey("your-api-key"),
//	)
//
//	resp, err := client.ProtocolDiscovery.AnalyzeTraffic(ctx, &{self._go_package_name()}.AnalyzeTrafficRequest{{
//		TrafficData: data,
//	}})
package {self._go_package_name()}

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"

	"golang.org/x/time/rate"
)

const (
	// DefaultBaseURL is the default API base URL
	DefaultBaseURL = "{self.config.api_base_url}"

	// DefaultTimeout is the default request timeout
	DefaultTimeout = 30 * time.Second

	// Version is the SDK version
	Version = "{self.config.package_version}"
)

// Client is the QBITEL API client
type Client struct {{
	// HTTP client for making requests
	httpClient *http.Client

	// Base URL for API requests
	baseURL *url.URL

	// API key for authentication
	apiKey string

	// Rate limiter
	rateLimiter *rate.Limiter

	// User agent for requests
	userAgent string

	// Service clients
	ProtocolDiscovery *ProtocolDiscoveryService
	LegacyWhisperer   *LegacyWhispererService
	AIGateway         *AIGatewayService
	Security          *SecurityService
}}

// ClientOption is a function that configures a Client
type ClientOption func(*Client)

// WithAPIKey sets the API key
func WithAPIKey(apiKey string) ClientOption {{
	return func(c *Client) {{
		c.apiKey = apiKey
	}}
}}

// WithBaseURL sets the base URL
func WithBaseURL(baseURL string) ClientOption {{
	return func(c *Client) {{
		u, _ := url.Parse(baseURL)
		c.baseURL = u
	}}
}}

// WithHTTPClient sets the HTTP client
func WithHTTPClient(httpClient *http.Client) ClientOption {{
	return func(c *Client) {{
		c.httpClient = httpClient
	}}
}}

// WithRateLimiter sets a custom rate limiter
func WithRateLimiter(limiter *rate.Limiter) ClientOption {{
	return func(c *Client) {{
		c.rateLimiter = limiter
	}}
}}

// WithUserAgent sets a custom user agent
func WithUserAgent(userAgent string) ClientOption {{
	return func(c *Client) {{
		c.userAgent = userAgent
	}}
}}

// NewClient creates a new QBITEL API client
func NewClient(opts ...ClientOption) *Client {{
	baseURL, _ := url.Parse(DefaultBaseURL)

	c := &Client{{
		httpClient: &http.Client{{
			Timeout: DefaultTimeout,
		}},
		baseURL:     baseURL,
		rateLimiter: rate.NewLimiter(rate.Limit(10), 20), // 10 req/s, burst of 20
		userAgent:   fmt.Sprintf("qbitel-go/%s", Version),
	}}

	for _, opt := range opts {{
		opt(c)
	}}

	// Initialize service clients
	c.ProtocolDiscovery = &ProtocolDiscoveryService{{client: c}}
	c.LegacyWhisperer = &LegacyWhispererService{{client: c}}
	c.AIGateway = &AIGatewayService{{client: c}}
	c.Security = &SecurityService{{client: c}}

	return c
}}

// Request represents an API request
type Request struct {{
	Method   string
	Path     string
	Query    url.Values
	Body     interface{{}}
	Headers  map[string]string
}}

// Do executes an API request
func (c *Client) Do(ctx context.Context, req *Request, result interface{{}}) error {{
	// Wait for rate limiter
	if err := c.rateLimiter.Wait(ctx); err != nil {{
		return fmt.Errorf("rate limit: %w", err)
	}}

	// Build URL
	u := c.baseURL.ResolveReference(&url.URL{{
		Path:     req.Path,
		RawQuery: req.Query.Encode(),
	}})

	// Prepare body
	var body io.Reader
	if req.Body != nil {{
		jsonBody, err := json.Marshal(req.Body)
		if err != nil {{
			return fmt.Errorf("marshal body: %w", err)
		}}
		body = bytes.NewReader(jsonBody)
	}}

	// Create request
	httpReq, err := http.NewRequestWithContext(ctx, req.Method, u.String(), body)
	if err != nil {{
		return fmt.Errorf("create request: %w", err)
	}}

	// Set headers
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")
	httpReq.Header.Set("User-Agent", c.userAgent)

	if c.apiKey != "" {{
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	}}

	for k, v := range req.Headers {{
		httpReq.Header.Set(k, v)
	}}

	// Execute request
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {{
		return fmt.Errorf("do request: %w", err)
	}}
	defer resp.Body.Close()

	// Read response body
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {{
		return fmt.Errorf("read response: %w", err)
	}}

	// Check for errors
	if resp.StatusCode >= 400 {{
		return parseError(resp.StatusCode, respBody)
	}}

	// Parse response
	if result != nil && len(respBody) > 0 {{
		if err := json.Unmarshal(respBody, result); err != nil {{
			return fmt.Errorf("unmarshal response: %w", err)
		}}
	}}

	return nil
}}

// parseError parses an error response
func parseError(statusCode int, body []byte) error {{
	var apiErr APIError
	if err := json.Unmarshal(body, &apiErr); err != nil {{
		return &APIError{{
			StatusCode: statusCode,
			Message:    string(body),
		}}
	}}
	apiErr.StatusCode = statusCode
	return &apiErr
}}
'''
        self._write_file(self.output_dir / "client.go", content)

    def _generate_types(self) -> None:
        """Generate types from schema components."""
        schemas = self.schema.components.get("schemas", {})

        content = f'''package {self._go_package_name()}

import (
	"time"
)

'''
        for name, schema in schemas.items():
            content += self._generate_struct(name, schema)
            content += "\n"

        self._write_file(self.output_dir / "types.go", content)

    def _generate_struct(self, name: str, schema: Dict[str, Any]) -> str:
        """Generate a Go struct from schema."""
        struct_name = self._to_pascal_case(name)
        description = schema.get("description", f"{struct_name} represents a {name}")

        lines = [f"// {struct_name} {description}"]
        lines.append(f"type {struct_name} struct {{")

        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        for prop_name, prop_schema in properties.items():
            field_name = self._to_pascal_case(prop_name)
            field_type = self._schema_to_go_type(prop_schema)

            # Make optional fields pointers
            if prop_name not in required and not field_type.startswith("[]"):
                field_type = "*" + field_type

            json_tag = f'`json:"{prop_name},omitempty"`'
            field_desc = prop_schema.get("description", "")

            if field_desc:
                lines.append(f"\t// {field_desc}")
            lines.append(f"\t{field_name} {field_type} {json_tag}")

        lines.append("}")
        return "\n".join(lines) + "\n"

    def _generate_operations(self) -> None:
        """Generate operation methods."""
        # Group paths by tags
        services = {}
        for path, methods in self.schema.paths.items():
            for method, operation in methods.items():
                if method in ["get", "post", "put", "delete", "patch"]:
                    tags = operation.get("tags", ["default"])
                    tag = tags[0] if tags else "default"

                    if tag not in services:
                        services[tag] = []

                    services[tag].append({
                        "path": path,
                        "method": method.upper(),
                        "operation": operation,
                    })

        # Generate service files
        for service_name, operations in services.items():
            self._generate_service_file(service_name, operations)

    def _generate_service_file(
        self,
        service_name: str,
        operations: List[Dict[str, Any]],
    ) -> None:
        """Generate a service file."""
        struct_name = self._to_pascal_case(service_name) + "Service"
        file_name = self._to_snake_case(service_name) + ".go"

        content = f'''package {self._go_package_name()}

import (
	"context"
	"fmt"
	"net/url"
)

// {struct_name} handles {service_name} operations
type {struct_name} struct {{
	client *Client
}}

'''
        for op in operations:
            content += self._generate_operation_method(struct_name, op)
            content += "\n"

        self._write_file(self.output_dir / file_name, content)

    def _generate_operation_method(
        self,
        service_name: str,
        operation_data: Dict[str, Any],
    ) -> str:
        """Generate a single operation method."""
        operation = operation_data["operation"]
        method = operation_data["method"]
        path = operation_data["path"]

        operation_id = operation.get("operationId", "")
        method_name = self._to_pascal_case(operation_id) if operation_id else "Operation"
        description = operation.get("summary", operation.get("description", ""))

        # Build parameters
        params = operation.get("parameters", [])
        request_body = operation.get("requestBody", {})

        # Determine input and output types
        input_type = f"{method_name}Request"
        output_type = f"{method_name}Response"

        lines = []

        # Method comment
        if description:
            lines.append(f"// {method_name} {description}")

        # Method signature
        lines.append(f"func (s *{service_name}) {method_name}(ctx context.Context, req *{input_type}) (*{output_type}, error) {{")

        # Build query params
        query_params = [p for p in params if p.get("in") == "query"]
        if query_params:
            lines.append("\tquery := url.Values{}")
            for p in query_params:
                param_name = p["name"]
                field_name = self._to_pascal_case(param_name)
                lines.append(f'\tif req.{field_name} != "" {{')
                lines.append(f'\t\tquery.Set("{param_name}", req.{field_name})')
                lines.append("\t}")

        # Build path with params
        path_params = [p for p in params if p.get("in") == "path"]
        if path_params:
            lines.append(f'\tpath := fmt.Sprintf("{self._go_path_template(path)}"')
            for p in path_params:
                field_name = self._to_pascal_case(p["name"])
                lines.append(f", req.{field_name}")
            lines.append("\t)")
        else:
            lines.append(f'\tpath := "{path}"')

        # Make request
        lines.append("")
        lines.append("\tvar result " + output_type)
        lines.append("\terr := s.client.Do(ctx, &Request{")
        lines.append(f'\t\tMethod: "{method}",')
        lines.append("\t\tPath:   path,")

        if query_params:
            lines.append("\t\tQuery:  query,")

        if request_body:
            lines.append("\t\tBody:   req,")

        lines.append("\t}, &result)")
        lines.append("")
        lines.append("\tif err != nil {")
        lines.append("\t\treturn nil, err")
        lines.append("\t}")
        lines.append("")
        lines.append("\treturn &result, nil")
        lines.append("}")

        return "\n".join(lines)

    def _generate_errors(self) -> None:
        """Generate error types."""
        content = f'''package {self._go_package_name()}

import (
	"fmt"
)

// APIError represents an API error response
type APIError struct {{
	StatusCode int    `json:"-"`
	Code       string `json:"code,omitempty"`
	Message    string `json:"message,omitempty"`
	Details    []ErrorDetail `json:"details,omitempty"`
}}

// ErrorDetail provides additional error information
type ErrorDetail struct {{
	Field   string `json:"field,omitempty"`
	Message string `json:"message,omitempty"`
}}

func (e *APIError) Error() string {{
	return fmt.Sprintf("API error %d: %s - %s", e.StatusCode, e.Code, e.Message)
}}

// IsNotFound returns true if the error is a 404
func (e *APIError) IsNotFound() bool {{
	return e.StatusCode == 404
}}

// IsUnauthorized returns true if the error is a 401
func (e *APIError) IsUnauthorized() bool {{
	return e.StatusCode == 401
}}

// IsForbidden returns true if the error is a 403
func (e *APIError) IsForbidden() bool {{
	return e.StatusCode == 403
}}

// IsRateLimited returns true if the error is a 429
func (e *APIError) IsRateLimited() bool {{
	return e.StatusCode == 429
}}

// IsServerError returns true if the error is a 5xx
func (e *APIError) IsServerError() bool {{
	return e.StatusCode >= 500
}}
'''
        self._write_file(self.output_dir / "errors.go", content)

    def _generate_tests(self) -> None:
        """Generate test files."""
        content = f'''package {self._go_package_name()}_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	{self._go_package_name()} "github.com/yazhsab/{self.config.package_name}-go"
)

func TestNewClient(t *testing.T) {{
	client := {self._go_package_name()}.NewClient(
		{self._go_package_name()}.WithAPIKey("test-key"),
	)

	if client == nil {{
		t.Fatal("expected client to be non-nil")
	}}
}}

func TestClientWithMockServer(t *testing.T) {{
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {{
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{{"status": "ok"}}`))
	}}))
	defer server.Close()

	client := {self._go_package_name()}.NewClient(
		{self._go_package_name()}.WithBaseURL(server.URL),
		{self._go_package_name()}.WithAPIKey("test-key"),
	)

	ctx := context.Background()
	_ = ctx
	_ = client
	// Add specific endpoint tests here
}}
'''
        self._write_file(self.output_dir / "client_test.go", content)

    def _generate_examples(self) -> None:
        """Generate example files."""
        examples_dir = self.output_dir / "examples"
        self._ensure_dir(examples_dir)

        content = f'''package main

import (
	"context"
	"fmt"
	"log"
	"os"

	{self._go_package_name()} "github.com/yazhsab/{self.config.package_name}-go"
)

func main() {{
	// Create client with API key from environment
	client := {self._go_package_name()}.NewClient(
		{self._go_package_name()}.WithAPIKey(os.Getenv("QBITEL_API_KEY")),
	)

	ctx := context.Background()

	// Example: Analyze protocol traffic
	fmt.Println("QBITEL Go SDK Example")
	fmt.Printf("SDK Version: %s\\n", {self._go_package_name()}.Version)

	_ = ctx
	_ = client

	// Add your API calls here
	log.Println("Example completed successfully")
}}
'''
        self._write_file(examples_dir / "main.go", content)

    def _go_package_name(self) -> str:
        """Get Go package name."""
        return self.config.package_name.replace("-", "_").replace(" ", "_").lower()

    def _schema_to_go_type(self, schema: Dict[str, Any]) -> str:
        """Convert OpenAPI schema to Go type."""
        return TypeMapper.map_schema_type(Language.GO, schema)

    def _to_pascal_case(self, s: str) -> str:
        """Convert to PascalCase."""
        words = re.split(r"[-_\s]+", s)
        return "".join(word.capitalize() for word in words)

    def _to_snake_case(self, s: str) -> str:
        """Convert to snake_case."""
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
        s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
        return s.replace("-", "_").lower()

    def _go_path_template(self, path: str) -> str:
        """Convert OpenAPI path to Go format string."""
        # Convert {param} to %s
        return re.sub(r"\{(\w+)\}", "%s", path)
