"""
C# SDK Generator

Generates idiomatic C# SDK with:
- Strongly typed classes
- Async/await with Task
- System.Text.Json serialization
- Builder pattern
- Comprehensive XML documentation
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


class CSharpGenerator(BaseLanguageGenerator):
    """C# SDK generator."""

    language = Language.CSHARP
    file_extension = ".cs"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.namespace = self.language_config.get(
            "namespace", "QbitelAI.SDK"
        )

    def generate(self) -> None:
        """Generate C# SDK."""
        src_dir = self.output_dir / "src" / self.namespace
        self._ensure_dir(src_dir)

        # Generate project files
        self._generate_csproj()
        self._generate_solution()

        # Generate source files
        self._generate_client(src_dir)
        self._generate_config(src_dir)
        self._generate_types(src_dir)
        self._generate_exceptions(src_dir)
        self._generate_services(src_dir)

        if self.config.generate_tests:
            self._generate_tests()

    def _generate_csproj(self) -> None:
        """Generate .csproj file."""
        content = f'''<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
    <LangVersion>latest</LangVersion>

    <PackageId>{self.namespace}</PackageId>
    <Version>{self.config.package_version}</Version>
    <Authors>{self.config.author}</Authors>
    <Description>{self.config.description}</Description>
    <PackageLicenseExpression>{self.config.license}</PackageLicenseExpression>
    <RepositoryUrl>https://github.com/yazhsab/{self.config.package_name}-dotnet</RepositoryUrl>
    <PackageTags>qbitel;ai;security;sdk</PackageTags>

    <GenerateDocumentationFile>true</GenerateDocumentationFile>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="System.Text.Json" Version="8.0.0" />
    <PackageReference Include="Microsoft.Extensions.Http" Version="8.0.0" />
    <PackageReference Include="Polly" Version="8.2.0" />
  </ItemGroup>

</Project>
'''
        self._write_file(
            self.output_dir / "src" / self.namespace / f"{self.namespace}.csproj",
            content
        )

    def _generate_solution(self) -> None:
        """Generate .sln file."""
        # Generate a basic solution file
        content = f'''
Microsoft Visual Studio Solution File, Format Version 12.00
# Visual Studio Version 17
Project("{{FAE04EC0-301F-11D3-BF4B-00C04F79EFBC}}") = "{self.namespace}", "src\\{self.namespace}\\{self.namespace}.csproj", "{{00000000-0000-0000-0000-000000000001}}"
EndProject
Global
    GlobalSection(SolutionConfigurationPlatforms) = preSolution
        Debug|Any CPU = Debug|Any CPU
        Release|Any CPU = Release|Any CPU
    EndGlobalSection
    GlobalSection(ProjectConfigurationPlatforms) = postSolution
        {{00000000-0000-0000-0000-000000000001}}.Debug|Any CPU.ActiveCfg = Debug|Any CPU
        {{00000000-0000-0000-0000-000000000001}}.Debug|Any CPU.Build.0 = Debug|Any CPU
        {{00000000-0000-0000-0000-000000000001}}.Release|Any CPU.ActiveCfg = Release|Any CPU
        {{00000000-0000-0000-0000-000000000001}}.Release|Any CPU.Build.0 = Release|Any CPU
    EndGlobalSection
EndGlobal
'''
        self._write_file(self.output_dir / f"{self.namespace}.sln", content)

    def _generate_client(self, src_dir: Path) -> None:
        """Generate main client class."""
        content = f'''using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

namespace {self.namespace};

/// <summary>
/// QBITEL API Client.
/// </summary>
/// <remarks>
/// Main entry point for interacting with the QBITEL API.
///
/// <example>
/// <code>
/// var client = new QbitelClientBuilder()
///     .WithApiKey("your-api-key")
///     .Build();
///
/// var response = await client.ProtocolDiscovery.AnalyzeTrafficAsync(request);
/// </code>
/// </example>
/// </remarks>
public sealed class QbitelClient : IDisposable
{{
    /// <summary>
    /// Default API base URL.
    /// </summary>
    public const string DefaultBaseUrl = "{self.config.api_base_url}";

    /// <summary>
    /// SDK version.
    /// </summary>
    public const string Version = "{self.config.package_version}";

    private readonly HttpClient _httpClient;
    private readonly JsonSerializerOptions _jsonOptions;
    private bool _disposed;

    /// <summary>
    /// Gets the Protocol Discovery service.
    /// </summary>
    public ProtocolDiscoveryService ProtocolDiscovery {{ get; }}

    /// <summary>
    /// Gets the Legacy Whisperer service.
    /// </summary>
    public LegacyWhispererService LegacyWhisperer {{ get; }}

    /// <summary>
    /// Gets the AI Gateway service.
    /// </summary>
    public AIGatewayService AIGateway {{ get; }}

    /// <summary>
    /// Gets the Security service.
    /// </summary>
    public SecurityService Security {{ get; }}

    internal QbitelClient(HttpClient httpClient)
    {{
        _httpClient = httpClient ?? throw new ArgumentNullException(nameof(httpClient));

        _jsonOptions = new JsonSerializerOptions
        {{
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            WriteIndented = false,
            DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
        }};

        // Initialize services
        ProtocolDiscovery = new ProtocolDiscoveryService(this);
        LegacyWhisperer = new LegacyWhispererService(this);
        AIGateway = new AIGatewayService(this);
        Security = new SecurityService(this);
    }}

    /// <summary>
    /// Creates a new client builder.
    /// </summary>
    /// <returns>A new QbitelClientBuilder instance.</returns>
    public static QbitelClientBuilder CreateBuilder() => new();

    /// <summary>
    /// Executes a GET request.
    /// </summary>
    public async Task<T> GetAsync<T>(string path, CancellationToken cancellationToken = default)
    {{
        return await SendAsync<T>(HttpMethod.Get, path, null, cancellationToken);
    }}

    /// <summary>
    /// Executes a POST request.
    /// </summary>
    public async Task<T> PostAsync<T, TBody>(string path, TBody body, CancellationToken cancellationToken = default)
    {{
        return await SendAsync<T>(HttpMethod.Post, path, body, cancellationToken);
    }}

    /// <summary>
    /// Executes a PUT request.
    /// </summary>
    public async Task<T> PutAsync<T, TBody>(string path, TBody body, CancellationToken cancellationToken = default)
    {{
        return await SendAsync<T>(HttpMethod.Put, path, body, cancellationToken);
    }}

    /// <summary>
    /// Executes a DELETE request.
    /// </summary>
    public async Task<T> DeleteAsync<T>(string path, CancellationToken cancellationToken = default)
    {{
        return await SendAsync<T>(HttpMethod.Delete, path, null, cancellationToken);
    }}

    private async Task<T> SendAsync<T>(
        HttpMethod method,
        string path,
        object? body,
        CancellationToken cancellationToken)
    {{
        using var request = new HttpRequestMessage(method, path);

        if (body is not null)
        {{
            var json = JsonSerializer.Serialize(body, _jsonOptions);
            request.Content = new StringContent(json, Encoding.UTF8, "application/json");
        }}

        using var response = await _httpClient.SendAsync(request, cancellationToken);

        var responseBody = await response.Content.ReadAsStringAsync(cancellationToken);

        if (!response.IsSuccessStatusCode)
        {{
            throw new QbitelApiException((int)response.StatusCode, responseBody);
        }}

        if (string.IsNullOrEmpty(responseBody))
        {{
            return default!;
        }}

        return JsonSerializer.Deserialize<T>(responseBody, _jsonOptions)!;
    }}

    /// <inheritdoc />
    public void Dispose()
    {{
        if (_disposed) return;

        _httpClient.Dispose();
        _disposed = true;
    }}
}}

/// <summary>
/// Builder for creating QbitelClient instances.
/// </summary>
public sealed class QbitelClientBuilder
{{
    private string? _apiKey;
    private string _baseUrl = QbitelClient.DefaultBaseUrl;
    private TimeSpan _timeout = TimeSpan.FromSeconds(30);
    private HttpMessageHandler? _handler;

    /// <summary>
    /// Sets the API key.
    /// </summary>
    /// <param name="apiKey">The API key.</param>
    /// <returns>This builder instance.</returns>
    public QbitelClientBuilder WithApiKey(string apiKey)
    {{
        _apiKey = apiKey ?? throw new ArgumentNullException(nameof(apiKey));
        return this;
    }}

    /// <summary>
    /// Sets the base URL.
    /// </summary>
    /// <param name="baseUrl">The base URL.</param>
    /// <returns>This builder instance.</returns>
    public QbitelClientBuilder WithBaseUrl(string baseUrl)
    {{
        _baseUrl = baseUrl ?? throw new ArgumentNullException(nameof(baseUrl));
        return this;
    }}

    /// <summary>
    /// Sets the request timeout.
    /// </summary>
    /// <param name="timeout">The timeout duration.</param>
    /// <returns>This builder instance.</returns>
    public QbitelClientBuilder WithTimeout(TimeSpan timeout)
    {{
        _timeout = timeout;
        return this;
    }}

    /// <summary>
    /// Sets a custom HTTP message handler.
    /// </summary>
    /// <param name="handler">The HTTP message handler.</param>
    /// <returns>This builder instance.</returns>
    public QbitelClientBuilder WithHandler(HttpMessageHandler handler)
    {{
        _handler = handler;
        return this;
    }}

    /// <summary>
    /// Builds the client.
    /// </summary>
    /// <returns>A new QbitelClient instance.</returns>
    /// <exception cref="InvalidOperationException">Thrown when API key is not set.</exception>
    public QbitelClient Build()
    {{
        if (string.IsNullOrEmpty(_apiKey))
        {{
            throw new InvalidOperationException("API key is required.");
        }}

        var httpClient = _handler is not null
            ? new HttpClient(_handler)
            : new HttpClient();

        httpClient.BaseAddress = new Uri(_baseUrl);
        httpClient.Timeout = _timeout;
        httpClient.DefaultRequestHeaders.Authorization =
            new AuthenticationHeaderValue("Bearer", _apiKey);
        httpClient.DefaultRequestHeaders.Accept.Add(
            new MediaTypeWithQualityHeaderValue("application/json"));
        httpClient.DefaultRequestHeaders.UserAgent.ParseAdd(
            $"qbitel-dotnet/{{QbitelClient.Version}}");

        return new QbitelClient(httpClient);
    }}
}}
'''
        self._write_file(src_dir / "QbitelClient.cs", content)

    def _generate_config(self, src_dir: Path) -> None:
        """Generate configuration class."""
        content = f'''namespace {self.namespace};

/// <summary>
/// SDK configuration constants.
/// </summary>
public static class QbitelConfig
{{
    /// <summary>
    /// Default API base URL.
    /// </summary>
    public const string DefaultBaseUrl = "{self.config.api_base_url}";

    /// <summary>
    /// API version.
    /// </summary>
    public const string ApiVersion = "{self.config.api_version}";

    /// <summary>
    /// SDK version.
    /// </summary>
    public const string SdkVersion = "{self.config.package_version}";
}}
'''
        self._write_file(src_dir / "QbitelConfig.cs", content)

    def _generate_types(self, src_dir: Path) -> None:
        """Generate type classes."""
        models_dir = src_dir / "Models"
        self._ensure_dir(models_dir)

        schemas = self.schema.components.get("schemas", {})

        for name, schema in schemas.items():
            content = self._generate_csharp_class(name, schema)
            class_name = self._to_pascal_case(name)
            self._write_file(models_dir / f"{class_name}.cs", content)

    def _generate_csharp_class(self, name: str, schema: Dict[str, Any]) -> str:
        """Generate a C# class."""
        class_name = self._to_pascal_case(name)
        description = schema.get("description", f"{class_name} model")

        lines = [f"using System.Text.Json.Serialization;"]
        lines.append("")
        lines.append(f"namespace {self.namespace}.Models;")
        lines.append("")
        lines.append(f"/// <summary>")
        lines.append(f"/// {description}")
        lines.append(f"/// </summary>")
        lines.append(f"public sealed class {class_name}")
        lines.append("{")

        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        for prop_name, prop_schema in properties.items():
            prop_name_pascal = self._to_pascal_case(prop_name)
            prop_type = self._schema_to_csharp_type(prop_schema)

            # Make optional fields nullable
            if prop_name not in required and not prop_type.endswith("?"):
                prop_type = f"{prop_type}?"

            prop_desc = prop_schema.get("description", "")
            if prop_desc:
                lines.append(f"    /// <summary>")
                lines.append(f"    /// {prop_desc}")
                lines.append(f"    /// </summary>")

            lines.append(f'    [JsonPropertyName("{prop_name}")]')
            lines.append(f"    public {prop_type} {prop_name_pascal} {{ get; set; }}")
            lines.append("")

        lines.append("}")
        return "\n".join(lines)

    def _generate_exceptions(self, src_dir: Path) -> None:
        """Generate exception classes."""
        content = f'''namespace {self.namespace};

/// <summary>
/// Base exception for QBITEL SDK errors.
/// </summary>
public class QbitelException : Exception
{{
    /// <summary>
    /// Initializes a new instance of the <see cref="QbitelException"/> class.
    /// </summary>
    public QbitelException() {{ }}

    /// <summary>
    /// Initializes a new instance with a message.
    /// </summary>
    /// <param name="message">The error message.</param>
    public QbitelException(string message) : base(message) {{ }}

    /// <summary>
    /// Initializes a new instance with a message and inner exception.
    /// </summary>
    /// <param name="message">The error message.</param>
    /// <param name="innerException">The inner exception.</param>
    public QbitelException(string message, Exception innerException)
        : base(message, innerException) {{ }}
}}

/// <summary>
/// Exception for API error responses.
/// </summary>
public class QbitelApiException : QbitelException
{{
    /// <summary>
    /// Gets the HTTP status code.
    /// </summary>
    public int StatusCode {{ get; }}

    /// <summary>
    /// Gets the response body.
    /// </summary>
    public string ResponseBody {{ get; }}

    /// <summary>
    /// Initializes a new instance of the <see cref="QbitelApiException"/> class.
    /// </summary>
    /// <param name="statusCode">The HTTP status code.</param>
    /// <param name="responseBody">The response body.</param>
    public QbitelApiException(int statusCode, string responseBody)
        : base($"API error {{statusCode}}: {{responseBody}}")
    {{
        StatusCode = statusCode;
        ResponseBody = responseBody;
    }}

    /// <summary>
    /// Returns true if this is a not found error (404).
    /// </summary>
    public bool IsNotFound => StatusCode == 404;

    /// <summary>
    /// Returns true if this is an unauthorized error (401).
    /// </summary>
    public bool IsUnauthorized => StatusCode == 401;

    /// <summary>
    /// Returns true if this is a forbidden error (403).
    /// </summary>
    public bool IsForbidden => StatusCode == 403;

    /// <summary>
    /// Returns true if this is a rate limit error (429).
    /// </summary>
    public bool IsRateLimited => StatusCode == 429;

    /// <summary>
    /// Returns true if this is a server error (5xx).
    /// </summary>
    public bool IsServerError => StatusCode >= 500;
}}
'''
        self._write_file(src_dir / "Exceptions.cs", content)

    def _generate_services(self, src_dir: Path) -> None:
        """Generate service classes."""
        services_dir = src_dir / "Services"
        self._ensure_dir(services_dir)

        services = [
            ("ProtocolDiscoveryService", "Protocol Discovery"),
            ("LegacyWhispererService", "Legacy Whisperer"),
            ("AIGatewayService", "AI Gateway"),
            ("SecurityService", "Security"),
        ]

        for class_name, description in services:
            content = f'''namespace {self.namespace};

/// <summary>
/// {description} service client.
/// </summary>
public sealed class {class_name}
{{
    private readonly QbitelClient _client;

    internal {class_name}(QbitelClient client)
    {{
        _client = client ?? throw new ArgumentNullException(nameof(client));
    }}

    // Add service methods here
}}
'''
            self._write_file(services_dir / f"{class_name}.cs", content)

    def _generate_tests(self) -> None:
        """Generate test files."""
        test_dir = self.output_dir / "tests" / f"{self.namespace}.Tests"
        self._ensure_dir(test_dir)

        # Test project file
        csproj_content = f'''<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
    <IsPackable>false</IsPackable>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="Microsoft.NET.Test.Sdk" Version="17.8.0" />
    <PackageReference Include="xunit" Version="2.6.2" />
    <PackageReference Include="xunit.runner.visualstudio" Version="2.5.4" />
    <PackageReference Include="Moq" Version="4.20.70" />
  </ItemGroup>

  <ItemGroup>
    <ProjectReference Include="..\\..\\src\\{self.namespace}\\{self.namespace}.csproj" />
  </ItemGroup>

</Project>
'''
        self._write_file(test_dir / f"{self.namespace}.Tests.csproj", csproj_content)

        # Test file
        test_content = f'''using Xunit;

namespace {self.namespace}.Tests;

public class QbitelClientTests
{{
    [Fact]
    public void Builder_RequiresApiKey()
    {{
        var exception = Assert.Throws<InvalidOperationException>(() =>
        {{
            QbitelClient.CreateBuilder().Build();
        }});

        Assert.Contains("API key", exception.Message);
    }}

    [Fact]
    public void Builder_WithApiKey_Succeeds()
    {{
        var client = QbitelClient.CreateBuilder()
            .WithApiKey("test-key")
            .Build();

        Assert.NotNull(client);
        client.Dispose();
    }}
}}
'''
        self._write_file(test_dir / "QbitelClientTests.cs", test_content)

    def _schema_to_csharp_type(self, schema: Dict[str, Any]) -> str:
        """Convert OpenAPI schema to C# type."""
        return TypeMapper.map_schema_type(Language.CSHARP, schema)

    def _to_pascal_case(self, s: str) -> str:
        """Convert to PascalCase."""
        words = re.split(r"[-_\s]+", s)
        return "".join(word.capitalize() for word in words)
