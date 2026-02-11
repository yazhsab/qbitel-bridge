"""
Java SDK Generator

Generates idiomatic Java SDK with:
- Strongly typed POJOs
- Async support with CompletableFuture
- Builder pattern
- Jackson serialization
- Comprehensive Javadoc
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


class JavaGenerator(BaseLanguageGenerator):
    """Java SDK generator."""

    language = Language.JAVA
    file_extension = ".java"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.package_name = self.language_config.get(
            "package", "com.qbitelai.sdk"
        )

    def generate(self) -> None:
        """Generate Java SDK."""
        src_dir = self.output_dir / "src" / "main" / "java"
        package_path = src_dir / self.package_name.replace(".", "/")
        self._ensure_dir(package_path)

        # Generate project files
        self._generate_pom()
        self._generate_gradle()

        # Generate source files
        self._generate_client(package_path)
        self._generate_config(package_path)
        self._generate_types(package_path)
        self._generate_exceptions(package_path)
        self._generate_services(package_path)

        if self.config.generate_tests:
            self._generate_tests()

    def _generate_pom(self) -> None:
        """Generate pom.xml."""
        content = f'''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>com.qbitelai</groupId>
    <artifactId>{self.config.package_name}-java</artifactId>
    <version>{self.config.package_version}</version>
    <packaging>jar</packaging>

    <name>QBITEL Java SDK</name>
    <description>{self.config.description}</description>
    <url>https://github.com/yazhsab/{self.config.package_name}-java</url>

    <licenses>
        <license>
            <name>{self.config.license}</name>
        </license>
    </licenses>

    <properties>
        <maven.compiler.source>17</maven.compiler.source>
        <maven.compiler.target>17</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
        <jackson.version>2.16.0</jackson.version>
        <okhttp.version>4.12.0</okhttp.version>
    </properties>

    <dependencies>
        <dependency>
            <groupId>com.squareup.okhttp3</groupId>
            <artifactId>okhttp</artifactId>
            <version>${{okhttp.version}}</version>
        </dependency>
        <dependency>
            <groupId>com.fasterxml.jackson.core</groupId>
            <artifactId>jackson-databind</artifactId>
            <version>${{jackson.version}}</version>
        </dependency>
        <dependency>
            <groupId>com.fasterxml.jackson.datatype</groupId>
            <artifactId>jackson-datatype-jsr310</artifactId>
            <version>${{jackson.version}}</version>
        </dependency>

        <!-- Test dependencies -->
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter</artifactId>
            <version>5.10.0</version>
            <scope>test</scope>
        </dependency>
        <dependency>
            <groupId>org.mockito</groupId>
            <artifactId>mockito-core</artifactId>
            <version>5.7.0</version>
            <scope>test</scope>
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <version>3.11.0</version>
            </plugin>
        </plugins>
    </build>
</project>
'''
        self._write_file(self.output_dir / "pom.xml", content)

    def _generate_gradle(self) -> None:
        """Generate build.gradle."""
        content = f'''plugins {{
    id 'java-library'
    id 'maven-publish'
}}

group = 'com.qbitelai'
version = '{self.config.package_version}'

java {{
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}}

repositories {{
    mavenCentral()
}}

dependencies {{
    implementation 'com.squareup.okhttp3:okhttp:4.12.0'
    implementation 'com.fasterxml.jackson.core:jackson-databind:2.16.0'
    implementation 'com.fasterxml.jackson.datatype:jackson-datatype-jsr310:2.16.0'

    testImplementation 'org.junit.jupiter:junit-jupiter:5.10.0'
    testImplementation 'org.mockito:mockito-core:5.7.0'
}}

test {{
    useJUnitPlatform()
}}

publishing {{
    publications {{
        maven(MavenPublication) {{
            from components.java
        }}
    }}
}}
'''
        self._write_file(self.output_dir / "build.gradle", content)

    def _generate_client(self, package_path: Path) -> None:
        """Generate main client class."""
        content = f'''package {self.package_name};

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import okhttp3.*;

import java.io.IOException;
import java.time.Duration;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

/**
 * QBITEL API Client.
 *
 * <p>Main entry point for interacting with the QBITEL API.
 *
 * <p>Example usage:
 * <pre>{{@code
 * QbitelClient client = QbitelClient.builder()
 *     .apiKey("your-api-key")
 *     .build();
 *
 * // Sync call
 * Response response = client.protocolDiscovery().analyzeTraffic(request);
 *
 * // Async call
 * CompletableFuture<Response> future = client.protocolDiscovery().analyzeTrafficAsync(request);
 * }}</pre>
 */
public class QbitelClient {{

    /** Default API base URL */
    public static final String DEFAULT_BASE_URL = "{self.config.api_base_url}";

    /** SDK version */
    public static final String VERSION = "{self.config.package_version}";

    private final OkHttpClient httpClient;
    private final String baseUrl;
    private final String apiKey;
    private final ObjectMapper objectMapper;

    private final ProtocolDiscoveryService protocolDiscovery;
    private final LegacyWhispererService legacyWhisperer;
    private final AIGatewayService aiGateway;
    private final SecurityService security;

    private QbitelClient(Builder builder) {{
        this.baseUrl = builder.baseUrl != null ? builder.baseUrl : DEFAULT_BASE_URL;
        this.apiKey = builder.apiKey;

        this.httpClient = new OkHttpClient.Builder()
            .connectTimeout(builder.connectTimeout)
            .readTimeout(builder.readTimeout)
            .writeTimeout(builder.writeTimeout)
            .addInterceptor(chain -> {{
                Request original = chain.request();
                Request.Builder requestBuilder = original.newBuilder()
                    .header("Authorization", "Bearer " + apiKey)
                    .header("Content-Type", "application/json")
                    .header("Accept", "application/json")
                    .header("User-Agent", "qbitel-java/" + VERSION);
                return chain.proceed(requestBuilder.build());
            }})
            .build();

        this.objectMapper = new ObjectMapper()
            .registerModule(new JavaTimeModule())
            .disable(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS);

        // Initialize services
        this.protocolDiscovery = new ProtocolDiscoveryService(this);
        this.legacyWhisperer = new LegacyWhispererService(this);
        this.aiGateway = new AIGatewayService(this);
        this.security = new SecurityService(this);
    }}

    /**
     * Create a new client builder.
     * @return A new Builder instance
     */
    public static Builder builder() {{
        return new Builder();
    }}

    /**
     * Get the Protocol Discovery service.
     * @return ProtocolDiscoveryService instance
     */
    public ProtocolDiscoveryService protocolDiscovery() {{
        return protocolDiscovery;
    }}

    /**
     * Get the Legacy Whisperer service.
     * @return LegacyWhispererService instance
     */
    public LegacyWhispererService legacyWhisperer() {{
        return legacyWhisperer;
    }}

    /**
     * Get the AI Gateway service.
     * @return AIGatewayService instance
     */
    public AIGatewayService aiGateway() {{
        return aiGateway;
    }}

    /**
     * Get the Security service.
     * @return SecurityService instance
     */
    public SecurityService security() {{
        return security;
    }}

    /**
     * Execute a synchronous GET request.
     */
    public <T> T get(String path, Class<T> responseType) throws QbitelException {{
        return executeRequest("GET", path, null, responseType);
    }}

    /**
     * Execute a synchronous POST request.
     */
    public <T, B> T post(String path, B body, Class<T> responseType) throws QbitelException {{
        return executeRequest("POST", path, body, responseType);
    }}

    /**
     * Execute a synchronous PUT request.
     */
    public <T, B> T put(String path, B body, Class<T> responseType) throws QbitelException {{
        return executeRequest("PUT", path, body, responseType);
    }}

    /**
     * Execute a synchronous DELETE request.
     */
    public <T> T delete(String path, Class<T> responseType) throws QbitelException {{
        return executeRequest("DELETE", path, null, responseType);
    }}

    /**
     * Execute an asynchronous request.
     */
    public <T, B> CompletableFuture<T> executeAsync(
            String method, String path, B body, Class<T> responseType) {{
        return CompletableFuture.supplyAsync(() -> {{
            try {{
                return executeRequest(method, path, body, responseType);
            }} catch (QbitelException e) {{
                throw new RuntimeException(e);
            }}
        }});
    }}

    private <T, B> T executeRequest(
            String method, String path, B body, Class<T> responseType) throws QbitelException {{
        String url = baseUrl + path;

        RequestBody requestBody = null;
        if (body != null) {{
            try {{
                String json = objectMapper.writeValueAsString(body);
                requestBody = RequestBody.create(json, MediaType.parse("application/json"));
            }} catch (Exception e) {{
                throw new QbitelException("Failed to serialize request body", e);
            }}
        }}

        Request request = new Request.Builder()
            .url(url)
            .method(method, requestBody)
            .build();

        try (Response response = httpClient.newCall(request).execute()) {{
            String responseBody = response.body() != null ? response.body().string() : "";

            if (!response.isSuccessful()) {{
                throw new QbitelApiException(response.code(), responseBody);
            }}

            if (responseType == Void.class || responseBody.isEmpty()) {{
                return null;
            }}

            return objectMapper.readValue(responseBody, responseType);
        }} catch (IOException e) {{
            throw new QbitelException("Request failed", e);
        }}
    }}

    ObjectMapper getObjectMapper() {{
        return objectMapper;
    }}

    /**
     * Client builder.
     */
    public static class Builder {{
        private String apiKey;
        private String baseUrl;
        private Duration connectTimeout = Duration.ofSeconds(10);
        private Duration readTimeout = Duration.ofSeconds(30);
        private Duration writeTimeout = Duration.ofSeconds(30);

        /**
         * Set the API key.
         * @param apiKey The API key
         * @return This builder
         */
        public Builder apiKey(String apiKey) {{
            this.apiKey = apiKey;
            return this;
        }}

        /**
         * Set the base URL.
         * @param baseUrl The base URL
         * @return This builder
         */
        public Builder baseUrl(String baseUrl) {{
            this.baseUrl = baseUrl;
            return this;
        }}

        /**
         * Set the connect timeout.
         * @param timeout The timeout duration
         * @return This builder
         */
        public Builder connectTimeout(Duration timeout) {{
            this.connectTimeout = timeout;
            return this;
        }}

        /**
         * Set the read timeout.
         * @param timeout The timeout duration
         * @return This builder
         */
        public Builder readTimeout(Duration timeout) {{
            this.readTimeout = timeout;
            return this;
        }}

        /**
         * Set the write timeout.
         * @param timeout The timeout duration
         * @return This builder
         */
        public Builder writeTimeout(Duration timeout) {{
            this.writeTimeout = timeout;
            return this;
        }}

        /**
         * Build the client.
         * @return A new QbitelClient instance
         */
        public QbitelClient build() {{
            if (apiKey == null || apiKey.isEmpty()) {{
                throw new IllegalArgumentException("API key is required");
            }}
            return new QbitelClient(this);
        }}
    }}
}}
'''
        self._write_file(package_path / "QbitelClient.java", content)

    def _generate_config(self, package_path: Path) -> None:
        """Generate configuration class."""
        content = f'''package {self.package_name};

/**
 * SDK configuration constants.
 */
public final class QbitelConfig {{

    private QbitelConfig() {{
        // Utility class
    }}

    /** Default API base URL */
    public static final String DEFAULT_BASE_URL = "{self.config.api_base_url}";

    /** API version */
    public static final String API_VERSION = "{self.config.api_version}";

    /** SDK version */
    public static final String SDK_VERSION = "{self.config.package_version}";
}}
'''
        self._write_file(package_path / "QbitelConfig.java", content)

    def _generate_types(self, package_path: Path) -> None:
        """Generate type classes."""
        models_path = package_path / "models"
        self._ensure_dir(models_path)

        schemas = self.schema.components.get("schemas", {})

        for name, schema in schemas.items():
            content = self._generate_java_class(name, schema)
            class_name = self._to_pascal_case(name)
            self._write_file(models_path / f"{class_name}.java", content)

    def _generate_java_class(self, name: str, schema: Dict[str, Any]) -> str:
        """Generate a Java class."""
        class_name = self._to_pascal_case(name)
        description = schema.get("description", f"{class_name} model")

        lines = [f"package {self.package_name}.models;"]
        lines.append("")
        lines.append("import com.fasterxml.jackson.annotation.JsonProperty;")
        lines.append("import java.time.OffsetDateTime;")
        lines.append("import java.util.List;")
        lines.append("import java.util.Map;")
        lines.append("")
        lines.append(f"/**")
        lines.append(f" * {description}")
        lines.append(f" */")
        lines.append(f"public class {class_name} {{")

        properties = schema.get("properties", {})

        # Generate fields
        for prop_name, prop_schema in properties.items():
            field_name = self._to_camel_case(prop_name)
            field_type = self._schema_to_java_type(prop_schema)

            field_desc = prop_schema.get("description", "")
            if field_desc:
                lines.append(f"    /** {field_desc} */")

            lines.append(f'    @JsonProperty("{prop_name}")')
            lines.append(f"    private {field_type} {field_name};")
            lines.append("")

        # Generate getters and setters
        for prop_name, prop_schema in properties.items():
            field_name = self._to_camel_case(prop_name)
            field_type = self._schema_to_java_type(prop_schema)
            method_name = self._to_pascal_case(prop_name)

            # Getter
            lines.append(f"    public {field_type} get{method_name}() {{")
            lines.append(f"        return {field_name};")
            lines.append("    }")
            lines.append("")

            # Setter
            lines.append(f"    public void set{method_name}({field_type} {field_name}) {{")
            lines.append(f"        this.{field_name} = {field_name};")
            lines.append("    }")
            lines.append("")

        lines.append("}")
        return "\n".join(lines)

    def _generate_exceptions(self, package_path: Path) -> None:
        """Generate exception classes."""
        # Base exception
        content = f'''package {self.package_name};

/**
 * Base exception for QBITEL SDK errors.
 */
public class QbitelException extends Exception {{

    public QbitelException(String message) {{
        super(message);
    }}

    public QbitelException(String message, Throwable cause) {{
        super(message, cause);
    }}
}}
'''
        self._write_file(package_path / "QbitelException.java", content)

        # API exception
        content = f'''package {self.package_name};

/**
 * Exception for API error responses.
 */
public class QbitelApiException extends QbitelException {{

    private final int statusCode;
    private final String responseBody;

    public QbitelApiException(int statusCode, String responseBody) {{
        super(String.format("API error %d: %s", statusCode, responseBody));
        this.statusCode = statusCode;
        this.responseBody = responseBody;
    }}

    /**
     * Get the HTTP status code.
     * @return The status code
     */
    public int getStatusCode() {{
        return statusCode;
    }}

    /**
     * Get the response body.
     * @return The response body
     */
    public String getResponseBody() {{
        return responseBody;
    }}

    /**
     * Check if this is a not found error (404).
     */
    public boolean isNotFound() {{
        return statusCode == 404;
    }}

    /**
     * Check if this is an unauthorized error (401).
     */
    public boolean isUnauthorized() {{
        return statusCode == 401;
    }}

    /**
     * Check if this is a rate limit error (429).
     */
    public boolean isRateLimited() {{
        return statusCode == 429;
    }}

    /**
     * Check if this is a server error (5xx).
     */
    public boolean isServerError() {{
        return statusCode >= 500;
    }}
}}
'''
        self._write_file(package_path / "QbitelApiException.java", content)

    def _generate_services(self, package_path: Path) -> None:
        """Generate service classes."""
        services = [
            ("ProtocolDiscoveryService", "Protocol Discovery"),
            ("LegacyWhispererService", "Legacy Whisperer"),
            ("AIGatewayService", "AI Gateway"),
            ("SecurityService", "Security"),
        ]

        for class_name, description in services:
            content = f'''package {self.package_name};

import java.util.concurrent.CompletableFuture;

/**
 * {description} service client.
 */
public class {class_name} {{

    private final QbitelClient client;

    {class_name}(QbitelClient client) {{
        this.client = client;
    }}

    // Add service methods here
}}
'''
            self._write_file(package_path / f"{class_name}.java", content)

    def _generate_tests(self) -> None:
        """Generate test files."""
        test_dir = self.output_dir / "src" / "test" / "java"
        package_path = test_dir / self.package_name.replace(".", "/")
        self._ensure_dir(package_path)

        content = f'''package {self.package_name};

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class QbitelClientTest {{

    @Test
    void testBuilder() {{
        QbitelClient client = QbitelClient.builder()
            .apiKey("test-key")
            .build();

        assertNotNull(client);
    }}

    @Test
    void testBuilderRequiresApiKey() {{
        assertThrows(IllegalArgumentException.class, () -> {{
            QbitelClient.builder().build();
        }});
    }}
}}
'''
        self._write_file(package_path / "QbitelClientTest.java", content)

    def _schema_to_java_type(self, schema: Dict[str, Any]) -> str:
        """Convert OpenAPI schema to Java type."""
        return TypeMapper.map_schema_type(Language.JAVA, schema)

    def _to_pascal_case(self, s: str) -> str:
        """Convert to PascalCase."""
        words = re.split(r"[-_\s]+", s)
        return "".join(word.capitalize() for word in words)

    def _to_camel_case(self, s: str) -> str:
        """Convert to camelCase."""
        pascal = self._to_pascal_case(s)
        return pascal[0].lower() + pascal[1:] if pascal else ""
