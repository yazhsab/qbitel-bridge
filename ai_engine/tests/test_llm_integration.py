"""
Tests for LLM Integration (Unified LLM Service).
Covers LLMProvider, LLMRequest, LLMResponse, and UnifiedLLMService.
"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import asdict

from llm.unified_llm_service import (
    UnifiedLLMService,
    LLMProvider,
    LLMRequest,
    LLMResponse,
)
from core.config import Config
from core.exceptions import LLMException


class TestLLMProvider:
    """Test LLMProvider enum."""

    def test_provider_values(self):
        """Test LLM provider enum values."""
        assert LLMProvider.OPENAI_GPT4.value == "openai_gpt4"
        assert LLMProvider.ANTHROPIC_CLAUDE.value == "anthropic_claude"
        assert LLMProvider.OLLAMA_LOCAL.value == "ollama_local"

    def test_provider_from_string(self):
        """Test creating provider from string."""
        provider = LLMProvider("openai_gpt4")
        assert provider == LLMProvider.OPENAI_GPT4


class TestLLMRequest:
    """Test LLMRequest dataclass."""

    def test_create_basic_request(self):
        """Test creating basic LLM request."""
        request = LLMRequest(
            prompt="Explain TCP protocol",
            feature_domain="protocol_analysis",
        )

        assert request.prompt == "Explain TCP protocol"
        assert request.feature_domain == "protocol_analysis"
        assert request.max_tokens == 2000
        assert request.temperature == 0.3
        assert request.stream is False

    def test_create_request_with_context(self):
        """Test request with context."""
        context = {"protocol": "TCP", "layer": "transport"}
        request = LLMRequest(
            prompt="Analyze this protocol",
            feature_domain="protocol_analysis",
            context=context,
            max_tokens=1000,
            temperature=0.5,
        )

        assert request.context == context
        assert request.max_tokens == 1000
        assert request.temperature == 0.5

    def test_request_with_system_prompt(self):
        """Test request with system prompt."""
        request = LLMRequest(
            prompt="User query",
            feature_domain="copilot",
            system_prompt="You are a protocol analysis expert",
        )

        assert request.system_prompt == "You are a protocol analysis expert"

    def test_request_streaming(self):
        """Test streaming request."""
        request = LLMRequest(
            prompt="Generate long analysis",
            feature_domain="analysis",
            stream=True,
        )

        assert request.stream is True

    def test_request_to_dict(self):
        """Test converting request to dict."""
        request = LLMRequest(
            prompt="Test prompt",
            feature_domain="test",
            max_tokens=500,
        )

        data = asdict(request)
        assert data["prompt"] == "Test prompt"
        assert data["max_tokens"] == 500


class TestLLMResponse:
    """Test LLMResponse dataclass."""

    def test_create_response(self):
        """Test creating LLM response."""
        response = LLMResponse(
            content="This is the LLM response",
            provider="openai_gpt4",
            tokens_used=150,
            processing_time=1.25,
            confidence=0.95,
        )

        assert response.content == "This is the LLM response"
        assert response.provider == "openai_gpt4"
        assert response.tokens_used == 150
        assert response.processing_time == 1.25
        assert response.confidence == 0.95

    def test_response_with_metadata(self):
        """Test response with metadata."""
        metadata = {"model": "gpt-4", "finish_reason": "stop"}
        response = LLMResponse(
            content="Response text",
            provider="openai_gpt4",
            tokens_used=100,
            processing_time=0.5,
            confidence=0.9,
            metadata=metadata,
        )

        assert response.metadata == metadata
        assert response.metadata["model"] == "gpt-4"

    def test_response_to_dict(self):
        """Test converting response to dict."""
        response = LLMResponse(
            content="Test response",
            provider="anthropic_claude",
            tokens_used=200,
            processing_time=2.0,
            confidence=0.85,
        )

        data = asdict(response)
        assert data["content"] == "Test response"
        assert data["provider"] == "anthropic_claude"


class TestUnifiedLLMServiceInitialization:
    """Test UnifiedLLMService initialization."""

    def test_init_with_config(self):
        """Test service initialization."""
        config = Config()
        service = UnifiedLLMService(config)

        assert service.config == config
        assert service.openai_client is None
        assert service.anthropic_client is None
        assert service.ollama_client is None

    @pytest.mark.asyncio
    async def test_initialize_openai_provider(self):
        """Test initializing OpenAI provider."""
        config = Config()
        config.llm_provider = "openai_gpt4"
        config.openai_api_key = "test_key_123"

        service = UnifiedLLMService(config)

        with patch("llm.unified_llm_service.openai") as mock_openai:
            mock_openai.AsyncOpenAI = Mock()
            await service.initialize()

            assert service.primary_provider == LLMProvider.OPENAI_GPT4

    @pytest.mark.asyncio
    async def test_initialize_anthropic_provider(self):
        """Test initializing Anthropic provider."""
        config = Config()
        config.llm_provider = "anthropic_claude"
        config.anthropic_api_key = "test_key_456"

        service = UnifiedLLMService(config)

        with patch("llm.unified_llm_service.AsyncAnthropic") as mock_anthropic:
            mock_anthropic.return_value = AsyncMock()
            await service.initialize()

            assert service.primary_provider == LLMProvider.ANTHROPIC_CLAUDE

    @pytest.mark.asyncio
    async def test_initialize_ollama_provider(self):
        """Test initializing Ollama provider."""
        config = Config()
        config.llm_provider = "ollama_local"
        config.ollama_host = "http://localhost:11434"

        service = UnifiedLLMService(config)

        with patch("llm.unified_llm_service.ollama") as mock_ollama:
            mock_ollama.AsyncClient = Mock()
            await service.initialize()

            assert service.primary_provider == LLMProvider.OLLAMA_LOCAL


class TestLLMServiceGeneration:
    """Test LLM text generation."""

    @pytest_asyncio.fixture
    async def service(self):
        """Create LLM service instance."""
        config = Config()
        config.llm_provider = "openai_gpt4"
        svc = UnifiedLLMService(config)
        svc.openai_client = AsyncMock()
        svc.primary_provider = LLMProvider.OPENAI_GPT4
        return svc

    @pytest.mark.asyncio
    async def test_generate_basic(self, service):
        """Test basic text generation."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Generated text"))]
        mock_response.usage = Mock(total_tokens=100)

        service.openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        request = LLMRequest(
            prompt="Test prompt",
            feature_domain="test",
        )

        with patch("time.time", side_effect=[0, 1.5]):
            response = await service.generate(request)

        assert response.content == "Generated text"
        assert response.provider == "openai_gpt4"
        assert response.tokens_used == 100
        assert response.processing_time == 1.5

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self, service):
        """Test generation with system prompt."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response with system"))]
        mock_response.usage = Mock(total_tokens=150)

        service.openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        request = LLMRequest(
            prompt="User query",
            feature_domain="copilot",
            system_prompt="You are an expert",
        )

        with patch("time.time", side_effect=[0, 1.0]):
            response = await service.generate(request)

        # Verify system prompt was used
        call_args = service.openai_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert any(msg["role"] == "system" for msg in messages)

    @pytest.mark.asyncio
    async def test_generate_with_context(self, service):
        """Test generation with context."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Context-aware response"))]
        mock_response.usage = Mock(total_tokens=200)

        service.openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        request = LLMRequest(
            prompt="Analyze this",
            feature_domain="analysis",
            context={"protocol": "HTTP", "version": "1.1"},
        )

        with patch("time.time", side_effect=[0, 2.0]):
            response = await service.generate(request)

        assert response.content == "Context-aware response"
        assert response.metadata is not None


class TestLLMServiceFallback:
    """Test LLM provider fallback mechanisms."""

    @pytest_asyncio.fixture
    async def service(self):
        """Create service with multiple providers."""
        config = Config()
        svc = UnifiedLLMService(config)
        svc.primary_provider = LLMProvider.OPENAI_GPT4
        svc.fallback_providers = [LLMProvider.ANTHROPIC_CLAUDE]
        svc.openai_client = AsyncMock()
        svc.anthropic_client = AsyncMock()
        return svc

    @pytest.mark.asyncio
    async def test_fallback_on_primary_failure(self, service):
        """Test fallback when primary provider fails."""
        # Primary fails
        service.openai_client.chat.completions.create = AsyncMock(
            side_effect=Exception("OpenAI error")
        )

        # Fallback succeeds
        mock_response = Mock()
        mock_response.content = [Mock(text="Fallback response")]
        mock_response.usage = Mock(input_tokens=50, output_tokens=50)

        service.anthropic_client.messages.create = AsyncMock(
            return_value=mock_response
        )

        request = LLMRequest(prompt="Test", feature_domain="test")

        with patch("time.time", side_effect=[0, 1.0]):
            with patch.object(service, "_generate_anthropic") as mock_gen:
                mock_gen.return_value = LLMResponse(
                    content="Fallback response",
                    provider="anthropic_claude",
                    tokens_used=100,
                    processing_time=1.0,
                    confidence=0.9,
                )
                response = await service.generate_with_fallback(request)

        assert response.provider == "anthropic_claude"


class TestLLMServiceStreaming:
    """Test streaming responses."""

    @pytest_asyncio.fixture
    async def service(self):
        """Create service."""
        config = Config()
        svc = UnifiedLLMService(config)
        svc.openai_client = AsyncMock()
        svc.primary_provider = LLMProvider.OPENAI_GPT4
        return svc

    @pytest.mark.asyncio
    async def test_streaming_generation(self, service):
        """Test streaming text generation."""

        async def mock_stream():
            chunks = ["Chunk 1", " Chunk 2", " Chunk 3"]
            for chunk in chunks:
                yield Mock(choices=[Mock(delta=Mock(content=chunk))])

        service.openai_client.chat.completions.create = AsyncMock(
            return_value=mock_stream()
        )

        request = LLMRequest(
            prompt="Generate stream",
            feature_domain="test",
            stream=True,
        )

        chunks_received = []
        async for chunk in service.generate_stream(request):
            chunks_received.append(chunk)

        assert len(chunks_received) > 0


class TestLLMServiceMetrics:
    """Test metrics collection."""

    @pytest_asyncio.fixture
    async def service(self):
        """Create service."""
        config = Config()
        svc = UnifiedLLMService(config)
        svc.openai_client = AsyncMock()
        svc.primary_provider = LLMProvider.OPENAI_GPT4
        return svc

    @pytest.mark.asyncio
    async def test_metrics_recorded(self, service):
        """Test that metrics are recorded."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test"))]
        mock_response.usage = Mock(total_tokens=50)

        service.openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        request = LLMRequest(prompt="Test", feature_domain="test")

        with patch("time.time", side_effect=[0, 0.5]):
            with patch("llm.unified_llm_service.LLM_REQUEST_COUNTER") as mock_counter:
                with patch(
                    "llm.unified_llm_service.LLM_TOKEN_USAGE"
                ) as mock_token_counter:
                    await service.generate(request)

                    # Verify metrics were incremented
                    mock_counter.labels.assert_called()
                    mock_token_counter.labels.assert_called()


class TestLLMServiceErrorHandling:
    """Test error handling."""

    @pytest_asyncio.fixture
    async def service(self):
        """Create service."""
        config = Config()
        svc = UnifiedLLMService(config)
        svc.openai_client = AsyncMock()
        svc.primary_provider = LLMProvider.OPENAI_GPT4
        return svc

    @pytest.mark.asyncio
    async def test_generation_api_error(self, service):
        """Test handling API errors."""
        service.openai_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        request = LLMRequest(prompt="Test", feature_domain="test")

        with pytest.raises(LLMException):
            await service.generate(request)

    @pytest.mark.asyncio
    async def test_generation_timeout(self, service):
        """Test handling timeouts."""

        async def slow_response():
            await asyncio.sleep(10)
            return Mock()

        service.openai_client.chat.completions.create = slow_response
        service.request_timeout = 1  # 1 second timeout

        request = LLMRequest(prompt="Test", feature_domain="test")

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(service.generate(request), timeout=1)


class TestLLMServiceEdgeCases:
    """Test edge cases."""

    @pytest_asyncio.fixture
    async def service(self):
        """Create service."""
        config = Config()
        svc = UnifiedLLMService(config)
        svc.openai_client = AsyncMock()
        svc.primary_provider = LLMProvider.OPENAI_GPT4
        return svc

    @pytest.mark.asyncio
    async def test_empty_response(self, service):
        """Test handling empty response."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=""))]
        mock_response.usage = Mock(total_tokens=10)

        service.openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        request = LLMRequest(prompt="Test", feature_domain="test")

        with patch("time.time", side_effect=[0, 0.1]):
            response = await service.generate(request)

        assert response.content == ""
        assert response.tokens_used == 10

    @pytest.mark.asyncio
    async def test_very_long_prompt(self, service):
        """Test handling very long prompt."""
        long_prompt = "x" * 10000

        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response"))]
        mock_response.usage = Mock(total_tokens=5000)

        service.openai_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        request = LLMRequest(
            prompt=long_prompt,
            feature_domain="test",
            max_tokens=4000,
        )

        with patch("time.time", side_effect=[0, 3.0]):
            response = await service.generate(request)

        assert response.content == "Response"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
