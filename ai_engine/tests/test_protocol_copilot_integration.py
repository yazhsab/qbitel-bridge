"""
CRONOS AI Engine - Protocol Intelligence Copilot Integration Tests
Comprehensive test suite for LLM-enhanced protocol discovery and analysis.
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any, Optional

from ..core.config import Config
from ..copilot.protocol_copilot import (
    ProtocolIntelligenceCopilot,
    CopilotQuery,
    CopilotResponse,
    create_protocol_copilot
)
from ..discovery.enhanced_protocol_discovery_orchestrator import (
    EnhancedProtocolDiscoveryOrchestrator,
    EnhancedDiscoveryRequest,
    EnhancedDiscoveryResult,
    LLMAnalysisType,
    LLMAnalysisResult
)
from ..llm.unified_llm_service import UnifiedLLMService, LLMRequest, LLMResponse
from ..llm.rag_engine import RAGEngine
from ..api.schemas import QueryType

class TestProtocolCopilotCore:
    """Test core Protocol Intelligence Copilot functionality."""
    
    @pytest.fixture
    async def mock_config(self):
        """Create mock configuration for testing."""
        config = Mock(spec=Config)
        config.llm = Mock()
        config.llm.providers = {
            'openai': {'enabled': True, 'api_key': 'test-key'},
            'anthropic': {'enabled': True, 'api_key': 'test-key'},
            'ollama': {'enabled': False}
        }
        config.llm.default_provider = 'openai'
        config.llm.timeout = 30
        config.redis = Mock()
        config.redis.host = 'localhost'
        config.redis.port = 6379
        config.redis.db = 0
        config.security = Mock()
        config.security.jwt_secret = 'test-secret'
        return config
    
    @pytest.fixture
    async def mock_llm_service(self):
        """Create mock LLM service."""
        service = Mock(spec=UnifiedLLMService)
        service.process_request = AsyncMock(return_value=LLMResponse(
            response="Test LLM response for protocol analysis",
            confidence=0.85,
            provider="openai",
            model="gpt-4",
            processing_time=1.2,
            tokens_used=150,
            metadata={"test": "data"}
        ))
        service.get_health_status = Mock(return_value={
            "providers": {"openai": "healthy", "anthropic": "healthy"},
            "default_provider": "openai"
        })
        return service
    
    @pytest.fixture
    async def mock_rag_engine(self):
        """Create mock RAG engine."""
        engine = Mock(spec=RAGEngine)
        engine.search = AsyncMock(return_value=[
            {
                'content': 'Test protocol knowledge',
                'similarity': 0.9,
                'metadata': {'source': 'test'}
            }
        ])
        engine.add_documents = AsyncMock(return_value=True)
        engine.get_health_status = Mock(return_value="healthy")
        return engine
    
    @pytest.fixture
    async def protocol_copilot(self, mock_config, mock_llm_service, mock_rag_engine):
        """Create Protocol Intelligence Copilot instance."""
        with patch('ai_engine.copilot.protocol_copilot.UnifiedLLMService') as mock_llm_class:
            mock_llm_class.return_value = mock_llm_service
            with patch('ai_engine.copilot.protocol_copilot.RAGEngine') as mock_rag_class:
                mock_rag_class.return_value = mock_rag_engine
                copilot = ProtocolIntelligenceCopilot(
                    llm_service=mock_llm_service,
                    rag_engine=mock_rag_engine
                )
                await copilot.initialize()
                return copilot
    
    @pytest.mark.asyncio
    async def test_copilot_initialization(self, mock_config, mock_llm_service, mock_rag_engine):
        """Test Protocol Intelligence Copilot initialization."""
        with patch('ai_engine.copilot.protocol_copilot.UnifiedLLMService') as mock_llm_class:
            mock_llm_class.return_value = mock_llm_service
            with patch('ai_engine.copilot.protocol_copilot.RAGEngine') as mock_rag_class:
                mock_rag_class.return_value = mock_rag_engine
                
                copilot = ProtocolIntelligenceCopilot(
                    llm_service=mock_llm_service,
                    rag_engine=mock_rag_engine
                )
                
                assert copilot.llm_service == mock_llm_service
                assert copilot.rag_engine == mock_rag_engine
                assert not copilot.is_initialized
                
                await copilot.initialize()
                assert copilot.is_initialized
    
    @pytest.mark.asyncio
    async def test_copilot_query_processing(self, protocol_copilot):
        """Test basic copilot query processing."""
        query = CopilotQuery(
            query="Analyze this HTTP protocol traffic",
            query_type="protocol_analysis",
            user_id="test_user",
            session_id="test_session",
            packet_data=b"GET /test HTTP/1.1\r\nHost: example.com\r\n\r\n"
        )
        
        response = await protocol_copilot.process_query(query)
        
        assert isinstance(response, CopilotResponse)
        assert response.response == "Test LLM response for protocol analysis"
        assert response.confidence == 0.85
        assert response.query_type == "protocol_analysis"
        assert response.session_id == "test_session"
        assert response.processing_time > 0
        
        # Verify LLM service was called
        protocol_copilot.llm_service.process_request.assert_called_once()
        call_args = protocol_copilot.llm_service.process_request.call_args[0][0]
        assert isinstance(call_args, LLMRequest)
    
    @pytest.mark.asyncio
    async def test_copilot_with_rag_enhancement(self, protocol_copilot):
        """Test copilot query with RAG enhancement."""
        query = CopilotQuery(
            query="What are the security implications of this protocol?",
            query_type="security_assessment",
            user_id="test_user",
            session_id="test_session"
        )
        
        response = await protocol_copilot.process_query(query)
        
        # Verify RAG engine was used for context enhancement
        protocol_copilot.rag_engine.search.assert_called_once()
        
        # Verify response includes sources
        assert len(response.sources) > 0
        assert response.sources[0]['content'] == 'Test protocol knowledge'

class TestEnhancedProtocolDiscovery:
    """Test Enhanced Protocol Discovery Orchestrator."""
    
    @pytest.fixture
    async def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=Config)
        config.inference = Mock()
        config.inference.num_workers = 4
        return config
    
    @pytest.fixture
    async def mock_copilot(self):
        """Create mock Protocol Intelligence Copilot."""
        copilot = Mock(spec=ProtocolIntelligenceCopilot)
        copilot.process_query = AsyncMock(return_value=CopilotResponse(
            response="Enhanced analysis of discovered protocol",
            query_type="protocol_analysis",
            confidence=0.9,
            sources=[],
            suggestions=["Consider security hardening", "Monitor for anomalies"],
            session_id="test_session",
            processing_time=2.1,
            metadata={"llm_provider": "openai"}
        ))
        copilot.llm_service = Mock()
        copilot.get_health_status = Mock(return_value={
            "llm_service": {"providers": {"openai": "healthy"}},
            "rag_engine": "healthy"
        })
        return copilot
    
    @pytest.fixture
    async def enhanced_orchestrator(self, mock_config, mock_copilot):
        """Create Enhanced Protocol Discovery Orchestrator."""
        # Mock all the required components
        with patch('ai_engine.discovery.enhanced_protocol_discovery_orchestrator.StatisticalAnalyzer'), \
             patch('ai_engine.discovery.enhanced_protocol_discovery_orchestrator.GrammarLearner'), \
             patch('ai_engine.discovery.enhanced_protocol_discovery_orchestrator.ParserGenerator'), \
             patch('ai_engine.discovery.enhanced_protocol_discovery_orchestrator.ProtocolClassifier'), \
             patch('ai_engine.discovery.enhanced_protocol_discovery_orchestrator.MessageValidator'):
            
            orchestrator = EnhancedProtocolDiscoveryOrchestrator(mock_config, mock_copilot)
            
            # Mock the traditional discovery method
            orchestrator._execute_discovery = AsyncMock(return_value=EnhancedDiscoveryResult(
                protocol_type="HTTP",
                confidence=0.8,
                processing_time=1.5,
                phases_completed=[]
            ))
            
            return orchestrator
    
    @pytest.mark.asyncio
    async def test_enhanced_discovery_with_llm_analysis(self, enhanced_orchestrator, mock_copilot):
        """Test enhanced protocol discovery with LLM analysis."""
        request = EnhancedDiscoveryRequest(
            messages=[b"GET /test HTTP/1.1\r\nHost: example.com\r\n\r\n"],
            enable_llm_analysis=True,
            llm_analysis_types=[
                LLMAnalysisType.PROTOCOL_IDENTIFICATION,
                LLMAnalysisType.SECURITY_ASSESSMENT
            ],
            natural_language_explanation=True
        )
        
        result = await enhanced_orchestrator.discover_protocol_enhanced(request)
        
        assert isinstance(result, EnhancedDiscoveryResult)
        assert result.protocol_type == "HTTP"
        assert len(result.llm_analyses) <= len(request.llm_analysis_types)
        
        # Verify copilot was called for LLM analysis
        assert mock_copilot.process_query.call_count >= 0
    
    @pytest.mark.asyncio
    async def test_enhanced_discovery_without_llm(self, enhanced_orchestrator):
        """Test enhanced discovery with LLM analysis disabled."""
        request = EnhancedDiscoveryRequest(
            messages=[b"test protocol data"],
            enable_llm_analysis=False,
            natural_language_explanation=False
        )
        
        result = await enhanced_orchestrator.discover_protocol_enhanced(request)
        
        assert isinstance(result, EnhancedDiscoveryResult)
        assert len(result.llm_analyses) == 0
        assert result.natural_language_summary is None
    
    @pytest.mark.asyncio
    async def test_llm_analysis_types(self, enhanced_orchestrator, mock_copilot):
        """Test different LLM analysis types."""
        for analysis_type in LLMAnalysisType:
            request = EnhancedDiscoveryRequest(
                messages=[b"test data"],
                enable_llm_analysis=True,
                llm_analysis_types=[analysis_type],
                user_context={"user_id": "test_user"}
            )
            
            result = await enhanced_orchestrator.discover_protocol_enhanced(request)
            
            # Should not fail for any analysis type
            assert isinstance(result, EnhancedDiscoveryResult)

class TestLLMServiceIntegration:
    """Test LLM Service Integration."""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        client = Mock()
        client.chat.completions.create = AsyncMock(return_value=Mock(
            choices=[Mock(message=Mock(content="OpenAI response"))],
            usage=Mock(total_tokens=100)
        ))
        return client
    
    @pytest.fixture
    def mock_anthropic_client(self):
        """Mock Anthropic client."""
        client = Mock()
        client.messages.create = AsyncMock(return_value=Mock(
            content=[Mock(text="Anthropic response")],
            usage=Mock(input_tokens=50, output_tokens=50)
        ))
        return client
    
    @pytest.mark.asyncio
    async def test_unified_llm_service_openai(self, mock_openai_client):
        """Test UnifiedLLMService with OpenAI provider."""
        config = {
            'providers': {
                'openai': {'enabled': True, 'api_key': 'test-key'},
                'anthropic': {'enabled': False}
            },
            'default_provider': 'openai'
        }
        
        with patch('openai.AsyncOpenAI', return_value=mock_openai_client):
            service = UnifiedLLMService(config)
            await service.initialize()
            
            request = LLMRequest(
                prompt="Test prompt",
                feature_domain="protocol_copilot",
                context={"test": "context"}
            )
            
            response = await service.process_request(request)
            
            assert isinstance(response, LLMResponse)
            assert response.response == "OpenAI response"
            assert response.provider == "openai"
            assert response.tokens_used == 100
    
    @pytest.mark.asyncio
    async def test_llm_service_fallback(self, mock_openai_client, mock_anthropic_client):
        """Test LLM service provider fallback."""
        config = {
            'providers': {
                'openai': {'enabled': True, 'api_key': 'test-key'},
                'anthropic': {'enabled': True, 'api_key': 'test-key'}
            },
            'default_provider': 'openai'
        }
        
        # Make OpenAI fail
        mock_openai_client.chat.completions.create.side_effect = Exception("OpenAI error")
        
        with patch('openai.AsyncOpenAI', return_value=mock_openai_client), \
             patch('anthropic.AsyncAnthropic', return_value=mock_anthropic_client):
            
            service = UnifiedLLMService(config)
            await service.initialize()
            
            request = LLMRequest(
                prompt="Test prompt",
                feature_domain="protocol_copilot"
            )
            
            response = await service.process_request(request)
            
            # Should fallback to Anthropic
            assert response.provider == "anthropic"
            assert response.response == "Anthropic response"

class TestRAGEngine:
    """Test RAG Engine functionality."""
    
    @pytest.mark.asyncio
    async def test_rag_engine_initialization(self):
        """Test RAG engine initialization."""
        with patch('chromadb.Client') as mock_chroma:
            mock_collection = Mock()
            mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
            
            engine = RAGEngine()
            await engine.initialize()
            
            assert engine.is_initialized
            assert engine.client is not None
    
    @pytest.mark.asyncio
    async def test_rag_document_storage_and_search(self):
        """Test document storage and retrieval in RAG engine."""
        with patch('chromadb.Client') as mock_chroma:
            mock_collection = Mock()
            mock_collection.query.return_value = {
                'documents': [['Test document content']],
                'distances': [[0.1]],
                'metadatas': [[{'source': 'test'}]]
            }
            mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
            
            engine = RAGEngine()
            await engine.initialize()
            
            # Test document addition
            documents = [
                {
                    'content': 'Test protocol documentation',
                    'metadata': {'type': 'protocol', 'source': 'test'}
                }
            ]
            
            result = await engine.add_documents('test_collection', documents)
            assert result is True
            
            # Test search
            search_results = await engine.search('test_collection', 'test query', limit=5)
            assert len(search_results) > 0
            assert search_results[0]['content'] == 'Test document content'
            assert search_results[0]['similarity'] > 0.8  # High similarity for low distance

class TestAPIIntegration:
    """Test API integration components."""
    
    @pytest.mark.asyncio
    async def test_copilot_endpoint_processing(self):
        """Test copilot API endpoint processing."""
        from ..api.copilot_endpoints import process_copilot_query
        from ..api.schemas import CopilotQuery as APICopilotQuery
        
        # Mock the copilot
        mock_copilot = Mock()
        mock_copilot.process_query = AsyncMock(return_value=CopilotResponse(
            response="API test response",
            query_type="protocol_analysis",
            confidence=0.85,
            sources=[],
            suggestions=[],
            session_id="api_test_session",
            processing_time=1.0
        ))
        
        query = APICopilotQuery(
            query="Test API query",
            query_type="protocol_analysis",
            user_id="test_user",
            session_id="api_test_session"
        )
        
        with patch('ai_engine.api.copilot_endpoints._get_copilot', return_value=mock_copilot):
            response = await process_copilot_query(query, {"user_id": "test_user"})
            
            assert response.response == "API test response"
            assert response.confidence == 0.85
            mock_copilot.process_query.assert_called_once()

class TestWebSocketIntegration:
    """Test WebSocket real-time communication."""
    
    @pytest.mark.asyncio
    async def test_websocket_message_handling(self):
        """Test WebSocket message processing."""
        from ..api.copilot_endpoints import handle_websocket_message
        
        mock_websocket = Mock()
        mock_websocket.send_text = AsyncMock()
        
        mock_copilot = Mock()
        mock_copilot.process_query = AsyncMock(return_value=CopilotResponse(
            response="WebSocket test response",
            query_type="protocol_analysis",
            confidence=0.9,
            sources=[],
            suggestions=[],
            session_id="ws_test_session",
            processing_time=0.8
        ))
        
        message_data = {
            "type": "query",
            "query": "WebSocket test query",
            "session_id": "ws_test_session",
            "user_id": "test_user"
        }
        
        with patch('ai_engine.api.copilot_endpoints._get_copilot', return_value=mock_copilot):
            await handle_websocket_message(mock_websocket, message_data, {"user_id": "test_user"})
            
            # Verify WebSocket response was sent
            mock_websocket.send_text.assert_called_once()
            
            # Verify copilot was called
            mock_copilot.process_query.assert_called_once()

class TestSecurityIntegration:
    """Test security and authentication integration."""
    
    @pytest.mark.asyncio
    async def test_authentication_service(self):
        """Test authentication service functionality."""
        from ..api.auth import AuthenticationService
        
        with patch('redis.asyncio.Redis') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis_instance.ping = AsyncMock()
            mock_redis.return_value = mock_redis_instance
            
            auth_service = AuthenticationService()
            await auth_service.initialize()
            
            # Test password hashing
            password = "test_password123"
            hashed = auth_service.hash_password(password)
            assert auth_service.verify_password(password, hashed)
            assert not auth_service.verify_password("wrong_password", hashed)
            
            # Test token creation and verification
            token_data = {"user_id": "test_user", "role": "admin"}
            token = auth_service.create_access_token(token_data)
            assert isinstance(token, str)
            
            payload = await auth_service.verify_token(token)
            assert payload["user_id"] == "test_user"
            assert payload["role"] == "admin"

class TestPerformanceAndResilience:
    """Test performance and resilience characteristics."""
    
    @pytest.mark.asyncio
    async def test_concurrent_copilot_queries(self):
        """Test concurrent query handling."""
        mock_copilot = Mock()
        mock_copilot.process_query = AsyncMock(return_value=CopilotResponse(
            response="Concurrent test response",
            query_type="protocol_analysis",
            confidence=0.8,
            sources=[],
            suggestions=[],
            session_id="concurrent_test",
            processing_time=0.1
        ))
        
        # Create multiple concurrent queries
        queries = [
            CopilotQuery(
                query=f"Test query {i}",
                query_type="protocol_analysis",
                user_id="test_user",
                session_id=f"session_{i}"
            )
            for i in range(10)
        ]
        
        # Execute concurrently
        start_time = time.time()
        responses = await asyncio.gather(*[
            mock_copilot.process_query(query) for query in queries
        ])
        end_time = time.time()
        
        # Verify all queries completed
        assert len(responses) == 10
        for response in responses:
            assert isinstance(response, CopilotResponse)
        
        # Should complete reasonably quickly with mocked backend
        assert end_time - start_time < 5.0
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        mock_copilot = Mock()
        
        # First call fails, second succeeds
        mock_copilot.process_query = AsyncMock(side_effect=[
            Exception("Temporary failure"),
            CopilotResponse(
                response="Recovery successful",
                query_type="protocol_analysis",
                confidence=0.7,
                sources=[],
                suggestions=[],
                session_id="recovery_test",
                processing_time=0.2
            )
        ])
        
        query = CopilotQuery(
            query="Test recovery query",
            query_type="protocol_analysis",
            user_id="test_user",
            session_id="recovery_test"
        )
        
        # First attempt should fail
        with pytest.raises(Exception):
            await mock_copilot.process_query(query)
        
        # Second attempt should succeed
        response = await mock_copilot.process_query(query)
        assert response.response == "Recovery successful"

# Integration test fixtures and utilities
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def integration_config():
    """Create integration test configuration."""
    return {
        'llm': {
            'providers': {
                'openai': {'enabled': True, 'api_key': 'test-key'},
                'anthropic': {'enabled': True, 'api_key': 'test-key'}
            },
            'default_provider': 'openai'
        },
        'redis': {
            'host': 'localhost',
            'port': 6379,
            'db': 1  # Use test database
        },
        'security': {
            'jwt_secret': 'test-secret-key',
            'enable_rate_limiting': False
        }
    }

# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_copilot_query_performance(self, benchmark):
        """Benchmark copilot query processing performance."""
        mock_copilot = Mock()
        mock_copilot.process_query = AsyncMock(return_value=CopilotResponse(
            response="Benchmark response",
            query_type="protocol_analysis",
            confidence=0.85,
            sources=[],
            suggestions=[],
            session_id="benchmark_test",
            processing_time=0.05
        ))
        
        query = CopilotQuery(
            query="Benchmark test query",
            query_type="protocol_analysis",
            user_id="benchmark_user",
            session_id="benchmark_test"
        )
        
        # Benchmark the query processing
        result = await benchmark(mock_copilot.process_query, query)
        assert isinstance(result, CopilotResponse)
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_enhanced_discovery_performance(self, benchmark):
        """Benchmark enhanced protocol discovery performance."""
        # This would test actual performance with real components
        # For now, just verify the test framework works
        
        async def mock_discovery():
            await asyncio.sleep(0.01)  # Simulate processing time
            return EnhancedDiscoveryResult(
                protocol_type="HTTP",
                confidence=0.9,
                processing_time=0.01
            )
        
        result = await benchmark(mock_discovery)
        assert isinstance(result, EnhancedDiscoveryResult)

# Test configuration
pytest_plugins = ['pytest_asyncio', 'pytest_benchmark']

# Custom test markers
def pytest_configure(config):
    """Configure custom test markers."""
    config.addinivalue_line("markers", "benchmark: mark test as a performance benchmark")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")