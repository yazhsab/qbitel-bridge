"""
CRONOS AI - Unified LLM Service
Enterprise-grade LLM integration with multiple providers and fallback mechanisms.
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any, Optional, List, Union, AsyncIterator
from dataclasses import dataclass, asdict
from enum import Enum
import openai
from anthropic import AsyncAnthropic
import ollama
from concurrent.futures import ThreadPoolExecutor
from prometheus_client import Counter, Histogram, Gauge

from ..core.config import Config
from ..core.exceptions import CronosAIException
from ..monitoring.metrics import MetricsCollector

# Prometheus metrics for LLM operations
LLM_REQUEST_COUNTER = Counter('cronos_llm_requests_total', 'Total LLM requests', ['provider', 'feature_domain', 'status'])
LLM_REQUEST_DURATION = Histogram('cronos_llm_request_duration_seconds', 'LLM request duration', ['provider', 'feature_domain'])
LLM_TOKEN_USAGE = Counter('cronos_llm_tokens_total', 'Total tokens used', ['provider', 'type'])
LLM_ACTIVE_CONNECTIONS = Gauge('cronos_llm_active_connections', 'Active LLM connections', ['provider'])

logger = logging.getLogger(__name__)

class LLMException(CronosAIException):
    """LLM-specific exception."""
    pass

class LLMProvider(Enum):
    """LLM provider types."""
    OPENAI_GPT4 = "openai_gpt4"
    ANTHROPIC_CLAUDE = "anthropic_claude"
    OLLAMA_LOCAL = "ollama_local"

@dataclass
class LLMRequest:
    """LLM request structure."""
    prompt: str
    feature_domain: str
    context: Dict[str, Any] = None
    max_tokens: int = 2000
    temperature: float = 0.3
    system_prompt: Optional[str] = None
    stream: bool = False

@dataclass
class LLMResponse:
    """LLM response structure."""
    content: str
    provider: str
    tokens_used: int
    processing_time: float
    confidence: float
    metadata: Dict[str, Any] = None

class UnifiedLLMService:
    """
    Unified LLM service supporting multiple providers with intelligent fallback.
    Integrates seamlessly with existing CRONOS AI architecture.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize providers
        self.openai_client: Optional[openai.AsyncOpenAI] = None
        self.anthropic_client: Optional[AsyncAnthropic] = None
        self.ollama_client = None
        
        # Provider health tracking
        self.provider_health = {
            LLMProvider.OPENAI_GPT4: True,
            LLMProvider.ANTHROPIC_CLAUDE: True,
            LLMProvider.OLLAMA_LOCAL: True
        }
        
        # Domain-specific routing configuration
        self.domain_routing = {
            'protocol_copilot': {
                'primary': LLMProvider.OPENAI_GPT4,
                'fallback': [LLMProvider.ANTHROPIC_CLAUDE, LLMProvider.OLLAMA_LOCAL],
                'system_prompt': """You are a protocol analysis expert with deep knowledge of networking protocols, 
                packet analysis, and cybersecurity. Provide accurate, technical responses about protocol behavior, 
                structure, and security implications. Always cite your reasoning and provide actionable insights."""
            },
            'compliance_reporter': {
                'primary': LLMProvider.ANTHROPIC_CLAUDE,
                'fallback': [LLMProvider.OPENAI_GPT4, LLMProvider.OLLAMA_LOCAL],
                'system_prompt': """You are a compliance and regulatory expert specializing in cybersecurity frameworks 
                like PCI-DSS, HIPAA, SOX, and Basel III. Provide precise regulatory interpretations and compliance 
                gap analysis."""
            },
            'legacy_whisperer': {
                'primary': LLMProvider.OLLAMA_LOCAL,  # Privacy for sensitive legacy systems
                'fallback': [LLMProvider.OPENAI_GPT4, LLMProvider.ANTHROPIC_CLAUDE],
                'system_prompt': """You are a legacy systems expert with deep knowledge of mainframes, COBOL, 
                industrial protocols, and system maintenance. Provide practical advice for maintaining and 
                troubleshooting legacy infrastructure."""
            },
            'security_orchestrator': {
                'primary': LLMProvider.ANTHROPIC_CLAUDE,
                'fallback': [LLMProvider.OPENAI_GPT4, LLMProvider.OLLAMA_LOCAL],
                'system_prompt': """You are a cybersecurity expert specializing in threat analysis, incident response, 
                and automated security orchestration. Provide clear, actionable security recommendations with 
                risk assessments."""
            },
            'translation_studio': {
                'primary': LLMProvider.OPENAI_GPT4,  # Best for code generation
                'fallback': [LLMProvider.ANTHROPIC_CLAUDE, LLMProvider.OLLAMA_LOCAL],
                'system_prompt': """You are a protocol translation and API generation expert. Generate clean, 
                production-ready code and comprehensive API documentation. Focus on security, performance, 
                and maintainability."""
            }
        }
        
        # Request queue for rate limiting
        self.request_queues = {
            provider: asyncio.Queue(maxsize=100) 
            for provider in LLMProvider
        }
        
        # Thread pool for synchronous operations
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def initialize(self) -> None:
        """Initialize all LLM providers."""
        try:
            self.logger.info("Initializing Unified LLM Service...")
            
            # Initialize OpenAI
            openai_key = getattr(self.config, 'openai_api_key', None)
            if openai_key:
                self.openai_client = openai.AsyncOpenAI(
                    api_key=openai_key,
                    timeout=30.0
                )
                await self._test_provider(LLMProvider.OPENAI_GPT4)
            
            # Initialize Anthropic
            anthropic_key = getattr(self.config, 'anthropic_api_key', None)
            if anthropic_key:
                self.anthropic_client = AsyncAnthropic(
                    api_key=anthropic_key,
                    timeout=30.0
                )
                await self._test_provider(LLMProvider.ANTHROPIC_CLAUDE)
            
            # Initialize Ollama (local)
            try:
                self.ollama_client = ollama.AsyncClient(
                    host=getattr(self.config, 'ollama_host', 'http://localhost:11434')
                )
                await self._test_provider(LLMProvider.OLLAMA_LOCAL)
            except Exception as e:
                self.logger.warning(f"Ollama initialization failed: {e}")
                self.provider_health[LLMProvider.OLLAMA_LOCAL] = False
            
            # Start health monitoring
            asyncio.create_task(self._health_monitor())
            
            self.logger.info("Unified LLM Service initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM service: {e}")
            raise LLMException(f"LLM service initialization failed: {e}")
    
    async def process_request(self, request: LLMRequest) -> LLMResponse:
        """
        Process LLM request with intelligent routing and fallback.
        
        Args:
            request: LLM request with prompt and context
            
        Returns:
            LLM response with content and metadata
        """
        start_time = time.time()
        
        # Get domain configuration
        domain_config = self.domain_routing.get(
            request.feature_domain, 
            self.domain_routing['protocol_copilot']  # Default
        )
        
        # Try primary provider first
        primary_provider = domain_config['primary']
        providers_to_try = [primary_provider] + domain_config['fallback']
        
        last_error = None
        
        for provider in providers_to_try:
            if not self.provider_health[provider]:
                continue
                
            try:
                response = await self._execute_request(
                    request, provider, domain_config['system_prompt']
                )
                
                # Update metrics
                LLM_REQUEST_COUNTER.labels(
                    provider=provider.value,
                    feature_domain=request.feature_domain,
                    status='success'
                ).inc()
                
                LLM_REQUEST_DURATION.labels(
                    provider=provider.value,
                    feature_domain=request.feature_domain
                ).observe(time.time() - start_time)
                
                return response
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"Provider {provider.value} failed: {e}")
                
                # Mark provider as unhealthy temporarily
                self.provider_health[provider] = False
                
                LLM_REQUEST_COUNTER.labels(
                    provider=provider.value,
                    feature_domain=request.feature_domain,
                    status='error'
                ).inc()
        
        # All providers failed
        raise LLMException(f"All LLM providers failed. Last error: {last_error}")
    
    async def _execute_request(
        self, 
        request: LLMRequest, 
        provider: LLMProvider, 
        system_prompt: str
    ) -> LLMResponse:
        """Execute request on specific provider."""
        start_time = time.time()
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add context if available
        if request.context:
            context_str = self._format_context(request.context)
            messages.append({"role": "user", "content": f"Context: {context_str}"})
        
        messages.append({"role": "user", "content": request.prompt})
        
        # Execute based on provider
        if provider == LLMProvider.OPENAI_GPT4:
            return await self._execute_openai(request, messages, start_time)
        elif provider == LLMProvider.ANTHROPIC_CLAUDE:
            return await self._execute_anthropic(request, messages, start_time)
        elif provider == LLMProvider.OLLAMA_LOCAL:
            return await self._execute_ollama(request, messages, start_time)
        else:
            raise LLMException(f"Unsupported provider: {provider}")
    
    async def _execute_openai(
        self, 
        request: LLMRequest, 
        messages: List[Dict], 
        start_time: float
    ) -> LLMResponse:
        """Execute request using OpenAI."""
        response = await self.openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=request.stream
        )
        
        content = response.choices[0].message.content
        tokens_used = response.usage.total_tokens if response.usage else 0
        
        LLM_TOKEN_USAGE.labels(provider='openai', type='total').inc(tokens_used)
        
        return LLMResponse(
            content=content,
            provider=LLMProvider.OPENAI_GPT4.value,
            tokens_used=tokens_used,
            processing_time=time.time() - start_time,
            confidence=0.9,  # High confidence for GPT-4
            metadata={
                'model': 'gpt-4-turbo',
                'finish_reason': response.choices[0].finish_reason
            }
        )
    
    async def _execute_anthropic(
        self, 
        request: LLMRequest, 
        messages: List[Dict], 
        start_time: float
    ) -> LLMResponse:
        """Execute request using Anthropic Claude."""
        # Convert messages format for Anthropic
        system_msg = None
        user_messages = []
        
        for msg in messages:
            if msg['role'] == 'system':
                system_msg = msg['content']
            else:
                user_messages.append(msg)
        
        response = await self.anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=user_messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system=system_msg
        )
        
        content = response.content[0].text
        tokens_used = response.usage.input_tokens + response.usage.output_tokens
        
        LLM_TOKEN_USAGE.labels(provider='anthropic', type='total').inc(tokens_used)
        
        return LLMResponse(
            content=content,
            provider=LLMProvider.ANTHROPIC_CLAUDE.value,
            tokens_used=tokens_used,
            processing_time=time.time() - start_time,
            confidence=0.9,  # High confidence for Claude
            metadata={
                'model': 'claude-3-sonnet-20240229',
                'stop_reason': response.stop_reason
            }
        )
    
    async def _execute_ollama(
        self, 
        request: LLMRequest, 
        messages: List[Dict], 
        start_time: float
    ) -> LLMResponse:
        """Execute request using Ollama (local)."""
        # Format messages for Ollama
        prompt = self._format_ollama_prompt(messages)
        
        response = await self.ollama_client.generate(
            model='llama2',  # Configurable
            prompt=prompt,
            options={
                'temperature': request.temperature,
                'num_predict': request.max_tokens
            }
        )
        
        content = response['response']
        tokens_used = len(content.split())  # Approximate
        
        LLM_TOKEN_USAGE.labels(provider='ollama', type='total').inc(tokens_used)
        
        return LLMResponse(
            content=content,
            provider=LLMProvider.OLLAMA_LOCAL.value,
            tokens_used=tokens_used,
            processing_time=time.time() - start_time,
            confidence=0.7,  # Lower confidence for local model
            metadata={
                'model': 'llama2',
                'local': True
            }
        )
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context for LLM consumption."""
        formatted_parts = []
        
        for key, value in context.items():
            if isinstance(value, dict):
                formatted_parts.append(f"{key}: {json.dumps(value, indent=2)}")
            elif isinstance(value, list):
                formatted_parts.append(f"{key}: {', '.join(str(v) for v in value)}")
            else:
                formatted_parts.append(f"{key}: {value}")
        
        return "\n".join(formatted_parts)
    
    def _format_ollama_prompt(self, messages: List[Dict]) -> str:
        """Format messages for Ollama."""
        prompt_parts = []
        
        for msg in messages:
            role = msg['role'].upper()
            content = msg['content']
            prompt_parts.append(f"{role}: {content}")
        
        return "\n\n".join(prompt_parts) + "\n\nASSISTANT:"
    
    async def _test_provider(self, provider: LLMProvider) -> bool:
        """Test provider health."""
        try:
            test_request = LLMRequest(
                prompt="Hello, please respond with 'OK' if you're working.",
                feature_domain="health_check",
                max_tokens=10,
                temperature=0.0
            )
            
            response = await self._execute_request(
                test_request, provider, "You are a helpful assistant."
            )
            
            self.provider_health[provider] = True
            self.logger.info(f"Provider {provider.value} is healthy")
            return True
            
        except Exception as e:
            self.logger.warning(f"Provider {provider.value} health check failed: {e}")
            self.provider_health[provider] = False
            return False
    
    async def _health_monitor(self) -> None:
        """Background health monitoring for all providers."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                for provider in LLMProvider:
                    if not self.provider_health[provider]:
                        await self._test_provider(provider)
                        
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all providers."""
        return {
            "providers": {
                provider.value: {
                    "healthy": self.provider_health[provider],
                    "queue_size": self.request_queues[provider].qsize()
                }
                for provider in LLMProvider
            },
            "total_requests": sum([
                counter._value._value 
                for counter in LLM_REQUEST_COUNTER._metrics.values()
            ]),
            "active_connections": sum([
                gauge._value._value 
                for gauge in LLM_ACTIVE_CONNECTIONS._metrics.values()
            ])
        }
    
    async def shutdown(self) -> None:
        """Shutdown LLM service."""
        self.logger.info("Shutting down Unified LLM Service...")
        
        if self.openai_client:
            await self.openai_client.close()
        
        if self.anthropic_client:
            await self.anthropic_client.close()
        
        self.executor.shutdown(wait=True)
        
        self.logger.info("Unified LLM Service shutdown complete")

# Global LLM service instance
_llm_service: Optional[UnifiedLLMService] = None

def get_llm_service() -> UnifiedLLMService:
    """Get global LLM service instance."""
    global _llm_service
    if _llm_service is None:
        from ..core.config import get_config
        _llm_service = UnifiedLLMService(get_config())
    return _llm_service