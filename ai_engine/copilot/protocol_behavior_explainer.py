"""
CRONOS AI - Protocol Behavior Explainer
Provides natural language explanations of protocol behavior and message patterns.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

from ..core.exceptions import CronosAIException
from ..llm.unified_llm_service import UnifiedLLMService, LLMRequest
from ..llm.rag_engine import RAGEngine

logger = logging.getLogger(__name__)

@dataclass
class ProtocolBehaviorQuery:
    """Query for protocol behavior explanation."""
    protocol_type: str
    messages: List[bytes]
    question: str
    context: Dict[str, Any] = field(default_factory=dict)
    include_examples: bool = True
    detail_level: str = "standard"  # basic, standard, detailed

@dataclass
class BehaviorExplanation:
    """Protocol behavior explanation."""
    explanation: str
    key_observations: List[str]
    message_patterns: List[Dict[str, Any]]
    sequence_analysis: Optional[str] = None
    security_implications: List[str] = field(default_factory=list)
    performance_notes: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.8
    sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MessagePattern:
    """Identified message pattern."""
    pattern_type: str
    description: str
    frequency: int
    example_messages: List[bytes]
    characteristics: Dict[str, Any]
    significance: str  # low, medium, high

class ProtocolBehaviorExplainer:
    """
    Explains protocol behavior in natural language.
    
    Analyzes message sequences, identifies patterns, and provides
    comprehensive explanations of protocol behavior with examples.
    """
    
    def __init__(self, llm_service: UnifiedLLMService, rag_engine: RAGEngine):
        self.llm_service = llm_service
        self.rag_engine = rag_engine
        self.logger = logging.getLogger(__name__)
        
        # Pattern recognition templates
        self.pattern_templates = {
            'request_response': {
                'description': 'Request-response pattern',
                'indicators': ['request', 'response', 'query', 'reply']
            },
            'handshake': {
                'description': 'Connection handshake sequence',
                'indicators': ['syn', 'ack', 'hello', 'init']
            },
            'streaming': {
                'description': 'Continuous data streaming',
                'indicators': ['stream', 'chunk', 'fragment', 'segment']
            },
            'heartbeat': {
                'description': 'Keep-alive or heartbeat messages',
                'indicators': ['ping', 'pong', 'keepalive', 'heartbeat']
            },
            'error_handling': {
                'description': 'Error or exception handling',
                'indicators': ['error', 'exception', 'fail', 'retry']
            }
        }
    
    async def explain_protocol_behavior(
        self,
        query: ProtocolBehaviorQuery
    ) -> BehaviorExplanation:
        """
        Explain protocol behavior based on message analysis.
        
        Args:
            query: Protocol behavior query with messages and question
            
        Returns:
            Comprehensive behavior explanation
        """
        try:
            self.logger.info(f"Explaining behavior for {query.protocol_type} protocol")
            
            # Analyze message patterns
            patterns = await self._analyze_message_patterns(
                query.messages,
                query.protocol_type
            )
            
            # Analyze message sequence
            sequence_analysis = await self._analyze_message_sequence(
                query.messages,
                query.protocol_type
            )
            
            # Get relevant protocol knowledge
            protocol_knowledge = await self._get_protocol_knowledge(
                query.protocol_type,
                query.question
            )
            
            # Generate explanation using LLM
            explanation = await self._generate_explanation(
                query,
                patterns,
                sequence_analysis,
                protocol_knowledge
            )
            
            # Extract key observations
            key_observations = self._extract_key_observations(
                patterns,
                sequence_analysis
            )
            
            # Identify security implications
            security_implications = await self._identify_security_implications(
                query.protocol_type,
                patterns,
                query.messages
            )
            
            # Generate examples if requested
            examples = []
            if query.include_examples:
                examples = self._generate_examples(patterns, query.messages)
            
            return BehaviorExplanation(
                explanation=explanation,
                key_observations=key_observations,
                message_patterns=[self._pattern_to_dict(p) for p in patterns],
                sequence_analysis=sequence_analysis,
                security_implications=security_implications,
                performance_notes=self._analyze_performance_characteristics(patterns),
                examples=examples,
                confidence=0.85,
                sources=self._extract_sources(protocol_knowledge),
                metadata={
                    'protocol_type': query.protocol_type,
                    'message_count': len(query.messages),
                    'pattern_count': len(patterns),
                    'detail_level': query.detail_level,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to explain protocol behavior: {e}")
            raise CronosAIException(f"Behavior explanation failed: {e}")
    
    async def _analyze_message_patterns(
        self,
        messages: List[bytes],
        protocol_type: str
    ) -> List[MessagePattern]:
        """Analyze and identify message patterns."""
        patterns = []
        
        try:
            # Group similar messages
            message_groups = self._group_similar_messages(messages)
            
            # Identify patterns in each group
            for group_type, group_messages in message_groups.items():
                if len(group_messages) < 2:
                    continue
                
                # Analyze characteristics
                characteristics = self._analyze_message_characteristics(group_messages)
                
                # Determine pattern type
                pattern_type = self._classify_pattern_type(
                    group_type,
                    characteristics,
                    protocol_type
                )
                
                # Create pattern
                pattern = MessagePattern(
                    pattern_type=pattern_type,
                    description=self._get_pattern_description(pattern_type),
                    frequency=len(group_messages),
                    example_messages=group_messages[:3],  # First 3 examples
                    characteristics=characteristics,
                    significance=self._assess_pattern_significance(
                        pattern_type,
                        len(group_messages),
                        len(messages)
                    )
                )
                
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Pattern analysis failed: {e}")
            return []
    
    def _group_similar_messages(self, messages: List[bytes]) -> Dict[str, List[bytes]]:
        """Group similar messages together."""
        groups = {}
        
        for message in messages:
            # Simple grouping by message size and first few bytes
            if len(message) == 0:
                continue
            
            # Create group key based on size range and first bytes
            size_range = (len(message) // 100) * 100
            first_bytes = message[:4].hex() if len(message) >= 4 else message.hex()
            group_key = f"size_{size_range}_start_{first_bytes[:8]}"
            
            if group_key not in groups:
                groups[group_key] = []
            
            groups[group_key].append(message)
        
        return groups
    
    def _analyze_message_characteristics(
        self,
        messages: List[bytes]
    ) -> Dict[str, Any]:
        """Analyze characteristics of message group."""
        if not messages:
            return {}
        
        sizes = [len(m) for m in messages]
        
        return {
            'count': len(messages),
            'avg_size': sum(sizes) / len(sizes),
            'min_size': min(sizes),
            'max_size': max(sizes),
            'size_variance': self._calculate_variance(sizes),
            'has_text': any(self._contains_text(m) for m in messages),
            'has_binary': any(not self._contains_text(m) for m in messages),
            'common_bytes': self._find_common_bytes(messages)
        }
    
    def _contains_text(self, data: bytes) -> bool:
        """Check if data contains text."""
        try:
            # Try to decode as UTF-8
            text = data.decode('utf-8', errors='strict')
            # Check if mostly printable
            printable = sum(1 for c in text if c.isprintable() or c.isspace())
            return printable / len(text) > 0.7 if text else False
        except:
            return False
    
    def _find_common_bytes(self, messages: List[bytes]) -> List[int]:
        """Find bytes common across all messages."""
        if not messages:
            return []
        
        # Find common positions
        min_len = min(len(m) for m in messages)
        common = []
        
        for i in range(min(min_len, 10)):  # Check first 10 bytes
            bytes_at_pos = [m[i] for m in messages if len(m) > i]
            if len(set(bytes_at_pos)) == 1:  # All same
                common.append(bytes_at_pos[0])
        
        return common
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values."""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def _classify_pattern_type(
        self,
        group_type: str,
        characteristics: Dict[str, Any],
        protocol_type: str
    ) -> str:
        """Classify the type of pattern."""
        # Check against known patterns
        for pattern_name, template in self.pattern_templates.items():
            if any(indicator in group_type.lower() 
                   for indicator in template['indicators']):
                return pattern_name
        
        # Classify based on characteristics
        if characteristics.get('size_variance', 0) < 10:
            return 'fixed_format'
        elif characteristics.get('has_text'):
            return 'text_based'
        else:
            return 'binary_protocol'
    
    def _get_pattern_description(self, pattern_type: str) -> str:
        """Get description for pattern type."""
        if pattern_type in self.pattern_templates:
            return self.pattern_templates[pattern_type]['description']
        
        descriptions = {
            'fixed_format': 'Fixed-format messages with consistent structure',
            'text_based': 'Text-based protocol messages',
            'binary_protocol': 'Binary protocol with variable structure'
        }
        
        return descriptions.get(pattern_type, 'Unclassified message pattern')
    
    def _assess_pattern_significance(
        self,
        pattern_type: str,
        frequency: int,
        total_messages: int
    ) -> str:
        """Assess significance of pattern."""
        ratio = frequency / max(total_messages, 1)
        
        # High significance patterns
        if pattern_type in ['handshake', 'error_handling']:
            return 'high'
        
        # Based on frequency
        if ratio > 0.5:
            return 'high'
        elif ratio > 0.2:
            return 'medium'
        else:
            return 'low'
    
    async def _analyze_message_sequence(
        self,
        messages: List[bytes],
        protocol_type: str
    ) -> str:
        """Analyze the sequence of messages."""
        if len(messages) < 2:
            return "Insufficient messages for sequence analysis."
        
        analysis_parts = []
        
        # Analyze message flow
        analysis_parts.append(f"Message Sequence Analysis ({len(messages)} messages):")
        
        # Check for alternating patterns
        sizes = [len(m) for m in messages]
        if self._is_alternating(sizes):
            analysis_parts.append("- Alternating message sizes detected (likely request-response pattern)")
        
        # Check for increasing/decreasing trends
        if self._is_increasing(sizes):
            analysis_parts.append("- Message sizes increasing (possible data transfer or streaming)")
        elif self._is_decreasing(sizes):
            analysis_parts.append("- Message sizes decreasing (possible connection teardown)")
        
        # Check for periodic patterns
        if self._has_periodic_pattern(sizes):
            analysis_parts.append("- Periodic pattern detected (possible heartbeat or polling)")
        
        # Analyze timing if available
        analysis_parts.append(f"- Message size range: {min(sizes)} to {max(sizes)} bytes")
        analysis_parts.append(f"- Average message size: {sum(sizes) / len(sizes):.1f} bytes")
        
        return '\n'.join(analysis_parts)
    
    def _is_alternating(self, values: List[float]) -> bool:
        """Check if values alternate between high and low."""
        if len(values) < 4:
            return False
        
        median = sorted(values)[len(values) // 2]
        above_median = [v > median for v in values]
        
        # Check for alternating pattern
        alternations = sum(1 for i in range(len(above_median) - 1) 
                          if above_median[i] != above_median[i + 1])
        
        return alternations > len(values) * 0.6
    
    def _is_increasing(self, values: List[float]) -> bool:
        """Check if values are generally increasing."""
        if len(values) < 3:
            return False
        
        increases = sum(1 for i in range(len(values) - 1) if values[i + 1] > values[i])
        return increases > len(values) * 0.7
    
    def _is_decreasing(self, values: List[float]) -> bool:
        """Check if values are generally decreasing."""
        if len(values) < 3:
            return False
        
        decreases = sum(1 for i in range(len(values) - 1) if values[i + 1] < values[i])
        return decreases > len(values) * 0.7
    
    def _has_periodic_pattern(self, values: List[float]) -> bool:
        """Check for periodic patterns in values."""
        if len(values) < 6:
            return False
        
        # Simple periodicity check
        for period in range(2, min(6, len(values) // 2)):
            matches = 0
            for i in range(len(values) - period):
                if abs(values[i] - values[i + period]) < values[i] * 0.1:
                    matches += 1
            
            if matches > len(values) * 0.5:
                return True
        
        return False
    
    async def _get_protocol_knowledge(
        self,
        protocol_type: str,
        question: str
    ) -> List[Dict[str, Any]]:
        """Get relevant protocol knowledge from RAG."""
        try:
            query = f"{protocol_type} protocol behavior {question}"
            results = await self.rag_engine.query_similar(
                query,
                collection_name='protocol_knowledge',
                n_results=5,
                similarity_threshold=0.6
            )
            
            return [
                {
                    'content': doc.content,
                    'metadata': doc.metadata,
                    'similarity': score
                }
                for doc, score in zip(results.documents, results.similarity_scores)
            ]
            
        except Exception as e:
            self.logger.error(f"Failed to get protocol knowledge: {e}")
            return []
    
    async def _generate_explanation(
        self,
        query: ProtocolBehaviorQuery,
        patterns: List[MessagePattern],
        sequence_analysis: str,
        protocol_knowledge: List[Dict[str, Any]]
    ) -> str:
        """Generate natural language explanation using LLM."""
        try:
            # Build context
            context_parts = [
                f"Protocol: {query.protocol_type}",
                f"Question: {query.question}",
                f"\nMessage Analysis:",
                f"- Total messages: {len(query.messages)}",
                f"- Identified patterns: {len(patterns)}",
                f"\nSequence Analysis:",
                sequence_analysis
            ]
            
            # Add pattern information
            if patterns:
                context_parts.append("\nIdentified Patterns:")
                for pattern in patterns[:5]:
                    context_parts.append(
                        f"- {pattern.description} "
                        f"(frequency: {pattern.frequency}, "
                        f"significance: {pattern.significance})"
                    )
            
            # Add protocol knowledge
            if protocol_knowledge:
                context_parts.append("\nRelevant Protocol Knowledge:")
                for knowledge in protocol_knowledge[:3]:
                    context_parts.append(f"- {knowledge['content'][:200]}...")
            
            context_str = '\n'.join(context_parts)
            
            # Request LLM explanation
            llm_request = LLMRequest(
                prompt=f"""
                Explain the protocol behavior based on the following analysis.
                
                {context_str}
                
                Provide a clear, comprehensive explanation that:
                1. Directly answers the user's question
                2. Explains the observed message patterns
                3. Describes the protocol behavior and purpose
                4. Highlights any notable characteristics
                5. Provides practical insights
                
                Use natural language and be specific about what the messages show.
                Detail level: {query.detail_level}
                """,
                feature_domain="protocol_copilot",
                max_tokens=1500,
                temperature=0.3
            )
            
            llm_response = await self.llm_service.process_request(llm_request)
            return llm_response.content
            
        except Exception as e:
            self.logger.error(f"Failed to generate explanation: {e}")
            return f"Unable to generate detailed explanation. Observed {len(patterns)} message patterns in {len(query.messages)} messages."
    
    def _extract_key_observations(
        self,
        patterns: List[MessagePattern],
        sequence_analysis: str
    ) -> List[str]:
        """Extract key observations from analysis."""
        observations = []
        
        # Pattern-based observations
        for pattern in patterns:
            if pattern.significance == 'high':
                observations.append(
                    f"{pattern.description} appears {pattern.frequency} times"
                )
        
        # Sequence-based observations
        if "alternating" in sequence_analysis.lower():
            observations.append("Request-response pattern detected")
        
        if "increasing" in sequence_analysis.lower():
            observations.append("Progressive data transfer observed")
        
        if "periodic" in sequence_analysis.lower():
            observations.append("Periodic communication pattern present")
        
        return observations[:5]  # Top 5 observations
    
    async def _identify_security_implications(
        self,
        protocol_type: str,
        patterns: List[MessagePattern],
        messages: List[bytes]
    ) -> List[str]:
        """Identify security implications."""
        implications = []
        
        # Check for unencrypted data
        text_messages = sum(1 for m in messages if self._contains_text(m))
        if text_messages > len(messages) * 0.5:
            implications.append(
                "Messages appear to contain unencrypted text data"
            )
        
        # Check for large messages (potential DoS)
        large_messages = sum(1 for m in messages if len(m) > 10000)
        if large_messages > 0:
            implications.append(
                f"{large_messages} large messages detected (potential DoS risk)"
            )
        
        # Check for error patterns
        error_patterns = [p for p in patterns if 'error' in p.pattern_type.lower()]
        if error_patterns:
            implications.append(
                "Error handling patterns observed (review error information disclosure)"
            )
        
        return implications
    
    def _analyze_performance_characteristics(
        self,
        patterns: List[MessagePattern]
    ) -> List[str]:
        """Analyze performance characteristics."""
        notes = []
        
        # Check message frequency
        total_messages = sum(p.frequency for p in patterns)
        if total_messages > 100:
            notes.append(
                f"High message volume ({total_messages} messages) - consider batching"
            )
        
        # Check for small messages
        small_message_patterns = [
            p for p in patterns 
            if p.characteristics.get('avg_size', 0) < 100
        ]
        if small_message_patterns:
            notes.append(
                "Multiple small messages detected - potential for optimization"
            )
        
        # Check for large messages
        large_message_patterns = [
            p for p in patterns 
            if p.characteristics.get('avg_size', 0) > 10000
        ]
        if large_message_patterns:
            notes.append(
                "Large messages present - consider streaming or chunking"
            )
        
        return notes
    
    def _generate_examples(
        self,
        patterns: List[MessagePattern],
        messages: List[bytes]
    ) -> List[Dict[str, Any]]:
        """Generate example explanations."""
        examples = []
        
        for pattern in patterns[:3]:  # Top 3 patterns
            if pattern.example_messages:
                example_msg = pattern.example_messages[0]
                
                examples.append({
                    'pattern_type': pattern.pattern_type,
                    'description': pattern.description,
                    'example_size': len(example_msg),
                    'example_preview': self._format_message_preview(example_msg),
                    'characteristics': pattern.characteristics
                })
        
        return examples
    
    def _format_message_preview(self, message: bytes) -> str:
        """Format message preview for display."""
        if self._contains_text(message):
            try:
                text = message.decode('utf-8', errors='replace')
                return text[:100] + ('...' if len(text) > 100 else '')
            except:
                pass
        
        # Show hex preview
        hex_str = message[:50].hex()
        return f"0x{hex_str}" + ('...' if len(message) > 50 else '')
    
    def _extract_sources(self, protocol_knowledge: List[Dict[str, Any]]) -> List[str]:
        """Extract source references."""
        sources = []
        
        for knowledge in protocol_knowledge:
            metadata = knowledge.get('metadata', {})
            source = metadata.get('source', 'knowledge_base')
            if source not in sources:
                sources.append(source)
        
        return sources
    
    def _pattern_to_dict(self, pattern: MessagePattern) -> Dict[str, Any]:
        """Convert pattern to dictionary."""
        return {
            'pattern_type': pattern.pattern_type,
            'description': pattern.description,
            'frequency': pattern.frequency,
            'significance': pattern.significance,
            'characteristics': pattern.characteristics
        }