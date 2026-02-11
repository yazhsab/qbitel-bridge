"""
QBITEL - Protocol Knowledge Base
Specialized knowledge base for protocol intelligence with enterprise features.
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib

from ..llm.rag_engine import RAGEngine, RAGDocument
from ..core.config import get_config
from ..discovery.protocol_discovery_orchestrator import ProtocolDiscoveryOrchestrator
from prometheus_client import Counter, Histogram

# Metrics
KB_DOCUMENT_COUNTER = Counter(
    "qbitel_kb_documents_total", "Total knowledge base documents", ["collection"]
)
KB_QUERY_COUNTER = Counter(
    "qbitel_kb_queries_total", "Total knowledge base queries", ["type"]
)
KB_UPDATE_DURATION = Histogram(
    "qbitel_kb_update_duration_seconds", "Knowledge base update duration"
)

logger = logging.getLogger(__name__)


@dataclass
class ProtocolKnowledge:
    """Protocol-specific knowledge structure."""

    protocol_name: str
    description: str
    technical_details: Dict[str, Any]
    security_implications: List[str]
    compliance_requirements: List[str]
    field_definitions: List[Dict[str, Any]]
    common_patterns: List[str]
    threat_indicators: List[str]
    performance_characteristics: Dict[str, Any]
    created_at: datetime
    last_updated: datetime
    confidence_score: float = 0.9


@dataclass
class InteractionRecord:
    """Record of copilot interactions for learning."""

    query: str
    response: str
    query_type: str
    confidence: float
    user_feedback: Optional[float] = None
    protocol_context: Optional[str] = None
    timestamp: datetime = None


class ProtocolKnowledgeBase:
    """
    Specialized knowledge base for protocol intelligence.

    Features:
    - Protocol-specific knowledge management
    - Learning from user interactions
    - Automatic knowledge extraction from protocol discoveries
    - Expert knowledge integration
    - Continuous improvement through feedback
    """

    def __init__(self, rag_engine: RAGEngine = None):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)

        # Core components
        self.rag_engine = rag_engine or RAGEngine()

        # Knowledge categories
        self.knowledge_categories = {
            "protocol_definitions": "Detailed protocol specifications and behaviors",
            "security_patterns": "Security threats and vulnerability patterns",
            "compliance_mappings": "Regulatory compliance requirements by protocol",
            "field_schemas": "Protocol field definitions and parsing rules",
            "performance_baselines": "Performance characteristics and benchmarks",
            "threat_intelligence": "Known attack patterns and indicators",
            "best_practices": "Implementation and configuration best practices",
            "troubleshooting": "Common issues and resolution procedures",
        }

        # Learning and feedback
        self.interaction_history: List[InteractionRecord] = []
        self.feedback_threshold = 0.8  # Minimum confidence for auto-learning
        self.learning_batch_size = 50

        # Update tracking
        self.last_knowledge_update = datetime.now()
        self.update_frequency = timedelta(hours=6)  # Update every 6 hours
        self.pending_updates = []

    async def initialize(self) -> None:
        """Initialize the protocol knowledge base."""
        try:
            self.logger.info("Initializing Protocol Knowledge Base...")

            # Initialize RAG engine
            await self.rag_engine.initialize()

            # Load base protocol knowledge
            await self._load_base_protocol_knowledge()

            # Load expert knowledge
            await self._load_expert_knowledge()

            # Start background learning task
            asyncio.create_task(self._continuous_learning_loop())

            self.logger.info("Protocol Knowledge Base initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Protocol Knowledge Base: {e}")
            raise

    async def search_protocol_knowledge(
        self,
        query: str,
        protocol_type: Optional[str] = None,
        knowledge_category: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant protocol knowledge.

        Args:
            query: Search query
            protocol_type: Specific protocol to search for
            knowledge_category: Category of knowledge to search
            limit: Maximum number of results

        Returns:
            List of relevant knowledge documents
        """
        try:
            KB_QUERY_COUNTER.labels(type="search").inc()

            # Enhance query with protocol context
            enhanced_query = query
            if protocol_type:
                enhanced_query = f"{protocol_type} protocol: {query}"

            # Determine collection to search
            collection_name = None
            if knowledge_category and knowledge_category in self.knowledge_categories:
                # Map category to collection
                category_mapping = {
                    "protocol_definitions": "protocol_knowledge",
                    "security_patterns": "security_patterns",
                    "compliance_mappings": "compliance_rules",
                    "field_schemas": "field_definitions",
                    "threat_intelligence": "threat_intelligence",
                }
                collection_name = category_mapping.get(knowledge_category)

            # Search using RAG engine
            results = await self.rag_engine.query_similar(
                enhanced_query,
                collection_name=collection_name,
                n_results=limit,
                similarity_threshold=0.6,
            )

            # Format results for protocol context
            formatted_results = []
            for doc, score in zip(results.documents, results.similarity_scores):
                formatted_results.append(
                    {
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "similarity_score": score,
                        "knowledge_category": doc.metadata.get("category", "general"),
                        "protocol": doc.metadata.get("protocol", "unknown"),
                        "last_updated": doc.metadata.get("last_updated", "unknown"),
                    }
                )

            return formatted_results

        except Exception as e:
            self.logger.error(f"Protocol knowledge search failed: {e}")
            return []

    async def add_protocol_knowledge(self, knowledge: ProtocolKnowledge) -> bool:
        """Add new protocol knowledge to the knowledge base."""
        try:
            start_time = time.time()

            # Create RAG documents for different aspects
            documents = []

            # Main protocol definition
            main_doc = RAGDocument(
                id=f"protocol_{knowledge.protocol_name}_{int(time.time())}",
                content=f"""
                Protocol: {knowledge.protocol_name}
                
                Description:
                {knowledge.description}
                
                Technical Details:
                {json.dumps(knowledge.technical_details, indent=2)}
                
                Security Implications:
                {chr(10).join(f"- {impl}" for impl in knowledge.security_implications)}
                
                Performance Characteristics:
                {json.dumps(knowledge.performance_characteristics, indent=2)}
                """,
                metadata={
                    "protocol": knowledge.protocol_name,
                    "category": "protocol_definition",
                    "confidence": knowledge.confidence_score,
                    "created_at": knowledge.created_at.isoformat(),
                    "last_updated": knowledge.last_updated.isoformat(),
                },
                created_at=knowledge.created_at,
            )
            documents.append(main_doc)

            # Security implications document
            if knowledge.security_implications:
                security_doc = RAGDocument(
                    id=f"security_{knowledge.protocol_name}_{int(time.time())}",
                    content=f"""
                    Security Analysis for {knowledge.protocol_name}:
                    
                    Security Implications:
                    {chr(10).join(f"- {impl}" for impl in knowledge.security_implications)}
                    
                    Threat Indicators:
                    {chr(10).join(f"- {indicator}" for indicator in knowledge.threat_indicators)}
                    
                    Security Recommendations:
                    - Enable encryption where possible
                    - Monitor for unusual traffic patterns
                    - Implement proper access controls
                    """,
                    metadata={
                        "protocol": knowledge.protocol_name,
                        "category": "security_analysis",
                        "confidence": knowledge.confidence_score,
                        "created_at": knowledge.created_at.isoformat(),
                    },
                    created_at=knowledge.created_at,
                )
                documents.append(security_doc)

            # Field definitions document
            if knowledge.field_definitions:
                field_doc = RAGDocument(
                    id=f"fields_{knowledge.protocol_name}_{int(time.time())}",
                    content=f"""
                    Field Definitions for {knowledge.protocol_name}:
                    
                    {json.dumps(knowledge.field_definitions, indent=2)}
                    
                    Common Patterns:
                    {chr(10).join(f"- {pattern}" for pattern in knowledge.common_patterns)}
                    """,
                    metadata={
                        "protocol": knowledge.protocol_name,
                        "category": "field_definitions",
                        "confidence": knowledge.confidence_score,
                        "created_at": knowledge.created_at.isoformat(),
                    },
                    created_at=knowledge.created_at,
                )
                documents.append(field_doc)

            # Store documents in appropriate collections
            collection_mapping = {
                "protocol_definition": "protocol_knowledge",
                "security_analysis": "security_patterns",
                "field_definitions": "field_definitions",
            }

            for doc in documents:
                collection = collection_mapping.get(
                    doc.metadata["category"], "protocol_knowledge"
                )
                await self.rag_engine.add_documents(collection, [doc])
                KB_DOCUMENT_COUNTER.labels(collection=collection).inc()

            KB_UPDATE_DURATION.observe(time.time() - start_time)
            self.logger.info(f"Added protocol knowledge for {knowledge.protocol_name}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to add protocol knowledge: {e}")
            return False

    async def learn_from_interaction(self, interaction: InteractionRecord) -> None:
        """Learn from copilot interaction."""
        try:
            # Store interaction
            self.interaction_history.append(interaction)

            # Extract learnable knowledge if confidence is high
            if interaction.confidence >= self.feedback_threshold:
                await self._extract_knowledge_from_interaction(interaction)

            # Batch process learning
            if len(self.interaction_history) >= self.learning_batch_size:
                await self._process_learning_batch()

        except Exception as e:
            self.logger.error(f"Learning from interaction failed: {e}")

    async def update_from_discovery_results(
        self, discovery_results: Dict[str, Any], protocol_profiles: Dict[str, Any]
    ) -> None:
        """Update knowledge base from protocol discovery results."""
        try:
            self.logger.info("Updating knowledge base from discovery results...")

            for protocol_name, profile in protocol_profiles.items():
                # Create knowledge from discovery profile
                knowledge = await self._create_knowledge_from_profile(
                    protocol_name, profile, discovery_results
                )

                if knowledge:
                    await self.add_protocol_knowledge(knowledge)

            self.last_knowledge_update = datetime.now()

        except Exception as e:
            self.logger.error(f"Failed to update from discovery results: {e}")

    async def get_protocol_summary(self, protocol_name: str) -> Dict[str, Any]:
        """Get comprehensive summary of a protocol."""
        try:
            # Search for all knowledge about the protocol
            results = await self.search_protocol_knowledge(
                f"comprehensive information about {protocol_name}",
                protocol_type=protocol_name,
                limit=10,
            )

            if not results:
                return {"protocol": protocol_name, "status": "not_found"}

            # Aggregate information
            summary = {
                "protocol": protocol_name,
                "description": "",
                "security_level": "unknown",
                "complexity": "unknown",
                "common_uses": [],
                "security_considerations": [],
                "field_count": 0,
                "knowledge_confidence": 0.0,
                "last_updated": "unknown",
            }

            total_confidence = 0.0

            for result in results:
                total_confidence += result["similarity_score"]

                # Extract different types of information
                content = result["content"].lower()

                if "description:" in content:
                    description_start = content.find("description:") + len(
                        "description:"
                    )
                    description_end = content.find("\n\n", description_start)
                    if description_end == -1:
                        description_end = description_start + 200
                    summary["description"] = content[
                        description_start:description_end
                    ].strip()

                if "security" in content:
                    if any(
                        word in content
                        for word in ["high risk", "vulnerable", "insecure"]
                    ):
                        summary["security_level"] = "high_risk"
                    elif any(
                        word in content for word in ["secure", "encrypted", "safe"]
                    ):
                        summary["security_level"] = "secure"
                    else:
                        summary["security_level"] = "moderate"

                # Extract field information
                if "field" in content and "definitions" in content:
                    # Simple field count estimation
                    field_matches = content.count("field")
                    summary["field_count"] = max(summary["field_count"], field_matches)

            summary["knowledge_confidence"] = (
                total_confidence / len(results) if results else 0.0
            )

            return summary

        except Exception as e:
            self.logger.error(f"Failed to get protocol summary: {e}")
            return {"protocol": protocol_name, "status": "error", "error": str(e)}

    async def export_knowledge_base(self, format: str = "json") -> Dict[str, Any]:
        """Export knowledge base for backup or analysis."""
        try:
            stats = self.rag_engine.get_collection_stats()

            export_data = {
                "export_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "format": format,
                    "total_collections": len(stats),
                    "total_documents": sum(
                        stat.get("document_count", 0) for stat in stats.values()
                    ),
                },
                "collection_stats": stats,
                "knowledge_categories": self.knowledge_categories,
                "interaction_stats": {
                    "total_interactions": len(self.interaction_history),
                    "avg_confidence": (
                        sum(i.confidence for i in self.interaction_history)
                        / len(self.interaction_history)
                        if self.interaction_history
                        else 0.0
                    ),
                    "last_update": self.last_knowledge_update.isoformat(),
                },
            }

            return export_data

        except Exception as e:
            self.logger.error(f"Knowledge base export failed: {e}")
            return {"error": str(e)}

    # Private methods for knowledge processing

    async def _load_base_protocol_knowledge(self) -> None:
        """Load foundational protocol knowledge."""
        try:
            # Extended protocol definitions
            base_protocols = [
                ProtocolKnowledge(
                    protocol_name="HTTP",
                    description="HyperText Transfer Protocol - Application layer protocol for web communication",
                    technical_details={
                        "default_ports": [80, 443],
                        "methods": [
                            "GET",
                            "POST",
                            "PUT",
                            "DELETE",
                            "HEAD",
                            "OPTIONS",
                            "PATCH",
                        ],
                        "versions": ["HTTP/1.0", "HTTP/1.1", "HTTP/2", "HTTP/3"],
                        "transport": "TCP",
                        "encryption": "Optional (HTTPS with TLS/SSL)",
                    },
                    security_implications=[
                        "Plain HTTP transmits data in clear text",
                        "Susceptible to man-in-the-middle attacks",
                        "Cookie and session management vulnerabilities",
                        "HTTPS provides encryption and authentication",
                    ],
                    compliance_requirements=[
                        "PCI-DSS requires HTTPS for payment data",
                        "GDPR recommends encryption in transit",
                        "HIPAA requires secure transmission of PHI",
                    ],
                    field_definitions=[
                        {
                            "name": "method",
                            "type": "string",
                            "description": "HTTP request method",
                        },
                        {
                            "name": "uri",
                            "type": "string",
                            "description": "Resource identifier",
                        },
                        {
                            "name": "version",
                            "type": "string",
                            "description": "HTTP version",
                        },
                        {
                            "name": "headers",
                            "type": "key-value",
                            "description": "Request/response headers",
                        },
                        {
                            "name": "body",
                            "type": "bytes",
                            "description": "Message body",
                        },
                    ],
                    common_patterns=[
                        "Request-response pattern",
                        "Stateless operation",
                        "Header-based metadata transmission",
                    ],
                    threat_indicators=[
                        "Unusual user-agent strings",
                        "Suspicious request patterns",
                        "Large payload sizes",
                        "Frequent requests from single source",
                    ],
                    performance_characteristics={
                        "latency": "Low to moderate",
                        "throughput": "High",
                        "connection_overhead": "Moderate",
                        "caching": "Extensive support",
                    },
                    created_at=datetime.now(),
                    last_updated=datetime.now(),
                    confidence_score=0.95,
                ),
                ProtocolKnowledge(
                    protocol_name="DNS",
                    description="Domain Name System - Hierarchical naming system for network resources",
                    technical_details={
                        "default_port": 53,
                        "transport": "UDP (primary), TCP (for large responses)",
                        "record_types": [
                            "A",
                            "AAAA",
                            "CNAME",
                            "MX",
                            "NS",
                            "PTR",
                            "SOA",
                            "TXT",
                        ],
                        "query_types": ["Recursive", "Iterative"],
                        "security_extensions": [
                            "DNSSEC",
                            "DNS over HTTPS (DoH)",
                            "DNS over TLS (DoT)",
                        ],
                    },
                    security_implications=[
                        "DNS spoofing and cache poisoning attacks",
                        "DNS tunneling for data exfiltration",
                        "DDoS amplification attacks",
                        "Privacy concerns with plain DNS queries",
                    ],
                    compliance_requirements=[
                        "NIST recommends DNSSEC implementation",
                        "Some regulations require secure DNS resolution",
                    ],
                    field_definitions=[
                        {
                            "name": "header",
                            "type": "struct",
                            "description": "DNS message header",
                        },
                        {
                            "name": "question",
                            "type": "struct",
                            "description": "Query information",
                        },
                        {
                            "name": "answer",
                            "type": "resource_record",
                            "description": "Response records",
                        },
                        {
                            "name": "authority",
                            "type": "resource_record",
                            "description": "Authority records",
                        },
                        {
                            "name": "additional",
                            "type": "resource_record",
                            "description": "Additional records",
                        },
                    ],
                    common_patterns=[
                        "Query-response pattern",
                        "Hierarchical resolution",
                        "Caching mechanism",
                    ],
                    threat_indicators=[
                        "Queries to suspicious domains",
                        "Unusual query patterns",
                        "DNS tunneling signatures",
                        "Response size anomalies",
                    ],
                    performance_characteristics={
                        "latency": "Very low",
                        "cache_hit_ratio": "High",
                        "query_rate": "Very high",
                        "availability_requirement": "Critical",
                    },
                    created_at=datetime.now(),
                    last_updated=datetime.now(),
                    confidence_score=0.95,
                ),
            ]

            # Add base protocols to knowledge base
            for protocol in base_protocols:
                await self.add_protocol_knowledge(protocol)

            self.logger.info("Base protocol knowledge loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load base protocol knowledge: {e}")

    async def _load_expert_knowledge(self) -> None:
        """Load expert knowledge from various sources."""
        try:
            # Security best practices
            security_docs = [
                RAGDocument(
                    id="security_best_practices_general",
                    content="""
                    Protocol Security Best Practices:
                    
                    1. Encryption in Transit:
                       - Always use TLS/SSL for sensitive data
                       - Implement perfect forward secrecy
                       - Use strong cipher suites
                    
                    2. Authentication and Authorization:
                       - Implement strong authentication mechanisms
                       - Use role-based access control
                       - Validate all inputs
                    
                    3. Monitoring and Logging:
                       - Log all security-relevant events
                       - Monitor for unusual patterns
                       - Implement real-time alerting
                    
                    4. Protocol Hardening:
                       - Disable unnecessary features
                       - Use the latest protocol versions
                       - Implement proper error handling
                    """,
                    metadata={
                        "category": "security_best_practices",
                        "confidence": 0.9,
                        "source": "security_expert",
                    },
                    created_at=datetime.now(),
                ),
                RAGDocument(
                    id="threat_detection_patterns",
                    content="""
                    Common Attack Patterns in Network Protocols:
                    
                    1. Man-in-the-Middle Attacks:
                       - Certificate validation bypassing
                       - Protocol downgrade attacks
                       - Session hijacking
                    
                    2. Denial of Service Attacks:
                       - Resource exhaustion attacks
                       - Protocol flooding
                       - Amplification attacks
                    
                    3. Data Exfiltration:
                       - DNS tunneling
                       - HTTP/HTTPS covert channels
                       - Protocol field abuse
                    
                    4. Injection Attacks:
                       - SQL injection via HTTP parameters
                       - Command injection in protocol fields
                       - Header injection attacks
                    """,
                    metadata={
                        "category": "threat_patterns",
                        "confidence": 0.9,
                        "source": "threat_intelligence",
                    },
                    created_at=datetime.now(),
                ),
            ]

            await self.rag_engine.add_documents("security_patterns", security_docs)

            self.logger.info("Expert knowledge loaded successfully")

        except Exception as e:
            self.logger.error(f"Failed to load expert knowledge: {e}")

    async def _extract_knowledge_from_interaction(
        self, interaction: InteractionRecord
    ) -> None:
        """Extract learnable knowledge from user interaction."""
        try:
            # Simple knowledge extraction based on query patterns
            query_lower = interaction.query.lower()

            # Extract protocol mentions
            protocols = [
                "http",
                "https",
                "tcp",
                "udp",
                "dns",
                "ftp",
                "ssh",
                "tls",
                "ssl",
            ]
            mentioned_protocols = [p for p in protocols if p in query_lower]

            if mentioned_protocols and interaction.confidence > 0.8:
                # Create a knowledge entry from the interaction
                knowledge_content = f"""
                User Query: {interaction.query}
                
                Assistant Response: {interaction.response}
                
                Context: {interaction.protocol_context or 'General protocol discussion'}
                
                Query Type: {interaction.query_type}
                Confidence: {interaction.confidence}
                Protocols Mentioned: {', '.join(mentioned_protocols)}
                """

                doc = RAGDocument(
                    id=f"learned_{int(time.time())}_{hash(interaction.query) % 10000}",
                    content=knowledge_content,
                    metadata={
                        "category": "learned_knowledge",
                        "protocols": mentioned_protocols,
                        "confidence": interaction.confidence,
                        "query_type": interaction.query_type,
                        "source": "user_interaction",
                        "created_at": datetime.now().isoformat(),
                    },
                    created_at=datetime.now(),
                )

                # Store in appropriate collection
                await self.rag_engine.add_documents("protocol_knowledge", [doc])

                self.logger.debug(
                    f"Learned from interaction: {interaction.query[:50]}..."
                )

        except Exception as e:
            self.logger.error(f"Failed to extract knowledge from interaction: {e}")

    async def _process_learning_batch(self) -> None:
        """Process a batch of interactions for learning."""
        try:
            # Simple batch processing - in production, this would be more sophisticated
            recent_interactions = self.interaction_history[-self.learning_batch_size :]

            # Analyze patterns
            query_types = {}
            protocol_mentions = {}

            for interaction in recent_interactions:
                query_types[interaction.query_type] = (
                    query_types.get(interaction.query_type, 0) + 1
                )

                if interaction.protocol_context:
                    protocol_mentions[interaction.protocol_context] = (
                        protocol_mentions.get(interaction.protocol_context, 0) + 1
                    )

            # Create learning summary
            learning_summary = f"""
            Learning Summary from {len(recent_interactions)} interactions:
            
            Most common query types:
            {json.dumps(dict(sorted(query_types.items(), key=lambda x: x[1], reverse=True)[:5]), indent=2)}
            
            Most discussed protocols:
            {json.dumps(dict(sorted(protocol_mentions.items(), key=lambda x: x[1], reverse=True)[:5]), indent=2)}
            
            Average confidence: {sum(i.confidence for i in recent_interactions) / len(recent_interactions):.2f}
            """

            # Store learning summary
            doc = RAGDocument(
                id=f"learning_summary_{int(time.time())}",
                content=learning_summary,
                metadata={
                    "category": "learning_analytics",
                    "batch_size": len(recent_interactions),
                    "source": "batch_learning",
                    "created_at": datetime.now().isoformat(),
                },
                created_at=datetime.now(),
            )

            await self.rag_engine.add_documents("protocol_knowledge", [doc])

            # Clear processed interactions to prevent memory buildup
            self.interaction_history = self.interaction_history[
                self.learning_batch_size :
            ]

            self.logger.info(
                f"Processed learning batch of {len(recent_interactions)} interactions"
            )

        except Exception as e:
            self.logger.error(f"Failed to process learning batch: {e}")

    async def _create_knowledge_from_profile(
        self,
        protocol_name: str,
        profile: Dict[str, Any],
        discovery_results: Dict[str, Any],
    ) -> Optional[ProtocolKnowledge]:
        """Create knowledge from discovery profile."""
        try:
            # Extract information from profile
            description = f"Protocol discovered through analysis with {profile.get('usage_count', 0)} observations"

            # Technical details from grammar
            technical_details = {
                "grammar_rules": profile.get("grammar", {}).get("rules_count", 0),
                "symbols_identified": profile.get("grammar", {}).get(
                    "symbols_count", 0
                ),
                "parser_available": profile.get("parser") is not None,
                "discovery_confidence": profile.get("average_confidence", 0.0),
            }

            # Basic security implications
            security_implications = [
                "Protocol behavior analyzed through traffic observation",
                "Security assessment needed for production use",
                "Monitor for unusual traffic patterns",
            ]

            # Performance characteristics from statistics
            performance_characteristics = profile.get("statistics", {})

            return ProtocolKnowledge(
                protocol_name=protocol_name,
                description=description,
                technical_details=technical_details,
                security_implications=security_implications,
                compliance_requirements=[],  # Would need compliance analysis
                field_definitions=[],  # Would extract from parser if available
                common_patterns=[],
                threat_indicators=[],
                performance_characteristics=performance_characteristics,
                created_at=datetime.now(),
                last_updated=datetime.now(),
                confidence_score=profile.get("average_confidence", 0.7),
            )

        except Exception as e:
            self.logger.error(f"Failed to create knowledge from profile: {e}")
            return None

    async def _continuous_learning_loop(self) -> None:
        """Background task for continuous learning and knowledge updates."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour

                # Check if it's time for a knowledge update
                if datetime.now() - self.last_knowledge_update > self.update_frequency:
                    await self._perform_scheduled_update()

                # Process any pending updates
                if self.pending_updates:
                    await self._process_pending_updates()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in continuous learning loop: {e}")

    async def _perform_scheduled_update(self) -> None:
        """Perform scheduled knowledge base updates."""
        try:
            self.logger.info("Performing scheduled knowledge base update...")

            # Clear old cached queries in RAG engine
            self.rag_engine.clear_cache()

            # Update knowledge statistics
            stats = self.rag_engine.get_collection_stats()
            self.logger.info(f"Knowledge base stats: {stats}")

            self.last_knowledge_update = datetime.now()

        except Exception as e:
            self.logger.error(f"Scheduled update failed: {e}")

    async def _process_pending_updates(self) -> None:
        """Process pending knowledge updates."""
        try:
            updates_to_process = self.pending_updates[:10]  # Process up to 10 at a time
            self.pending_updates = self.pending_updates[10:]

            for update in updates_to_process:
                # Process update based on type
                if update.get("type") == "protocol_knowledge":
                    await self.add_protocol_knowledge(update["data"])
                elif update.get("type") == "interaction_learning":
                    await self.learn_from_interaction(update["data"])

            if updates_to_process:
                self.logger.info(f"Processed {len(updates_to_process)} pending updates")

        except Exception as e:
            self.logger.error(f"Failed to process pending updates: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        return {
            "rag_engine_stats": self.rag_engine.get_collection_stats(),
            "knowledge_categories": len(self.knowledge_categories),
            "interaction_history_size": len(self.interaction_history),
            "last_update": self.last_knowledge_update.isoformat(),
            "pending_updates": len(self.pending_updates),
            "learning_threshold": self.feedback_threshold,
            "batch_size": self.learning_batch_size,
        }
