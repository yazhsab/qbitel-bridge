"""
CRONOS AI - Retrieval Augmented Generation (RAG) Engine
Advanced RAG implementation for protocol intelligence with vector similarity search.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import chromadb
from sentence_transformers import SentenceTransformer
import json
from datetime import datetime
from prometheus_client import Counter, Histogram

# Metrics
RAG_QUERY_COUNTER = Counter('cronos_rag_queries_total', 'Total RAG queries', ['query_type'])
RAG_QUERY_DURATION = Histogram('cronos_rag_query_duration_seconds', 'RAG query duration')
RAG_SIMILARITY_SCORE = Histogram('cronos_rag_similarity_score', 'RAG similarity scores')

logger = logging.getLogger(__name__)

@dataclass
class RAGDocument:
    """Document structure for RAG storage."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    created_at: datetime = None

@dataclass
class RAGQueryResult:
    """RAG query result structure."""
    documents: List[RAGDocument]
    similarity_scores: List[float]
    query_embedding: List[float]
    processing_time: float
    total_results: int

class RAGEngine:
    """
    Enterprise RAG engine for protocol intelligence with ChromaDB and SentenceTransformers.
    Provides semantic search capabilities for protocol knowledge and documentation.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=self.config.get('chroma_db_path', './data/chroma_db')
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(
            self.config.get('embedding_model', 'all-MiniLM-L6-v2')
        )
        
        # Collections for different knowledge types
        self.collections = {}
        self.collection_names = [
            'protocol_knowledge',
            'compliance_rules',
            'security_patterns',
            'field_definitions',
            'threat_intelligence'
        ]
        
        # Query cache
        self.query_cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    async def initialize(self) -> None:
        """Initialize RAG engine and create collections."""
        try:
            self.logger.info("Initializing RAG Engine...")
            
            # Create or get collections
            for collection_name in self.collection_names:
                try:
                    collection = self.chroma_client.get_collection(name=collection_name)
                    self.logger.info(f"Loaded existing collection: {collection_name}")
                except:
                    collection = self.chroma_client.create_collection(
                        name=collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    self.logger.info(f"Created new collection: {collection_name}")
                
                self.collections[collection_name] = collection
            
            # Load initial knowledge base
            await self._load_initial_knowledge()
            
            self.logger.info("RAG Engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG engine: {e}")
            raise
    
    async def add_documents(
        self, 
        collection_name: str, 
        documents: List[RAGDocument]
    ) -> None:
        """Add documents to a specific collection."""
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} not found")
        
        collection = self.collections[collection_name]
        
        # Generate embeddings for documents
        texts = [doc.content for doc in documents]
        embeddings = self.embedding_model.encode(texts).tolist()
        
        # Prepare data for ChromaDB
        ids = [doc.id for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Store in ChromaDB
        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        self.logger.info(f"Added {len(documents)} documents to {collection_name}")
    
    async def query_similar(
        self, 
        query: str, 
        collection_name: str = None,
        n_results: int = 5,
        similarity_threshold: float = 0.7
    ) -> RAGQueryResult:
        """
        Query for similar documents using semantic search.
        
        Args:
            query: Search query
            collection_name: Specific collection to search (None for all)
            n_results: Number of results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            RAG query result with similar documents
        """
        start_time = time.time()
        
        try:
            # Check cache
            cache_key = f"{query}_{collection_name}_{n_results}_{similarity_threshold}"
            if cache_key in self.query_cache:
                cache_entry = self.query_cache[cache_key]
                if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                    return cache_entry['result']
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            
            all_documents = []
            all_scores = []
            
            # Search in specified collection or all collections
            collections_to_search = (
                [collection_name] if collection_name 
                else self.collection_names
            )
            
            for coll_name in collections_to_search:
                if coll_name in self.collections:
                    collection = self.collections[coll_name]
                    
                    # Query ChromaDB
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=n_results,
                        include=['documents', 'metadatas', 'distances']
                    )
                    
                    # Process results
                    if results['documents'] and results['documents'][0]:
                        for i, (doc, metadata, distance) in enumerate(zip(
                            results['documents'][0],
                            results['metadatas'][0],
                            results['distances'][0]
                        )):
                            # Convert distance to similarity score
                            similarity = 1.0 - distance
                            
                            if similarity >= similarity_threshold:
                                rag_doc = RAGDocument(
                                    id=results['ids'][0][i] if results['ids'] else f"{coll_name}_{i}",
                                    content=doc,
                                    metadata={
                                        **metadata,
                                        'collection': coll_name,
                                        'similarity': similarity
                                    }
                                )
                                all_documents.append(rag_doc)
                                all_scores.append(similarity)
            
            # Sort by similarity score
            if all_documents:
                sorted_pairs = sorted(
                    zip(all_documents, all_scores),
                    key=lambda x: x[1],
                    reverse=True
                )
                all_documents, all_scores = zip(*sorted_pairs)
                all_documents = list(all_documents)[:n_results]
                all_scores = list(all_scores)[:n_results]
            
            result = RAGQueryResult(
                documents=all_documents,
                similarity_scores=all_scores,
                query_embedding=query_embedding,
                processing_time=time.time() - start_time,
                total_results=len(all_documents)
            )
            
            # Cache result
            self.query_cache[cache_key] = {
                'result': result,
                'timestamp': time.time()
            }
            
            # Update metrics
            RAG_QUERY_COUNTER.labels(query_type=collection_name or 'all').inc()
            RAG_QUERY_DURATION.observe(result.processing_time)
            
            for score in all_scores:
                RAG_SIMILARITY_SCORE.observe(score)
            
            return result
            
        except Exception as e:
            self.logger.error(f"RAG query failed: {e}")
            return RAGQueryResult(
                documents=[],
                similarity_scores=[],
                query_embedding=[],
                processing_time=time.time() - start_time,
                total_results=0
            )
    
    async def enhance_query_context(
        self, 
        query: str, 
        existing_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Enhance query with relevant context from knowledge base.
        
        Args:
            query: Original query
            existing_context: Existing context to enhance
            
        Returns:
            Enhanced context with relevant knowledge
        """
        enhanced_context = existing_context or {}
        
        # Search for relevant protocol knowledge
        protocol_results = await self.query_similar(
            query,
            collection_name='protocol_knowledge',
            n_results=3,
            similarity_threshold=0.6
        )
        
        # Search for compliance information
        compliance_results = await self.query_similar(
            query,
            collection_name='compliance_rules',
            n_results=2,
            similarity_threshold=0.6
        )
        
        # Search for security patterns
        security_results = await self.query_similar(
            query,
            collection_name='security_patterns',
            n_results=2,
            similarity_threshold=0.6
        )
        
        # Combine results into context
        enhanced_context.update({
            'relevant_protocols': [
                {
                    'content': doc.content,
                    'metadata': doc.metadata,
                    'similarity': score
                }
                for doc, score in zip(protocol_results.documents, protocol_results.similarity_scores)
            ],
            'compliance_context': [
                {
                    'content': doc.content,
                    'metadata': doc.metadata,
                    'similarity': score
                }
                for doc, score in zip(compliance_results.documents, compliance_results.similarity_scores)
            ],
            'security_context': [
                {
                    'content': doc.content,
                    'metadata': doc.metadata,
                    'similarity': score
                }
                for doc, score in zip(security_results.documents, security_results.similarity_scores)
            ],
            'rag_metadata': {
                'query_processed_at': datetime.now().isoformat(),
                'total_results': (
                    protocol_results.total_results + 
                    compliance_results.total_results + 
                    security_results.total_results
                ),
                'avg_similarity': np.mean([
                    *protocol_results.similarity_scores,
                    *compliance_results.similarity_scores,
                    *security_results.similarity_scores
                ]) if any([
                    protocol_results.similarity_scores,
                    compliance_results.similarity_scores,
                    security_results.similarity_scores
                ]) else 0.0
            }
        })
        
        return enhanced_context
    
    async def _load_initial_knowledge(self) -> None:
        """Load initial knowledge base with protocol information."""
        try:
            # Load protocol knowledge
            protocol_docs = [
                RAGDocument(
                    id="http_protocol_basics",
                    content="""HTTP (HyperText Transfer Protocol) is an application-layer protocol for distributed, 
                    collaborative, hypermedia information systems. HTTP is the foundation of data communication for the World Wide Web.
                    
                    Key characteristics:
                    - Request-response protocol
                    - Stateless protocol
                    - Uses TCP as transport layer
                    - Default ports: 80 (HTTP), 443 (HTTPS)
                    - Methods: GET, POST, PUT, DELETE, HEAD, OPTIONS, PATCH
                    
                    Security considerations:
                    - Plain HTTP transmits data in clear text
                    - HTTPS provides encryption via TLS/SSL
                    - Common vulnerabilities: SQL injection, XSS, CSRF
                    """,
                    metadata={
                        'protocol': 'HTTP',
                        'layer': 'application',
                        'security_level': 'low_without_tls',
                        'common_ports': [80, 443]
                    },
                    created_at=datetime.now()
                ),
                RAGDocument(
                    id="tcp_protocol_basics",
                    content="""TCP (Transmission Control Protocol) is a connection-oriented, reliable, 
                    byte stream service. TCP provides reliable, ordered, and error-checked delivery of data between applications.
                    
                    Key characteristics:
                    - Connection-oriented protocol
                    - Reliable data delivery with acknowledgments
                    - Flow control and congestion control
                    - Three-way handshake for connection establishment
                    - Four-way handshake for connection termination
                    
                    Security considerations:
                    - Vulnerable to SYN flood attacks
                    - Connection hijacking possible
                    - No built-in encryption
                    """,
                    metadata={
                        'protocol': 'TCP',
                        'layer': 'transport',
                        'reliability': 'high',
                        'connection_type': 'connection_oriented'
                    },
                    created_at=datetime.now()
                ),
                RAGDocument(
                    id="dns_protocol_basics",
                    content="""DNS (Domain Name System) translates domain names to IP addresses. 
                    It's a hierarchical decentralized naming system for computers, services, or other resources connected to the Internet.
                    
                    Key characteristics:
                    - Hierarchical structure
                    - Uses UDP port 53 (TCP for large responses)
                    - Caching mechanism for performance
                    - Recursive and iterative query types
                    
                    Security considerations:
                    - DNS spoofing/cache poisoning
                    - DNS tunneling for data exfiltration
                    - DNSSEC provides authentication
                    - DNS over HTTPS (DoH) and DNS over TLS (DoT) for privacy
                    """,
                    metadata={
                        'protocol': 'DNS',
                        'layer': 'application',
                        'default_port': 53,
                        'security_extensions': ['DNSSEC', 'DoH', 'DoT']
                    },
                    created_at=datetime.now()
                )
            ]
            
            await self.add_documents('protocol_knowledge', protocol_docs)
            
            # Load compliance rules
            compliance_docs = [
                RAGDocument(
                    id="pci_dss_network_security",
                    content="""PCI DSS Requirement 1: Install and maintain a firewall configuration to protect cardholder data.
                    
                    Key requirements:
                    - Establish firewall standards
                    - Build firewall configuration that restricts connections
                    - Prohibit direct public access between Internet and cardholder data environment
                    - Install personal firewall software on mobile devices
                    
                    Network segmentation requirements:
                    - Implement network segmentation to isolate cardholder data environment
                    - Restrict inbound and outbound traffic to minimum necessary
                    - Document and justify any protocols in use
                    """,
                    metadata={
                        'framework': 'PCI-DSS',
                        'requirement': '1',
                        'domain': 'network_security',
                        'criticality': 'high'
                    },
                    created_at=datetime.now()
                )
            ]
            
            await self.add_documents('compliance_rules', compliance_docs)
            
            # Load security patterns
            security_docs = [
                RAGDocument(
                    id="anomaly_detection_patterns",
                    content="""Common network anomaly patterns for protocol analysis:
                    
                    1. Unusual traffic patterns:
                       - Sudden spikes in traffic volume
                       - Traffic to/from unusual destinations
                       - Unexpected protocol usage
                    
                    2. Protocol violations:
                       - Malformed packets
                       - Protocol field inconsistencies
                       - Unexpected protocol sequences
                    
                    3. Security indicators:
                       - Port scanning patterns
                       - Brute force attempt patterns
                       - Data exfiltration patterns
                    """,
                    metadata={
                        'category': 'anomaly_detection',
                        'type': 'security_patterns',
                        'confidence': 'high'
                    },
                    created_at=datetime.now()
                )
            ]
            
            await self.add_documents('security_patterns', security_docs)
            
            self.logger.info("Initial knowledge base loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load initial knowledge base: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections."""
        stats = {}
        
        for name, collection in self.collections.items():
            try:
                count = collection.count()
                stats[name] = {
                    'document_count': count,
                    'collection_name': name
                }
            except Exception as e:
                stats[name] = {
                    'document_count': 0,
                    'error': str(e)
                }
        
        return stats
    
    async def update_document(
        self, 
        collection_name: str, 
        document_id: str, 
        new_content: str, 
        new_metadata: Dict[str, Any] = None
    ) -> bool:
        """Update an existing document."""
        try:
            if collection_name not in self.collections:
                return False
            
            collection = self.collections[collection_name]
            
            # Generate new embedding
            new_embedding = self.embedding_model.encode([new_content])[0].tolist()
            
            # Update in ChromaDB
            collection.update(
                ids=[document_id],
                documents=[new_content],
                embeddings=[new_embedding],
                metadatas=[new_metadata or {}]
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update document {document_id}: {e}")
            return False
    
    async def delete_document(self, collection_name: str, document_id: str) -> bool:
        """Delete a document from collection."""
        try:
            if collection_name not in self.collections:
                return False
            
            collection = self.collections[collection_name]
            collection.delete(ids=[document_id])
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    def clear_cache(self) -> None:
        """Clear the query cache."""
        self.query_cache.clear()
        self.logger.info("RAG query cache cleared")