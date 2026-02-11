"""
QBITEL - Advanced RAG Features
Multi-query expansion, contextual compression, and enhanced retrieval.

Updated for 2024-2025 AI/ML trends:
- Multi-query expansion using LLM
- Contextual compression to reduce noise
- Query decomposition for complex questions
- Fusion retrieval combining multiple strategies
"""

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel


class QueryExpansionStrategy(Enum):
    """Strategies for query expansion."""
    SYNONYMS = "synonyms"  # Add synonym terms
    HYPOTHETICAL_ANSWER = "hypothetical_answer"  # HyDE - generate hypothetical answer
    MULTI_PERSPECTIVE = "multi_perspective"  # Generate queries from different angles
    STEP_BACK = "step_back"  # Generate more abstract query
    DECOMPOSITION = "decomposition"  # Break into sub-queries


class CompressionStrategy(Enum):
    """Strategies for contextual compression."""
    EXTRACTIVE = "extractive"  # Extract relevant sentences
    ABSTRACTIVE = "abstractive"  # Summarize relevant content
    FILTER_ONLY = "filter_only"  # Remove irrelevant documents
    HYBRID = "hybrid"  # Combination of strategies


@dataclass
class ExpandedQuery:
    """Represents an expanded query with metadata."""
    original_query: str
    expanded_queries: List[str]
    expansion_strategy: QueryExpansionStrategy
    expansion_metadata: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)


@dataclass
class CompressedContext:
    """Represents compressed context from retrieved documents."""
    original_documents: List[Dict[str, Any]]
    compressed_content: str
    compression_ratio: float
    relevant_excerpts: List[str]
    compression_strategy: CompressionStrategy
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiQueryResult:
    """Result from multi-query retrieval."""
    queries_used: List[str]
    documents: List[Dict[str, Any]]
    fusion_scores: List[float]
    processing_time: float
    deduplication_applied: bool = True
    total_candidates: int = 0


class QueryExpander:
    """
    Expands queries using various strategies for improved retrieval.

    Implements multiple expansion strategies:
    - HyDE (Hypothetical Document Embeddings)
    - Multi-perspective query generation
    - Step-back prompting
    - Query decomposition
    """

    def __init__(
        self,
        llm_generate_func: Optional[Callable] = None,
        default_strategy: QueryExpansionStrategy = QueryExpansionStrategy.MULTI_PERSPECTIVE,
        max_expansions: int = 3,
    ):
        self.llm_generate = llm_generate_func
        self.default_strategy = default_strategy
        self.max_expansions = max_expansions
        self.logger = logging.getLogger(__name__)

        # Expansion prompts for different strategies
        self.expansion_prompts = {
            QueryExpansionStrategy.SYNONYMS: """Generate {n} alternative phrasings of this query using synonyms and related terms.
Original query: {query}

Return only the alternative queries, one per line. No numbering or extra text.""",

            QueryExpansionStrategy.HYPOTHETICAL_ANSWER: """Imagine you are an expert answering this question. Write a brief, hypothetical answer that a relevant document might contain.
Question: {query}

Write a 2-3 sentence hypothetical answer that would be found in a relevant document.""",

            QueryExpansionStrategy.MULTI_PERSPECTIVE: """Generate {n} different search queries for this question, each from a different perspective or angle.
Original question: {query}

Consider different aspects like:
- Technical details
- User intent
- Related concepts
- Broader context

Return only the queries, one per line. No numbering or extra text.""",

            QueryExpansionStrategy.STEP_BACK: """Given this specific question, generate a more general/abstract question that captures the underlying concept.
Specific question: {query}

Return only the step-back question, no extra text.""",

            QueryExpansionStrategy.DECOMPOSITION: """Break this complex question into {n} simpler sub-questions that, when answered together, would answer the main question.
Complex question: {query}

Return only the sub-questions, one per line. No numbering or extra text.""",
        }

    async def expand_query(
        self,
        query: str,
        strategy: Optional[QueryExpansionStrategy] = None,
        num_expansions: Optional[int] = None,
    ) -> ExpandedQuery:
        """
        Expand a query using the specified strategy.

        Args:
            query: Original query to expand
            strategy: Expansion strategy to use
            num_expansions: Number of expansions to generate

        Returns:
            ExpandedQuery with original and expanded queries
        """
        strategy = strategy or self.default_strategy
        num_expansions = num_expansions or self.max_expansions

        start_time = time.time()
        expanded_queries = [query]  # Always include original

        try:
            if self.llm_generate is None:
                # Fallback: simple rule-based expansion
                expanded_queries.extend(self._rule_based_expansion(query, num_expansions))
            else:
                # LLM-based expansion
                prompt = self.expansion_prompts[strategy].format(
                    query=query,
                    n=num_expansions
                )

                response = await self.llm_generate(prompt)

                # Parse response into individual queries
                new_queries = self._parse_expansion_response(response, strategy)
                expanded_queries.extend(new_queries[:num_expansions])

            return ExpandedQuery(
                original_query=query,
                expanded_queries=expanded_queries,
                expansion_strategy=strategy,
                expansion_metadata={
                    "processing_time": time.time() - start_time,
                    "num_expansions": len(expanded_queries) - 1,
                    "llm_used": self.llm_generate is not None,
                }
            )

        except Exception as e:
            self.logger.warning(f"Query expansion failed: {e}, using original query only")
            return ExpandedQuery(
                original_query=query,
                expanded_queries=[query],
                expansion_strategy=strategy,
                expansion_metadata={"error": str(e)}
            )

    def _rule_based_expansion(self, query: str, num_expansions: int) -> List[str]:
        """Simple rule-based query expansion as fallback."""
        expansions = []

        # Add "what is" prefix if not present
        lower_query = query.lower()
        if not lower_query.startswith(("what", "how", "why", "when", "where", "who")):
            expansions.append(f"what is {query}")

        # Add "explain" prefix
        if not lower_query.startswith("explain"):
            expansions.append(f"explain {query}")

        # Add common related terms
        protocol_terms = ["protocol", "network", "security", "implementation"]
        for term in protocol_terms:
            if term not in lower_query:
                expansions.append(f"{query} {term}")
                if len(expansions) >= num_expansions:
                    break

        return expansions[:num_expansions]

    def _parse_expansion_response(
        self,
        response: str,
        strategy: QueryExpansionStrategy
    ) -> List[str]:
        """Parse LLM response into individual queries."""
        # Split by newlines and clean up
        lines = [line.strip() for line in response.split("\n")]

        # Remove empty lines, numbering, and bullet points
        queries = []
        for line in lines:
            if not line:
                continue
            # Remove common prefixes
            cleaned = re.sub(r"^[\d\.\-\*\•]+\s*", "", line)
            cleaned = cleaned.strip()
            if cleaned and len(cleaned) > 5:  # Minimum query length
                queries.append(cleaned)

        return queries

    async def decompose_complex_query(
        self,
        query: str,
        max_subqueries: int = 4
    ) -> List[str]:
        """
        Decompose a complex query into simpler sub-queries.
        Useful for multi-hop reasoning.
        """
        expanded = await self.expand_query(
            query,
            strategy=QueryExpansionStrategy.DECOMPOSITION,
            num_expansions=max_subqueries
        )
        return expanded.expanded_queries


class ContextualCompressor:
    """
    Compresses retrieved context to reduce noise and improve relevance.

    Implements:
    - Extractive compression (extract relevant sentences)
    - Abstractive compression (summarize content)
    - Relevance filtering
    """

    def __init__(
        self,
        llm_generate_func: Optional[Callable] = None,
        default_strategy: CompressionStrategy = CompressionStrategy.EXTRACTIVE,
        max_compressed_length: int = 2000,
    ):
        self.llm_generate = llm_generate_func
        self.default_strategy = default_strategy
        self.max_compressed_length = max_compressed_length
        self.logger = logging.getLogger(__name__)

        self.compression_prompts = {
            CompressionStrategy.EXTRACTIVE: """Given the following context and query, extract only the sentences that are directly relevant to answering the query.

Query: {query}

Context:
{context}

Return only the relevant sentences, preserving their original wording. If no sentences are relevant, respond with "NO_RELEVANT_CONTENT".""",

            CompressionStrategy.ABSTRACTIVE: """Given the following context and query, provide a concise summary of the information relevant to the query.

Query: {query}

Context:
{context}

Provide a focused summary (max 3-4 sentences) containing only information relevant to the query. If no relevant information exists, respond with "NO_RELEVANT_CONTENT".""",

            CompressionStrategy.FILTER_ONLY: """Given the following documents and query, identify which documents are relevant to the query.

Query: {query}

Documents:
{documents}

Return only the document IDs that are relevant, one per line. If none are relevant, respond with "NONE".""",
        }

    async def compress(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        strategy: Optional[CompressionStrategy] = None,
    ) -> CompressedContext:
        """
        Compress retrieved documents to focus on relevant content.

        Args:
            query: Original query for relevance assessment
            documents: Retrieved documents to compress
            strategy: Compression strategy to use

        Returns:
            CompressedContext with compressed content
        """
        strategy = strategy or self.default_strategy
        start_time = time.time()

        if not documents:
            return CompressedContext(
                original_documents=[],
                compressed_content="",
                compression_ratio=1.0,
                relevant_excerpts=[],
                compression_strategy=strategy,
            )

        try:
            if self.llm_generate is None:
                # Fallback: simple keyword-based filtering
                return self._keyword_based_compression(query, documents, strategy)

            if strategy == CompressionStrategy.FILTER_ONLY:
                return await self._filter_documents(query, documents)
            elif strategy == CompressionStrategy.HYBRID:
                # First filter, then compress
                filtered = await self._filter_documents(query, documents)
                if filtered.compressed_content:
                    return await self._compress_content(
                        query,
                        filtered.original_documents,
                        CompressionStrategy.EXTRACTIVE
                    )
                return filtered
            else:
                return await self._compress_content(query, documents, strategy)

        except Exception as e:
            self.logger.warning(f"Compression failed: {e}, returning uncompressed")
            combined = "\n\n".join(
                doc.get("content", str(doc)) for doc in documents
            )
            return CompressedContext(
                original_documents=documents,
                compressed_content=combined[:self.max_compressed_length],
                compression_ratio=1.0,
                relevant_excerpts=[],
                compression_strategy=strategy,
                metadata={"error": str(e)}
            )

    async def _compress_content(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        strategy: CompressionStrategy
    ) -> CompressedContext:
        """Compress document content using LLM."""
        # Combine documents into context
        context = "\n\n---\n\n".join(
            doc.get("content", str(doc)) for doc in documents
        )
        original_length = len(context)

        prompt = self.compression_prompts[strategy].format(
            query=query,
            context=context
        )

        response = await self.llm_generate(prompt)

        if "NO_RELEVANT_CONTENT" in response:
            return CompressedContext(
                original_documents=documents,
                compressed_content="",
                compression_ratio=0.0,
                relevant_excerpts=[],
                compression_strategy=strategy,
                metadata={"no_relevant_content": True}
            )

        # Parse excerpts (split by periods or newlines)
        excerpts = [
            s.strip() for s in re.split(r'[.\n]+', response)
            if s.strip() and len(s.strip()) > 20
        ]

        return CompressedContext(
            original_documents=documents,
            compressed_content=response[:self.max_compressed_length],
            compression_ratio=len(response) / max(original_length, 1),
            relevant_excerpts=excerpts,
            compression_strategy=strategy,
        )

    async def _filter_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> CompressedContext:
        """Filter documents by relevance."""
        # Format documents with IDs
        doc_text = "\n\n".join(
            f"[DOC_{i}]: {doc.get('content', str(doc))[:500]}"
            for i, doc in enumerate(documents)
        )

        prompt = self.compression_prompts[CompressionStrategy.FILTER_ONLY].format(
            query=query,
            documents=doc_text
        )

        response = await self.llm_generate(prompt)

        if "NONE" in response:
            return CompressedContext(
                original_documents=documents,
                compressed_content="",
                compression_ratio=0.0,
                relevant_excerpts=[],
                compression_strategy=CompressionStrategy.FILTER_ONLY,
                metadata={"no_relevant_documents": True}
            )

        # Parse document IDs from response
        relevant_ids = set()
        for match in re.finditer(r'DOC_(\d+)', response):
            relevant_ids.add(int(match.group(1)))

        # Filter documents
        filtered_docs = [
            doc for i, doc in enumerate(documents)
            if i in relevant_ids
        ]

        combined = "\n\n".join(
            doc.get("content", str(doc)) for doc in filtered_docs
        )

        return CompressedContext(
            original_documents=filtered_docs,
            compressed_content=combined[:self.max_compressed_length],
            compression_ratio=len(filtered_docs) / max(len(documents), 1),
            relevant_excerpts=[],
            compression_strategy=CompressionStrategy.FILTER_ONLY,
            metadata={"filtered_count": len(filtered_docs)}
        )

    def _keyword_based_compression(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        strategy: CompressionStrategy
    ) -> CompressedContext:
        """Fallback keyword-based compression."""
        query_words = set(query.lower().split())

        scored_docs = []
        for doc in documents:
            content = doc.get("content", str(doc))
            doc_words = set(content.lower().split())
            overlap = len(query_words & doc_words)
            scored_docs.append((doc, overlap, content))

        # Sort by relevance and take top documents
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        relevant_docs = []
        relevant_excerpts = []
        total_length = 0

        for doc, score, content in scored_docs:
            if score > 0 and total_length < self.max_compressed_length:
                relevant_docs.append(doc)
                # Extract sentences containing query words
                sentences = re.split(r'[.!?]+', content)
                for sentence in sentences:
                    if any(word in sentence.lower() for word in query_words):
                        relevant_excerpts.append(sentence.strip())
                total_length += len(content)

        combined = "\n\n".join(
            doc.get("content", str(doc)) for doc in relevant_docs
        )

        original_length = sum(
            len(doc.get("content", str(doc))) for doc in documents
        )

        return CompressedContext(
            original_documents=relevant_docs,
            compressed_content=combined[:self.max_compressed_length],
            compression_ratio=len(combined) / max(original_length, 1),
            relevant_excerpts=relevant_excerpts[:10],
            compression_strategy=strategy,
            metadata={"method": "keyword_based"}
        )


class FusionRetriever:
    """
    Implements Reciprocal Rank Fusion (RRF) for combining results from multiple queries.

    RRF is a robust method for combining rankings that:
    - Doesn't require score normalization
    - Handles different score distributions
    - Produces stable, high-quality rankings
    """

    def __init__(
        self,
        k: int = 60,  # RRF constant
        deduplication: bool = True,
    ):
        self.k = k
        self.deduplication = deduplication
        self.logger = logging.getLogger(__name__)

    def fuse_results(
        self,
        query_results: List[List[Dict[str, Any]]],
        id_field: str = "id",
    ) -> MultiQueryResult:
        """
        Fuse results from multiple queries using Reciprocal Rank Fusion.

        Args:
            query_results: List of result lists, one per query
            id_field: Field to use for document identification

        Returns:
            MultiQueryResult with fused rankings
        """
        start_time = time.time()

        # Calculate RRF scores
        doc_scores: Dict[str, Tuple[float, Dict[str, Any]]] = {}

        for query_idx, results in enumerate(query_results):
            for rank, doc in enumerate(results):
                doc_id = self._get_doc_id(doc, id_field)
                rrf_score = 1.0 / (self.k + rank + 1)

                if doc_id in doc_scores:
                    current_score, current_doc = doc_scores[doc_id]
                    doc_scores[doc_id] = (current_score + rrf_score, current_doc)
                else:
                    doc_scores[doc_id] = (rrf_score, doc)

        # Sort by fused score
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1][0],
            reverse=True
        )

        # Extract results
        documents = [doc for _, (score, doc) in sorted_docs]
        scores = [score for _, (score, doc) in sorted_docs]

        return MultiQueryResult(
            queries_used=[],  # Set by caller
            documents=documents,
            fusion_scores=scores,
            processing_time=time.time() - start_time,
            deduplication_applied=self.deduplication,
            total_candidates=sum(len(r) for r in query_results)
        )

    def _get_doc_id(self, doc: Dict[str, Any], id_field: str) -> str:
        """Get document ID, generating one if needed."""
        if id_field in doc:
            return str(doc[id_field])

        # Generate ID from content hash
        content = doc.get("content", str(doc))
        return hashlib.md5(content.encode()).hexdigest()[:16]


class AdvancedRAGPipeline:
    """
    Complete advanced RAG pipeline combining all features.

    Pipeline stages:
    1. Query expansion (multi-query, HyDE, etc.)
    2. Multi-query retrieval
    3. Result fusion (RRF)
    4. Contextual compression
    5. Final context assembly
    """

    def __init__(
        self,
        retrieval_func: Callable,
        llm_generate_func: Optional[Callable] = None,
        enable_expansion: bool = True,
        enable_compression: bool = True,
        expansion_strategy: QueryExpansionStrategy = QueryExpansionStrategy.MULTI_PERSPECTIVE,
        compression_strategy: CompressionStrategy = CompressionStrategy.EXTRACTIVE,
        max_results: int = 10,
        fusion_k: int = 60,
    ):
        self.retrieve = retrieval_func
        self.llm_generate = llm_generate_func
        self.enable_expansion = enable_expansion
        self.enable_compression = enable_compression
        self.max_results = max_results
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.query_expander = QueryExpander(
            llm_generate_func=llm_generate_func,
            default_strategy=expansion_strategy,
        )

        self.contextual_compressor = ContextualCompressor(
            llm_generate_func=llm_generate_func,
            default_strategy=compression_strategy,
        )

        self.fusion_retriever = FusionRetriever(k=fusion_k)

    async def retrieve_with_expansion(
        self,
        query: str,
        collection_name: Optional[str] = None,
        expansion_strategy: Optional[QueryExpansionStrategy] = None,
        num_expansions: int = 3,
        **retrieval_kwargs
    ) -> MultiQueryResult:
        """
        Retrieve documents using query expansion and fusion.

        Args:
            query: Original query
            collection_name: Optional collection to search
            expansion_strategy: Strategy for query expansion
            num_expansions: Number of expanded queries to generate
            **retrieval_kwargs: Additional arguments for retrieval

        Returns:
            MultiQueryResult with fused rankings
        """
        start_time = time.time()

        # Step 1: Expand query
        if self.enable_expansion:
            expanded = await self.query_expander.expand_query(
                query,
                strategy=expansion_strategy,
                num_expansions=num_expansions
            )
            queries = expanded.expanded_queries
        else:
            queries = [query]

        # Step 2: Retrieve for each query
        all_results = []
        for q in queries:
            try:
                results = await self.retrieve(
                    query=q,
                    collection_name=collection_name,
                    **retrieval_kwargs
                )
                # Handle both list and RAGQueryResult
                if hasattr(results, "documents"):
                    docs = [
                        {
                            "id": doc.id,
                            "content": doc.content,
                            "metadata": doc.metadata,
                            "score": score
                        }
                        for doc, score in zip(results.documents, results.similarity_scores)
                    ]
                else:
                    docs = results if isinstance(results, list) else []
                all_results.append(docs)
            except Exception as e:
                self.logger.warning(f"Retrieval failed for query '{q[:50]}...': {e}")
                all_results.append([])

        # Step 3: Fuse results
        fused = self.fusion_retriever.fuse_results(all_results)
        fused.queries_used = queries
        fused.processing_time = time.time() - start_time

        # Limit results
        fused.documents = fused.documents[:self.max_results]
        fused.fusion_scores = fused.fusion_scores[:self.max_results]

        return fused

    async def retrieve_and_compress(
        self,
        query: str,
        collection_name: Optional[str] = None,
        expansion_strategy: Optional[QueryExpansionStrategy] = None,
        compression_strategy: Optional[CompressionStrategy] = None,
        **retrieval_kwargs
    ) -> Tuple[MultiQueryResult, CompressedContext]:
        """
        Full pipeline: expand, retrieve, fuse, and compress.

        Returns:
            Tuple of (MultiQueryResult, CompressedContext)
        """
        # Retrieve with expansion
        fused_results = await self.retrieve_with_expansion(
            query,
            collection_name=collection_name,
            expansion_strategy=expansion_strategy,
            **retrieval_kwargs
        )

        # Compress results
        if self.enable_compression and fused_results.documents:
            compressed = await self.contextual_compressor.compress(
                query,
                fused_results.documents,
                strategy=compression_strategy
            )
        else:
            combined = "\n\n".join(
                doc.get("content", str(doc)) for doc in fused_results.documents
            )
            compressed = CompressedContext(
                original_documents=fused_results.documents,
                compressed_content=combined,
                compression_ratio=1.0,
                relevant_excerpts=[],
                compression_strategy=CompressionStrategy.FILTER_ONLY,
            )

        return fused_results, compressed

    async def get_context_for_llm(
        self,
        query: str,
        collection_name: Optional[str] = None,
        max_context_length: int = 4000,
        **kwargs
    ) -> str:
        """
        Get optimized context string for LLM consumption.

        This is a convenience method that runs the full pipeline
        and returns a formatted context string.
        """
        _, compressed = await self.retrieve_and_compress(
            query,
            collection_name=collection_name,
            **kwargs
        )

        context = compressed.compressed_content
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."

        return context


# Factory function for creating pipeline
def create_advanced_rag_pipeline(
    rag_engine,
    llm_service=None,
    **config
) -> AdvancedRAGPipeline:
    """
    Factory function to create an AdvancedRAGPipeline.

    Args:
        rag_engine: RAGEngine instance for retrieval
        llm_service: Optional UnifiedLLMService for LLM operations
        **config: Additional configuration options

    Returns:
        Configured AdvancedRAGPipeline instance
    """
    async def retrieval_func(query, collection_name=None, **kwargs):
        return await rag_engine.query_similar(
            query,
            collection_name=collection_name,
            **kwargs
        )

    llm_func = None
    if llm_service is not None:
        async def llm_func(prompt):
            from .unified_llm_service import LLMRequest
            request = LLMRequest(
                prompt=prompt,
                feature_domain="protocol_copilot",
                max_tokens=500,
                temperature=0.3,
            )
            response = await llm_service.process_request(request)
            return response.content

    return AdvancedRAGPipeline(
        retrieval_func=retrieval_func,
        llm_generate_func=llm_func,
        **config
    )
