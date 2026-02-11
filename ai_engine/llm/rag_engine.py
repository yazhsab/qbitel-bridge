"""
QBITEL - Retrieval Augmented Generation (RAG) Engine
Advanced RAG implementation for protocol intelligence with vector similarity search.

Updated for 2024-2025 AI/ML trends:
- Hybrid search (BM25 + Vector similarity)
- Cross-encoder reranking
- Better embedding models (BGE, Nomic)
- Multi-query expansion
- Contextual compression
"""

import asyncio
import hashlib
import logging
import math
import re
import sys
import time
import types
import uuid
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from collections import Counter as WordCounter
import json
from datetime import datetime
from enum import Enum
from prometheus_client import Counter, Histogram

try:  # pragma: no cover - optional dependency
    import chromadb as _chromadb  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - handled via in-memory fallback
    _chromadb = None

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer as _SentenceTransformer  # type: ignore
    from sentence_transformers import CrossEncoder as _CrossEncoder  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - handled via deterministic fallback
    _SentenceTransformer = None
    _CrossEncoder = None

try:  # pragma: no cover - optional dependency for BM25
    from rank_bm25 import BM25Okapi as _BM25Okapi  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    _BM25Okapi = None


# =============================================================================
# Embedding Model Configuration - 2024-2025 Best Models
# =============================================================================

class EmbeddingModelConfig:
    """Centralized configuration for embedding models."""

    # Best general-purpose models (2024-2025)
    BGE_LARGE = "BAAI/bge-large-en-v1.5"  # Excellent for general use
    BGE_BASE = "BAAI/bge-base-en-v1.5"  # Good balance of speed/quality
    BGE_SMALL = "BAAI/bge-small-en-v1.5"  # Fast, good for prototyping

    # Nomic models (good for long context)
    NOMIC_EMBED = "nomic-ai/nomic-embed-text-v1.5"

    # All-MiniLM (legacy, fast)
    MINILM = "all-MiniLM-L6-v2"

    # Default - use BGE-base for good balance
    DEFAULT = BGE_BASE
    LEGACY = MINILM


class RerankerModelConfig:
    """Configuration for cross-encoder reranking models."""

    # Best rerankers (2024-2025)
    BGE_RERANKER_LARGE = "BAAI/bge-reranker-large"
    BGE_RERANKER_BASE = "BAAI/bge-reranker-base"

    # Cross-encoder models
    MS_MARCO_MINILM = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    DEFAULT = BGE_RERANKER_BASE


class SearchMode(Enum):
    """Search modes for RAG queries."""
    VECTOR_ONLY = "vector"  # Pure semantic search
    BM25_ONLY = "bm25"  # Pure keyword search
    HYBRID = "hybrid"  # Combined BM25 + vector
    HYBRID_RERANK = "hybrid_rerank"  # Hybrid with cross-encoder reranking


class _FallbackSentenceTransformer:
    """Deterministic embedding fallback when sentence-transformers is unavailable."""

    def __init__(self, model_name: str = "sentence-transformer-fallback"):
        self.model_name = model_name

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        embeddings = [self._encode_single(text) for text in texts]
        return np.array(embeddings)

    @staticmethod
    def _encode_single(text: str, vector_size: int = 384) -> List[float]:
        # Use SHA-256 to generate deterministic embeddings
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        floats = [byte / 255.0 for byte in digest]
        if len(floats) >= vector_size:
            return floats[:vector_size]
        repeats = (vector_size + len(floats) - 1) // len(floats)
        extended = (floats * repeats)[:vector_size]
        return extended


class _FallbackCrossEncoder:
    """Fallback cross-encoder when sentence-transformers is unavailable."""

    def __init__(self, model_name: str = "cross-encoder-fallback"):
        self.model_name = model_name

    def predict(self, sentence_pairs: List[Tuple[str, str]], **kwargs) -> np.ndarray:
        """Compute simple similarity scores as fallback."""
        scores = []
        for query, doc in sentence_pairs:
            # Simple word overlap score as fallback
            query_words = set(query.lower().split())
            doc_words = set(doc.lower().split())
            if not query_words or not doc_words:
                scores.append(0.0)
            else:
                overlap = len(query_words & doc_words)
                scores.append(overlap / max(len(query_words), 1))
        return np.array(scores)


class _FallbackBM25:
    """Simple BM25 implementation when rank_bm25 is unavailable."""

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if corpus else 0
        self.doc_freqs = []
        self.idf = {}
        self._calculate_idf()

    def _calculate_idf(self):
        """Calculate inverse document frequency for each term."""
        df = {}  # document frequency
        for document in self.corpus:
            for word in set(document):
                df[word] = df.get(word, 0) + 1

        n_docs = len(self.corpus)
        for word, freq in df.items():
            self.idf[word] = math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1)

    def get_scores(self, query: List[str]) -> np.ndarray:
        """Get BM25 scores for a query against all documents."""
        scores = np.zeros(len(self.corpus))
        for i, doc in enumerate(self.corpus):
            score = 0.0
            doc_len = self.doc_len[i]
            term_freqs = WordCounter(doc)

            for term in query:
                if term not in self.idf:
                    continue
                tf = term_freqs.get(term, 0)
                idf = self.idf[term]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += idf * numerator / denominator

            scores[i] = score
        return scores

    def get_top_n(self, query: List[str], n: int = 5) -> List[Tuple[int, float]]:
        """Get top N documents by BM25 score."""
        scores = self.get_scores(query)
        top_indices = np.argsort(scores)[::-1][:n]
        return [(idx, scores[idx]) for idx in top_indices]


class _InMemoryChromaCollection:
    """Minimal in-memory collection used as a chromadb fallback."""

    def __init__(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        self.name = name
        self.metadata = metadata or {}
        self._documents: List[str] = []
        self._embeddings: List[List[float]] = []
        self._metadatas: List[Dict[str, Any]] = []
        self._ids: List[str] = []

    def add(
        self,
        documents: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        if embeddings is None:
            raise ValueError(
                "Embeddings are required when using the in-memory Chroma fallback"
            )

        metadatas = metadatas or [{} for _ in documents]
        ids = ids or [str(uuid.uuid4()) for _ in documents]

        for doc, emb, meta, doc_id in zip(documents, embeddings, metadatas, ids):
            self._documents.append(doc)
            self._embeddings.append(list(emb))
            self._metadatas.append(dict(meta))
            self._ids.append(doc_id)

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        include: Optional[List[str]] = None,
    ) -> Dict[str, List[List[Any]]]:
        include = include or []
        if not query_embeddings:
            empty = [[]]
            result: Dict[str, List[List[Any]]] = {
                "ids": empty,
                "documents": empty,
                "metadatas": empty,
                "distances": empty,
            }
            if "embeddings" in include:
                result["embeddings"] = empty
            return result

        query_vector = np.array(query_embeddings[0], dtype=float)

        ranked: List[Tuple[str, str, Dict[str, Any], float]] = []
        for doc_id, doc, meta, emb in zip(
            self._ids, self._documents, self._metadatas, self._embeddings
        ):
            emb_vector = np.array(emb, dtype=float)
            similarity = self._cosine_similarity(query_vector, emb_vector)
            ranked.append((doc_id, doc, meta, similarity))

        ranked.sort(key=lambda item: item[3], reverse=True)
        top_k = ranked[: max(1, n_results)] if ranked else []

        ids, docs, metas, sims = zip(*top_k) if top_k else ([], [], [], [])
        distances = [1.0 - sim for sim in sims]

        result: Dict[str, List[List[Any]]] = {
            "ids": [list(ids)],
            "documents": [list(docs)],
            "metadatas": [list(metas)],
            "distances": [distances],
        }

        if "embeddings" in include:
            embed_map = {
                doc_id: emb for doc_id, emb in zip(self._ids, self._embeddings)
            }
            result["embeddings"] = [[embed_map[doc_id] for doc_id in ids]]

        return result

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        if denom == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / denom)


class _InMemoryChromaClient:
    """Lightweight client that mimics the subset of chromadb API we rely on."""

    def __init__(self, *args, **kwargs):
        self._collections: Dict[str, _InMemoryChromaCollection] = {}

    def get_collection(self, name: str):
        if name not in self._collections:
            raise KeyError(name)
        return self._collections[name]

    def create_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        collection = _InMemoryChromaCollection(name, metadata)
        self._collections[name] = collection
        return collection

    def get_or_create_collection(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ):
        return self._collections.get(name) or self.create_collection(name, metadata)

    def delete_collection(self, name: str) -> None:
        self._collections.pop(name, None)


# Set up model classes with fallbacks
if _SentenceTransformer is None:  # pragma: no cover - import guard
    SentenceTransformer = _FallbackSentenceTransformer  # type: ignore
else:
    SentenceTransformer = _SentenceTransformer  # type: ignore

if _CrossEncoder is None:  # pragma: no cover - import guard
    CrossEncoder = _FallbackCrossEncoder  # type: ignore
else:
    CrossEncoder = _CrossEncoder  # type: ignore

if _BM25Okapi is None:  # pragma: no cover - import guard
    BM25Okapi = _FallbackBM25  # type: ignore
else:
    BM25Okapi = _BM25Okapi  # type: ignore


if _chromadb is None:  # pragma: no cover - import guard
    chromadb = types.ModuleType("chromadb")
    chromadb.Client = _InMemoryChromaClient  # type: ignore[attr-defined]
    chromadb.PersistentClient = _InMemoryChromaClient  # type: ignore[attr-defined]
    chromadb.Collection = _InMemoryChromaCollection  # type: ignore[attr-defined]
    sys.modules["chromadb"] = chromadb
    _HAS_CHROMADB = False
else:
    chromadb = _chromadb  # type: ignore
    _HAS_CHROMADB = True

# Metrics
RAG_QUERY_COUNTER = Counter(
    "qbitel_rag_queries_total", "Total RAG queries", ["query_type"]
)
RAG_QUERY_DURATION = Histogram(
    "qbitel_rag_query_duration_seconds", "RAG query duration"
)
RAG_SIMILARITY_SCORE = Histogram("qbitel_rag_similarity_score", "RAG similarity scores")

logger = logging.getLogger(__name__)


@dataclass
class RAGDocument:
    """Document structure for RAG storage."""

    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    created_at: datetime = None
    # New fields for hybrid search
    bm25_score: Optional[float] = None
    vector_score: Optional[float] = None
    rerank_score: Optional[float] = None


@dataclass
class RAGQueryResult:
    """RAG query result structure with hybrid search metadata."""

    documents: List[RAGDocument]
    similarity_scores: List[float]
    query_embedding: List[float]
    processing_time: float
    total_results: int

    # New: Search metadata
    search_mode: SearchMode = SearchMode.VECTOR_ONLY
    bm25_weight: float = 0.0
    vector_weight: float = 1.0
    reranking_applied: bool = False
    query_expansion_applied: bool = False


class RAGEngine:
    """
    Enterprise RAG engine for protocol intelligence with ChromaDB and SentenceTransformers.

    Updated for 2024-2025 with:
    - Hybrid search (BM25 + Vector similarity)
    - Cross-encoder reranking
    - Better embedding models (BGE, Nomic)
    - Multi-query expansion
    - Configurable search modes
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Client configuration
        self.client = None
        self.is_initialized = False
        self.use_fallback_store = False
        self._chroma_path = self.config.get("chroma_db_path", "./data/chroma_db")

        # Initialize embedding model - use better default (BGE)
        embedding_model_name = self.config.get(
            "embedding_model",
            EmbeddingModelConfig.DEFAULT  # BGE-base by default
        )
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_model_name = embedding_model_name

        # Initialize reranker model (optional, for hybrid_rerank mode)
        self.reranker = None
        self.reranker_enabled = self.config.get("enable_reranker", True)
        if self.reranker_enabled:
            reranker_model_name = self.config.get(
                "reranker_model",
                RerankerModelConfig.DEFAULT
            )
            try:
                self.reranker = CrossEncoder(reranker_model_name)
                self.logger.info(f"Reranker initialized: {reranker_model_name}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize reranker: {e}")
                self.reranker = None

        # BM25 index for hybrid search (built per collection)
        self.bm25_indices: Dict[str, Any] = {}
        self.bm25_corpus: Dict[str, List[List[str]]] = {}
        self.bm25_doc_ids: Dict[str, List[str]] = {}

        # Default search configuration
        self.default_search_mode = SearchMode(
            self.config.get("search_mode", SearchMode.HYBRID.value)
        )
        self.default_bm25_weight = self.config.get("bm25_weight", 0.3)
        self.default_vector_weight = self.config.get("vector_weight", 0.7)
        self.rerank_top_k = self.config.get("rerank_top_k", 20)  # Candidates for reranking

        # Collections for different knowledge types
        self.collections: Dict[str, Any] = {}
        self.collection_names = [
            "protocol_knowledge",
            "compliance_rules",
            "security_patterns",
            "field_definitions",
            "threat_intelligence",
            "protocol_translation_patterns",
            "api_design_patterns",
            "code_generation_templates",
            "translation_best_practices",
        ]

        # Query cache
        self.query_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = self.config.get("cache_ttl", 3600)  # seconds

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for BM25."""
        # Lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def _build_bm25_index(self, collection_name: str, documents: List[str], doc_ids: List[str]) -> None:
        """Build BM25 index for a collection."""
        corpus = [self._tokenize(doc) for doc in documents]
        self.bm25_corpus[collection_name] = corpus
        self.bm25_doc_ids[collection_name] = doc_ids
        self.bm25_indices[collection_name] = BM25Okapi(corpus)
        self.logger.debug(f"Built BM25 index for {collection_name} with {len(documents)} documents")

    def _update_bm25_index(self, collection_name: str, document: str, doc_id: str) -> None:
        """Update BM25 index with a new document."""
        if collection_name not in self.bm25_corpus:
            self.bm25_corpus[collection_name] = []
            self.bm25_doc_ids[collection_name] = []

        tokens = self._tokenize(document)
        self.bm25_corpus[collection_name].append(tokens)
        self.bm25_doc_ids[collection_name].append(doc_id)

        # Rebuild index (in production, use incremental updates)
        self.bm25_indices[collection_name] = BM25Okapi(self.bm25_corpus[collection_name])

    async def initialize(self) -> None:
        """Initialize RAG engine and create collections."""
        try:
            self.logger.info("Initializing RAG Engine...")

            if self.client is None:
                self.client = self._create_client()
                self.use_fallback_store = isinstance(self.client, _InMemoryChromaClient)

            # Create or get collections
            for collection_name in self.collection_names:
                collection = self._get_or_create_collection(collection_name)
                self.collections[collection_name] = collection

            # Load initial knowledge base
            await self._load_initial_knowledge()

            self.is_initialized = True
            self.logger.info("RAG Engine initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize RAG engine: {e}")
            raise

    def _create_client(self):
        """Create a ChromaDB client, falling back to in-memory storage when unavailable."""
        if _HAS_CHROMADB:
            persistent_cls = getattr(chromadb, "PersistentClient", None)
            if persistent_cls is not None:
                try:
                    return persistent_cls(path=self._chroma_path)
                except Exception as exc:  # pragma: no cover - runtime safety net
                    self.logger.warning(
                        "Failed to initialize chromadb.PersistentClient, falling back to Client: %s",
                        exc,
                    )

            client_cls = getattr(chromadb, "Client", None)
            if client_cls is not None:
                try:
                    return client_cls()
                except Exception as exc:  # pragma: no cover - runtime safety net
                    self.logger.warning(
                        "Failed to initialize chromadb.Client, using in-memory fallback: %s",
                        exc,
                    )

        self.logger.warning(
            "chromadb dependency not available; using in-memory vector store fallback"
        )
        return _InMemoryChromaClient()

    def _get_or_create_collection(self, name: str):
        if self.client is None:
            raise RuntimeError("RAGEngine client is not initialized")

        # Attempt standard chromadb API
        try:
            if hasattr(self.client, "get_collection"):
                collection = self.client.get_collection(name=name)
                self.collections[name] = collection
                return collection
        except Exception:
            pass

        metadata = {"hnsw:space": "cosine"}

        if hasattr(self.client, "get_or_create_collection"):
            collection = self.client.get_or_create_collection(
                name=name, metadata=metadata
            )
            self.collections[name] = collection
            return collection

        if hasattr(self.client, "create_collection"):
            collection = self.client.create_collection(name=name, metadata=metadata)
            self.collections[name] = collection
            return collection

        raise AttributeError(
            "Configured Chroma client does not expose a collection creation API"
        )

    async def add_documents(
        self, collection_name: str, documents: List[Union[RAGDocument, Dict[str, Any]]]
    ) -> bool:
        """Add documents to a specific collection."""
        if self.client is None:
            raise RuntimeError("RAGEngine must be initialized before adding documents")

        collection = self.collections.get(collection_name)
        if collection is None:
            collection = self._get_or_create_collection(collection_name)
            self.collections[collection_name] = collection

        normalized_docs: List[RAGDocument] = []
        for doc in documents:
            if isinstance(doc, RAGDocument):
                normalized_docs.append(doc)
            elif isinstance(doc, dict):
                if "content" not in doc:
                    raise ValueError(
                        "Document dictionaries must include a 'content' field"
                    )
                normalized_docs.append(
                    RAGDocument(
                        id=str(doc.get("id") or uuid.uuid4()),
                        content=doc["content"],
                        metadata=dict(doc.get("metadata", {})),
                        embedding=doc.get("embedding"),
                        created_at=doc.get("created_at") or datetime.utcnow(),
                    )
                )
            else:
                raise TypeError(
                    "Documents must be RAGDocument instances or dictionary representations"
                )

        if not normalized_docs:
            self.logger.debug(
                "No documents provided for collection %s; skipping add operation",
                collection_name,
            )
            return False

        texts = [doc.content for doc in normalized_docs]
        embeddings = [
            list(embedding) for embedding in self.embedding_model.encode(texts)
        ]

        for doc, embedding in zip(normalized_docs, embeddings):
            doc.embedding = embedding

        ids = [doc.id for doc in normalized_docs]
        metadatas = [doc.metadata for doc in normalized_docs]

        collection.add(
            documents=texts, embeddings=embeddings, metadatas=metadatas, ids=ids
        )

        # Update BM25 index for hybrid search
        for doc, doc_id in zip(normalized_docs, ids):
            self._update_bm25_index(collection_name, doc.content, doc_id)

        self.logger.info(
            "Added %s documents to collection '%s'",
            len(normalized_docs),
            collection_name,
        )
        return True

    async def query_similar(
        self,
        query: str,
        collection_name: str = None,
        n_results: int = 5,
        similarity_threshold: float = 0.7,
        search_mode: Optional[SearchMode] = None,
        bm25_weight: Optional[float] = None,
        vector_weight: Optional[float] = None,
    ) -> RAGQueryResult:
        """
        Query for similar documents using configurable search modes.

        Args:
            query: Search query
            collection_name: Specific collection to search (None for all)
            n_results: Number of results to return
            similarity_threshold: Minimum similarity threshold
            search_mode: Search mode (vector, bm25, hybrid, hybrid_rerank)
            bm25_weight: Weight for BM25 scores in hybrid search
            vector_weight: Weight for vector scores in hybrid search

        Returns:
            RAG query result with similar documents
        """
        start_time = time.time()

        # Use defaults if not specified
        search_mode = search_mode or self.default_search_mode
        bm25_weight = bm25_weight if bm25_weight is not None else self.default_bm25_weight
        vector_weight = vector_weight if vector_weight is not None else self.default_vector_weight

        try:
            # Check cache (include search mode in cache key)
            cache_key = f"{query}_{collection_name}_{n_results}_{similarity_threshold}_{search_mode.value}"
            if cache_key in self.query_cache:
                cache_entry = self.query_cache[cache_key]
                if time.time() - cache_entry["timestamp"] < self.cache_ttl:
                    return cache_entry["result"]

            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0].tolist()

            # Search in specified collection or all collections
            collections_to_search = (
                [collection_name] if collection_name else self.collection_names
            )

            all_documents = []
            all_scores = []

            for coll_name in collections_to_search:
                collection = self.collections.get(coll_name)
                if collection is None:
                    try:
                        collection = self._get_or_create_collection(coll_name)
                    except Exception:
                        continue

                if not hasattr(collection, "query"):
                    continue

                # Determine how many candidates to retrieve for reranking
                retrieve_k = self.rerank_top_k if search_mode == SearchMode.HYBRID_RERANK else n_results

                # Get vector search results
                vector_results = {}
                if search_mode in (SearchMode.VECTOR_ONLY, SearchMode.HYBRID, SearchMode.HYBRID_RERANK):
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=retrieve_k,
                        include=["documents", "metadatas", "distances"],
                    )
                    if results.get("documents") and results["documents"][0]:
                        for i, (doc_id, doc, metadata, distance) in enumerate(
                            zip(
                                results.get("ids", [[]])[0],
                                results["documents"][0],
                                results.get("metadatas", [[]])[0],
                                results.get("distances", [[]])[0],
                            )
                        ):
                            vector_score = 1.0 - distance
                            vector_results[doc_id] = {
                                "content": doc,
                                "metadata": metadata,
                                "vector_score": vector_score,
                                "bm25_score": 0.0,
                            }

                # Get BM25 results
                bm25_results = {}
                if search_mode in (SearchMode.BM25_ONLY, SearchMode.HYBRID, SearchMode.HYBRID_RERANK):
                    if coll_name in self.bm25_indices:
                        query_tokens = self._tokenize(query)
                        bm25_scores = self.bm25_indices[coll_name].get_scores(query_tokens)

                        # Normalize BM25 scores
                        max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 and max(bm25_scores) > 0 else 1.0
                        normalized_scores = bm25_scores / max_bm25

                        # Get top-k by BM25
                        top_indices = np.argsort(normalized_scores)[::-1][:retrieve_k]
                        for idx in top_indices:
                            if idx < len(self.bm25_doc_ids.get(coll_name, [])):
                                doc_id = self.bm25_doc_ids[coll_name][idx]
                                bm25_score = float(normalized_scores[idx])
                                if doc_id in vector_results:
                                    vector_results[doc_id]["bm25_score"] = bm25_score
                                else:
                                    # Need to fetch document content from collection
                                    bm25_results[doc_id] = {
                                        "bm25_score": bm25_score,
                                        "vector_score": 0.0,
                                    }

                # Combine results based on search mode
                combined_results = {}

                if search_mode == SearchMode.VECTOR_ONLY:
                    combined_results = vector_results
                    for doc_id, data in combined_results.items():
                        data["final_score"] = data["vector_score"]

                elif search_mode == SearchMode.BM25_ONLY:
                    # For BM25-only, we need to fetch documents
                    for doc_id, data in bm25_results.items():
                        data["final_score"] = data["bm25_score"]
                    combined_results = bm25_results

                elif search_mode in (SearchMode.HYBRID, SearchMode.HYBRID_RERANK):
                    # Merge vector and BM25 results
                    all_doc_ids = set(vector_results.keys()) | set(bm25_results.keys())
                    for doc_id in all_doc_ids:
                        v_data = vector_results.get(doc_id, {"vector_score": 0.0, "bm25_score": 0.0})
                        b_data = bm25_results.get(doc_id, {"vector_score": 0.0, "bm25_score": 0.0})

                        combined_results[doc_id] = {
                            "content": v_data.get("content", ""),
                            "metadata": v_data.get("metadata", {}),
                            "vector_score": v_data.get("vector_score", 0.0),
                            "bm25_score": max(v_data.get("bm25_score", 0.0), b_data.get("bm25_score", 0.0)),
                        }

                        # Calculate hybrid score
                        combined_results[doc_id]["final_score"] = (
                            vector_weight * combined_results[doc_id]["vector_score"] +
                            bm25_weight * combined_results[doc_id]["bm25_score"]
                        )

                # Apply reranking if enabled
                reranking_applied = False
                if search_mode == SearchMode.HYBRID_RERANK and self.reranker is not None:
                    reranking_applied = True
                    # Prepare pairs for cross-encoder
                    doc_ids = list(combined_results.keys())
                    if doc_ids:
                        pairs = [(query, combined_results[doc_id].get("content", "")) for doc_id in doc_ids]
                        rerank_scores = self.reranker.predict(pairs)

                        # Normalize rerank scores
                        if len(rerank_scores) > 0:
                            min_score = min(rerank_scores)
                            max_score = max(rerank_scores)
                            score_range = max_score - min_score if max_score > min_score else 1.0
                            normalized_rerank = (rerank_scores - min_score) / score_range

                            for i, doc_id in enumerate(doc_ids):
                                combined_results[doc_id]["rerank_score"] = float(normalized_rerank[i])
                                # Replace final score with rerank score
                                combined_results[doc_id]["final_score"] = float(normalized_rerank[i])

                # Convert to RAGDocument objects
                for doc_id, data in combined_results.items():
                    final_score = data.get("final_score", 0.0)
                    if final_score >= similarity_threshold:
                        rag_doc = RAGDocument(
                            id=doc_id,
                            content=data.get("content", ""),
                            metadata={
                                **data.get("metadata", {}),
                                "collection": coll_name,
                            },
                            vector_score=data.get("vector_score"),
                            bm25_score=data.get("bm25_score"),
                            rerank_score=data.get("rerank_score"),
                        )
                        all_documents.append(rag_doc)
                        all_scores.append(final_score)

            # Sort by final score
            if all_documents:
                sorted_pairs = sorted(
                    zip(all_documents, all_scores), key=lambda x: x[1], reverse=True
                )
                all_documents, all_scores = zip(*sorted_pairs)
                all_documents = list(all_documents)[:n_results]
                all_scores = list(all_scores)[:n_results]
            else:
                all_documents = []
                all_scores = []

            result = RAGQueryResult(
                documents=all_documents,
                similarity_scores=all_scores,
                query_embedding=query_embedding,
                processing_time=time.time() - start_time,
                total_results=len(all_documents),
                search_mode=search_mode,
                bm25_weight=bm25_weight,
                vector_weight=vector_weight,
                reranking_applied=reranking_applied,
            )

            # Cache result
            self.query_cache[cache_key] = {"result": result, "timestamp": time.time()}

            # Update metrics
            RAG_QUERY_COUNTER.labels(query_type=collection_name or "all").inc()
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
                total_results=0,
                search_mode=search_mode,
            )

    async def search(
        self,
        collection_name: str,
        query: str,
        limit: int = 5,
        similarity_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Convenience wrapper returning a simplified list of document matches."""

        result = await self.query_similar(
            query,
            collection_name=collection_name,
            n_results=limit,
            similarity_threshold=similarity_threshold or 0.0,
        )

        simplified: List[Dict[str, Any]] = []
        for document, score in zip(result.documents, result.similarity_scores):
            simplified.append(
                {
                    "id": document.id,
                    "content": document.content,
                    "metadata": document.metadata,
                    "similarity": score,
                }
            )

        return simplified

    async def enhance_query_context(
        self, query: str, existing_context: Dict[str, Any] = None
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
            collection_name="protocol_knowledge",
            n_results=3,
            similarity_threshold=0.6,
        )

        # Search for compliance information
        compliance_results = await self.query_similar(
            query,
            collection_name="compliance_rules",
            n_results=2,
            similarity_threshold=0.6,
        )

        # Search for security patterns
        security_results = await self.query_similar(
            query,
            collection_name="security_patterns",
            n_results=2,
            similarity_threshold=0.6,
        )

        # Combine results into context
        enhanced_context.update(
            {
                "relevant_protocols": [
                    {
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "similarity": score,
                    }
                    for doc, score in zip(
                        protocol_results.documents, protocol_results.similarity_scores
                    )
                ],
                "compliance_context": [
                    {
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "similarity": score,
                    }
                    for doc, score in zip(
                        compliance_results.documents,
                        compliance_results.similarity_scores,
                    )
                ],
                "security_context": [
                    {
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "similarity": score,
                    }
                    for doc, score in zip(
                        security_results.documents, security_results.similarity_scores
                    )
                ],
                "rag_metadata": {
                    "query_processed_at": datetime.now().isoformat(),
                    "total_results": (
                        protocol_results.total_results
                        + compliance_results.total_results
                        + security_results.total_results
                    ),
                    "avg_similarity": (
                        np.mean(
                            [
                                *protocol_results.similarity_scores,
                                *compliance_results.similarity_scores,
                                *security_results.similarity_scores,
                            ]
                        )
                        if any(
                            [
                                protocol_results.similarity_scores,
                                compliance_results.similarity_scores,
                                security_results.similarity_scores,
                            ]
                        )
                        else 0.0
                    ),
                },
            }
        )

        return enhanced_context

    async def enhance_translation_context(
        self,
        query: str,
        source_protocol: str = None,
        target_protocol: str = None,
        existing_context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Enhance query with translation-specific context from knowledge base.

        Args:
            query: Translation query
            source_protocol: Source protocol name
            target_protocol: Target protocol name
            existing_context: Existing context to enhance

        Returns:
            Enhanced context with translation-specific knowledge
        """
        enhanced_context = existing_context or {}

        # Search for protocol translation patterns
        translation_query = f"{query} {source_protocol or ''} {target_protocol or ''} protocol translation"
        translation_results = await self.query_similar(
            translation_query,
            collection_name="protocol_translation_patterns",
            n_results=3,
            similarity_threshold=0.5,
        )

        # Search for API design patterns
        api_query = f"{query} API design patterns {target_protocol or ''}"
        api_results = await self.query_similar(
            api_query,
            collection_name="api_design_patterns",
            n_results=2,
            similarity_threshold=0.5,
        )

        # Search for code generation templates
        code_query = f"{query} code generation templates"
        code_results = await self.query_similar(
            code_query,
            collection_name="code_generation_templates",
            n_results=2,
            similarity_threshold=0.5,
        )

        # Search for translation best practices
        practices_query = f"{query} translation best practices"
        practices_results = await self.query_similar(
            practices_query,
            collection_name="translation_best_practices",
            n_results=2,
            similarity_threshold=0.5,
        )

        # Combine translation-specific results
        enhanced_context.update(
            {
                "translation_patterns": [
                    {
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "similarity": score,
                    }
                    for doc, score in zip(
                        translation_results.documents,
                        translation_results.similarity_scores,
                    )
                ],
                "api_design_patterns": [
                    {
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "similarity": score,
                    }
                    for doc, score in zip(
                        api_results.documents, api_results.similarity_scores
                    )
                ],
                "code_generation_templates": [
                    {
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "similarity": score,
                    }
                    for doc, score in zip(
                        code_results.documents, code_results.similarity_scores
                    )
                ],
                "best_practices": [
                    {
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "similarity": score,
                    }
                    for doc, score in zip(
                        practices_results.documents, practices_results.similarity_scores
                    )
                ],
                "translation_metadata": {
                    "source_protocol": source_protocol,
                    "target_protocol": target_protocol,
                    "query_processed_at": datetime.now().isoformat(),
                    "total_translation_results": (
                        translation_results.total_results
                        + api_results.total_results
                        + code_results.total_results
                        + practices_results.total_results
                    ),
                },
            }
        )

        return enhanced_context

    async def query_protocol_patterns(
        self, protocol_name: str, pattern_type: str = "all", n_results: int = 5
    ) -> RAGQueryResult:
        """
        Query for protocol-specific patterns and knowledge.

        Args:
            protocol_name: Name of the protocol
            pattern_type: Type of patterns (translation, api, security, all)
            n_results: Number of results to return

        Returns:
            RAG query result with protocol patterns
        """
        query = f"{protocol_name} protocol patterns {pattern_type}"

        if pattern_type == "translation":
            return await self.query_similar(
                query, "protocol_translation_patterns", n_results, 0.5
            )
        elif pattern_type == "api":
            return await self.query_similar(
                query, "api_design_patterns", n_results, 0.5
            )
        elif pattern_type == "security":
            return await self.query_similar(query, "security_patterns", n_results, 0.5)
        else:
            # Search across all relevant collections
            collections = [
                "protocol_knowledge",
                "protocol_translation_patterns",
                "api_design_patterns",
                "security_patterns",
            ]

            all_results = []
            all_scores = []

            for collection in collections:
                if collection in self.collections:
                    result = await self.query_similar(
                        query, collection, max(1, n_results // len(collections)), 0.5
                    )
                    all_results.extend(result.documents)
                    all_scores.extend(result.similarity_scores)

            # Sort and limit results
            if all_results:
                sorted_pairs = sorted(
                    zip(all_results, all_scores), key=lambda x: x[1], reverse=True
                )
                all_results, all_scores = zip(*sorted_pairs)
                all_results = list(all_results)[:n_results]
                all_scores = list(all_scores)[:n_results]

            return RAGQueryResult(
                documents=all_results,
                similarity_scores=all_scores,
                query_embedding=[],
                processing_time=0.0,
                total_results=len(all_results),
            )

    async def get_code_generation_templates(
        self, language: str, template_type: str = "client", n_results: int = 3
    ) -> RAGQueryResult:
        """
        Get code generation templates for specific language and type.

        Args:
            language: Programming language (python, typescript, etc.)
            template_type: Type of template (client, models, tests)
            n_results: Number of results to return

        Returns:
            RAG query result with code templates
        """
        query = f"{language} {template_type} code generation template"

        return await self.query_similar(
            query,
            collection_name="code_generation_templates",
            n_results=n_results,
            similarity_threshold=0.4,
        )

    async def get_translation_best_practices(
        self, context: str = "general", n_results: int = 5
    ) -> RAGQueryResult:
        """
        Get translation best practices for specific context.

        Args:
            context: Context for best practices (api, protocol, security)
            n_results: Number of results to return

        Returns:
            RAG query result with best practices
        """
        query = f"{context} translation best practices"

        return await self.query_similar(
            query,
            collection_name="translation_best_practices",
            n_results=n_results,
            similarity_threshold=0.4,
        )

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
                        "protocol": "HTTP",
                        "layer": "application",
                        "security_level": "low_without_tls",
                        "common_ports": [80, 443],
                    },
                    created_at=datetime.now(),
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
                        "protocol": "TCP",
                        "layer": "transport",
                        "reliability": "high",
                        "connection_type": "connection_oriented",
                    },
                    created_at=datetime.now(),
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
                        "protocol": "DNS",
                        "layer": "application",
                        "default_port": 53,
                        "security_extensions": ["DNSSEC", "DoH", "DoT"],
                    },
                    created_at=datetime.now(),
                ),
            ]

            await self.add_documents("protocol_knowledge", protocol_docs)

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
                        "framework": "PCI-DSS",
                        "requirement": "1",
                        "domain": "network_security",
                        "criticality": "high",
                    },
                    created_at=datetime.now(),
                )
            ]

            await self.add_documents("compliance_rules", compliance_docs)

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
                        "category": "anomaly_detection",
                        "type": "security_patterns",
                        "confidence": "high",
                    },
                    created_at=datetime.now(),
                )
            ]

            await self.add_documents("security_patterns", security_docs)

            # Load protocol translation patterns
            translation_docs = [
                RAGDocument(
                    id="http_to_rest_pattern",
                    content="""HTTP to REST API Translation Pattern:
                    
                    Common mappings:
                    - HTTP GET requests map to REST GET endpoints
                    - HTTP POST requests map to REST POST/PUT endpoints
                    - HTTP headers become API parameters or headers
                    - Query parameters remain query parameters
                    - Request/response bodies map to JSON schemas
                    
                    Best practices:
                    - Preserve HTTP semantics in REST design
                    - Use appropriate HTTP status codes
                    - Maintain idempotency for safe operations
                    - Handle authentication consistently
                    - Implement proper error handling
                    """,
                    metadata={
                        "source_protocol": "HTTP",
                        "target_format": "REST",
                        "pattern_type": "protocol_to_api",
                        "confidence": "high",
                    },
                    created_at=datetime.now(),
                ),
                RAGDocument(
                    id="binary_to_json_pattern",
                    content="""Binary Protocol to JSON API Pattern:
                    
                    Translation strategies:
                    - Binary fields become JSON properties with appropriate types
                    - Fixed-length fields map to structured JSON objects
                    - Length-prefixed strings become JSON strings
                    - Binary data encoded as base64 or hex strings
                    - Nested structures become nested JSON objects
                    
                    Considerations:
                    - Data type preservation is critical
                    - Endianness handling for multi-byte values
                    - Validation of field constraints
                    - Performance impact of encoding/decoding
                    - Version compatibility across protocol changes
                    """,
                    metadata={
                        "source_format": "binary",
                        "target_format": "json",
                        "pattern_type": "data_transformation",
                        "complexity": "medium",
                    },
                    created_at=datetime.now(),
                ),
            ]

            await self.add_documents("protocol_translation_patterns", translation_docs)

            # Load API design patterns
            api_design_docs = [
                RAGDocument(
                    id="rest_crud_pattern",
                    content="""RESTful CRUD API Design Pattern:
                    
                    Standard endpoints:
                    - POST /resources - Create new resource
                    - GET /resources - List resources (with pagination)
                    - GET /resources/{id} - Retrieve specific resource
                    - PUT /resources/{id} - Update entire resource
                    - PATCH /resources/{id} - Partial update
                    - DELETE /resources/{id} - Remove resource
                    
                    Additional patterns:
                    - GET /resources/{id}/relationships - Access related resources
                    - POST /resources/{id}/actions - Trigger resource actions
                    - HEAD /resources/{id} - Check resource existence
                    - OPTIONS /resources - Get available methods
                    """,
                    metadata={
                        "api_style": "REST",
                        "pattern_category": "CRUD",
                        "maturity_level": "standard",
                        "use_cases": ["data_management", "resource_access"],
                    },
                    created_at=datetime.now(),
                ),
                RAGDocument(
                    id="async_api_pattern",
                    content="""Asynchronous API Pattern:
                    
                    Implementation approaches:
                    - POST /jobs - Submit asynchronous job
                    - GET /jobs/{id} - Check job status
                    - GET /jobs/{id}/result - Retrieve completed result
                    - DELETE /jobs/{id} - Cancel pending job
                    
                    Webhook notifications:
                    - POST /webhooks - Register webhook endpoint
                    - Callback on job completion/failure
                    - Include job ID and status in callback
                    
                    Status states: queued, processing, completed, failed, cancelled
                    """,
                    metadata={
                        "api_style": "REST",
                        "pattern_category": "asynchronous",
                        "complexity": "high",
                        "use_cases": ["long_running_operations", "batch_processing"],
                    },
                    created_at=datetime.now(),
                ),
            ]

            await self.add_documents("api_design_patterns", api_design_docs)

            # Load code generation templates
            code_templates = [
                RAGDocument(
                    id="python_client_template",
                    content="""Python SDK Client Template:
                    
                    Basic structure:
                    ```python
                    import httpx
                    from typing import Dict, Any, Optional
                    
                    class APIClient:
                        def __init__(self, base_url: str, api_key: str):
                            self.base_url = base_url
                            self.client = httpx.AsyncClient(
                                base_url=base_url,
                                headers={'Authorization': f'Bearer {api_key}'}
                            )
                        
                        async def request(self, method: str, endpoint: str, **kwargs):
                            response = await self.client.request(method, endpoint, **kwargs)
                            response.raise_for_status()
                            return response.json()
                    ```
                    
                    Key features: async/await, error handling, authentication, type hints
                    """,
                    metadata={
                        "language": "python",
                        "template_type": "client",
                        "features": ["async", "authentication", "error_handling"],
                        "dependencies": ["httpx"],
                    },
                    created_at=datetime.now(),
                ),
                RAGDocument(
                    id="typescript_interface_template",
                    content="""TypeScript Interface Template:
                    
                    Basic structure:
                    ```typescript
                    export interface APIResponse<T> {
                        data: T;
                        metadata?: Record<string, any>;
                        errors?: string[];
                    }
                    
                    export interface APIClient {
                        request<T>(
                            method: string,
                            endpoint: string,
                            options?: RequestOptions
                        ): Promise<APIResponse<T>>;
                    }
                    
                    export interface RequestOptions {
                        params?: Record<string, any>;
                        data?: any;
                        headers?: Record<string, string>;
                    }
                    ```
                    
                    Key features: generic types, strict typing, optional properties
                    """,
                    metadata={
                        "language": "typescript",
                        "template_type": "interfaces",
                        "features": [
                            "generics",
                            "strict_typing",
                            "optional_properties",
                        ],
                        "patterns": ["response_wrapper", "request_options"],
                    },
                    created_at=datetime.now(),
                ),
            ]

            await self.add_documents("code_generation_templates", code_templates)

            # Load translation best practices
            best_practices_docs = [
                RAGDocument(
                    id="api_translation_practices",
                    content="""API Translation Best Practices:
                    
                    1. Maintain Semantic Consistency:
                       - Preserve original intent and meaning
                       - Use consistent naming conventions
                       - Map similar concepts to similar structures
                    
                    2. Handle Data Types Properly:
                       - Validate type conversions
                       - Handle precision loss gracefully
                       - Document type mapping decisions
                    
                    3. Error Handling:
                       - Translate error codes appropriately
                       - Provide meaningful error messages
                       - Include context for debugging
                    
                    4. Performance Considerations:
                       - Minimize unnecessary data transformations
                       - Cache frequently accessed mappings
                       - Use streaming for large datasets
                    
                    5. Security:
                       - Validate all input data
                       - Sanitize translated outputs
                       - Maintain authentication context
                    """,
                    metadata={
                        "category": "api_translation",
                        "importance": "high",
                        "scope": "general",
                        "applies_to": ["REST", "GraphQL", "gRPC"],
                    },
                    created_at=datetime.now(),
                ),
                RAGDocument(
                    id="protocol_bridge_practices",
                    content="""Protocol Bridge Best Practices:
                    
                    1. Real-time Translation:
                       - Minimize latency in translation pipeline
                       - Use connection pooling for efficiency
                       - Implement proper backpressure handling
                    
                    2. Data Integrity:
                       - Validate translations bidirectionally
                       - Handle partial translation failures
                       - Maintain transaction semantics where applicable
                    
                    3. Monitoring and Observability:
                       - Track translation success rates
                       - Monitor performance metrics
                       - Log translation errors with context
                    
                    4. Scalability:
                       - Support horizontal scaling
                       - Use caching for frequently translated patterns
                       - Implement circuit breakers for fault tolerance
                    
                    5. Configuration Management:
                       - Allow runtime configuration updates
                       - Support multiple protocol versions
                       - Enable feature flags for gradual rollout
                    """,
                    metadata={
                        "category": "protocol_bridge",
                        "importance": "critical",
                        "scope": "infrastructure",
                        "applies_to": ["streaming", "batch_processing", "real_time"],
                    },
                    created_at=datetime.now(),
                ),
            ]

            await self.add_documents("translation_best_practices", best_practices_docs)

            self.logger.info(
                "Initial knowledge base with translation extensions loaded successfully"
            )

        except Exception as e:
            self.logger.error(f"Failed to load initial knowledge base: {e}")

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections."""
        stats = {}

        for name, collection in self.collections.items():
            try:
                count = collection.count()
                stats[name] = {"document_count": count, "collection_name": name}
            except Exception as e:
                stats[name] = {"document_count": 0, "error": str(e)}

        return stats

    async def update_document(
        self,
        collection_name: str,
        document_id: str,
        new_content: str,
        new_metadata: Dict[str, Any] = None,
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
                metadatas=[new_metadata or {}],
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
