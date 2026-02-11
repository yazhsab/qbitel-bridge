"""Add vector embeddings and semantic search tables

Revision ID: 006
Revises: 005
Create Date: 2025-02-03 10:30:00.000000

This migration adds:
- Vector embeddings for protocol semantic search
- RAG knowledge base tables
- Semantic cache for LLM responses
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '006'
down_revision = '005'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create vector embeddings and semantic search tables."""

    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ═══════════════════════════════════════════════════════════════════
    # PROTOCOL EMBEDDINGS
    # ═══════════════════════════════════════════════════════════════════
    op.create_table(
        'protocol_embeddings',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('protocol_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('protocol_name', sa.String(255), nullable=False),
        sa.Column('chunk_id', sa.String(255), nullable=False),
        sa.Column('chunk_type', sa.String(50), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('content_hash', sa.String(64), nullable=False),
        sa.Column('embedding', postgresql.ARRAY(sa.Float()), nullable=False),
        sa.Column('embedding_model', sa.String(100), nullable=False),
        sa.Column('embedding_dimension', sa.Integer(), nullable=False),
        sa.Column('token_count', sa.Integer(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(), default={}),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.UniqueConstraint('protocol_name', 'chunk_id', name='unique_protocol_chunk'),
    )

    # Create indexes for protocol embeddings
    op.create_index('idx_pe_protocol_id', 'protocol_embeddings', ['protocol_id'])
    op.create_index('idx_pe_protocol_name', 'protocol_embeddings', ['protocol_name'])
    op.create_index('idx_pe_chunk_type', 'protocol_embeddings', ['chunk_type'])
    op.create_index('idx_pe_content_hash', 'protocol_embeddings', ['content_hash'])

    # Create vector index for similarity search (using HNSW)
    op.execute("""
        CREATE INDEX idx_pe_embedding_hnsw ON protocol_embeddings
        USING hnsw ((embedding::vector(1536)) vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)

    # ═══════════════════════════════════════════════════════════════════
    # RAG KNOWLEDGE BASE
    # ═══════════════════════════════════════════════════════════════════
    op.create_table(
        'knowledge_base_documents',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('source_type', sa.String(50), nullable=False),
        sa.Column('source_url', sa.Text(), nullable=True),
        sa.Column('source_file_path', sa.Text(), nullable=True),
        sa.Column('content_type', sa.String(100), nullable=False),
        sa.Column('domain', sa.String(50), nullable=True),
        sa.Column('category', sa.String(100), nullable=True),
        sa.Column('tags', postgresql.ARRAY(sa.String()), default=[]),
        sa.Column('version', sa.String(50), nullable=True),
        sa.Column('language', sa.String(10), default='en'),
        sa.Column('content_hash', sa.String(64), nullable=False, unique=True),
        sa.Column('total_chunks', sa.Integer(), nullable=False, default=0),
        sa.Column('total_tokens', sa.Integer(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('last_indexed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('metadata', postgresql.JSONB(), default={}),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    )

    # Create indexes for knowledge base documents
    op.create_index('idx_kbd_source_type', 'knowledge_base_documents', ['source_type'])
    op.create_index('idx_kbd_domain', 'knowledge_base_documents', ['domain'])
    op.create_index('idx_kbd_category', 'knowledge_base_documents', ['category'])
    op.create_index('idx_kbd_tags', 'knowledge_base_documents', ['tags'], postgresql_using='gin')
    op.create_index('idx_kbd_is_active', 'knowledge_base_documents', ['is_active'])

    # Knowledge base chunks with embeddings
    op.create_table(
        'knowledge_base_chunks',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('content_hash', sa.String(64), nullable=False),
        sa.Column('embedding', postgresql.ARRAY(sa.Float()), nullable=False),
        sa.Column('embedding_model', sa.String(100), nullable=False),
        sa.Column('embedding_dimension', sa.Integer(), nullable=False),
        sa.Column('token_count', sa.Integer(), nullable=True),
        sa.Column('start_char', sa.Integer(), nullable=True),
        sa.Column('end_char', sa.Integer(), nullable=True),
        sa.Column('page_number', sa.Integer(), nullable=True),
        sa.Column('section_title', sa.String(500), nullable=True),
        sa.Column('metadata', postgresql.JSONB(), default={}),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['document_id'], ['knowledge_base_documents.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('document_id', 'chunk_index', name='unique_document_chunk'),
    )

    # Create indexes for knowledge base chunks
    op.create_index('idx_kbc_document_id', 'knowledge_base_chunks', ['document_id'])
    op.create_index('idx_kbc_content_hash', 'knowledge_base_chunks', ['content_hash'])

    # Create vector index for similarity search
    op.execute("""
        CREATE INDEX idx_kbc_embedding_hnsw ON knowledge_base_chunks
        USING hnsw ((embedding::vector(1536)) vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)

    # ═══════════════════════════════════════════════════════════════════
    # SEMANTIC CACHE FOR LLM RESPONSES
    # ═══════════════════════════════════════════════════════════════════
    op.create_table(
        'semantic_cache',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('query_hash', sa.String(64), nullable=False),
        sa.Column('query_text', sa.Text(), nullable=False),
        sa.Column('query_embedding', postgresql.ARRAY(sa.Float()), nullable=False),
        sa.Column('embedding_model', sa.String(100), nullable=False),
        sa.Column('response_text', sa.Text(), nullable=False),
        sa.Column('response_metadata', postgresql.JSONB(), default={}),
        sa.Column('model_used', sa.String(100), nullable=False),
        sa.Column('provider', sa.String(50), nullable=False),
        sa.Column('temperature', sa.DECIMAL(3, 2), nullable=True),
        sa.Column('prompt_tokens', sa.Integer(), nullable=True),
        sa.Column('completion_tokens', sa.Integer(), nullable=True),
        sa.Column('latency_ms', sa.Integer(), nullable=True),
        sa.Column('context_keys', postgresql.ARRAY(sa.String()), default=[]),
        sa.Column('hit_count', sa.Integer(), nullable=False, default=0),
        sa.Column('last_hit_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    )

    # Create indexes for semantic cache
    op.create_index('idx_sc_query_hash', 'semantic_cache', ['query_hash'])
    op.create_index('idx_sc_model_used', 'semantic_cache', ['model_used'])
    op.create_index('idx_sc_expires_at', 'semantic_cache', ['expires_at'])
    op.create_index('idx_sc_context_keys', 'semantic_cache', ['context_keys'], postgresql_using='gin')

    # Create vector index for semantic similarity cache lookup
    op.execute("""
        CREATE INDEX idx_sc_embedding_hnsw ON semantic_cache
        USING hnsw ((query_embedding::vector(1536)) vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)

    # ═══════════════════════════════════════════════════════════════════
    # THREAT INTELLIGENCE EMBEDDINGS
    # ═══════════════════════════════════════════════════════════════════
    op.create_table(
        'threat_intelligence_embeddings',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('ioc_type', sa.String(50), nullable=False),
        sa.Column('ioc_value', sa.Text(), nullable=False),
        sa.Column('ioc_hash', sa.String(64), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('embedding', postgresql.ARRAY(sa.Float()), nullable=False),
        sa.Column('embedding_model', sa.String(100), nullable=False),
        sa.Column('threat_type', sa.String(100), nullable=True),
        sa.Column('severity', sa.String(20), nullable=True),
        sa.Column('confidence', sa.DECIMAL(5, 4), nullable=True),
        sa.Column('mitre_techniques', postgresql.ARRAY(sa.String()), default=[]),
        sa.Column('source', sa.String(255), nullable=True),
        sa.Column('source_url', sa.Text(), nullable=True),
        sa.Column('first_seen', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_seen', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('metadata', postgresql.JSONB(), default={}),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    )

    # Create indexes for threat intelligence
    op.create_index('idx_tie_ioc_type', 'threat_intelligence_embeddings', ['ioc_type'])
    op.create_index('idx_tie_ioc_hash', 'threat_intelligence_embeddings', ['ioc_hash'])
    op.create_index('idx_tie_threat_type', 'threat_intelligence_embeddings', ['threat_type'])
    op.create_index('idx_tie_severity', 'threat_intelligence_embeddings', ['severity'])
    op.create_index('idx_tie_mitre', 'threat_intelligence_embeddings', ['mitre_techniques'], postgresql_using='gin')
    op.create_index('idx_tie_is_active', 'threat_intelligence_embeddings', ['is_active'])

    # Create vector index for threat similarity search
    op.execute("""
        CREATE INDEX idx_tie_embedding_hnsw ON threat_intelligence_embeddings
        USING hnsw ((embedding::vector(1536)) vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)

    # ═══════════════════════════════════════════════════════════════════
    # CREATE HELPER FUNCTIONS FOR SIMILARITY SEARCH
    # ═══════════════════════════════════════════════════════════════════

    op.execute("""
        CREATE OR REPLACE FUNCTION search_protocol_embeddings(
            query_embedding float[],
            match_threshold float DEFAULT 0.7,
            match_count int DEFAULT 10
        )
        RETURNS TABLE (
            id uuid,
            protocol_name varchar(255),
            chunk_type varchar(50),
            content text,
            similarity float
        )
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RETURN QUERY
            SELECT
                pe.id,
                pe.protocol_name,
                pe.chunk_type,
                pe.content,
                1 - (pe.embedding::vector(1536) <=> query_embedding::vector(1536)) as similarity
            FROM protocol_embeddings pe
            WHERE 1 - (pe.embedding::vector(1536) <=> query_embedding::vector(1536)) > match_threshold
            ORDER BY pe.embedding::vector(1536) <=> query_embedding::vector(1536)
            LIMIT match_count;
        END;
        $$
    """)

    op.execute("""
        CREATE OR REPLACE FUNCTION search_knowledge_base(
            query_embedding float[],
            domain_filter varchar DEFAULT NULL,
            match_threshold float DEFAULT 0.7,
            match_count int DEFAULT 10
        )
        RETURNS TABLE (
            chunk_id uuid,
            document_id uuid,
            title varchar(500),
            content text,
            section_title varchar(500),
            similarity float
        )
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RETURN QUERY
            SELECT
                kbc.id as chunk_id,
                kbd.id as document_id,
                kbd.title,
                kbc.content,
                kbc.section_title,
                1 - (kbc.embedding::vector(1536) <=> query_embedding::vector(1536)) as similarity
            FROM knowledge_base_chunks kbc
            JOIN knowledge_base_documents kbd ON kbc.document_id = kbd.id
            WHERE kbd.is_active = true
              AND (domain_filter IS NULL OR kbd.domain = domain_filter)
              AND 1 - (kbc.embedding::vector(1536) <=> query_embedding::vector(1536)) > match_threshold
            ORDER BY kbc.embedding::vector(1536) <=> query_embedding::vector(1536)
            LIMIT match_count;
        END;
        $$
    """)

    op.execute("""
        CREATE OR REPLACE FUNCTION get_semantic_cache(
            query_embedding float[],
            similarity_threshold float DEFAULT 0.95
        )
        RETURNS TABLE (
            id uuid,
            query_text text,
            response_text text,
            similarity float
        )
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RETURN QUERY
            SELECT
                sc.id,
                sc.query_text,
                sc.response_text,
                1 - (sc.query_embedding::vector(1536) <=> query_embedding::vector(1536)) as similarity
            FROM semantic_cache sc
            WHERE (sc.expires_at IS NULL OR sc.expires_at > now())
              AND 1 - (sc.query_embedding::vector(1536) <=> query_embedding::vector(1536)) > similarity_threshold
            ORDER BY sc.query_embedding::vector(1536) <=> query_embedding::vector(1536)
            LIMIT 1;
        END;
        $$
    """)


def downgrade() -> None:
    """Drop vector embeddings tables and functions."""

    # Drop functions
    op.execute("DROP FUNCTION IF EXISTS get_semantic_cache(float[], float)")
    op.execute("DROP FUNCTION IF EXISTS search_knowledge_base(float[], varchar, float, int)")
    op.execute("DROP FUNCTION IF EXISTS search_protocol_embeddings(float[], float, int)")

    # Drop tables
    op.drop_table('threat_intelligence_embeddings')
    op.drop_table('semantic_cache')
    op.drop_table('knowledge_base_chunks')
    op.drop_table('knowledge_base_documents')
    op.drop_table('protocol_embeddings')
