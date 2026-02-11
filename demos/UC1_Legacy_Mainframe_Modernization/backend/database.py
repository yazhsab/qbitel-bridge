"""
QBITEL - UC1 Demo Database Persistence Layer

Production-grade database layer supporting:
- SQLAlchemy ORM with async support
- PostgreSQL for production
- SQLite for development/testing
- Migrations via Alembic patterns
"""

import os
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass, field, asdict
from contextlib import asynccontextmanager
from enum import Enum
import json

# Database imports
try:
    from sqlalchemy import (
        Column, String, Integer, Float, Boolean, DateTime, Text, JSON,
        ForeignKey, Index, create_engine, MetaData, Table,
        select, insert, update, delete, and_, or_,
    )
    from sqlalchemy.ext.asyncio import (
        AsyncSession, create_async_engine, async_sessionmaker,
    )
    from sqlalchemy.orm import (
        DeclarativeBase, Mapped, mapped_column, relationship,
        selectinload,
    )
    from sqlalchemy.pool import NullPool

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DatabaseConfig:
    """Database configuration."""
    driver: str = "sqlite"  # sqlite, postgresql, mysql
    host: str = "localhost"
    port: int = 5432
    database: str = "uc1_demo"
    username: str = ""
    password: str = ""
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create config from environment variables."""
        return cls(
            driver=os.getenv("DB_DRIVER", "sqlite"),
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "uc1_demo"),
            username=os.getenv("DB_USER", ""),
            password=os.getenv("DB_PASSWORD", ""),
            pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
            echo=os.getenv("DB_ECHO", "false").lower() == "true",
        )

    def get_connection_url(self, async_mode: bool = True) -> str:
        """Get database connection URL."""
        if self.driver == "sqlite":
            db_path = os.path.join(
                os.path.dirname(__file__),
                f"{self.database}.db"
            )
            if async_mode:
                return f"sqlite+aiosqlite:///{db_path}"
            return f"sqlite:///{db_path}"
        elif self.driver == "postgresql":
            if async_mode:
                driver = "postgresql+asyncpg"
            else:
                driver = "postgresql"
            return f"{driver}://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        else:
            raise ValueError(f"Unsupported database driver: {self.driver}")


# =============================================================================
# ORM Models (when SQLAlchemy is available)
# =============================================================================

if SQLALCHEMY_AVAILABLE:

    class Base(DeclarativeBase):
        """Base class for ORM models."""
        pass

    class SystemRecord(Base):
        """Legacy system record."""
        __tablename__ = "systems"

        id: Mapped[str] = mapped_column(String(36), primary_key=True)
        name: Mapped[str] = mapped_column(String(200), nullable=False)
        system_type: Mapped[str] = mapped_column(String(50), nullable=False)
        manufacturer: Mapped[Optional[str]] = mapped_column(String(100))
        model: Mapped[Optional[str]] = mapped_column(String(100))
        version: Mapped[Optional[str]] = mapped_column(String(50))
        location: Mapped[Optional[str]] = mapped_column(String(200))
        criticality: Mapped[str] = mapped_column(String(20), default="medium")
        business_function: Mapped[Optional[str]] = mapped_column(Text)
        metadata_json: Mapped[Optional[str]] = mapped_column(Text)  # JSON stored as text
        created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
        updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

        # Relationships
        analyses = relationship("AnalysisRecord", back_populates="system")
        knowledge_items = relationship("KnowledgeRecord", back_populates="system")

        __table_args__ = (
            Index("idx_systems_type", "system_type"),
            Index("idx_systems_criticality", "criticality"),
        )

    class AnalysisRecord(Base):
        """Analysis record."""
        __tablename__ = "analyses"

        id: Mapped[str] = mapped_column(String(36), primary_key=True)
        system_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("systems.id"))
        analysis_type: Mapped[str] = mapped_column(String(50), nullable=False)  # cobol, protocol, health
        status: Mapped[str] = mapped_column(String(20), default="completed")
        result_json: Mapped[Optional[str]] = mapped_column(Text)  # JSON stored as text
        confidence_score: Mapped[Optional[float]] = mapped_column(Float)
        llm_provider: Mapped[Optional[str]] = mapped_column(String(50))
        duration_ms: Mapped[Optional[int]] = mapped_column(Integer)
        created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

        # Relationships
        system = relationship("SystemRecord", back_populates="analyses")

        __table_args__ = (
            Index("idx_analyses_type", "analysis_type"),
            Index("idx_analyses_system", "system_id"),
            Index("idx_analyses_created", "created_at"),
        )

    class KnowledgeRecord(Base):
        """Captured knowledge record."""
        __tablename__ = "knowledge"

        id: Mapped[str] = mapped_column(String(36), primary_key=True)
        system_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("systems.id"))
        expert_id: Mapped[str] = mapped_column(String(100), nullable=False)
        session_type: Mapped[str] = mapped_column(String(50), default="interview")
        title: Mapped[Optional[str]] = mapped_column(String(500))
        content: Mapped[str] = mapped_column(Text, nullable=False)
        formalized_json: Mapped[Optional[str]] = mapped_column(Text)  # JSON stored as text
        confidence_score: Mapped[Optional[float]] = mapped_column(Float)
        status: Mapped[str] = mapped_column(String(20), default="captured")
        created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
        updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

        # Relationships
        system = relationship("SystemRecord", back_populates="knowledge_items")

        __table_args__ = (
            Index("idx_knowledge_system", "system_id"),
            Index("idx_knowledge_expert", "expert_id"),
        )

    class ModernizationPlanRecord(Base):
        """Modernization plan record."""
        __tablename__ = "modernization_plans"

        id: Mapped[str] = mapped_column(String(36), primary_key=True)
        system_id: Mapped[str] = mapped_column(String(36), ForeignKey("systems.id"))
        target_technology: Mapped[str] = mapped_column(String(100), nullable=False)
        status: Mapped[str] = mapped_column(String(20), default="draft")  # draft, approved, in_progress, completed
        phases_json: Mapped[Optional[str]] = mapped_column(Text)
        risk_assessment_json: Mapped[Optional[str]] = mapped_column(Text)
        estimated_effort: Mapped[Optional[str]] = mapped_column(String(100))
        generated_code_json: Mapped[Optional[str]] = mapped_column(Text)
        created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
        updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

        __table_args__ = (
            Index("idx_plans_system", "system_id"),
            Index("idx_plans_status", "status"),
        )

    class GeneratedCodeRecord(Base):
        """Generated code record."""
        __tablename__ = "generated_code"

        id: Mapped[str] = mapped_column(String(36), primary_key=True)
        source_type: Mapped[str] = mapped_column(String(50), nullable=False)  # adapter, migration, test
        target_language: Mapped[str] = mapped_column(String(50), nullable=False)
        target_protocol: Mapped[Optional[str]] = mapped_column(String(50))
        code: Mapped[str] = mapped_column(Text, nullable=False)
        test_code: Mapped[Optional[str]] = mapped_column(Text)
        documentation: Mapped[Optional[str]] = mapped_column(Text)
        dependencies_json: Mapped[Optional[str]] = mapped_column(Text)
        quality_score: Mapped[Optional[float]] = mapped_column(Float)
        created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

        __table_args__ = (
            Index("idx_code_language", "target_language"),
        )


# =============================================================================
# Database Manager
# =============================================================================

class DatabaseManager:
    """Async database manager."""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig.from_env()
        self._engine = None
        self._session_factory = None
        self._initialized = False

    async def initialize(self):
        """Initialize database connection and create tables."""
        if not SQLALCHEMY_AVAILABLE:
            logger.warning("SQLAlchemy not available. Using in-memory storage.")
            self._initialized = True
            return

        try:
            # Create async engine
            url = self.config.get_connection_url(async_mode=True)
            self._engine = create_async_engine(
                url,
                echo=self.config.echo,
                pool_pre_ping=True,
            )

            # Create session factory
            self._session_factory = async_sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )

            # Create tables
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            self._initialized = True
            logger.info(f"Database initialized: {self.config.driver}")

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    async def shutdown(self):
        """Shutdown database connection."""
        if self._engine:
            await self._engine.dispose()
            logger.info("Database connection closed")

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session."""
        if not self._session_factory:
            raise RuntimeError("Database not initialized")

        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    # =========================================================================
    # System Operations
    # =========================================================================

    async def create_system(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new system record."""
        if not SQLALCHEMY_AVAILABLE or not self._initialized:
            return system_data

        async with self.session() as session:
            record = SystemRecord(
                id=system_data["system_id"],
                name=system_data["system_name"],
                system_type=system_data.get("system_type", "unknown"),
                manufacturer=system_data.get("manufacturer"),
                model=system_data.get("model"),
                version=system_data.get("version"),
                location=system_data.get("location"),
                criticality=system_data.get("criticality", "medium"),
                business_function=system_data.get("business_function"),
                metadata_json=json.dumps(system_data.get("metadata", {})),
            )
            session.add(record)
            await session.commit()

            logger.info(f"Created system record: {record.id}")
            return system_data

    async def get_system(self, system_id: str) -> Optional[Dict[str, Any]]:
        """Get system by ID."""
        if not SQLALCHEMY_AVAILABLE or not self._initialized:
            return None

        async with self.session() as session:
            result = await session.execute(
                select(SystemRecord).where(SystemRecord.id == system_id)
            )
            record = result.scalar_one_or_none()

            if record:
                return {
                    "system_id": record.id,
                    "system_name": record.name,
                    "system_type": record.system_type,
                    "manufacturer": record.manufacturer,
                    "model": record.model,
                    "version": record.version,
                    "location": record.location,
                    "criticality": record.criticality,
                    "business_function": record.business_function,
                    "metadata": json.loads(record.metadata_json or "{}"),
                    "created_at": record.created_at.isoformat(),
                    "updated_at": record.updated_at.isoformat(),
                }
            return None

    async def list_systems(
        self,
        system_type: Optional[str] = None,
        criticality: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List systems with optional filters."""
        if not SQLALCHEMY_AVAILABLE or not self._initialized:
            return []

        async with self.session() as session:
            query = select(SystemRecord)

            if system_type:
                query = query.where(SystemRecord.system_type == system_type)
            if criticality:
                query = query.where(SystemRecord.criticality == criticality)

            query = query.order_by(SystemRecord.created_at.desc())
            query = query.limit(limit).offset(offset)

            result = await session.execute(query)
            records = result.scalars().all()

            return [
                {
                    "system_id": r.id,
                    "system_name": r.name,
                    "system_type": r.system_type,
                    "criticality": r.criticality,
                    "created_at": r.created_at.isoformat(),
                }
                for r in records
            ]

    async def delete_system(self, system_id: str) -> bool:
        """Delete system record."""
        if not SQLALCHEMY_AVAILABLE or not self._initialized:
            return False

        async with self.session() as session:
            result = await session.execute(
                delete(SystemRecord).where(SystemRecord.id == system_id)
            )
            return result.rowcount > 0

    # =========================================================================
    # Analysis Operations
    # =========================================================================

    async def save_analysis(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save analysis result."""
        if not SQLALCHEMY_AVAILABLE or not self._initialized:
            return analysis_data

        async with self.session() as session:
            record = AnalysisRecord(
                id=analysis_data["analysis_id"],
                system_id=analysis_data.get("system_id"),
                analysis_type=analysis_data["analysis_type"],
                status=analysis_data.get("status", "completed"),
                result_json=json.dumps(analysis_data.get("result", {})),
                confidence_score=analysis_data.get("confidence_score"),
                llm_provider=analysis_data.get("llm_provider"),
                duration_ms=analysis_data.get("duration_ms"),
            )
            session.add(record)
            await session.commit()

            logger.info(f"Saved analysis: {record.id} ({record.analysis_type})")
            return analysis_data

    async def get_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis by ID."""
        if not SQLALCHEMY_AVAILABLE or not self._initialized:
            return None

        async with self.session() as session:
            result = await session.execute(
                select(AnalysisRecord).where(AnalysisRecord.id == analysis_id)
            )
            record = result.scalar_one_or_none()

            if record:
                return {
                    "analysis_id": record.id,
                    "system_id": record.system_id,
                    "analysis_type": record.analysis_type,
                    "status": record.status,
                    "result": json.loads(record.result_json or "{}"),
                    "confidence_score": record.confidence_score,
                    "llm_provider": record.llm_provider,
                    "duration_ms": record.duration_ms,
                    "created_at": record.created_at.isoformat(),
                }
            return None

    async def list_analyses(
        self,
        system_id: Optional[str] = None,
        analysis_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List analyses with optional filters."""
        if not SQLALCHEMY_AVAILABLE or not self._initialized:
            return []

        async with self.session() as session:
            query = select(AnalysisRecord)

            if system_id:
                query = query.where(AnalysisRecord.system_id == system_id)
            if analysis_type:
                query = query.where(AnalysisRecord.analysis_type == analysis_type)

            query = query.order_by(AnalysisRecord.created_at.desc())
            query = query.limit(limit).offset(offset)

            result = await session.execute(query)
            records = result.scalars().all()

            return [
                {
                    "analysis_id": r.id,
                    "system_id": r.system_id,
                    "analysis_type": r.analysis_type,
                    "status": r.status,
                    "confidence_score": r.confidence_score,
                    "created_at": r.created_at.isoformat(),
                }
                for r in records
            ]

    # =========================================================================
    # Knowledge Operations
    # =========================================================================

    async def save_knowledge(self, knowledge_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save captured knowledge."""
        if not SQLALCHEMY_AVAILABLE or not self._initialized:
            return knowledge_data

        async with self.session() as session:
            record = KnowledgeRecord(
                id=knowledge_data["knowledge_id"],
                system_id=knowledge_data.get("system_id"),
                expert_id=knowledge_data["expert_id"],
                session_type=knowledge_data.get("session_type", "interview"),
                title=knowledge_data.get("title"),
                content=knowledge_data["content"],
                formalized_json=json.dumps(knowledge_data.get("formalized", {})),
                confidence_score=knowledge_data.get("confidence_score"),
                status=knowledge_data.get("status", "captured"),
            )
            session.add(record)
            await session.commit()

            logger.info(f"Saved knowledge: {record.id}")
            return knowledge_data

    async def get_knowledge(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """Get knowledge by ID."""
        if not SQLALCHEMY_AVAILABLE or not self._initialized:
            return None

        async with self.session() as session:
            result = await session.execute(
                select(KnowledgeRecord).where(KnowledgeRecord.id == knowledge_id)
            )
            record = result.scalar_one_or_none()

            if record:
                return {
                    "knowledge_id": record.id,
                    "system_id": record.system_id,
                    "expert_id": record.expert_id,
                    "session_type": record.session_type,
                    "title": record.title,
                    "content": record.content,
                    "formalized": json.loads(record.formalized_json or "{}"),
                    "confidence_score": record.confidence_score,
                    "status": record.status,
                    "created_at": record.created_at.isoformat(),
                }
            return None

    async def search_knowledge(
        self,
        query: str,
        system_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search knowledge base."""
        if not SQLALCHEMY_AVAILABLE or not self._initialized:
            return []

        async with self.session() as session:
            stmt = select(KnowledgeRecord)

            # Basic text search (for production, use full-text search)
            stmt = stmt.where(
                or_(
                    KnowledgeRecord.title.contains(query),
                    KnowledgeRecord.content.contains(query),
                )
            )

            if system_id:
                stmt = stmt.where(KnowledgeRecord.system_id == system_id)

            stmt = stmt.order_by(KnowledgeRecord.created_at.desc()).limit(limit)

            result = await session.execute(stmt)
            records = result.scalars().all()

            return [
                {
                    "knowledge_id": r.id,
                    "title": r.title,
                    "content": r.content[:200] + "..." if len(r.content) > 200 else r.content,
                    "system_id": r.system_id,
                    "confidence_score": r.confidence_score,
                }
                for r in records
            ]

    # =========================================================================
    # Modernization Plan Operations
    # =========================================================================

    async def save_plan(self, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save modernization plan."""
        if not SQLALCHEMY_AVAILABLE or not self._initialized:
            return plan_data

        async with self.session() as session:
            record = ModernizationPlanRecord(
                id=plan_data["plan_id"],
                system_id=plan_data["system_id"],
                target_technology=plan_data["target_technology"],
                status=plan_data.get("status", "draft"),
                phases_json=json.dumps(plan_data.get("phases", [])),
                risk_assessment_json=json.dumps(plan_data.get("risk_assessment", {})),
                estimated_effort=plan_data.get("estimated_effort"),
                generated_code_json=json.dumps(plan_data.get("generated_code")) if plan_data.get("generated_code") else None,
            )
            session.add(record)
            await session.commit()

            logger.info(f"Saved modernization plan: {record.id}")
            return plan_data

    async def get_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get modernization plan by ID."""
        if not SQLALCHEMY_AVAILABLE or not self._initialized:
            return None

        async with self.session() as session:
            result = await session.execute(
                select(ModernizationPlanRecord).where(ModernizationPlanRecord.id == plan_id)
            )
            record = result.scalar_one_or_none()

            if record:
                return {
                    "plan_id": record.id,
                    "system_id": record.system_id,
                    "target_technology": record.target_technology,
                    "status": record.status,
                    "phases": json.loads(record.phases_json or "[]"),
                    "risk_assessment": json.loads(record.risk_assessment_json or "{}"),
                    "estimated_effort": record.estimated_effort,
                    "generated_code": json.loads(record.generated_code_json) if record.generated_code_json else None,
                    "created_at": record.created_at.isoformat(),
                    "updated_at": record.updated_at.isoformat(),
                }
            return None

    # =========================================================================
    # Generated Code Operations
    # =========================================================================

    async def save_generated_code(self, code_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save generated code."""
        if not SQLALCHEMY_AVAILABLE or not self._initialized:
            return code_data

        async with self.session() as session:
            record = GeneratedCodeRecord(
                id=code_data["generation_id"],
                source_type=code_data.get("source_type", "adapter"),
                target_language=code_data["target_language"],
                target_protocol=code_data.get("target_protocol"),
                code=code_data["code"],
                test_code=code_data.get("test_code"),
                documentation=code_data.get("documentation"),
                dependencies_json=json.dumps(code_data.get("dependencies", [])),
                quality_score=code_data.get("quality_score"),
            )
            session.add(record)
            await session.commit()

            logger.info(f"Saved generated code: {record.id}")
            return code_data


# =============================================================================
# In-Memory Fallback Storage
# =============================================================================

class InMemoryStorage:
    """In-memory storage fallback when database is not available."""

    def __init__(self):
        self.systems: Dict[str, Dict] = {}
        self.analyses: Dict[str, Dict] = {}
        self.knowledge: Dict[str, Dict] = {}
        self.plans: Dict[str, Dict] = {}
        self.code: Dict[str, Dict] = {}

    async def create_system(self, data: Dict) -> Dict:
        self.systems[data["system_id"]] = data
        return data

    async def get_system(self, system_id: str) -> Optional[Dict]:
        return self.systems.get(system_id)

    async def list_systems(self, **kwargs) -> List[Dict]:
        return list(self.systems.values())

    async def save_analysis(self, data: Dict) -> Dict:
        self.analyses[data["analysis_id"]] = data
        return data

    async def get_analysis(self, analysis_id: str) -> Optional[Dict]:
        return self.analyses.get(analysis_id)

    async def save_knowledge(self, data: Dict) -> Dict:
        self.knowledge[data["knowledge_id"]] = data
        return data

    async def get_knowledge(self, knowledge_id: str) -> Optional[Dict]:
        return self.knowledge.get(knowledge_id)

    async def save_plan(self, data: Dict) -> Dict:
        self.plans[data["plan_id"]] = data
        return data

    async def get_plan(self, plan_id: str) -> Optional[Dict]:
        return self.plans.get(plan_id)


# =============================================================================
# Global Database Instance
# =============================================================================

_db_manager: Optional[DatabaseManager] = None
_memory_storage: Optional[InMemoryStorage] = None


async def get_database() -> DatabaseManager:
    """Get database manager instance."""
    global _db_manager, _memory_storage

    if _db_manager is None:
        _db_manager = DatabaseManager()
        try:
            await _db_manager.initialize()
        except Exception as e:
            logger.warning(f"Database initialization failed, using in-memory storage: {e}")
            _memory_storage = InMemoryStorage()

    return _db_manager


def get_storage():
    """Get storage (database or in-memory fallback)."""
    if _db_manager and _db_manager._initialized:
        return _db_manager
    if _memory_storage:
        return _memory_storage
    return InMemoryStorage()


async def shutdown_database():
    """Shutdown database connection."""
    global _db_manager
    if _db_manager:
        await _db_manager.shutdown()
        _db_manager = None
