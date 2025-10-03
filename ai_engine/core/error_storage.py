"""
CRONOS AI Engine - Persistent Error Storage
Provides persistent storage for error records using Redis and PostgreSQL.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import asdict
import redis.asyncio as redis
from sqlalchemy import Column, String, Float, Integer, Text, Boolean, JSON, Index
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base

from .error_handling import ErrorRecord, ErrorSeverity, ErrorCategory, RecoveryStrategy
from .exceptions import CronosAIException

logger = logging.getLogger(__name__)

Base = declarative_base()


class ErrorRecordModel(Base):
    """SQLAlchemy model for error records."""
    __tablename__ = 'error_records'
    
    error_id = Column(String(36), primary_key=True)
    timestamp = Column(Float, nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)
    category = Column(String(50), nullable=False, index=True)
    component = Column(String(100), nullable=False, index=True)
    operation = Column(String(100), nullable=False)
    exception_type = Column(String(100), nullable=False)
    exception_message = Column(Text, nullable=False)
    stack_trace = Column(Text, nullable=False)
    context = Column(JSON, nullable=False)
    recovery_attempted = Column(Boolean, default=False)
    recovery_successful = Column(Boolean, default=False)
    recovery_strategy = Column(String(50), nullable=True)
    retry_count = Column(Integer, default=0)
    metadata = Column(JSON, nullable=True)
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_component_timestamp', 'component', 'timestamp'),
        Index('idx_severity_timestamp', 'severity', 'timestamp'),
        Index('idx_category_timestamp', 'category', 'timestamp'),
    )


class PersistentErrorStorage:
    """
    Persistent error storage using Redis for fast access and PostgreSQL for long-term storage.
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        postgres_url: str = "postgresql+asyncpg://user:pass@localhost/cronos_ai",
        redis_ttl: int = 86400,  # 24 hours
        postgres_retention_days: int = 90
    ):
        self.redis_url = redis_url
        self.postgres_url = postgres_url
        self.redis_ttl = redis_ttl
        self.postgres_retention_days = postgres_retention_days
        
        self.redis_client: Optional[redis.Redis] = None
        self.db_engine = None
        self.async_session_maker = None
        
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize storage connections."""
        try:
            # Initialize Redis
            self.redis_client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis_client.ping()
            self.logger.info("Redis connection established")
            
            # Initialize PostgreSQL
            self.db_engine = create_async_engine(
                self.postgres_url,
                echo=False,
                pool_size=10,
                max_overflow=20
            )
            
            # Create tables
            async with self.db_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            self.async_session_maker = async_sessionmaker(
                self.db_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            self.logger.info("PostgreSQL connection established")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize error storage: {e}")
            raise CronosAIException(f"Error storage initialization failed: {e}")
    
    async def store_error(self, error_record: ErrorRecord) -> bool:
        """
        Store error record in both Redis and PostgreSQL.
        
        Args:
            error_record: Error record to store
            
        Returns:
            True if stored successfully
        """
        try:
            error_dict = error_record.to_dict()
            error_json = json.dumps(error_dict)
            
            # Store in Redis for fast access
            if self.redis_client:
                redis_key = f"error:{error_record.error_id}"
                await self.redis_client.setex(
                    redis_key,
                    self.redis_ttl,
                    error_json
                )
                
                # Add to component-specific sorted set
                component_key = f"errors:component:{error_record.component}"
                await self.redis_client.zadd(
                    component_key,
                    {error_record.error_id: error_record.timestamp}
                )
                await self.redis_client.expire(component_key, self.redis_ttl)
                
                # Add to severity-specific sorted set
                severity_key = f"errors:severity:{error_record.severity.value}"
                await self.redis_client.zadd(
                    severity_key,
                    {error_record.error_id: error_record.timestamp}
                )
                await self.redis_client.expire(severity_key, self.redis_ttl)
            
            # Store in PostgreSQL for long-term storage
            if self.async_session_maker:
                async with self.async_session_maker() as session:
                    db_record = ErrorRecordModel(
                        error_id=error_record.error_id,
                        timestamp=error_record.timestamp,
                        severity=error_record.severity.value,
                        category=error_record.category.value,
                        component=error_record.component,
                        operation=error_record.operation,
                        exception_type=error_record.exception_type,
                        exception_message=error_record.exception_message,
                        stack_trace=error_record.stack_trace,
                        context=error_dict['context'],
                        recovery_attempted=error_record.recovery_attempted,
                        recovery_successful=error_record.recovery_successful,
                        recovery_strategy=error_record.recovery_strategy.value if error_record.recovery_strategy else None,
                        retry_count=error_record.retry_count,
                        metadata=error_record.metadata
                    )
                    session.add(db_record)
                    await session.commit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store error record {error_record.error_id}: {e}")
            return False
    
    async def get_error(self, error_id: str) -> Optional[Dict[str, Any]]:
        """Get error record by ID."""
        try:
            # Try Redis first
            if self.redis_client:
                redis_key = f"error:{error_id}"
                error_json = await self.redis_client.get(redis_key)
                if error_json:
                    return json.loads(error_json)
            
            # Fall back to PostgreSQL
            if self.async_session_maker:
                async with self.async_session_maker() as session:
                    result = await session.get(ErrorRecordModel, error_id)
                    if result:
                        return {
                            'error_id': result.error_id,
                            'timestamp': result.timestamp,
                            'severity': result.severity,
                            'category': result.category,
                            'component': result.component,
                            'operation': result.operation,
                            'exception_type': result.exception_type,
                            'exception_message': result.exception_message,
                            'stack_trace': result.stack_trace,
                            'context': result.context,
                            'recovery_attempted': result.recovery_attempted,
                            'recovery_successful': result.recovery_successful,
                            'recovery_strategy': result.recovery_strategy,
                            'retry_count': result.retry_count,
                            'metadata': result.metadata
                        }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get error record {error_id}: {e}")
            return None
    
    async def get_errors_by_component(
        self,
        component: str,
        limit: int = 100,
        since: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get recent errors for a component."""
        try:
            errors = []
            
            # Try Redis first
            if self.redis_client:
                component_key = f"errors:component:{component}"
                min_score = since if since else 0
                error_ids = await self.redis_client.zrangebyscore(
                    component_key,
                    min_score,
                    '+inf',
                    start=0,
                    num=limit
                )
                
                for error_id in error_ids:
                    error = await self.get_error(error_id)
                    if error:
                        errors.append(error)
            
            # If not enough from Redis, query PostgreSQL
            if len(errors) < limit and self.async_session_maker:
                async with self.async_session_maker() as session:
                    from sqlalchemy import select
                    
                    query = select(ErrorRecordModel).where(
                        ErrorRecordModel.component == component
                    )
                    
                    if since:
                        query = query.where(ErrorRecordModel.timestamp >= since)
                    
                    query = query.order_by(ErrorRecordModel.timestamp.desc()).limit(limit)
                    
                    result = await session.execute(query)
                    db_errors = result.scalars().all()
                    
                    for db_error in db_errors:
                        if len(errors) >= limit:
                            break
                        error_dict = {
                            'error_id': db_error.error_id,
                            'timestamp': db_error.timestamp,
                            'severity': db_error.severity,
                            'category': db_error.category,
                            'component': db_error.component,
                            'operation': db_error.operation,
                            'exception_type': db_error.exception_type,
                            'exception_message': db_error.exception_message,
                            'stack_trace': db_error.stack_trace,
                            'context': db_error.context,
                            'recovery_attempted': db_error.recovery_attempted,
                            'recovery_successful': db_error.recovery_successful,
                            'recovery_strategy': db_error.recovery_strategy,
                            'retry_count': db_error.retry_count,
                            'metadata': db_error.metadata
                        }
                        if error_dict not in errors:
                            errors.append(error_dict)
            
            return errors
            
        except Exception as e:
            self.logger.error(f"Failed to get errors for component {component}: {e}")
            return []
    
    async def get_error_statistics(
        self,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Get error statistics for the specified time window."""
        try:
            cutoff_time = time.time() - (time_window_hours * 3600)
            
            stats = {
                'total_errors': 0,
                'by_severity': {},
                'by_category': {},
                'by_component': {},
                'recovery_rate': 0.0,
                'time_window_hours': time_window_hours
            }
            
            if self.async_session_maker:
                async with self.async_session_maker() as session:
                    from sqlalchemy import select, func
                    
                    # Total errors
                    total_query = select(func.count(ErrorRecordModel.error_id)).where(
                        ErrorRecordModel.timestamp >= cutoff_time
                    )
                    result = await session.execute(total_query)
                    stats['total_errors'] = result.scalar() or 0
                    
                    # By severity
                    severity_query = select(
                        ErrorRecordModel.severity,
                        func.count(ErrorRecordModel.error_id)
                    ).where(
                        ErrorRecordModel.timestamp >= cutoff_time
                    ).group_by(ErrorRecordModel.severity)
                    
                    result = await session.execute(severity_query)
                    stats['by_severity'] = {row[0]: row[1] for row in result}
                    
                    # By category
                    category_query = select(
                        ErrorRecordModel.category,
                        func.count(ErrorRecordModel.error_id)
                    ).where(
                        ErrorRecordModel.timestamp >= cutoff_time
                    ).group_by(ErrorRecordModel.category)
                    
                    result = await session.execute(category_query)
                    stats['by_category'] = {row[0]: row[1] for row in result}
                    
                    # By component
                    component_query = select(
                        ErrorRecordModel.component,
                        func.count(ErrorRecordModel.error_id)
                    ).where(
                        ErrorRecordModel.timestamp >= cutoff_time
                    ).group_by(ErrorRecordModel.component)
                    
                    result = await session.execute(component_query)
                    stats['by_component'] = {row[0]: row[1] for row in result}
                    
                    # Recovery rate
                    recovery_query = select(
                        func.count(ErrorRecordModel.error_id).filter(
                            ErrorRecordModel.recovery_attempted == True
                        ),
                        func.count(ErrorRecordModel.error_id).filter(
                            ErrorRecordModel.recovery_successful == True
                        )
                    ).where(
                        ErrorRecordModel.timestamp >= cutoff_time
                    )
                    
                    result = await session.execute(recovery_query)
                    row = result.first()
                    if row and row[0] > 0:
                        stats['recovery_rate'] = row[1] / row[0]
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get error statistics: {e}")
            return {}
    
    async def cleanup_old_errors(self):
        """Clean up old error records from PostgreSQL."""
        try:
            if self.async_session_maker:
                cutoff_time = time.time() - (self.postgres_retention_days * 86400)
                
                async with self.async_session_maker() as session:
                    from sqlalchemy import delete
                    
                    delete_query = delete(ErrorRecordModel).where(
                        ErrorRecordModel.timestamp < cutoff_time
                    )
                    
                    result = await session.execute(delete_query)
                    await session.commit()
                    
                    deleted_count = result.rowcount
                    self.logger.info(f"Cleaned up {deleted_count} old error records")
                    
                    return deleted_count
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old errors: {e}")
            return 0
    
    async def close(self):
        """Close storage connections."""
        try:
            if self.redis_client:
                await self.redis_client.close()
                self.logger.info("Redis connection closed")
            
            if self.db_engine:
                await self.db_engine.dispose()
                self.logger.info("PostgreSQL connection closed")
                
        except Exception as e:
            self.logger.error(f"Error closing storage connections: {e}")


# Global storage instance
_error_storage: Optional[PersistentErrorStorage] = None


async def get_error_storage(
    redis_url: Optional[str] = None,
    postgres_url: Optional[str] = None
) -> PersistentErrorStorage:
    """Get or create global error storage instance."""
    global _error_storage
    
    if _error_storage is None:
        _error_storage = PersistentErrorStorage(
            redis_url=redis_url or "redis://localhost:6379/0",
            postgres_url=postgres_url or "postgresql+asyncpg://user:pass@localhost/cronos_ai"
        )
        await _error_storage.initialize()
    
    return _error_storage