#!/usr/bin/env python3
"""
QBITEL - TimescaleDB Integration
High-performance time-series database integration for metrics, logs, and analysis data.
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
import asyncpg
from asyncpg import Connection, Pool
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import uuid
import hashlib

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from integration.config.unified_config import get_config, get_service_config
from integration.orchestrator.service_integration import get_orchestrator, Message
from integration.streaming.kafka_streaming_service import StreamEvent, StreamEventType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesRecord:
    """Time series record structure"""

    timestamp: datetime
    metric_name: str
    value: float
    labels: Dict[str, str]
    source_component: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LogRecord:
    """Log record structure"""

    timestamp: datetime
    level: str
    component: str
    message: str
    correlation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SecurityEventRecord:
    """Security event record structure"""

    timestamp: datetime
    event_type: str
    severity: str
    source_ip: str
    dest_ip: str
    threat_level: float
    details: Dict[str, Any]
    correlation_id: Optional[str] = None


@dataclass
class ProtocolAnalysisRecord:
    """Protocol analysis record structure"""

    timestamp: datetime
    session_id: str
    protocol: str
    confidence: float
    analysis_result: Dict[str, Any]
    processing_time_ms: float
    correlation_id: Optional[str] = None


class TimescaleDBIntegration:
    """
    Production TimescaleDB integration for QBITEL.
    Handles time-series data storage and retrieval with optimal performance.
    """

    def __init__(self):
        self.config = get_service_config("database")
        self.orchestrator = get_orchestrator()

        # Database connection pool
        self.pool: Optional[Pool] = None

        # Data buffers for batch inserts
        self.metrics_buffer: List[TimeSeriesRecord] = []
        self.logs_buffer: List[LogRecord] = []
        self.security_buffer: List[SecurityEventRecord] = []
        self.protocol_buffer: List[ProtocolAnalysisRecord] = []

        # Buffer management
        self.max_buffer_size = 10000
        self.flush_interval = 30  # seconds
        self.last_flush_time = time.time()

        # Performance tracking
        self.records_inserted = 0
        self.batch_insert_count = 0
        self.insert_errors = 0
        self.query_count = 0

        # Thread pool for database operations
        self.db_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="db_")

        # Hypertables configuration
        self.hypertables = {
            "metrics": {
                "time_column": "timestamp",
                "chunk_time_interval": "1 hour",
                "compression_after": "7 days",
                "retention_policy": "30 days",
            },
            "logs": {
                "time_column": "timestamp",
                "chunk_time_interval": "6 hours",
                "compression_after": "3 days",
                "retention_policy": "90 days",
            },
            "security_events": {
                "time_column": "timestamp",
                "chunk_time_interval": "1 hour",
                "compression_after": "7 days",
                "retention_policy": "365 days",
            },
            "protocol_analysis": {
                "time_column": "timestamp",
                "chunk_time_interval": "30 minutes",
                "compression_after": "1 day",
                "retention_policy": "7 days",
            },
        }

        # Running state
        self.running = False

    async def initialize(self):
        """Initialize TimescaleDB integration"""
        logger.info("Initializing TimescaleDB Integration...")

        try:
            # Create database connection pool
            await self._create_connection_pool()

            # Initialize database schema
            await self._initialize_schema()

            # Set up hypertables
            await self._setup_hypertables()

            # Create indexes for optimal performance
            await self._create_indexes()

            # Set up retention policies
            await self._setup_retention_policies()

            # Start background tasks
            await self._start_background_tasks()

            logger.info("TimescaleDB Integration initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize TimescaleDB integration: {e}")
            raise

    async def _create_connection_pool(self):
        """Create database connection pool"""
        try:
            db_config = {
                "host": self.config.get("host", "localhost"),
                "port": self.config.get("port", 5432),
                "database": self.config.get("database", "qbitel"),
                "user": self.config.get("username", "qbitel_user"),
                "password": self.config.get("password", ""),
                "ssl": self.config.get("ssl_mode", "prefer"),
                "min_size": 5,
                "max_size": self.config.get("connection_pool_size", 20),
                "command_timeout": self.config.get("connection_timeout", 30),
            }

            self.pool = await asyncpg.create_pool(**db_config)

            # Test connection
            async with self.pool.acquire() as conn:
                version = await conn.fetchval("SELECT version()")
                logger.info(f"Connected to database: {version}")

        except Exception as e:
            logger.error(f"Failed to create database connection pool: {e}")
            raise

    async def _initialize_schema(self):
        """Initialize database schema"""
        try:
            async with self.pool.acquire() as conn:
                # Enable TimescaleDB extension
                await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")

                # Create metrics table
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS metrics (
                        timestamp TIMESTAMPTZ NOT NULL,
                        metric_name VARCHAR(255) NOT NULL,
                        value DOUBLE PRECISION NOT NULL,
                        labels JSONB,
                        source_component VARCHAR(100) NOT NULL,
                        metadata JSONB,
                        PRIMARY KEY (timestamp, metric_name, source_component)
                    )
                """
                )

                # Create logs table
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS logs (
                        timestamp TIMESTAMPTZ NOT NULL,
                        level VARCHAR(20) NOT NULL,
                        component VARCHAR(100) NOT NULL,
                        message TEXT NOT NULL,
                        correlation_id VARCHAR(50),
                        metadata JSONB,
                        PRIMARY KEY (timestamp, component)
                    )
                """
                )

                # Create security events table
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS security_events (
                        timestamp TIMESTAMPTZ NOT NULL,
                        event_type VARCHAR(100) NOT NULL,
                        severity VARCHAR(20) NOT NULL,
                        source_ip INET NOT NULL,
                        dest_ip INET NOT NULL,
                        threat_level DOUBLE PRECISION NOT NULL,
                        details JSONB NOT NULL,
                        correlation_id VARCHAR(50),
                        PRIMARY KEY (timestamp, source_ip, dest_ip)
                    )
                """
                )

                # Create protocol analysis table
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS protocol_analysis (
                        timestamp TIMESTAMPTZ NOT NULL,
                        session_id VARCHAR(100) NOT NULL,
                        protocol VARCHAR(50) NOT NULL,
                        confidence DOUBLE PRECISION NOT NULL,
                        analysis_result JSONB NOT NULL,
                        processing_time_ms DOUBLE PRECISION NOT NULL,
                        correlation_id VARCHAR(50),
                        PRIMARY KEY (timestamp, session_id)
                    )
                """
                )

                logger.info("Database schema initialized")

        except Exception as e:
            logger.error(f"Failed to initialize database schema: {e}")
            raise

    async def _setup_hypertables(self):
        """Set up TimescaleDB hypertables"""
        try:
            async with self.pool.acquire() as conn:
                for table_name, config in self.hypertables.items():
                    # Check if hypertable already exists
                    exists = await conn.fetchval(
                        """
                        SELECT EXISTS (
                            SELECT 1 FROM timescaledb_information.hypertables 
                            WHERE hypertable_name = $1
                        )
                    """,
                        table_name,
                    )

                    if not exists:
                        # Create hypertable
                        await conn.execute(
                            f"""
                            SELECT create_hypertable(
                                '{table_name}', 
                                '{config["time_column"]}',
                                chunk_time_interval => INTERVAL '{config["chunk_time_interval"]}'
                            )
                        """
                        )
                        logger.info(f"Created hypertable: {table_name}")

                    # Enable compression if not already enabled
                    compression_enabled = await conn.fetchval(
                        """
                        SELECT EXISTS (
                            SELECT 1 FROM timescaledb_information.compression_settings 
                            WHERE hypertable_name = $1
                        )
                    """,
                        table_name,
                    )

                    if not compression_enabled:
                        await conn.execute(
                            f"""
                            ALTER TABLE {table_name} SET (
                                timescaledb.compress,
                                timescaledb.compress_segmentby = 'source_component'
                            )
                        """
                        )

                        # Add compression policy
                        await conn.execute(
                            f"""
                            SELECT add_compression_policy(
                                '{table_name}', 
                                INTERVAL '{config["compression_after"]}'
                            )
                        """
                        )
                        logger.info(f"Enabled compression for: {table_name}")

        except Exception as e:
            logger.error(f"Failed to setup hypertables: {e}")
            raise

    async def _create_indexes(self):
        """Create indexes for optimal query performance"""
        try:
            async with self.pool.acquire() as conn:
                # Metrics indexes
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_metrics_metric_name_time 
                    ON metrics (metric_name, timestamp DESC)
                """
                )

                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_metrics_labels 
                    ON metrics USING GIN (labels)
                """
                )

                # Logs indexes
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_logs_component_time 
                    ON logs (component, timestamp DESC)
                """
                )

                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_logs_level 
                    ON logs (level, timestamp DESC)
                """
                )

                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_logs_correlation_id 
                    ON logs (correlation_id) WHERE correlation_id IS NOT NULL
                """
                )

                # Security events indexes
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_security_events_type_time 
                    ON security_events (event_type, timestamp DESC)
                """
                )

                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_security_events_severity 
                    ON security_events (severity, timestamp DESC)
                """
                )

                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_security_events_source_ip 
                    ON security_events (source_ip, timestamp DESC)
                """
                )

                # Protocol analysis indexes
                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_protocol_analysis_protocol 
                    ON protocol_analysis (protocol, timestamp DESC)
                """
                )

                await conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_protocol_analysis_session 
                    ON protocol_analysis (session_id, timestamp DESC)
                """
                )

                logger.info("Database indexes created")

        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            raise

    async def _setup_retention_policies(self):
        """Set up data retention policies"""
        try:
            async with self.pool.acquire() as conn:
                for table_name, config in self.hypertables.items():
                    # Check if retention policy exists
                    policy_exists = await conn.fetchval(
                        """
                        SELECT EXISTS (
                            SELECT 1 FROM timescaledb_information.jobs 
                            WHERE proc_name = 'policy_retention' 
                            AND hypertable_name = $1
                        )
                    """,
                        table_name,
                    )

                    if not policy_exists:
                        await conn.execute(
                            f"""
                            SELECT add_retention_policy(
                                '{table_name}', 
                                INTERVAL '{config["retention_policy"]}'
                            )
                        """
                        )
                        logger.info(
                            f"Added retention policy for {table_name}: {config['retention_policy']}"
                        )

        except Exception as e:
            logger.error(f"Failed to setup retention policies: {e}")
            raise

    async def _start_background_tasks(self):
        """Start background processing tasks"""
        self.running = True

        # Start buffer flush task
        asyncio.create_task(self._buffer_flush_task())

        # Start metrics collector
        asyncio.create_task(self._metrics_collector())

        # Start health monitor
        asyncio.create_task(self._health_monitor())

        logger.info("Background tasks started")

    async def _buffer_flush_task(self):
        """Flush data buffers to database"""
        while self.running:
            try:
                current_time = time.time()
                should_flush = (
                    len(self.metrics_buffer) >= self.max_buffer_size
                    or len(self.logs_buffer) >= self.max_buffer_size
                    or len(self.security_buffer) >= self.max_buffer_size
                    or len(self.protocol_buffer) >= self.max_buffer_size
                    or (current_time - self.last_flush_time) > self.flush_interval
                )

                if should_flush:
                    await self._flush_all_buffers()
                    self.last_flush_time = current_time

                await asyncio.sleep(1)  # Check every second

            except Exception as e:
                logger.error(f"Error in buffer flush task: {e}")
                await asyncio.sleep(5)

    async def _flush_all_buffers(self):
        """Flush all data buffers"""
        try:
            tasks = []

            if self.metrics_buffer:
                tasks.append(self._flush_metrics_buffer())

            if self.logs_buffer:
                tasks.append(self._flush_logs_buffer())

            if self.security_buffer:
                tasks.append(self._flush_security_buffer())

            if self.protocol_buffer:
                tasks.append(self._flush_protocol_buffer())

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                self.batch_insert_count += 1

        except Exception as e:
            logger.error(f"Error flushing buffers: {e}")
            self.insert_errors += 1

    async def _flush_metrics_buffer(self):
        """Flush metrics buffer to database"""
        if not self.metrics_buffer:
            return

        try:
            records = self.metrics_buffer.copy()
            self.metrics_buffer.clear()

            async with self.pool.acquire() as conn:
                # Prepare data for batch insert
                data = [
                    (
                        record.timestamp,
                        record.metric_name,
                        record.value,
                        json.dumps(record.labels),
                        record.source_component,
                        json.dumps(record.metadata) if record.metadata else None,
                    )
                    for record in records
                ]

                # Batch insert
                await conn.executemany(
                    """
                    INSERT INTO metrics (
                        timestamp, metric_name, value, labels, 
                        source_component, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (timestamp, metric_name, source_component) 
                    DO UPDATE SET value = EXCLUDED.value, labels = EXCLUDED.labels
                """,
                    data,
                )

                self.records_inserted += len(records)
                logger.debug(f"Flushed {len(records)} metrics records")

        except Exception as e:
            logger.error(f"Error flushing metrics buffer: {e}")
            self.insert_errors += 1

    async def _flush_logs_buffer(self):
        """Flush logs buffer to database"""
        if not self.logs_buffer:
            return

        try:
            records = self.logs_buffer.copy()
            self.logs_buffer.clear()

            async with self.pool.acquire() as conn:
                data = [
                    (
                        record.timestamp,
                        record.level,
                        record.component,
                        record.message,
                        record.correlation_id,
                        json.dumps(record.metadata) if record.metadata else None,
                    )
                    for record in records
                ]

                await conn.executemany(
                    """
                    INSERT INTO logs (
                        timestamp, level, component, message, 
                        correlation_id, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (timestamp, component) DO NOTHING
                """,
                    data,
                )

                self.records_inserted += len(records)
                logger.debug(f"Flushed {len(records)} log records")

        except Exception as e:
            logger.error(f"Error flushing logs buffer: {e}")
            self.insert_errors += 1

    async def _flush_security_buffer(self):
        """Flush security events buffer to database"""
        if not self.security_buffer:
            return

        try:
            records = self.security_buffer.copy()
            self.security_buffer.clear()

            async with self.pool.acquire() as conn:
                data = [
                    (
                        record.timestamp,
                        record.event_type,
                        record.severity,
                        record.source_ip,
                        record.dest_ip,
                        record.threat_level,
                        json.dumps(record.details),
                        record.correlation_id,
                    )
                    for record in records
                ]

                await conn.executemany(
                    """
                    INSERT INTO security_events (
                        timestamp, event_type, severity, source_ip, dest_ip,
                        threat_level, details, correlation_id
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (timestamp, source_ip, dest_ip) DO NOTHING
                """,
                    data,
                )

                self.records_inserted += len(records)
                logger.debug(f"Flushed {len(records)} security event records")

        except Exception as e:
            logger.error(f"Error flushing security buffer: {e}")
            self.insert_errors += 1

    async def _flush_protocol_buffer(self):
        """Flush protocol analysis buffer to database"""
        if not self.protocol_buffer:
            return

        try:
            records = self.protocol_buffer.copy()
            self.protocol_buffer.clear()

            async with self.pool.acquire() as conn:
                data = [
                    (
                        record.timestamp,
                        record.session_id,
                        record.protocol,
                        record.confidence,
                        json.dumps(record.analysis_result),
                        record.processing_time_ms,
                        record.correlation_id,
                    )
                    for record in records
                ]

                await conn.executemany(
                    """
                    INSERT INTO protocol_analysis (
                        timestamp, session_id, protocol, confidence,
                        analysis_result, processing_time_ms, correlation_id
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (timestamp, session_id) DO NOTHING
                """,
                    data,
                )

                self.records_inserted += len(records)
                logger.debug(f"Flushed {len(records)} protocol analysis records")

        except Exception as e:
            logger.error(f"Error flushing protocol buffer: {e}")
            self.insert_errors += 1

    async def _metrics_collector(self):
        """Collect integration performance metrics"""
        while self.running:
            try:
                await asyncio.sleep(60)  # Collect every minute

                # Calculate metrics
                uptime = time.time() - self.last_flush_time
                buffer_utilization = {
                    "metrics_buffer": len(self.metrics_buffer) / self.max_buffer_size,
                    "logs_buffer": len(self.logs_buffer) / self.max_buffer_size,
                    "security_buffer": len(self.security_buffer) / self.max_buffer_size,
                    "protocol_buffer": len(self.protocol_buffer) / self.max_buffer_size,
                }

                metrics = {
                    "records_inserted": self.records_inserted,
                    "batch_insert_count": self.batch_insert_count,
                    "insert_errors": self.insert_errors,
                    "error_rate": self.insert_errors / max(self.batch_insert_count, 1),
                    "buffer_utilization": buffer_utilization,
                    "records_per_second": self.records_inserted / max(uptime, 1),
                }

                # Send metrics to orchestrator
                message = Message(
                    id=f"timescaledb_metrics_{time.time()}",
                    timestamp=time.time(),
                    source="timescaledb_integration",
                    destination="orchestrator",
                    message_type="metric_update",
                    payload={
                        "component": "timescaledb_integration",
                        "metrics": metrics,
                    },
                )

                await self.orchestrator.send_message(message)

            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(60)

    async def _health_monitor(self):
        """Monitor database health"""
        while self.running:
            try:
                await asyncio.sleep(120)  # Check every 2 minutes

                # Test database connection
                async with self.pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")

                health_status = "healthy"

                # Check buffer levels
                max_buffer_usage = max(
                    [
                        len(self.metrics_buffer) / self.max_buffer_size,
                        len(self.logs_buffer) / self.max_buffer_size,
                        len(self.security_buffer) / self.max_buffer_size,
                        len(self.protocol_buffer) / self.max_buffer_size,
                    ]
                )

                if max_buffer_usage > 0.8:
                    health_status = "degraded"

                # Send health status
                message = Message(
                    id=f"timescaledb_health_{time.time()}",
                    timestamp=time.time(),
                    source="timescaledb_integration",
                    destination="orchestrator",
                    message_type="health_check",
                    payload={
                        "component": "timescaledb_integration",
                        "status": health_status,
                        "buffer_usage": max_buffer_usage,
                        "connection_pool_size": self.pool.get_size(),
                    },
                )

                await self.orchestrator.send_message(message)

            except Exception as e:
                logger.error(f"Database health check failed: {e}")
                await asyncio.sleep(120)

    # Public API methods
    async def insert_metric(self, metric: TimeSeriesRecord):
        """Insert a single metric record"""
        self.metrics_buffer.append(metric)

        # Flush if buffer is full
        if len(self.metrics_buffer) >= self.max_buffer_size:
            await self._flush_metrics_buffer()

    async def insert_log(self, log: LogRecord):
        """Insert a single log record"""
        self.logs_buffer.append(log)

        if len(self.logs_buffer) >= self.max_buffer_size:
            await self._flush_logs_buffer()

    async def insert_security_event(self, event: SecurityEventRecord):
        """Insert a single security event record"""
        self.security_buffer.append(event)

        if len(self.security_buffer) >= self.max_buffer_size:
            await self._flush_security_buffer()

    async def insert_protocol_analysis(self, analysis: ProtocolAnalysisRecord):
        """Insert a single protocol analysis record"""
        self.protocol_buffer.append(analysis)

        if len(self.protocol_buffer) >= self.max_buffer_size:
            await self._flush_protocol_buffer()

    async def query_metrics(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        labels: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Query metrics data"""
        try:
            async with self.pool.acquire() as conn:
                query = """
                    SELECT timestamp, value, labels, source_component
                    FROM metrics 
                    WHERE metric_name = $1 
                    AND timestamp BETWEEN $2 AND $3
                """
                params = [metric_name, start_time, end_time]

                if labels:
                    query += " AND labels @> $4"
                    params.append(json.dumps(labels))

                query += " ORDER BY timestamp DESC"

                rows = await conn.fetch(query, *params)
                self.query_count += 1

                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Error querying metrics: {e}")
            return []

    async def query_security_events(
        self,
        start_time: datetime,
        end_time: datetime,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query security events"""
        try:
            async with self.pool.acquire() as conn:
                query = """
                    SELECT timestamp, event_type, severity, source_ip, dest_ip,
                           threat_level, details, correlation_id
                    FROM security_events 
                    WHERE timestamp BETWEEN $1 AND $2
                """
                params = [start_time, end_time]

                if event_type:
                    query += " AND event_type = $3"
                    params.append(event_type)

                if severity:
                    query += f" AND severity = ${len(params) + 1}"
                    params.append(severity)

                query += " ORDER BY timestamp DESC LIMIT 1000"

                rows = await conn.fetch(query, *params)
                self.query_count += 1

                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Error querying security events: {e}")
            return []

    async def get_protocol_analysis_stats(
        self, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """Get protocol analysis statistics"""
        try:
            async with self.pool.acquire() as conn:
                # Get protocol distribution
                protocol_stats = await conn.fetch(
                    """
                    SELECT protocol, COUNT(*) as count, AVG(confidence) as avg_confidence
                    FROM protocol_analysis 
                    WHERE timestamp BETWEEN $1 AND $2
                    GROUP BY protocol
                    ORDER BY count DESC
                """,
                    start_time,
                    end_time,
                )

                # Get average processing time
                avg_processing_time = await conn.fetchval(
                    """
                    SELECT AVG(processing_time_ms)
                    FROM protocol_analysis 
                    WHERE timestamp BETWEEN $1 AND $2
                """,
                    start_time,
                    end_time,
                )

                self.query_count += 1

                return {
                    "protocol_distribution": [dict(row) for row in protocol_stats],
                    "average_processing_time_ms": float(avg_processing_time or 0),
                    "time_range": {
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat(),
                    },
                }

        except Exception as e:
            logger.error(f"Error getting protocol analysis stats: {e}")
            return {}

    async def get_system_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get system metrics summary"""
        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours)

            async with self.pool.acquire() as conn:
                # Get total records by table
                metrics_count = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM metrics 
                    WHERE timestamp BETWEEN $1 AND $2
                """,
                    start_time,
                    end_time,
                )

                logs_count = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM logs 
                    WHERE timestamp BETWEEN $1 AND $2
                """,
                    start_time,
                    end_time,
                )

                security_count = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM security_events 
                    WHERE timestamp BETWEEN $1 AND $2
                """,
                    start_time,
                    end_time,
                )

                protocol_count = await conn.fetchval(
                    """
                    SELECT COUNT(*) FROM protocol_analysis 
                    WHERE timestamp BETWEEN $1 AND $2
                """,
                    start_time,
                    end_time,
                )

                self.query_count += 1

                return {
                    "time_range_hours": hours,
                    "record_counts": {
                        "metrics": metrics_count,
                        "logs": logs_count,
                        "security_events": security_count,
                        "protocol_analysis": protocol_count,
                        "total": metrics_count
                        + logs_count
                        + security_count
                        + protocol_count,
                    },
                    "integration_stats": {
                        "records_inserted": self.records_inserted,
                        "query_count": self.query_count,
                        "error_rate": self.insert_errors
                        / max(self.batch_insert_count, 1),
                    },
                }

        except Exception as e:
            logger.error(f"Error getting system metrics summary: {e}")
            return {}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            "records_inserted": self.records_inserted,
            "batch_insert_count": self.batch_insert_count,
            "insert_errors": self.insert_errors,
            "query_count": self.query_count,
            "buffer_sizes": {
                "metrics": len(self.metrics_buffer),
                "logs": len(self.logs_buffer),
                "security": len(self.security_buffer),
                "protocol": len(self.protocol_buffer),
            },
            "connection_pool_size": self.pool.get_size() if self.pool else 0,
        }

    async def shutdown(self):
        """Shutdown TimescaleDB integration"""
        logger.info("Shutting down TimescaleDB Integration...")

        self.running = False

        # Flush remaining buffers
        await self._flush_all_buffers()

        # Close database pool
        if self.pool:
            await self.pool.close()

        # Shutdown thread pool
        self.db_executor.shutdown(wait=True)

        logger.info("TimescaleDB Integration shutdown complete")


# Global integration instance
_timescaledb_integration = None


def get_timescaledb_integration() -> TimescaleDBIntegration:
    """Get global TimescaleDB integration instance"""
    global _timescaledb_integration
    if _timescaledb_integration is None:
        _timescaledb_integration = TimescaleDBIntegration()
    return _timescaledb_integration


async def main():
    """Main entry point for TimescaleDB integration"""
    integration = TimescaleDBIntegration()

    try:
        await integration.initialize()

        # Keep running
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"TimescaleDB integration error: {e}")
    finally:
        await integration.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
