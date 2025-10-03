"""
CRONOS AI - Log Aggregation System
Production-ready log aggregation with ELK Stack and Grafana Loki integration.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import aiohttp
from urllib.parse import urljoin
import gzip
from collections import deque

from ..core.config import Config
from ..core.exceptions import ObservabilityException

logger = logging.getLogger(__name__)


class LogLevel(str, Enum):
    """Log levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class StructuredLog:
    """Structured log entry."""
    timestamp: datetime
    level: LogLevel
    message: str
    logger_name: str
    service: str
    environment: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    fields: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
            "logger": self.logger_name,
            "service": self.service,
            "environment": self.environment,
        }
        
        if self.trace_id:
            data["trace_id"] = self.trace_id
        if self.span_id:
            data["span_id"] = self.span_id
        if self.user_id:
            data["user_id"] = self.user_id
        if self.request_id:
            data["request_id"] = self.request_id
        if self.fields:
            data["fields"] = self.fields
        if self.tags:
            data["tags"] = self.tags
        
        return data


class ElasticsearchLogAggregator:
    """
    Elasticsearch log aggregator for ELK stack integration.
    
    Sends structured logs to Elasticsearch for indexing and analysis.
    """
    
    def __init__(self, config: Config):
        """Initialize Elasticsearch log aggregator."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Elasticsearch configuration
        self.es_hosts = getattr(config, 'elasticsearch_hosts', ['http://localhost:9200'])
        self.es_index_prefix = getattr(config, 'elasticsearch_index_prefix', 'cronos-ai-logs')
        self.es_username = getattr(config, 'elasticsearch_username', None)
        self.es_password = getattr(config, 'elasticsearch_password', None)
        
        # Batching configuration
        self._batch_queue: deque = deque(maxlen=10000)
        self._batch_size = 500
        self._flush_interval = 5  # seconds
        self._session: Optional[aiohttp.ClientSession] = None
        self._flush_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.logs_sent = 0
        self.logs_failed = 0
        
        self.logger.info(f"ElasticsearchLogAggregator initialized (hosts: {self.es_hosts})")
    
    async def initialize(self):
        """Initialize HTTP session and background tasks."""
        auth = None
        if self.es_username and self.es_password:
            auth = aiohttp.BasicAuth(self.es_username, self.es_password)
        
        self._session = aiohttp.ClientSession(auth=auth)
        self._flush_task = asyncio.create_task(self._flush_loop())
        
        # Create index template
        await self._create_index_template()
        
        self.logger.info("Elasticsearch log aggregator initialized")
    
    async def shutdown(self):
        """Shutdown aggregator and flush pending logs."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        await self.flush()
        
        if self._session:
            await self._session.close()
    
    async def send_log(self, log: StructuredLog):
        """Send log entry to Elasticsearch."""
        try:
            self._batch_queue.append(log)
            
            if len(self._batch_queue) >= self._batch_size:
                await self._flush_batch()
                
        except Exception as e:
            self.logger.error(f"Failed to queue log: {e}")
            self.logs_failed += 1
    
    async def flush(self):
        """Flush pending logs."""
        if self._batch_queue:
            await self._flush_batch()
    
    async def _flush_loop(self):
        """Background task to flush logs periodically."""
        while True:
            try:
                await asyncio.sleep(self._flush_interval)
                if self._batch_queue:
                    await self._flush_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Flush loop error: {e}")
    
    async def _flush_batch(self):
        """Flush batch of logs to Elasticsearch using bulk API."""
        if not self._batch_queue or not self._session:
            return
        
        try:
            # Get batch
            batch_size = min(len(self._batch_queue), self._batch_size)
            batch = [self._batch_queue.popleft() for _ in range(batch_size)]
            
            # Build bulk request
            bulk_data = []
            index_name = f"{self.es_index_prefix}-{datetime.utcnow().strftime('%Y.%m.%d')}"
            
            for log in batch:
                # Index action
                bulk_data.append(json.dumps({
                    "index": {
                        "_index": index_name,
                        "_type": "_doc"
                    }
                }))
                # Document
                bulk_data.append(json.dumps(log.to_dict()))
            
            bulk_body = "\n".join(bulk_data) + "\n"
            
            # Send to Elasticsearch
            for host in self.es_hosts:
                try:
                    endpoint = urljoin(host, "/_bulk")
                    
                    async with self._session.post(
                        endpoint,
                        data=bulk_body,
                        headers={"Content-Type": "application/x-ndjson"}
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            if not result.get('errors', False):
                                self.logs_sent += len(batch)
                                self.logger.debug(f"Flushed {len(batch)} logs to Elasticsearch")
                                return
                            else:
                                self.logger.error(f"Elasticsearch bulk errors: {result}")
                        else:
                            self.logger.error(f"Elasticsearch returned status {response.status}")
                            
                except Exception as e:
                    self.logger.error(f"Failed to send to {host}: {e}")
                    continue
            
            # If all hosts failed, increment failure count
            self.logs_failed += len(batch)
            
        except Exception as e:
            self.logger.error(f"Failed to flush batch: {e}")
            self.logs_failed += len(batch) if batch else 0
    
    async def _create_index_template(self):
        """Create Elasticsearch index template for logs."""
        template = {
            "index_patterns": [f"{self.es_index_prefix}-*"],
            "settings": {
                "number_of_shards": 3,
                "number_of_replicas": 1,
                "index.lifecycle.name": "cronos-ai-logs-policy",
                "index.lifecycle.rollover_alias": f"{self.es_index_prefix}"
            },
            "mappings": {
                "properties": {
                    "timestamp": {"type": "date"},
                    "level": {"type": "keyword"},
                    "message": {"type": "text"},
                    "logger": {"type": "keyword"},
                    "service": {"type": "keyword"},
                    "environment": {"type": "keyword"},
                    "trace_id": {"type": "keyword"},
                    "span_id": {"type": "keyword"},
                    "user_id": {"type": "keyword"},
                    "request_id": {"type": "keyword"},
                    "tags": {"type": "keyword"},
                    "fields": {"type": "object", "enabled": True}
                }
            }
        }
        
        try:
            for host in self.es_hosts:
                endpoint = urljoin(host, f"/_index_template/{self.es_index_prefix}-template")
                
                async with self._session.put(
                    endpoint,
                    json=template
                ) as response:
                    if response.status in (200, 201):
                        self.logger.info("Elasticsearch index template created")
                        return
                        
        except Exception as e:
            self.logger.warning(f"Failed to create index template: {e}")


class LokiLogAggregator:
    """
    Grafana Loki log aggregator.
    
    Sends structured logs to Loki for efficient log storage and querying.
    """
    
    def __init__(self, config: Config):
        """Initialize Loki log aggregator."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Loki configuration
        self.loki_url = getattr(config, 'loki_url', 'http://localhost:3100')
        self.loki_username = getattr(config, 'loki_username', None)
        self.loki_password = getattr(config, 'loki_password', None)
        
        # Batching configuration
        self._batch_queue: deque = deque(maxlen=10000)
        self._batch_size = 100
        self._flush_interval = 5  # seconds
        self._session: Optional[aiohttp.ClientSession] = None
        self._flush_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.logs_sent = 0
        self.logs_failed = 0
        
        self.logger.info(f"LokiLogAggregator initialized (url: {self.loki_url})")
    
    async def initialize(self):
        """Initialize HTTP session and background tasks."""
        auth = None
        if self.loki_username and self.loki_password:
            auth = aiohttp.BasicAuth(self.loki_username, self.loki_password)
        
        self._session = aiohttp.ClientSession(auth=auth)
        self._flush_task = asyncio.create_task(self._flush_loop())
        
        self.logger.info("Loki log aggregator initialized")
    
    async def shutdown(self):
        """Shutdown aggregator and flush pending logs."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        await self.flush()
        
        if self._session:
            await self._session.close()
    
    async def send_log(self, log: StructuredLog):
        """Send log entry to Loki."""
        try:
            self._batch_queue.append(log)
            
            if len(self._batch_queue) >= self._batch_size:
                await self._flush_batch()
                
        except Exception as e:
            self.logger.error(f"Failed to queue log: {e}")
            self.logs_failed += 1
    
    async def flush(self):
        """Flush pending logs."""
        if self._batch_queue:
            await self._flush_batch()
    
    async def _flush_loop(self):
        """Background task to flush logs periodically."""
        while True:
            try:
                await asyncio.sleep(self._flush_interval)
                if self._batch_queue:
                    await self._flush_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Flush loop error: {e}")
    
    async def _flush_batch(self):
        """Flush batch of logs to Loki."""
        if not self._batch_queue or not self._session:
            return
        
        try:
            # Get batch
            batch_size = min(len(self._batch_queue), self._batch_size)
            batch = [self._batch_queue.popleft() for _ in range(batch_size)]
            
            # Group logs by labels
            streams = {}
            for log in batch:
                labels = self._build_labels(log)
                label_key = json.dumps(labels, sort_keys=True)
                
                if label_key not in streams:
                    streams[label_key] = {
                        "stream": labels,
                        "values": []
                    }
                
                # Loki expects [timestamp_ns, log_line]
                timestamp_ns = str(int(log.timestamp.timestamp() * 1e9))
                log_line = json.dumps({
                    "message": log.message,
                    "fields": log.fields
                })
                
                streams[label_key]["values"].append([timestamp_ns, log_line])
            
            # Build Loki push request
            loki_data = {
                "streams": list(streams.values())
            }
            
            # Send to Loki
            endpoint = urljoin(self.loki_url, "/loki/api/v1/push")
            
            async with self._session.post(
                endpoint,
                json=loki_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 204:
                    self.logs_sent += len(batch)
                    self.logger.debug(f"Flushed {len(batch)} logs to Loki")
                else:
                    self.logger.error(f"Loki returned status {response.status}")
                    self.logs_failed += len(batch)
                    
        except Exception as e:
            self.logger.error(f"Failed to flush batch to Loki: {e}")
            self.logs_failed += len(batch) if batch else 0
    
    def _build_labels(self, log: StructuredLog) -> Dict[str, str]:
        """Build Loki labels from log entry."""
        labels = {
            "level": log.level.value,
            "logger": log.logger_name,
            "service": log.service,
            "environment": log.environment
        }
        
        if log.trace_id:
            labels["trace_id"] = log.trace_id
        if log.user_id:
            labels["user_id"] = log.user_id
        
        # Add tags as labels
        for tag in log.tags:
            labels[f"tag_{tag}"] = "true"
        
        return labels


class LogAggregationManager:
    """
    Central log aggregation manager.
    
    Coordinates multiple log aggregators and provides unified interface.
    """
    
    def __init__(self, config: Config):
        """Initialize log aggregation manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Service information
        self.service_name = getattr(config, 'service_name', 'cronos-ai')
        self.environment = getattr(config.environment, 'value', 'development')
        
        # Aggregators
        self.aggregators: List[Union[ElasticsearchLogAggregator, LokiLogAggregator]] = []
        
        # Initialize aggregators based on configuration
        if getattr(config, 'enable_elasticsearch_logging', False):
            self.aggregators.append(ElasticsearchLogAggregator(config))
        
        if getattr(config, 'enable_loki_logging', False):
            self.aggregators.append(LokiLogAggregator(config))
        
        self.logger.info(f"LogAggregationManager initialized with {len(self.aggregators)} aggregators")
    
    async def initialize(self):
        """Initialize all aggregators."""
        for aggregator in self.aggregators:
            await aggregator.initialize()
        
        self.logger.info("Log aggregation manager initialized")
    
    async def shutdown(self):
        """Shutdown all aggregators."""
        for aggregator in self.aggregators:
            await aggregator.shutdown()
        
        self.logger.info("Log aggregation manager shut down")
    
    async def send_log(
        self,
        level: LogLevel,
        message: str,
        logger_name: str,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        fields: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        """Send log to all configured aggregators."""
        log = StructuredLog(
            timestamp=datetime.utcnow(),
            level=level,
            message=message,
            logger_name=logger_name,
            service=self.service_name,
            environment=self.environment,
            trace_id=trace_id,
            span_id=span_id,
            user_id=user_id,
            request_id=request_id,
            fields=fields or {},
            tags=tags or []
        )
        
        # Send to all aggregators
        tasks = [aggregator.send_log(log) for aggregator in self.aggregators]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregation statistics."""
        stats = {
            "aggregators": [],
            "total_logs_sent": 0,
            "total_logs_failed": 0
        }
        
        for aggregator in self.aggregators:
            aggregator_stats = {
                "type": type(aggregator).__name__,
                "logs_sent": aggregator.logs_sent,
                "logs_failed": aggregator.logs_failed
            }
            stats["aggregators"].append(aggregator_stats)
            stats["total_logs_sent"] += aggregator.logs_sent
            stats["total_logs_failed"] += aggregator.logs_failed
        
        return stats


# Global log aggregation manager
_log_aggregation_manager: Optional[LogAggregationManager] = None


async def initialize_log_aggregation(config: Config) -> LogAggregationManager:
    """Initialize global log aggregation manager."""
    global _log_aggregation_manager
    
    _log_aggregation_manager = LogAggregationManager(config)
    await _log_aggregation_manager.initialize()
    
    return _log_aggregation_manager


def get_log_aggregation_manager() -> Optional[LogAggregationManager]:
    """Get global log aggregation manager."""
    return _log_aggregation_manager