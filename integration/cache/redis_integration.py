#!/usr/bin/env python3
"""
CRONOS AI - Redis Integration
High-performance caching and session management using Redis.
"""

import asyncio
import logging
import json
import time
import pickle
import hashlib
from typing import Dict, Any, Optional, List, Union, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import aioredis
from aioredis import Redis
from concurrent.futures import ThreadPoolExecutor
import msgpack
from pathlib import Path
import sys
import uuid

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from integration.config.unified_config import get_config, get_service_config
from integration.orchestrator.service_integration import get_orchestrator, Message

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry structure"""
    key: str
    value: Any
    ttl: Optional[int] = None
    created_at: float = None
    accessed_count: int = 0
    last_accessed: float = None

@dataclass
class SessionData:
    """Session data structure"""
    session_id: str
    user_id: Optional[str] = None
    device_id: Optional[str] = None
    created_at: float = None
    last_activity: float = None
    data: Dict[str, Any] = None
    expires_at: Optional[float] = None

@dataclass
class RateLimitInfo:
    """Rate limiting information"""
    key: str
    current_count: int
    limit: int
    window_seconds: int
    reset_time: float

class SerializationMethod:
    """Serialization methods for different data types"""
    JSON = "json"
    PICKLE = "pickle"
    MSGPACK = "msgpack"
    RAW = "raw"

class RedisIntegration:
    """
    Production Redis integration for CRONOS AI.
    Provides caching, session management, rate limiting, and pub/sub messaging.
    """
    
    def __init__(self):
        self.config = get_service_config('redis')
        self.orchestrator = get_orchestrator()
        
        # Redis connections
        self.redis_client: Optional[Redis] = None
        self.pub_sub_client: Optional[Redis] = None
        
        # Connection configuration
        self.redis_config = {
            'host': self.config.get('host', 'localhost'),
            'port': self.config.get('port', 6379),
            'db': self.config.get('database', 0),
            'password': self.config.get('password') if self.config.get('password') else None,
            'ssl': self.config.get('ssl', False),
            'retry_on_timeout': True,
            'socket_connect_timeout': self.config.get('connection_timeout', 5),
            'socket_keepalive': True,
            'socket_keepalive_options': {},
            'health_check_interval': 30,
        }
        
        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_sets = 0
        self.cache_deletes = 0
        self.pub_sub_messages = 0
        
        # Key prefixes for different use cases
        self.prefixes = {
            'cache': 'cronos:cache:',
            'session': 'cronos:session:',
            'rate_limit': 'cronos:rate_limit:',
            'lock': 'cronos:lock:',
            'queue': 'cronos:queue:',
            'pub_sub': 'cronos:pubsub:',
            'metrics': 'cronos:metrics:',
        }
        
        # Default TTLs (in seconds)
        self.default_ttls = {
            'cache': 3600,  # 1 hour
            'session': 86400,  # 24 hours
            'rate_limit': 3600,  # 1 hour
            'lock': 300,  # 5 minutes
            'metrics': 1800,  # 30 minutes
        }
        
        # Serialization preferences
        self.serialization_method = SerializationMethod.MSGPACK
        
        # Running state
        self.running = False
        
        # Thread pool for blocking operations
        self.executor = ThreadPoolExecutor(
            max_workers=4, 
            thread_name_prefix="redis_"
        )
        
        # Pub/sub subscriptions
        self.subscriptions: Dict[str, List[callable]] = {}
        
        # Connection pool settings
        self.max_connections = self.config.get('connection_pool_size', 20)
        
    async def initialize(self):
        """Initialize Redis integration"""
        logger.info("Initializing Redis Integration...")
        
        try:
            # Create main Redis client
            await self._create_redis_client()
            
            # Create pub/sub client
            await self._create_pubsub_client()
            
            # Test connections
            await self._test_connections()
            
            # Set up key expiration notifications if needed
            await self._setup_key_notifications()
            
            # Start background tasks
            await self._start_background_tasks()
            
            logger.info("Redis Integration initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis integration: {e}")
            raise
    
    async def _create_redis_client(self):
        """Create main Redis client"""
        try:
            self.redis_client = await aioredis.from_url(
                self._build_redis_url(),
                max_connections=self.max_connections,
                retry_on_timeout=self.redis_config['retry_on_timeout'],
                socket_connect_timeout=self.redis_config['socket_connect_timeout'],
                socket_keepalive=self.redis_config['socket_keepalive'],
                health_check_interval=self.redis_config['health_check_interval'],
            )
            
            logger.info(f"Created Redis client: {self.redis_config['host']}:{self.redis_config['port']}")
            
        except Exception as e:
            logger.error(f"Failed to create Redis client: {e}")
            raise
    
    async def _create_pubsub_client(self):
        """Create separate Redis client for pub/sub"""
        try:
            self.pub_sub_client = await aioredis.from_url(
                self._build_redis_url(),
                max_connections=5,  # Fewer connections for pub/sub
                retry_on_timeout=True,
                socket_connect_timeout=self.redis_config['socket_connect_timeout'],
            )
            
            logger.info("Created Redis pub/sub client")
            
        except Exception as e:
            logger.error(f"Failed to create Redis pub/sub client: {e}")
            raise
    
    def _build_redis_url(self) -> str:
        """Build Redis connection URL"""
        protocol = "rediss" if self.redis_config['ssl'] else "redis"
        host = self.redis_config['host']
        port = self.redis_config['port']
        db = self.redis_config['db']
        password = self.redis_config['password']
        
        if password:
            return f"{protocol}://:{password}@{host}:{port}/{db}"
        else:
            return f"{protocol}://{host}:{port}/{db}"
    
    async def _test_connections(self):
        """Test Redis connections"""
        try:
            # Test main client
            await self.redis_client.ping()
            logger.info("Main Redis connection test successful")
            
            # Test pub/sub client
            await self.pub_sub_client.ping()
            logger.info("Pub/sub Redis connection test successful")
            
            # Get Redis info
            info = await self.redis_client.info()
            logger.info(f"Redis version: {info.get('redis_version')}")
            logger.info(f"Redis memory usage: {info.get('used_memory_human')}")
            
        except Exception as e:
            logger.error(f"Redis connection test failed: {e}")
            raise
    
    async def _setup_key_notifications(self):
        """Set up Redis key expiration notifications"""
        try:
            # Enable keyspace notifications for expired events
            await self.redis_client.config_set('notify-keyspace-events', 'Ex')
            logger.info("Redis key expiration notifications enabled")
            
        except Exception as e:
            logger.warning(f"Could not enable key notifications: {e}")
    
    async def _start_background_tasks(self):
        """Start background processing tasks"""
        self.running = True
        
        # Start metrics collector
        asyncio.create_task(self._metrics_collector())
        
        # Start connection health monitor
        asyncio.create_task(self._connection_monitor())
        
        # Start pub/sub listener
        asyncio.create_task(self._pubsub_listener())
        
        # Start cache cleanup task
        asyncio.create_task(self._cache_cleanup_task())
        
        logger.info("Redis background tasks started")
    
    # Caching methods
    async def set_cache(self, key: str, value: Any, ttl: Optional[int] = None, 
                       serialization: Optional[str] = None) -> bool:
        """Set cache value"""
        try:
            full_key = self.prefixes['cache'] + key
            serialized_value = await self._serialize_value(value, serialization)
            
            if ttl is None:
                ttl = self.default_ttls['cache']
            
            result = await self.redis_client.setex(full_key, ttl, serialized_value)
            
            if result:
                self.cache_sets += 1
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    async def get_cache(self, key: str, serialization: Optional[str] = None) -> Optional[Any]:
        """Get cache value"""
        try:
            full_key = self.prefixes['cache'] + key
            serialized_value = await self.redis_client.get(full_key)
            
            if serialized_value is None:
                self.cache_misses += 1
                return None
            
            self.cache_hits += 1
            return await self._deserialize_value(serialized_value, serialization)
            
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return None
    
    async def delete_cache(self, key: str) -> bool:
        """Delete cache value"""
        try:
            full_key = self.prefixes['cache'] + key
            result = await self.redis_client.delete(full_key)
            
            if result > 0:
                self.cache_deletes += 1
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False
    
    async def exists_cache(self, key: str) -> bool:
        """Check if cache key exists"""
        try:
            full_key = self.prefixes['cache'] + key
            return bool(await self.redis_client.exists(full_key))
            
        except Exception as e:
            logger.error(f"Error checking cache key existence {key}: {e}")
            return False
    
    async def set_cache_many(self, data: Dict[str, Any], ttl: Optional[int] = None) -> int:
        """Set multiple cache values"""
        try:
            pipeline = self.redis_client.pipeline()
            
            if ttl is None:
                ttl = self.default_ttls['cache']
            
            for key, value in data.items():
                full_key = self.prefixes['cache'] + key
                serialized_value = await self._serialize_value(value)
                pipeline.setex(full_key, ttl, serialized_value)
            
            results = await pipeline.execute()
            success_count = sum(1 for result in results if result)
            self.cache_sets += success_count
            
            return success_count
            
        except Exception as e:
            logger.error(f"Error setting multiple cache keys: {e}")
            return 0
    
    async def get_cache_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple cache values"""
        try:
            full_keys = [self.prefixes['cache'] + key for key in keys]
            values = await self.redis_client.mget(full_keys)
            
            result = {}
            for i, (key, value) in enumerate(zip(keys, values)):
                if value is not None:
                    self.cache_hits += 1
                    result[key] = await self._deserialize_value(value)
                else:
                    self.cache_misses += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting multiple cache keys: {e}")
            return {}
    
    # Session management methods
    async def create_session(self, session_id: Optional[str] = None, 
                           user_id: Optional[str] = None, 
                           device_id: Optional[str] = None,
                           ttl: Optional[int] = None) -> str:
        """Create user session"""
        try:
            if session_id is None:
                session_id = str(uuid.uuid4())
            
            if ttl is None:
                ttl = self.default_ttls['session']
            
            session_data = SessionData(
                session_id=session_id,
                user_id=user_id,
                device_id=device_id,
                created_at=time.time(),
                last_activity=time.time(),
                data={},
                expires_at=time.time() + ttl
            )
            
            full_key = self.prefixes['session'] + session_id
            serialized_data = await self._serialize_value(asdict(session_data))
            
            await self.redis_client.setex(full_key, ttl, serialized_data)
            
            logger.info(f"Created session: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return None
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session data"""
        try:
            full_key = self.prefixes['session'] + session_id
            data = await self.redis_client.get(full_key)
            
            if data is None:
                return None
            
            session_dict = await self._deserialize_value(data)
            return SessionData(**session_dict)
            
        except Exception as e:
            logger.error(f"Error getting session {session_id}: {e}")
            return None
    
    async def update_session(self, session_id: str, data: Dict[str, Any], 
                           extend_ttl: bool = True) -> bool:
        """Update session data"""
        try:
            session = await self.get_session(session_id)
            if session is None:
                return False
            
            # Update session data
            session.data.update(data)
            session.last_activity = time.time()
            
            full_key = self.prefixes['session'] + session_id
            serialized_data = await self._serialize_value(asdict(session))
            
            if extend_ttl:
                ttl = self.default_ttls['session']
                session.expires_at = time.time() + ttl
                await self.redis_client.setex(full_key, ttl, serialized_data)
            else:
                await self.redis_client.set(full_key, serialized_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating session {session_id}: {e}")
            return False
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session"""
        try:
            full_key = self.prefixes['session'] + session_id
            result = await self.redis_client.delete(full_key)
            
            if result > 0:
                logger.info(f"Deleted session: {session_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False
    
    # Rate limiting methods
    async def check_rate_limit(self, key: str, limit: int, window_seconds: int) -> RateLimitInfo:
        """Check and update rate limit"""
        try:
            full_key = self.prefixes['rate_limit'] + key
            current_time = time.time()
            window_start = current_time - window_seconds
            
            # Use sliding window rate limiting with Redis sorted sets
            pipeline = self.redis_client.pipeline()
            
            # Remove old entries
            pipeline.zremrangebyscore(full_key, 0, window_start)
            
            # Count current requests
            pipeline.zcard(full_key)
            
            # Add current request
            pipeline.zadd(full_key, {str(uuid.uuid4()): current_time})
            
            # Set TTL
            pipeline.expire(full_key, window_seconds + 1)
            
            results = await pipeline.execute()
            current_count = results[1]
            
            # Check if limit exceeded
            if current_count >= limit:
                # Remove the request we just added since it's over the limit
                await self.redis_client.zpopmax(full_key)
                is_allowed = False
            else:
                is_allowed = True
            
            # Calculate reset time
            reset_time = current_time + window_seconds
            
            return RateLimitInfo(
                key=key,
                current_count=current_count,
                limit=limit,
                window_seconds=window_seconds,
                reset_time=reset_time
            )
            
        except Exception as e:
            logger.error(f"Error checking rate limit for {key}: {e}")
            # Return permissive result on error
            return RateLimitInfo(
                key=key,
                current_count=0,
                limit=limit,
                window_seconds=window_seconds,
                reset_time=time.time() + window_seconds
            )
    
    # Distributed locking methods
    async def acquire_lock(self, lock_name: str, timeout: int = 10, 
                          ttl: Optional[int] = None) -> Optional[str]:
        """Acquire distributed lock"""
        try:
            if ttl is None:
                ttl = self.default_ttls['lock']
            
            lock_key = self.prefixes['lock'] + lock_name
            lock_value = str(uuid.uuid4())
            
            end_time = time.time() + timeout
            
            while time.time() < end_time:
                # Try to acquire lock
                acquired = await self.redis_client.set(
                    lock_key, lock_value, 
                    nx=True, ex=ttl
                )
                
                if acquired:
                    return lock_value
                
                await asyncio.sleep(0.1)  # Wait 100ms before retrying
            
            return None
            
        except Exception as e:
            logger.error(f"Error acquiring lock {lock_name}: {e}")
            return None
    
    async def release_lock(self, lock_name: str, lock_value: str) -> bool:
        """Release distributed lock"""
        try:
            lock_key = self.prefixes['lock'] + lock_name
            
            # Lua script to ensure we only release our own lock
            lua_script = """
                if redis.call('get', KEYS[1]) == ARGV[1] then
                    return redis.call('del', KEYS[1])
                else
                    return 0
                end
            """
            
            result = await self.redis_client.eval(lua_script, 1, lock_key, lock_value)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error releasing lock {lock_name}: {e}")
            return False
    
    # Pub/Sub methods
    async def publish(self, channel: str, message: Any) -> int:
        """Publish message to channel"""
        try:
            full_channel = self.prefixes['pub_sub'] + channel
            serialized_message = await self._serialize_value(message)
            
            subscriber_count = await self.redis_client.publish(full_channel, serialized_message)
            self.pub_sub_messages += 1
            
            return subscriber_count
            
        except Exception as e:
            logger.error(f"Error publishing to channel {channel}: {e}")
            return 0
    
    async def subscribe(self, channel: str, handler: callable):
        """Subscribe to channel"""
        try:
            full_channel = self.prefixes['pub_sub'] + channel
            
            if channel not in self.subscriptions:
                self.subscriptions[channel] = []
            
            self.subscriptions[channel].append(handler)
            logger.info(f"Subscribed to channel: {channel}")
            
        except Exception as e:
            logger.error(f"Error subscribing to channel {channel}: {e}")
    
    # Queue methods
    async def enqueue(self, queue_name: str, item: Any, priority: int = 0) -> bool:
        """Add item to queue"""
        try:
            full_queue = self.prefixes['queue'] + queue_name
            serialized_item = await self._serialize_value(item)
            
            # Use sorted set for priority queue
            await self.redis_client.zadd(full_queue, {serialized_item: priority})
            return True
            
        except Exception as e:
            logger.error(f"Error enqueuing to {queue_name}: {e}")
            return False
    
    async def dequeue(self, queue_name: str, count: int = 1) -> List[Any]:
        """Remove items from queue"""
        try:
            full_queue = self.prefixes['queue'] + queue_name
            
            # Get highest priority items
            items = await self.redis_client.zpopmax(full_queue, count)
            
            result = []
            for item_data, priority in items:
                deserialized_item = await self._deserialize_value(item_data)
                result.append(deserialized_item)
            
            return result
            
        except Exception as e:
            logger.error(f"Error dequeuing from {queue_name}: {e}")
            return []
    
    # Utility methods
    async def _serialize_value(self, value: Any, method: Optional[str] = None) -> bytes:
        """Serialize value for storage"""
        try:
            if method is None:
                method = self.serialization_method
            
            if method == SerializationMethod.JSON:
                return json.dumps(value, default=str).encode('utf-8')
            elif method == SerializationMethod.PICKLE:
                return pickle.dumps(value)
            elif method == SerializationMethod.MSGPACK:
                return msgpack.packb(value, use_bin_type=True)
            elif method == SerializationMethod.RAW:
                if isinstance(value, str):
                    return value.encode('utf-8')
                elif isinstance(value, bytes):
                    return value
                else:
                    return str(value).encode('utf-8')
            else:
                # Default to msgpack
                return msgpack.packb(value, use_bin_type=True)
                
        except Exception as e:
            logger.error(f"Error serializing value: {e}")
            return json.dumps({'error': 'serialization_failed'}).encode('utf-8')
    
    async def _deserialize_value(self, data: bytes, method: Optional[str] = None) -> Any:
        """Deserialize value from storage"""
        try:
            if method is None:
                method = self.serialization_method
            
            if method == SerializationMethod.JSON:
                return json.loads(data.decode('utf-8'))
            elif method == SerializationMethod.PICKLE:
                return pickle.loads(data)
            elif method == SerializationMethod.MSGPACK:
                return msgpack.unpackb(data, raw=False)
            elif method == SerializationMethod.RAW:
                return data.decode('utf-8')
            else:
                # Default to msgpack
                return msgpack.unpackb(data, raw=False)
                
        except Exception as e:
            logger.error(f"Error deserializing value: {e}")
            return None
    
    # Background tasks
    async def _metrics_collector(self):
        """Collect Redis performance metrics"""
        while self.running:
            try:
                await asyncio.sleep(60)  # Collect every minute
                
                # Get Redis info
                info = await self.redis_client.info()
                
                # Calculate cache hit rate
                total_requests = self.cache_hits + self.cache_misses
                hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
                
                metrics = {
                    'cache_hits': self.cache_hits,
                    'cache_misses': self.cache_misses,
                    'cache_hit_rate': hit_rate,
                    'cache_sets': self.cache_sets,
                    'cache_deletes': self.cache_deletes,
                    'pub_sub_messages': self.pub_sub_messages,
                    'redis_memory_usage': info.get('used_memory'),
                    'redis_connected_clients': info.get('connected_clients'),
                    'redis_ops_per_sec': info.get('instantaneous_ops_per_sec'),
                }
                
                # Send metrics to orchestrator
                message = Message(
                    id=f"redis_metrics_{time.time()}",
                    timestamp=time.time(),
                    source="redis_integration",
                    destination="orchestrator",
                    message_type="metric_update",
                    payload={
                        'component': 'redis_integration',
                        'metrics': metrics,
                    }
                )
                
                await self.orchestrator.send_message(message)
                
            except Exception as e:
                logger.error(f"Error collecting Redis metrics: {e}")
                await asyncio.sleep(60)
    
    async def _connection_monitor(self):
        """Monitor Redis connection health"""
        while self.running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Test connections
                await self.redis_client.ping()
                await self.pub_sub_client.ping()
                
                health_status = "healthy"
                
                # Send health status
                message = Message(
                    id=f"redis_health_{time.time()}",
                    timestamp=time.time(),
                    source="redis_integration",
                    destination="orchestrator",
                    message_type="health_check",
                    payload={
                        'component': 'redis_integration',
                        'status': health_status,
                    }
                )
                
                await self.orchestrator.send_message(message)
                
            except Exception as e:
                logger.error(f"Redis connection health check failed: {e}")
                await asyncio.sleep(30)
    
    async def _pubsub_listener(self):
        """Listen for pub/sub messages"""
        while self.running:
            try:
                if not self.subscriptions:
                    await asyncio.sleep(1)
                    continue
                
                # Subscribe to all channels
                pubsub = self.pub_sub_client.pubsub()
                
                for channel in self.subscriptions.keys():
                    full_channel = self.prefixes['pub_sub'] + channel
                    await pubsub.subscribe(full_channel)
                
                # Listen for messages
                async for message in pubsub.listen():
                    if message['type'] == 'message':
                        channel_name = message['channel'].decode().replace(self.prefixes['pub_sub'], '')
                        data = await self._deserialize_value(message['data'])
                        
                        # Call handlers
                        handlers = self.subscriptions.get(channel_name, [])
                        for handler in handlers:
                            try:
                                await handler(channel_name, data)
                            except Exception as e:
                                logger.error(f"Error in pub/sub handler: {e}")
                
            except Exception as e:
                logger.error(f"Error in pub/sub listener: {e}")
                await asyncio.sleep(5)
    
    async def _cache_cleanup_task(self):
        """Clean up expired cache entries"""
        while self.running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # This is handled automatically by Redis TTL,
                # but we can add custom cleanup logic here if needed
                
            except Exception as e:
                logger.error(f"Error in cache cleanup task: {e}")
                await asyncio.sleep(300)
    
    # Public API
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        total_requests = self.cache_hits + self.cache_misses
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / total_requests if total_requests > 0 else 0,
            'cache_sets': self.cache_sets,
            'cache_deletes': self.cache_deletes,
            'pub_sub_messages': self.pub_sub_messages,
            'active_subscriptions': len(self.subscriptions),
        }
    
    async def flush_all_caches(self) -> bool:
        """Flush all cache entries (use with caution)"""
        try:
            # Only flush cache keys, not sessions or other data
            cache_pattern = self.prefixes['cache'] + '*'
            keys = []
            
            async for key in self.redis_client.scan_iter(match=cache_pattern):
                keys.append(key)
            
            if keys:
                await self.redis_client.delete(*keys)
                logger.info(f"Flushed {len(keys)} cache entries")
            
            return True
            
        except Exception as e:
            logger.error(f"Error flushing caches: {e}")
            return False
    
    async def get_redis_info(self) -> Dict[str, Any]:
        """Get Redis server information"""
        try:
            return await self.redis_client.info()
        except Exception as e:
            logger.error(f"Error getting Redis info: {e}")
            return {}
    
    async def shutdown(self):
        """Shutdown Redis integration"""
        logger.info("Shutting down Redis Integration...")
        
        self.running = False
        
        # Close Redis connections
        if self.redis_client:
            await self.redis_client.close()
        
        if self.pub_sub_client:
            await self.pub_sub_client.close()
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        logger.info("Redis Integration shutdown complete")

# Global integration instance
_redis_integration = None

def get_redis_integration() -> RedisIntegration:
    """Get global Redis integration instance"""
    global _redis_integration
    if _redis_integration is None:
        _redis_integration = RedisIntegration()
    return _redis_integration

async def main():
    """Main entry point for Redis integration"""
    integration = RedisIntegration()
    
    try:
        await integration.initialize()
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Redis integration error: {e}")
    finally:
        await integration.shutdown()

if __name__ == "__main__":
    asyncio.run(main())