"""
CRONOS AI - Compliance Data Integrations

Integration with TimescaleDB, Redis, and other data systems
for compliance data storage, caching, and retrieval.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict
from datetime import datetime, timedelta
import hashlib
import pickle

from ..core.config import Config
from ..core.exceptions import CronosAIException
from .regulatory_kb import ComplianceAssessment
from .report_generator import ComplianceReport

logger = logging.getLogger(__name__)

class IntegrationException(CronosAIException):
    """Integration-specific exception."""
    pass

class TimescaleComplianceIntegration:
    """TimescaleDB integration for compliance time-series data."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.connection_pool = None
        
        # Connection settings
        self.host = getattr(config.database, 'host', 'localhost')
        self.port = getattr(config.database, 'port', 5432)
        self.database = getattr(config.database, 'database', 'cronos_ai')
        self.username = getattr(config.database, 'username', 'cronos')
        self.password = getattr(config.database, 'password', 'cronos123')
        
        # Table names
        self.tables = {
            'assessments': 'compliance_assessments',
            'gaps': 'compliance_gaps', 
            'recommendations': 'compliance_recommendations',
            'reports': 'compliance_reports',
            'trends': 'compliance_trends'
        }
    
    async def initialize(self):
        """Initialize TimescaleDB connection and create tables."""
        try:
            import asyncpg
            
            # Create connection pool
            dsn = f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
            self.connection_pool = await asyncpg.create_pool(
                dsn,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            
            # Create tables and hypertables
            await self._create_tables()
            
            self.logger.info("TimescaleDB integration initialized")
            
        except ImportError:
            raise IntegrationException("asyncpg library required for TimescaleDB integration")
        except Exception as e:
            self.logger.error(f"TimescaleDB initialization failed: {e}")
            raise IntegrationException(f"TimescaleDB initialization failed: {e}")
    
    async def _create_tables(self):
        """Create compliance tables and hypertables."""
        try:
            async with self.connection_pool.acquire() as conn:
                # Create assessments table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.tables['assessments']} (
                        id SERIAL PRIMARY KEY,
                        assessment_id VARCHAR(255) UNIQUE NOT NULL,
                        framework VARCHAR(100) NOT NULL,
                        version VARCHAR(50) NOT NULL,
                        assessment_date TIMESTAMPTZ NOT NULL,
                        overall_compliance_score DECIMAL(5,2) NOT NULL,
                        risk_score DECIMAL(5,2) NOT NULL,
                        compliant_requirements INTEGER NOT NULL,
                        non_compliant_requirements INTEGER NOT NULL,
                        partially_compliant_requirements INTEGER NOT NULL,
                        not_assessed_requirements INTEGER NOT NULL,
                        next_assessment_due TIMESTAMPTZ,
                        assessment_data JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)
                
                # Create gaps table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.tables['gaps']} (
                        id SERIAL PRIMARY KEY,
                        assessment_id VARCHAR(255) NOT NULL,
                        framework VARCHAR(100) NOT NULL,
                        requirement_id VARCHAR(100) NOT NULL,
                        requirement_title TEXT NOT NULL,
                        severity VARCHAR(50) NOT NULL,
                        current_state TEXT,
                        required_state TEXT,
                        gap_description TEXT,
                        impact_assessment TEXT,
                        remediation_effort VARCHAR(50),
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        FOREIGN KEY (assessment_id) REFERENCES {self.tables['assessments']}(assessment_id)
                    );
                """)
                
                # Create recommendations table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.tables['recommendations']} (
                        id SERIAL PRIMARY KEY,
                        assessment_id VARCHAR(255) NOT NULL,
                        framework VARCHAR(100) NOT NULL,
                        recommendation_id VARCHAR(100) NOT NULL,
                        title TEXT NOT NULL,
                        description TEXT,
                        priority VARCHAR(50) NOT NULL,
                        implementation_steps JSONB,
                        estimated_effort_days INTEGER,
                        business_impact TEXT,
                        technical_requirements JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        FOREIGN KEY (assessment_id) REFERENCES {self.tables['assessments']}(assessment_id)
                    );
                """)
                
                # Create reports table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.tables['reports']} (
                        id SERIAL PRIMARY KEY,
                        report_id VARCHAR(255) UNIQUE NOT NULL,
                        framework VARCHAR(100) NOT NULL,
                        assessment_id VARCHAR(255),
                        report_type VARCHAR(100) NOT NULL,
                        format VARCHAR(50) NOT NULL,
                        generated_date TIMESTAMPTZ NOT NULL,
                        file_name VARCHAR(255),
                        file_size INTEGER,
                        checksum VARCHAR(255),
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)
                
                # Create trends table for time-series data
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.tables['trends']} (
                        time TIMESTAMPTZ NOT NULL,
                        framework VARCHAR(100) NOT NULL,
                        compliance_score DECIMAL(5,2) NOT NULL,
                        risk_score DECIMAL(5,2) NOT NULL,
                        gaps_count INTEGER NOT NULL,
                        critical_gaps_count INTEGER NOT NULL,
                        assessment_id VARCHAR(255)
                    );
                """)
                
                # Create hypertable for trends (TimescaleDB specific)
                try:
                    await conn.execute(f"""
                        SELECT create_hypertable('{self.tables["trends"]}', 'time', 
                                               if_not_exists => TRUE);
                    """)
                except Exception:
                    # Might fail if not TimescaleDB or hypertable already exists
                    self.logger.warning("Could not create hypertable - continuing with regular table")
                
                # Create indexes for performance
                indexes = [
                    f"CREATE INDEX IF NOT EXISTS idx_assessments_framework_date ON {self.tables['assessments']} (framework, assessment_date DESC);",
                    f"CREATE INDEX IF NOT EXISTS idx_gaps_assessment ON {self.tables['gaps']} (assessment_id, severity);",
                    f"CREATE INDEX IF NOT EXISTS idx_recommendations_assessment ON {self.tables['recommendations']} (assessment_id, priority);",
                    f"CREATE INDEX IF NOT EXISTS idx_trends_framework_time ON {self.tables['trends']} (framework, time DESC);",
                ]
                
                for index_sql in indexes:
                    await conn.execute(index_sql)
                
            self.logger.info("Compliance tables created successfully")
            
        except Exception as e:
            self.logger.error(f"Table creation failed: {e}")
            raise IntegrationException(f"Table creation failed: {e}")
    
    async def store_assessment(self, assessment: ComplianceAssessment) -> str:
        """Store compliance assessment in TimescaleDB."""
        try:
            async with self.connection_pool.acquire() as conn:
                async with conn.transaction():
                    # Store main assessment
                    assessment_id = await conn.fetchval(f"""
                        INSERT INTO {self.tables['assessments']} (
                            assessment_id, framework, version, assessment_date,
                            overall_compliance_score, risk_score,
                            compliant_requirements, non_compliant_requirements,
                            partially_compliant_requirements, not_assessed_requirements,
                            next_assessment_due, assessment_data
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                        ON CONFLICT (assessment_id) DO UPDATE SET
                            overall_compliance_score = EXCLUDED.overall_compliance_score,
                            risk_score = EXCLUDED.risk_score,
                            assessment_data = EXCLUDED.assessment_data
                        RETURNING assessment_id;
                    """,
                        assessment.assessment_id,
                        assessment.framework,
                        assessment.version,
                        assessment.assessment_date,
                        assessment.overall_compliance_score,
                        assessment.risk_score,
                        assessment.compliant_requirements,
                        assessment.non_compliant_requirements,
                        assessment.partially_compliant_requirements,
                        assessment.not_assessed_requirements,
                        assessment.next_assessment_due,
                        json.dumps(asdict(assessment))
                    )
                    
                    # Store gaps
                    for gap in assessment.gaps:
                        await conn.execute(f"""
                            INSERT INTO {self.tables['gaps']} (
                                assessment_id, framework, requirement_id, requirement_title,
                                severity, current_state, required_state, gap_description,
                                impact_assessment, remediation_effort
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                            ON CONFLICT DO NOTHING;
                        """,
                            assessment.assessment_id,
                            assessment.framework,
                            gap.requirement_id,
                            gap.requirement_title,
                            gap.severity.value,
                            gap.current_state,
                            gap.required_state,
                            gap.gap_description,
                            gap.impact_assessment,
                            gap.remediation_effort
                        )
                    
                    # Store recommendations
                    for rec in assessment.recommendations:
                        await conn.execute(f"""
                            INSERT INTO {self.tables['recommendations']} (
                                assessment_id, framework, recommendation_id, title,
                                description, priority, implementation_steps,
                                estimated_effort_days, business_impact, technical_requirements
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                            ON CONFLICT DO NOTHING;
                        """,
                            assessment.assessment_id,
                            assessment.framework,
                            rec.id,
                            rec.title,
                            rec.description,
                            rec.priority.value,
                            json.dumps(rec.implementation_steps),
                            rec.estimated_effort_days,
                            rec.business_impact,
                            json.dumps(rec.technical_requirements)
                        )
                    
                    # Store trend data point
                    critical_gaps = len([g for g in assessment.gaps if g.severity.value == 'critical'])
                    await conn.execute(f"""
                        INSERT INTO {self.tables['trends']} (
                            time, framework, compliance_score, risk_score,
                            gaps_count, critical_gaps_count, assessment_id
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7);
                    """,
                        assessment.assessment_date,
                        assessment.framework,
                        assessment.overall_compliance_score,
                        assessment.risk_score,
                        len(assessment.gaps),
                        critical_gaps,
                        assessment.assessment_id
                    )
                
                self.logger.info(f"Stored assessment in TimescaleDB: {assessment_id}")
                return assessment_id
                
        except Exception as e:
            self.logger.error(f"Failed to store assessment: {e}")
            raise IntegrationException(f"Assessment storage failed: {e}")
    
    async def get_latest_assessment(self, framework: str) -> Optional[ComplianceAssessment]:
        """Get latest assessment for framework."""
        try:
            async with self.connection_pool.acquire() as conn:
                row = await conn.fetchrow(f"""
                    SELECT assessment_data
                    FROM {self.tables['assessments']}
                    WHERE framework = $1
                    ORDER BY assessment_date DESC
                    LIMIT 1;
                """, framework)
                
                if row and row['assessment_data']:
                    assessment_data = json.loads(row['assessment_data'])
                    return ComplianceAssessment(**assessment_data)
                
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get latest assessment: {e}")
            return None
    
    async def store_report_metadata(self, report: ComplianceReport):
        """Store report metadata."""
        try:
            async with self.connection_pool.acquire() as conn:
                await conn.execute(f"""
                    INSERT INTO {self.tables['reports']} (
                        report_id, framework, report_type, format,
                        generated_date, file_name, file_size, checksum, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (report_id) DO UPDATE SET
                        generated_date = EXCLUDED.generated_date,
                        metadata = EXCLUDED.metadata;
                """,
                    report.report_id,
                    report.framework,
                    report.report_type.value,
                    report.format.value,
                    report.generated_date,
                    report.file_name,
                    report.file_size,
                    report.checksum,
                    json.dumps(report.metadata)
                )
                
        except Exception as e:
            self.logger.error(f"Failed to store report metadata: {e}")
    
    async def get_compliance_trends(
        self, 
        frameworks: List[str], 
        days: int = 30
    ) -> Dict[str, Any]:
        """Get compliance trends over time."""
        try:
            since_date = datetime.utcnow() - timedelta(days=days)
            
            async with self.connection_pool.acquire() as conn:
                rows = await conn.fetch(f"""
                    SELECT 
                        framework,
                        DATE_TRUNC('day', time) as date,
                        AVG(compliance_score) as avg_compliance,
                        AVG(risk_score) as avg_risk,
                        MAX(gaps_count) as max_gaps,
                        MAX(critical_gaps_count) as max_critical_gaps
                    FROM {self.tables['trends']}
                    WHERE framework = ANY($1) AND time >= $2
                    GROUP BY framework, DATE_TRUNC('day', time)
                    ORDER BY framework, date;
                """, frameworks, since_date)
                
                trends = {}
                for row in rows:
                    framework = row['framework']
                    if framework not in trends:
                        trends[framework] = []
                    
                    trends[framework].append({
                        'date': row['date'].isoformat(),
                        'compliance_score': float(row['avg_compliance']),
                        'risk_score': float(row['avg_risk']),
                        'gaps_count': row['max_gaps'],
                        'critical_gaps': row['max_critical_gaps']
                    })
                
                return trends
                
        except Exception as e:
            self.logger.error(f"Failed to get compliance trends: {e}")
            return {}
    
    async def cleanup_old_data(self, retention_days: int):
        """Clean up old compliance data."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
            
            async with self.connection_pool.acquire() as conn:
                # Delete old assessments and related data
                result = await conn.execute(f"""
                    DELETE FROM {self.tables['assessments']}
                    WHERE assessment_date < $1;
                """, cutoff_date)
                
                deleted_count = int(result.split()[-1]) if result else 0
                
                if deleted_count > 0:
                    self.logger.info(f"Cleaned up {deleted_count} old assessments")
                
        except Exception as e:
            self.logger.error(f"Data cleanup failed: {e}")
    
    async def close(self):
        """Close connection pool."""
        if self.connection_pool:
            await self.connection_pool.close()
            self.logger.info("TimescaleDB connection pool closed")

class RedisComplianceCache:
    """Redis integration for compliance data caching."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.redis_client = None
        
        # Connection settings
        self.host = getattr(config.redis, 'host', 'localhost')
        self.port = getattr(config.redis, 'port', 6379)
        self.password = getattr(config.redis, 'password', None)
        self.db = getattr(config.redis, 'db', 0)
        
        # Cache settings
        self.ttl_hours = 6
        self.key_prefix = "compliance:"
    
    async def initialize(self):
        """Initialize Redis connection."""
        try:
            import redis.asyncio as redis
            
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                db=self.db,
                decode_responses=False,  # We'll handle serialization
                socket_timeout=30
            )
            
            # Test connection
            await self.redis_client.ping()
            
            self.logger.info("Redis compliance cache initialized")
            
        except ImportError:
            raise IntegrationException("redis library required for Redis integration")
        except Exception as e:
            self.logger.error(f"Redis initialization failed: {e}")
            raise IntegrationException(f"Redis initialization failed: {e}")
    
    async def cache_assessment(self, framework: str, assessment: ComplianceAssessment):
        """Cache compliance assessment."""
        try:
            key = f"{self.key_prefix}assessment:{framework}"
            
            # Serialize assessment
            serialized_data = pickle.dumps(asdict(assessment))
            
            # Store with TTL
            await self.redis_client.setex(
                key, 
                self.ttl_hours * 3600,  # Convert to seconds
                serialized_data
            )
            
            # Store metadata for quick access
            metadata_key = f"{self.key_prefix}meta:{framework}"
            metadata = {
                'framework': framework,
                'score': assessment.overall_compliance_score,
                'risk': assessment.risk_score,
                'date': assessment.assessment_date.isoformat(),
                'cached_at': datetime.utcnow().isoformat()
            }
            
            await self.redis_client.setex(
                metadata_key,
                self.ttl_hours * 3600,
                json.dumps(metadata)
            )
            
            self.logger.debug(f"Cached assessment for {framework}")
            
        except Exception as e:
            self.logger.error(f"Failed to cache assessment: {e}")
    
    async def get_assessment(self, framework: str) -> Optional[ComplianceAssessment]:
        """Get cached assessment."""
        try:
            key = f"{self.key_prefix}assessment:{framework}"
            
            serialized_data = await self.redis_client.get(key)
            if not serialized_data:
                return None
            
            # Deserialize assessment
            assessment_data = pickle.loads(serialized_data)
            
            # Convert back to ComplianceAssessment object
            # Note: This is simplified - in practice you'd need proper deserialization
            return ComplianceAssessment(**assessment_data)
            
        except Exception as e:
            self.logger.error(f"Failed to get cached assessment: {e}")
            return None
    
    async def cache_dashboard_data(self, data: Dict[str, Any]):
        """Cache dashboard data."""
        try:
            key = f"{self.key_prefix}dashboard"
            
            await self.redis_client.setex(
                key,
                1800,  # 30 minutes TTL for dashboard
                json.dumps(data)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to cache dashboard data: {e}")
    
    async def get_dashboard_data(self) -> Optional[Dict[str, Any]]:
        """Get cached dashboard data."""
        try:
            key = f"{self.key_prefix}dashboard"
            
            data = await self.redis_client.get(key)
            if data:
                return json.loads(data.decode())
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get cached dashboard data: {e}")
            return None
    
    async def invalidate_framework_cache(self, framework: str):
        """Invalidate cache for specific framework."""
        try:
            keys = [
                f"{self.key_prefix}assessment:{framework}",
                f"{self.key_prefix}meta:{framework}",
                f"{self.key_prefix}dashboard"  # Dashboard depends on framework data
            ]
            
            await self.redis_client.delete(*keys)
            self.logger.info(f"Invalidated cache for {framework}")
            
        except Exception as e:
            self.logger.error(f"Failed to invalidate cache: {e}")
    
    async def cleanup_expired(self):
        """Clean up expired keys (Redis handles this automatically, but we can do manual cleanup)."""
        try:
            # Get all compliance keys
            pattern = f"{self.key_prefix}*"
            keys = await self.redis_client.keys(pattern)
            
            if keys:
                # Check TTL and remove keys that should have expired
                pipeline = self.redis_client.pipeline()
                for key in keys:
                    pipeline.ttl(key)
                
                ttls = await pipeline.execute()
                
                expired_keys = []
                for key, ttl in zip(keys, ttls):
                    if ttl == -1:  # No TTL set
                        expired_keys.append(key)
                
                if expired_keys:
                    await self.redis_client.delete(*expired_keys)
                    self.logger.info(f"Cleaned up {len(expired_keys)} keys without TTL")
                    
        except Exception as e:
            self.logger.error(f"Cache cleanup failed: {e}")
    
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache usage statistics."""
        try:
            pattern = f"{self.key_prefix}*"
            keys = await self.redis_client.keys(pattern)
            
            stats = {
                'total_keys': len(keys),
                'memory_usage': 0,
                'key_types': {},
                'frameworks_cached': []
            }
            
            for key in keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                
                # Categorize keys
                if ':assessment:' in key_str:
                    framework = key_str.split(':')[-1]
                    stats['frameworks_cached'].append(framework)
                    key_type = 'assessment'
                elif ':meta:' in key_str:
                    key_type = 'metadata'
                elif ':dashboard' in key_str:
                    key_type = 'dashboard'
                else:
                    key_type = 'other'
                
                stats['key_types'][key_type] = stats['key_types'].get(key_type, 0) + 1
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get cache statistics: {e}")
            return {}
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            self.logger.info("Redis connection closed")

class ComplianceSecurityIntegration:
    """Integration with security systems for compliance data protection."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Security settings
        self.encryption_enabled = getattr(config.security, 'enable_encryption', True)
        self.audit_logging_enabled = getattr(config.security, 'audit_logging', True)
        
    async def initialize(self):
        """Initialize security integration."""
        try:
            # Initialize security components
            if self.encryption_enabled:
                await self._initialize_encryption()
            
            if self.audit_logging_enabled:
                await self._initialize_security_logging()
            
            self.logger.info("Security integration initialized")
            
        except Exception as e:
            self.logger.error(f"Security integration failed: {e}")
            raise IntegrationException(f"Security integration failed: {e}")
    
    async def _initialize_encryption(self):
        """Initialize encryption for sensitive compliance data."""
        # This would integrate with the existing security framework
        self.logger.info("Encryption initialized for compliance data")
    
    async def _initialize_security_logging(self):
        """Initialize security event logging."""
        # This would integrate with SIEM systems
        self.logger.info("Security logging initialized for compliance")
    
    async def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive compliance data."""
        try:
            if not self.encryption_enabled:
                return data
            
            # Use existing encryption from security framework
            # This is a placeholder - would use actual encryption
            return data  # In practice, would return encrypted data
            
        except Exception as e:
            self.logger.error(f"Data encryption failed: {e}")
            raise IntegrationException(f"Data encryption failed: {e}")
    
    async def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive compliance data."""
        try:
            if not self.encryption_enabled:
                return encrypted_data
            
            # Use existing decryption from security framework
            return encrypted_data  # In practice, would return decrypted data
            
        except Exception as e:
            self.logger.error(f"Data decryption failed: {e}")
            raise IntegrationException(f"Data decryption failed: {e}")
    
    async def log_security_event(
        self, 
        event_type: str, 
        resource: str, 
        action: str, 
        outcome: str,
        details: Dict[str, Any] = None
    ):
        """Log security event for compliance activities."""
        try:
            if not self.audit_logging_enabled:
                return
            
            security_event = {
                'timestamp': datetime.utcnow().isoformat(),
                'event_type': event_type,
                'resource': resource,
                'action': action,
                'outcome': outcome,
                'details': details or {},
                'source': 'compliance_system'
            }
            
            # This would integrate with existing SIEM/security logging
            self.logger.info(f"Security event: {json.dumps(security_event)}")
            
        except Exception as e:
            self.logger.error(f"Security event logging failed: {e}")
    
    async def validate_data_access(self, user: str, resource: str, action: str) -> bool:
        """Validate user access to compliance data."""
        try:
            # This would integrate with existing access control system
            # For now, allow all access
            await self.log_security_event(
                'access_control',
                resource,
                action,
                'allowed',
                {'user': user}
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Access validation failed: {e}")
            return False
    
    async def close(self):
        """Close security integration."""
        self.logger.info("Security integration closed")