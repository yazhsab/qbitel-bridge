"""
CRONOS AI - Blockchain-based Audit Trail System

Enterprise-grade audit trail management with blockchain integration
for immutable compliance record keeping and regulatory verification.
"""

import asyncio
import logging
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import base64
from collections import defaultdict
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import secrets

from ..core.config import Config
from ..core.exceptions import CronosAIException
from ..monitoring.enterprise_metrics import get_enterprise_metrics

logger = logging.getLogger(__name__)

class AuditException(CronosAIException):
    """Audit trail specific exception."""
    pass

class EventType(Enum):
    """Types of audit events."""
    ASSESSMENT_STARTED = "assessment_started"
    ASSESSMENT_COMPLETED = "assessment_completed"
    REQUIREMENT_EVALUATED = "requirement_evaluated"
    GAP_IDENTIFIED = "gap_identified"
    RECOMMENDATION_GENERATED = "recommendation_generated"
    REPORT_GENERATED = "report_generated"
    REPORT_ACCESSED = "report_accessed"
    CONFIGURATION_CHANGED = "configuration_changed"
    USER_ACTION = "user_action"
    SYSTEM_ACTION = "system_action"
    SECURITY_EVENT = "security_event"
    COMPLIANCE_VIOLATION = "compliance_violation"
    REMEDIATION_STARTED = "remediation_started"
    REMEDIATION_COMPLETED = "remediation_completed"

class EventSeverity(Enum):
    """Severity levels for audit events."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class AuditEvent:
    """Individual audit event record."""
    event_id: str
    timestamp: datetime
    event_type: EventType
    severity: EventSeverity
    actor: str  # User, system, or service responsible
    resource: str  # What was affected
    action: str  # What action was performed
    outcome: str  # Success, failure, etc.
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    compliance_framework: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['event_type'] = self.event_type.value
        data['severity'] = self.severity.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['event_type'] = EventType(data['event_type'])
        data['severity'] = EventSeverity(data['severity'])
        return cls(**data)

@dataclass
class AuditBlock:
    """Blockchain block containing audit events."""
    block_number: int
    timestamp: datetime
    previous_hash: str
    events: List[AuditEvent]
    merkle_root: str
    block_hash: str
    nonce: int = 0
    validator: str = "cronos_ai"
    signature: str = ""
    
    def calculate_hash(self) -> str:
        """Calculate block hash."""
        block_data = {
            'block_number': self.block_number,
            'timestamp': self.timestamp.isoformat(),
            'previous_hash': self.previous_hash,
            'merkle_root': self.merkle_root,
            'nonce': self.nonce,
            'validator': self.validator,
            'events_hash': self._calculate_events_hash()
        }
        
        block_string = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def _calculate_events_hash(self) -> str:
        """Calculate combined hash of all events in block."""
        events_data = [event.to_dict() for event in self.events]
        events_string = json.dumps(events_data, sort_keys=True)
        return hashlib.sha256(events_string.encode()).hexdigest()
    
    def calculate_merkle_root(self) -> str:
        """Calculate Merkle root of events."""
        if not self.events:
            return hashlib.sha256(b"").hexdigest()
        
        # Create leaf hashes
        leaves = []
        for event in self.events:
            event_string = json.dumps(event.to_dict(), sort_keys=True)
            leaves.append(hashlib.sha256(event_string.encode()).hexdigest())
        
        # Build Merkle tree
        return self._build_merkle_tree(leaves)
    
    def _build_merkle_tree(self, hashes: List[str]) -> str:
        """Build Merkle tree from hash list."""
        if len(hashes) == 1:
            return hashes[0]
        
        if len(hashes) % 2 == 1:
            hashes.append(hashes[-1])  # Duplicate last hash if odd number
        
        next_level = []
        for i in range(0, len(hashes), 2):
            combined = hashes[i] + hashes[i + 1]
            next_level.append(hashlib.sha256(combined.encode()).hexdigest())
        
        return self._build_merkle_tree(next_level)

class CryptographicManager:
    """Manages cryptographic operations for audit trail security."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Generate or load keys
        self.private_key = self._load_or_generate_private_key()
        self.public_key = self.private_key.public_key()
        
        # Encryption key for sensitive data
        self.encryption_key = self._derive_encryption_key()
    
    def _load_or_generate_private_key(self) -> rsa.RSAPrivateKey:
        """Load existing private key or generate new one."""
        try:
            # Try to load existing key (in production, this would come from secure storage)
            key_path = getattr(self.config, 'audit_private_key_path', None)
            if key_path and Path(key_path).exists():
                with open(key_path, 'rb') as f:
                    private_key = serialization.load_pem_private_key(
                        f.read(), 
                        password=None
                    )
                return private_key
        except Exception as e:
            self.logger.warning(f"Could not load existing key: {e}")
        
        # Generate new key
        self.logger.info("Generating new RSA key pair for audit trail")
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        return private_key
    
    def _derive_encryption_key(self) -> bytes:
        """Derive encryption key from configuration."""
        # In production, this would use proper key derivation
        master_key = getattr(self.config, 'audit_master_key', 'cronos_ai_audit_2024')
        return hashlib.pbkdf2_hmac(
            'sha256',
            master_key.encode(),
            b'cronos_ai_salt',
            100000,
            32
        )
    
    def sign_block(self, block: AuditBlock) -> str:
        """Digitally sign audit block."""
        try:
            block_hash_bytes = block.block_hash.encode()
            signature = self.private_key.sign(
                block_hash_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return base64.b64encode(signature).decode()
        except Exception as e:
            raise AuditException(f"Block signing failed: {e}")
    
    def verify_block_signature(self, block: AuditBlock, signature: str) -> bool:
        """Verify block digital signature."""
        try:
            signature_bytes = base64.b64decode(signature)
            block_hash_bytes = block.block_hash.encode()
            
            self.public_key.verify(
                signature_bytes,
                block_hash_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive audit data."""
        try:
            # Generate random IV
            iv = secrets.token_bytes(16)
            
            # Encrypt data
            cipher = Cipher(
                algorithms.AES(self.encryption_key),
                modes.CBC(iv)
            )
            encryptor = cipher.encryptor()
            
            # Pad data to block size
            padded_data = self._pad_data(data.encode())
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()
            
            # Combine IV and ciphertext
            encrypted = iv + ciphertext
            return base64.b64encode(encrypted).decode()
            
        except Exception as e:
            raise AuditException(f"Data encryption failed: {e}")
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive audit data."""
        try:
            # Decode and split IV and ciphertext
            encrypted_bytes = base64.b64decode(encrypted_data)
            iv = encrypted_bytes[:16]
            ciphertext = encrypted_bytes[16:]
            
            # Decrypt data
            cipher = Cipher(
                algorithms.AES(self.encryption_key),
                modes.CBC(iv)
            )
            decryptor = cipher.decryptor()
            padded_data = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Remove padding
            data = self._unpad_data(padded_data)
            return data.decode()
            
        except Exception as e:
            raise AuditException(f"Data decryption failed: {e}")
    
    def _pad_data(self, data: bytes) -> bytes:
        """PKCS7 padding."""
        block_size = 16
        padding_len = block_size - (len(data) % block_size)
        padding = bytes([padding_len] * padding_len)
        return data + padding
    
    def _unpad_data(self, data: bytes) -> bytes:
        """Remove PKCS7 padding."""
        padding_len = data[-1]
        return data[:-padding_len]

class BlockchainAuditTrail:
    """Blockchain-based audit trail implementation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.crypto_manager = CryptographicManager(config)
        self.metrics = get_enterprise_metrics()
        
        # Blockchain storage
        self.blocks: List[AuditBlock] = []
        self.pending_events: List[AuditEvent] = []
        
        # Configuration
        self.block_size = getattr(config, 'audit_block_size', 100)  # Events per block
        self.block_interval = getattr(config, 'audit_block_interval', 300)  # 5 minutes
        
        # Initialize genesis block
        self._create_genesis_block()
        
        # Background tasks
        self._running = False
        self._block_creation_task = None
    
    async def start(self):
        """Start audit trail service."""
        if not self._running:
            self._running = True
            self._block_creation_task = asyncio.create_task(self._block_creation_loop())
            self.logger.info("Blockchain audit trail started")
    
    async def stop(self):
        """Stop audit trail service."""
        self._running = False
        if self._block_creation_task:
            self._block_creation_task.cancel()
            try:
                await self._block_creation_task
            except asyncio.CancelledError:
                pass
        
        # Finalize any pending events
        if self.pending_events:
            await self._create_block()
        
        self.logger.info("Blockchain audit trail stopped")
    
    def _create_genesis_block(self):
        """Create the genesis block."""
        genesis_event = AuditEvent(
            event_id="genesis",
            timestamp=datetime.utcnow(),
            event_type=EventType.SYSTEM_ACTION,
            severity=EventSeverity.INFO,
            actor="system",
            resource="audit_trail",
            action="initialize",
            outcome="success",
            details={"message": "Audit trail blockchain initialized"}
        )
        
        genesis_block = AuditBlock(
            block_number=0,
            timestamp=datetime.utcnow(),
            previous_hash="0" * 64,  # Genesis block has no previous hash
            events=[genesis_event],
            merkle_root="",
            block_hash="",
            validator="cronos_ai"
        )
        
        genesis_block.merkle_root = genesis_block.calculate_merkle_root()
        genesis_block.block_hash = genesis_block.calculate_hash()
        genesis_block.signature = self.crypto_manager.sign_block(genesis_block)
        
        self.blocks.append(genesis_block)
        self.logger.info("Genesis block created")
    
    async def record_event(self, event: AuditEvent) -> str:
        """Record audit event to blockchain."""
        try:
            # Add event to pending queue
            self.pending_events.append(event)
            
            # Create block if we've reached the block size limit
            if len(self.pending_events) >= self.block_size:
                await self._create_block()
            
            # Record metrics
            self.metrics.increment_protocol_discovery_counter(
                "audit_events_recorded_total",
                labels={
                    "event_type": event.event_type.value,
                    "severity": event.severity.value
                }
            )
            
            return event.event_id
            
        except Exception as e:
            self.logger.error(f"Failed to record audit event: {e}")
            raise AuditException(f"Event recording failed: {e}")
    
    async def _create_block(self):
        """Create new block from pending events."""
        if not self.pending_events:
            return
        
        try:
            # Get previous block hash
            previous_hash = self.blocks[-1].block_hash if self.blocks else "0" * 64
            
            # Create new block
            new_block = AuditBlock(
                block_number=len(self.blocks),
                timestamp=datetime.utcnow(),
                previous_hash=previous_hash,
                events=self.pending_events.copy(),
                merkle_root="",
                block_hash="",
                validator="cronos_ai"
            )
            
            # Calculate hashes
            new_block.merkle_root = new_block.calculate_merkle_root()
            new_block.block_hash = new_block.calculate_hash()
            
            # Sign block
            new_block.signature = self.crypto_manager.sign_block(new_block)
            
            # Add to blockchain
            self.blocks.append(new_block)
            
            # Clear pending events
            self.pending_events.clear()
            
            # Record metrics
            self.metrics.increment_protocol_discovery_counter(
                "audit_blocks_created_total",
                labels={"validator": new_block.validator}
            )
            
            self.metrics.record_protocol_discovery_metric(
                "audit_events_per_block",
                len(new_block.events),
                {"block_number": str(new_block.block_number)}
            )
            
            self.logger.info(f"Block {new_block.block_number} created with {len(new_block.events)} events")
            
        except Exception as e:
            self.logger.error(f"Failed to create audit block: {e}")
            raise AuditException(f"Block creation failed: {e}")
    
    async def _block_creation_loop(self):
        """Background task to create blocks at regular intervals."""
        while self._running:
            try:
                await asyncio.sleep(self.block_interval)
                
                if self.pending_events:
                    await self._create_block()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Block creation loop error: {e}")
    
    def verify_blockchain_integrity(self) -> Dict[str, Any]:
        """Verify the integrity of the entire blockchain."""
        try:
            verification_result = {
                'valid': True,
                'total_blocks': len(self.blocks),
                'total_events': sum(len(block.events) for block in self.blocks),
                'issues': [],
                'verified_at': datetime.utcnow().isoformat()
            }
            
            # Verify each block
            for i, block in enumerate(self.blocks):
                block_issues = self._verify_block(block, i)
                if block_issues:
                    verification_result['issues'].extend(block_issues)
                    verification_result['valid'] = False
            
            # Verify chain continuity
            chain_issues = self._verify_chain_continuity()
            if chain_issues:
                verification_result['issues'].extend(chain_issues)
                verification_result['valid'] = False
            
            return verification_result
            
        except Exception as e:
            self.logger.error(f"Blockchain verification failed: {e}")
            return {
                'valid': False,
                'error': str(e),
                'verified_at': datetime.utcnow().isoformat()
            }
    
    def _verify_block(self, block: AuditBlock, index: int) -> List[str]:
        """Verify individual block integrity."""
        issues = []
        
        # Verify block hash
        calculated_hash = block.calculate_hash()
        if calculated_hash != block.block_hash:
            issues.append(f"Block {index}: Hash mismatch")
        
        # Verify Merkle root
        calculated_merkle = block.calculate_merkle_root()
        if calculated_merkle != block.merkle_root:
            issues.append(f"Block {index}: Merkle root mismatch")
        
        # Verify signature
        if not self.crypto_manager.verify_block_signature(block, block.signature):
            issues.append(f"Block {index}: Invalid signature")
        
        # Verify previous hash (except genesis block)
        if index > 0 and block.previous_hash != self.blocks[index - 1].block_hash:
            issues.append(f"Block {index}: Previous hash mismatch")
        
        return issues
    
    def _verify_chain_continuity(self) -> List[str]:
        """Verify blockchain continuity."""
        issues = []
        
        for i in range(1, len(self.blocks)):
            current_block = self.blocks[i]
            previous_block = self.blocks[i - 1]
            
            if current_block.previous_hash != previous_block.block_hash:
                issues.append(f"Chain break between blocks {i-1} and {i}")
            
            if current_block.block_number != previous_block.block_number + 1:
                issues.append(f"Block number sequence error at block {i}")
        
        return issues
    
    def get_blockchain_summary(self) -> Dict[str, Any]:
        """Get blockchain summary statistics."""
        if not self.blocks:
            return {'total_blocks': 0, 'total_events': 0}
        
        latest_block = self.blocks[-1]
        
        # Event type distribution
        event_types = defaultdict(int)
        for block in self.blocks:
            for event in block.events:
                event_types[event.event_type.value] += 1
        
        return {
            'total_blocks': len(self.blocks),
            'total_events': sum(len(block.events) for block in self.blocks),
            'pending_events': len(self.pending_events),
            'latest_block': {
                'number': latest_block.block_number,
                'timestamp': latest_block.timestamp.isoformat(),
                'events_count': len(latest_block.events),
                'hash': latest_block.block_hash[:16] + "..."
            },
            'event_type_distribution': dict(event_types),
            'blockchain_size_mb': self._calculate_blockchain_size() / (1024 * 1024),
            'genesis_timestamp': self.blocks[0].timestamp.isoformat() if self.blocks else None
        }
    
    def _calculate_blockchain_size(self) -> int:
        """Calculate approximate blockchain size in bytes."""
        total_size = 0
        for block in self.blocks:
            # Approximate size calculation
            block_data = {
                'block_number': block.block_number,
                'timestamp': block.timestamp.isoformat(),
                'previous_hash': block.previous_hash,
                'events': [event.to_dict() for event in block.events],
                'merkle_root': block.merkle_root,
                'block_hash': block.block_hash,
                'signature': block.signature
            }
            total_size += len(json.dumps(block_data).encode())
        
        return total_size
    
    async def query_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[EventType]] = None,
        actor: Optional[str] = None,
        resource: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """Query audit events with filters."""
        try:
            matching_events = []
            
            for block in self.blocks:
                for event in block.events:
                    # Time filter
                    if start_time and event.timestamp < start_time:
                        continue
                    if end_time and event.timestamp > end_time:
                        continue
                    
                    # Event type filter
                    if event_types and event.event_type not in event_types:
                        continue
                    
                    # Actor filter
                    if actor and event.actor != actor:
                        continue
                    
                    # Resource filter
                    if resource and event.resource != resource:
                        continue
                    
                    matching_events.append(event)
                    
                    if len(matching_events) >= limit:
                        break
                
                if len(matching_events) >= limit:
                    break
            
            return matching_events
            
        except Exception as e:
            self.logger.error(f"Event query failed: {e}")
            raise AuditException(f"Event query failed: {e}")
    
    def export_blockchain(self, format: str = 'json') -> bytes:
        """Export blockchain data."""
        try:
            blockchain_data = {
                'metadata': {
                    'export_timestamp': datetime.utcnow().isoformat(),
                    'total_blocks': len(self.blocks),
                    'total_events': sum(len(block.events) for block in self.blocks),
                    'format_version': '1.0'
                },
                'blocks': []
            }
            
            for block in self.blocks:
                block_data = {
                    'block_number': block.block_number,
                    'timestamp': block.timestamp.isoformat(),
                    'previous_hash': block.previous_hash,
                    'merkle_root': block.merkle_root,
                    'block_hash': block.block_hash,
                    'validator': block.validator,
                    'signature': block.signature,
                    'events': [event.to_dict() for event in block.events]
                }
                blockchain_data['blocks'].append(block_data)
            
            if format.lower() == 'json':
                return json.dumps(blockchain_data, indent=2).encode()
            else:
                raise AuditException(f"Unsupported export format: {format}")
                
        except Exception as e:
            self.logger.error(f"Blockchain export failed: {e}")
            raise AuditException(f"Blockchain export failed: {e}")

class AuditTrailManager:
    """Main audit trail manager coordinating all audit operations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.blockchain = BlockchainAuditTrail(config)
        self.metrics = get_enterprise_metrics()
        
        # Event tracking
        self.session_events: Dict[str, List[str]] = defaultdict(list)
        
    async def start(self):
        """Start audit trail manager."""
        await self.blockchain.start()
        self.logger.info("Audit trail manager started")
    
    async def stop(self):
        """Stop audit trail manager."""
        await self.blockchain.stop()
        self.logger.info("Audit trail manager stopped")
    
    async def record_compliance_event(
        self,
        event_type: EventType,
        actor: str,
        resource: str,
        action: str,
        outcome: str,
        details: Optional[Dict[str, Any]] = None,
        severity: EventSeverity = EventSeverity.INFO,
        compliance_framework: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> str:
        """Record compliance-specific audit event."""
        try:
            event_id = self._generate_event_id()
            
            audit_event = AuditEvent(
                event_id=event_id,
                timestamp=datetime.utcnow(),
                event_type=event_type,
                severity=severity,
                actor=actor,
                resource=resource,
                action=action,
                outcome=outcome,
                details=details or {},
                compliance_framework=compliance_framework,
                session_id=session_id
            )
            
            # Record to blockchain
            await self.blockchain.record_event(audit_event)
            
            # Track session events
            if session_id:
                self.session_events[session_id].append(event_id)
            
            return event_id
            
        except Exception as e:
            self.logger.error(f"Failed to record compliance event: {e}")
            raise AuditException(f"Compliance event recording failed: {e}")
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        timestamp = int(time.time() * 1000000)  # Microsecond precision
        random_part = secrets.token_hex(4)
        return f"EVT_{timestamp}_{random_part}"
    
    async def create_compliance_trail(
        self,
        framework: str,
        assessment_id: str,
        events: List[Dict[str, Any]]
    ) -> str:
        """Create comprehensive compliance audit trail."""
        try:
            trail_id = f"TRAIL_{framework}_{assessment_id}_{int(time.time())}"
            
            # Record trail start
            await self.record_compliance_event(
                EventType.ASSESSMENT_STARTED,
                "system",
                f"compliance_trail_{trail_id}",
                "create_trail",
                "success",
                {
                    "trail_id": trail_id,
                    "framework": framework,
                    "assessment_id": assessment_id,
                    "event_count": len(events)
                },
                compliance_framework=framework
            )
            
            # Record all events
            for event_data in events:
                await self.record_compliance_event(
                    EventType(event_data.get('event_type', 'system_action')),
                    event_data.get('actor', 'system'),
                    event_data.get('resource', 'unknown'),
                    event_data.get('action', 'unknown'),
                    event_data.get('outcome', 'unknown'),
                    event_data.get('details', {}),
                    EventSeverity(event_data.get('severity', 'info')),
                    framework
                )
            
            # Record trail completion
            await self.record_compliance_event(
                EventType.ASSESSMENT_COMPLETED,
                "system",
                f"compliance_trail_{trail_id}",
                "complete_trail",
                "success",
                {
                    "trail_id": trail_id,
                    "events_recorded": len(events)
                },
                compliance_framework=framework
            )
            
            return trail_id
            
        except Exception as e:
            self.logger.error(f"Failed to create compliance trail: {e}")
            raise AuditException(f"Compliance trail creation failed: {e}")
    
    async def generate_compliance_audit_report(
        self,
        framework: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate compliance audit report for specific timeframe."""
        try:
            # Query relevant events
            events = await self.blockchain.query_events(
                start_time=start_date,
                end_time=end_date,
                limit=10000  # Large limit for comprehensive report
            )
            
            # Filter by framework if specified
            if framework:
                events = [e for e in events if e.compliance_framework == framework]
            
            # Analyze events
            analysis = self._analyze_audit_events(events)
            
            # Generate report
            report = {
                'report_metadata': {
                    'framework': framework,
                    'period_start': start_date.isoformat(),
                    'period_end': end_date.isoformat(),
                    'generated_at': datetime.utcnow().isoformat(),
                    'total_events': len(events)
                },
                'event_summary': analysis['summary'],
                'compliance_activities': analysis['activities'],
                'security_events': analysis['security'],
                'risk_indicators': analysis['risks'],
                'blockchain_integrity': self.blockchain.verify_blockchain_integrity(),
                'recommendations': analysis['recommendations']
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate audit report: {e}")
            raise AuditException(f"Audit report generation failed: {e}")
    
    def _analyze_audit_events(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Analyze audit events for reporting."""
        analysis = {
            'summary': {
                'total_events': len(events),
                'event_types': defaultdict(int),
                'severity_distribution': defaultdict(int),
                'actors': defaultdict(int),
                'outcomes': defaultdict(int)
            },
            'activities': [],
            'security': [],
            'risks': [],
            'recommendations': []
        }
        
        for event in events:
            # Summary statistics
            analysis['summary']['event_types'][event.event_type.value] += 1
            analysis['summary']['severity_distribution'][event.severity.value] += 1
            analysis['summary']['actors'][event.actor] += 1
            analysis['summary']['outcomes'][event.outcome] += 1
            
            # Categorize events
            if event.event_type in [EventType.ASSESSMENT_STARTED, EventType.ASSESSMENT_COMPLETED]:
                analysis['activities'].append({
                    'timestamp': event.timestamp.isoformat(),
                    'type': event.event_type.value,
                    'resource': event.resource,
                    'outcome': event.outcome,
                    'details': event.details
                })
            
            if event.event_type in [EventType.SECURITY_EVENT, EventType.COMPLIANCE_VIOLATION]:
                analysis['security'].append({
                    'timestamp': event.timestamp.isoformat(),
                    'severity': event.severity.value,
                    'type': event.event_type.value,
                    'details': event.details
                })
            
            if event.severity in [EventSeverity.ERROR, EventSeverity.CRITICAL]:
                analysis['risks'].append({
                    'timestamp': event.timestamp.isoformat(),
                    'severity': event.severity.value,
                    'event_type': event.event_type.value,
                    'resource': event.resource,
                    'outcome': event.outcome
                })
        
        # Generate recommendations
        if analysis['summary']['severity_distribution']['critical'] > 0:
            analysis['recommendations'].append(
                "Critical events detected - immediate investigation required"
            )
        
        if analysis['summary']['outcomes']['failure'] > analysis['summary']['outcomes'].get('success', 0):
            analysis['recommendations'].append(
                "High failure rate detected - review system processes"
            )
        
        return analysis
    
    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get comprehensive audit trail statistics."""
        blockchain_summary = self.blockchain.get_blockchain_summary()
        
        return {
            'blockchain': blockchain_summary,
            'active_sessions': len(self.session_events),
            'session_events': {
                session: len(events) 
                for session, events in self.session_events.items()
            },
            'integrity_status': self.blockchain.verify_blockchain_integrity()['valid'],
            'uptime': 'active'  # Would track actual uptime in production
        }