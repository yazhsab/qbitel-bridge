"""
Automation Engine

Main orchestration engine for zero-touch automation.
Combines all automation components into a unified interface.

Capabilities:
- Protocol adapter auto-generation
- Self-configuring infrastructure
- Automated key and certificate lifecycle
- Self-healing orchestration
- Compliance automation
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set
import uuid
import threading

from ai_engine.automation.protocol_adapter_generator import (
    ProtocolAdapterGenerator,
    GeneratedAdapter,
    AdapterConfig,
)
from ai_engine.automation.configuration_manager import (
    ConfigurationManager,
    SecurityConfiguration,
    ConfigurationChange,
)
from ai_engine.automation.key_lifecycle_manager import (
    KeyLifecycleManager,
    KeyPolicy,
    KeyMetadata,
    KeyType,
    KeyState,
)
from ai_engine.automation.certificate_automation import (
    CertificateAutomation,
    CertificateType,
    CertificateState,
    Certificate,
)
from ai_engine.automation.self_healing_orchestrator import (
    SelfHealingOrchestrator,
    ComponentType,
    HealthStatus,
    RecoveryAction,
    Incident,
)


logger = logging.getLogger(__name__)


@dataclass
class AutomationConfig:
    """Configuration for Automation Engine."""

    # Feature enablement
    enable_adapter_generation: bool = True
    enable_config_management: bool = True
    enable_key_lifecycle: bool = True
    enable_cert_automation: bool = True
    enable_self_healing: bool = True

    # Key management
    key_rotation_check_interval: int = 3600  # 1 hour
    default_key_rotation_days: int = 90

    # Certificate management
    cert_renewal_check_interval: int = 86400  # 1 day
    default_cert_validity_days: int = 90
    cert_renewal_threshold_days: int = 30

    # Health monitoring
    health_check_interval: int = 30  # seconds
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60

    # Callbacks
    alert_callback: Optional[Callable[[Incident], None]] = None
    key_rotation_callback: Optional[Callable[[Any], None]] = None
    cert_renewal_callback: Optional[Callable[[Any], None]] = None


@dataclass
class AutomationStatus:
    """Status of automation engine."""

    status: str
    started_at: Optional[datetime]
    components: Dict[str, bool]
    stats: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "components": self.components,
            "stats": self.stats,
        }


@dataclass
class AutomationTask:
    """Automated task record."""

    task_id: str
    task_type: str
    description: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class AutomationEngine:
    """
    Main Zero-Touch Automation Engine.

    Orchestrates all automation components to provide:
    - Automated protocol adaptation
    - Self-configuring security
    - Automated key and certificate management
    - Self-healing infrastructure
    """

    def __init__(
        self,
        config: Optional[AutomationConfig] = None,
        hsm_provider: Optional[Any] = None,
        ca_provider: Optional[Any] = None,
    ):
        self._config = config or AutomationConfig()
        self._hsm_provider = hsm_provider
        self._ca_provider = ca_provider
        self._started_at: Optional[datetime] = None
        self._lock = threading.RLock()

        # Initialize components
        self._adapter_generator = ProtocolAdapterGenerator()
        self._config_manager = ConfigurationManager()
        self._key_manager = KeyLifecycleManager(
            hsm_provider=hsm_provider,
            rotation_callback=self._config.key_rotation_callback,
        )
        self._cert_automation = CertificateAutomation(
            ca_provider=ca_provider,
            key_manager=self._key_manager,
            renewal_callback=self._config.cert_renewal_callback,
        )
        self._self_healing = SelfHealingOrchestrator(
            alert_callback=self._config.alert_callback,
        )

        # Task tracking
        self._tasks: Dict[str, AutomationTask] = {}

        # Register self-healing components
        self._register_internal_components()

        logger.info("Automation Engine initialized")

    def start(self) -> None:
        """Start all automation services."""
        with self._lock:
            if self._started_at:
                logger.warning("Automation Engine already started")
                return

            self._started_at = datetime.utcnow()

            # Start key lifecycle management
            if self._config.enable_key_lifecycle:
                self._key_manager.start_rotation_scheduler(
                    self._config.key_rotation_check_interval
                )

            # Start certificate automation
            if self._config.enable_cert_automation:
                self._cert_automation.start_renewal_scheduler(
                    self._config.cert_renewal_check_interval
                )

            # Start self-healing monitoring
            if self._config.enable_self_healing:
                self._self_healing.start_monitoring(
                    self._config.health_check_interval
                )

            logger.info("Automation Engine started")

    def stop(self) -> None:
        """Stop all automation services."""
        with self._lock:
            if not self._started_at:
                return

            # Stop all schedulers
            self._key_manager.stop_rotation_scheduler()
            self._cert_automation.stop_renewal_scheduler()
            self._self_healing.stop_monitoring()

            self._started_at = None

            logger.info("Automation Engine stopped")

    # === Protocol Adapter Generation ===

    def generate_adapter(
        self,
        source_analysis: Dict[str, Any],
        target_analysis: Dict[str, Any],
        adapter_name: str,
        config: Optional[AdapterConfig] = None,
    ) -> GeneratedAdapter:
        """
        Generate a protocol adapter.

        Args:
            source_analysis: Source protocol analysis
            target_analysis: Target protocol analysis
            adapter_name: Name for the adapter
            config: Optional adapter configuration

        Returns:
            GeneratedAdapter
        """
        task = self._create_task(
            "adapter_generation",
            f"Generate adapter: {adapter_name}",
        )

        try:
            adapter = self._adapter_generator.generate_adapter(
                source_analysis,
                target_analysis,
                adapter_name,
                config,
            )

            self._complete_task(task, {"adapter_id": adapter.adapter_id})
            return adapter

        except Exception as e:
            self._fail_task(task, str(e))
            raise

    def generate_adapter_from_samples(
        self,
        source_samples: List[bytes],
        target_samples: List[bytes],
        adapter_name: str,
    ) -> GeneratedAdapter:
        """
        Generate adapter from sample data.

        Args:
            source_samples: Source protocol samples
            target_samples: Target protocol samples
            adapter_name: Adapter name

        Returns:
            GeneratedAdapter
        """
        task = self._create_task(
            "adapter_generation",
            f"Generate adapter from samples: {adapter_name}",
        )

        try:
            adapter = self._adapter_generator.generate_from_samples(
                source_samples,
                target_samples,
                adapter_name,
            )

            self._complete_task(task, {"adapter_id": adapter.adapter_id})
            return adapter

        except Exception as e:
            self._fail_task(task, str(e))
            raise

    # === Configuration Management ===

    def auto_configure(
        self,
        system_analysis: Dict[str, Any],
        environment: str = "production",
    ) -> SecurityConfiguration:
        """
        Auto-configure security based on system analysis.

        Args:
            system_analysis: System analysis result
            environment: Target environment

        Returns:
            SecurityConfiguration
        """
        task = self._create_task(
            "auto_configuration",
            f"Auto-configure for {environment}",
        )

        try:
            config = self._config_manager.auto_configure(
                system_analysis,
                environment,
            )

            self._complete_task(task, {"config_id": config.config_id})
            return config

        except Exception as e:
            self._fail_task(task, str(e))
            raise

    def get_configuration(self) -> Optional[SecurityConfiguration]:
        """Get current security configuration."""
        return self._config_manager.get_current_config()

    def apply_configuration(
        self,
        config_id: str,
        dry_run: bool = False,
    ) -> List[ConfigurationChange]:
        """Apply a configuration."""
        return self._config_manager.apply_configuration(config_id, dry_run)

    # === Key Management ===

    def generate_key(
        self,
        key_type: KeyType,
        tags: Optional[Dict[str, str]] = None,
    ) -> KeyMetadata:
        """
        Generate a new cryptographic key.

        Args:
            key_type: Type of key
            tags: Optional tags

        Returns:
            KeyMetadata
        """
        return self._key_manager.generate_key(key_type, tags=tags)

    def rotate_key(
        self,
        key_id: str,
        reason: str = "Manual rotation",
    ) -> Any:
        """Rotate a key."""
        return self._key_manager.rotate_key(key_id, reason, automatic=False)

    def get_expiring_keys(self, days: int = 14) -> List[KeyMetadata]:
        """Get keys expiring within specified days."""
        return self._key_manager.get_expiring_keys(days)

    def get_key(self, key_id: str) -> Optional[KeyMetadata]:
        """Get key metadata."""
        return self._key_manager.get_key(key_id)

    def list_keys(
        self,
        key_type: Optional[KeyType] = None,
        state: Optional[KeyState] = None,
    ) -> List[KeyMetadata]:
        """List keys with optional filtering."""
        return self._key_manager.list_keys(key_type, state)

    # === Certificate Management ===

    def issue_certificate(
        self,
        common_name: str,
        cert_type: CertificateType,
        subject_alt_names: Optional[List[str]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Certificate:
        """
        Issue a new certificate.

        Args:
            common_name: Certificate common name
            cert_type: Certificate type
            subject_alt_names: Subject alternative names
            tags: Optional tags

        Returns:
            Certificate
        """
        task = self._create_task(
            "certificate_issuance",
            f"Issue certificate for {common_name}",
        )

        try:
            # Create request
            request = self._cert_automation.create_certificate_request(
                common_name=common_name,
                cert_type=cert_type,
                subject_alt_names=subject_alt_names,
            )

            # Auto-approve for automation engine (policy-based approval)
            self._cert_automation.approve_request(
                request_id=request.request_id,
                approved_by="automation_engine",
            )

            # Issue certificate
            cert = self._cert_automation.issue_certificate(
                request_id=request.request_id,
                tags=tags,
            )

            self._complete_task(task, {"cert_id": cert.cert_id})
            return cert

        except Exception as e:
            self._fail_task(task, str(e))
            raise

    def renew_certificate(
        self,
        cert_id: str,
        reason: str = "Manual renewal",
    ) -> Any:
        """Renew a certificate."""
        return self._cert_automation.renew_certificate(cert_id, reason, automatic=False)

    def get_expiring_certificates(self, days: int = 30) -> List[Certificate]:
        """Get certificates expiring within specified days."""
        return self._cert_automation.get_expiring_certificates(days)

    def get_certificate(self, cert_id: str) -> Optional[Certificate]:
        """Get certificate by ID."""
        return self._cert_automation.get_certificate(cert_id)

    def list_certificates(
        self,
        cert_type: Optional[CertificateType] = None,
        state: Optional[CertificateState] = None,
    ) -> List[Certificate]:
        """List certificates with optional filtering."""
        return self._cert_automation.list_certificates(cert_type, state)

    # === Self-Healing ===

    def register_component(
        self,
        component_id: str,
        component_type: ComponentType,
        name: str,
        health_check: Optional[Callable[[], bool]] = None,
        check_interval: int = 30,
    ) -> Any:
        """
        Register a component for health monitoring.

        Args:
            component_id: Unique component ID
            component_type: Type of component
            name: Human-readable name
            health_check: Health check function
            check_interval: Check interval in seconds

        Returns:
            ComponentState
        """
        return self._self_healing.register_component(
            component_id,
            component_type,
            name,
            health_check,
            check_interval,
        )

    def check_component_health(self, component_id: str) -> Any:
        """Check health of a component."""
        return self._self_healing.check_health(component_id)

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health."""
        return self._self_healing.get_system_health()

    def get_open_incidents(self) -> List[Incident]:
        """Get open incidents."""
        return self._self_healing.get_open_incidents()

    def resolve_incident(
        self,
        incident_id: str,
        root_cause: str = "",
    ) -> bool:
        """Resolve an incident."""
        return self._self_healing.resolve_incident(incident_id, root_cause)

    # === Automation Workflows ===

    def provision_secure_service(
        self,
        service_name: str,
        service_type: str = "api",
        environment: str = "production",
    ) -> Dict[str, Any]:
        """
        Provision a fully secured service.

        This workflow:
        1. Generates encryption keys
        2. Issues TLS certificates
        3. Configures security settings
        4. Registers for health monitoring

        Args:
            service_name: Service name
            service_type: Type of service
            environment: Target environment

        Returns:
            Provisioning result
        """
        task = self._create_task(
            "service_provisioning",
            f"Provision secure service: {service_name}",
        )

        try:
            result = {
                "service_name": service_name,
                "environment": environment,
                "resources": {},
            }

            # Generate encryption key
            dek = self._key_manager.generate_key(
                KeyType.DEK,
                tags={"service": service_name, "env": environment},
            )
            result["resources"]["encryption_key"] = dek.key_id

            # Generate signing key
            signing_key = self._key_manager.generate_key(
                KeyType.SIGNING,
                tags={"service": service_name, "env": environment},
            )
            result["resources"]["signing_key"] = signing_key.key_id

            # Issue TLS certificate
            cert = self.issue_certificate(
                common_name=f"{service_name}.internal",
                cert_type=CertificateType.TLS_SERVER,
                subject_alt_names=[
                    f"{service_name}.internal",
                    f"{service_name}.{environment}.internal",
                ],
                tags={"service": service_name, "env": environment},
            )
            result["resources"]["tls_certificate"] = cert.cert_id

            # Register for monitoring
            self._self_healing.register_component(
                component_id=f"{service_name}_{environment}",
                component_type=ComponentType.API_ENDPOINT,
                name=f"{service_name} ({environment})",
            )
            result["resources"]["monitoring"] = f"{service_name}_{environment}"

            self._complete_task(task, result)

            logger.info(f"Provisioned secure service: {service_name}")

            return result

        except Exception as e:
            self._fail_task(task, str(e))
            raise

    def rotate_all_expiring(
        self,
        key_days: int = 14,
        cert_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Rotate all expiring keys and certificates.

        Args:
            key_days: Days threshold for keys
            cert_days: Days threshold for certificates

        Returns:
            Rotation summary
        """
        task = self._create_task(
            "rotation_sweep",
            f"Rotate expiring keys ({key_days}d) and certs ({cert_days}d)",
        )

        try:
            result = {
                "keys_rotated": [],
                "certs_renewed": [],
                "errors": [],
            }

            # Rotate expiring keys
            for key in self._key_manager.get_expiring_keys(key_days):
                try:
                    event = self._key_manager.rotate_key(
                        key.key_id,
                        "Expiration sweep rotation",
                        automatic=True,
                    )
                    result["keys_rotated"].append({
                        "old_key": key.key_id,
                        "new_key": event.new_key_id,
                    })
                except Exception as e:
                    result["errors"].append({
                        "type": "key_rotation",
                        "key_id": key.key_id,
                        "error": str(e),
                    })

            # Renew expiring certificates
            for cert in self._cert_automation.get_expiring_certificates(cert_days):
                try:
                    event = self._cert_automation.renew_certificate(
                        cert.cert_id,
                        "Expiration sweep renewal",
                        automatic=True,
                    )
                    result["certs_renewed"].append({
                        "old_cert": cert.cert_id,
                        "new_cert": event.new_cert_id,
                    })
                except Exception as e:
                    result["errors"].append({
                        "type": "cert_renewal",
                        "cert_id": cert.cert_id,
                        "error": str(e),
                    })

            self._complete_task(task, result)

            logger.info(
                f"Rotation sweep: {len(result['keys_rotated'])} keys, "
                f"{len(result['certs_renewed'])} certs"
            )

            return result

        except Exception as e:
            self._fail_task(task, str(e))
            raise

    def run_compliance_check(self) -> Dict[str, Any]:
        """
        Run compliance check across all managed resources.

        Returns:
            Compliance check results
        """
        task = self._create_task(
            "compliance_check",
            "Run compliance verification",
        )

        try:
            result = {
                "checked_at": datetime.utcnow().isoformat(),
                "findings": [],
                "summary": {},
            }

            # Check key compliance
            keys = self._key_manager.list_keys()
            pqc_keys = [k for k in keys if "ML-" in k.algorithm]
            classical_keys = [k for k in keys if "ML-" not in k.algorithm]

            result["summary"]["total_keys"] = len(keys)
            result["summary"]["pqc_ready_keys"] = len(pqc_keys)
            result["summary"]["classical_keys"] = len(classical_keys)

            if classical_keys:
                result["findings"].append({
                    "severity": "medium",
                    "type": "pqc_migration",
                    "message": f"{len(classical_keys)} keys using classical algorithms",
                    "recommendation": "Migrate to ML-KEM or ML-DSA",
                })

            # Check certificate compliance
            certs = self._cert_automation.list_certificates()
            pqc_certs = [c for c in certs if "ML-" in c.key_algorithm]
            expiring = self._cert_automation.get_expiring_certificates(30)

            result["summary"]["total_certs"] = len(certs)
            result["summary"]["pqc_ready_certs"] = len(pqc_certs)
            result["summary"]["expiring_soon"] = len(expiring)

            if expiring:
                result["findings"].append({
                    "severity": "high",
                    "type": "cert_expiration",
                    "message": f"{len(expiring)} certificates expiring within 30 days",
                    "recommendation": "Renew certificates immediately",
                })

            # Check system health
            health = self._self_healing.get_system_health()
            result["summary"]["system_health"] = health["status"]

            if health["unhealthy"] > 0:
                result["findings"].append({
                    "severity": "critical",
                    "type": "system_health",
                    "message": f"{health['unhealthy']} components unhealthy",
                    "recommendation": "Investigate and resolve incidents",
                })

            open_incidents = self._self_healing.get_open_incidents()
            result["summary"]["open_incidents"] = len(open_incidents)

            self._complete_task(task, result)

            return result

        except Exception as e:
            self._fail_task(task, str(e))
            raise

    # === Internal Methods ===

    def _register_internal_components(self) -> None:
        """Register internal components for monitoring."""
        # Register key manager
        self._self_healing.register_component(
            "key_lifecycle_manager",
            ComponentType.KEY_SERVICE,
            "Key Lifecycle Manager",
            health_check=lambda: True,  # Simple alive check
        )

        # Register cert automation
        self._self_healing.register_component(
            "cert_automation",
            ComponentType.CERT_SERVICE,
            "Certificate Automation",
            health_check=lambda: True,
        )

        # Register config manager
        self._self_healing.register_component(
            "config_manager",
            ComponentType.EXTERNAL_SERVICE,
            "Configuration Manager",
            health_check=lambda: True,
        )

    def _create_task(
        self,
        task_type: str,
        description: str,
    ) -> AutomationTask:
        """Create an automation task."""
        task = AutomationTask(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            description=description,
            created_at=datetime.utcnow(),
            status="running",
        )
        self._tasks[task.task_id] = task
        return task

    def _complete_task(
        self,
        task: AutomationTask,
        result: Dict[str, Any],
    ) -> None:
        """Complete a task successfully."""
        task.status = "completed"
        task.completed_at = datetime.utcnow()
        task.result = result

    def _fail_task(
        self,
        task: AutomationTask,
        error: str,
    ) -> None:
        """Mark a task as failed."""
        task.status = "failed"
        task.completed_at = datetime.utcnow()
        task.error = error

    def get_status(self) -> AutomationStatus:
        """Get automation engine status."""
        return AutomationStatus(
            status="running" if self._started_at else "stopped",
            started_at=self._started_at,
            components={
                "adapter_generation": self._config.enable_adapter_generation,
                "config_management": self._config.enable_config_management,
                "key_lifecycle": self._config.enable_key_lifecycle,
                "cert_automation": self._config.enable_cert_automation,
                "self_healing": self._config.enable_self_healing,
            },
            stats={
                "total_keys": len(self._key_manager.list_keys()),
                "total_certs": len(self._cert_automation.list_certificates()),
                "open_incidents": len(self._self_healing.get_open_incidents()),
                "tasks_completed": sum(
                    1 for t in self._tasks.values() if t.status == "completed"
                ),
                "tasks_failed": sum(
                    1 for t in self._tasks.values() if t.status == "failed"
                ),
            },
        )

    def get_task_history(
        self,
        task_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[AutomationTask]:
        """Get task history."""
        tasks = list(self._tasks.values())

        if task_type:
            tasks = [t for t in tasks if t.task_type == task_type]

        # Sort by creation time, newest first
        tasks.sort(key=lambda t: t.created_at, reverse=True)

        return tasks[:limit]


# Convenience function for quick automation setup
def create_automation_engine(
    hsm_provider: Optional[Any] = None,
    ca_provider: Optional[Any] = None,
    auto_start: bool = True,
) -> AutomationEngine:
    """
    Create and optionally start an automation engine.

    Args:
        hsm_provider: Optional HSM provider
        ca_provider: Optional CA provider
        auto_start: Whether to start automatically

    Returns:
        AutomationEngine
    """
    engine = AutomationEngine(
        hsm_provider=hsm_provider,
        ca_provider=ca_provider,
    )

    if auto_start:
        engine.start()

    return engine
