"""
CRONOS AI Engine - Resilience Manager

Unified resilience management system that orchestrates circuit breakers,
retry policies, bulkhead isolation, health checking, timeout management,
and error recovery for the Zero-Touch Security Orchestrator.
"""

import asyncio
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerManager, 
    get_circuit_breaker_manager
)
from .retry_policy import (
    RetryPolicy, RetryManager, ExponentialBackoff, CommonRetryPolicies,
    get_retry_manager
)
from .bulkhead import (
    BulkheadManager, ResourcePool, AllocationStrategy, ResourceContext,
    get_bulkhead_manager
)
from .health_checker import (
    HealthChecker, HealthCheck, BasicHealthCheck, HTTPHealthCheck,
    HealthCheckConfig, CheckType, HealthStatus, get_health_checker
)
from .timeout_manager import (
    TimeoutManager, TimeoutPolicy, TimeoutConfig, TimeoutStrategy,
    get_timeout_manager
)
from .error_recovery import (
    ErrorRecoveryManager, RecoveryPlan, RecoveryAction, RecoveryStrategy,
    ErrorSeverity, get_error_recovery_manager
)

from ..config import get_security_config
from ..logging import get_security_logger, SecurityLogType, LogLevel


class ResilienceLevel(str, Enum):
    """Resilience configuration levels."""
    MINIMAL = "minimal"      # Basic resilience features
    STANDARD = "standard"    # Standard enterprise resilience
    HIGH = "high"           # High availability resilience
    MAXIMUM = "maximum"     # Maximum resilience with all features


@dataclass
class ResilienceConfig:
    """Configuration for resilience management."""
    level: ResilienceLevel = ResilienceLevel.STANDARD
    enable_circuit_breakers: bool = True
    enable_retry_policies: bool = True
    enable_bulkhead_isolation: bool = True
    enable_health_checking: bool = True
    enable_timeout_management: bool = True
    enable_error_recovery: bool = True
    
    # Global settings
    default_timeout: float = 30.0
    default_retries: int = 3
    default_circuit_breaker_threshold: int = 5
    health_check_interval: float = 60.0
    
    # Performance settings
    max_concurrent_operations: int = 1000
    resource_pool_sizes: Dict[str, int] = field(default_factory=lambda: {
        'api_calls': 100,
        'database_connections': 50,
        'file_operations': 25,
        'network_requests': 200
    })


class ResilienceManager:
    """
    Unified resilience manager that orchestrates all resilience patterns
    and provides a single interface for resilient operations.
    """
    
    def __init__(self, config: Optional[ResilienceConfig] = None):
        self.config = config or ResilienceConfig()
        self.logger = get_security_logger("cronos.security.resilience.manager")
        
        # Component managers
        self._circuit_breaker_manager: Optional[CircuitBreakerManager] = None
        self._retry_manager: Optional[RetryManager] = None
        self._bulkhead_manager: Optional[BulkheadManager] = None
        self._health_checker: Optional[HealthChecker] = None
        self._timeout_manager: Optional[TimeoutManager] = None
        self._error_recovery_manager: Optional[ErrorRecoveryManager] = None
        
        # State management
        self._initialized = False
        self._running = False
        self._background_tasks: List[asyncio.Task] = []
        
        # Metrics
        self._start_time = datetime.utcnow()
        self._total_operations = 0
        self._successful_operations = 0
        self._failed_operations = 0
        self._recovered_operations = 0
        
        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            "Resilience Manager initialized",
            level=LogLevel.INFO,
            metadata={
                'level': self.config.level.value,
                'circuit_breakers': self.config.enable_circuit_breakers,
                'retry_policies': self.config.enable_retry_policies,
                'bulkhead_isolation': self.config.enable_bulkhead_isolation,
                'health_checking': self.config.enable_health_checking,
                'timeout_management': self.config.enable_timeout_management,
                'error_recovery': self.config.enable_error_recovery
            }
        )
    
    @property
    def circuit_breaker_manager(self) -> CircuitBreakerManager:
        """Get circuit breaker manager."""
        if not self._circuit_breaker_manager:
            self._circuit_breaker_manager = get_circuit_breaker_manager()
        return self._circuit_breaker_manager
    
    @property
    def retry_manager(self) -> RetryManager:
        """Get retry manager."""
        if not self._retry_manager:
            self._retry_manager = get_retry_manager()
        return self._retry_manager
    
    @property
    def bulkhead_manager(self) -> BulkheadManager:
        """Get bulkhead manager."""
        if not self._bulkhead_manager:
            self._bulkhead_manager = get_bulkhead_manager()
        return self._bulkhead_manager
    
    @property
    def health_checker(self) -> HealthChecker:
        """Get health checker."""
        if not self._health_checker:
            self._health_checker = get_health_checker()
        return self._health_checker
    
    @property
    def timeout_manager(self) -> TimeoutManager:
        """Get timeout manager."""
        if not self._timeout_manager:
            self._timeout_manager = get_timeout_manager()
        return self._timeout_manager
    
    @property
    def error_recovery_manager(self) -> ErrorRecoveryManager:
        """Get error recovery manager."""
        if not self._error_recovery_manager:
            self._error_recovery_manager = get_error_recovery_manager()
        return self._error_recovery_manager
    
    async def initialize(self):
        """Initialize all resilience components."""
        if self._initialized:
            return
        
        try:
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                "Initializing resilience components",
                level=LogLevel.INFO
            )
            
            # Initialize components based on configuration
            if self.config.enable_circuit_breakers:
                await self._initialize_circuit_breakers()
            
            if self.config.enable_retry_policies:
                await self._initialize_retry_policies()
            
            if self.config.enable_bulkhead_isolation:
                await self._initialize_bulkhead_isolation()
            
            if self.config.enable_health_checking:
                await self._initialize_health_checking()
            
            if self.config.enable_timeout_management:
                await self._initialize_timeout_management()
            
            if self.config.enable_error_recovery:
                await self._initialize_error_recovery()
            
            # Start background monitoring
            await self._start_background_monitoring()
            
            self._initialized = True
            self._running = True
            
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                "Resilience Manager initialization complete",
                level=LogLevel.INFO,
                metadata={'resilience_level': self.config.level.value}
            )
            
        except Exception as e:
            self.logger.log_security_event(
                SecurityLogType.CONFIGURATION_CHANGE,
                f"Resilience Manager initialization failed: {str(e)}",
                level=LogLevel.ERROR,
                error_code="RESILIENCE_INIT_FAILED"
            )
            raise
    
    async def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for common operations."""
        
        # Create circuit breakers for different operation types
        circuit_breakers = {
            'api_calls': CircuitBreakerConfig(
                failure_threshold=self.config.default_circuit_breaker_threshold,
                recovery_timeout=60,
                success_threshold=3,
                timeout=self.config.default_timeout
            ),
            'database_operations': CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=120,
                success_threshold=2,
                timeout=45.0
            ),
            'external_integrations': CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=90,
                success_threshold=3,
                timeout=30.0
            ),
            'security_operations': CircuitBreakerConfig(
                failure_threshold=2,
                recovery_timeout=30,
                success_threshold=1,
                timeout=60.0
            )
        }
        
        for name, config in circuit_breakers.items():
            self.circuit_breaker_manager.get_circuit_breaker(name, config)
        
        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            f"Initialized {len(circuit_breakers)} circuit breakers",
            level=LogLevel.INFO
        )
    
    async def _initialize_retry_policies(self):
        """Initialize retry policies for different scenarios."""
        
        # Create retry policies
        policies = {
            'fast_operations': {
                'max_attempts': 3,
                'base_delay': 0.5,
                'backoff_strategy': ExponentialBackoff(multiplier=1.5, max_delay=5.0)
            },
            'standard_operations': {
                'max_attempts': self.config.default_retries,
                'base_delay': 1.0,
                'backoff_strategy': ExponentialBackoff(multiplier=2.0, max_delay=30.0)
            },
            'slow_operations': {
                'max_attempts': 5,
                'base_delay': 5.0,
                'backoff_strategy': ExponentialBackoff(multiplier=2.0, max_delay=120.0)
            },
            'critical_operations': {
                'max_attempts': 5,
                'base_delay': 2.0,
                'backoff_strategy': ExponentialBackoff(multiplier=1.8, max_delay=60.0)
            }
        }
        
        for name, config in policies.items():
            self.retry_manager.create_policy(name, **config)
        
        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            f"Initialized {len(policies)} retry policies",
            level=LogLevel.INFO
        )
    
    async def _initialize_bulkhead_isolation(self):
        """Initialize bulkhead resource pools."""
        
        # Create resource pools based on configuration
        for pool_name, capacity in self.config.resource_pool_sizes.items():
            if capacity > 0:
                self.bulkhead_manager.create_pool(
                    name=pool_name,
                    capacity=capacity,
                    allocation_strategy=AllocationStrategy.FIFO
                )
        
        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            f"Initialized {len(self.config.resource_pool_sizes)} resource pools",
            level=LogLevel.INFO
        )
    
    async def _initialize_health_checking(self):
        """Initialize health checking system."""
        
        # Create basic system health checks
        health_checks = [
            {
                'name': 'system_health',
                'check_type': CheckType.LIVENESS,
                'interval': self.config.health_check_interval,
                'check_function': self._check_system_health
            },
            {
                'name': 'memory_usage',
                'check_type': CheckType.PERFORMANCE,
                'interval': 30.0,
                'check_function': self._check_memory_usage
            },
            {
                'name': 'resilience_components',
                'check_type': CheckType.READINESS,
                'interval': 60.0,
                'check_function': self._check_resilience_components
            }
        ]
        
        for check_config in health_checks:
            name = check_config.pop('name')
            check_function = check_config.pop('check_function')
            
            config = HealthCheckConfig(name=name, **check_config)
            health_check = BasicHealthCheck(config, check_function)
            self.health_checker.register_check(health_check)
        
        # Start health monitoring
        await self.health_checker.start_monitoring()
        
        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            f"Initialized {len(health_checks)} health checks",
            level=LogLevel.INFO
        )
    
    async def _initialize_timeout_management(self):
        """Initialize timeout management."""
        
        # Timeout policies are created automatically by TimeoutManager
        # We can customize them here if needed
        
        # Create custom timeout policies for security operations
        security_timeout_config = {
            'default_timeout': 60.0,
            'strategy': TimeoutStrategy.ADAPTIVE,
            'min_timeout': 10.0,
            'max_timeout': 180.0
        }
        
        self.timeout_manager.create_policy(
            name="security_operations",
            **security_timeout_config
        )
        
        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            "Initialized timeout management",
            level=LogLevel.INFO
        )
    
    async def _initialize_error_recovery(self):
        """Initialize error recovery system."""
        
        # Error recovery patterns are created automatically by ErrorRecoveryManager
        # We can add custom recovery plans here if needed
        
        # Create security-specific recovery plan
        from .error_recovery import RecoveryPlan, RecoveryAction, RecoveryStrategy
        
        security_recovery_plan = RecoveryPlan(
            name="security_operations",
            actions=[
                RecoveryAction(strategy=RecoveryStrategy.RETRY),
                RecoveryAction(strategy=RecoveryStrategy.DEGRADED_MODE),
                RecoveryAction(strategy=RecoveryStrategy.COMPENSATE)
            ],
            max_recovery_attempts=3,
            recovery_timeout=120.0
        )
        
        self.error_recovery_manager.register_recovery_plan(security_recovery_plan)
        self.error_recovery_manager.register_error_pattern("SecurityOperationError", "security_operations")
        
        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            "Initialized error recovery system",
            level=LogLevel.INFO
        )
    
    async def _start_background_monitoring(self):
        """Start background monitoring tasks."""
        
        # Resilience metrics monitoring task
        metrics_task = asyncio.create_task(self._background_metrics_monitoring())
        self._background_tasks.append(metrics_task)
        
        # Component health monitoring task
        health_task = asyncio.create_task(self._background_component_health_monitoring())
        self._background_tasks.append(health_task)
        
        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            f"Started {len(self._background_tasks)} background monitoring tasks",
            level=LogLevel.INFO
        )
    
    async def execute_resilient_operation(
        self,
        operation: Callable,
        operation_name: str,
        *args,
        circuit_breaker_name: Optional[str] = None,
        retry_policy_name: Optional[str] = None,
        timeout_policy_name: Optional[str] = None,
        resource_pool_name: Optional[str] = None,
        enable_error_recovery: bool = True,
        **kwargs
    ) -> Any:
        """
        Execute an operation with full resilience capabilities.
        
        Args:
            operation: Function to execute
            operation_name: Name of the operation for logging
            *args: Operation arguments
            circuit_breaker_name: Circuit breaker to use
            retry_policy_name: Retry policy to use
            timeout_policy_name: Timeout policy to use
            resource_pool_name: Resource pool to use
            enable_error_recovery: Whether to enable error recovery
            **kwargs: Operation keyword arguments
            
        Returns:
            Operation result
        """
        
        self._total_operations += 1
        start_time = asyncio.get_event_loop().time()
        
        self.logger.log_security_event(
            SecurityLogType.PERFORMANCE_METRIC,
            f"Starting resilient operation: {operation_name}",
            level=LogLevel.DEBUG,
            metadata={
                'operation_name': operation_name,
                'circuit_breaker': circuit_breaker_name,
                'retry_policy': retry_policy_name,
                'timeout_policy': timeout_policy_name,
                'resource_pool': resource_pool_name
            }
        )
        
        # Resource allocation (bulkhead)
        resource_context = None
        if resource_pool_name and self.config.enable_bulkhead_isolation:
            pool = self.bulkhead_manager.get_pool(resource_pool_name)
            if pool:
                resource_context = ResourceContext(pool, f"{operation_name}_{start_time}")
        
        try:
            # Wrap the operation with resilience patterns
            resilient_operation = operation
            
            # Apply timeout management
            if timeout_policy_name and self.config.enable_timeout_management:
                async def timeout_wrapped_operation():
                    return await self.timeout_manager.execute_with_policy(
                        timeout_policy_name,
                        operation,
                        *args,
                        **kwargs
                    )
                resilient_operation = timeout_wrapped_operation
            
            # Apply retry policy
            if retry_policy_name and self.config.enable_retry_policies:
                policy = self.retry_manager.get_policy(retry_policy_name)
                if policy:
                    async def retry_wrapped_operation():
                        result = await policy.execute(resilient_operation, *args, **kwargs)
                        return result.recovered_data if hasattr(result, 'recovered_data') else result
                    resilient_operation = retry_wrapped_operation
            
            # Apply circuit breaker
            if circuit_breaker_name and self.config.enable_circuit_breakers:
                circuit_breaker = self.circuit_breaker_manager.get_circuit_breaker(circuit_breaker_name)
                async def circuit_breaker_wrapped_operation():
                    return await circuit_breaker.call(resilient_operation, *args, **kwargs)
                resilient_operation = circuit_breaker_wrapped_operation
            
            # Execute with resource allocation
            if resource_context:
                async with resource_context as resource_id:
                    if resource_id is not None:
                        result = await resilient_operation()
                    else:
                        raise RuntimeError(f"Failed to acquire resource from pool: {resource_pool_name}")
            else:
                result = await resilient_operation()
            
            # Success
            execution_time = asyncio.get_event_loop().time() - start_time
            self._successful_operations += 1
            
            self.logger.log_security_event(
                SecurityLogType.PERFORMANCE_METRIC,
                f"Resilient operation completed: {operation_name}",
                level=LogLevel.DEBUG,
                metadata={
                    'operation_name': operation_name,
                    'execution_time': execution_time,
                    'success': True
                }
            )
            
            return result
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Attempt error recovery if enabled
            if enable_error_recovery and self.config.enable_error_recovery:
                try:
                    recovery_result = await self.error_recovery_manager.recover_from_error(
                        error=e,
                        operation_name=operation_name,
                        severity=ErrorSeverity.MEDIUM
                    )
                    
                    if recovery_result.success:
                        self._recovered_operations += 1
                        
                        self.logger.log_security_event(
                            SecurityLogType.PERFORMANCE_METRIC,
                            f"Operation recovered after error: {operation_name}",
                            level=LogLevel.INFO,
                            metadata={
                                'operation_name': operation_name,
                                'execution_time': execution_time,
                                'recovery_strategy': recovery_result.strategy_used.value if recovery_result.strategy_used else None
                            }
                        )
                        
                        return recovery_result.recovered_data
                
                except Exception as recovery_error:
                    self.logger.log_security_event(
                        SecurityLogType.PERFORMANCE_METRIC,
                        f"Error recovery failed for operation: {operation_name} - {str(recovery_error)}",
                        level=LogLevel.ERROR,
                        error_code="ERROR_RECOVERY_FAILED"
                    )
            
            # Final failure
            self._failed_operations += 1
            
            self.logger.log_security_event(
                SecurityLogType.PERFORMANCE_METRIC,
                f"Resilient operation failed: {operation_name} - {str(e)}",
                level=LogLevel.ERROR,
                metadata={
                    'operation_name': operation_name,
                    'execution_time': execution_time,
                    'error': str(e),
                    'error_type': type(e).__name__
                },
                error_code="RESILIENT_OPERATION_FAILED"
            )
            
            raise
    
    async def _check_system_health(self) -> bool:
        """Basic system health check."""
        try:
            # Basic system checks
            import psutil
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                return "degraded"
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 85:
                return "degraded"
            
            return True
        except Exception:
            return False
    
    async def _check_memory_usage(self) -> bool:
        """Memory usage health check."""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            if memory.percent > 95:
                return False
            elif memory.percent > 80:
                return "degraded"
            else:
                return True
        except Exception:
            return False
    
    async def _check_resilience_components(self) -> bool:
        """Check health of resilience components."""
        try:
            healthy_components = 0
            total_components = 0
            
            # Check circuit breakers
            if self.config.enable_circuit_breakers:
                total_components += 1
                cb_metrics = self.circuit_breaker_manager.get_all_metrics()
                open_breakers = sum(1 for cb in cb_metrics.values() if cb['state'] == 'open')
                if open_breakers == 0:
                    healthy_components += 1
            
            # Check resource pools
            if self.config.enable_bulkhead_isolation:
                total_components += 1
                pool_metrics = self.bulkhead_manager.get_global_metrics()
                if pool_metrics['global_utilization'] < 0.9:
                    healthy_components += 1
            
            # Health check itself
            if self.config.enable_health_checking:
                total_components += 1
                overall_health = self.health_checker.get_overall_health()
                if overall_health in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
                    healthy_components += 1
            
            if total_components == 0:
                return True
            
            health_ratio = healthy_components / total_components
            if health_ratio >= 0.8:
                return True
            elif health_ratio >= 0.6:
                return "degraded"
            else:
                return False
                
        except Exception:
            return False
    
    async def _background_metrics_monitoring(self):
        """Background task for monitoring resilience metrics."""
        
        while self._running:
            try:
                # Log periodic metrics
                metrics = self.get_global_metrics()
                
                self.logger.log_security_event(
                    SecurityLogType.PERFORMANCE_METRIC,
                    "Resilience metrics update",
                    level=LogLevel.INFO,
                    metadata=metrics
                )
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.log_security_event(
                    SecurityLogType.PERFORMANCE_METRIC,
                    f"Metrics monitoring error: {str(e)}",
                    level=LogLevel.ERROR
                )
                await asyncio.sleep(60)
    
    async def _background_component_health_monitoring(self):
        """Background task for monitoring component health."""
        
        while self._running:
            try:
                # Check component health and take corrective actions
                
                # Check circuit breakers
                if self.config.enable_circuit_breakers:
                    cb_metrics = self.circuit_breaker_manager.get_all_metrics()
                    open_breakers = [name for name, metrics in cb_metrics.items() if metrics['state'] == 'open']
                    
                    if open_breakers:
                        self.logger.log_security_event(
                            SecurityLogType.PERFORMANCE_METRIC,
                            f"Open circuit breakers detected: {len(open_breakers)}",
                            level=LogLevel.WARNING,
                            metadata={'open_breakers': open_breakers}
                        )
                
                # Check resource pools
                if self.config.enable_bulkhead_isolation:
                    pool_metrics = self.bulkhead_manager.get_global_metrics()
                    if pool_metrics['global_utilization'] > 0.9:
                        self.logger.log_security_event(
                            SecurityLogType.PERFORMANCE_METRIC,
                            "High resource pool utilization detected",
                            level=LogLevel.WARNING,
                            metadata={'utilization': pool_metrics['global_utilization']}
                        )
                
                await asyncio.sleep(60)  # Every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.log_security_event(
                    SecurityLogType.PERFORMANCE_METRIC,
                    f"Component health monitoring error: {str(e)}",
                    level=LogLevel.ERROR
                )
                await asyncio.sleep(30)
    
    def get_global_metrics(self) -> Dict[str, Any]:
        """Get comprehensive resilience metrics."""
        
        uptime = datetime.utcnow() - self._start_time
        
        base_metrics = {
            'resilience_level': self.config.level.value,
            'uptime_seconds': uptime.total_seconds(),
            'initialized': self._initialized,
            'running': self._running,
            'total_operations': self._total_operations,
            'successful_operations': self._successful_operations,
            'failed_operations': self._failed_operations,
            'recovered_operations': self._recovered_operations,
            'success_rate': self._successful_operations / self._total_operations if self._total_operations > 0 else 0,
            'recovery_rate': self._recovered_operations / self._failed_operations if self._failed_operations > 0 else 0,
            'enabled_components': {
                'circuit_breakers': self.config.enable_circuit_breakers,
                'retry_policies': self.config.enable_retry_policies,
                'bulkhead_isolation': self.config.enable_bulkhead_isolation,
                'health_checking': self.config.enable_health_checking,
                'timeout_management': self.config.enable_timeout_management,
                'error_recovery': self.config.enable_error_recovery
            }
        }
        
        # Add component-specific metrics
        if self.config.enable_circuit_breakers:
            base_metrics['circuit_breakers'] = self.circuit_breaker_manager.get_all_metrics()
        
        if self.config.enable_retry_policies:
            base_metrics['retry_policies'] = self.retry_manager.get_all_stats()
        
        if self.config.enable_bulkhead_isolation:
            base_metrics['bulkhead'] = self.bulkhead_manager.get_global_metrics()
        
        if self.config.enable_health_checking:
            base_metrics['health_checks'] = self.health_checker.get_health_summary()
        
        if self.config.enable_timeout_management:
            base_metrics['timeout_management'] = self.timeout_manager.get_global_metrics()
        
        if self.config.enable_error_recovery:
            base_metrics['error_recovery'] = self.error_recovery_manager.get_recovery_metrics()
        
        return base_metrics
    
    async def shutdown(self):
        """Shutdown the resilience manager and all components."""
        
        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            "Shutting down Resilience Manager",
            level=LogLevel.INFO
        )
        
        self._running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Shutdown components
        if self.config.enable_health_checking and self._health_checker:
            await self.health_checker.stop_monitoring()
        
        if self.config.enable_bulkhead_isolation and self._bulkhead_manager:
            await self.bulkhead_manager.shutdown()
        
        self.logger.log_security_event(
            SecurityLogType.CONFIGURATION_CHANGE,
            "Resilience Manager shutdown complete",
            level=LogLevel.INFO
        )


# Global resilience manager instance
_resilience_manager: Optional[ResilienceManager] = None

def get_resilience_manager(config: Optional[ResilienceConfig] = None) -> ResilienceManager:
    """Get global resilience manager instance."""
    global _resilience_manager
    if _resilience_manager is None:
        _resilience_manager = ResilienceManager(config)
    return _resilience_manager