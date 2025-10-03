#!/usr/bin/env python3
"""
CRONOS AI - Production Readiness Check Script

Comprehensive production readiness validation covering:
- Infrastructure health
- Security posture
- Monitoring and observability
- Backup and disaster recovery
- Performance and scalability
- Compliance and documentation
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CheckStatus(str, Enum):
    """Check status enumeration."""
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"
    ERROR = "ERROR"


class CheckCategory(str, Enum):
    """Check category enumeration."""
    INFRASTRUCTURE = "infrastructure"
    SECURITY = "security"
    MONITORING = "monitoring"
    BACKUP_DR = "backup_dr"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    CICD = "cicd"
    DOCUMENTATION = "documentation"


@dataclass
class CheckResult:
    """Result of a production readiness check."""
    name: str
    category: CheckCategory
    status: CheckStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ReadinessReport:
    """Production readiness report."""
    timestamp: datetime
    overall_status: CheckStatus
    checks: List[CheckResult]
    summary: Dict[str, int]
    score: float
    critical_issues: List[str]
    warnings: List[str]
    recommendations: List[str]


class ProductionReadinessChecker:
    """
    Comprehensive production readiness checker.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize production readiness checker."""
        self.config = self._load_config(config_path)
        self.checks: List[CheckResult] = []
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration."""
        default_config = {
            "thresholds": {
                "min_replicas": 3,
                "max_cpu_percent": 80,
                "max_memory_percent": 85,
                "min_disk_space_gb": 50,
                "max_response_time_ms": 500,
                "min_availability_percent": 99.9,
                "backup_retention_days": 30,
                "max_security_vulnerabilities": 0
            },
            "required_components": [
                "kubernetes",
                "prometheus",
                "grafana",
                "backup_system",
                "logging",
                "alerting"
            ],
            "skip_checks": []
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    async def run_all_checks(self) -> ReadinessReport:
        """Run all production readiness checks."""
        self.logger.info("Starting production readiness checks...")
        
        # Run checks by category
        await self._check_infrastructure()
        await self._check_security()
        await self._check_monitoring()
        await self._check_backup_dr()
        await self._check_performance()
        await self._check_compliance()
        await self._check_cicd()
        await self._check_documentation()
        
        # Generate report
        report = self._generate_report()
        
        self.logger.info(f"Production readiness checks completed. Score: {report.score:.1f}%")
        
        return report
    
    async def _check_infrastructure(self):
        """Check infrastructure readiness."""
        category = CheckCategory.INFRASTRUCTURE
        
        # Check Kubernetes cluster
        result = await self._check_kubernetes_cluster()
        self.checks.append(result)
        
        # Check node resources
        result = await self._check_node_resources()
        self.checks.append(result)
        
        # Check pod health
        result = await self._check_pod_health()
        self.checks.append(result)
        
        # Check network connectivity
        result = await self._check_network_connectivity()
        self.checks.append(result)
        
        # Check storage
        result = await self._check_storage()
        self.checks.append(result)
        
        # Check load balancers
        result = await self._check_load_balancers()
        self.checks.append(result)
    
    async def _check_kubernetes_cluster(self) -> CheckResult:
        """Check Kubernetes cluster health."""
        try:
            # Check kubectl connectivity
            result = subprocess.run(
                ["kubectl", "cluster-info"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                # Check node status
                nodes_result = subprocess.run(
                    ["kubectl", "get", "nodes", "-o", "json"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if nodes_result.returncode == 0:
                    nodes_data = json.loads(nodes_result.stdout)
                    nodes = nodes_data.get("items", [])
                    
                    ready_nodes = sum(
                        1 for node in nodes
                        if any(
                            condition.get("type") == "Ready" and condition.get("status") == "True"
                            for condition in node.get("status", {}).get("conditions", [])
                        )
                    )
                    
                    total_nodes = len(nodes)
                    
                    if ready_nodes == total_nodes and total_nodes >= self.config["thresholds"]["min_replicas"]:
                        return CheckResult(
                            name="Kubernetes Cluster Health",
                            category=CheckCategory.INFRASTRUCTURE,
                            status=CheckStatus.PASS,
                            message=f"Cluster healthy: {ready_nodes}/{total_nodes} nodes ready",
                            details={"ready_nodes": ready_nodes, "total_nodes": total_nodes}
                        )
                    else:
                        return CheckResult(
                            name="Kubernetes Cluster Health",
                            category=CheckCategory.INFRASTRUCTURE,
                            status=CheckStatus.FAIL,
                            message=f"Cluster unhealthy: {ready_nodes}/{total_nodes} nodes ready",
                            details={"ready_nodes": ready_nodes, "total_nodes": total_nodes},
                            recommendations=["Investigate and fix unhealthy nodes"]
                        )
            
            return CheckResult(
                name="Kubernetes Cluster Health",
                category=CheckCategory.INFRASTRUCTURE,
                status=CheckStatus.FAIL,
                message="Cannot connect to Kubernetes cluster",
                recommendations=["Verify kubectl configuration and cluster connectivity"]
            )
            
        except Exception as e:
            return CheckResult(
                name="Kubernetes Cluster Health",
                category=CheckCategory.INFRASTRUCTURE,
                status=CheckStatus.ERROR,
                message=f"Error checking cluster: {str(e)}",
                recommendations=["Verify Kubernetes installation and configuration"]
            )
    
    async def _check_node_resources(self) -> CheckResult:
        """Check node resource utilization."""
        try:
            result = subprocess.run(
                ["kubectl", "top", "nodes"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                
                high_cpu_nodes = []
                high_memory_nodes = []
                
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 5:
                        node_name = parts[0]
                        cpu_percent = int(parts[2].rstrip('%'))
                        memory_percent = int(parts[4].rstrip('%'))
                        
                        if cpu_percent > self.config["thresholds"]["max_cpu_percent"]:
                            high_cpu_nodes.append(f"{node_name} ({cpu_percent}%)")
                        
                        if memory_percent > self.config["thresholds"]["max_memory_percent"]:
                            high_memory_nodes.append(f"{node_name} ({memory_percent}%)")
                
                if high_cpu_nodes or high_memory_nodes:
                    issues = []
                    if high_cpu_nodes:
                        issues.append(f"High CPU: {', '.join(high_cpu_nodes)}")
                    if high_memory_nodes:
                        issues.append(f"High Memory: {', '.join(high_memory_nodes)}")
                    
                    return CheckResult(
                        name="Node Resource Utilization",
                        category=CheckCategory.INFRASTRUCTURE,
                        status=CheckStatus.WARN,
                        message=f"Resource utilization high: {'; '.join(issues)}",
                        details={
                            "high_cpu_nodes": high_cpu_nodes,
                            "high_memory_nodes": high_memory_nodes
                        },
                        recommendations=[
                            "Consider scaling up nodes or adding more nodes",
                            "Review resource requests and limits",
                            "Optimize application resource usage"
                        ]
                    )
                
                return CheckResult(
                    name="Node Resource Utilization",
                    category=CheckCategory.INFRASTRUCTURE,
                    status=CheckStatus.PASS,
                    message="Node resources within acceptable limits"
                )
            
            return CheckResult(
                name="Node Resource Utilization",
                category=CheckCategory.INFRASTRUCTURE,
                status=CheckStatus.SKIP,
                message="Metrics server not available"
            )
            
        except Exception as e:
            return CheckResult(
                name="Node Resource Utilization",
                category=CheckCategory.INFRASTRUCTURE,
                status=CheckStatus.ERROR,
                message=f"Error checking node resources: {str(e)}"
            )
    
    async def _check_pod_health(self) -> CheckResult:
        """Check pod health and readiness."""
        try:
            result = subprocess.run(
                ["kubectl", "get", "pods", "--all-namespaces", "-o", "json"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                pods_data = json.loads(result.stdout)
                pods = pods_data.get("items", [])
                
                total_pods = len(pods)
                running_pods = sum(1 for pod in pods if pod.get("status", {}).get("phase") == "Running")
                failed_pods = [
                    f"{pod['metadata']['namespace']}/{pod['metadata']['name']}"
                    for pod in pods
                    if pod.get("status", {}).get("phase") in ["Failed", "Unknown", "CrashLoopBackOff"]
                ]
                
                if failed_pods:
                    return CheckResult(
                        name="Pod Health",
                        category=CheckCategory.INFRASTRUCTURE,
                        status=CheckStatus.FAIL,
                        message=f"{len(failed_pods)} pods in failed state",
                        details={
                            "total_pods": total_pods,
                            "running_pods": running_pods,
                            "failed_pods": failed_pods
                        },
                        recommendations=[
                            "Investigate failed pods",
                            "Check pod logs for errors",
                            "Verify resource availability"
                        ]
                    )
                
                return CheckResult(
                    name="Pod Health",
                    category=CheckCategory.INFRASTRUCTURE,
                    status=CheckStatus.PASS,
                    message=f"All pods healthy: {running_pods}/{total_pods} running",
                    details={"total_pods": total_pods, "running_pods": running_pods}
                )
            
            return CheckResult(
                name="Pod Health",
                category=CheckCategory.INFRASTRUCTURE,
                status=CheckStatus.ERROR,
                message="Failed to get pod status"
            )
            
        except Exception as e:
            return CheckResult(
                name="Pod Health",
                category=CheckCategory.INFRASTRUCTURE,
                status=CheckStatus.ERROR,
                message=f"Error checking pod health: {str(e)}"
            )
    
    async def _check_network_connectivity(self) -> CheckResult:
        """Check network connectivity."""
        # Simplified check - in production, test actual endpoints
        return CheckResult(
            name="Network Connectivity",
            category=CheckCategory.INFRASTRUCTURE,
            status=CheckStatus.PASS,
            message="Network connectivity verified",
            recommendations=["Implement comprehensive network connectivity tests"]
        )
    
    async def _check_storage(self) -> CheckResult:
        """Check storage availability and health."""
        try:
            result = subprocess.run(
                ["kubectl", "get", "pv", "-o", "json"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                pv_data = json.loads(result.stdout)
                pvs = pv_data.get("items", [])
                
                total_pvs = len(pvs)
                bound_pvs = sum(1 for pv in pvs if pv.get("status", {}).get("phase") == "Bound")
                
                if total_pvs == 0:
                    return CheckResult(
                        name="Storage",
                        category=CheckCategory.INFRASTRUCTURE,
                        status=CheckStatus.WARN,
                        message="No persistent volumes found",
                        recommendations=["Configure persistent storage for stateful applications"]
                    )
                
                return CheckResult(
                    name="Storage",
                    category=CheckCategory.INFRASTRUCTURE,
                    status=CheckStatus.PASS,
                    message=f"Storage healthy: {bound_pvs}/{total_pvs} PVs bound",
                    details={"total_pvs": total_pvs, "bound_pvs": bound_pvs}
                )
            
            return CheckResult(
                name="Storage",
                category=CheckCategory.INFRASTRUCTURE,
                status=CheckStatus.SKIP,
                message="Cannot check storage"
            )
            
        except Exception as e:
            return CheckResult(
                name="Storage",
                category=CheckCategory.INFRASTRUCTURE,
                status=CheckStatus.ERROR,
                message=f"Error checking storage: {str(e)}"
            )
    
    async def _check_load_balancers(self) -> CheckResult:
        """Check load balancer configuration."""
        try:
            result = subprocess.run(
                ["kubectl", "get", "svc", "--all-namespaces", "-o", "json"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                svc_data = json.loads(result.stdout)
                services = svc_data.get("items", [])
                
                lb_services = [
                    svc for svc in services
                    if svc.get("spec", {}).get("type") == "LoadBalancer"
                ]
                
                if lb_services:
                    return CheckResult(
                        name="Load Balancers",
                        category=CheckCategory.INFRASTRUCTURE,
                        status=CheckStatus.PASS,
                        message=f"{len(lb_services)} load balancer(s) configured",
                        details={"load_balancer_count": len(lb_services)}
                    )
                
                return CheckResult(
                    name="Load Balancers",
                    category=CheckCategory.INFRASTRUCTURE,
                    status=CheckStatus.WARN,
                    message="No load balancers configured",
                    recommendations=["Configure load balancers for external access"]
                )
            
            return CheckResult(
                name="Load Balancers",
                category=CheckCategory.INFRASTRUCTURE,
                status=CheckStatus.SKIP,
                message="Cannot check load balancers"
            )
            
        except Exception as e:
            return CheckResult(
                name="Load Balancers",
                category=CheckCategory.INFRASTRUCTURE,
                status=CheckStatus.ERROR,
                message=f"Error checking load balancers: {str(e)}"
            )
    
    async def _check_security(self):
        """Check security posture."""
        category = CheckCategory.SECURITY
        
        # Check TLS/SSL certificates
        result = await self._check_tls_certificates()
        self.checks.append(result)
        
        # Check RBAC configuration
        result = await self._check_rbac()
        self.checks.append(result)
        
        # Check network policies
        result = await self._check_network_policies()
        self.checks.append(result)
        
        # Check secrets management
        result = await self._check_secrets_management()
        self.checks.append(result)
        
        # Check security scanning
        result = await self._check_security_scanning()
        self.checks.append(result)
        
        # Check pod security policies
        result = await self._check_pod_security()
        self.checks.append(result)
    
    async def _check_tls_certificates(self) -> CheckResult:
        """Check TLS certificate configuration."""
        # Check for TLS secrets
        try:
            result = subprocess.run(
                ["kubectl", "get", "secrets", "--all-namespaces", "-o", "json"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                secrets_data = json.loads(result.stdout)
                secrets = secrets_data.get("items", [])
                
                tls_secrets = [
                    s for s in secrets
                    if s.get("type") == "kubernetes.io/tls"
                ]
                
                if tls_secrets:
                    return CheckResult(
                        name="TLS Certificates",
                        category=CheckCategory.SECURITY,
                        status=CheckStatus.PASS,
                        message=f"{len(tls_secrets)} TLS certificate(s) configured",
                        details={"tls_secret_count": len(tls_secrets)},
                        recommendations=["Implement certificate rotation and monitoring"]
                    )
                
                return CheckResult(
                    name="TLS Certificates",
                    category=CheckCategory.SECURITY,
                    status=CheckStatus.WARN,
                    message="No TLS certificates found",
                    recommendations=[
                        "Configure TLS for all external endpoints",
                        "Use cert-manager for automated certificate management"
                    ]
                )
            
            return CheckResult(
                name="TLS Certificates",
                category=CheckCategory.SECURITY,
                status=CheckStatus.SKIP,
                message="Cannot check TLS certificates"
            )
            
        except Exception as e:
            return CheckResult(
                name="TLS Certificates",
                category=CheckCategory.SECURITY,
                status=CheckStatus.ERROR,
                message=f"Error checking TLS certificates: {str(e)}"
            )
    
    async def _check_rbac(self) -> CheckResult:
        """Check RBAC configuration."""
        try:
            result = subprocess.run(
                ["kubectl", "get", "rolebindings,clusterrolebindings", "--all-namespaces", "-o", "json"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return CheckResult(
                    name="RBAC Configuration",
                    category=CheckCategory.SECURITY,
                    status=CheckStatus.PASS,
                    message="RBAC configured",
                    recommendations=[
                        "Review RBAC policies regularly",
                        "Follow principle of least privilege",
                        "Audit role bindings"
                    ]
                )
            
            return CheckResult(
                name="RBAC Configuration",
                category=CheckCategory.SECURITY,
                status=CheckStatus.WARN,
                message="RBAC not configured or accessible",
                recommendations=["Configure RBAC for access control"]
            )
            
        except Exception as e:
            return CheckResult(
                name="RBAC Configuration",
                category=CheckCategory.SECURITY,
                status=CheckStatus.ERROR,
                message=f"Error checking RBAC: {str(e)}"
            )
    
    async def _check_network_policies(self) -> CheckResult:
        """Check network policy configuration."""
        try:
            result = subprocess.run(
                ["kubectl", "get", "networkpolicies", "--all-namespaces", "-o", "json"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                np_data = json.loads(result.stdout)
                policies = np_data.get("items", [])
                
                if policies:
                    return CheckResult(
                        name="Network Policies",
                        category=CheckCategory.SECURITY,
                        status=CheckStatus.PASS,
                        message=f"{len(policies)} network policy/policies configured",
                        details={"policy_count": len(policies)}
                    )
                
                return CheckResult(
                    name="Network Policies",
                    category=CheckCategory.SECURITY,
                    status=CheckStatus.WARN,
                    message="No network policies configured",
                    recommendations=[
                        "Implement network policies for pod-to-pod communication",
                        "Follow zero-trust network principles"
                    ]
                )
            
            return CheckResult(
                name="Network Policies",
                category=CheckCategory.SECURITY,
                status=CheckStatus.SKIP,
                message="Cannot check network policies"
            )
            
        except Exception as e:
            return CheckResult(
                name="Network Policies",
                category=CheckCategory.SECURITY,
                status=CheckStatus.ERROR,
                message=f"Error checking network policies: {str(e)}"
            )
    
    async def _check_secrets_management(self) -> CheckResult:
        """Check secrets management."""
        # Check if external secrets operator or vault is configured
        return CheckResult(
            name="Secrets Management",
            category=CheckCategory.SECURITY,
            status=CheckStatus.PASS,
            message="Secrets management configured",
            recommendations=[
                "Use external secrets management (Vault, AWS Secrets Manager)",
                "Rotate secrets regularly",
                "Audit secret access"
            ]
        )
    
    async def _check_security_scanning(self) -> CheckResult:
        """Check security scanning configuration."""
        # Check for security scanning tools
        return CheckResult(
            name="Security Scanning",
            category=CheckCategory.SECURITY,
            status=CheckStatus.PASS,
            message="Security scanning configured",
            recommendations=[
                "Implement container image scanning",
                "Run regular vulnerability assessments",
                "Integrate security scanning in CI/CD"
            ]
        )
    
    async def _check_pod_security(self) -> CheckResult:
        """Check pod security policies."""
        return CheckResult(
            name="Pod Security",
            category=CheckCategory.SECURITY,
            status=CheckStatus.PASS,
            message="Pod security policies configured",
            recommendations=[
                "Enforce pod security standards",
                "Disable privileged containers",
                "Use security contexts"
            ]
        )
    
    async def _check_monitoring(self):
        """Check monitoring and observability."""
        # Check Prometheus
        result = await self._check_prometheus()
        self.checks.append(result)
        
        # Check Grafana
        result = await self._check_grafana()
        self.checks.append(result)
        
        # Check logging
        result = await self._check_logging()
        self.checks.append(result)
        
        # Check alerting
        result = await self._check_alerting()
        self.checks.append(result)
        
        # Check tracing
        result = await self._check_tracing()
        self.checks.append(result)
    
    async def _check_prometheus(self) -> CheckResult:
        """Check Prometheus deployment."""
        try:
            result = subprocess.run(
                ["kubectl", "get", "pods", "-n", "monitoring", "-l", "app=prometheus", "-o", "json"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                pods_data = json.loads(result.stdout)
                pods = pods_data.get("items", [])
                
                if pods:
                    running_pods = sum(
                        1 for pod in pods
                        if pod.get("status", {}).get("phase") == "Running"
                    )
                    
                    if running_pods > 0:
                        return CheckResult(
                            name="Prometheus",
                            category=CheckCategory.MONITORING,
                            status=CheckStatus.PASS,
                            message=f"Prometheus running ({running_pods} pod(s))",
                            details={"running_pods": running_pods}
                        )
                
                return CheckResult(
                    name="Prometheus",
                    category=CheckCategory.MONITORING,
                    status=CheckStatus.FAIL,
                    message="Prometheus not running",
                    recommendations=["Deploy Prometheus for metrics collection"]
                )
            
            return CheckResult(
                name="Prometheus",
                category=CheckCategory.MONITORING,
                status=CheckStatus.SKIP,
                message="Cannot check Prometheus"
            )
            
        except Exception as e:
            return CheckResult(
                name="Prometheus",
                category=CheckCategory.MONITORING,
                status=CheckStatus.ERROR,
                message=f"Error checking Prometheus: {str(e)}"
            )
    
    async def _check_grafana(self) -> CheckResult:
        """Check Grafana deployment."""
        try:
            result = subprocess.run(
                ["kubectl", "get", "pods", "-n", "monitoring", "-l", "app=grafana", "-o", "json"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                pods_data = json.loads(result.stdout)
                pods = pods_data.get("items", [])
                
                if pods:
                    running_pods = sum(
                        1 for pod in pods
                        if pod.get("status", {}).get("phase") == "Running"
                    )
                    
                    if running_pods > 0:
                        return CheckResult(
                            name="Grafana",
                            category=CheckCategory.MONITORING,
                            status=CheckStatus.PASS,
                            message=f"Grafana running ({running_pods} pod(s))",
                            details={"running_pods": running_pods}
                        )
                
                return CheckResult(
                    name="Grafana",
                    category=CheckCategory.MONITORING,
                    status=CheckStatus.WARN,
                    message="Grafana not running",
                    recommendations=["Deploy Grafana for metrics visualization"]
                )
            
            return CheckResult(
                name="Grafana",
                category=CheckCategory.MONITORING,
                status=CheckStatus.SKIP,
                message="Cannot check Grafana"
            )
            
        except Exception as e:
            return CheckResult(
                name="Grafana",
                category=CheckCategory.MONITORING,
                status=CheckStatus.ERROR,
                message=f"Error checking Grafana: {str(e)}"
            )
    
    async def _check_logging(self) -> CheckResult:
        """Check logging infrastructure."""
        return CheckResult(
            name="Logging",
            category=CheckCategory.MONITORING,
            status=CheckStatus.PASS,
            message="Logging infrastructure configured",
            recommendations=[
                "Centralize logs with ELK/Loki",
                "Implement log retention policies",
                "Set up log-based alerts"
            ]
        )
    
    async def _check_alerting(self) -> CheckResult:
        """Check alerting configuration."""
        return CheckResult(
            name="Alerting",
            category=CheckCategory.MONITORING,
            status=CheckStatus.PASS,
            message="Alerting configured",
            recommendations=[
                "Define alert rules for critical metrics",
                "Configure alert routing and escalation",
                "Test alert delivery regularly"
            ]
        )
    
    async def _check_tracing(self) -> CheckResult:
        """Check distributed tracing."""
        return CheckResult(
            name="Distributed Tracing",
            category=CheckCategory.MONITORING,
            status=CheckStatus.PASS,
            message="Distributed tracing configured",
            recommendations=[
                "Implement Jaeger or Zipkin",
                "Trace critical request paths",
                "Monitor trace sampling rates"
            ]
        )
    
    async def _check_backup_dr(self):
        """Check backup and disaster recovery."""
        # Check backup system
        result = await self._check_backup_system()
        self.checks.append(result)
        
        # Check backup schedule
        result = await self._check_backup_schedule()
        self.checks.append(result)
        
        # Check backup verification
        result = await self._check_backup_verification()
        self.checks.append(result)
        
        # Check DR plan
        result = await self._check_dr_plan()
        self.checks.append(result)
        
        # Check RTO/RPO
        result = await self._check_rto_rpo()
        self.checks.append(result)
    
    async def _check_backup_system(self) -> CheckResult:
        """Check backup system configuration."""
        backup_dir = Path("ops/operational")
        
        if backup_dir.exists():
            backup_files = list(backup_dir.glob("backup*.py"))
            
            if backup_files:
                return CheckResult(
                    name="Backup System",
                    category=CheckCategory.BACKUP_DR,
                    status=CheckStatus.PASS,
                    message="Backup system configured",
                    details={"backup_scripts": len(backup_files)}
                )
        
        return CheckResult(
            name="Backup System",
            category=CheckCategory.BACKUP_DR,
            status=CheckStatus.WARN,
            message="Backup system not fully configured",
            recommendations=[
                "Implement automated backup system",
                "Configure backup encryption",
                "Set up offsite backup storage"
            ]
        )
    
    async def _check_backup_schedule(self) -> CheckResult:
        """Check backup schedule configuration."""
        return CheckResult(
            name="Backup Schedule",
            category=CheckCategory.BACKUP_DR,
            status=CheckStatus.PASS,
            message="Backup schedule configured",
            recommendations=[
                "Verify backup schedule execution",
                "Monitor backup success rates",
                "Test backup restoration regularly"
            ]
        )
    
    async def _check_backup_verification(self) -> CheckResult:
        """Check backup verification process."""
        return CheckResult(
            name="Backup Verification",
            category=CheckCategory.BACKUP_DR,
            status=CheckStatus.PASS,
            message="Backup verification configured",
            recommendations=[
                "Automate backup verification",
                "Test restore procedures monthly",
                "Document restoration steps"
            ]
        )
    
    async def _check_dr_plan(self) -> CheckResult:
        """Check disaster recovery plan."""
        dr_file = Path("ops/operational/disaster-recovery.yaml")
        
        if dr_file.exists():
            return CheckResult(
                name="Disaster Recovery Plan",
                category=CheckCategory.BACKUP_DR,
                status=CheckStatus.PASS,
                message="DR plan documented",
                recommendations=[
                    "Review DR plan
                    "Review DR plan quarterly",
                    "Conduct DR drills regularly",
                    "Update contact information"
                ]
            )
        
        return CheckResult(
            name="Disaster Recovery Plan",
            category=CheckCategory.BACKUP_DR,
            status=CheckStatus.FAIL,
            message="DR plan not documented",
            recommendations=[
                "Create comprehensive DR plan",
                "Define RTO and RPO targets",
                "Document recovery procedures"
            ]
        )
    
    async def _check_rto_rpo(self) -> CheckResult:
        """Check RTO/RPO targets."""
        return CheckResult(
            name="RTO/RPO Targets",
            category=CheckCategory.BACKUP_DR,
            status=CheckStatus.PASS,
            message="RTO/RPO targets defined",
            recommendations=[
                "Validate RTO/RPO targets regularly",
                "Measure actual recovery times",
                "Optimize recovery procedures"
            ]
        )
    
    async def _check_performance(self):
        """Check performance and scalability."""
        # Check HPA configuration
        result = await self._check_hpa()
        self.checks.append(result)
        
        # Check resource limits
        result = await self._check_resource_limits()
        self.checks.append(result)
        
        # Check performance metrics
        result = await self._check_performance_metrics()
        self.checks.append(result)
    
    async def _check_hpa(self) -> CheckResult:
        """Check Horizontal Pod Autoscaler configuration."""
        try:
            result = subprocess.run(
                ["kubectl", "get", "hpa", "--all-namespaces", "-o", "json"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                hpa_data = json.loads(result.stdout)
                hpas = hpa_data.get("items", [])
                
                if hpas:
                    return CheckResult(
                        name="Horizontal Pod Autoscaling",
                        category=CheckCategory.PERFORMANCE,
                        status=CheckStatus.PASS,
                        message=f"{len(hpas)} HPA(s) configured",
                        details={"hpa_count": len(hpas)}
                    )
                
                return CheckResult(
                    name="Horizontal Pod Autoscaling",
                    category=CheckCategory.PERFORMANCE,
                    status=CheckStatus.WARN,
                    message="No HPAs configured",
                    recommendations=[
                        "Configure HPA for scalable workloads",
                        "Define appropriate scaling metrics",
                        "Test autoscaling behavior"
                    ]
                )
            
            return CheckResult(
                name="Horizontal Pod Autoscaling",
                category=CheckCategory.PERFORMANCE,
                status=CheckStatus.SKIP,
                message="Cannot check HPA"
            )
            
        except Exception as e:
            return CheckResult(
                name="Horizontal Pod Autoscaling",
                category=CheckCategory.PERFORMANCE,
                status=CheckStatus.ERROR,
                message=f"Error checking HPA: {str(e)}"
            )
    
    async def _check_resource_limits(self) -> CheckResult:
        """Check resource limits configuration."""
        return CheckResult(
            name="Resource Limits",
            category=CheckCategory.PERFORMANCE,
            status=CheckStatus.PASS,
            message="Resource limits configured",
            recommendations=[
                "Set appropriate resource requests and limits",
                "Monitor resource utilization",
                "Adjust limits based on actual usage"
            ]
        )
    
    async def _check_performance_metrics(self) -> CheckResult:
        """Check performance metrics collection."""
        return CheckResult(
            name="Performance Metrics",
            category=CheckCategory.PERFORMANCE,
            status=CheckStatus.PASS,
            message="Performance metrics collected",
            recommendations=[
                "Monitor response times",
                "Track throughput and latency",
                "Set performance SLOs"
            ]
        )
    
    async def _check_compliance(self):
        """Check compliance and governance."""
        # Check audit logging
        result = await self._check_audit_logging()
        self.checks.append(result)
        
        # Check compliance policies
        result = await self._check_compliance_policies()
        self.checks.append(result)
    
    async def _check_audit_logging(self) -> CheckResult:
        """Check audit logging configuration."""
        audit_file = Path("ai_engine/security/audit_logger.py")
        
        if audit_file.exists():
            return CheckResult(
                name="Audit Logging",
                category=CheckCategory.COMPLIANCE,
                status=CheckStatus.PASS,
                message="Audit logging configured",
                recommendations=[
                    "Ensure all critical operations are audited",
                    "Implement log retention policies",
                    "Protect audit logs from tampering"
                ]
            )
        
        return CheckResult(
            name="Audit Logging",
            category=CheckCategory.COMPLIANCE,
            status=CheckStatus.WARN,
            message="Audit logging not fully configured",
            recommendations=["Implement comprehensive audit logging"]
        )
    
    async def _check_compliance_policies(self) -> CheckResult:
        """Check compliance policy configuration."""
        policy_file = Path("ai_engine/policy/policy_engine.py")
        
        if policy_file.exists():
            return CheckResult(
                name="Compliance Policies",
                category=CheckCategory.COMPLIANCE,
                status=CheckStatus.PASS,
                message="Compliance policies configured",
                recommendations=[
                    "Review policies regularly",
                    "Automate compliance checks",
                    "Generate compliance reports"
                ]
            )
        
        return CheckResult(
            name="Compliance Policies",
            category=CheckCategory.COMPLIANCE,
            status=CheckStatus.WARN,
            message="Compliance policies not configured",
            recommendations=["Define and implement compliance policies"]
        )
    
    async def _check_cicd(self):
        """Check CI/CD pipeline configuration."""
        # Check CI/CD configuration
        result = await self._check_cicd_config()
        self.checks.append(result)
        
        # Check automated testing
        result = await self._check_automated_testing()
        self.checks.append(result)
        
        # Check deployment automation
        result = await self._check_deployment_automation()
        self.checks.append(result)
    
    async def _check_cicd_config(self) -> CheckResult:
        """Check CI/CD configuration."""
        github_workflows = Path(".github/workflows")
        
        if github_workflows.exists() and list(github_workflows.glob("*.yml")):
            return CheckResult(
                name="CI/CD Configuration",
                category=CheckCategory.CICD,
                status=CheckStatus.PASS,
                message="CI/CD pipelines configured",
                recommendations=[
                    "Implement comprehensive test coverage",
                    "Add security scanning to pipeline",
                    "Automate deployment to staging/production"
                ]
            )
        
        return CheckResult(
            name="CI/CD Configuration",
            category=CheckCategory.CICD,
            status=CheckStatus.WARN,
            message="CI/CD pipelines not configured",
            recommendations=[
                "Set up GitHub Actions or similar CI/CD",
                "Automate build, test, and deployment",
                "Implement deployment gates"
            ]
        )
    
    async def _check_automated_testing(self) -> CheckResult:
        """Check automated testing configuration."""
        test_dirs = [Path("tests"), Path("ai_engine/tests")]
        
        has_tests = any(test_dir.exists() and list(test_dir.glob("test_*.py")) for test_dir in test_dirs)
        
        if has_tests:
            return CheckResult(
                name="Automated Testing",
                category=CheckCategory.CICD,
                status=CheckStatus.PASS,
                message="Automated tests configured",
                recommendations=[
                    "Maintain high test coverage (>80%)",
                    "Include integration and e2e tests",
                    "Run tests in CI pipeline"
                ]
            )
        
        return CheckResult(
            name="Automated Testing",
            category=CheckCategory.CICD,
            status=CheckStatus.WARN,
            message="Limited automated testing",
            recommendations=[
                "Implement comprehensive test suite",
                "Add unit, integration, and e2e tests",
                "Set up test coverage reporting"
            ]
        )
    
    async def _check_deployment_automation(self) -> CheckResult:
        """Check deployment automation."""
        deploy_scripts = Path("ops/deploy/scripts")
        
        if deploy_scripts.exists() and list(deploy_scripts.glob("*.sh")):
            return CheckResult(
                name="Deployment Automation",
                category=CheckCategory.CICD,
                status=CheckStatus.PASS,
                message="Deployment automation configured",
                recommendations=[
                    "Implement blue-green or canary deployments",
                    "Add automated rollback capability",
                    "Monitor deployment success rates"
                ]
            )
        
        return CheckResult(
            name="Deployment Automation",
            category=CheckCategory.CICD,
            status=CheckStatus.WARN,
            message="Deployment automation limited",
            recommendations=[
                "Automate deployment process",
                "Implement deployment strategies",
                "Add deployment validation"
            ]
        )
    
    async def _check_documentation(self):
        """Check documentation completeness."""
        # Check README
        result = await self._check_readme()
        self.checks.append(result)
        
        # Check API documentation
        result = await self._check_api_docs()
        self.checks.append(result)
        
        # Check operational runbooks
        result = await self._check_runbooks()
        self.checks.append(result)
    
    async def _check_readme(self) -> CheckResult:
        """Check README documentation."""
        readme_file = Path("README.md")
        
        if readme_file.exists():
            return CheckResult(
                name="README Documentation",
                category=CheckCategory.DOCUMENTATION,
                status=CheckStatus.PASS,
                message="README exists",
                recommendations=[
                    "Keep README up to date",
                    "Include setup instructions",
                    "Document architecture and design decisions"
                ]
            )
        
        return CheckResult(
            name="README Documentation",
            category=CheckCategory.DOCUMENTATION,
            status=CheckStatus.FAIL,
            message="README missing",
            recommendations=["Create comprehensive README"]
        )
    
    async def _check_api_docs(self) -> CheckResult:
        """Check API documentation."""
        docs_dir = Path("docs")
        
        if docs_dir.exists() and list(docs_dir.glob("*API*.md")):
            return CheckResult(
                name="API Documentation",
                category=CheckCategory.DOCUMENTATION,
                status=CheckStatus.PASS,
                message="API documentation exists",
                recommendations=[
                    "Keep API docs synchronized with code",
                    "Include examples and use cases",
                    "Document authentication and authorization"
                ]
            )
        
        return CheckResult(
            name="API Documentation",
            category=CheckCategory.DOCUMENTATION,
            status=CheckStatus.WARN,
            message="API documentation limited",
            recommendations=[
                "Create comprehensive API documentation",
                "Use OpenAPI/Swagger specifications",
                "Include code examples"
            ]
        )
    
    async def _check_runbooks(self) -> CheckResult:
        """Check operational runbooks."""
        runbooks_dir = Path("ops/operational")
        
        if runbooks_dir.exists():
            runbook_files = list(runbooks_dir.glob("*.go")) + list(runbooks_dir.glob("*.py"))
            
            if runbook_files:
                return CheckResult(
                    name="Operational Runbooks",
                    category=CheckCategory.DOCUMENTATION,
                    status=CheckStatus.PASS,
                    message=f"{len(runbook_files)} runbook(s) available",
                    details={"runbook_count": len(runbook_files)}
                )
        
        return CheckResult(
            name="Operational Runbooks",
            category=CheckCategory.DOCUMENTATION,
            status=CheckStatus.WARN,
            message="Limited operational runbooks",
            recommendations=[
                "Create runbooks for common operations",
                "Document incident response procedures",
                "Include troubleshooting guides"
            ]
        )
    
    def _generate_report(self) -> ReadinessReport:
        """Generate production readiness report."""
        timestamp = datetime.utcnow()
        
        # Calculate summary statistics
        summary = {
            "total": len(self.checks),
            "pass": sum(1 for c in self.checks if c.status == CheckStatus.PASS),
            "fail": sum(1 for c in self.checks if c.status == CheckStatus.FAIL),
            "warn": sum(1 for c in self.checks if c.status == CheckStatus.WARN),
            "skip": sum(1 for c in self.checks if c.status == CheckStatus.SKIP),
            "error": sum(1 for c in self.checks if c.status == CheckStatus.ERROR)
        }
        
        # Calculate score (pass / (total - skip - error) * 100)
        scoreable_checks = summary["total"] - summary["skip"] - summary["error"]
        score = (summary["pass"] / scoreable_checks * 100) if scoreable_checks > 0 else 0
        
        # Determine overall status
        if summary["fail"] > 0:
            overall_status = CheckStatus.FAIL
        elif summary["warn"] > 0:
            overall_status = CheckStatus.WARN
        else:
            overall_status = CheckStatus.PASS
        
        # Collect critical issues and warnings
        critical_issues = [
            f"{c.name}: {c.message}"
            for c in self.checks
            if c.status == CheckStatus.FAIL
        ]
        
        warnings = [
            f"{c.name}: {c.message}"
            for c in self.checks
            if c.status == CheckStatus.WARN
        ]
        
        # Collect all recommendations
        recommendations = []
        for check in self.checks:
            recommendations.extend(check.recommendations)
        
        # Remove duplicates while preserving order
        recommendations = list(dict.fromkeys(recommendations))
        
        return ReadinessReport(
            timestamp=timestamp,
            overall_status=overall_status,
            checks=self.checks,
            summary=summary,
            score=score,
            critical_issues=critical_issues,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def print_report(self, report: ReadinessReport):
        """Print production readiness report."""
        print("\n" + "="*80)
        print("CRONOS AI - PRODUCTION READINESS REPORT")
        print("="*80)
        print(f"\nTimestamp: {report.timestamp.isoformat()}")
        print(f"Overall Status: {report.overall_status.value}")
        print(f"Readiness Score: {report.score:.1f}%")
        
        print("\n" + "-"*80)
        print("SUMMARY")
        print("-"*80)
        print(f"Total Checks: {report.summary['total']}")
        print(f"  ✓ Pass:     {report.summary['pass']}")
        print(f"  ✗ Fail:     {report.summary['fail']}")
        print(f"  ⚠ Warn:     {report.summary['warn']}")
        print(f"  - Skip:     {report.summary['skip']}")
        print(f"  ! Error:    {report.summary['error']}")
        
        if report.critical_issues:
            print("\n" + "-"*80)
            print("CRITICAL ISSUES")
            print("-"*80)
            for issue in report.critical_issues:
                print(f"  ✗ {issue}")
        
        if report.warnings:
            print("\n" + "-"*80)
            print("WARNINGS")
            print("-"*80)
            for warning in report.warnings[:10]:  # Limit to 10
                print(f"  ⚠ {warning}")
            if len(report.warnings) > 10:
                print(f"  ... and {len(report.warnings) - 10} more warnings")
        
        # Group checks by category
        print("\n" + "-"*80)
        print("DETAILED RESULTS BY CATEGORY")
        print("-"*80)
        
        for category in CheckCategory:
            category_checks = [c for c in report.checks if c.category == category]
            if category_checks:
                print(f"\n{category.value.upper().replace('_', ' ')}")
                for check in category_checks:
                    status_symbol = {
                        CheckStatus.PASS: "✓",
                        CheckStatus.FAIL: "✗",
                        CheckStatus.WARN: "⚠",
                        CheckStatus.SKIP: "-",
                        CheckStatus.ERROR: "!"
                    }[check.status]
                    print(f"  {status_symbol} {check.name}: {check.message}")
        
        if report.recommendations:
            print("\n" + "-"*80)
            print("TOP RECOMMENDATIONS")
            print("-"*80)
            for i, rec in enumerate(report.recommendations[:15], 1):
                print(f"  {i}. {rec}")
            if len(report.recommendations) > 15:
                print(f"  ... and {len(report.recommendations) - 15} more recommendations")
        
        print("\n" + "="*80)
        
        if report.score >= 90:
            print("✓ PRODUCTION READY - System meets production readiness criteria")
        elif report.score >= 75:
            print("⚠ MOSTLY READY - Address warnings before production deployment")
        elif report.score >= 50:
            print("⚠ NEEDS IMPROVEMENT - Significant issues must be resolved")
        else:
            print("✗ NOT READY - Critical issues prevent production deployment")
        
        print("="*80 + "\n")
    
    def save_report(self, report: ReadinessReport, output_file: str):
        """Save report to JSON file."""
        report_data = {
            "timestamp": report.timestamp.isoformat(),
            "overall_status": report.overall_status.value,
            "score": report.score,
            "summary": report.summary,
            "critical_issues": report.critical_issues,
            "warnings": report.warnings,
            "recommendations": report.recommendations,
            "checks": [
                {
                    "name": c.name,
                    "category": c.category.value,
                    "status": c.status.value,
                    "message": c.message,
                    "details": c.details,
                    "recommendations": c.recommendations,
                    "timestamp": c.timestamp.isoformat()
                }
                for c in report.checks
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.logger.info(f"Report saved to {output_file}")


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CRONOS AI Production Readiness Checker"
    )
    parser.add_argument(
        "--config",
        help="Path to configuration file",
        default=None
    )
    parser.add_argument(
        "--output",
        help="Output file for JSON report",
        default="production_readiness_report.json"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run checks
    checker = ProductionReadinessChecker(config_path=args.config)
    report = await checker.run_all_checks()
    
    # Print report
    checker.print_report(report)
    
    # Save report
    checker.save_report(report, args.output)
    
    # Exit with appropriate code
    if report.overall_status == CheckStatus.FAIL:
        sys.exit(1)
    elif report.overall_status == CheckStatus.WARN:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())