"""
Istio Sidecar Injector

Automatically injects CRONOS AI quantum-safe security sidecars into Kubernetes pods
for transparent encryption of service-to-service traffic.
"""

import base64
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import yaml

logger = logging.getLogger(__name__)


@dataclass
class SidecarConfig:
    """Configuration for CRONOS AI security sidecar"""

    image: str = "cronos-ai/quantum-sidecar:latest"
    cpu_request: str = "100m"
    cpu_limit: str = "500m"
    memory_request: str = "128Mi"
    memory_limit: str = "512Mi"
    quantum_algorithm: str = "kyber-1024"
    signature_algorithm: str = "dilithium-5"
    enable_metrics: bool = True
    metrics_port: int = 15090
    proxy_port: int = 15001
    admin_port: int = 15000


class IstioSidecarInjector:
    """
    Manages automatic injection of CRONOS AI quantum-safe security sidecars
    into Istio service mesh pods.

    This replaces or augments the default Istio Envoy sidecar with quantum-safe
    encryption capabilities.
    """

    def __init__(
        self,
        namespace: str = "cronos-system",
        config: Optional[SidecarConfig] = None,
        webhook_name: str = "cronos-sidecar-injector"
    ):
        """
        Initialize the Istio sidecar injector.

        Args:
            namespace: Kubernetes namespace for CRONOS AI components
            config: Sidecar configuration settings
            webhook_name: Name of the mutating webhook configuration
        """
        self.namespace = namespace
        self.config = config or SidecarConfig()
        self.webhook_name = webhook_name
        logger.info(f"Initialized IstioSidecarInjector in namespace: {namespace}")

    def create_webhook_configuration(self) -> Dict[str, Any]:
        """
        Create Kubernetes MutatingWebhookConfiguration for automatic sidecar injection.

        Returns:
            Dict containing the webhook configuration YAML
        """
        webhook_config = {
            "apiVersion": "admissionregistration.k8s.io/v1",
            "kind": "MutatingWebhookConfiguration",
            "metadata": {
                "name": self.webhook_name,
                "labels": {
                    "app": "cronos-ai",
                    "component": "sidecar-injector"
                }
            },
            "webhooks": [
                {
                    "name": f"{self.webhook_name}.cronos-ai.io",
                    "clientConfig": {
                        "service": {
                            "name": "cronos-sidecar-injector",
                            "namespace": self.namespace,
                            "path": "/inject"
                        },
                        "caBundle": ""  # Will be populated with actual CA certificate
                    },
                    "rules": [
                        {
                            "operations": ["CREATE"],
                            "apiGroups": [""],
                            "apiVersions": ["v1"],
                            "resources": ["pods"]
                        }
                    ],
                    "failurePolicy": "Fail",
                    "admissionReviewVersions": ["v1", "v1beta1"],
                    "sideEffects": "None",
                    "namespaceSelector": {
                        "matchLabels": {
                            "cronos-injection": "enabled"
                        }
                    }
                }
            ]
        }

        logger.info(f"Created webhook configuration: {self.webhook_name}")
        return webhook_config

    def inject_sidecar(self, pod_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inject CRONOS AI quantum-safe sidecar into a pod specification.

        Args:
            pod_spec: Original Kubernetes pod specification

        Returns:
            Modified pod specification with injected sidecar
        """
        # Create sidecar container
        sidecar_container = self._create_sidecar_container()

        # Add sidecar to pod spec
        if "containers" not in pod_spec["spec"]:
            pod_spec["spec"]["containers"] = []

        pod_spec["spec"]["containers"].append(sidecar_container)

        # Add init container for setup
        init_container = self._create_init_container()
        if "initContainers" not in pod_spec["spec"]:
            pod_spec["spec"]["initContainers"] = []

        pod_spec["spec"]["initContainers"].append(init_container)

        # Add volumes for shared data
        if "volumes" not in pod_spec["spec"]:
            pod_spec["spec"]["volumes"] = []

        pod_spec["spec"]["volumes"].extend(self._create_volumes())

        # Add annotations
        if "annotations" not in pod_spec["metadata"]:
            pod_spec["metadata"]["annotations"] = {}

        pod_spec["metadata"]["annotations"].update({
            "cronos.ai/sidecar-injected": "true",
            "cronos.ai/quantum-algorithm": self.config.quantum_algorithm,
            "cronos.ai/signature-algorithm": self.config.signature_algorithm,
            "cronos.ai/version": "1.0.0"
        })

        logger.info("Successfully injected CRONOS AI sidecar into pod")
        return pod_spec

    def _create_sidecar_container(self) -> Dict[str, Any]:
        """Create the sidecar container specification"""
        return {
            "name": "cronos-quantum-proxy",
            "image": self.config.image,
            "imagePullPolicy": "IfNotPresent",
            "args": [
                "proxy",
                "--quantum-algo", self.config.quantum_algorithm,
                "--signature-algo", self.config.signature_algorithm,
                "--proxy-port", str(self.config.proxy_port),
                "--admin-port", str(self.config.admin_port),
                "--metrics-port", str(self.config.metrics_port),
                "--log-level", "info"
            ],
            "ports": [
                {
                    "name": "proxy",
                    "containerPort": self.config.proxy_port,
                    "protocol": "TCP"
                },
                {
                    "name": "admin",
                    "containerPort": self.config.admin_port,
                    "protocol": "TCP"
                },
                {
                    "name": "metrics",
                    "containerPort": self.config.metrics_port,
                    "protocol": "TCP"
                }
            ],
            "env": [
                {
                    "name": "POD_NAME",
                    "valueFrom": {
                        "fieldRef": {
                            "fieldPath": "metadata.name"
                        }
                    }
                },
                {
                    "name": "POD_NAMESPACE",
                    "valueFrom": {
                        "fieldRef": {
                            "fieldPath": "metadata.namespace"
                        }
                    }
                },
                {
                    "name": "CRONOS_QUANTUM_ENABLED",
                    "value": "true"
                }
            ],
            "resources": {
                "requests": {
                    "cpu": self.config.cpu_request,
                    "memory": self.config.memory_request
                },
                "limits": {
                    "cpu": self.config.cpu_limit,
                    "memory": self.config.memory_limit
                }
            },
            "volumeMounts": [
                {
                    "name": "cronos-certs",
                    "mountPath": "/etc/cronos/certs",
                    "readOnly": True
                },
                {
                    "name": "cronos-config",
                    "mountPath": "/etc/cronos/config",
                    "readOnly": True
                }
            ],
            "securityContext": {
                "runAsNonRoot": True,
                "runAsUser": 1337,
                "capabilities": {
                    "add": ["NET_ADMIN", "NET_RAW"],
                    "drop": ["ALL"]
                }
            },
            "livenessProbe": {
                "httpGet": {
                    "path": "/healthz",
                    "port": self.config.admin_port
                },
                "initialDelaySeconds": 10,
                "periodSeconds": 10
            },
            "readinessProbe": {
                "httpGet": {
                    "path": "/ready",
                    "port": self.config.admin_port
                },
                "initialDelaySeconds": 5,
                "periodSeconds": 5
            }
        }

    def _create_init_container(self) -> Dict[str, Any]:
        """Create the init container for iptables setup"""
        return {
            "name": "cronos-init",
            "image": self.config.image,
            "imagePullPolicy": "IfNotPresent",
            "command": ["cronos-iptables"],
            "args": [
                "-p", str(self.config.proxy_port),
                "-u", "1337",
                "-m", "REDIRECT",
                "-i", "*",
                "-b", "*"
            ],
            "securityContext": {
                "capabilities": {
                    "add": ["NET_ADMIN", "NET_RAW"],
                    "drop": ["ALL"]
                },
                "privileged": False,
                "runAsNonRoot": False,
                "runAsUser": 0
            },
            "resources": {
                "requests": {
                    "cpu": "10m",
                    "memory": "10Mi"
                },
                "limits": {
                    "cpu": "50m",
                    "memory": "50Mi"
                }
            }
        }

    def _create_volumes(self) -> List[Dict[str, Any]]:
        """Create volume specifications for certificates and config"""
        return [
            {
                "name": "cronos-certs",
                "secret": {
                    "secretName": "cronos-quantum-certs",
                    "optional": False
                }
            },
            {
                "name": "cronos-config",
                "configMap": {
                    "name": "cronos-mesh-config",
                    "optional": False
                }
            }
        ]

    def should_inject(self, pod: Dict[str, Any]) -> bool:
        """
        Determine if a pod should have the sidecar injected.

        Args:
            pod: Kubernetes pod specification

        Returns:
            bool: True if sidecar should be injected
        """
        # Check for explicit annotation to skip injection
        annotations = pod.get("metadata", {}).get("annotations", {})
        if annotations.get("cronos.ai/inject") == "false":
            return False

        # Check if already injected
        if annotations.get("cronos.ai/sidecar-injected") == "true":
            return False

        # Check namespace label
        namespace = pod.get("metadata", {}).get("namespace", "default")

        # Check pod labels
        labels = pod.get("metadata", {}).get("labels", {})

        # Inject if explicitly requested or if namespace is labeled
        if annotations.get("cronos.ai/inject") == "true":
            return True

        # Add more sophisticated injection logic here
        return True

    def generate_injection_template(self) -> Dict[str, Any]:
        """
        Generate the complete injection template for ConfigMap storage.

        Returns:
            Dict containing the injection template
        """
        template = {
            "policy": "enabled",
            "alwaysInjectSelector": [],
            "neverInjectSelector": [],
            "injectedAnnotations": {},
            "template": {
                "metadata": {
                    "annotations": {
                        "cronos.ai/sidecar-injected": "true"
                    }
                },
                "spec": {
                    "containers": [
                        self._create_sidecar_container()
                    ],
                    "initContainers": [
                        self._create_init_container()
                    ],
                    "volumes": self._create_volumes()
                }
            }
        }

        return template


def create_injection_webhook_service() -> Dict[str, Any]:
    """
    Create Kubernetes Service for the sidecar injection webhook.

    Returns:
        Dict containing the Service YAML
    """
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": "cronos-sidecar-injector",
            "namespace": "cronos-system",
            "labels": {
                "app": "cronos-ai",
                "component": "sidecar-injector"
            }
        },
        "spec": {
            "ports": [
                {
                    "name": "https-webhook",
                    "port": 443,
                    "targetPort": 9443,
                    "protocol": "TCP"
                }
            ],
            "selector": {
                "app": "cronos-ai",
                "component": "sidecar-injector"
            },
            "type": "ClusterIP"
        }
    }
