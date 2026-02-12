"""
Admission Webhook Server

Kubernetes ValidatingWebhookConfiguration and MutatingWebhookConfiguration
server for enforcing container security policies.
"""

import logging
import json
import base64
import hashlib
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
from aiohttp import web
import asyncio
import ssl

logger = logging.getLogger(__name__)


class AdmissionAction(Enum):
    """Admission webhook actions"""

    ALLOW = "allow"
    DENY = "deny"
    PATCH = "patch"


@dataclass
class AdmissionRequest:
    """Kubernetes admission request"""

    uid: str
    kind: str
    operation: str
    object: Dict[str, Any]
    old_object: Optional[Dict[str, Any]] = None
    namespace: str = ""
    user_info: Optional[Dict[str, Any]] = None


@dataclass
class SecurityPolicy:
    """Container security policy"""

    name: str
    enabled: bool = True
    enforce: bool = True

    # Image policies
    require_image_signature: bool = True
    allowed_registries: List[str] = None
    blocked_registries: List[str] = None

    # Vulnerability policies
    max_critical_vulnerabilities: int = 0
    max_high_vulnerabilities: int = 5

    # Runtime policies
    disallow_privileged: bool = True
    disallow_host_network: bool = True
    disallow_host_pid: bool = True
    require_read_only_root: bool = False

    # Quantum-safe requirements
    require_quantum_safe_crypto: bool = True

    def __post_init__(self):
        if self.allowed_registries is None:
            self.allowed_registries = []
        if self.blocked_registries is None:
            self.blocked_registries = []


class AdmissionWebhookServer:
    """
    Kubernetes admission webhook server for container security.
    """

    def __init__(self, port: int = 8443, tls_cert_path: Optional[str] = None, tls_key_path: Optional[str] = None):
        """Initialize admission webhook server"""
        self.port = port
        self.tls_cert_path = tls_cert_path
        self.tls_key_path = tls_key_path
        self._policies: Dict[str, SecurityPolicy] = {}
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None

        # Load default policy
        self._load_default_policy()

        # Vulnerable library patterns
        self._quantum_vulnerable_libs = [
            r"openssl.*1\.0\.",  # Old OpenSSL versions
            r"libssl.*1\.0\.",
            r"crypto.*[^q]",  # Non-quantum crypto libs
            r"rsa-.*",
            r"dsa-.*",
            r"ecdsa-.*",
        ]

        logger.info(f"Initialized AdmissionWebhookServer on port {port}")

    def _load_default_policy(self):
        """Load default security policy"""
        default_policy = SecurityPolicy(
            name="default",
            require_image_signature=True,
            allowed_registries=["gcr.io", "docker.io", "quay.io", "registry.k8s.io"],
            blocked_registries=["untrusted.io"],
            max_critical_vulnerabilities=0,
            max_high_vulnerabilities=5,
            disallow_privileged=True,
            disallow_host_network=True,
            disallow_host_pid=True,
            require_quantum_safe_crypto=True,
        )
        self._policies["default"] = default_policy
        logger.info("Loaded default security policy")

    def validate_pod(self, pod: Dict[str, Any], policy_name: str = "default") -> Dict[str, Any]:
        """
        Validate pod against security policies.

        Args:
            pod: Pod specification
            policy_name: Name of security policy to apply

        Returns:
            Dict with validation result
        """
        policy = self._policies.get(policy_name, self._policies["default"])
        violations = []

        # Extract pod spec
        spec = pod.get("spec", {})
        containers = spec.get("containers", [])
        init_containers = spec.get("initContainers", [])
        all_containers = containers + init_containers

        # Validate containers
        for idx, container in enumerate(all_containers):
            container_name = container.get("name", f"container-{idx}")

            # 1. Validate image registry
            image = container.get("image", "")
            if not self._validate_image_registry(image, policy):
                violations.append(f"Container '{container_name}' uses disallowed registry: {image}")

            # 2. Validate image signature
            if policy.require_image_signature:
                if not self._verify_image_signature(image):
                    violations.append(f"Container '{container_name}' image not signed or signature invalid: {image}")

            # 3. Check for quantum-vulnerable libraries
            if policy.require_quantum_safe_crypto:
                vulnerable_libs = self._check_quantum_vulnerable_libs(image)
                if vulnerable_libs:
                    violations.append(
                        f"Container '{container_name}' contains quantum-vulnerable libraries: {', '.join(vulnerable_libs)}"
                    )

            # 4. Check security context
            security_context = container.get("securityContext", {})

            if policy.disallow_privileged and security_context.get("privileged", False):
                violations.append(f"Container '{container_name}' runs in privileged mode")

            if policy.require_read_only_root and not security_context.get("readOnlyRootFilesystem", False):
                violations.append(f"Container '{container_name}' does not have read-only root filesystem")

        # 5. Check pod-level security
        if policy.disallow_host_network and spec.get("hostNetwork", False):
            violations.append("Pod uses host network")

        if policy.disallow_host_pid and spec.get("hostPID", False):
            violations.append("Pod uses host PID namespace")

        # Determine if pod is allowed
        if violations:
            allowed = not policy.enforce
            status_code = 403 if policy.enforce else 200
            message = "Pod validation failed:\n" + "\n".join(f"- {v}" for v in violations)
            logger.warning(f"Pod validation failed: {message}")
        else:
            allowed = True
            status_code = 200
            message = "Pod meets all security requirements"
            logger.info("Pod validation passed")

        return {"allowed": allowed, "status": {"code": status_code, "message": message}, "violations": violations}

    def _validate_image_registry(self, image: str, policy: SecurityPolicy) -> bool:
        """
        Validate image registry against policy.

        Args:
            image: Container image
            policy: Security policy

        Returns:
            True if allowed, False otherwise
        """
        # Extract registry from image
        parts = image.split("/")
        registry = parts[0] if len(parts) > 1 and "." in parts[0] else "docker.io"

        # Check blocked registries
        for blocked in policy.blocked_registries:
            if blocked in registry:
                return False

        # Check allowed registries
        if policy.allowed_registries:
            for allowed in policy.allowed_registries:
                if allowed in registry:
                    return True
            return False

        return True

    def _verify_image_signature(self, image: str) -> bool:
        """
        Verify image signature using cosign or similar tool.

        Args:
            image: Container image

        Returns:
            True if signature is valid, False otherwise
        """
        # In production, this would integrate with cosign, notary, or similar
        # For now, we'll simulate signature verification

        # Check if image has a digest (signed images typically use digests)
        if "@sha256:" in image:
            logger.debug(f"Image {image} uses digest, likely signed")
            return True

        # Check for known trusted images
        trusted_prefixes = [
            "gcr.io/distroless/",
            "registry.k8s.io/",
        ]

        for prefix in trusted_prefixes:
            if image.startswith(prefix):
                logger.debug(f"Image {image} from trusted registry")
                return True

        logger.warning(f"Image {image} signature could not be verified")
        return False

    def _check_quantum_vulnerable_libs(self, image: str) -> List[str]:
        """
        Check image for quantum-vulnerable cryptographic libraries.

        Args:
            image: Container image

        Returns:
            List of vulnerable libraries found
        """
        # In production, this would scan the image using trivy, grype, or similar
        # For now, we'll check based on image name patterns
        vulnerable = []

        for pattern in self._quantum_vulnerable_libs:
            if re.search(pattern, image, re.IGNORECASE):
                vulnerable.append(pattern)

        # Check for known vulnerable base images
        vulnerable_bases = [
            "ubuntu:14.04",
            "ubuntu:16.04",
            "debian:jessie",
            "centos:6",
            "centos:7",
        ]

        for base in vulnerable_bases:
            if base in image:
                vulnerable.append(f"outdated base image: {base}")

        return vulnerable

    async def handle_validate(self, request: web.Request) -> web.Response:
        """
        Handle validation webhook request from Kubernetes.

        Args:
            request: HTTP request

        Returns:
            HTTP response with admission review
        """
        try:
            # Parse admission review request
            body = await request.json()
            admission_request = body.get("request", {})

            uid = admission_request.get("uid")
            kind = admission_request.get("kind", {}).get("kind")
            operation = admission_request.get("operation")
            obj = admission_request.get("object", {})
            namespace = admission_request.get("namespace", "default")

            logger.info(f"Admission request: {operation} {kind} in namespace {namespace}")

            # Validate based on kind
            if kind == "Pod":
                validation_result = self.validate_pod(obj)
            else:
                # Allow other resources
                validation_result = {
                    "allowed": True,
                    "status": {"code": 200, "message": f"Resource type {kind} not validated"},
                }

            # Build admission review response
            admission_response = {
                "apiVersion": "admission.k8s.io/v1",
                "kind": "AdmissionReview",
                "response": {"uid": uid, "allowed": validation_result["allowed"], "status": validation_result["status"]},
            }

            return web.json_response(admission_response)

        except Exception as e:
            logger.error(f"Error handling admission request: {e}")
            return web.json_response(
                {
                    "apiVersion": "admission.k8s.io/v1",
                    "kind": "AdmissionReview",
                    "response": {
                        "uid": admission_request.get("uid", "unknown"),
                        "allowed": False,
                        "status": {"code": 500, "message": f"Internal error: {str(e)}"},
                    },
                },
                status=500,
            )

    async def handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint"""
        return web.json_response({"status": "healthy"})

    async def start(self):
        """Start the admission webhook server"""
        logger.info(f"Starting admission webhook server on port {self.port}")

        # Create aiohttp application
        self._app = web.Application()

        # Add routes
        self._app.router.add_post("/validate", self.handle_validate)
        self._app.router.add_get("/health", self.handle_health)
        self._app.router.add_get("/healthz", self.handle_health)

        # Create runner
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        # Create SSL context if TLS configured
        ssl_context = None
        if self.tls_cert_path and self.tls_key_path:
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(self.tls_cert_path, self.tls_key_path)
            logger.info("TLS enabled for webhook server")

        # Create and start site
        site = web.TCPSite(self._runner, host="0.0.0.0", port=self.port, ssl_context=ssl_context)
        await site.start()

        logger.info(f"Admission webhook server started on port {self.port}")
        logger.info("Kubernetes can now send admission requests to /validate")

    async def stop(self):
        """Stop the admission webhook server"""
        if self._runner:
            logger.info("Stopping admission webhook server...")
            await self._runner.cleanup()
            logger.info("Admission webhook server stopped")

    def add_policy(self, policy: SecurityPolicy):
        """
        Add or update a security policy.

        Args:
            policy: Security policy to add
        """
        self._policies[policy.name] = policy
        logger.info(f"Added security policy: {policy.name}")

    def remove_policy(self, policy_name: str):
        """
        Remove a security policy.

        Args:
            policy_name: Name of policy to remove
        """
        if policy_name in self._policies and policy_name != "default":
            del self._policies[policy_name]
            logger.info(f"Removed security policy: {policy_name}")

    def create_webhook_config(self) -> Dict[str, Any]:
        """Create webhook configuration manifest"""
        return {
            "apiVersion": "admissionregistration.k8s.io/v1",
            "kind": "ValidatingWebhookConfiguration",
            "metadata": {"name": "qbitel-container-security"},
            "webhooks": [
                {
                    "name": "validate.qbitel.io",
                    "clientConfig": {"service": {"name": "qbitel-webhook", "namespace": "qbitel-system", "path": "/validate"}},
                    "rules": [
                        {"operations": ["CREATE", "UPDATE"], "apiGroups": [""], "apiVersions": ["v1"], "resources": ["pods"]}
                    ],
                    "admissionReviewVersions": ["v1"],
                }
            ],
        }
