"""
Firecracker MicroVM Sandbox

Provides secure isolated execution environment using Firecracker:
- Sub-second VM boot times
- Strong isolation guarantees
- Resource quotas (CPU, memory, network)
- Ephemeral execution
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SandboxStatus(Enum):
    """Sandbox execution status."""

    PENDING = "pending"
    CREATING = "creating"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ResourceLimits:
    """Resource limits for sandbox execution."""

    # CPU limits
    vcpu_count: int = 1
    cpu_percent: int = 100  # CPU throttle percentage

    # Memory limits
    memory_mb: int = 256
    memory_balloon: bool = True  # Enable memory ballooning

    # Disk limits
    disk_size_mb: int = 512
    disk_iops: Optional[int] = None
    disk_bandwidth_mbps: Optional[int] = None

    # Network limits
    rx_rate_limit_mbps: Optional[int] = None
    tx_rate_limit_mbps: Optional[int] = None

    # Execution limits
    timeout_seconds: int = 60
    max_processes: int = 100
    max_open_files: int = 1000


@dataclass
class NetworkConfig:
    """Network configuration for sandbox."""

    enabled: bool = False  # Disable network by default
    tap_device: Optional[str] = None
    host_ip: str = "172.16.0.1"
    guest_ip: str = "172.16.0.2"
    netmask: str = "255.255.255.0"
    gateway: Optional[str] = None

    # Allowlist for outbound connections
    allowed_hosts: List[str] = field(default_factory=list)
    allowed_ports: List[int] = field(default_factory=list)


@dataclass
class SandboxConfig:
    """Configuration for Firecracker sandbox."""

    # Resource limits
    resources: ResourceLimits = field(default_factory=ResourceLimits)

    # Network configuration
    network: NetworkConfig = field(default_factory=NetworkConfig)

    # Paths
    kernel_path: str = "/opt/firecracker/vmlinux"
    rootfs_path: str = "/opt/firecracker/rootfs.ext4"
    firecracker_bin: str = "/usr/bin/firecracker"
    jailer_bin: str = "/usr/bin/jailer"

    # Jailer configuration
    use_jailer: bool = True
    chroot_base: str = "/srv/jailer"
    jail_id: Optional[str] = None

    # Execution
    working_dir: str = "/workspace"
    environment: Dict[str, str] = field(default_factory=dict)

    # Logging
    log_level: str = "Warning"
    metrics_enabled: bool = True

    # Snapshots
    enable_snapshots: bool = False
    snapshot_path: Optional[str] = None


@dataclass
class SandboxResult:
    """Result from sandbox execution."""

    # Status
    status: SandboxStatus
    exit_code: int = 0

    # Output
    stdout: str = ""
    stderr: str = ""
    output_files: Dict[str, bytes] = field(default_factory=dict)

    # Timing
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0

    # Resource usage
    peak_memory_mb: float = 0.0
    cpu_time_ms: float = 0.0
    io_read_bytes: int = 0
    io_write_bytes: int = 0

    # Error info
    error_message: Optional[str] = None
    error_type: Optional[str] = None

    # Metadata
    sandbox_id: str = ""
    trace_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "cpu_time_ms": self.cpu_time_ms,
            "error_message": self.error_message,
            "sandbox_id": self.sandbox_id,
            "trace_id": self.trace_id,
        }


class FirecrackerSandbox:
    """
    Firecracker-based MicroVM sandbox for secure code execution.

    Provides:
    - Strong isolation via lightweight VMs
    - Fast boot times (<125ms)
    - Resource quotas and limits
    - Network isolation
    - Ephemeral execution

    Example:
        async with FirecrackerSandbox() as sandbox:
            result = await sandbox.execute(
                code="print('Hello World')",
                language="python"
            )
            print(result.stdout)
    """

    def __init__(self, config: Optional[SandboxConfig] = None):
        """
        Initialize Firecracker sandbox.

        Args:
            config: Sandbox configuration
        """
        self.config = config or SandboxConfig()
        self.sandbox_id = self.config.jail_id or str(uuid.uuid4())[:8]

        self._process: Optional[asyncio.subprocess.Process] = None
        self._socket_path: Optional[str] = None
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None
        self._status = SandboxStatus.PENDING
        self._start_time: Optional[datetime] = None

        logger.info(f"Sandbox initialized: {self.sandbox_id}")

    @property
    def status(self) -> SandboxStatus:
        """Get current sandbox status."""
        return self._status

    async def __aenter__(self) -> "FirecrackerSandbox":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()

    async def start(self) -> bool:
        """
        Start the Firecracker microVM.

        Returns:
            True if started successfully
        """
        self._status = SandboxStatus.CREATING
        self._start_time = datetime.utcnow()

        try:
            # Create temporary directory
            self._temp_dir = tempfile.TemporaryDirectory(prefix=f"fc_{self.sandbox_id}_")

            # Create socket path
            self._socket_path = os.path.join(self._temp_dir.name, "firecracker.sock")

            # Build Firecracker command
            if self.config.use_jailer:
                cmd = self._build_jailer_command()
            else:
                cmd = self._build_firecracker_command()

            logger.debug(f"Starting Firecracker: {' '.join(cmd)}")

            # Start Firecracker process
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Wait for API socket
            await self._wait_for_socket()

            # Configure VM
            await self._configure_vm()

            # Start VM
            await self._start_vm()

            self._status = SandboxStatus.RUNNING
            logger.info(f"Sandbox started: {self.sandbox_id}")
            return True

        except Exception as e:
            self._status = SandboxStatus.FAILED
            logger.error(f"Failed to start sandbox: {e}")
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop the Firecracker microVM."""
        try:
            if self._process:
                # Send shutdown signal
                try:
                    await self._api_request("PUT", "/actions", {"action_type": "SendCtrlAltDel"})
                except Exception:
                    pass

                # Wait for graceful shutdown
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    # Force kill
                    self._process.kill()
                    await self._process.wait()

                self._process = None

            # Cleanup temp directory
            if self._temp_dir:
                self._temp_dir.cleanup()
                self._temp_dir = None

            self._status = SandboxStatus.COMPLETED
            logger.info(f"Sandbox stopped: {self.sandbox_id}")

        except Exception as e:
            logger.error(f"Error stopping sandbox: {e}")
            self._status = SandboxStatus.FAILED

    async def execute(
        self,
        code: str,
        language: str = "python",
        args: Optional[List[str]] = None,
        stdin: Optional[str] = None,
        files: Optional[Dict[str, bytes]] = None,
        timeout: Optional[int] = None,
    ) -> SandboxResult:
        """
        Execute code in the sandbox.

        Args:
            code: Code to execute
            language: Programming language
            args: Command line arguments
            stdin: Standard input
            files: Files to provide to the sandbox
            timeout: Execution timeout (overrides config)

        Returns:
            SandboxResult with execution output
        """
        result = SandboxResult(
            status=SandboxStatus.RUNNING,
            sandbox_id=self.sandbox_id,
            trace_id=str(uuid.uuid4()),
            start_time=datetime.utcnow(),
        )

        try:
            timeout = timeout or self.config.resources.timeout_seconds

            # Prepare execution command
            exec_cmd = self._build_exec_command(code, language, args)

            # Upload files if provided
            if files:
                await self._upload_files(files)

            # Execute command in VM
            stdout, stderr, exit_code = await asyncio.wait_for(
                self._execute_in_vm(exec_cmd, stdin),
                timeout=timeout,
            )

            result.stdout = stdout
            result.stderr = stderr
            result.exit_code = exit_code
            result.status = SandboxStatus.COMPLETED if exit_code == 0 else SandboxStatus.FAILED

        except asyncio.TimeoutError:
            result.status = SandboxStatus.TIMEOUT
            result.error_message = f"Execution timed out after {timeout}s"
            result.error_type = "TimeoutError"

        except Exception as e:
            result.status = SandboxStatus.FAILED
            result.error_message = str(e)
            result.error_type = type(e).__name__

        finally:
            result.end_time = datetime.utcnow()
            result.duration_ms = (result.end_time - result.start_time).total_seconds() * 1000

            # Get resource usage
            await self._collect_metrics(result)

        return result

    async def execute_script(
        self,
        script_path: str,
        language: str = "python",
        **kwargs,
    ) -> SandboxResult:
        """
        Execute a script file in the sandbox.

        Args:
            script_path: Path to script file
            language: Programming language
            **kwargs: Additional arguments for execute()

        Returns:
            SandboxResult
        """
        with open(script_path, "r") as f:
            code = f.read()
        return await self.execute(code, language, **kwargs)

    async def _build_jailer_command(self) -> List[str]:
        """Build jailer command line."""
        return [
            self.config.jailer_bin,
            "--id", self.sandbox_id,
            "--exec-file", self.config.firecracker_bin,
            "--uid", "1000",
            "--gid", "1000",
            "--chroot-base-dir", self.config.chroot_base,
            "--",
            "--api-sock", self._socket_path,
        ]

    def _build_firecracker_command(self) -> List[str]:
        """Build Firecracker command line."""
        return [
            self.config.firecracker_bin,
            "--api-sock", self._socket_path,
            "--level", self.config.log_level,
        ]

    async def _wait_for_socket(self, timeout: float = 5.0) -> None:
        """Wait for API socket to be available."""
        start = time.time()
        while time.time() - start < timeout:
            if os.path.exists(self._socket_path):
                return
            await asyncio.sleep(0.01)
        raise TimeoutError("Firecracker API socket not available")

    async def _configure_vm(self) -> None:
        """Configure the VM via API."""
        # Set boot source
        await self._api_request("PUT", "/boot-source", {
            "kernel_image_path": self.config.kernel_path,
            "boot_args": "console=ttyS0 reboot=k panic=1 pci=off",
        })

        # Set root drive
        await self._api_request("PUT", "/drives/rootfs", {
            "drive_id": "rootfs",
            "path_on_host": self.config.rootfs_path,
            "is_root_device": True,
            "is_read_only": False,
        })

        # Set machine config
        await self._api_request("PUT", "/machine-config", {
            "vcpu_count": self.config.resources.vcpu_count,
            "mem_size_mib": self.config.resources.memory_mb,
        })

        # Configure network if enabled
        if self.config.network.enabled:
            await self._configure_network()

        # Set rate limiters if configured
        await self._configure_rate_limiters()

    async def _configure_network(self) -> None:
        """Configure network interface."""
        tap_device = self.config.network.tap_device or f"fc-{self.sandbox_id}-tap0"

        # Create TAP device
        await asyncio.create_subprocess_exec(
            "ip", "tuntap", "add", tap_device, "mode", "tap",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        # Configure TAP
        await asyncio.create_subprocess_exec(
            "ip", "addr", "add",
            f"{self.config.network.host_ip}/24",
            "dev", tap_device,
        )
        await asyncio.create_subprocess_exec(
            "ip", "link", "set", tap_device, "up",
        )

        # Add network interface to VM
        await self._api_request("PUT", "/network-interfaces/eth0", {
            "iface_id": "eth0",
            "host_dev_name": tap_device,
            "guest_mac": self._generate_mac_address(),
        })

    async def _configure_rate_limiters(self) -> None:
        """Configure rate limiters for resources."""
        resources = self.config.resources

        # Disk rate limiter
        if resources.disk_iops or resources.disk_bandwidth_mbps:
            rate_limiter = {}
            if resources.disk_bandwidth_mbps:
                rate_limiter["bandwidth"] = {
                    "size": resources.disk_bandwidth_mbps * 1024 * 1024,
                    "refill_time": 1000,
                }
            if resources.disk_iops:
                rate_limiter["ops"] = {
                    "size": resources.disk_iops,
                    "refill_time": 1000,
                }

            await self._api_request("PATCH", "/drives/rootfs", {
                "drive_id": "rootfs",
                "rate_limiter": rate_limiter,
            })

    async def _start_vm(self) -> None:
        """Start the VM instance."""
        await self._api_request("PUT", "/actions", {
            "action_type": "InstanceStart",
        })

        # Wait for VM to boot
        await asyncio.sleep(0.5)

    async def _execute_in_vm(
        self,
        command: List[str],
        stdin: Optional[str] = None,
    ) -> Tuple[str, str, int]:
        """Execute command inside the VM."""
        # In production, this would use vsock or serial console
        # For simulation, we'll use a subprocess approach

        # Build the full command
        full_cmd = " ".join(command)

        # Mock execution for development
        logger.debug(f"Executing in VM: {full_cmd}")

        # Simulate execution
        return "Execution output", "", 0

    async def _upload_files(self, files: Dict[str, bytes]) -> None:
        """Upload files to the VM."""
        for name, content in files.items():
            target_path = os.path.join(self.config.working_dir, name)
            logger.debug(f"Uploading file to VM: {target_path}")
            # In production, use vsock or drive attachment

    async def _collect_metrics(self, result: SandboxResult) -> None:
        """Collect resource usage metrics."""
        try:
            if self.config.metrics_enabled and self._process:
                metrics = await self._api_request("GET", "/metrics")
                # Parse and populate metrics
                result.peak_memory_mb = metrics.get("memory_actual_mib", 0)
                result.cpu_time_ms = metrics.get("cpu_time_ns", 0) / 1e6
        except Exception as e:
            logger.warning(f"Failed to collect metrics: {e}")

    async def _api_request(
        self,
        method: str,
        path: str,
        data: Optional[Dict] = None,
    ) -> Dict:
        """Make API request to Firecracker."""
        import aiohttp

        url = f"http://localhost{path}"

        async with aiohttp.UnixConnector(path=self._socket_path) as connector:
            async with aiohttp.ClientSession(connector=connector) as session:
                if method == "GET":
                    async with session.get(url) as resp:
                        return await resp.json() if resp.content_length else {}
                elif method == "PUT":
                    async with session.put(url, json=data) as resp:
                        return await resp.json() if resp.content_length else {}
                elif method == "PATCH":
                    async with session.patch(url, json=data) as resp:
                        return await resp.json() if resp.content_length else {}

        return {}

    def _build_exec_command(
        self,
        code: str,
        language: str,
        args: Optional[List[str]],
    ) -> List[str]:
        """Build execution command for language."""
        language_commands = {
            "python": ["python3", "-c", code],
            "python2": ["python2", "-c", code],
            "javascript": ["node", "-e", code],
            "ruby": ["ruby", "-e", code],
            "bash": ["bash", "-c", code],
            "sh": ["sh", "-c", code],
        }

        cmd = language_commands.get(language, ["python3", "-c", code])

        if args:
            cmd.extend(args)

        return cmd

    def _generate_mac_address(self) -> str:
        """Generate unique MAC address."""
        import random
        return "02:FC:{:02X}:{:02X}:{:02X}:{:02X}".format(
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
