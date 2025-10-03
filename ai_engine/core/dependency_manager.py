"""
CRONOS AI - Dependency Manager
Handles optional dependencies with fallback mechanisms and install checks.
Supports air-gapped deployments with vendor shims.
"""

import importlib
import logging
import sys
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DependencyStatus(Enum):
    """Dependency availability status."""
    AVAILABLE = "available"
    MISSING = "missing"
    VERSION_MISMATCH = "version_mismatch"
    IMPORT_ERROR = "import_error"


@dataclass
class DependencyInfo:
    """Information about a dependency."""
    name: str
    package: str
    required: bool
    min_version: Optional[str] = None
    status: DependencyStatus = DependencyStatus.MISSING
    installed_version: Optional[str] = None
    error_message: Optional[str] = None
    fallback_available: bool = False


class DependencyManager:
    """
    Manages optional dependencies with fallback mechanisms.
    Provides install checks and vendor shims for air-gapped deployments.
    """
    
    # LLM Provider SDKs (all optional)
    LLM_DEPENDENCIES = {
        'openai': DependencyInfo(
            name='OpenAI SDK',
            package='openai',
            required=False,
            min_version='1.10.0'
        ),
        'anthropic': DependencyInfo(
            name='Anthropic SDK',
            package='anthropic',
            required=False,
            min_version='0.8.0'
        ),
        'ollama': DependencyInfo(
            name='Ollama SDK',
            package='ollama',
            required=False,
            min_version='0.1.0'
        ),
    }
    
    # ML/DL Dependencies
    ML_DEPENDENCIES = {
        'torch': DependencyInfo(
            name='PyTorch',
            package='torch',
            required=True,
            min_version='2.1.0'
        ),
        'torchcrf': DependencyInfo(
            name='TorchCRF',
            package='torchcrf',
            required=True,
            min_version='1.1.0',
            fallback_available=True  # Can try TorchCRF (capitalized)
        ),
    }
    
    # Monitoring Dependencies
    MONITORING_DEPENDENCIES = {
        'psutil': DependencyInfo(
            name='psutil',
            package='psutil',
            required=True,
            min_version='5.9.0'
        ),
        'prometheus_client': DependencyInfo(
            name='Prometheus Client',
            package='prometheus_client',
            required=True,
            min_version='0.19.0'
        ),
    }
    
    def __init__(self):
        """Initialize dependency manager."""
        self.dependency_status: Dict[str, DependencyInfo] = {}
        self._check_all_dependencies()
    
    def _check_all_dependencies(self) -> None:
        """Check status of all dependencies."""
        all_deps = {
            **self.LLM_DEPENDENCIES,
            **self.ML_DEPENDENCIES,
            **self.MONITORING_DEPENDENCIES
        }
        
        for key, dep_info in all_deps.items():
            self.dependency_status[key] = self._check_dependency(dep_info)
    
    def _check_dependency(self, dep_info: DependencyInfo) -> DependencyInfo:
        """
        Check if a dependency is available and meets version requirements.
        
        Args:
            dep_info: Dependency information
            
        Returns:
            Updated dependency information with status
        """
        try:
            # Try to import the package
            module = importlib.import_module(dep_info.package)
            
            # Get version if available
            version = None
            for attr in ['__version__', 'VERSION', 'version']:
                if hasattr(module, attr):
                    version = getattr(module, attr)
                    if callable(version):
                        version = version()
                    break
            
            dep_info.installed_version = str(version) if version else "unknown"
            
            # Check version if minimum is specified
            if dep_info.min_version and version:
                if self._compare_versions(str(version), dep_info.min_version) < 0:
                    dep_info.status = DependencyStatus.VERSION_MISMATCH
                    dep_info.error_message = (
                        f"Version {version} is below minimum required {dep_info.min_version}"
                    )
                    logger.warning(
                        f"{dep_info.name}: {dep_info.error_message}"
                    )
                else:
                    dep_info.status = DependencyStatus.AVAILABLE
                    logger.info(
                        f"{dep_info.name} v{version} is available"
                    )
            else:
                dep_info.status = DependencyStatus.AVAILABLE
                logger.info(
                    f"{dep_info.name} is available (version: {version or 'unknown'})"
                )
                
        except ImportError as e:
            dep_info.status = DependencyStatus.MISSING
            dep_info.error_message = str(e)
            
            if dep_info.required:
                logger.error(
                    f"Required dependency {dep_info.name} is missing: {e}"
                )
            else:
                logger.info(
                    f"Optional dependency {dep_info.name} is not installed"
                )
                
            # Try fallback if available
            if dep_info.fallback_available:
                fallback_dep = self._try_fallback(dep_info)
                if fallback_dep:
                    return fallback_dep
                    
        except Exception as e:
            dep_info.status = DependencyStatus.IMPORT_ERROR
            dep_info.error_message = f"Import error: {e}"
            logger.error(
                f"Error checking {dep_info.name}: {e}"
            )
        
        return dep_info
    
    def _try_fallback(self, dep_info: DependencyInfo) -> Optional[DependencyInfo]:
        """
        Try fallback package names for dependencies.
        
        Args:
            dep_info: Original dependency information
            
        Returns:
            Updated dependency info if fallback succeeds, None otherwise
        """
        fallback_packages = {
            'torchcrf': ['TorchCRF', 'torch_crf'],
        }
        
        if dep_info.package not in fallback_packages:
            return None
        
        for fallback_pkg in fallback_packages[dep_info.package]:
            try:
                module = importlib.import_module(fallback_pkg)
                logger.info(
                    f"Using fallback package {fallback_pkg} for {dep_info.name}"
                )
                
                # Update dependency info with fallback
                fallback_info = DependencyInfo(
                    name=dep_info.name,
                    package=fallback_pkg,
                    required=dep_info.required,
                    min_version=dep_info.min_version,
                    status=DependencyStatus.AVAILABLE,
                    installed_version="unknown",
                    fallback_available=False
                )
                return fallback_info
                
            except ImportError:
                continue
        
        return None
    
    def _compare_versions(self, version1: str, version2: str) -> int:
        """
        Compare two version strings.
        
        Args:
            version1: First version string
            version2: Second version string
            
        Returns:
            -1 if version1 < version2, 0 if equal, 1 if version1 > version2
        """
        def normalize(v):
            return [int(x) for x in v.split('.') if x.isdigit()]
        
        v1_parts = normalize(version1)
        v2_parts = normalize(version2)
        
        for i in range(max(len(v1_parts), len(v2_parts))):
            v1 = v1_parts[i] if i < len(v1_parts) else 0
            v2 = v2_parts[i] if i < len(v2_parts) else 0
            
            if v1 < v2:
                return -1
            elif v1 > v2:
                return 1
        
        return 0
    
    def get_status(self, package: str) -> Optional[DependencyInfo]:
        """
        Get status of a specific dependency.
        
        Args:
            package: Package name
            
        Returns:
            Dependency information or None if not tracked
        """
        return self.dependency_status.get(package)
    
    def is_available(self, package: str) -> bool:
        """
        Check if a dependency is available.
        
        Args:
            package: Package name
            
        Returns:
            True if dependency is available, False otherwise
        """
        dep_info = self.get_status(package)
        return dep_info is not None and dep_info.status == DependencyStatus.AVAILABLE
    
    def get_missing_required(self) -> List[DependencyInfo]:
        """
        Get list of missing required dependencies.
        
        Returns:
            List of missing required dependencies
        """
        return [
            dep for dep in self.dependency_status.values()
            if dep.required and dep.status != DependencyStatus.AVAILABLE
        ]
    
    def get_missing_optional(self) -> List[DependencyInfo]:
        """
        Get list of missing optional dependencies.
        
        Returns:
            List of missing optional dependencies
        """
        return [
            dep for dep in self.dependency_status.values()
            if not dep.required and dep.status != DependencyStatus.AVAILABLE
        ]
    
    def validate_installation(self) -> Tuple[bool, List[str]]:
        """
        Validate that all required dependencies are installed.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        missing = self.get_missing_required()
        
        if not missing:
            return True, []
        
        errors = [
            f"Missing required dependency: {dep.name} ({dep.package})"
            for dep in missing
        ]
        
        return False, errors
    
    def get_installation_report(self) -> str:
        """
        Generate a detailed installation report.
        
        Returns:
            Formatted installation report
        """
        lines = ["=" * 80]
        lines.append("CRONOS AI - Dependency Installation Report")
        lines.append("=" * 80)
        lines.append("")
        
        # Required dependencies
        lines.append("Required Dependencies:")
        lines.append("-" * 80)
        for dep in self.dependency_status.values():
            if dep.required:
                status_symbol = "✓" if dep.status == DependencyStatus.AVAILABLE else "✗"
                version_info = f"v{dep.installed_version}" if dep.installed_version else "N/A"
                lines.append(
                    f"  {status_symbol} {dep.name:30} {version_info:15} [{dep.status.value}]"
                )
                if dep.error_message:
                    lines.append(f"     Error: {dep.error_message}")
        
        lines.append("")
        
        # Optional dependencies
        lines.append("Optional Dependencies (LLM Providers):")
        lines.append("-" * 80)
        for dep in self.dependency_status.values():
            if not dep.required:
                status_symbol = "✓" if dep.status == DependencyStatus.AVAILABLE else "○"
                version_info = f"v{dep.installed_version}" if dep.installed_version else "N/A"
                lines.append(
                    f"  {status_symbol} {dep.name:30} {version_info:15} [{dep.status.value}]"
                )
        
        lines.append("")
        lines.append("=" * 80)
        
        # Validation summary
        is_valid, errors = self.validate_installation()
        if is_valid:
            lines.append("✓ All required dependencies are installed")
        else:
            lines.append("✗ Installation validation failed:")
            for error in errors:
                lines.append(f"  - {error}")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def install_instructions(self) -> str:
        """
        Generate installation instructions for missing dependencies.
        
        Returns:
            Installation instructions
        """
        missing_required = self.get_missing_required()
        missing_optional = self.get_missing_optional()
        
        if not missing_required and not missing_optional:
            return "All dependencies are installed."
        
        lines = ["Installation Instructions:"]
        lines.append("")
        
        if missing_required:
            lines.append("Required dependencies:")
            packages = [dep.package for dep in missing_required]
            lines.append(f"  pip install {' '.join(packages)}")
            lines.append("")
        
        if missing_optional:
            lines.append("Optional dependencies (LLM providers):")
            for dep in missing_optional:
                lines.append(f"  pip install {dep.package}  # {dep.name}")
            lines.append("")
        
        lines.append("Or install all dependencies:")
        lines.append("  pip install -r requirements.txt")
        
        return "\n".join(lines)


# Global dependency manager instance
_dependency_manager: Optional[DependencyManager] = None


def get_dependency_manager() -> DependencyManager:
    """Get or create global dependency manager instance."""
    global _dependency_manager
    if _dependency_manager is None:
        _dependency_manager = DependencyManager()
    return _dependency_manager


def check_dependencies() -> bool:
    """
    Check all dependencies and print report.
    
    Returns:
        True if all required dependencies are available
    """
    manager = get_dependency_manager()
    print(manager.get_installation_report())
    
    is_valid, errors = manager.validate_installation()
    if not is_valid:
        print("\n" + manager.install_instructions())
    
    return is_valid