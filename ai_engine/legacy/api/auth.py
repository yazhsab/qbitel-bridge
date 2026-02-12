"""
QBITEL Engine - Legacy System Whisperer Authentication & Authorization

Authentication and authorization for Legacy System Whisperer API endpoints.
"""

import jwt
from jwt import ExpiredSignatureError, PyJWTError
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from fastapi import HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN

from ...core.config import Config, get_config
from ..logging import get_legacy_logger
from ..models import LegacySystemContext

# Initialize security and logging
security = HTTPBearer(auto_error=False)
logger = get_legacy_logger(__name__)


# Permission constants
class LegacyPermissions:
    """Permission constants for Legacy System Whisperer."""

    # System management
    SYSTEM_REGISTER = "legacy:system:register"
    SYSTEM_VIEW = "legacy:system:view"
    SYSTEM_MODIFY = "legacy:system:modify"
    SYSTEM_DELETE = "legacy:system:delete"
    SYSTEM_ADMIN = "legacy:system:admin"

    # Analysis and prediction
    ANALYSIS_REQUEST = "legacy:analysis:request"
    ANALYSIS_VIEW = "legacy:analysis:view"
    PREDICTION_REQUEST = "legacy:prediction:request"
    PREDICTION_VIEW = "legacy:prediction:view"

    # Knowledge management
    KNOWLEDGE_CAPTURE = "legacy:knowledge:capture"
    KNOWLEDGE_VIEW = "legacy:knowledge:view"
    KNOWLEDGE_MODIFY = "legacy:knowledge:modify"
    KNOWLEDGE_SEARCH = "legacy:knowledge:search"

    # Decision support
    DECISION_REQUEST = "legacy:decision:request"
    DECISION_VIEW = "legacy:decision:view"
    DECISION_APPROVE = "legacy:decision:approve"

    # Maintenance scheduling
    MAINTENANCE_SCHEDULE = "legacy:maintenance:schedule"
    MAINTENANCE_VIEW = "legacy:maintenance:view"
    MAINTENANCE_MODIFY = "legacy:maintenance:modify"
    MAINTENANCE_APPROVE = "legacy:maintenance:approve"

    # Monitoring and alerts
    MONITORING_VIEW = "legacy:monitoring:view"
    ALERT_MANAGE = "legacy:alert:manage"
    ALERT_VIEW = "legacy:alert:view"

    # Configuration and administration
    CONFIG_VIEW = "legacy:config:view"
    CONFIG_MODIFY = "legacy:config:modify"
    SERVICE_ADMIN = "legacy:service:admin"

    # Bulk operations
    BULK_OPERATIONS = "legacy:bulk:operations"


class UserRole:
    """User role definitions with associated permissions."""

    VIEWER = "legacy_viewer"
    OPERATOR = "legacy_operator"
    ANALYST = "legacy_analyst"
    MAINTAINER = "legacy_maintainer"
    ADMIN = "legacy_admin"
    SUPER_ADMIN = "legacy_super_admin"


# Role-to-permissions mapping
# Define base permissions for each role
_VIEWER_PERMS = [
    LegacyPermissions.SYSTEM_VIEW,
    LegacyPermissions.ANALYSIS_VIEW,
    LegacyPermissions.PREDICTION_VIEW,
    LegacyPermissions.KNOWLEDGE_VIEW,
    LegacyPermissions.DECISION_VIEW,
    LegacyPermissions.MAINTENANCE_VIEW,
    LegacyPermissions.MONITORING_VIEW,
    LegacyPermissions.ALERT_VIEW,
    LegacyPermissions.CONFIG_VIEW,
]

_OPERATOR_PERMS = [
    *_VIEWER_PERMS,
    LegacyPermissions.SYSTEM_MODIFY,
    LegacyPermissions.ANALYSIS_REQUEST,
    LegacyPermissions.PREDICTION_REQUEST,
    LegacyPermissions.KNOWLEDGE_SEARCH,
]

_ANALYST_PERMS = [
    *_OPERATOR_PERMS,
    LegacyPermissions.KNOWLEDGE_CAPTURE,
    LegacyPermissions.DECISION_REQUEST,
    LegacyPermissions.MAINTENANCE_SCHEDULE,
]

_MAINTAINER_PERMS = [
    *_ANALYST_PERMS,
    LegacyPermissions.KNOWLEDGE_MODIFY,
    LegacyPermissions.MAINTENANCE_MODIFY,
    LegacyPermissions.MAINTENANCE_APPROVE,
]

_ADMIN_PERMS = [
    *_MAINTAINER_PERMS,
    LegacyPermissions.SYSTEM_REGISTER,
    LegacyPermissions.SYSTEM_DELETE,
    LegacyPermissions.DECISION_APPROVE,
    LegacyPermissions.ALERT_MANAGE,
    LegacyPermissions.CONFIG_MODIFY,
    LegacyPermissions.BULK_OPERATIONS,
]

_SUPER_ADMIN_PERMS = [
    *_ADMIN_PERMS,
    LegacyPermissions.SYSTEM_ADMIN,
    LegacyPermissions.SERVICE_ADMIN,
]

ROLE_PERMISSIONS = {
    UserRole.VIEWER: _VIEWER_PERMS,
    UserRole.OPERATOR: _OPERATOR_PERMS,
    UserRole.ANALYST: _ANALYST_PERMS,
    UserRole.MAINTAINER: _MAINTAINER_PERMS,
    UserRole.ADMIN: _ADMIN_PERMS,
    UserRole.SUPER_ADMIN: _SUPER_ADMIN_PERMS,
}


class User:
    """User model for authentication and authorization."""

    def __init__(
        self,
        user_id: str,
        username: str,
        email: str,
        roles: List[str] = None,
        permissions: List[str] = None,
        system_access: List[str] = None,
        metadata: Dict[str, Any] = None,
    ):
        self.user_id = user_id
        self.username = username
        self.email = email
        self.roles = roles or []
        self.permissions = self._resolve_permissions(permissions or [])
        self.system_access = system_access or []  # Specific system IDs user can access
        self.metadata = metadata or {}
        self.authenticated_at = datetime.now()

    def _resolve_permissions(self, explicit_permissions: List[str]) -> List[str]:
        """Resolve permissions from roles and explicit permissions."""
        all_permissions = set(explicit_permissions)

        # Add permissions from roles
        for role in self.roles:
            role_perms = ROLE_PERMISSIONS.get(role, [])
            all_permissions.update(role_perms)

        return list(all_permissions)

    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions

    def has_role(self, role: str) -> bool:
        """Check if user has specific role."""
        return role in self.roles

    def can_access_system(self, system_id: str) -> bool:
        """Check if user can access specific system."""
        # Super admin can access all systems
        if self.has_role(UserRole.SUPER_ADMIN):
            return True

        # Check system-specific access
        if self.system_access:
            return system_id in self.system_access

        # If no system restrictions, allow access based on permissions
        return self.has_permission(LegacyPermissions.SYSTEM_VIEW)

    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "roles": self.roles,
            "permissions": self.permissions,
            "system_access": self.system_access,
            "authenticated_at": self.authenticated_at.isoformat(),
        }


class LegacySystemAuth:
    """Authentication and authorization handler for Legacy System Whisperer."""

    def __init__(self, config: Config):
        self.config = config
        self.secret_key = config.security.jwt_secret or "legacy-system-whisperer-secret"
        self.algorithm = "HS256"
        self.token_expiry_hours = config.security.jwt_expiry_hours

        # Mock user database (in production, would integrate with actual user service)
        self.users_db = self._initialize_mock_users()

        logger.info("Legacy System Whisperer authentication initialized")

    def _initialize_mock_users(self) -> Dict[str, User]:
        """Initialize mock users for development/testing."""
        return {
            "admin": User(
                user_id="usr_001",
                username="admin",
                email="admin@qbitel.com",
                roles=[UserRole.SUPER_ADMIN],
            ),
            "analyst": User(
                user_id="usr_002",
                username="analyst",
                email="analyst@qbitel.com",
                roles=[UserRole.ANALYST],
            ),
            "operator": User(
                user_id="usr_003",
                username="operator",
                email="operator@qbitel.com",
                roles=[UserRole.OPERATOR],
            ),
            "viewer": User(
                user_id="usr_004",
                username="viewer",
                email="viewer@qbitel.com",
                roles=[UserRole.VIEWER],
            ),
        }

    def create_access_token(self, user: User) -> str:
        """Create JWT access token for user."""
        payload = {
            "sub": user.user_id,
            "username": user.username,
            "email": user.email,
            "roles": user.roles,
            "permissions": user.permissions,
            "system_access": user.system_access,
            "iat": datetime.now(),
            "exp": datetime.now() + timedelta(hours=self.token_expiry_hours),
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        logger.info(f"Access token created for user: {user.username}")
        return token

    def verify_token(self, token: str) -> User:
        """Verify JWT token and return user."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Check expiration
            exp_timestamp = payload.get("exp")
            if exp_timestamp and datetime.fromtimestamp(exp_timestamp) < datetime.now():
                raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Token expired")

            # Reconstruct user from payload
            user = User(
                user_id=payload["sub"],
                username=payload["username"],
                email=payload["email"],
                roles=payload.get("roles", []),
                permissions=payload.get("permissions", []),
                system_access=payload.get("system_access", []),
            )

            return user

        except ExpiredSignatureError:
            logger.warning("Token expired")
            raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Token expired")
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Invalid token")
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Token verification failed")

    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username/password (mock implementation)."""
        # Mock authentication - in production, would validate against real auth service
        user = self.users_db.get(username)
        if user and password == "password":  # Mock password validation
            logger.info(f"User authenticated: {username}")
            return user

        logger.warning(f"Authentication failed for user: {username}")
        return None

    def authorize_permission(self, user: User, required_permission: str) -> bool:
        """Check if user has required permission."""
        has_permission = user.has_permission(required_permission)

        if not has_permission:
            logger.warning(f"Authorization failed: user {user.username} lacks permission {required_permission}")

        return has_permission

    def authorize_system_access(self, user: User, system_id: str) -> bool:
        """Check if user can access specific system."""
        can_access = user.can_access_system(system_id)

        if not can_access:
            logger.warning(f"System access denied: user {user.username} cannot access system {system_id}")

        return can_access


# Global auth instance
_auth_instance: Optional[LegacySystemAuth] = None


def get_auth() -> LegacySystemAuth:
    """Get authentication instance."""
    global _auth_instance
    if _auth_instance is None:
        config = get_config()
        _auth_instance = LegacySystemAuth(config)
    return _auth_instance


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> User:
    """
    Dependency to get current authenticated user.

    Extracts and validates JWT token from Authorization header.
    """
    if not credentials:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    auth = get_auth()

    try:
        user = auth.verify_token(credentials.credentials)
        return user

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_permission(permission: str):
    """
    Decorator factory for requiring specific permission.

    Usage:
        @app.get("/endpoint")
        async def protected_endpoint(user: User = Depends(require_permission("permission"))):
            ...
    """

    def permission_dependency(user: User = Depends(get_current_user)) -> User:
        auth = get_auth()

        if not auth.authorize_permission(user, permission):
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission}",
            )

        return user

    return permission_dependency


def require_role(role: str):
    """
    Decorator factory for requiring specific role.

    Usage:
        @app.get("/endpoint")
        async def protected_endpoint(user: User = Depends(require_role("admin"))):
            ...
    """

    def role_dependency(user: User = Depends(get_current_user)) -> User:
        if not user.has_role(role):
            raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail=f"Role required: {role}")

        return user

    return role_dependency


def require_system_access(system_id: str):
    """
    Decorator factory for requiring access to specific system.

    Usage:
        @app.get("/systems/{system_id}")
        async def get_system(
            system_id: str,
            user: User = Depends(require_system_access(system_id))
        ):
            ...
    """

    def system_access_dependency(user: User = Depends(get_current_user)) -> User:
        auth = get_auth()

        if not auth.authorize_system_access(user, system_id):
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN,
                detail=f"System access denied: {system_id}",
            )

        return user

    return system_access_dependency


def require_any_permission(*permissions: str):
    """
    Decorator factory for requiring any of the specified permissions.

    Usage:
        @app.get("/endpoint")
        async def endpoint(
            user: User = Depends(require_any_permission("perm1", "perm2"))
        ):
            ...
    """

    def any_permission_dependency(user: User = Depends(get_current_user)) -> User:
        auth = get_auth()

        for permission in permissions:
            if auth.authorize_permission(user, permission):
                return user

        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail=f"One of these permissions required: {', '.join(permissions)}",
        )

    return any_permission_dependency


def require_all_permissions(*permissions: str):
    """
    Decorator factory for requiring all of the specified permissions.

    Usage:
        @app.get("/endpoint")
        async def endpoint(
            user: User = Depends(require_all_permissions("perm1", "perm2"))
        ):
            ...
    """

    def all_permissions_dependency(user: User = Depends(get_current_user)) -> User:
        auth = get_auth()

        for permission in permissions:
            if not auth.authorize_permission(user, permission):
                raise HTTPException(
                    status_code=HTTP_403_FORBIDDEN,
                    detail=f"Permission required: {permission}",
                )

        return user

    return all_permissions_dependency


# Authentication endpoint (for login)
async def authenticate_user_endpoint(username: str, password: str) -> Dict[str, Any]:
    """
    Authenticate user and return access token.

    This would typically be part of a separate auth service,
    but included here for completeness.
    """
    auth = get_auth()
    user = auth.authenticate_user(username, password)

    if not user:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    access_token = auth.create_access_token(user)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": auth.token_expiry_hours * 3600,
        "user": user.to_dict(),
    }


# Utility functions
def get_user_permissions(user: User) -> List[str]:
    """Get all permissions for user."""
    return user.permissions


def get_user_systems(user: User) -> List[str]:
    """Get systems user can access."""
    return user.system_access


def check_system_permission(user: User, system_id: str, permission: str) -> bool:
    """Check if user has permission for specific system."""
    auth = get_auth()
    return auth.authorize_permission(user, permission) and auth.authorize_system_access(user, system_id)


def is_system_admin(user: User) -> bool:
    """Check if user is system administrator."""
    return user.has_role(UserRole.ADMIN) or user.has_role(UserRole.SUPER_ADMIN)


def can_manage_alerts(user: User) -> bool:
    """Check if user can manage alerts."""
    return user.has_permission(LegacyPermissions.ALERT_MANAGE)


def can_schedule_maintenance(user: User) -> bool:
    """Check if user can schedule maintenance."""
    return user.has_permission(LegacyPermissions.MAINTENANCE_SCHEDULE)


def can_capture_knowledge(user: User) -> bool:
    """Check if user can capture expert knowledge."""
    return user.has_permission(LegacyPermissions.KNOWLEDGE_CAPTURE)


# Security audit logging
def log_security_event(
    event_type: str,
    user: Optional[User] = None,
    resource: Optional[str] = None,
    action: Optional[str] = None,
    outcome: str = "success",
    details: Optional[Dict[str, Any]] = None,
):
    """Log security-related events for audit trail."""
    log_data = {
        "event_type": event_type,
        "timestamp": datetime.now().isoformat(),
        "user_id": user.user_id if user else None,
        "username": user.username if user else None,
        "resource": resource,
        "action": action,
        "outcome": outcome,
        "details": details or {},
    }

    logger.info(f"Security Event: {event_type}", extra={"security_event": log_data})


# Rate limiting helpers (basic implementation)
_rate_limit_cache: Dict[str, List[datetime]] = {}


def check_rate_limit(user_id: str, limit: int, window_seconds: int) -> bool:
    """
    Check if user is within rate limits.

    Args:
        user_id: User identifier
        limit: Maximum requests allowed
        window_seconds: Time window in seconds

    Returns:
        True if within limits, False otherwise
    """
    now = datetime.now()
    cutoff = now - timedelta(seconds=window_seconds)

    if user_id not in _rate_limit_cache:
        _rate_limit_cache[user_id] = []

    # Clean old entries
    _rate_limit_cache[user_id] = [timestamp for timestamp in _rate_limit_cache[user_id] if timestamp > cutoff]

    # Check limit
    if len(_rate_limit_cache[user_id]) >= limit:
        return False

    # Add current request
    _rate_limit_cache[user_id].append(now)
    return True


def require_rate_limit(limit: int, window_seconds: int = 60):
    """
    Decorator factory for rate limiting.

    Usage:
        @app.get("/endpoint")
        async def endpoint(user: User = Depends(require_rate_limit(10, 60))):
            ...
    """

    def rate_limit_dependency(user: User = Depends(get_current_user)) -> User:
        if not check_rate_limit(user.user_id, limit, window_seconds):
            raise HTTPException(
                status_code=429,  # Too Many Requests
                detail=f"Rate limit exceeded: {limit} requests per {window_seconds} seconds",
            )
        return user

    return rate_limit_dependency
