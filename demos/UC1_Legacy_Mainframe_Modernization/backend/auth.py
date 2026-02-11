"""
QBITEL - UC1 Demo Authentication Module

Production-grade authentication middleware supporting:
- API Key authentication
- JWT token authentication
- Role-based access control
"""

import os
import time
import hmac
import hashlib
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

from fastapi import HTTPException, Security, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from pydantic import BaseModel

import logging

logger = logging.getLogger(__name__)


class AuthMethod(str, Enum):
    """Supported authentication methods."""
    API_KEY = "api_key"
    JWT = "jwt"
    NONE = "none"


class Role(str, Enum):
    """User roles for RBAC."""
    ADMIN = "admin"
    OPERATOR = "operator"
    ANALYST = "analyst"
    VIEWER = "viewer"


@dataclass
class AuthConfig:
    """Authentication configuration."""
    enabled: bool = True
    method: AuthMethod = AuthMethod.API_KEY
    api_key_header: str = "X-API-Key"
    jwt_secret: str = field(default_factory=lambda: os.getenv("JWT_SECRET", secrets.token_hex(32)))
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class User:
    """Authenticated user."""
    user_id: str
    username: str
    roles: List[Role]
    api_key: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    authenticated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TokenPayload(BaseModel):
    """JWT token payload."""
    sub: str  # user_id
    username: str
    roles: List[str]
    exp: int
    iat: int


# Security schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


class AuthManager:
    """Authentication manager."""

    def __init__(self, config: Optional[AuthConfig] = None):
        self.config = config or AuthConfig()
        self._api_keys: Dict[str, User] = {}
        self._rate_limits: Dict[str, List[float]] = {}
        self._setup_default_keys()

    def _setup_default_keys(self):
        """Setup default API keys from environment."""
        # Demo API key (for testing)
        demo_key = os.getenv("DEMO_API_KEY", "demo-key-for-testing-only")
        self._api_keys[demo_key] = User(
            user_id="demo-user",
            username="demo",
            roles=[Role.ANALYST],
            api_key=demo_key,
        )

        # Admin API key
        admin_key = os.getenv("ADMIN_API_KEY")
        if admin_key:
            self._api_keys[admin_key] = User(
                user_id="admin-user",
                username="admin",
                roles=[Role.ADMIN],
                api_key=admin_key,
            )

        # Production API keys from env (comma-separated)
        prod_keys = os.getenv("API_KEYS", "")
        for key in prod_keys.split(","):
            key = key.strip()
            if key and key not in self._api_keys:
                self._api_keys[key] = User(
                    user_id=f"user-{hashlib.sha256(key.encode()).hexdigest()[:8]}",
                    username=f"api-user-{len(self._api_keys)}",
                    roles=[Role.OPERATOR],
                    api_key=key,
                )

    def validate_api_key(self, api_key: str) -> Optional[User]:
        """Validate API key and return user."""
        if not api_key:
            return None

        user = self._api_keys.get(api_key)
        if user:
            logger.debug(f"API key authenticated: {user.username}")
            return user

        return None

    def validate_jwt(self, token: str) -> Optional[User]:
        """Validate JWT token and return user."""
        try:
            import jwt
            payload = jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=[self.config.jwt_algorithm],
            )

            # Check expiration
            if payload.get("exp", 0) < time.time():
                logger.warning("JWT token expired")
                return None

            return User(
                user_id=payload["sub"],
                username=payload["username"],
                roles=[Role(r) for r in payload.get("roles", [])],
            )
        except Exception as e:
            logger.warning(f"JWT validation failed: {e}")
            return None

    def create_jwt(self, user: User) -> str:
        """Create JWT token for user."""
        import jwt

        now = int(time.time())
        payload = {
            "sub": user.user_id,
            "username": user.username,
            "roles": [r.value for r in user.roles],
            "iat": now,
            "exp": now + (self.config.jwt_expiry_hours * 3600),
        }

        return jwt.encode(
            payload,
            self.config.jwt_secret,
            algorithm=self.config.jwt_algorithm,
        )

    def check_rate_limit(self, identifier: str) -> bool:
        """Check if request is within rate limits."""
        now = time.time()
        window_start = now - self.config.rate_limit_window_seconds

        # Clean old entries
        if identifier in self._rate_limits:
            self._rate_limits[identifier] = [
                ts for ts in self._rate_limits[identifier]
                if ts > window_start
            ]
        else:
            self._rate_limits[identifier] = []

        # Check limit
        if len(self._rate_limits[identifier]) >= self.config.rate_limit_requests:
            return False

        # Record request
        self._rate_limits[identifier].append(now)
        return True

    def check_permission(self, user: User, required_roles: List[Role]) -> bool:
        """Check if user has required roles."""
        if Role.ADMIN in user.roles:
            return True

        return any(role in user.roles for role in required_roles)


# Global auth manager
_auth_manager: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    """Get global auth manager instance."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager


def set_auth_manager(manager: AuthManager):
    """Set global auth manager instance."""
    global _auth_manager
    _auth_manager = manager


# =============================================================================
# FastAPI Dependencies
# =============================================================================

async def get_api_key(
    api_key: Optional[str] = Security(api_key_header),
) -> Optional[str]:
    """Extract API key from header."""
    return api_key


async def get_bearer_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> Optional[str]:
    """Extract bearer token from header."""
    if credentials:
        return credentials.credentials
    return None


async def authenticate(
    request: Request,
    api_key: Optional[str] = Depends(get_api_key),
    bearer_token: Optional[str] = Depends(get_bearer_token),
) -> User:
    """Authenticate request and return user."""
    auth_manager = get_auth_manager()

    # Check if auth is disabled
    if not auth_manager.config.enabled:
        return User(
            user_id="anonymous",
            username="anonymous",
            roles=[Role.VIEWER],
        )

    # Check rate limit by IP
    client_ip = request.client.host if request.client else "unknown"
    if not auth_manager.check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(auth_manager.config.rate_limit_window_seconds)},
        )

    # Try API key authentication
    if api_key:
        user = auth_manager.validate_api_key(api_key)
        if user:
            return user

    # Try JWT authentication
    if bearer_token:
        user = auth_manager.validate_jwt(bearer_token)
        if user:
            return user

    # Check if authentication is required
    auth_required = os.getenv("AUTH_REQUIRED", "false").lower() == "true"
    if auth_required:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer, ApiKey"},
        )

    # Return anonymous user
    return User(
        user_id="anonymous",
        username="anonymous",
        roles=[Role.VIEWER],
    )


async def require_auth(user: User = Depends(authenticate)) -> User:
    """Require authentication."""
    if user.user_id == "anonymous":
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
        )
    return user


def require_roles(*roles: Role):
    """Create dependency that requires specific roles."""
    async def check_roles(user: User = Depends(authenticate)) -> User:
        auth_manager = get_auth_manager()
        if not auth_manager.check_permission(user, list(roles)):
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required roles: {[r.value for r in roles]}",
            )
        return user
    return check_roles


# Role-specific dependencies
require_admin = require_roles(Role.ADMIN)
require_operator = require_roles(Role.ADMIN, Role.OPERATOR)
require_analyst = require_roles(Role.ADMIN, Role.OPERATOR, Role.ANALYST)


# =============================================================================
# API Key Management Endpoints (for admin use)
# =============================================================================

class CreateAPIKeyRequest(BaseModel):
    """Request to create new API key."""
    username: str
    roles: List[str] = ["analyst"]


class APIKeyResponse(BaseModel):
    """API key response."""
    api_key: str
    user_id: str
    username: str
    roles: List[str]
    created_at: str


def create_api_key(request: CreateAPIKeyRequest) -> APIKeyResponse:
    """Create new API key (admin only)."""
    auth_manager = get_auth_manager()

    # Generate secure API key
    api_key = secrets.token_urlsafe(32)

    # Create user
    roles = [Role(r) for r in request.roles if r in [e.value for e in Role]]
    if not roles:
        roles = [Role.ANALYST]

    user = User(
        user_id=f"user-{secrets.token_hex(4)}",
        username=request.username,
        roles=roles,
        api_key=api_key,
    )

    # Store
    auth_manager._api_keys[api_key] = user

    logger.info(f"Created API key for user: {user.username}")

    return APIKeyResponse(
        api_key=api_key,
        user_id=user.user_id,
        username=user.username,
        roles=[r.value for r in user.roles],
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def revoke_api_key(api_key: str) -> bool:
    """Revoke API key (admin only)."""
    auth_manager = get_auth_manager()

    if api_key in auth_manager._api_keys:
        user = auth_manager._api_keys.pop(api_key)
        logger.info(f"Revoked API key for user: {user.username}")
        return True

    return False
