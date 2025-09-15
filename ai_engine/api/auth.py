"""
CRONOS AI Engine - Authentication

This module provides authentication and authorization for API requests.
"""

import logging
import time
import hashlib
import secrets
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import jwt

from fastapi import HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
import bcrypt

from ..core.config import Config
from ..core.exceptions import AuthenticationException


class APIKeyManager:
    """
    API Key management for authentication.
    
    This class handles API key generation, validation, and management
    for securing API endpoints.
    """
    
    def __init__(self, config: Config):
        """Initialize API key manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # In-memory storage for demo (use database in production)
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        
        # Default admin key for initial setup
        self._create_default_keys()
    
    def _create_default_keys(self) -> None:
        """Create default API keys for initial setup."""
        # Create admin key
        admin_key = self.generate_api_key(
            name="admin",
            permissions=["admin", "read", "write"],
            expires_in_days=365
        )
        
        # Create read-only key
        readonly_key = self.generate_api_key(
            name="readonly",
            permissions=["read"],
            expires_in_days=90
        )
        
        self.logger.info(f"Created default admin API key: {admin_key}")
        self.logger.info(f"Created default readonly API key: {readonly_key}")
    
    def generate_api_key(
        self,
        name: str,
        permissions: list = None,
        expires_in_days: int = 30,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Generate a new API key.
        
        Args:
            name: Key name/identifier
            permissions: List of permissions
            expires_in_days: Expiration period in days
            metadata: Additional metadata
            
        Returns:
            Generated API key
        """
        if permissions is None:
            permissions = ["read"]
        
        if metadata is None:
            metadata = {}
        
        # Generate secure random key
        api_key = f"cronos_ai_{secrets.token_urlsafe(32)}"
        
        # Calculate expiry
        expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        # Store key info
        self.api_keys[api_key] = {
            "name": name,
            "permissions": permissions,
            "created_at": datetime.utcnow(),
            "expires_at": expires_at,
            "last_used": None,
            "usage_count": 0,
            "metadata": metadata,
            "is_active": True
        }
        
        self.logger.info(f"Generated API key for '{name}' with permissions: {permissions}")
        return api_key
    
    def validate_api_key(self, api_key: str) -> Dict[str, Any]:
        """
        Validate API key and return key info.
        
        Args:
            api_key: API key to validate
            
        Returns:
            Key information if valid
            
        Raises:
            AuthenticationException: If key is invalid
        """
        if not api_key or api_key not in self.api_keys:
            raise AuthenticationException("Invalid API key")
        
        key_info = self.api_keys[api_key]
        
        # Check if key is active
        if not key_info["is_active"]:
            raise AuthenticationException("API key is deactivated")
        
        # Check expiry
        if key_info["expires_at"] < datetime.utcnow():
            raise AuthenticationException("API key has expired")
        
        # Update usage info
        key_info["last_used"] = datetime.utcnow()
        key_info["usage_count"] += 1
        
        return key_info.copy()
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        if api_key in self.api_keys:
            self.api_keys[api_key]["is_active"] = False
            self.logger.info(f"Revoked API key: {api_key[:20]}...")
            return True
        return False
    
    def list_api_keys(self) -> Dict[str, Dict[str, Any]]:
        """List all API keys (without actual key values)."""
        result = {}
        for key, info in self.api_keys.items():
            masked_key = f"{key[:20]}...{key[-4:]}"
            result[masked_key] = {
                "name": info["name"],
                "permissions": info["permissions"],
                "created_at": info["created_at"],
                "expires_at": info["expires_at"],
                "last_used": info["last_used"],
                "usage_count": info["usage_count"],
                "is_active": info["is_active"]
            }
        return result
    
    def has_permission(self, api_key: str, required_permission: str) -> bool:
        """Check if API key has required permission."""
        try:
            key_info = self.validate_api_key(api_key)
            return required_permission in key_info["permissions"] or "admin" in key_info["permissions"]
        except AuthenticationException:
            return False


class JWTManager:
    """
    JWT token management for session-based authentication.
    
    This class handles JWT token generation and validation for
    web-based sessions and temporary access tokens.
    """
    
    def __init__(self, config: Config):
        """Initialize JWT manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # JWT configuration
        self.secret_key = getattr(config, 'jwt_secret_key', secrets.token_urlsafe(32))
        self.algorithm = getattr(config, 'jwt_algorithm', 'HS256')
        self.access_token_expire_minutes = getattr(config, 'access_token_expire_minutes', 30)
        self.refresh_token_expire_days = getattr(config, 'refresh_token_expire_days', 7)
        
        # Blacklisted tokens (use Redis in production)
        self.blacklisted_tokens: set = set()
    
    def create_access_token(
        self,
        subject: str,
        permissions: list = None,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT access token.
        
        Args:
            subject: Token subject (user ID, API key name, etc.)
            permissions: List of permissions
            expires_delta: Custom expiration time
            
        Returns:
            JWT token string
        """
        if permissions is None:
            permissions = []
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        payload = {
            "sub": subject,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access",
            "permissions": permissions
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token
    
    def create_refresh_token(
        self,
        subject: str,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT refresh token.
        
        Args:
            subject: Token subject
            expires_delta: Custom expiration time
            
        Returns:
            JWT refresh token string
        """
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        
        payload = {
            "sub": subject,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verify JWT token and return payload.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Token payload if valid
            
        Raises:
            AuthenticationException: If token is invalid
        """
        try:
            # Check if token is blacklisted
            if token in self.blacklisted_tokens:
                raise AuthenticationException("Token has been revoked")
            
            # Decode and verify token
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationException("Token has expired")
        except jwt.JWTError as e:
            raise AuthenticationException(f"Token verification failed: {e}")
    
    def refresh_access_token(self, refresh_token: str) -> str:
        """
        Generate new access token from refresh token.
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            New access token
        """
        payload = self.verify_token(refresh_token)
        
        if payload.get("type") != "refresh":
            raise AuthenticationException("Invalid token type for refresh")
        
        # Create new access token
        return self.create_access_token(
            subject=payload["sub"],
            permissions=[]  # Would fetch from user store in real implementation
        )
    
    def revoke_token(self, token: str) -> bool:
        """Add token to blacklist."""
        try:
            # Verify token is valid before blacklisting
            self.verify_token(token)
            self.blacklisted_tokens.add(token)
            return True
        except AuthenticationException:
            return False


# Global instances (initialized in main application)
api_key_manager: Optional[APIKeyManager] = None
jwt_manager: Optional[JWTManager] = None


def initialize_auth(config: Config) -> None:
    """Initialize authentication managers."""
    global api_key_manager, jwt_manager
    
    api_key_manager = APIKeyManager(config)
    jwt_manager = JWTManager(config)
    
    logging.getLogger(__name__).info("Authentication managers initialized")


async def authenticate_request(credentials: HTTPAuthorizationCredentials) -> Dict[str, Any]:
    """
    Authenticate API request using Bearer token.
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        Authentication context
        
    Raises:
        HTTPException: If authentication fails
    """
    global api_key_manager, jwt_manager
    
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    
    try:
        # Try API key authentication first
        if token.startswith("cronos_ai_"):
            if not api_key_manager:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="API key manager not initialized"
                )
            
            key_info = api_key_manager.validate_api_key(token)
            return {
                "auth_type": "api_key",
                "subject": key_info["name"],
                "permissions": key_info["permissions"],
                "key_info": key_info
            }
        
        # Try JWT token authentication
        else:
            if not jwt_manager:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="JWT manager not initialized"
                )
            
            payload = jwt_manager.verify_token(token)
            return {
                "auth_type": "jwt",
                "subject": payload["sub"],
                "permissions": payload.get("permissions", []),
                "token_payload": payload
            }
    
    except AuthenticationException as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logging.getLogger(__name__).error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error"
        )


def get_api_key() -> str:
    """
    Get a valid API key for testing/development.
    
    Returns:
        A valid API key
    """
    global api_key_manager
    
    if not api_key_manager:
        # Return a mock key for testing
        return "cronos_ai_mock_key_for_testing"
    
    # Return first available admin key
    for key, info in api_key_manager.api_keys.items():
        if info["is_active"] and "admin" in info["permissions"]:
            return key
    
    # Generate a new admin key if none exists
    return api_key_manager.generate_api_key(
        name="auto_generated_admin",
        permissions=["admin", "read", "write"]
    )


def require_permission(required_permission: str):
    """
    Decorator to require specific permission for endpoint access.
    
    Args:
        required_permission: Required permission string
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # This would be implemented as a FastAPI dependency
            # For now, just return the function
            return await func(*args, **kwargs)
        return wrapper
    return decorator


class RateLimiter:
    """
    Rate limiting for API requests.
    
    This class implements token bucket rate limiting per API key/user.
    """
    
    def __init__(self, max_requests: int = 100, time_window: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: Dict[str, list] = {}
        self.logger = logging.getLogger(__name__)
    
    def is_allowed(self, identifier: str) -> bool:
        """
        Check if request is allowed for identifier.
        
        Args:
            identifier: API key or user identifier
            
        Returns:
            True if request is allowed
        """
        current_time = time.time()
        
        # Initialize if not exists
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        # Clean old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if current_time - req_time < self.time_window
        ]
        
        # Check limit
        if len(self.requests[identifier]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[identifier].append(current_time)
        return True
    
    def get_remaining_requests(self, identifier: str) -> int:
        """Get remaining requests for identifier."""
        current_time = time.time()
        
        if identifier not in self.requests:
            return self.max_requests
        
        # Clean old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if current_time - req_time < self.time_window
        ]
        
        return max(0, self.max_requests - len(self.requests[identifier]))
    
    def reset_limit(self, identifier: str) -> None:
        """Reset rate limit for identifier."""
        if identifier in self.requests:
            del self.requests[identifier]


# Global rate limiter instance
rate_limiter = RateLimiter()


def check_rate_limit(identifier: str) -> bool:
    """
    Check rate limit for identifier.
    
    Args:
        identifier: Request identifier
        
    Returns:
        True if request is allowed
    """
    return rate_limiter.is_allowed(identifier)


def get_rate_limit_info(identifier: str) -> Dict[str, Any]:
    """
    Get rate limit information for identifier.
    
    Args:
        identifier: Request identifier
        
    Returns:
        Rate limit information
    """
    return {
        "max_requests": rate_limiter.max_requests,
        "time_window": rate_limiter.time_window,
        "remaining_requests": rate_limiter.get_remaining_requests(identifier),
        "reset_time": time.time() + rate_limiter.time_window
    }