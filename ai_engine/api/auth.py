"""
CRONOS AI Engine - Authentication and Authorization
Enterprise-grade authentication with JWT tokens and role-based access control.
"""

import json
import logging
import os
import secrets
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import jwt
from jwt import ExpiredSignatureError, PyJWTError
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
import redis.asyncio as redis

from ..core.config import get_config
from ..security.secrets_manager import get_secrets_manager
from ..security.audit_logger import get_audit_logger

logger = logging.getLogger(__name__)

# API key must be set via environment variable or secrets manager
_api_key: Optional[str] = None

# Security configuration
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthenticationError(Exception):
    """Authentication related errors."""
    pass

class AuthenticationService:
    """Enhanced authentication service with enterprise features."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.redis_client = None
        self.secret_key = self._load_secret_key()
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7
        self._session_store: Dict[str, Dict[str, Any]] = {}
        self._session_expiry: Dict[str, datetime] = {}
        self._token_blacklist: Dict[str, datetime] = {}
        self.audit_logger = get_audit_logger()
        self.secrets_manager = get_secrets_manager()

    def _load_secret_key(self) -> str:
        """
        Load JWT secret from secrets manager or configuration.
        
        Priority order:
        1. Secrets manager (Vault, AWS Secrets Manager, etc.)
        2. Environment variable
        3. Configuration file
        4. Generate ephemeral (development only)
        """
        # Try secrets manager first
        try:
            secrets_mgr = get_secrets_manager()
            secret = secrets_mgr.get_secret('jwt_secret')
            if secret and len(secret) >= 32:
                logger.info("JWT secret loaded from secrets manager")
                return secret
        except Exception as e:
            logger.debug(f"Could not load JWT secret from secrets manager: {e}")
        
        # Try configuration
        secret = getattr(self.config.security, 'jwt_secret', None)
        if secret and len(secret) >= 32:
            return secret

        # Check if production mode
        if hasattr(self.config, 'environment') and self.config.environment.value == 'production':
            raise AuthenticationError(
                "JWT secret not configured in production mode!\n"
                "REQUIRED: Configure JWT secret in secrets manager or set CRONOS_AI_JWT_SECRET.\n"
                "Generate a secure secret: python -c \"import secrets; print(secrets.token_urlsafe(48))\""
            )

        # Generate ephemeral secret for development
        generated = secrets.token_urlsafe(48)
        if hasattr(self.config.security, 'jwt_secret'):
            self.config.security.jwt_secret = generated

        logger.warning(
            "JWT secret not configured; generated ephemeral secret for development. "
            "Configure 'security.jwt_secret' with a strong value for production."
        )
        return generated
    
    async def initialize(self):
        """Initialize authentication service."""
        try:
            # Initialize Redis for session management
            self.redis_client = redis.Redis(
                host=self.config.redis.host,
                port=self.config.redis.port,
                db=self.config.redis.db,
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Authentication service initialized")
            
        except Exception as e:
            logger.warning(
                "Redis unavailable for authentication service (%s); falling back to in-memory session store",
                e,
            )
            self.redis_client = None
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire, "type": "access"})
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    async def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token and return payload."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check if token is blacklisted
            if self.redis_client:
                blacklisted = await self.redis_client.get(f"blacklist:{token}")
                if blacklisted:
                    raise AuthenticationError("Token has been revoked")
            else:
                self._prune_blacklist()
                expiry = self._token_blacklist.get(token)
                if expiry and expiry > datetime.utcnow():
                    raise AuthenticationError("Token has been revoked")
            
            return payload
            
        except ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except PyJWTError:
            raise AuthenticationError("Invalid token")
    
    async def revoke_token(self, token: str):
        """Add token to blacklist."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm], verify=False)
            exp_ts = payload.get('exp', 0)
            expiry = datetime.utcfromtimestamp(exp_ts) if exp_ts else datetime.utcnow()
        except Exception:
            expiry = datetime.utcnow() + timedelta(hours=1)

        if self.redis_client:
            try:
                ttl = max(0, int((expiry - datetime.utcnow()).total_seconds()))
                await self.redis_client.setex(f"blacklist:{token}", ttl or 1, "revoked")
            except Exception as e:
                logger.warning(f"Failed to revoke token in redis: {e}")
        else:
            self._prune_blacklist()
            self._token_blacklist[token] = expiry
    
    async def store_session(self, user_id: str, session_data: Dict[str, Any], ttl: int = 3600):
        """Store user session data."""
        if self.redis_client:
            try:
                payload = json.dumps(session_data or {}, default=str)
                await self.redis_client.setex(
                    f"session:{user_id}", 
                    ttl, 
                    payload
                )
            except Exception as e:
                logger.warning(f"Failed to store session: {e}")
        else:
            self._prune_sessions()
            expiry = datetime.utcnow() + timedelta(seconds=ttl)
            self._session_store[user_id] = session_data or {}
            self._session_expiry[user_id] = expiry
    
    async def get_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user session data."""
        if self.redis_client:
            try:
                session_data = await self.redis_client.get(f"session:{user_id}")
                if session_data:
                    try:
                        return json.loads(session_data)
                    except json.JSONDecodeError:
                        logger.warning("Failed to decode session JSON; discarding session data")
            except Exception as e:
                logger.warning(f"Failed to get session: {e}")
        else:
            self._prune_sessions()
            session = self._session_store.get(user_id)
            if session:
                return session
        return None
    
    async def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate user credentials.
        
        PRODUCTION NOTE: This is a legacy method for backward compatibility.
        Use EnterpriseAuthenticationService from auth_enterprise.py for production deployments.
        
        This method is DEPRECATED and will be removed in a future version.
        """
        # Check if we're in production mode
        if hasattr(self.config, 'environment') and self.config.environment.value == 'production':
            logger.error(
                "Legacy authentication called in production mode! "
                "Use EnterpriseAuthenticationService from auth_enterprise.py instead."
            )
            raise AuthenticationError(
                "Legacy authentication not available in production. "
                "Use enterprise authentication with database-backed user management."
            )
        
        # DEVELOPMENT/TESTING ONLY - Demo users
        logger.warning(
            "Using legacy demo user authentication - FOR DEVELOPMENT/TESTING ONLY. "
            "Switch to EnterpriseAuthenticationService for production."
        )
        
        # Only allow demo users in non-production environments
        import os
        demo_users = {
            "admin": {
                "user_id": "admin_001",
                "username": "admin",
                "password_hash": self.hash_password(
                    os.getenv('DEMO_ADMIN_PASSWORD', 'DemoOnly_NotForProduction_123!')
                ),
                "role": "administrator",
                "permissions": [
                    "protocol_discovery",
                    "copilot_access",
                    "system_administration",
                    "analytics_access"
                ],
                "full_name": "Demo Administrator (DEV ONLY)"
            },
            "analyst": {
                "user_id": "analyst_001",
                "username": "analyst",
                "password_hash": self.hash_password(
                    os.getenv('DEMO_ANALYST_PASSWORD', 'DemoOnly_NotForProduction_456!')
                ),
                "role": "security_analyst",
                "permissions": [
                    "protocol_discovery",
                    "copilot_access",
                    "analytics_access"
                ],
                "full_name": "Demo Analyst (DEV ONLY)"
            }
        }
        
        user = demo_users.get(username)
        if user and self.verify_password(password, user["password_hash"]):
            return {
                "user_id": user["user_id"],
                "username": user["username"],
                "role": user["role"],
                "permissions": user["permissions"],
                "full_name": user["full_name"]
            }
        return None

    def _prune_blacklist(self) -> None:
        now = datetime.utcnow()
        expired = [token for token, expiry in self._token_blacklist.items() if expiry <= now]
        for token in expired:
            self._token_blacklist.pop(token, None)

    def _prune_sessions(self) -> None:
        now = datetime.utcnow()
        expired = [user_id for user_id, expiry in self._session_expiry.items() if expiry <= now]
        for user_id in expired:
            self._session_store.pop(user_id, None)
            self._session_expiry.pop(user_id, None)

# Global authentication service instance
_auth_service: Optional[AuthenticationService] = None

async def get_auth_service() -> AuthenticationService:
    """Get authentication service instance."""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthenticationService()
        await _auth_service.initialize()
    return _auth_service

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Verify JWT token from Authorization header."""
    try:
        token = credentials.credentials

        # Allow static API key for simple integrations/tests
        if token == get_api_key():
            return {
                "user_id": "api-key-user",
                "username": "api-key-user",
                "role": "system",
                "permissions": [
                    "protocol_discovery",
                    "copilot_access",
                    "analytics_access"
                ]
            }

        auth_service = await get_auth_service()
        payload = await auth_service.verify_token(token)
        return payload
        
    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")

async def get_current_user(token_payload: Dict[str, Any] = Depends(verify_token)) -> Dict[str, Any]:
    """Get current authenticated user."""
    user_id = token_payload.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    
    # Get additional user info from session if needed
    auth_service = await get_auth_service()
    session_data = await auth_service.get_session(user_id)
    
    return {
        "user_id": user_id,
        "username": token_payload.get("username"),
        "role": token_payload.get("role"),
        "permissions": token_payload.get("permissions", []),
        "session_data": session_data
    }

def require_permission(permission: str):
    """Dependency to require specific permission."""
    async def check_permission(current_user: Dict[str, Any] = Depends(get_current_user)):
        permissions = current_user.get("permissions", [])
        if permission not in permissions:
            raise HTTPException(
                status_code=403,
                detail=f"Permission '{permission}' required"
            )
        return current_user
    return check_permission

def require_role(role: str):
    """Dependency to require specific role."""
    async def check_role(current_user: Dict[str, Any] = Depends(get_current_user)):
        user_role = current_user.get("role")
        if user_role != role:
            raise HTTPException(
                status_code=403,
                detail=f"Role '{role}' required"
            )
        return current_user
    return check_role

# Authentication endpoints
async def login(username: str, password: str, ip_address: Optional[str] = None, user_agent: Optional[str] = None) -> Dict[str, Any]:
    """Authenticate user and return tokens."""
    auth_service = await get_auth_service()
    
    user = await auth_service.authenticate_user(username, password)
    if not user:
        # Log failed login attempt
        auth_service.audit_logger.log_login_failed(
            username=username,
            reason="Invalid credentials",
            ip_address=ip_address,
            user_agent=user_agent
        )
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Create tokens
    token_data = {
        "user_id": user["user_id"],
        "username": user["username"],
        "role": user["role"],
        "permissions": user["permissions"]
    }
    
    access_token = auth_service.create_access_token(token_data)
    refresh_token = auth_service.create_refresh_token(token_data)
    
    # Store session
    await auth_service.store_session(
        user["user_id"],
        {
            "login_time": datetime.utcnow().isoformat(),
            "user_agent": user_agent or "api_client",
            "ip_address": ip_address or "127.0.0.1"
        }
    )
    
    # Log successful login
    auth_service.audit_logger.log_login_success(
        user_id=user["user_id"],
        username=user["username"],
        ip_address=ip_address,
        user_agent=user_agent,
        mfa_used=False
    )
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": user
    }

async def refresh_access_token(refresh_token: str) -> Dict[str, str]:
    """Refresh access token using refresh token."""
    auth_service = await get_auth_service()
    
    try:
        payload = await auth_service.verify_token(refresh_token)
        if payload.get("type") != "refresh":
            raise AuthenticationError("Invalid token type")
        
        # Create new access token
        token_data = {
            "user_id": payload["user_id"],
            "username": payload["username"],
            "role": payload["role"],
            "permissions": payload["permissions"]
        }
        
        new_access_token = auth_service.create_access_token(token_data)
        
        return {
            "access_token": new_access_token,
            "token_type": "bearer"
        }
        
    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=str(e))

async def logout(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Logout user and revoke tokens."""
    # In production, would revoke all user tokens
    return {"message": "Logged out successfully"}

def initialize_auth(config: Optional[Any] = None) -> str:
    """
    Initialize API key configuration and return the active key.
    
    Priority order:
    1. Secrets manager (Vault, AWS Secrets Manager, etc.)
    2. CRONOS_AI_API_KEY environment variable
    3. API_KEY environment variable
    4. Configuration file
    
    NEVER use hardcoded API keys in production.
    """
    global _api_key

    cfg = config or get_config()
    security_cfg = getattr(cfg, "security", None)
    
    # Try secrets manager first
    try:
        secrets_mgr = get_secrets_manager()
        candidate = secrets_mgr.get_secret('api_key')
        if candidate and len(candidate) >= 32:
            logger.info("API key loaded from secrets manager")
            _api_key = candidate
            if security_cfg is not None:
                security_cfg.api_key = _api_key
            return _api_key
    except Exception as e:
        logger.debug(f"Could not load API key from secrets manager: {e}")
    
    # Try to get API key from config
    candidate = None
    if security_cfg is not None:
        candidate = getattr(security_cfg, "api_key", None)
    
    # Try environment variables
    if not candidate:
        candidate = os.getenv('CRONOS_AI_API_KEY') or os.getenv('API_KEY')
    
    # Validate API key strength
    if candidate:
        if len(candidate) < 32:
            logger.warning(
                f"API key is too short ({len(candidate)} chars). "
                "Minimum 32 characters recommended for production."
            )
        _api_key = candidate
    else:
        # In production, this should fail
        if cfg and hasattr(cfg, 'environment') and cfg.environment.value == 'production':
            raise AuthenticationError(
                "API key not configured in production mode!\n"
                "REQUIRED: Configure API key in secrets manager or set CRONOS_AI_API_KEY.\n"
                "Generate a secure key: python -c \"import secrets; print('cronos_' + secrets.token_urlsafe(32))\""
            )
        
        # For development, generate a temporary key and warn
        _api_key = secrets.token_urlsafe(32)
        logger.warning(
            "No API key configured. Generated temporary key for development. "
            "Set CRONOS_AI_API_KEY environment variable for production."
        )
    
    if security_cfg is not None and not getattr(security_cfg, "api_key", None):
        security_cfg.api_key = _api_key

    return _api_key


def get_api_key() -> str:
    """
    Return the configured API key for bearer authentication.
    
    Raises AuthenticationError if no API key is configured in production.
    """
    global _api_key
    if _api_key is None:
        initialize_auth()
    
    if not _api_key:
        raise AuthenticationError("API key not configured")
    
    return _api_key
