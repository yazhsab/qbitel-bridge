"""
CRONOS AI Engine - Authentication and Authorization
Enterprise-grade authentication with JWT tokens and role-based access control.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import jwt
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
import redis.asyncio as redis

from ..core.config import get_config

logger = logging.getLogger(__name__)

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
        self.secret_key = self.config.security.jwt_secret
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7
    
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
            logger.error(f"Failed to initialize authentication service: {e}")
            raise AuthenticationError(f"Authentication initialization failed: {e}")
    
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
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.JWTError:
            raise AuthenticationError("Invalid token")
    
    async def revoke_token(self, token: str):
        """Add token to blacklist."""
        if self.redis_client:
            try:
                # Decode to get expiration
                payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm], verify=False)
                exp = payload.get('exp', 0)
                
                # Add to blacklist with TTL
                ttl = max(0, exp - int(datetime.utcnow().timestamp()))
                await self.redis_client.setex(f"blacklist:{token}", ttl, "revoked")
                
            except Exception as e:
                logger.warning(f"Failed to revoke token: {e}")
    
    async def store_session(self, user_id: str, session_data: Dict[str, Any], ttl: int = 3600):
        """Store user session data."""
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    f"session:{user_id}", 
                    ttl, 
                    str(session_data)
                )
            except Exception as e:
                logger.warning(f"Failed to store session: {e}")
    
    async def get_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user session data."""
        if self.redis_client:
            try:
                session_data = await self.redis_client.get(f"session:{user_id}")
                return eval(session_data) if session_data else None
            except Exception as e:
                logger.warning(f"Failed to get session: {e}")
        return None
    
    async def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user credentials."""
        # In production, this would query a user database
        # For now, we'll use a simple demo user
        demo_users = {
            "admin": {
                "user_id": "admin_001",
                "username": "admin",
                "password_hash": self.hash_password("admin123"),
                "role": "administrator",
                "permissions": [
                    "protocol_discovery",
                    "copilot_access", 
                    "system_administration",
                    "analytics_access"
                ],
                "full_name": "System Administrator"
            },
            "analyst": {
                "user_id": "analyst_001", 
                "username": "analyst",
                "password_hash": self.hash_password("analyst123"),
                "role": "security_analyst",
                "permissions": [
                    "protocol_discovery",
                    "copilot_access",
                    "analytics_access"
                ],
                "full_name": "Security Analyst"
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
        auth_service = await get_auth_service()
        payload = await auth_service.verify_token(credentials.credentials)
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
async def login(username: str, password: str) -> Dict[str, Any]:
    """Authenticate user and return tokens."""
    auth_service = await get_auth_service()
    
    user = await auth_service.authenticate_user(username, password)
    if not user:
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
            "user_agent": "api_client",
            "ip_address": "127.0.0.1"  # Would be actual IP in production
        }
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