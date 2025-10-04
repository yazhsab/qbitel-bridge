"""
CRONOS AI Engine - Enterprise Authentication

Production-ready authentication with OAuth2, SAML, MFA, and comprehensive security features.
"""

import os
import hashlib
import secrets
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import pyotp
import qrcode
import io
import base64

from fastapi import HTTPException, Depends, Request
from fastapi.security import (
    HTTPBearer,
    HTTPAuthorizationCredentials,
    OAuth2PasswordBearer,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from passlib.context import CryptContext
import jwt
from jwt import ExpiredSignatureError, PyJWTError

from ..core.config import get_config
from ..security.audit_logger import get_audit_logger
from ..models.database import (
    User,
    APIKey,
    UserSession,
    AuditLog,
    OAuthProvider,
    SAMLProvider,
    PasswordResetToken,
    UserRole,
    MFAMethod,
    APIKeyStatus,
    AuditAction,
)

logger = logging.getLogger(__name__)

# Security configuration
security = HTTPBearer()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class EnterpriseAuthenticationService:
    """Enterprise-grade authentication service with OAuth2, SAML, and MFA support."""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.secret_key = self._load_secret_key()
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7
        self.audit_logger = get_audit_logger()

        # MFA policy configuration
        self.mfa_required_for_admin = True
        self.mfa_required_for_privileged = True
        self.mfa_grace_period_days = 7

    def _load_secret_key(self) -> str:
        """Load JWT secret from configuration."""
        secret = getattr(self.config.security, "jwt_secret", None)
        if not secret or len(secret) < 32:
            raise ValueError(
                "JWT secret not configured or too short. "
                "Set CRONOS_AI_JWT_SECRET environment variable with at least 32 characters."
            )
        return secret

    # ==================== Password Management ====================

    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        return pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return pwd_context.verify(plain_password, hashed_password)

    def validate_password_strength(self, password: str) -> tuple[bool, Optional[str]]:
        """
        Validate password meets security requirements.

        Returns:
            (is_valid, error_message)
        """
        if len(password) < 12:
            return False, "Password must be at least 12 characters long"

        if not any(c.isupper() for c in password):
            return False, "Password must contain at least one uppercase letter"

        if not any(c.islower() for c in password):
            return False, "Password must contain at least one lowercase letter"

        if not any(c.isdigit() for c in password):
            return False, "Password must contain at least one digit"

        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            return False, "Password must contain at least one special character"

        # Check for common patterns
        common_patterns = ["password", "123456", "qwerty", "admin", "letmein"]
        if any(pattern in password.lower() for pattern in common_patterns):
            return False, "Password contains common patterns"

        return True, None

    async def change_password(
        self, db: AsyncSession, user_id: str, old_password: str, new_password: str
    ) -> bool:
        """Change user password with validation."""
        # Get user
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Verify old password
        if not self.verify_password(old_password, user.password_hash):
            await self._log_audit(
                db,
                user_id,
                AuditAction.PASSWORD_CHANGE,
                success=False,
                error_message="Invalid old password",
            )
            raise HTTPException(status_code=400, detail="Invalid old password")

        # Validate new password
        is_valid, error = self.validate_password_strength(new_password)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error)

        # Update password
        user.password_hash = self.hash_password(new_password)
        user.password_changed_at = datetime.utcnow()
        user.must_change_password = False

        await db.commit()

        await self._log_audit(db, user_id, AuditAction.PASSWORD_CHANGE, success=True)

        return True

    # ==================== Multi-Factor Authentication ====================

    def generate_totp_secret(self) -> str:
        """Generate TOTP secret for MFA."""
        return pyotp.random_base32()

    def generate_totp_qr_code(self, username: str, secret: str) -> str:
        """Generate QR code for TOTP setup."""
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=username, issuer_name="CRONOS AI"
        )

        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        return base64.b64encode(buffer.getvalue()).decode()

    def verify_totp(self, secret: str, token: str) -> bool:
        """Verify TOTP token."""
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=1)

    def generate_backup_codes(self, count: int = 10) -> List[str]:
        """Generate backup codes for MFA."""
        return [secrets.token_hex(4).upper() for _ in range(count)]

    async def enable_mfa(
        self,
        db: AsyncSession,
        user_id: str,
        method: MFAMethod,
        secret: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Enable MFA for user."""
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if method == MFAMethod.TOTP:
            if not secret:
                secret = self.generate_totp_secret()

            # Generate QR code
            qr_code = self.generate_totp_qr_code(user.username, secret)

            # Generate backup codes
            backup_codes = self.generate_backup_codes()
            backup_codes_hashed = [self.hash_password(code) for code in backup_codes]

            user.mfa_enabled = True
            user.mfa_method = method
            user.mfa_secret = secret  # Should be encrypted in production
            user.mfa_backup_codes = backup_codes_hashed

            await db.commit()

            await self._log_audit(
                db,
                user_id,
                AuditAction.MFA_ENABLED,
                success=True,
                details={"method": method.value},
            )

            return {"qr_code": qr_code, "secret": secret, "backup_codes": backup_codes}

        raise HTTPException(
            status_code=400, detail=f"MFA method {method} not implemented"
        )

    async def verify_mfa_token(
        self, db: AsyncSession, user_id: str, token: str
    ) -> bool:
        """Verify MFA token."""
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()

        if not user or not user.mfa_enabled:
            return False

        if user.mfa_method == MFAMethod.TOTP:
            return self.verify_totp(user.mfa_secret, token)

        return False

    # ==================== API Key Management ====================

    def generate_api_key(self) -> tuple[str, str]:
        """
        Generate API key and its hash.

        Returns:
            (api_key, key_hash)
        """
        # Generate key with prefix
        key = f"cronos_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return key, key_hash

    async def create_api_key(
        self,
        db: AsyncSession,
        user_id: str,
        name: str,
        description: Optional[str] = None,
        expires_days: Optional[int] = None,
        permissions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create new API key for user."""
        # Generate key
        api_key, key_hash = self.generate_api_key()
        key_prefix = api_key[:15]

        # Calculate expiration
        expires_at = None
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)

        # Create API key record
        api_key_record = APIKey(
            key_hash=key_hash,
            key_prefix=key_prefix,
            name=name,
            description=description,
            user_id=user_id,
            expires_at=expires_at,
            permissions=permissions or [],
        )

        db.add(api_key_record)
        await db.commit()
        await db.refresh(api_key_record)

        await self._log_audit(
            db,
            user_id,
            AuditAction.API_KEY_CREATED,
            success=True,
            details={"key_name": name, "key_id": str(api_key_record.id)},
        )

        return {
            "api_key": api_key,  # Only returned once
            "key_id": str(api_key_record.id),
            "key_prefix": key_prefix,
            "expires_at": expires_at.isoformat() if expires_at else None,
        }

    async def revoke_api_key(
        self,
        db: AsyncSession,
        key_id: str,
        revoked_by: str,
        reason: Optional[str] = None,
    ) -> bool:
        """Revoke API key."""
        result = await db.execute(select(APIKey).where(APIKey.id == key_id))
        api_key = result.scalar_one_or_none()

        if not api_key:
            raise HTTPException(status_code=404, detail="API key not found")

        api_key.status = APIKeyStatus.REVOKED
        api_key.revoked_at = datetime.utcnow()
        api_key.revoked_by = revoked_by
        api_key.revoked_reason = reason

        await db.commit()

        await self._log_audit(
            db,
            revoked_by,
            AuditAction.API_KEY_REVOKED,
            success=True,
            details={"key_id": key_id, "reason": reason},
        )

        return True

    async def verify_api_key(self, db: AsyncSession, api_key: str) -> Optional[User]:
        """Verify API key and return associated user."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        result = await db.execute(
            select(APIKey, User)
            .join(User)
            .where(
                and_(
                    APIKey.key_hash == key_hash,
                    APIKey.status == APIKeyStatus.ACTIVE,
                    User.is_active == True,
                )
            )
        )
        row = result.first()

        if not row:
            return None

        api_key_record, user = row

        # Check expiration
        if api_key_record.expires_at and api_key_record.expires_at < datetime.utcnow():
            api_key_record.status = APIKeyStatus.EXPIRED
            await db.commit()
            return None

        # Update usage
        api_key_record.last_used_at = datetime.utcnow()
        api_key_record.usage_count += 1
        await db.commit()

        return user

    # ==================== JWT Token Management ====================

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
            return payload
        except ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except PyJWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

    # ==================== Session Management ====================

    async def create_session(
        self,
        db: AsyncSession,
        user_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> UserSession:
        """Create user session."""
        session_token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(hours=24)

        session = UserSession(
            session_token=session_token,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=expires_at,
        )

        db.add(session)
        await db.commit()
        await db.refresh(session)

        return session

    async def revoke_session(self, db: AsyncSession, session_token: str) -> bool:
        """Revoke user session."""
        result = await db.execute(
            select(UserSession).where(UserSession.session_token == session_token)
        )
        session = result.scalar_one_or_none()

        if session:
            session.revoked_at = datetime.utcnow()
            await db.commit()
            return True

        return False

    # ==================== Audit Logging ====================

    async def _log_audit(
        self,
        db: AsyncSession,
        user_id: Optional[str],
        action: AuditAction,
        success: bool = True,
        error_message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ):
        """Log audit event."""
        audit_log = AuditLog(
            action=action,
            user_id=user_id,
            success=success,
            error_message=error_message,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        db.add(audit_log)
        await db.commit()

    # ==================== User Authentication ====================

    async def authenticate_user(
        self,
        db: AsyncSession,
        username: str,
        password: str,
        ip_address: Optional[str] = None,
        mfa_token: Optional[str] = None,
    ) -> Optional[User]:
        """Authenticate user with username, password, and optional MFA."""
        result = await db.execute(
            select(User).where(and_(User.username == username, User.is_active == True))
        )
        user = result.scalar_one_or_none()

        if not user:
            await self._log_audit(
                db,
                None,
                AuditAction.LOGIN_FAILED,
                success=False,
                error_message="User not found",
                details={"username": username},
                ip_address=ip_address,
            )
            self.audit_logger.log_login_failed(
                username=username, reason="User not found", ip_address=ip_address
            )
            return None

        # Check account lockout
        if user.account_locked_until and user.account_locked_until > datetime.utcnow():
            await self._log_audit(
                db,
                str(user.id),
                AuditAction.LOGIN_FAILED,
                success=False,
                error_message="Account locked",
                ip_address=ip_address,
            )
            self.audit_logger.log_login_failed(
                username=username, reason="Account locked", ip_address=ip_address
            )
            raise HTTPException(
                status_code=403,
                detail=f"Account locked until {user.account_locked_until.isoformat()}",
            )

        # Verify password
        if not user.password_hash or not self.verify_password(
            password, user.password_hash
        ):
            user.failed_login_attempts += 1

            # Lock account after 5 failed attempts
            if user.failed_login_attempts >= 5:
                user.account_locked_until = datetime.utcnow() + timedelta(minutes=30)
                self.audit_logger.log_security_alert(
                    alert_type="account_locked",
                    description=f"Account {username} locked after 5 failed login attempts",
                    user_id=str(user.id),
                    ip_address=ip_address,
                )

            await db.commit()

            await self._log_audit(
                db,
                str(user.id),
                AuditAction.LOGIN_FAILED,
                success=False,
                error_message="Invalid password",
                ip_address=ip_address,
            )
            self.audit_logger.log_login_failed(
                username=username,
                reason="Invalid password",
                ip_address=ip_address,
                failed_attempts=user.failed_login_attempts,
            )
            return None

        # Check MFA requirement
        if self._requires_mfa(user):
            if not user.mfa_enabled:
                # Enforce MFA for privileged accounts
                if self._is_privileged_user(user):
                    raise HTTPException(
                        status_code=403,
                        detail="MFA is required for this account. Please enable MFA before logging in.",
                    )
            elif not mfa_token:
                raise HTTPException(status_code=403, detail="MFA token required")
            else:
                # Verify MFA token
                if not await self.verify_mfa_token(db, str(user.id), mfa_token):
                    await self._log_audit(
                        db,
                        str(user.id),
                        AuditAction.LOGIN_FAILED,
                        success=False,
                        error_message="Invalid MFA token",
                        ip_address=ip_address,
                    )
                    self.audit_logger.log_login_failed(
                        username=username,
                        reason="Invalid MFA token",
                        ip_address=ip_address,
                    )
                    return None

        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.last_login = datetime.utcnow()
        user.last_login_ip = ip_address
        await db.commit()

        # Log successful authentication
        await self._log_audit(
            db,
            str(user.id),
            AuditAction.LOGIN_SUCCESS,
            success=True,
            ip_address=ip_address,
            details={"mfa_used": user.mfa_enabled},
        )
        self.audit_logger.log_login_success(
            user_id=str(user.id),
            username=username,
            ip_address=ip_address,
            mfa_used=user.mfa_enabled,
        )

        return user

    def _requires_mfa(self, user: User) -> bool:
        """Check if user requires MFA based on policy."""
        # Check if user is in grace period
        if user.created_at:
            grace_period_end = user.created_at + timedelta(
                days=self.mfa_grace_period_days
            )
            if datetime.utcnow() < grace_period_end:
                return False

        # Require MFA for admin users
        if self.mfa_required_for_admin and user.role == UserRole.ADMINISTRATOR:
            return True

        # Require MFA for privileged users
        if self.mfa_required_for_privileged and self._is_privileged_user(user):
            return True

        return False

    def _is_privileged_user(self, user: User) -> bool:
        """Check if user has privileged access."""
        privileged_roles = [
            UserRole.ADMINISTRATOR,
            UserRole.SECURITY_ADMIN,
            UserRole.COMPLIANCE_OFFICER,
        ]
        return user.role in privileged_roles


# Global service instance
_auth_service: Optional[EnterpriseAuthenticationService] = None


async def get_auth_service() -> EnterpriseAuthenticationService:
    """Get authentication service instance."""
    global _auth_service
    if _auth_service is None:
        _auth_service = EnterpriseAuthenticationService()
    return _auth_service
