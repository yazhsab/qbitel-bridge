# Enterprise Authentication Guide

## Overview

CRONOS AI Engine provides enterprise-grade authentication with support for:
- Database-backed user management
- OAuth2 integration (Google, GitHub, Azure AD, etc.)
- SAML 2.0 for enterprise SSO
- Multi-factor authentication (MFA)
- API key management with rotation
- Comprehensive audit logging
- Session management with Redis

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Authentication Layer                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Password   │  │    OAuth2    │  │     SAML     │      │
│  │     Auth     │  │     Auth     │  │     Auth     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │     MFA      │  │   API Keys   │  │   Sessions   │      │
│  │   (TOTP)     │  │  Management  │  │  (Redis)     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │            Audit Logging & Compliance                │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Environment Configuration

```bash
# JWT Configuration (REQUIRED)
export CRONOS_AI_JWT_SECRET="<generate-with: python -c 'import secrets; print(secrets.token_urlsafe(48))'>"

# Database Configuration (REQUIRED)
export CRONOS_AI_DB_HOST="localhost"
export CRONOS_AI_DB_PORT="5432"
export CRONOS_AI_DB_NAME="cronos_ai"
export CRONOS_AI_DB_USER="cronos"
export CRONOS_AI_DB_PASSWORD="<secure-password>"

# Redis Configuration (REQUIRED for sessions)
export CRONOS_AI_REDIS_HOST="localhost"
export CRONOS_AI_REDIS_PORT="6379"
export CRONOS_AI_REDIS_PASSWORD="<secure-password>"

# Encryption Key (REQUIRED if encryption enabled)
export CRONOS_AI_ENCRYPTION_KEY="<generate-with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'>"

# API Key (OPTIONAL for API authentication)
export CRONOS_AI_API_KEY="<generate-with: python -c 'import secrets; print(\"cronos_\" + secrets.token_urlsafe(32))'>"
```

### 2. Initialize Database

```bash
cd ai_engine
alembic upgrade head
```

### 3. Create Admin User

```python
import asyncio
from uuid import uuid4
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from ai_engine.models.database import User, UserRole
from ai_engine.api.auth_enterprise import EnterpriseAuthenticationService

async def create_admin():
    engine = create_async_engine(
        "postgresql+asyncpg://cronos:password@localhost/cronos_ai"
    )
    
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    
    auth_service = EnterpriseAuthenticationService()
    
    async with async_session() as session:
        admin = User(
            id=uuid4(),
            username="admin",
            email="admin@example.com",
            full_name="System Administrator",
            password_hash=auth_service.hash_password("SecurePassword123!"),
            role=UserRole.ADMINISTRATOR,
            permissions=[
                "protocol_discovery",
                "copilot_access",
                "system_administration",
                "analytics_access"
            ],
            is_active=True,
            is_verified=True
        )
        
        session.add(admin)
        await session.commit()
        print(f"✓ Admin user created: {admin.username}")

asyncio.run(create_admin())
```

## Password Authentication

### Password Requirements

- Minimum 12 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one digit
- At least one special character
- No common patterns (password, 123456, etc.)

### User Login

```python
from fastapi import FastAPI, Depends
from ai_engine.api.auth_enterprise import EnterpriseAuthenticationService, get_auth_service

app = FastAPI()

@app.post("/auth/login")
async def login(
    username: str,
    password: str,
    auth_service: EnterpriseAuthenticationService = Depends(get_auth_service)
):
    # Authenticate user
    user = await auth_service.authenticate_user(db, username, password)
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Check MFA if enabled
    if user.mfa_enabled:
        return {"requires_mfa": True, "user_id": str(user.id)}
    
    # Create tokens
    token_data = {
        "user_id": str(user.id),
        "username": user.username,
        "role": user.role.value,
        "permissions": user.permissions
    }
    
    access_token = auth_service.create_access_token(token_data)
    refresh_token = auth_service.create_refresh_token(token_data)
    
    # Create session
    session = await auth_service.create_session(
        db, str(user.id), 
        ip_address=request.client.host,
        user_agent=request.headers.get("user-agent")
    )
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": {
            "id": str(user.id),
            "username": user.username,
            "role": user.role.value
        }
    }
```

## Multi-Factor Authentication (MFA)

### Enable TOTP MFA

```python
@app.post("/auth/mfa/enable")
async def enable_mfa(
    current_user: User = Depends(get_current_user),
    auth_service: EnterpriseAuthenticationService = Depends(get_auth_service)
):
    result = await auth_service.enable_mfa(
        db, str(current_user.id), MFAMethod.TOTP
    )
    
    return {
        "qr_code": result["qr_code"],  # Base64 encoded QR code
        "secret": result["secret"],     # Manual entry key
        "backup_codes": result["backup_codes"]  # Save these securely!
    }
```

### Verify MFA Token

```python
@app.post("/auth/mfa/verify")
async def verify_mfa(
    user_id: str,
    token: str,
    auth_service: EnterpriseAuthenticationService = Depends(get_auth_service)
):
    is_valid = await auth_service.verify_mfa_token(db, user_id, token)
    
    if not is_valid:
        raise HTTPException(status_code=401, detail="Invalid MFA token")
    
    # Create tokens after successful MFA
    # ... (same as login)
```

## API Key Management

### Create API Key

```python
@app.post("/auth/api-keys")
async def create_api_key(
    name: str,
    description: str = None,
    expires_days: int = 90,
    current_user: User = Depends(get_current_user),
    auth_service: EnterpriseAuthenticationService = Depends(get_auth_service)
):
    result = await auth_service.create_api_key(
        db,
        user_id=str(current_user.id),
        name=name,
        description=description,
        expires_days=expires_days,
        permissions=current_user.permissions
    )
    
    return {
        "api_key": result["api_key"],  # SAVE THIS - shown only once!
        "key_id": result["key_id"],
        "key_prefix": result["key_prefix"],
        "expires_at": result["expires_at"]
    }
```

### Use API Key

```python
from fastapi import Header

@app.get("/api/protected")
async def protected_endpoint(
    authorization: str = Header(...),
    auth_service: EnterpriseAuthenticationService = Depends(get_auth_service)
):
    # Extract API key from Authorization header
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    
    api_key = authorization[7:]  # Remove "Bearer " prefix
    
    # Verify API key
    user = await auth_service.verify_api_key(db, api_key)
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return {"message": "Access granted", "user": user.username}
```

### Revoke API Key

```python
@app.delete("/auth/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    reason: str = None,
    current_user: User = Depends(get_current_user),
    auth_service: EnterpriseAuthenticationService = Depends(get_auth_service)
):
    await auth_service.revoke_api_key(
        db, key_id, str(current_user.id), reason
    )
    
    return {"message": "API key revoked successfully"}
```

## OAuth2 Integration

### Configure OAuth Provider

```python
from ai_engine.models.database import OAuthProvider
from cryptography.fernet import Fernet

async def configure_oauth_provider():
    # Encrypt client secret
    fernet = Fernet(os.getenv('CRONOS_AI_ENCRYPTION_KEY').encode())
    client_secret_encrypted = fernet.encrypt(
        "your-oauth-client-secret".encode()
    )
    
    provider = OAuthProvider(
        id=uuid4(),
        name="google",
        provider_type="google",
        client_id="your-client-id.apps.googleusercontent.com",
        client_secret_encrypted=client_secret_encrypted,
        authorization_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
        token_endpoint="https://oauth2.googleapis.com/token",
        userinfo_endpoint="https://www.googleapis.com/oauth2/v3/userinfo",
        scopes=["openid", "email", "profile"],
        is_active=True
    )
    
    db.add(provider)
    await db.commit()
```

### OAuth Login Flow

```python
@app.get("/auth/oauth/{provider}/login")
async def oauth_login(provider: str):
    # Get provider configuration
    result = await db.execute(
        select(OAuthProvider).where(OAuthProvider.name == provider)
    )
    oauth_provider = result.scalar_one_or_none()
    
    if not oauth_provider:
        raise HTTPException(status_code=404, detail="Provider not found")
    
    # Generate state token for CSRF protection
    state = secrets.token_urlsafe(32)
    
    # Build authorization URL
    auth_url = (
        f"{oauth_provider.authorization_endpoint}?"
        f"client_id={oauth_provider.client_id}&"
        f"redirect_uri={REDIRECT_URI}&"
        f"response_type=code&"
        f"scope={' '.join(oauth_provider.scopes)}&"
        f"state={state}"
    )
    
    return {"authorization_url": auth_url, "state": state}
```

## SAML 2.0 Integration

### Configure SAML Provider

```python
from ai_engine.models.database import SAMLProvider

async def configure_saml_provider():
    provider = SAMLProvider(
        id=uuid4(),
        name="corporate_sso",
        entity_id="https://sso.company.com/saml/metadata",
        sso_url="https://sso.company.com/saml/sso",
        slo_url="https://sso.company.com/saml/slo",
        x509_cert="-----BEGIN CERTIFICATE-----\n...\n-----END CERTIFICATE-----",
        attribute_mapping={
            "email": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress",
            "username": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name",
            "full_name": "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/givenname"
        },
        is_active=True
    )
    
    db.add(provider)
    await db.commit()
```

## Audit Logging

All authentication events are automatically logged:

```python
# Query audit logs
from ai_engine.models.database import AuditLog, AuditAction

# Get failed login attempts
result = await db.execute(
    select(AuditLog)
    .where(
        and_(
            AuditLog.action == AuditAction.LOGIN_FAILED,
            AuditLog.timestamp >= datetime.utcnow() - timedelta(hours=24)
        )
    )
    .order_by(AuditLog.timestamp.desc())
)
failed_logins = result.scalars().all()

# Get user activity
result = await db.execute(
    select(AuditLog)
    .where(AuditLog.user_id == user_id)
    .order_by(AuditLog.timestamp.desc())
    .limit(100)
)
user_activity = result.scalars().all()
```

## Security Best Practices

### 1. Password Management

- Enforce strong password requirements
- Implement password expiration policies
- Prevent password reuse
- Use bcrypt for password hashing

### 2. API Key Security

- Rotate API keys regularly (90 days recommended)
- Use key prefixes for identification
- Monitor API key usage
- Revoke unused keys

### 3. Session Management

- Use Redis for distributed session storage
- Implement session timeout (24 hours recommended)
- Track session activity
- Revoke sessions on logout

### 4. MFA Implementation

- Require MFA for administrators
- Provide backup codes
- Support multiple MFA methods
- Log MFA events

### 5. Audit Logging

- Log all authentication events
- Include IP addresses and user agents
- Retain logs for compliance (365 days recommended)
- Monitor for suspicious activity

## Troubleshooting

### Common Issues

**Issue: JWT token expired**
```python
# Solution: Implement token refresh
@app.post("/auth/refresh")
async def refresh_token(refresh_token: str):
    payload = await auth_service.verify_token(refresh_token)
    
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid token type")
    
    new_access_token = auth_service.create_access_token({
        "user_id": payload["user_id"],
        "username": payload["username"],
        "role": payload["role"]
    })
    
    return {"access_token": new_access_token}
```

**Issue: Account locked after failed attempts**
```python
# Solution: Implement unlock mechanism
@app.post("/auth/unlock")
async def unlock_account(email: str):
    # Send unlock email with token
    # Verify token and unlock account
    user.account_locked_until = None
    user.failed_login_attempts = 0
    await db.commit()
```

## Migration from Legacy Auth

```python
# Migrate existing users to new system
async def migrate_users():
    # Read from old system
    old_users = get_old_users()
    
    for old_user in old_users:
        new_user = User(
            id=uuid4(),
            username=old_user["username"],
            email=old_user["email"],
            full_name=old_user["name"],
            password_hash=old_user["password_hash"],  # Already hashed
            role=UserRole.VIEWER,
            permissions=[],
            is_active=True,
            must_change_password=True  # Force password change
        )
        
        db.add(new_user)
    
    await db.commit()
```

## Support

For authentication issues:
- Documentation: `docs/ENTERPRISE_AUTHENTICATION.md`
- Security: security@cronos-ai.com
- Support: support@cronos-ai.com