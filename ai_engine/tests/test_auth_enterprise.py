"""
CRONOS AI Engine - Enterprise Authentication Tests

Comprehensive test suite for enterprise authentication functionality.
"""

import pytest
import asyncio
import jwt
import hashlib
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, List, Any, Optional

from ai_engine.api.auth_enterprise import (
    EnterpriseAuthService,
    LDAPAuthenticator,
    SAMLAuthenticator,
    OIDCAuthenticator,
    EnterpriseUser,
    EnterpriseRole,
    EnterprisePermission,
    AuthProvider,
    AuthMethod,
    EnterpriseAuthConfig,
    EnterpriseAuthException,
    EnterpriseAuthValidationError,
)


class TestEnterpriseUser:
    """Test EnterpriseUser dataclass."""

    def test_enterprise_user_creation(self):
        """Test creating EnterpriseUser instance."""
        user = EnterpriseUser(
            user_id="user123",
            username="testuser",
            email="test@example.com",
            first_name="Test",
            last_name="User",
            department="Engineering",
            manager="manager@example.com",
            employee_id="EMP001",
            roles=["admin", "user"],
            permissions=["read", "write"],
            groups=["engineering", "admin"],
            last_login=datetime.now(timezone.utc),
            is_active=True,
            metadata={"location": "US", "timezone": "UTC"}
        )
        
        assert user.user_id == "user123"
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.first_name == "Test"
        assert user.last_name == "User"
        assert user.department == "Engineering"
        assert user.manager == "manager@example.com"
        assert user.employee_id == "EMP001"
        assert user.roles == ["admin", "user"]
        assert user.permissions == ["read", "write"]
        assert user.groups == ["engineering", "admin"]
        assert user.is_active is True
        assert user.metadata == {"location": "US", "timezone": "UTC"}

    def test_enterprise_user_defaults(self):
        """Test EnterpriseUser with default values."""
        user = EnterpriseUser(
            user_id="user456",
            username="testuser2",
            email="test2@example.com"
        )
        
        assert user.first_name is None
        assert user.last_name is None
        assert user.department is None
        assert user.manager is None
        assert user.employee_id is None
        assert user.roles == []
        assert user.permissions == []
        assert user.groups == []
        assert user.last_login is None
        assert user.is_active is True
        assert user.metadata == {}

    def test_enterprise_user_full_name(self):
        """Test EnterpriseUser full name property."""
        user = EnterpriseUser(
            user_id="user789",
            username="testuser3",
            email="test3@example.com",
            first_name="John",
            last_name="Doe"
        )
        
        assert user.full_name == "John Doe"

    def test_enterprise_user_full_name_partial(self):
        """Test EnterpriseUser full name with partial names."""
        user = EnterpriseUser(
            user_id="user101",
            username="testuser4",
            email="test4@example.com",
            first_name="John"
        )
        
        assert user.full_name == "John"

    def test_enterprise_user_has_role(self):
        """Test EnterpriseUser has_role method."""
        user = EnterpriseUser(
            user_id="user102",
            username="testuser5",
            email="test5@example.com",
            roles=["admin", "user", "viewer"]
        )
        
        assert user.has_role("admin") is True
        assert user.has_role("user") is True
        assert user.has_role("viewer") is True
        assert user.has_role("superuser") is False

    def test_enterprise_user_has_permission(self):
        """Test EnterpriseUser has_permission method."""
        user = EnterpriseUser(
            user_id="user103",
            username="testuser6",
            email="test6@example.com",
            permissions=["read", "write", "delete"]
        )
        
        assert user.has_permission("read") is True
        assert user.has_permission("write") is True
        assert user.has_permission("delete") is True
        assert user.has_permission("admin") is False

    def test_enterprise_user_in_group(self):
        """Test EnterpriseUser in_group method."""
        user = EnterpriseUser(
            user_id="user104",
            username="testuser7",
            email="test7@example.com",
            groups=["engineering", "admin", "devops"]
        )
        
        assert user.in_group("engineering") is True
        assert user.in_group("admin") is True
        assert user.in_group("devops") is True
        assert user.in_group("marketing") is False


class TestEnterpriseRole:
    """Test EnterpriseRole dataclass."""

    def test_enterprise_role_creation(self):
        """Test creating EnterpriseRole instance."""
        role = EnterpriseRole(
            role_id="role123",
            name="admin",
            description="Administrator role",
            permissions=["read", "write", "delete", "admin"],
            inherits_from=["user"],
            is_system_role=True,
            created_at=datetime.now(timezone.utc),
            created_by="system",
            metadata={"level": "high", "department": "IT"}
        )
        
        assert role.role_id == "role123"
        assert role.name == "admin"
        assert role.description == "Administrator role"
        assert role.permissions == ["read", "write", "delete", "admin"]
        assert role.inherits_from == ["user"]
        assert role.is_system_role is True
        assert role.created_by == "system"
        assert role.metadata == {"level": "high", "department": "IT"}

    def test_enterprise_role_has_permission(self):
        """Test EnterpriseRole has_permission method."""
        role = EnterpriseRole(
            role_id="role456",
            name="user",
            description="User role",
            permissions=["read", "write"]
        )
        
        assert role.has_permission("read") is True
        assert role.has_permission("write") is True
        assert role.has_permission("delete") is False

    def test_enterprise_role_inheritance(self):
        """Test EnterpriseRole inheritance."""
        base_role = EnterpriseRole(
            role_id="base_role",
            name="base",
            description="Base role",
            permissions=["read"]
        )
        
        derived_role = EnterpriseRole(
            role_id="derived_role",
            name="derived",
            description="Derived role",
            permissions=["write"],
            inherits_from=["base"]
        )
        
        # Test inheritance logic
        assert derived_role.inherits_from == ["base"]


class TestEnterprisePermission:
    """Test EnterprisePermission dataclass."""

    def test_enterprise_permission_creation(self):
        """Test creating EnterprisePermission instance."""
        permission = EnterprisePermission(
            permission_id="perm123",
            name="read",
            description="Read permission",
            resource="data",
            action="read",
            conditions={"department": "engineering"},
            is_system_permission=True,
            created_at=datetime.now(timezone.utc),
            created_by="system"
        )
        
        assert permission.permission_id == "perm123"
        assert permission.name == "read"
        assert permission.description == "Read permission"
        assert permission.resource == "data"
        assert permission.action == "read"
        assert permission.conditions == {"department": "engineering"}
        assert permission.is_system_permission is True
        assert permission.created_by == "system"

    def test_enterprise_permission_matches(self):
        """Test EnterprisePermission matches method."""
        permission = EnterprisePermission(
            permission_id="perm456",
            name="read_data",
            description="Read data permission",
            resource="data",
            action="read",
            conditions={"department": "engineering"}
        )
        
        # Test matching conditions
        assert permission.matches({"department": "engineering"}) is True
        assert permission.matches({"department": "marketing"}) is False
        assert permission.matches({}) is False


class TestLDAPAuthenticator:
    """Test LDAPAuthenticator functionality."""

    @pytest.fixture
    def ldap_config(self):
        """Create LDAP configuration."""
        return {
            "server": "ldap://ldap.example.com:389",
            "base_dn": "dc=example,dc=com",
            "bind_dn": "cn=admin,dc=example,dc=com",
            "bind_password": "admin_password",
            "user_search_base": "ou=users,dc=example,dc=com",
            "group_search_base": "ou=groups,dc=example,dc=com",
            "user_filter": "(uid={username})",
            "group_filter": "(member={user_dn})"
        }

    @pytest.fixture
    def ldap_authenticator(self, ldap_config):
        """Create LDAPAuthenticator instance."""
        with patch('ai_engine.api.auth_enterprise.ldap3') as mock_ldap3:
            return LDAPAuthenticator(ldap_config)

    def test_ldap_authenticator_initialization(self, ldap_authenticator, ldap_config):
        """Test LDAPAuthenticator initialization."""
        assert ldap_authenticator.config == ldap_config
        assert ldap_authenticator.server == ldap_config["server"]
        assert ldap_authenticator.base_dn == ldap_config["base_dn"]

    @pytest.mark.asyncio
    async def test_ldap_authenticate_success(self, ldap_authenticator):
        """Test successful LDAP authentication."""
        with patch('ai_engine.api.auth_enterprise.ldap3') as mock_ldap3:
            mock_connection = Mock()
            mock_connection.bind.return_value = True
            mock_connection.search.return_value = True
            mock_connection.entries = [Mock()]
            mock_connection.entries[0].entry_dn = "uid=testuser,ou=users,dc=example,dc=com"
            mock_connection.entries[0].uid.value = "testuser"
            mock_connection.entries[0].mail.value = "test@example.com"
            mock_connection.entries[0].cn.value = "Test User"
            mock_ldap3.Connection.return_value = mock_connection
            
            user = await ldap_authenticator.authenticate("testuser", "password")
            
            assert user is not None
            assert user.username == "testuser"
            assert user.email == "test@example.com"

    @pytest.mark.asyncio
    async def test_ldap_authenticate_failure(self, ldap_authenticator):
        """Test LDAP authentication failure."""
        with patch('ai_engine.api.auth_enterprise.ldap3') as mock_ldap3:
            mock_connection = Mock()
            mock_connection.bind.return_value = False
            mock_ldap3.Connection.return_value = mock_connection
            
            user = await ldap_authenticator.authenticate("testuser", "wrong_password")
            
            assert user is None

    @pytest.mark.asyncio
    async def test_ldap_get_user_groups(self, ldap_authenticator):
        """Test getting user groups from LDAP."""
        with patch('ai_engine.api.auth_enterprise.ldap3') as mock_ldap3:
            mock_connection = Mock()
            mock_connection.search.return_value = True
            mock_connection.entries = [Mock(), Mock()]
            mock_connection.entries[0].cn.value = "engineering"
            mock_connection.entries[1].cn.value = "admin"
            mock_ldap3.Connection.return_value = mock_connection
            
            groups = await ldap_authenticator.get_user_groups("testuser")
            
            assert "engineering" in groups
            assert "admin" in groups

    @pytest.mark.asyncio
    async def test_ldap_connection_error(self, ldap_authenticator):
        """Test LDAP connection error."""
        with patch('ai_engine.api.auth_enterprise.ldap3') as mock_ldap3:
            mock_ldap3.Connection.side_effect = Exception("Connection failed")
            
            with pytest.raises(EnterpriseAuthException):
                await ldap_authenticator.authenticate("testuser", "password")


class TestSAMLAuthenticator:
    """Test SAMLAuthenticator functionality."""

    @pytest.fixture
    def saml_config(self):
        """Create SAML configuration."""
        return {
            "idp_url": "https://idp.example.com/sso",
            "sp_entity_id": "https://app.example.com",
            "sp_acs_url": "https://app.example.com/saml/acs",
            "sp_sls_url": "https://app.example.com/saml/sls",
            "certificate": "-----BEGIN CERTIFICATE-----\n...\n-----END CERTIFICATE-----",
            "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----"
        }

    @pytest.fixture
    def saml_authenticator(self, saml_config):
        """Create SAMLAuthenticator instance."""
        with patch('ai_engine.api.auth_enterprise.OneLogin_Saml2_Auth') as mock_saml:
            return SAMLAuthenticator(saml_config)

    def test_saml_authenticator_initialization(self, saml_authenticator, saml_config):
        """Test SAMLAuthenticator initialization."""
        assert saml_authenticator.config == saml_config
        assert saml_authenticator.idp_url == saml_config["idp_url"]
        assert saml_authenticator.sp_entity_id == saml_config["sp_entity_id"]

    def test_saml_create_auth_request(self, saml_authenticator):
        """Test creating SAML authentication request."""
        with patch('ai_engine.api.auth_enterprise.OneLogin_Saml2_Auth') as mock_saml:
            mock_auth = Mock()
            mock_auth.login.return_value = "https://idp.example.com/sso?SAMLRequest=..."
            mock_saml.return_value = mock_auth
            
            auth_url = saml_authenticator.create_auth_request()
            
            assert auth_url.startswith("https://idp.example.com/sso")
            mock_auth.login.assert_called_once()

    def test_saml_process_response(self, saml_authenticator):
        """Test processing SAML response."""
        with patch('ai_engine.api.auth_enterprise.OneLogin_Saml2_Auth') as mock_saml:
            mock_auth = Mock()
            mock_auth.process_response.return_value = None
            mock_auth.get_errors.return_value = []
            mock_auth.get_attributes.return_value = {
                "username": ["testuser"],
                "email": ["test@example.com"],
                "first_name": ["Test"],
                "last_name": ["User"]
            }
            mock_saml.return_value = mock_auth
            
            user = saml_authenticator.process_response("saml_response")
            
            assert user is not None
            assert user.username == "testuser"
            assert user.email == "test@example.com"

    def test_saml_process_response_with_errors(self, saml_authenticator):
        """Test processing SAML response with errors."""
        with patch('ai_engine.api.auth_enterprise.OneLogin_Saml2_Auth') as mock_saml:
            mock_auth = Mock()
            mock_auth.process_response.return_value = None
            mock_auth.get_errors.return_value = ["Invalid signature"]
            mock_saml.return_value = mock_auth
            
            with pytest.raises(EnterpriseAuthException):
                saml_authenticator.process_response("invalid_response")

    def test_saml_logout(self, saml_authenticator):
        """Test SAML logout."""
        with patch('ai_engine.api.auth_enterprise.OneLogin_Saml2_Auth') as mock_saml:
            mock_auth = Mock()
            mock_auth.logout.return_value = "https://idp.example.com/slo?SAMLRequest=..."
            mock_saml.return_value = mock_auth
            
            logout_url = saml_authenticator.logout("session_id")
            
            assert logout_url.startswith("https://idp.example.com/slo")
            mock_auth.logout.assert_called_once_with("session_id")


class TestOIDCAuthenticator:
    """Test OIDCAuthenticator functionality."""

    @pytest.fixture
    def oidc_config(self):
        """Create OIDC configuration."""
        return {
            "issuer": "https://oidc.example.com",
            "client_id": "test_client",
            "client_secret": "test_secret",
            "redirect_uri": "https://app.example.com/oidc/callback",
            "scope": "openid profile email",
            "response_type": "code"
        }

    @pytest.fixture
    def oidc_authenticator(self, oidc_config):
        """Create OIDCAuthenticator instance."""
        with patch('ai_engine.api.auth_enterprise.requests') as mock_requests:
            return OIDCAuthenticator(oidc_config)

    def test_oidc_authenticator_initialization(self, oidc_authenticator, oidc_config):
        """Test OIDCAuthenticator initialization."""
        assert oidc_authenticator.config == oidc_config
        assert oidc_authenticator.issuer == oidc_config["issuer"]
        assert oidc_authenticator.client_id == oidc_config["client_id"]

    def test_oidc_create_auth_url(self, oidc_authenticator):
        """Test creating OIDC authentication URL."""
        auth_url = oidc_authenticator.create_auth_url("state123")
        
        assert "https://oidc.example.com" in auth_url
        assert "client_id=test_client" in auth_url
        assert "state=state123" in auth_url

    @pytest.mark.asyncio
    async def test_oidc_exchange_code_for_token(self, oidc_authenticator):
        """Test exchanging authorization code for token."""
        with patch('ai_engine.api.auth_enterprise.requests') as mock_requests:
            mock_response = Mock()
            mock_response.json.return_value = {
                "access_token": "access_token_123",
                "id_token": "id_token_123",
                "token_type": "Bearer",
                "expires_in": 3600
            }
            mock_requests.post.return_value = mock_response
            
            tokens = await oidc_authenticator.exchange_code_for_token("auth_code")
            
            assert tokens["access_token"] == "access_token_123"
            assert tokens["id_token"] == "id_token_123"

    @pytest.mark.asyncio
    async def test_oidc_get_user_info(self, oidc_authenticator):
        """Test getting user info from OIDC."""
        with patch('ai_engine.api.auth_enterprise.requests') as mock_requests:
            mock_response = Mock()
            mock_response.json.return_value = {
                "sub": "user123",
                "preferred_username": "testuser",
                "email": "test@example.com",
                "given_name": "Test",
                "family_name": "User"
            }
            mock_requests.get.return_value = mock_response
            
            user_info = await oidc_authenticator.get_user_info("access_token")
            
            assert user_info["sub"] == "user123"
            assert user_info["preferred_username"] == "testuser"
            assert user_info["email"] == "test@example.com"

    @pytest.mark.asyncio
    async def test_oidc_authenticate_complete_flow(self, oidc_authenticator):
        """Test complete OIDC authentication flow."""
        with patch('ai_engine.api.auth_enterprise.requests') as mock_requests:
            # Mock token exchange
            token_response = Mock()
            token_response.json.return_value = {
                "access_token": "access_token_123",
                "id_token": "id_token_123",
                "token_type": "Bearer",
                "expires_in": 3600
            }
            
            # Mock user info
            user_response = Mock()
            user_response.json.return_value = {
                "sub": "user123",
                "preferred_username": "testuser",
                "email": "test@example.com",
                "given_name": "Test",
                "family_name": "User"
            }
            
            mock_requests.post.return_value = token_response
            mock_requests.get.return_value = user_response
            
            user = await oidc_authenticator.authenticate("auth_code")
            
            assert user is not None
            assert user.user_id == "user123"
            assert user.username == "testuser"
            assert user.email == "test@example.com"


class TestEnterpriseAuthService:
    """Test EnterpriseAuthService main functionality."""

    @pytest.fixture
    def auth_config(self):
        """Create enterprise auth configuration."""
        return EnterpriseAuthConfig(
            providers={
                "ldap": {
                    "type": "ldap",
                    "enabled": True,
                    "server": "ldap://ldap.example.com:389"
                },
                "saml": {
                    "type": "saml",
                    "enabled": True,
                    "idp_url": "https://idp.example.com/sso"
                },
                "oidc": {
                    "type": "oidc",
                    "enabled": True,
                    "issuer": "https://oidc.example.com"
                }
            },
            default_provider="ldap",
            session_timeout=3600,
            token_expiry=7200,
            enable_mfa=True,
            password_policy={
                "min_length": 8,
                "require_uppercase": True,
                "require_lowercase": True,
                "require_numbers": True,
                "require_special": True
            }
        )

    @pytest.fixture
    def enterprise_auth_service(self, auth_config):
        """Create EnterpriseAuthService instance."""
        with patch('ai_engine.api.auth_enterprise.LDAPAuthenticator'), \
             patch('ai_engine.api.auth_enterprise.SAMLAuthenticator'), \
             patch('ai_engine.api.auth_enterprise.OIDCAuthenticator'):
            return EnterpriseAuthService(auth_config)

    def test_enterprise_auth_service_initialization(self, enterprise_auth_service, auth_config):
        """Test EnterpriseAuthService initialization."""
        assert enterprise_auth_service.config == auth_config
        assert enterprise_auth_service.default_provider == "ldap"
        assert enterprise_auth_service.session_timeout == 3600

    def test_get_available_providers(self, enterprise_auth_service):
        """Test getting available authentication providers."""
        providers = enterprise_auth_service.get_available_providers()
        
        assert "ldap" in providers
        assert "saml" in providers
        assert "oidc" in providers

    def test_get_provider_config(self, enterprise_auth_service):
        """Test getting provider configuration."""
        ldap_config = enterprise_auth_service.get_provider_config("ldap")
        
        assert ldap_config["type"] == "ldap"
        assert ldap_config["enabled"] is True

    def test_get_provider_config_not_found(self, enterprise_auth_service):
        """Test getting non-existent provider configuration."""
        with pytest.raises(EnterpriseAuthException):
            enterprise_auth_service.get_provider_config("nonexistent")

    @pytest.mark.asyncio
    async def test_authenticate_with_ldap(self, enterprise_auth_service):
        """Test authentication with LDAP provider."""
        with patch.object(enterprise_auth_service, 'providers') as mock_providers:
            mock_ldap = Mock()
            mock_user = EnterpriseUser(
                user_id="user123",
                username="testuser",
                email="test@example.com"
            )
            mock_ldap.authenticate.return_value = mock_user
            mock_providers.__getitem__.return_value = mock_ldap
            
            user = await enterprise_auth_service.authenticate(
                "testuser", "password", provider="ldap"
            )
            
            assert user is not None
            assert user.username == "testuser"

    @pytest.mark.asyncio
    async def test_authenticate_with_default_provider(self, enterprise_auth_service):
        """Test authentication with default provider."""
        with patch.object(enterprise_auth_service, 'providers') as mock_providers:
            mock_ldap = Mock()
            mock_user = EnterpriseUser(
                user_id="user123",
                username="testuser",
                email="test@example.com"
            )
            mock_ldap.authenticate.return_value = mock_user
            mock_providers.__getitem__.return_value = mock_ldap
            
            user = await enterprise_auth_service.authenticate("testuser", "password")
            
            assert user is not None
            assert user.username == "testuser"

    @pytest.mark.asyncio
    async def test_authenticate_provider_not_found(self, enterprise_auth_service):
        """Test authentication with non-existent provider."""
        with pytest.raises(EnterpriseAuthException):
            await enterprise_auth_service.authenticate(
                "testuser", "password", provider="nonexistent"
            )

    def test_create_session(self, enterprise_auth_service):
        """Test creating user session."""
        user = EnterpriseUser(
            user_id="user123",
            username="testuser",
            email="test@example.com"
        )
        
        session = enterprise_auth_service.create_session(user)
        
        assert session is not None
        assert session.user_id == "user123"
        assert session.provider == "ldap"

    def test_validate_session(self, enterprise_auth_service):
        """Test session validation."""
        user = EnterpriseUser(
            user_id="user123",
            username="testuser",
            email="test@example.com"
        )
        
        session = enterprise_auth_service.create_session(user)
        is_valid = enterprise_auth_service.validate_session(session.session_id)
        
        assert is_valid is True

    def test_invalidate_session(self, enterprise_auth_service):
        """Test session invalidation."""
        user = EnterpriseUser(
            user_id="user123",
            username="testuser",
            email="test@example.com"
        )
        
        session = enterprise_auth_service.create_session(user)
        enterprise_auth_service.invalidate_session(session.session_id)
        
        is_valid = enterprise_auth_service.validate_session(session.session_id)
        assert is_valid is False

    def test_generate_jwt_token(self, enterprise_auth_service):
        """Test JWT token generation."""
        user = EnterpriseUser(
            user_id="user123",
            username="testuser",
            email="test@example.com"
        )
        
        token = enterprise_auth_service.generate_jwt_token(user)
        
        assert token is not None
        # Verify token can be decoded
        decoded = jwt.decode(token, options={"verify_signature": False})
        assert decoded["user_id"] == "user123"
        assert decoded["username"] == "testuser"

    def test_validate_jwt_token(self, enterprise_auth_service):
        """Test JWT token validation."""
        user = EnterpriseUser(
            user_id="user123",
            username="testuser",
            email="test@example.com"
        )
        
        token = enterprise_auth_service.generate_jwt_token(user)
        validated_user = enterprise_auth_service.validate_jwt_token(token)
        
        assert validated_user is not None
        assert validated_user.user_id == "user123"

    def test_validate_jwt_token_expired(self, enterprise_auth_service):
        """Test JWT token validation with expired token."""
        # Create token with past expiry
        payload = {
            "user_id": "user123",
            "username": "testuser",
            "exp": datetime.now(timezone.utc) - timedelta(hours=1)
        }
        
        with patch('ai_engine.api.auth_enterprise.jwt.encode') as mock_encode:
            mock_encode.return_value = "expired_token"
            
            with pytest.raises(EnterpriseAuthException):
                enterprise_auth_service.validate_jwt_token("expired_token")

    def test_check_permission(self, enterprise_auth_service):
        """Test permission checking."""
        user = EnterpriseUser(
            user_id="user123",
            username="testuser",
            email="test@example.com",
            permissions=["read", "write"]
        )
        
        assert enterprise_auth_service.check_permission(user, "read") is True
        assert enterprise_auth_service.check_permission(user, "write") is True
        assert enterprise_auth_service.check_permission(user, "delete") is False

    def test_check_role(self, enterprise_auth_service):
        """Test role checking."""
        user = EnterpriseUser(
            user_id="user123",
            username="testuser",
            email="test@example.com",
            roles=["admin", "user"]
        )
        
        assert enterprise_auth_service.check_role(user, "admin") is True
        assert enterprise_auth_service.check_role(user, "user") is True
        assert enterprise_auth_service.check_role(user, "superuser") is False

    def test_validate_password_policy(self, enterprise_auth_service):
        """Test password policy validation."""
        # Valid password
        assert enterprise_auth_service.validate_password_policy("ValidPass123!") is True
        
        # Invalid passwords
        assert enterprise_auth_service.validate_password_policy("short") is False
        assert enterprise_auth_service.validate_password_policy("nouppercase123!") is False
        assert enterprise_auth_service.validate_password_policy("NOLOWERCASE123!") is False
        assert enterprise_auth_service.validate_password_policy("NoNumbers!") is False
        assert enterprise_auth_service.validate_password_policy("NoSpecial123") is False

    def test_enable_mfa(self, enterprise_auth_service):
        """Test MFA enablement."""
        user = EnterpriseUser(
            user_id="user123",
            username="testuser",
            email="test@example.com"
        )
        
        mfa_secret = enterprise_auth_service.enable_mfa(user)
        
        assert mfa_secret is not None
        assert len(mfa_secret) > 0

    def test_verify_mfa_token(self, enterprise_auth_service):
        """Test MFA token verification."""
        user = EnterpriseUser(
            user_id="user123",
            username="testuser",
            email="test@example.com"
        )
        
        mfa_secret = enterprise_auth_service.enable_mfa(user)
        
        # This would normally require a real TOTP token
        # For testing, we'll mock the verification
        with patch('ai_engine.api.auth_enterprise.pyotp') as mock_pyotp:
            mock_totp = Mock()
            mock_totp.verify.return_value = True
            mock_pyotp.TOTP.return_value = mock_totp
            
            is_valid = enterprise_auth_service.verify_mfa_token(user, "123456")
            assert is_valid is True

    def test_audit_log(self, enterprise_auth_service):
        """Test audit logging."""
        user = EnterpriseUser(
            user_id="user123",
            username="testuser",
            email="test@example.com"
        )
        
        with patch('ai_engine.api.auth_enterprise.logging') as mock_logging:
            enterprise_auth_service.audit_log("LOGIN", user, "192.168.1.1")
            
            mock_logging.getLogger.return_value.info.assert_called_once()

    def test_cleanup_expired_sessions(self, enterprise_auth_service):
        """Test cleanup of expired sessions."""
        # Create session with past expiry
        user = EnterpriseUser(
            user_id="user123",
            username="testuser",
            email="test@example.com"
        )
        
        session = enterprise_auth_service.create_session(user)
        session.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        
        enterprise_auth_service.cleanup_expired_sessions()
        
        # Session should be removed
        is_valid = enterprise_auth_service.validate_session(session.session_id)
        assert is_valid is False

    def test_get_user_sessions(self, enterprise_auth_service):
        """Test getting user sessions."""
        user = EnterpriseUser(
            user_id="user123",
            username="testuser",
            email="test@example.com"
        )
        
        session1 = enterprise_auth_service.create_session(user)
        session2 = enterprise_auth_service.create_session(user)
        
        sessions = enterprise_auth_service.get_user_sessions("user123")
        
        assert len(sessions) == 2
        assert session1.session_id in [s.session_id for s in sessions]
        assert session2.session_id in [s.session_id for s in sessions]

    def test_enterprise_auth_service_error_handling(self, enterprise_auth_service):
        """Test enterprise auth service error handling."""
        with pytest.raises(EnterpriseAuthValidationError):
            enterprise_auth_service.validate_password_policy("")

    def test_enterprise_auth_service_concurrent_access(self, enterprise_auth_service):
        """Test enterprise auth service concurrent access."""
        user = EnterpriseUser(
            user_id="user123",
            username="testuser",
            email="test@example.com"
        )
        
        def create_session():
            return enterprise_auth_service.create_session(user)
        
        # Run concurrent operations
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_session) for _ in range(10)]
            sessions = [future.result() for future in futures]
        
        # All sessions should be created successfully
        assert len(sessions) == 10
        assert all(session.user_id == "user123" for session in sessions)