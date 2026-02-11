import { User, UserManager, WebStorageStateStore } from 'oidc-client-ts';

export interface QBITELUser {
  id: string;
  email: string;
  name: string;
  roles: string[];
  permissions: string[];
  organizationId: string;
  organizationName: string;
  lastLogin: Date;
  sessionExpiry: Date;
}

export interface AuthConfig {
  authority: string;
  clientId: string;
  redirectUri: string;
  postLogoutRedirectUri: string;
  responseType: string;
  scope: string;
  automaticSilentRenew: boolean;
  silentRedirectUri: string;
  loadUserInfo: boolean;
}

export class OIDCAuthService {
  private userManager: UserManager;
  private config: AuthConfig;

  constructor(config: AuthConfig) {
    this.config = config;
    this.userManager = new UserManager({
      authority: config.authority,
      client_id: config.clientId,
      redirect_uri: config.redirectUri,
      post_logout_redirect_uri: config.postLogoutRedirectUri,
      response_type: config.responseType,
      scope: config.scope,
      automaticSilentRenew: config.automaticSilentRenew,
      silent_redirect_uri: config.silentRedirectUri,
      loadUserInfo: config.loadUserInfo,
      userStore: new WebStorageStateStore({ store: window.localStorage }),
      metadata: {
        issuer: config.authority,
        authorization_endpoint: `${config.authority}/auth`,
        token_endpoint: `${config.authority}/token`,
        userinfo_endpoint: `${config.authority}/userinfo`,
        end_session_endpoint: `${config.authority}/logout`,
        jwks_uri: `${config.authority}/.well-known/jwks.json`,
      },
    });

    // Set up event handlers
    this.userManager.events.addUserLoaded((user: User) => {
      console.log('User loaded:', user);
    });

    this.userManager.events.addUserUnloaded(() => {
      console.log('User unloaded');
    });

    this.userManager.events.addAccessTokenExpiring(() => {
      console.log('Access token expiring');
    });

    this.userManager.events.addAccessTokenExpired(() => {
      console.log('Access token expired');
      this.signoutRedirect();
    });

    this.userManager.events.addSilentRenewError((error: Error) => {
      console.error('Silent renew error:', error);
    });
  }

  async signinRedirect(): Promise<void> {
    return this.userManager.signinRedirect();
  }

  async signinRedirectCallback(): Promise<User> {
    return this.userManager.signinRedirectCallback();
  }

  async signoutRedirect(): Promise<void> {
    return this.userManager.signoutRedirect();
  }

  async signoutRedirectCallback(): Promise<void> {
    return this.userManager.signoutRedirectCallback();
  }

  async getUser(): Promise<User | null> {
    return this.userManager.getUser();
  }

  async removeUser(): Promise<void> {
    return this.userManager.removeUser();
  }

  async signinSilent(): Promise<User> {
    return this.userManager.signinSilent();
  }

  async renewToken(): Promise<User> {
    return this.userManager.signinSilent();
  }

  transformUser(oidcUser: User): QBITELUser {
    const profile = oidcUser.profile;
    
    return {
      id: profile.sub,
      email: profile.email || '',
      name: profile.name || profile.preferred_username || '',
      roles: profile.roles || [],
      permissions: profile.permissions || [],
      organizationId: profile.organization_id || '',
      organizationName: profile.organization_name || '',
      lastLogin: new Date(profile.last_login || Date.now()),
      sessionExpiry: new Date(oidcUser.expires_at * 1000),
    };
  }

  hasRole(user: QBITELUser, role: string): boolean {
    return user.roles.includes(role);
  }

  hasPermission(user: QBITELUser, permission: string): boolean {
    return user.permissions.includes(permission);
  }

  hasAnyRole(user: QBITELUser, roles: string[]): boolean {
    return roles.some(role => user.roles.includes(role));
  }

  hasAnyPermission(user: QBITELUser, permissions: string[]): boolean {
    return permissions.some(permission => user.permissions.includes(permission));
  }

  isTokenExpired(user: QBITELUser): boolean {
    return new Date() >= user.sessionExpiry;
  }

  getTokenExpiryTime(user: QBITELUser): number {
    return user.sessionExpiry.getTime() - Date.now();
  }

  // Role-based access control helpers
  canAccessDeviceManagement(user: QBITELUser): boolean {
    return this.hasAnyPermission(user, [
      'device:read',
      'device:write',
      'device:admin'
    ]);
  }

  canManageDevices(user: QBITELUser): boolean {
    return this.hasAnyPermission(user, [
      'device:write',
      'device:admin'
    ]);
  }

  canSuspendDevices(user: QBITELUser): boolean {
    return this.hasAnyPermission(user, [
      'device:suspend',
      'device:admin'
    ]);
  }

  canDecommissionDevices(user: QBITELUser): boolean {
    return this.hasAnyPermission(user, [
      'device:decommission',
      'device:admin'
    ]);
  }

  canAccessPolicyManagement(user: QBITELUser): boolean {
    return this.hasAnyPermission(user, [
      'policy:read',
      'policy:write',
      'policy:admin'
    ]);
  }

  canManagePolicies(user: QBITELUser): boolean {
    return this.hasAnyPermission(user, [
      'policy:write',
      'policy:admin'
    ]);
  }

  canAccessComplianceReports(user: QBITELUser): boolean {
    return this.hasAnyPermission(user, [
      'compliance:read',
      'compliance:admin'
    ]);
  }

  canAccessAuditLogs(user: QBITELUser): boolean {
    return this.hasAnyPermission(user, [
      'audit:read',
      'audit:admin'
    ]);
  }

  canAccessSystemSettings(user: QBITELUser): boolean {
    return this.hasAnyRole(user, ['admin', 'system_admin']);
  }

  canAccessOrganizationSettings(user: QBITELUser): boolean {
    return this.hasAnyRole(user, ['admin', 'org_admin']);
  }

  // Audit logging
  async logUserAction(user: QBITELUser, action: string, resource: string, details?: any): Promise<void> {
    try {
      const auditEvent = {
        userId: user.id,
        userEmail: user.email,
        organizationId: user.organizationId,
        action,
        resource,
        details,
        timestamp: new Date().toISOString(),
        sessionId: this.getSessionId(),
        userAgent: navigator.userAgent,
        ipAddress: await this.getClientIP(),
      };

      // Send audit event to backend
      await fetch('/api/audit/events', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${(await this.getUser())?.access_token}`,
        },
        body: JSON.stringify(auditEvent),
      });
    } catch (error) {
      console.error('Failed to log audit event:', error);
    }
  }

  private getSessionId(): string {
    // Generate or retrieve session ID
    let sessionId = sessionStorage.getItem('qbitel_session_id');
    if (!sessionId) {
      sessionId = crypto.randomUUID();
      sessionStorage.setItem('qbitel_session_id', sessionId);
    }
    return sessionId;
  }

  private async getClientIP(): Promise<string> {
    try {
      const response = await fetch('/api/client-ip');
      const data = await response.json();
      return data.ip || 'unknown';
    } catch {
      return 'unknown';
    }
  }

  // Session management
  async extendSession(): Promise<void> {
    try {
      await this.signinSilent();
    } catch (error) {
      console.error('Failed to extend session:', error);
      throw error;
    }
  }

  startSessionMonitoring(): void {
    setInterval(async () => {
      const user = await this.getUser();
      if (user) {
        const qbitelUser = this.transformUser(user);
        const timeToExpiry = this.getTokenExpiryTime(qbitelUser);
        
        // Renew token if expiring within 5 minutes
        if (timeToExpiry < 5 * 60 * 1000 && timeToExpiry > 0) {
          try {
            await this.extendSession();
          } catch (error) {
            console.error('Failed to renew token:', error);
            // Redirect to login if renewal fails
            await this.signoutRedirect();
          }
        }
      }
    }, 60000); // Check every minute
  }

  // Security headers and CSRF protection
  async getSecurityHeaders(): Promise<Record<string, string>> {
    const user = await this.getUser();
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (user?.access_token) {
      headers['Authorization'] = `Bearer ${user.access_token}`;
    }

    // Add CSRF token if available
    const csrfToken = this.getCSRFToken();
    if (csrfToken) {
      headers['X-CSRF-Token'] = csrfToken;
    }

    return headers;
  }

  private getCSRFToken(): string | null {
    const meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.getAttribute('content') : null;
  }

  // Multi-factor authentication support
  async initiateMFA(method: 'totp' | 'sms' | 'email'): Promise<{ challengeId: string }> {
    const headers = await this.getSecurityHeaders();
    const response = await fetch('/api/auth/mfa/initiate', {
      method: 'POST',
      headers,
      body: JSON.stringify({ method }),
    });

    if (!response.ok) {
      throw new Error('Failed to initiate MFA');
    }

    return response.json();
  }

  async verifyMFA(challengeId: string, code: string): Promise<{ success: boolean }> {
    const headers = await this.getSecurityHeaders();
    const response = await fetch('/api/auth/mfa/verify', {
      method: 'POST',
      headers,
      body: JSON.stringify({ challengeId, code }),
    });

    if (!response.ok) {
      throw new Error('Failed to verify MFA');
    }

    return response.json();
  }
}

// Default configuration
export const defaultAuthConfig: AuthConfig = {
  authority: (import.meta as any).env?.VITE_OIDC_AUTHORITY || 'https://auth.qbitel.local',
  clientId: (import.meta as any).env?.VITE_OIDC_CLIENT_ID || 'qbitel-console',
  redirectUri: `${window.location.origin}/auth/callback`,
  postLogoutRedirectUri: `${window.location.origin}/`,
  responseType: 'code',
  scope: 'openid profile email roles permissions organization',
  automaticSilentRenew: true,
  silentRedirectUri: `${window.location.origin}/auth/silent-callback`,
  loadUserInfo: true,
};

// Singleton instance
export const authService = new OIDCAuthService(defaultAuthConfig);

// Export with both names for compatibility
export { OIDCAuthService as OidcAuthService };