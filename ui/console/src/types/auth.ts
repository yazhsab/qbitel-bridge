export interface User {
  id: string;
  profile: {
    sub: string;
    name?: string;
    given_name?: string;
    family_name?: string;
    email?: string;
    email_verified?: boolean;
    preferred_username?: string;
    roles?: string[];
    groups?: string[];
    organization?: string;
    department?: string;
    job_title?: string;
    phone_number?: string;
    address?: {
      street_address?: string;
      locality?: string;
      region?: string;
      postal_code?: string;
      country?: string;
    };
    picture?: string;
    website?: string;
    locale?: string;
    zoneinfo?: string;
    updated_at?: number;
  };
  access_token: string;
  token_type: string;
  scope: string;
  expires_at?: number;
  expires_in?: number;
  refresh_token?: string;
  id_token?: string;
  session_state?: string;
  state?: any;
}

export interface AuthState {
  isAuthenticated: boolean;
  isLoading: boolean;
  user: User | null;
  error: string | null;
  lastActivity: number;
  sessionTimeout: number;
  mfaRequired: boolean;
  mfaVerified: boolean;
}

export interface LoginOptions {
  redirect_uri?: string;
  scope?: string;
  response_type?: string;
  state?: string;
  nonce?: string;
  prompt?: string;
  max_age?: number;
  ui_locales?: string;
  id_token_hint?: string;
  login_hint?: string;
  acr_values?: string;
  extraQueryParams?: Record<string, string>;
  extraTokenParams?: Record<string, string>;
}

export interface LogoutOptions {
  post_logout_redirect_uri?: string;
  id_token_hint?: string;
  state?: string;
  extraQueryParams?: Record<string, string>;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  refresh_token?: string;
  scope?: string;
  id_token?: string;
}

export interface RefreshTokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  refresh_token?: string;
  scope?: string;
}

export interface UserInfo {
  sub: string;
  name?: string;
  given_name?: string;
  family_name?: string;
  middle_name?: string;
  nickname?: string;
  preferred_username?: string;
  profile?: string;
  picture?: string;
  website?: string;
  email?: string;
  email_verified?: boolean;
  gender?: string;
  birthdate?: string;
  zoneinfo?: string;
  locale?: string;
  phone_number?: string;
  phone_number_verified?: boolean;
  address?: {
    formatted?: string;
    street_address?: string;
    locality?: string;
    region?: string;
    postal_code?: string;
    country?: string;
  };
  updated_at?: number;
  roles?: string[];
  groups?: string[];
  organization?: string;
  department?: string;
  job_title?: string;
  permissions?: string[];
}

export interface AuthConfig {
  authority: string;
  client_id: string;
  redirect_uri: string;
  post_logout_redirect_uri: string;
  response_type: string;
  scope: string;
  automaticSilentRenew: boolean;
  silent_redirect_uri: string;
  accessTokenExpiringNotificationTime: number;
  userStore: any;
  metadata?: {
    issuer?: string;
    authorization_endpoint?: string;
    token_endpoint?: string;
    userinfo_endpoint?: string;
    end_session_endpoint?: string;
    jwks_uri?: string;
    check_session_iframe?: string;
    revocation_endpoint?: string;
    introspection_endpoint?: string;
  };
  signingKeys?: any[];
  filterProtocolClaims: boolean;
  loadUserInfo: boolean;
  staleStateAge: number;
  clockSkew: number;
  revokeAccessTokenOnSignout: boolean;
  includeIdTokenInSilentRenew: boolean;
  extraQueryParams?: Record<string, string>;
  extraTokenParams?: Record<string, string>;
}

export interface AuthError {
  error: string;
  error_description?: string;
  error_uri?: string;
  state?: string;
}

export interface SessionInfo {
  user_id: string;
  session_id: string;
  created_at: number;
  last_activity: number;
  expires_at: number;
  ip_address: string;
  user_agent: string;
  location?: {
    country?: string;
    region?: string;
    city?: string;
  };
  mfa_verified: boolean;
  device_fingerprint?: string;
}

export interface AuditEvent {
  id: string;
  user_id: string;
  session_id: string;
  event_type: 'login' | 'logout' | 'token_refresh' | 'mfa_challenge' | 'mfa_verify' | 'session_timeout' | 'access_denied';
  timestamp: number;
  ip_address: string;
  user_agent: string;
  success: boolean;
  error_message?: string;
  metadata?: Record<string, any>;
}

export interface MfaChallenge {
  challenge_id: string;
  challenge_type: 'totp' | 'sms' | 'email' | 'push' | 'webauthn';
  expires_at: number;
  metadata?: Record<string, any>;
}

export interface MfaVerification {
  challenge_id: string;
  response: string;
  metadata?: Record<string, any>;
}

export interface Permission {
  id: string;
  name: string;
  description: string;
  resource: string;
  action: string;
  conditions?: Record<string, any>;
}

export interface Role {
  id: string;
  name: string;
  description: string;
  permissions: Permission[];
  is_system_role: boolean;
  created_at: string;
  updated_at: string;
}

export interface Group {
  id: string;
  name: string;
  description: string;
  roles: Role[];
  members: string[];
  created_at: string;
  updated_at: string;
}

// RBAC helper types
export type ResourceType = 'device' | 'policy' | 'user' | 'role' | 'group' | 'system' | 'audit';
export type ActionType = 'create' | 'read' | 'update' | 'delete' | 'execute' | 'approve' | 'manage';

export interface AccessRequest {
  resource: ResourceType;
  action: ActionType;
  resource_id?: string;
  context?: Record<string, any>;
}

export interface AccessResponse {
  allowed: boolean;
  reason?: string;
  conditions?: Record<string, any>;
}

// Event types for auth service
export interface AuthEventMap {
  'user_loaded': User;
  'user_unloaded': void;
  'access_token_expiring': void;
  'access_token_expired': void;
  'silent_renew_error': Error;
  'user_signed_out': void;
  'session_timeout': void;
  'mfa_required': MfaChallenge;
  'mfa_verified': void;
  'auth_error': AuthError;
}

export type AuthEventType = keyof AuthEventMap;
export type AuthEventHandler<T extends AuthEventType> = (data: AuthEventMap[T]) => void;

// Constants
export const AUTH_STORAGE_KEYS = {
  USER: 'qbitel_auth_user',
  ACCESS_TOKEN: 'qbitel_auth_access_token',
  REFRESH_TOKEN: 'qbitel_auth_refresh_token',
  ID_TOKEN: 'qbitel_auth_id_token',
  SESSION_STATE: 'qbitel_auth_session_state',
  LAST_ACTIVITY: 'qbitel_auth_last_activity',
  MFA_VERIFIED: 'qbitel_auth_mfa_verified',
} as const;

export const DEFAULT_SCOPES = [
  'openid',
  'profile',
  'email',
  'roles',
  'groups',
  'qbitel:admin',
  'qbitel:device:manage',
  'qbitel:policy:manage',
  'qbitel:audit:read',
] as const;

export const SESSION_TIMEOUT_WARNING = 5 * 60 * 1000; // 5 minutes before timeout
export const DEFAULT_SESSION_TIMEOUT = 8 * 60 * 60 * 1000; // 8 hours
export const TOKEN_REFRESH_THRESHOLD = 5 * 60 * 1000; // 5 minutes before expiry