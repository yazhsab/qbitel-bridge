/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string;
  readonly VITE_OIDC_AUTHORITY: string;
  readonly VITE_OIDC_CLIENT_ID: string;
  readonly VITE_OIDC_REDIRECT_URI: string;
  readonly VITE_OIDC_POST_LOGOUT_REDIRECT_URI: string;
  readonly VITE_OIDC_SCOPE: string;
  readonly VITE_ENABLE_MFA: string;
  readonly VITE_SESSION_TIMEOUT: string;
  readonly VITE_AUDIT_ENABLED: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

// Global environment variables for compatibility
declare global {
  namespace NodeJS {
    interface ProcessEnv {
      NODE_ENV: 'development' | 'production' | 'test';
      VITE_API_BASE_URL?: string;
      VITE_OIDC_AUTHORITY?: string;
      VITE_OIDC_CLIENT_ID?: string;
      VITE_OIDC_REDIRECT_URI?: string;
      VITE_OIDC_POST_LOGOUT_REDIRECT_URI?: string;
      VITE_OIDC_SCOPE?: string;
      VITE_ENABLE_MFA?: string;
      VITE_SESSION_TIMEOUT?: string;
      VITE_AUDIT_ENABLED?: string;
    }
  }
}

export {};