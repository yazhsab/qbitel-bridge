import React, { Suspense, lazy } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import {
  Box,
  CircularProgress,
  Alert,
  Typography,
  Paper,
} from '@mui/material';
import { DeviceApiClient } from '../api/devices';
import { User } from '../types/auth';
import { OidcAuthService } from '../auth/oidc';

// Lazy load components for better performance
const Dashboard = lazy(() => import('../components/Dashboard'));
const DeviceManagement = lazy(() => import('../components/DeviceManagement'));
const PolicyManagement = lazy(() => import('../components/PolicyManagement'));
const ComplianceReporting = lazy(() => import('../components/ComplianceReporting'));
const SecurityMonitoring = lazy(() => import('../components/SecurityMonitoring'));
const SystemSettings = lazy(() => import('../components/SystemSettings'));
const UserProfile = lazy(() => import('../components/UserProfile'));

// New Advanced Components
const ProtocolVisualization = lazy(() => import('../components/ProtocolVisualization'));
const AIModelMonitoring = lazy(() => import('../components/AIModelMonitoring'));
const ThreatIntelligence = lazy(() => import('../components/ThreatIntelligence'));
const AdvancedAnalytics = lazy(() => import('../components/AdvancedAnalytics'));

interface AppRoutesProps {
  user: User;
  apiClient: DeviceApiClient;
  authService: OidcAuthService;
}

interface RouteConfig {
  path: string;
  component: React.ComponentType<any>;
  permissions?: string[];
  roles?: string[];
  exact?: boolean;
  title: string;
  description: string;
  category: 'core' | 'advanced' | 'admin';
  beta?: boolean;
  enterprise?: boolean;
}

// Loading component
const LoadingFallback: React.FC<{ message?: string }> = ({ message = 'Loading...' }) => (
  <Box
    display="flex"
    flexDirection="column"
    alignItems="center"
    justifyContent="center"
    minHeight="400px"
    gap={2}
  >
    <CircularProgress size={48} />
    <Typography variant="h6" color="textSecondary">
      {message}
    </Typography>
  </Box>
);

// Error boundary fallback
const ErrorFallback: React.FC<{ error: Error }> = ({ error }) => (
  <Box p={3}>
    <Alert severity="error" sx={{ mb: 2 }}>
      <Typography variant="h6" gutterBottom>
        Component Error
      </Typography>
      <Typography variant="body2">
        {error.message || 'An unexpected error occurred while loading this component.'}
      </Typography>
    </Alert>
    <Paper sx={{ p: 3, textAlign: 'center' }}>
      <Typography variant="body1" color="textSecondary">
        Please try refreshing the page or contact support if the issue persists.
      </Typography>
    </Paper>
  </Box>
);

// Route configurations with RBAC
export const routeConfigs: RouteConfig[] = [
  {
    path: '/dashboard',
    component: Dashboard,
    title: 'Dashboard',
    description: 'Main dashboard with system overview',
    category: 'core',
    permissions: ['dashboard:read'],
  },
  {
    path: '/devices',
    component: DeviceManagement,
    title: 'Device Management',
    description: 'Manage and monitor devices',
    category: 'core',
    permissions: ['device:read'],
  },
  {
    path: '/policies',
    component: PolicyManagement,
    title: 'Policy Management',
    description: 'Configure and deploy policies',
    category: 'core',
    permissions: ['policy:read'],
  },
  {
    path: '/compliance',
    component: ComplianceReporting,
    title: 'Compliance Reporting',
    description: 'Compliance status and reports',
    category: 'core',
    permissions: ['compliance:read'],
  },
  {
    path: '/security',
    component: SecurityMonitoring,
    title: 'Security Monitoring',
    description: 'Security alerts and monitoring',
    category: 'core',
    permissions: ['security:read'],
  },
  {
    path: '/protocols',
    component: ProtocolVisualization,
    title: 'Protocol Discovery',
    description: 'Real-time protocol visualization and analysis',
    category: 'advanced',
    permissions: ['protocol:read'],
    enterprise: true,
  },
  {
    path: '/ai-models',
    component: AIModelMonitoring,
    title: 'AI Model Monitoring',
    description: 'Monitor and manage AI models',
    category: 'advanced',
    permissions: ['ai:read'],
    enterprise: true,
  },
  {
    path: '/threat-intelligence',
    component: ThreatIntelligence,
    title: 'Threat Intelligence',
    description: 'Security operations center and threat analysis',
    category: 'advanced',
    permissions: ['threat:read'],
    enterprise: true,
  },
  {
    path: '/analytics',
    component: AdvancedAnalytics,
    title: 'Advanced Analytics',
    description: 'Predictive analytics and insights',
    category: 'advanced',
    permissions: ['analytics:read'],
    enterprise: true,
    beta: true,
  },
  {
    path: '/settings',
    component: SystemSettings,
    title: 'System Settings',
    description: 'System configuration and administration',
    category: 'admin',
    roles: ['admin', 'system_admin'],
  },
  {
    path: '/profile',
    component: UserProfile,
    title: 'User Profile',
    description: 'User account and preferences',
    category: 'core',
  },
];

// Permission checker
function hasAccess(user: User, route: RouteConfig, authService: OidcAuthService): boolean {
  // Check roles if specified
  if (route.roles && route.roles.length > 0) {
    const userRoles = user.profile.roles || [];
    if (!route.roles.some(role => userRoles.includes(role))) {
      return false;
    }
  }

  // Check permissions if specified
  if (route.permissions && route.permissions.length > 0) {
    const userPermissions = user.profile.permissions || [];
    if (!route.permissions.some(permission => userPermissions.includes(permission))) {
      return false;
    }
  }

  return true;
}

// Higher-order component for protected routes
function ProtectedRoute({
  children,
  user,
  authService,
  route,
}: {
  children: React.ReactNode;
  user: User;
  authService: OidcAuthService;
  route: RouteConfig;
}) {
  if (!hasAccess(user, route, authService)) {
    return (
      <Box p={3}>
        <Alert severity="warning">
          <Typography variant="h6" gutterBottom>
            Access Denied
          </Typography>
          <Typography variant="body2">
            You don't have permission to access this page. Please contact your administrator if you believe this is an error.
          </Typography>
        </Alert>
      </Box>
    );
  }

  // Check for enterprise features
  if (route.enterprise && !user.profile.organization?.includes('enterprise')) {
    return (
      <Box p={3}>
        <Alert severity="info">
          <Typography variant="h6" gutterBottom>
            Enterprise Feature
          </Typography>
          <Typography variant="body2">
            This feature is available in the Enterprise edition. Contact sales to upgrade your plan.
          </Typography>
        </Alert>
      </Box>
    );
  }

  // Beta feature warning
  if (route.beta) {
    return (
      <Box>
        <Alert severity="info" sx={{ mb: 2 }}>
          <Typography variant="body2">
            <strong>Beta Feature:</strong> This feature is in beta and may have limited functionality.
          </Typography>
        </Alert>
        {children}
      </Box>
    );
  }

  return <>{children}</>;
}

const AppRoutes: React.FC<AppRoutesProps> = ({ user, apiClient, authService }) => {
  // Get accessible routes for current user
  const accessibleRoutes = routeConfigs.filter(route => 
    hasAccess(user, route, authService)
  );

  return (
    <Routes>
      {/* Default redirect */}
      <Route path="/" element={<Navigate to="/dashboard" replace />} />
      
      {/* Authentication callback */}
      <Route
        path="/callback"
        element={
          <LoadingFallback message="Processing authentication..." />
        }
      />
      
      {/* Silent callback for token renewal */}
      <Route
        path="/silent-callback"
        element={
          <LoadingFallback message="Renewing session..." />
        }
      />

      {/* Dynamic routes based on user permissions */}
      {accessibleRoutes.map((route) => {
        const Component = route.component;
        
        return (
          <Route
            key={route.path}
            path={route.path}
            element={
              <ProtectedRoute
                user={user}
                authService={authService}
                route={route}
              >
                <Suspense
                  fallback={<LoadingFallback message={`Loading ${route.title}...`} />}
                >
                  <Component
                    apiClient={apiClient}
                    user={user}
                    authService={authService}
                  />
                </Suspense>
              </ProtectedRoute>
            }
          />
        );
      })}

      {/* 404 Not Found */}
      <Route
        path="*"
        element={
          <Box p={3}>
            <Alert severity="error">
              <Typography variant="h6" gutterBottom>
                Page Not Found
              </Typography>
              <Typography variant="body2">
                The requested page could not be found or you don't have access to it.
              </Typography>
            </Alert>
          </Box>
        }
      />
    </Routes>
  );
};

export default AppRoutes;

// Navigation helper functions
export function getAccessibleRoutes(user: User, authService: OidcAuthService): RouteConfig[] {
  return routeConfigs.filter(route => hasAccess(user, route, authService));
}

export function getRoutesByCategory(
  category: 'core' | 'advanced' | 'admin',
  user: User,
  authService: OidcAuthService
): RouteConfig[] {
  return getAccessibleRoutes(user, authService).filter(route => route.category === category);
}

export function getRouteByPath(path: string): RouteConfig | undefined {
  return routeConfigs.find(route => route.path === path);
}

// Route metadata for navigation components
export interface NavigationItem {
  id: string;
  label: string;
  path: string;
  icon?: React.ComponentType;
  category: string;
  badge?: {
    text: string;
    color: 'default' | 'primary' | 'secondary' | 'error' | 'info' | 'success' | 'warning';
  };
  children?: NavigationItem[];
}

// Convert route configs to navigation items
export function createNavigationItems(
  user: User,
  authService: OidcAuthService,
  icons?: Record<string, React.ComponentType>
): NavigationItem[] {
  const accessibleRoutes = getAccessibleRoutes(user, authService);
  
  return accessibleRoutes
    .filter(route => route.path !== '/profile') // Exclude profile from main nav
    .map(route => ({
      id: route.path.slice(1), // Remove leading slash
      label: route.title,
      path: route.path,
      icon: icons?.[route.path.slice(1)],
      category: route.category,
      badge: route.beta ? { text: 'BETA', color: 'info' as const } : 
             route.enterprise ? { text: 'PRO', color: 'primary' as const } : 
             undefined,
    }));
}

// Breadcrumb helper
export function generateBreadcrumbs(pathname: string): { label: string; path?: string }[] {
  const route = getRouteByPath(pathname);
  const breadcrumbs = [{ label: 'Dashboard', path: '/dashboard' }];
  
  if (route && route.path !== '/dashboard') {
    breadcrumbs.push({ label: route.title });
  }
  
  return breadcrumbs;
}