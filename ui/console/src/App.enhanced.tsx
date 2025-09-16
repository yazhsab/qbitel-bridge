import React, { useEffect, useState, Suspense } from 'react';
import { BrowserRouter as Router } from 'react-router-dom';
import {
  Box,
  CircularProgress,
  Alert,
  Typography,
  Backdrop,
} from '@mui/material';

// Theme and Layout Components
import { EnterpriseThemeProvider } from './theme/ThemeProvider';
import ResponsiveLayout from './components/responsive/ResponsiveLayout';

// Routes and Auth
import AppRoutes from './routes/AppRoutes';
import { OidcAuthService, defaultAuthConfig } from './auth/oidc';
import { DeviceApiClient } from './api/devices';

// WebSocket Services
import { GlobalWebSocketManager } from './services/websocket';

// Types
import type { User } from './types/auth';

interface AppState {
  user: User | null;
  loading: boolean;
  error: string | null;
  initialized: boolean;
}

const EnhancedApp: React.FC = () => {
  const [state, setState] = useState<AppState>({
    user: null,
    loading: true,
    error: null,
    initialized: false,
  });

  // Initialize services
  const [authService] = useState(() => new OidcAuthService(defaultAuthConfig));
  const [apiClient] = useState(() => new DeviceApiClient(
    () => authService.getAccessToken()
  ));
  const [wsManager] = useState(() => GlobalWebSocketManager.getInstance());

  useEffect(() => {
    initializeApplication();
    
    return () => {
      // Cleanup WebSocket connections on unmount
      wsManager.destroy();
    };
  }, []);

  const initializeApplication = async () => {
    try {
      setState(prev => ({ ...prev, loading: true, error: null }));
      
      // Initialize WebSocket connections
      initializeWebSockets();
      
      // Handle authentication callback
      if (window.location.pathname === '/callback') {
        await authService.handleCallback();
        window.history.replaceState({}, document.title, '/dashboard');
        return;
      }

      // Check for existing session
      const user = await authService.getUser();
      if (user) {
        setState(prev => ({ 
          ...prev, 
          user, 
          loading: false, 
          initialized: true 
        }));
      } else {
        // Try silent renewal
        try {
          await authService.renewToken();
          const renewedUser = await authService.getUser();
          if (renewedUser) {
            setState(prev => ({ 
              ...prev, 
              user: renewedUser, 
              loading: false, 
              initialized: true 
            }));
          } else {
            setState(prev => ({ 
              ...prev, 
              loading: false, 
              initialized: true 
            }));
          }
        } catch {
          setState(prev => ({ 
            ...prev, 
            loading: false, 
            initialized: true 
          }));
        }
      }
    } catch (error) {
      console.error('Application initialization failed:', error);
      setState(prev => ({ 
        ...prev, 
        loading: false, 
        error: 'Failed to initialize application',
        initialized: true,
      }));
    }
  };

  const initializeWebSockets = () => {
    // Create main WebSocket connection
    const mainWsConfig = {
      url: `ws://${window.location.host}/api/ws/main`,
      reconnectAttempts: 10,
      reconnectInterval: 3000,
      heartbeatInterval: 30000,
      onOpen: () => console.log('Main WebSocket connected'),
      onClose: () => console.log('Main WebSocket disconnected'),
      onError: (error: Event) => console.error('Main WebSocket error:', error),
    };
    
    wsManager.createClient('main', mainWsConfig);
    
    // Create specialized WebSocket connections for different features
    const protocolWsConfig = {
      url: `ws://${window.location.host}/api/ws/protocols`,
      reconnectAttempts: 5,
      reconnectInterval: 2000,
      onOpen: () => console.log('Protocol WebSocket connected'),
    };
    
    wsManager.createClient('protocols', protocolWsConfig);
    
    const threatWsConfig = {
      url: `ws://${window.location.host}/api/ws/threats`,
      reconnectAttempts: 5,
      reconnectInterval: 2000,
      onOpen: () => console.log('Threat Intelligence WebSocket connected'),
    };
    
    wsManager.createClient('threats', threatWsConfig);
    
    // Connect all clients
    wsManager.getClient('main')?.connect();
    wsManager.getClient('protocols')?.connect();
    wsManager.getClient('threats')?.connect();
  };

  const handleLogin = async () => {
    try {
      setState(prev => ({ ...prev, loading: true, error: null }));
      await authService.login();
    } catch (error) {
      console.error('Login failed:', error);
      setState(prev => ({ 
        ...prev, 
        loading: false, 
        error: 'Login failed. Please try again.' 
      }));
    }
  };

  const handleLogout = async () => {
    try {
      await authService.logout();
      setState(prev => ({ ...prev, user: null }));
      // Disconnect WebSocket connections on logout
      wsManager.destroy();
    } catch (error) {
      console.error('Logout failed:', error);
      setState(prev => ({ 
        ...prev, 
        error: 'Logout failed. Please try again.' 
      }));
    }
  };

  // Loading screen
  if (!state.initialized || state.loading) {
    return (
      <EnterpriseThemeProvider>
        <Backdrop 
          open={true} 
          sx={{ 
            color: '#fff', 
            zIndex: (theme) => theme.zIndex.drawer + 1,
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          }}
        >
          <Box display="flex" flexDirection="column" alignItems="center" gap={3}>
            <CircularProgress 
              size={60} 
              thickness={4}
              sx={{ color: 'white' }}
            />
            <Typography variant="h5" sx={{ fontWeight: 600, color: 'white' }}>
              Loading CronosAI Console...
            </Typography>
            <Typography variant="body1" sx={{ color: 'rgba(255,255,255,0.8)' }}>
              Enterprise-grade AI-powered network security platform
            </Typography>
          </Box>
        </Backdrop>
      </EnterpriseThemeProvider>
    );
  }

  // Error state
  if (state.error) {
    return (
      <EnterpriseThemeProvider>
        <Box
          display="flex"
          flexDirection="column"
          alignItems="center"
          justifyContent="center"
          minHeight="100vh"
          bgcolor="background.default"
          p={3}
        >
          <Alert 
            severity="error" 
            sx={{ 
              mb: 3, 
              maxWidth: 600,
              '& .MuiAlert-message': { fontSize: '1.1rem' }
            }}
          >
            <Typography variant="h6" gutterBottom>
              Application Error
            </Typography>
            <Typography variant="body1">
              {state.error}
            </Typography>
          </Alert>
          <Box display="flex" gap={2}>
            <button
              onClick={() => window.location.reload()}
              style={{
                padding: '12px 24px',
                backgroundColor: '#1976d2',
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                fontSize: '16px',
                fontWeight: 500,
                cursor: 'pointer',
              }}
            >
              Retry
            </button>
          </Box>
        </Box>
      </EnterpriseThemeProvider>
    );
  }

  // Login screen
  if (!state.user) {
    return (
      <EnterpriseThemeProvider>
        <Box
          display="flex"
          flexDirection="column"
          alignItems="center"
          justifyContent="center"
          minHeight="100vh"
          sx={{
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            position: 'relative',
            overflow: 'hidden',
          }}
        >
          {/* Background Pattern */}
          <Box
            sx={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              opacity: 0.1,
              backgroundImage: 'radial-gradient(circle, white 1px, transparent 1px)',
              backgroundSize: '20px 20px',
            }}
          />
          
          <Box
            display="flex"
            flexDirection="column"
            alignItems="center"
            bgcolor="rgba(255, 255, 255, 0.95)"
            p={5}
            borderRadius={3}
            boxShadow={6}
            maxWidth={450}
            width="90%"
            textAlign="center"
            position="relative"
            sx={{
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(255, 255, 255, 0.2)',
            }}
          >
            <Typography 
              variant="h3" 
              component="h1" 
              gutterBottom 
              sx={{ 
                fontWeight: 700,
                background: 'linear-gradient(45deg, #667eea, #764ba2)',
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              CronosAI Console
            </Typography>
            
            <Typography 
              variant="h6" 
              color="text.secondary" 
              sx={{ mb: 2, fontWeight: 500 }}
            >
              Enterprise AI-Powered Network Security
            </Typography>
            
            <Typography variant="body1" color="text.secondary" paragraph>
              Access advanced protocol discovery, threat intelligence, AI model monitoring, 
              and predictive analytics in a unified enterprise dashboard.
            </Typography>
            
            <Box 
              sx={{ 
                display: 'grid', 
                gridTemplateColumns: 'repeat(2, 1fr)', 
                gap: 2, 
                mb: 3, 
                width: '100%' 
              }}
            >
              <Box textAlign="center">
                <Typography variant="h4" color="primary" fontWeight="bold">
                  Real-time
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Protocol Discovery
                </Typography>
              </Box>
              <Box textAlign="center">
                <Typography variant="h4" color="secondary" fontWeight="bold">
                  AI-Driven
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Threat Detection
                </Typography>
              </Box>
              <Box textAlign="center">
                <Typography variant="h4" color="success.main" fontWeight="bold">
                  Predictive
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Analytics
                </Typography>
              </Box>
              <Box textAlign="center">
                <Typography variant="h4" color="warning.main" fontWeight="bold">
                  Enterprise
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Security Operations
                </Typography>
              </Box>
            </Box>

            <button
              onClick={handleLogin}
              disabled={state.loading}
              style={{
                width: '100%',
                padding: '16px 24px',
                background: 'linear-gradient(45deg, #667eea, #764ba2)',
                color: 'white',
                border: 'none',
                borderRadius: '12px',
                fontSize: '16px',
                fontWeight: 600,
                cursor: state.loading ? 'not-allowed' : 'pointer',
                boxShadow: '0 4px 15px rgba(102, 126, 234, 0.4)',
                transition: 'all 0.3s ease',
                opacity: state.loading ? 0.7 : 1,
              }}
              onMouseOver={(e) => {
                if (!state.loading) {
                  e.currentTarget.style.transform = 'translateY(-2px)';
                  e.currentTarget.style.boxShadow = '0 6px 20px rgba(102, 126, 234, 0.6)';
                }
              }}
              onMouseOut={(e) => {
                if (!state.loading) {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.boxShadow = '0 4px 15px rgba(102, 126, 234, 0.4)';
                }
              }}
            >
              {state.loading ? (
                <Box display="flex" alignItems="center" justifyContent="center" gap={1}>
                  <CircularProgress size={20} sx={{ color: 'white' }} />
                  <span>Authenticating...</span>
                </Box>
              ) : (
                'Sign In with Enterprise SSO'
              )}
            </button>
            
            <Typography 
              variant="caption" 
              color="text.secondary" 
              sx={{ mt: 2 }}
            >
              Secure authentication powered by OpenID Connect
            </Typography>
          </Box>
        </Box>
      </EnterpriseThemeProvider>
    );
  }

  // Main application with authenticated user
  return (
    <EnterpriseThemeProvider>
      <Router>
        <ResponsiveLayout
          user={state.user}
          apiClient={apiClient}
          onLogout={handleLogout}
        >
          <Suspense
            fallback={
              <Box 
                display="flex" 
                alignItems="center" 
                justifyContent="center" 
                minHeight="400px"
              >
                <CircularProgress />
              </Box>
            }
          >
            <AppRoutes
              user={state.user}
              apiClient={apiClient}
              authService={authService}
            />
          </Suspense>
        </ResponsiveLayout>
      </Router>
    </EnterpriseThemeProvider>
  );
};

export default EnhancedApp;