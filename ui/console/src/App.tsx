import React, { useEffect, useState } from 'react';
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
  useLocation,
} from 'react-router-dom';
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  Box,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemButton,
  Divider,
  Avatar,
  Menu,
  MenuItem,
  Alert,
  Snackbar,
  CircularProgress,
  Backdrop,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  Devices as DevicesIcon,
  Security as SecurityIcon,
  Policy as PolicyIcon,
  Analytics as AnalyticsIcon,
  Settings as SettingsIcon,
  ExitToApp as LogoutIcon,
  AccountCircle as AccountIcon,
  Notifications as NotificationsIcon,
  Badge,
  Timeline as TimelineIcon,
  Psychology as AIIcon,
  Store as MarketplaceIcon,
} from '@mui/icons-material';

// Components
import DeviceManagement from './components/DeviceManagement';
import Dashboard from './components/Dashboard';
import PolicyManagement from './components/PolicyManagement';
import ComplianceReporting from './components/ComplianceReporting';
import SecurityMonitoring from './components/SecurityMonitoring';
import SystemSettings from './components/SystemSettings';
import UserProfile from './components/UserProfile';
import ProtocolVisualization from './components/ProtocolVisualization';
import AIModelMonitoring from './components/AIModelMonitoring';
import ThreatIntelligence from './components/ThreatIntelligence';
import AdvancedAnalytics from './components/AdvancedAnalytics';

// Auth and API
import { OidcAuthService, defaultAuthConfig } from './auth/oidc';
import { DeviceApiClient } from './api/devices';

// Types
import type { User } from './types/auth';

const drawerWidth = 240;

// Theme configuration
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
      dark: '#115293',
      light: '#42a5f5',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h6: {
      fontWeight: 600,
    },
  },
  components: {
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundColor: '#fafafa',
          borderRight: '1px solid #e0e0e0',
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: '#ffffff',
          color: '#333333',
          boxShadow: '0 1px 3px rgba(0,0,0,0.12)',
        },
      },
    },
  },
});

// Navigation items
const navigationItems = [
  { id: 'dashboard', label: 'Dashboard', icon: DashboardIcon, path: '/dashboard' },
  { id: 'marketplace', label: 'Marketplace', icon: MarketplaceIcon, path: '/marketplace' },
  { id: 'devices', label: 'Device Management', icon: DevicesIcon, path: '/devices' },
  { id: 'protocols', label: 'Protocol Visualization', icon: TimelineIcon, path: '/protocols' },
  { id: 'ai-models', label: 'AI Model Monitoring', icon: AIIcon, path: '/ai-models' },
  { id: 'threat-intel', label: 'Threat Intelligence', icon: SecurityIcon, path: '/threat-intelligence' },
  { id: 'analytics', label: 'Advanced Analytics', icon: AnalyticsIcon, path: '/analytics' },
  { id: 'policies', label: 'Policy Management', icon: PolicyIcon, path: '/policies' },
  { id: 'compliance', label: 'Compliance', icon: SecurityIcon, path: '/compliance' },
  { id: 'security', label: 'Security Monitoring', icon: AnalyticsIcon, path: '/security' },
  { id: 'settings', label: 'System Settings', icon: SettingsIcon, path: '/settings' },
];

interface AppState {
  user: User | null;
  loading: boolean;
  error: string | null;
  drawerOpen: boolean;
  userMenuAnchor: HTMLElement | null;
  notifications: number;
}

function AppContent() {
  const location = useLocation();
  const [state, setState] = useState<AppState>({
    user: null,
    loading: true,
    error: null,
    drawerOpen: false,
    userMenuAnchor: null,
    notifications: 0,
  });

  // Initialize services
  const [authService] = useState(() => new OidcAuthService(defaultAuthConfig));
  const [apiClient] = useState(() => new DeviceApiClient(
    () => authService.getAccessToken()
  ));

  useEffect(() => {
    initializeAuth();
  }, []);

  const initializeAuth = async () => {
    try {
      setState(prev => ({ ...prev, loading: true, error: null }));
      
      // Handle callback if present
      if (location.pathname === '/callback') {
        await authService.handleCallback();
        window.history.replaceState({}, document.title, '/dashboard');
        return;
      }

      // Check for existing session
      const user = await authService.getUser();
      if (user) {
        setState(prev => ({ ...prev, user, loading: false }));
        // Load notifications count
        loadNotifications();
      } else {
        // Try silent renewal
        try {
          await authService.renewToken();
          const renewedUser = await authService.getUser();
          if (renewedUser) {
            setState(prev => ({ ...prev, user: renewedUser, loading: false }));
            loadNotifications();
          } else {
            setState(prev => ({ ...prev, loading: false }));
          }
        } catch {
          setState(prev => ({ ...prev, loading: false }));
        }
      }
    } catch (error) {
      console.error('Auth initialization failed:', error);
      setState(prev => ({ 
        ...prev, 
        loading: false, 
        error: 'Authentication initialization failed' 
      }));
    }
  };

  const loadNotifications = async () => {
    try {
      const alerts = await apiClient.getDeviceAlerts(undefined, ['warning', 'error', 'critical'], undefined, false);
      setState(prev => ({ ...prev, notifications: alerts.length }));
    } catch (error) {
      console.error('Failed to load notifications:', error);
    }
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
      setState(prev => ({ ...prev, userMenuAnchor: null }));
      await authService.logout();
      setState(prev => ({ ...prev, user: null }));
    } catch (error) {
      console.error('Logout failed:', error);
      setState(prev => ({ 
        ...prev, 
        error: 'Logout failed. Please try again.' 
      }));
    }
  };

  const handleDrawerToggle = () => {
    setState(prev => ({ ...prev, drawerOpen: !prev.drawerOpen }));
  };

  const handleUserMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setState(prev => ({ ...prev, userMenuAnchor: event.currentTarget }));
  };

  const handleUserMenuClose = () => {
    setState(prev => ({ ...prev, userMenuAnchor: null }));
  };

  const handleErrorClose = () => {
    setState(prev => ({ ...prev, error: null }));
  };

  // Loading screen
  if (state.loading) {
    return (
      <Backdrop open={true} sx={{ color: '#fff', zIndex: theme.zIndex.drawer + 1 }}>
        <Box display="flex" flexDirection="column" alignItems="center">
          <CircularProgress color="inherit" />
          <Typography variant="h6" sx={{ mt: 2 }}>
            Loading QSLB Console...
          </Typography>
        </Box>
      </Backdrop>
    );
  }

  // Login screen
  if (!state.user) {
    return (
      <Box
        display="flex"
        flexDirection="column"
        alignItems="center"
        justifyContent="center"
        minHeight="100vh"
        bgcolor="background.default"
        p={3}
      >
        <Box
          display="flex"
          flexDirection="column"
          alignItems="center"
          bgcolor="background.paper"
          p={4}
          borderRadius={2}
          boxShadow={3}
          maxWidth={400}
          width="100%"
        >
          <Typography variant="h4" component="h1" gutterBottom color="primary">
            QSLB Console
          </Typography>
          <Typography variant="body1" color="text.secondary" align="center" paragraph>
            Quantum-Safe Load Balancer Management Console
          </Typography>
          <Typography variant="body2" color="text.secondary" align="center" paragraph>
            Please sign in to access the administrative interface.
          </Typography>
          <Box mt={3} width="100%">
            <button
              onClick={handleLogin}
              style={{
                width: '100%',
                padding: '12px 24px',
                backgroundColor: theme.palette.primary.main,
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                fontSize: '16px',
                fontWeight: 500,
                cursor: 'pointer',
                transition: 'background-color 0.2s',
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.backgroundColor = theme.palette.primary.dark;
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.backgroundColor = theme.palette.primary.main;
              }}
            >
              Sign In with SSO
            </button>
          </Box>
        </Box>
      </Box>
    );
  }

  // Main application
  return (
    <Box sx={{ display: 'flex' }}>
      {/* App Bar */}
      <AppBar
        position="fixed"
        sx={{
          width: { sm: `calc(100% - ${state.drawerOpen ? drawerWidth : 0}px)` },
          ml: { sm: state.drawerOpen ? `${drawerWidth}px` : 0 },
          transition: theme.transitions.create(['width', 'margin'], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2 }}
          >
            <MenuIcon />
          </IconButton>
          
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            QSLB Console
          </Typography>

          <IconButton color="inherit" sx={{ mr: 1 }}>
            <Badge badgeContent={state.notifications} color="error">
              <NotificationsIcon />
            </Badge>
          </IconButton>

          <IconButton
            color="inherit"
            onClick={handleUserMenuOpen}
            aria-label="user menu"
          >
            <Avatar sx={{ width: 32, height: 32, bgcolor: 'primary.main' }}>
              {state.user.profile?.name?.charAt(0) || state.user.profile?.email?.charAt(0) || 'U'}
            </Avatar>
          </IconButton>

          <Menu
            anchorEl={state.userMenuAnchor}
            open={Boolean(state.userMenuAnchor)}
            onClose={handleUserMenuClose}
            anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
            transformOrigin={{ vertical: 'top', horizontal: 'right' }}
          >
            <MenuItem onClick={handleUserMenuClose}>
              <ListItemIcon>
                <AccountIcon fontSize="small" />
              </ListItemIcon>
              Profile
            </MenuItem>
            <Divider />
            <MenuItem onClick={handleLogout}>
              <ListItemIcon>
                <LogoutIcon fontSize="small" />
              </ListItemIcon>
              Sign Out
            </MenuItem>
          </Menu>
        </Toolbar>
      </AppBar>

      {/* Navigation Drawer */}
      <Drawer
        variant="temporary"
        open={state.drawerOpen}
        onClose={handleDrawerToggle}
        ModalProps={{ keepMounted: true }}
        sx={{
          '& .MuiDrawer-paper': {
            boxSizing: 'border-box',
            width: drawerWidth,
          },
        }}
      >
        <Toolbar>
          <Typography variant="h6" noWrap component="div" color="primary">
            Navigation
          </Typography>
        </Toolbar>
        <Divider />
        <List>
          {navigationItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            
            return (
              <ListItem key={item.id} disablePadding>
                <ListItemButton
                  selected={isActive}
                  onClick={() => {
                    window.history.pushState({}, '', item.path);
                    handleDrawerToggle();
                  }}
                  sx={{
                    '&.Mui-selected': {
                      backgroundColor: 'primary.light',
                      color: 'primary.contrastText',
                      '&:hover': {
                        backgroundColor: 'primary.main',
                      },
                    },
                  }}
                >
                  <ListItemIcon sx={{ color: isActive ? 'inherit' : 'text.primary' }}>
                    <Icon />
                  </ListItemIcon>
                  <ListItemText primary={item.label} />
                </ListItemButton>
              </ListItem>
            );
          })}
        </List>
      </Drawer>

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: { sm: `calc(100% - ${state.drawerOpen ? drawerWidth : 0}px)` },
          transition: theme.transitions.create('width', {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
        }}
      >
        <Toolbar />
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/callback" element={<div>Processing authentication...</div>} />
          <Route path="/dashboard" element={<Dashboard apiClient={apiClient} />} />
          <Route path="/devices" element={<DeviceManagement apiClient={apiClient} />} />
          <Route path="/protocols" element={<ProtocolVisualization apiClient={apiClient} />} />
          <Route path="/ai-models" element={<AIModelMonitoring apiClient={apiClient} />} />
          <Route path="/threat-intelligence" element={<ThreatIntelligence apiClient={apiClient} />} />
          <Route path="/analytics" element={<AdvancedAnalytics apiClient={apiClient} />} />
          <Route path="/policies" element={<PolicyManagement apiClient={apiClient} />} />
          <Route path="/compliance" element={<ComplianceReporting apiClient={apiClient} />} />
          <Route path="/security" element={<SecurityMonitoring apiClient={apiClient} />} />
          <Route path="/settings" element={<SystemSettings apiClient={apiClient} />} />
          <Route path="/profile" element={<UserProfile user={state.user} authService={authService} />} />
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </Box>

      {/* Error Snackbar */}
      <Snackbar
        open={Boolean(state.error)}
        autoHideDuration={6000}
        onClose={handleErrorClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
      >
        <Alert onClose={handleErrorClose} severity="error" sx={{ width: '100%' }}>
          {state.error}
        </Alert>
      </Snackbar>
    </Box>
  );
}

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <AppContent />
      </Router>
    </ThemeProvider>
  );
}

export default App;