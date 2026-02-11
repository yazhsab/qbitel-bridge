import React, { useState, useEffect } from 'react';
import {
  Box,
  Drawer,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  useMediaQuery,
  useTheme,
  Fab,
  Zoom,
  SpeedDial,
  SpeedDialIcon,
  SpeedDialAction,
  BottomNavigation,
  BottomNavigationAction,
  Paper,
  Collapse,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Avatar,
  Badge,
  Menu,
  MenuItem,
  Divider,
  SwipeableDrawer,
  Card,
  CardContent,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Close as CloseIcon,
  Dashboard as DashboardIcon,
  Devices as DevicesIcon,
  Security as SecurityIcon,
  Analytics as AnalyticsIcon,
  Settings as SettingsIcon,
  Notifications as NotificationsIcon,
  AccountCircle as AccountIcon,
  Add as AddIcon,
  Timeline as TimelineIcon,
  Psychology as AIIcon,
  Shield as ThreatIcon,
  MoreVert as MoreIcon,
  Home as HomeIcon,
  Search as SearchIcon,
} from '@mui/icons-material';
import { useLocation, useNavigate } from 'react-router-dom';
import { User } from '../../types/auth';
import { DeviceApiClient } from '../../api/devices';

interface ResponsiveLayoutProps {
  children: React.ReactNode;
  user: User;
  apiClient: DeviceApiClient;
  onLogout: () => void;
}

interface NavigationItem {
  id: string;
  label: string;
  path: string;
  icon: React.ReactNode;
  badge?: number;
  enterprise?: boolean;
}

const navigationItems: NavigationItem[] = [
  {
    id: 'dashboard',
    label: 'Dashboard',
    path: '/dashboard',
    icon: <DashboardIcon />,
  },
  {
    id: 'devices',
    label: 'Devices',
    path: '/devices',
    icon: <DevicesIcon />,
  },
  {
    id: 'protocols',
    label: 'Protocols',
    path: '/protocols',
    icon: <TimelineIcon />,
    enterprise: true,
  },
  {
    id: 'ai-models',
    label: 'AI Models',
    path: '/ai-models',
    icon: <AIIcon />,
    enterprise: true,
  },
  {
    id: 'threat-intel',
    label: 'Threats',
    path: '/threat-intelligence',
    icon: <ThreatIcon />,
    enterprise: true,
  },
  {
    id: 'analytics',
    label: 'Analytics',
    path: '/analytics',
    icon: <AnalyticsIcon />,
    enterprise: true,
  },
  {
    id: 'security',
    label: 'Security',
    path: '/security',
    icon: <SecurityIcon />,
  },
  {
    id: 'settings',
    label: 'Settings',
    path: '/settings',
    icon: <SettingsIcon />,
  },
];

const ResponsiveLayout: React.FC<ResponsiveLayoutProps> = ({
  children,
  user,
  apiClient,
  onLogout,
}) => {
  const theme = useTheme();
  const location = useLocation();
  const navigate = useNavigate();
  
  // Responsive breakpoints
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const isTablet = useMediaQuery(theme.breakpoints.between('md', 'lg'));
  const isDesktop = useMediaQuery(theme.breakpoints.up('lg'));
  
  // State management
  const [mobileDrawerOpen, setMobileDrawerOpen] = useState(false);
  const [userMenuAnchor, setUserMenuAnchor] = useState<null | HTMLElement>(null);
  const [notificationMenuAnchor, setNotificationMenuAnchor] = useState<null | HTMLElement>(null);
  const [speedDialOpen, setSpeedDialOpen] = useState(false);
  const [bottomNavValue, setBottomNavValue] = useState(0);
  const [notifications, setNotifications] = useState(3); // Mock notification count

  // Update bottom navigation value based on current route
  useEffect(() => {
    const currentPath = location.pathname;
    const currentIndex = navigationItems.findIndex(item => item.path === currentPath);
    if (currentIndex !== -1) {
      setBottomNavValue(currentIndex);
    }
  }, [location.pathname]);

  // Handle navigation
  const handleNavigation = (path: string) => {
    navigate(path);
    if (isMobile) {
      setMobileDrawerOpen(false);
    }
  };

  // Drawer content for mobile/tablet
  const DrawerContent: React.FC = () => (
    <Box sx={{ width: 250 }}>
      <Toolbar>
        <Typography variant="h6" noWrap component="div">
          QbitelAI Console
        </Typography>
        {isMobile && (
          <IconButton
            edge="end"
            onClick={() => setMobileDrawerOpen(false)}
            sx={{ ml: 'auto' }}
          >
            <CloseIcon />
          </IconButton>
        )}
      </Toolbar>
      <Divider />
      <List>
        {navigationItems.map((item) => (
          <ListItem
            key={item.id}
            button
            selected={location.pathname === item.path}
            onClick={() => handleNavigation(item.path)}
            sx={{
              borderRadius: 1,
              mx: 1,
              mb: 0.5,
              '&.Mui-selected': {
                backgroundColor: theme.palette.primary.main,
                color: theme.palette.primary.contrastText,
                '& .MuiListItemIcon-root': {
                  color: theme.palette.primary.contrastText,
                },
                '&:hover': {
                  backgroundColor: theme.palette.primary.dark,
                },
              },
            }}
          >
            <ListItemIcon>
              {item.badge ? (
                <Badge badgeContent={item.badge} color="error">
                  {item.icon}
                </Badge>
              ) : (
                item.icon
              )}
            </ListItemIcon>
            <ListItemText primary={item.label} />
            {item.enterprise && (
              <Typography
                variant="caption"
                sx={{
                  backgroundColor: theme.palette.warning.main,
                  color: theme.palette.warning.contrastText,
                  px: 1,
                  py: 0.25,
                  borderRadius: 1,
                  fontSize: '0.6rem',
                  fontWeight: 'bold',
                }}
              >
                PRO
              </Typography>
            )}
          </ListItem>
        ))}
      </List>
    </Box>
  );

  // Speed dial actions for mobile
  const speedDialActions = [
    {
      icon: <AddIcon />,
      name: 'Add Device',
      action: () => navigate('/devices?action=add'),
    },
    {
      icon: <SearchIcon />,
      name: 'Search',
      action: () => {/* Implement search */},
    },
    {
      icon: <NotificationsIcon />,
      name: 'Notifications',
      action: () => {/* Open notifications */},
    },
  ];

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      {/* Mobile/Tablet App Bar */}
      {(isMobile || isTablet) && (
        <AppBar position="fixed" sx={{ zIndex: theme.zIndex.drawer + 1 }}>
          <Toolbar>
            <IconButton
              color="inherit"
              aria-label="open drawer"
              edge="start"
              onClick={() => setMobileDrawerOpen(true)}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>
            
            <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
              QbitelAI
            </Typography>

            <IconButton
              color="inherit"
              onClick={(e) => setNotificationMenuAnchor(e.currentTarget)}
            >
              <Badge badgeContent={notifications} color="error">
                <NotificationsIcon />
              </Badge>
            </IconButton>

            <IconButton
              color="inherit"
              onClick={(e) => setUserMenuAnchor(e.currentTarget)}
            >
              <Avatar
                sx={{
                  width: 32,
                  height: 32,
                  bgcolor: theme.palette.secondary.main,
                }}
              >
                {user.profile?.name?.charAt(0) || user.profile?.email?.charAt(0) || 'U'}
              </Avatar>
            </IconButton>
          </Toolbar>
        </AppBar>
      )}

      {/* Desktop Sidebar */}
      {isDesktop && (
        <Drawer
          variant="permanent"
          sx={{
            width: 280,
            flexShrink: 0,
            '& .MuiDrawer-paper': {
              width: 280,
              boxSizing: 'border-box',
              backgroundColor: theme.palette.background.default,
            },
          }}
        >
          <DrawerContent />
        </Drawer>
      )}

      {/* Mobile/Tablet Drawer */}
      {(isMobile || isTablet) && (
        <SwipeableDrawer
          anchor="left"
          open={mobileDrawerOpen}
          onClose={() => setMobileDrawerOpen(false)}
          onOpen={() => setMobileDrawerOpen(true)}
          ModalProps={{ keepMounted: true }}
        >
          <DrawerContent />
        </SwipeableDrawer>
      )}

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          ml: isDesktop ? '280px' : 0,
          pt: isMobile || isTablet ? 8 : 0,
          pb: isMobile ? 7 : 0, // Space for bottom navigation
        }}
      >
        <Box
          sx={{
            flexGrow: 1,
            overflow: 'auto',
            p: theme.spacing(isMobile ? 2 : 3),
          }}
        >
          {children}
        </Box>
      </Box>

      {/* Mobile Bottom Navigation */}
      {isMobile && (
        <Paper
          sx={{ position: 'fixed', bottom: 0, left: 0, right: 0, zIndex: 1000 }}
          elevation={3}
        >
          <BottomNavigation
            value={bottomNavValue}
            onChange={(event, newValue) => {
              setBottomNavValue(newValue);
              if (navigationItems[newValue]) {
                handleNavigation(navigationItems[newValue].path);
              }
            }}
            showLabels
          >
            {navigationItems.slice(0, 5).map((item, index) => (
              <BottomNavigationAction
                key={item.id}
                label={item.label}
                icon={
                  item.badge ? (
                    <Badge badgeContent={item.badge} color="error">
                      {item.icon}
                    </Badge>
                  ) : (
                    item.icon
                  )
                }
              />
            ))}
          </BottomNavigation>
        </Paper>
      )}

      {/* Mobile Speed Dial */}
      {isMobile && (
        <SpeedDial
          ariaLabel="Quick Actions"
          sx={{
            position: 'fixed',
            bottom: 80,
            right: 16,
          }}
          icon={<SpeedDialIcon />}
          onClose={() => setSpeedDialOpen(false)}
          onOpen={() => setSpeedDialOpen(true)}
          open={speedDialOpen}
        >
          {speedDialActions.map((action) => (
            <SpeedDialAction
              key={action.name}
              icon={action.icon}
              tooltipTitle={action.name}
              onClick={() => {
                action.action();
                setSpeedDialOpen(false);
              }}
            />
          ))}
        </SpeedDial>
      )}

      {/* User Menu */}
      <Menu
        anchorEl={userMenuAnchor}
        open={Boolean(userMenuAnchor)}
        onClose={() => setUserMenuAnchor(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
      >
        <MenuItem onClick={() => { setUserMenuAnchor(null); navigate('/profile'); }}>
          <ListItemIcon>
            <AccountIcon fontSize="small" />
          </ListItemIcon>
          Profile
        </MenuItem>
        <MenuItem onClick={() => { setUserMenuAnchor(null); navigate('/settings'); }}>
          <ListItemIcon>
            <SettingsIcon fontSize="small" />
          </ListItemIcon>
          Settings
        </MenuItem>
        <Divider />
        <MenuItem onClick={() => { setUserMenuAnchor(null); onLogout(); }}>
          <ListItemIcon>
            <CloseIcon fontSize="small" />
          </ListItemIcon>
          Sign Out
        </MenuItem>
      </Menu>

      {/* Notification Menu */}
      <Menu
        anchorEl={notificationMenuAnchor}
        open={Boolean(notificationMenuAnchor)}
        onClose={() => setNotificationMenuAnchor(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
        PaperProps={{
          sx: { width: 320, maxHeight: 400 },
        }}
      >
        <Box sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Notifications
          </Typography>
          <List dense>
            <ListItem>
              <ListItemIcon>
                <SecurityIcon color="warning" />
              </ListItemIcon>
              <ListItemText
                primary="Security Alert"
                secondary="Suspicious activity detected on device #1247"
              />
            </ListItem>
            <ListItem>
              <ListItemIcon>
                <AIIcon color="info" />
              </ListItemIcon>
              <ListItemText
                primary="Model Training Complete"
                secondary="Protocol classifier v2.1 training finished"
              />
            </ListItem>
            <ListItem>
              <ListItemIcon>
                <DevicesIcon color="success" />
              </ListItemIcon>
              <ListItemText
                primary="Device Enrolled"
                secondary="New IoT sensor added to network"
              />
            </ListItem>
          </List>
        </Box>
      </Menu>
    </Box>
  );
};

// Responsive Grid Component
export const ResponsiveGrid: React.FC<{
  children: React.ReactNode;
  spacing?: number;
}> = ({ children, spacing = 3 }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  return (
    <Box
      sx={{
        display: 'grid',
        gap: spacing,
        gridTemplateColumns: {
          xs: '1fr',
          sm: 'repeat(auto-fit, minmax(300px, 1fr))',
          md: 'repeat(auto-fit, minmax(350px, 1fr))',
          lg: 'repeat(auto-fit, minmax(400px, 1fr))',
        },
      }}
    >
      {children}
    </Box>
  );
};

// Responsive Card Component
export const ResponsiveCard: React.FC<{
  children: React.ReactNode;
  title?: string;
  action?: React.ReactNode;
  mobile?: boolean;
}> = ({ children, title, action, mobile }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  return (
    <Card
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        ...(mobile && {
          borderRadius: 0,
          boxShadow: 'none',
          borderBottom: `1px solid ${theme.palette.divider}`,
        }),
      }}
    >
      {title && (
        <Box
          sx={{
            p: 2,
            pb: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <Typography
            variant={isMobile ? "h6" : "h5"}
            component="h2"
            sx={{ fontWeight: 600 }}
          >
            {title}
          </Typography>
          {action}
        </Box>
      )}
      <CardContent sx={{ flexGrow: 1, pt: title ? 1 : 2 }}>
        {children}
      </CardContent>
    </Card>
  );
};

// Responsive Table Container
export const ResponsiveTable: React.FC<{
  children: React.ReactNode;
}> = ({ children }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  if (isMobile) {
    // On mobile, convert table to card-based layout
    return (
      <Box sx={{ width: '100%' }}>
        {children}
      </Box>
    );
  }

  return (
    <Box sx={{ width: '100%', overflow: 'auto' }}>
      {children}
    </Box>
  );
};

export default ResponsiveLayout;