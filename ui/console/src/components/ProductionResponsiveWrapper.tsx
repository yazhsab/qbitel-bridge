import React, { useState, useEffect, useMemo } from 'react';
import {
  Box,
  useMediaQuery,
  useTheme,
  ThemeProvider,
  createTheme,
  CssBaseline,
  GlobalStyles,
  Fade,
  Slide,
  Collapse,
} from '@mui/material';
import { ResponsiveLayout } from './responsive/ResponsiveLayout';
import { User } from '../types/auth';
import { DeviceApiClient } from '../api/devices';
import { config, PerformanceMonitor } from '../config/production';

interface ProductionResponsiveWrapperProps {
  children: React.ReactNode;
  user: User;
  apiClient: DeviceApiClient;
  onLogout: () => void;
}

// Enhanced responsive theme with production optimizations
const createResponsiveTheme = (prefersDarkMode: boolean, isMobile: boolean) => {
  const baseTheme = createTheme({
    palette: {
      mode: prefersDarkMode ? 'dark' : 'light',
      primary: {
        main: '#1976d2',
        light: '#42a5f5',
        dark: '#115293',
        contrastText: '#ffffff',
      },
      secondary: {
        main: '#dc004e',
        light: '#ff5983',
        dark: '#9a0036',
        contrastText: '#ffffff',
      },
      background: {
        default: prefersDarkMode ? '#121212' : '#f5f5f5',
        paper: prefersDarkMode ? '#1e1e1e' : '#ffffff',
      },
      text: {
        primary: prefersDarkMode ? '#ffffff' : '#212121',
        secondary: prefersDarkMode ? '#b3b3b3' : '#757575',
      },
    },
    typography: {
      fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
      fontSize: isMobile ? 14 : 16,
      h1: {
        fontSize: isMobile ? '2rem' : '2.5rem',
        fontWeight: 700,
        lineHeight: 1.2,
      },
      h2: {
        fontSize: isMobile ? '1.75rem' : '2rem',
        fontWeight: 600,
        lineHeight: 1.3,
      },
      h3: {
        fontSize: isMobile ? '1.5rem' : '1.75rem',
        fontWeight: 600,
        lineHeight: 1.4,
      },
      h4: {
        fontSize: isMobile ? '1.25rem' : '1.5rem',
        fontWeight: 600,
        lineHeight: 1.4,
      },
      h5: {
        fontSize: isMobile ? '1.1rem' : '1.25rem',
        fontWeight: 600,
        lineHeight: 1.5,
      },
      h6: {
        fontSize: isMobile ? '1rem' : '1.1rem',
        fontWeight: 600,
        lineHeight: 1.5,
      },
      body1: {
        fontSize: isMobile ? '0.875rem' : '1rem',
        lineHeight: 1.6,
      },
      body2: {
        fontSize: isMobile ? '0.8125rem' : '0.875rem',
        lineHeight: 1.5,
      },
      caption: {
        fontSize: isMobile ? '0.75rem' : '0.8125rem',
        lineHeight: 1.4,
      },
    },
    shape: {
      borderRadius: isMobile ? 8 : 12,
    },
    spacing: isMobile ? 4 : 8,
    components: {
      MuiCssBaseline: {
        styleOverrides: {
          body: {
            scrollbarGutter: 'stable',
          },
          '*::-webkit-scrollbar': {
            width: isMobile ? '4px' : '8px',
            height: isMobile ? '4px' : '8px',
          },
          '*::-webkit-scrollbar-track': {
            background: prefersDarkMode ? '#2c2c2c' : '#f1f1f1',
            borderRadius: '4px',
          },
          '*::-webkit-scrollbar-thumb': {
            backgroundColor: prefersDarkMode ? '#555' : '#c1c1c1',
            borderRadius: '4px',
            '&:hover': {
              backgroundColor: prefersDarkMode ? '#777' : '#a8a8a8',
            },
          },
        },
      },
      MuiCard: {
        styleOverrides: {
          root: {
            boxShadow: prefersDarkMode
              ? '0px 2px 4px rgba(0, 0, 0, 0.5)'
              : '0px 2px 4px rgba(0, 0, 0, 0.1)',
            transition: 'box-shadow 0.3s ease-in-out, transform 0.2s ease-in-out',
            '&:hover': {
              boxShadow: prefersDarkMode
                ? '0px 4px 8px rgba(0, 0, 0, 0.7)'
                : '0px 4px 8px rgba(0, 0, 0, 0.15)',
              transform: 'translateY(-2px)',
            },
          },
        },
      },
      MuiButton: {
        styleOverrides: {
          root: {
            textTransform: 'none',
            fontWeight: 600,
            borderRadius: isMobile ? '8px' : '12px',
            padding: isMobile ? '8px 16px' : '12px 24px',
            transition: 'all 0.2s ease-in-out',
          },
          containedPrimary: {
            background: 'linear-gradient(135deg, #1976d2 0%, #1565c0 100%)',
            '&:hover': {
              background: 'linear-gradient(135deg, #1565c0 0%, #0d47a1 100%)',
              transform: 'translateY(-1px)',
              boxShadow: '0 4px 12px rgba(25, 118, 210, 0.4)',
            },
          },
        },
      },
      MuiChip: {
        styleOverrides: {
          root: {
            fontSize: isMobile ? '0.75rem' : '0.8125rem',
            height: isMobile ? 24 : 28,
          },
        },
      },
      MuiTableCell: {
        styleOverrides: {
          root: {
            padding: isMobile ? '8px' : '16px',
            fontSize: isMobile ? '0.8125rem' : '0.875rem',
          },
        },
      },
      MuiDialog: {
        styleOverrides: {
          paper: {
            margin: isMobile ? 8 : 32,
            maxHeight: isMobile ? 'calc(100% - 16px)' : 'calc(100% - 64px)',
          },
        },
      },
    },
  });

  return createTheme({
    ...baseTheme,
    breakpoints: {
      values: {
        xs: 0,
        sm: 600,
        md: 960,
        lg: 1280,
        xl: 1920,
      },
    },
  });
};

// Global styles for production optimization
const globalStyles = (
  <GlobalStyles
    styles={{
      html: {
        // Enable smooth scrolling
        scrollBehavior: 'smooth',
        // Improve text rendering
        textRendering: 'optimizeLegibility',
        '-webkit-font-smoothing': 'antialiased',
        '-moz-osx-font-smoothing': 'grayscale',
      },
      body: {
        // Prevent horizontal scroll
        overflowX: 'hidden',
        // Optimize repaints
        transform: 'translateZ(0)',
      },
      // Optimize images
      img: {
        maxWidth: '100%',
        height: 'auto',
        // Improve image loading performance
        loading: 'lazy',
        // Prevent layout shift
        aspectRatio: 'attr(width) / attr(height)',
      },
      // Performance optimizations for animations
      '*': {
        willChange: 'auto',
      },
      '.animate-gpu': {
        transform: 'translateZ(0)',
        backfaceVisibility: 'hidden',
        perspective: 1000,
      },
      // Accessibility improvements
      '.sr-only': {
        position: 'absolute',
        width: '1px',
        height: '1px',
        padding: 0,
        margin: '-1px',
        overflow: 'hidden',
        clip: 'rect(0, 0, 0, 0)',
        whiteSpace: 'nowrap',
        border: 0,
      },
      // Touch optimization
      '.touch-action-none': {
        touchAction: 'none',
      },
      '.touch-action-pan-x': {
        touchAction: 'pan-x',
      },
      '.touch-action-pan-y': {
        touchAction: 'pan-y',
      },
      // Loading states
      '.loading-skeleton': {
        background: 'linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%)',
        backgroundSize: '200% 100%',
        animation: 'loading 1.5s infinite',
      },
      '@keyframes loading': {
        '0%': {
          backgroundPosition: '200% 0',
        },
        '100%': {
          backgroundPosition: '-200% 0',
        },
      },
    }}
  />
);

const ProductionResponsiveWrapper: React.FC<ProductionResponsiveWrapperProps> = ({
  children,
  user,
  apiClient,
  onLogout,
}) => {
  const [prefersDarkMode, setPrefersDarkMode] = useState(false);
  const [mounted, setMounted] = useState(false);

  // Detect system theme preference
  const systemPrefersDark = useMediaQuery('(prefers-color-scheme: dark)');
  const baseTheme = useTheme();
  const isMobile = useMediaQuery(baseTheme.breakpoints.down('md'));
  const isTablet = useMediaQuery(baseTheme.breakpoints.between('md', 'lg'));

  // Performance monitoring
  useEffect(() => {
    const endTimer = PerformanceMonitor.startTimer('ResponsiveWrapper.mount');
    setMounted(true);
    return endTimer;
  }, []);

  // Theme preference management
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme-preference');
    if (savedTheme) {
      setPrefersDarkMode(savedTheme === 'dark');
    } else {
      setPrefersDarkMode(systemPrefersDark);
    }
  }, [systemPrefersDark]);

  // Memoize theme to prevent unnecessary re-renders
  const theme = useMemo(() => {
    return createResponsiveTheme(prefersDarkMode, isMobile);
  }, [prefersDarkMode, isMobile]);

  // Handle theme toggle
  const toggleTheme = () => {
    const newPreference = !prefersDarkMode;
    setPrefersDarkMode(newPreference);
    localStorage.setItem('theme-preference', newPreference ? 'dark' : 'light');
  };

  // Performance optimization: prevent flash of unstyled content
  if (!mounted) {
    return null;
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      {globalStyles}
      
      <Fade in={mounted} timeout={300}>
        <Box
          sx={{
            minHeight: '100vh',
            display: 'flex',
            flexDirection: 'column',
            bgcolor: 'background.default',
          }}
          className="animate-gpu"
        >
          <ResponsiveLayout
            user={user}
            apiClient={apiClient}
            onLogout={onLogout}
          >
            <Slide
              direction="up"
              in={mounted}
              timeout={400}
              mountOnEnter
              unmountOnExit
            >
              <Box
                sx={{
                  flex: 1,
                  display: 'flex',
                  flexDirection: 'column',
                  p: {
                    xs: 1,
                    sm: 2,
                    md: 3,
                  },
                  transition: 'padding 0.2s ease-in-out',
                }}
              >
                {children}
              </Box>
            </Slide>
          </ResponsiveLayout>

          {/* Theme Toggle Button - Mobile */}
          {isMobile && (
            <Box
              sx={{
                position: 'fixed',
                bottom: 80,
                left: 16,
                zIndex: 1000,
              }}
            >
              <Collapse in={mounted}>
                <Box
                  component="button"
                  onClick={toggleTheme}
                  sx={{
                    width: 56,
                    height: 56,
                    borderRadius: '50%',
                    border: 'none',
                    bgcolor: 'primary.main',
                    color: 'primary.contrastText',
                    boxShadow: 3,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    transition: 'all 0.3s ease-in-out',
                    '&:hover': {
                      transform: 'scale(1.1)',
                      boxShadow: 6,
                    },
                    '&:active': {
                      transform: 'scale(0.95)',
                    },
                  }}
                  aria-label="Toggle theme"
                >
                  {prefersDarkMode ? '☀️' : '🌙'}
                </Box>
              </Collapse>
            </Box>
          )}

          {/* Performance Indicator - Development Only */}
          {process.env.NODE_ENV === 'development' && (
            <Box
              sx={{
                position: 'fixed',
                top: 8,
                right: 8,
                zIndex: 10000,
                bgcolor: 'rgba(0, 0, 0, 0.8)',
                color: 'white',
                px: 1,
                py: 0.5,
                borderRadius: 1,
                fontSize: '0.75rem',
                fontFamily: 'monospace',
                pointerEvents: 'none',
              }}
            >
              {isMobile ? 'Mobile' : isTablet ? 'Tablet' : 'Desktop'} |{' '}
              {prefersDarkMode ? 'Dark' : 'Light'}
            </Box>
          )}
        </Box>
      </Fade>
    </ThemeProvider>
  );
};

export default ProductionResponsiveWrapper;

// Hook for accessing responsive utilities
export const useResponsive = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const isTablet = useMediaQuery(theme.breakpoints.between('md', 'lg'));
  const isDesktop = useMediaQuery(theme.breakpoints.up('lg'));
  const prefersDarkMode = useMediaQuery('(prefers-color-scheme: dark)');

  return {
    isMobile,
    isTablet,
    isDesktop,
    prefersDarkMode,
    breakpoints: theme.breakpoints,
    spacing: theme.spacing,
  };
};

// Responsive component wrapper
export const ResponsiveComponent: React.FC<{
  children: React.ReactNode;
  mobile?: React.ReactNode;
  tablet?: React.ReactNode;
  desktop?: React.ReactNode;
}> = ({ children, mobile, tablet, desktop }) => {
  const { isMobile, isTablet } = useResponsive();

  if (isMobile && mobile) {
    return <>{mobile}</>;
  }

  if (isTablet && tablet) {
    return <>{tablet}</>;
  }

  if (desktop) {
    return <>{desktop}</>;
  }

  return <>{children}</>;
};