import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { ThemeProvider as MuiThemeProvider } from '@mui/material/styles';
import { CssBaseline } from '@mui/material';
import {
  createEnterpriseTheme,
  ThemeVariant,
  ThemeContextValue,
  availableThemes,
  saveThemePreference,
  getThemePreference,
} from './EnterpriseTheme';

// Create theme context
const ThemeContext = createContext<ThemeContextValue | undefined>(undefined);

// Theme provider props
interface EnterpriseThemeProviderProps {
  children: ReactNode;
  defaultTheme?: ThemeVariant;
}

// Theme provider component
export const EnterpriseThemeProvider: React.FC<EnterpriseThemeProviderProps> = ({
  children,
  defaultTheme = 'light',
}) => {
  const [currentTheme, setCurrentTheme] = useState<ThemeVariant>(() => {
    // Try to get saved theme preference, fallback to default
    try {
      return getThemePreference() || defaultTheme;
    } catch {
      return defaultTheme;
    }
  });

  const theme = createEnterpriseTheme(currentTheme);

  // Handle theme changes
  const handleThemeChange = (variant: ThemeVariant) => {
    setCurrentTheme(variant);
    saveThemePreference(variant);
    
    // Dispatch custom event for theme change
    window.dispatchEvent(new CustomEvent('theme-changed', { 
      detail: { theme: variant } 
    }));
  };

  // Listen for system theme changes
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    
    const handleSystemThemeChange = (e: MediaQueryListEvent) => {
      // Only auto-switch if user hasn't set a preference
      const savedTheme = localStorage.getItem('qbitel-theme');
      if (!savedTheme) {
        const newTheme = e.matches ? 'dark' : 'light';
        setCurrentTheme(newTheme);
      }
    };

    mediaQuery.addEventListener('change', handleSystemThemeChange);
    
    return () => {
      mediaQuery.removeEventListener('change', handleSystemThemeChange);
    };
  }, []);

  const contextValue: ThemeContextValue = {
    currentTheme,
    theme,
    setTheme: handleThemeChange,
    availableThemes,
  };

  return (
    <ThemeContext.Provider value={contextValue}>
      <MuiThemeProvider theme={theme}>
        <CssBaseline />
        {children}
      </MuiThemeProvider>
    </ThemeContext.Provider>
  );
};

// Hook to use theme context
export const useTheme = (): ThemeContextValue => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within an EnterpriseThemeProvider');
  }
  return context;
};

// Theme selector component
export const ThemeSelector: React.FC = () => {
  const { currentTheme, setTheme, availableThemes } = useTheme();
  
  return (
    <div>
      {availableThemes.map((themeOption) => (
        <button
          key={themeOption.key}
          onClick={() => setTheme(themeOption.key)}
          style={{
            padding: '8px 16px',
            margin: '4px',
            border: currentTheme === themeOption.key ? '2px solid #1976d2' : '1px solid #ccc',
            borderRadius: '8px',
            backgroundColor: currentTheme === themeOption.key ? '#e3f2fd' : '#fff',
            cursor: 'pointer',
          }}
        >
          <div style={{ fontWeight: 'bold' }}>{themeOption.name}</div>
          <div style={{ fontSize: '0.875rem', color: '#666' }}>
            {themeOption.description}
          </div>
        </button>
      ))}
    </div>
  );
};

// High-level theme utilities
export const withTheme = <P extends object>(
  Component: React.ComponentType<P>
): React.FC<P> => {
  const ThemedComponent: React.FC<P> = (props) => {
    const { theme } = useTheme();
    return <Component {...props} theme={theme} />;
  };
  
  ThemedComponent.displayName = `withTheme(${Component.displayName || Component.name})`;
  return ThemedComponent;
};

// Theme-aware responsive breakpoint hook
export const useResponsive = () => {
  const { theme } = useTheme();
  const [breakpoint, setBreakpoint] = useState<'xs' | 'sm' | 'md' | 'lg' | 'xl'>('md');
  
  useEffect(() => {
    const updateBreakpoint = () => {
      const width = window.innerWidth;
      const breakpoints = theme.breakpoints.values;
      
      if (width < breakpoints.sm) {
        setBreakpoint('xs');
      } else if (width < breakpoints.md) {
        setBreakpoint('sm');
      } else if (width < breakpoints.lg) {
        setBreakpoint('md');
      } else if (width < breakpoints.xl) {
        setBreakpoint('lg');
      } else {
        setBreakpoint('xl');
      }
    };
    
    updateBreakpoint();
    window.addEventListener('resize', updateBreakpoint);
    
    return () => window.removeEventListener('resize', updateBreakpoint);
  }, [theme.breakpoints.values]);
  
  return {
    breakpoint,
    isMobile: breakpoint === 'xs' || breakpoint === 'sm',
    isTablet: breakpoint === 'md',
    isDesktop: breakpoint === 'lg' || breakpoint === 'xl',
    up: (bp: 'xs' | 'sm' | 'md' | 'lg' | 'xl') => {
      const breakpoints = ['xs', 'sm', 'md', 'lg', 'xl'];
      return breakpoints.indexOf(breakpoint) >= breakpoints.indexOf(bp);
    },
    down: (bp: 'xs' | 'sm' | 'md' | 'lg' | 'xl') => {
      const breakpoints = ['xs', 'sm', 'md', 'lg', 'xl'];
      return breakpoints.indexOf(breakpoint) < breakpoints.indexOf(bp);
    },
  };
};

// Theme customization hook
export const useThemeCustomization = () => {
  const { theme, setTheme, currentTheme } = useTheme();
  
  const applyCustomColors = (colors: {
    primary?: string;
    secondary?: string;
    background?: string;
  }) => {
    // This would typically integrate with a theme customization system
    // For now, we'll just store the preferences
    const customThemeData = {
      variant: currentTheme,
      customColors: colors,
    };
    
    localStorage.setItem('qbitel-custom-theme', JSON.stringify(customThemeData));
  };
  
  const resetTheme = () => {
    localStorage.removeItem('qbitel-custom-theme');
    setTheme('light');
  };
  
  return {
    theme,
    applyCustomColors,
    resetTheme,
    currentVariant: currentTheme,
  };
};

// Performance optimization: Theme-aware memoization
export const useThemeMemo = function<T>(
  factory: () => T,
  deps: React.DependencyList,
): T {
  const { theme } = useTheme();
  
  // eslint-disable-next-line react-hooks/exhaustive-deps
  return React.useMemo(factory, [theme, ...deps]);
};

// Export theme context for advanced usage
export { ThemeContext };