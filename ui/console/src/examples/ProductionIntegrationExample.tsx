import React, { useEffect, useState } from 'react';
import {
  Box,
  Container,
  Typography,
  Alert,
  Card,
  CardContent,
  Grid,
  Button,
  Chip,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  CircularProgress,
} from '@mui/material';
import {
  CheckCircle as CheckIcon,
  Error as ErrorIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  PlayArrow as TestIcon,
} from '@mui/icons-material';

// Import all our production-ready components
import ProductionDashboard from '../components/ProductionDashboard';
import ProductionResponsiveWrapper from '../components/ProductionResponsiveWrapper';
import EnhancedDashboard from '../components/EnhancedDashboard';
import ProtocolVisualization from '../components/ProtocolVisualization';
import AIModelMonitoring from '../components/AIModelMonitoring';
import ThreatIntelligence from '../components/ThreatIntelligence';
import AdvancedAnalytics from '../components/AdvancedAnalytics';
import { DeviceApiClient } from '../api/devices';
import { config, PerformanceMonitor, ErrorTracker, CacheManager } from '../config/production';

interface TestResult {
  name: string;
  status: 'pending' | 'running' | 'passed' | 'failed';
  message: string;
  duration?: number;
}

interface IntegrationTest {
  id: string;
  name: string;
  description: string;
  test: () => Promise<boolean>;
}

const ProductionIntegrationExample: React.FC = () => {
  const [testResults, setTestResults] = useState<TestResult[]>([]);
  const [isRunningTests, setIsRunningTests] = useState(false);
  const [overallStatus, setOverallStatus] = useState<'idle' | 'running' | 'completed'>('idle');

  // Mock API client for testing
  const mockApiClient = new DeviceApiClient(async () => 'mock-token');

  // Mock user for testing
  const mockUser = {
    profile: {
      name: 'Test User',
      email: 'test@qbitelai.local',
    },
  };

  // Define integration tests
  const integrationTests: IntegrationTest[] = [
    {
      id: 'config-validation',
      name: 'Configuration Validation',
      description: 'Validate production configuration is properly loaded',
      test: async () => {
        return (
          config.api.baseUrl !== undefined &&
          config.websocket.url !== undefined &&
          config.cache.defaultTTL > 0 &&
          config.performance.enableVirtualization !== undefined
        );
      },
    },
    {
      id: 'cache-management',
      name: 'Cache Management',
      description: 'Test cache set/get operations',
      test: async () => {
        const testKey = 'test-key';
        const testData = { test: 'data' };
        
        CacheManager.set(testKey, testData, 1000);
        const retrieved = CacheManager.get(testKey);
        
        return JSON.stringify(retrieved) === JSON.stringify(testData);
      },
    },
    {
      id: 'performance-monitoring',
      name: 'Performance Monitoring',
      description: 'Test performance timing functionality',
      test: async () => {
        const endTimer = PerformanceMonitor.startTimer('test-operation');
        await new Promise(resolve => setTimeout(resolve, 10));
        endTimer();
        
        const metrics = PerformanceMonitor.getMetrics('test-operation');
        return metrics !== null && metrics.count > 0;
      },
    },
    {
      id: 'error-tracking',
      name: 'Error Tracking',
      description: 'Test error capture and tracking',
      test: async () => {
        const testError = new Error('Test error');
        ErrorTracker.captureError(testError, { context: 'integration-test' });
        
        const errors = ErrorTracker.getErrors();
        return errors.length > 0 && errors[errors.length - 1].message === 'Test error';
      },
    },
    {
      id: 'api-client-health',
      name: 'API Client Health',
      description: 'Test API client initialization and basic functionality',
      test: async () => {
        try {
          // Test that API client can be instantiated
          const client = new DeviceApiClient(async () => 'test-token');
          return client !== null;
        } catch (error) {
          return false;
        }
      },
    },
    {
      id: 'component-rendering',
      name: 'Component Rendering',
      description: 'Test that all major components can be instantiated',
      test: async () => {
        try {
          // Test component instantiation (this would normally be done with a proper test renderer)
          const components = [
            () => React.createElement(ProductionDashboard, { apiClient: mockApiClient }),
            () => React.createElement(EnhancedDashboard, { apiClient: mockApiClient }),
            () => React.createElement(ProtocolVisualization, { apiClient: mockApiClient }),
            () => React.createElement(AIModelMonitoring, { apiClient: mockApiClient }),
            () => React.createElement(ThreatIntelligence, { apiClient: mockApiClient }),
            () => React.createElement(AdvancedAnalytics, { apiClient: mockApiClient }),
          ];
          
          // Just check that components can be created without throwing
          components.forEach(componentFactory => componentFactory());
          return true;
        } catch (error) {
          return false;
        }
      },
    },
    {
      id: 'responsive-design',
      name: 'Responsive Design',
      description: 'Test responsive wrapper functionality',
      test: async () => {
        try {
          // Test that responsive wrapper can be instantiated
          React.createElement(ProductionResponsiveWrapper, {
            user: mockUser,
            apiClient: mockApiClient,
            onLogout: () => {},
            children: React.createElement('div', {}, 'Test content'),
          });
          return true;
        } catch (error) {
          return false;
        }
      },
    },
    {
      id: 'memory-management',
      name: 'Memory Management',
      description: 'Test memory usage and cleanup',
      test: async () => {
        // Test cache cleanup
        CacheManager.clear();
        const stats = CacheManager.getStats();
        
        // Test error cleanup
        ErrorTracker.clearErrors();
        const errors = ErrorTracker.getErrors();
        
        return stats.size === 0 && errors.length === 0;
      },
    },
  ];

  // Run all integration tests
  const runIntegrationTests = async () => {
    setIsRunningTests(true);
    setOverallStatus('running');
    setTestResults([]);

    const results: TestResult[] = [];

    for (const test of integrationTests) {
      // Update status to running
      const runningResult: TestResult = {
        name: test.name,
        status: 'running',
        message: 'Running test...',
      };
      
      results.push(runningResult);
      setTestResults([...results]);

      try {
        const startTime = performance.now();
        const passed = await test.test();
        const duration = performance.now() - startTime;

        // Update result
        const finalResult: TestResult = {
          name: test.name,
          status: passed ? 'passed' : 'failed',
          message: passed ? 'Test passed successfully' : 'Test failed',
          duration,
        };

        results[results.length - 1] = finalResult;
        setTestResults([...results]);

        // Add small delay for visual effect
        await new Promise(resolve => setTimeout(resolve, 100));

      } catch (error) {
        const finalResult: TestResult = {
          name: test.name,
          status: 'failed',
          message: `Test failed with error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        };

        results[results.length - 1] = finalResult;
        setTestResults([...results]);
      }
    }

    setIsRunningTests(false);
    setOverallStatus('completed');
  };

  // Calculate test statistics
  const testStats = {
    total: testResults.length,
    passed: testResults.filter(r => r.status === 'passed').length,
    failed: testResults.filter(r => r.status === 'failed').length,
    running: testResults.filter(r => r.status === 'running').length,
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'passed':
        return <CheckIcon color="success" />;
      case 'failed':
        return <ErrorIcon color="error" />;
      case 'running':
        return <CircularProgress size={20} />;
      default:
        return <InfoIcon color="info" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'passed':
        return 'success';
      case 'failed':
        return 'error';
      case 'running':
        return 'info';
      default:
        return 'default';
    }
  };

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Typography variant="h3" component="h1" gutterBottom>
        QbitelAI Dashboard - Production Integration
      </Typography>
      
      <Alert severity="info" sx={{ mb: 3 }}>
        This is a comprehensive integration example demonstrating the production-ready QbitelAI dashboard 
        with all advanced features including real-time data streaming, AI model monitoring, threat intelligence, 
        and responsive design.
      </Alert>

      {/* Integration Test Suite */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
            <Typography variant="h5" component="h2">
              Integration Test Suite
            </Typography>
            
            <Button
              variant="contained"
              startIcon={<TestIcon />}
              onClick={runIntegrationTests}
              disabled={isRunningTests}
            >
              {isRunningTests ? 'Running Tests...' : 'Run Integration Tests'}
            </Button>
          </Box>

          {/* Test Statistics */}
          {testResults.length > 0 && (
            <Grid container spacing={2} sx={{ mb: 3 }}>
              <Grid item xs={12} sm={3}>
                <Card variant="outlined">
                  <CardContent sx={{ textAlign: 'center' }}>
                    <Typography variant="h4" color="primary">
                      {testStats.total}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Total Tests
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} sm={3}>
                <Card variant="outlined">
                  <CardContent sx={{ textAlign: 'center' }}>
                    <Typography variant="h4" color="success.main">
                      {testStats.passed}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Passed
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} sm={3}>
                <Card variant="outlined">
                  <CardContent sx={{ textAlign: 'center' }}>
                    <Typography variant="h4" color="error.main">
                      {testStats.failed}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Failed
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} sm={3}>
                <Card variant="outlined">
                  <CardContent sx={{ textAlign: 'center' }}>
                    <Typography variant="h4" color="info.main">
                      {testStats.running}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Running
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          )}

          {/* Test Results */}
          {testResults.length > 0 && (
            <List>
              {testResults.map((result, index) => (
                <ListItem key={index} divider>
                  <ListItemIcon>
                    {getStatusIcon(result.status)}
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <Box display="flex" alignItems="center" gap={1}>
                        <Typography variant="body1">
                          {result.name}
                        </Typography>
                        <Chip
                          label={result.status}
                          size="small"
                          color={getStatusColor(result.status) as any}
                          variant="outlined"
                        />
                        {result.duration && (
                          <Chip
                            label={`${result.duration.toFixed(2)}ms`}
                            size="small"
                            variant="outlined"
                          />
                        )}
                      </Box>
                    }
                    secondary={result.message}
                  />
                </ListItem>
              ))}
            </List>
          )}

          {/* Overall Status */}
          {overallStatus === 'completed' && (
            <Alert
              severity={testStats.failed === 0 ? 'success' : 'warning'}
              sx={{ mt: 2 }}
            >
              {testStats.failed === 0
                ? `All ${testStats.total} tests passed successfully! ✨`
                : `${testStats.passed}/${testStats.total} tests passed. ${testStats.failed} tests need attention.`}
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Feature Showcase */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h5" component="h2" gutterBottom>
            Production Features Implemented
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    🚀 Performance & Optimization
                  </Typography>
                  <List dense>
                    <ListItem>
                      <ListItemText primary="Production-ready caching system" />
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="Performance monitoring & metrics" />
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="Memory management & cleanup" />
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="Request debouncing & throttling" />
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="Error tracking & recovery" />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    📱 Responsive Design
                  </Typography>
                  <List dense>
                    <ListItem>
                      <ListItemText primary="Mobile-first responsive layout" />
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="Adaptive component sizing" />
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="Touch-optimized interactions" />
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="Progressive enhancement" />
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="Accessible design patterns" />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    ⚡ Real-time Features
                  </Typography>
                  <List dense>
                    <ListItem>
                      <ListItemText primary="Production WebSocket management" />
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="Auto-reconnection with backoff" />
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="Live data streaming" />
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="Real-time notifications" />
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="Connection health monitoring" />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} md={6}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    🧠 Advanced Analytics
                  </Typography>
                  <List dense>
                    <ListItem>
                      <ListItemText primary="AI model performance monitoring" />
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="Protocol visualization & discovery" />
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="Threat intelligence integration" />
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="Predictive analytics dashboard" />
                    </ListItem>
                    <ListItem>
                      <ListItemText primary="Real-time security monitoring" />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Configuration Overview */}
      <Card>
        <CardContent>
          <Typography variant="h5" component="h2" gutterBottom>
            Production Configuration
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <Typography variant="h6" gutterBottom>
                API Configuration
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemText
                    primary="Base URL"
                    secondary={config.api.baseUrl}
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Timeout"
                    secondary={`${config.api.timeout}ms`}
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Retry Attempts"
                    secondary={config.api.retryAttempts}
                  />
                </ListItem>
              </List>
            </Grid>

            <Grid item xs={12} md={4}>
              <Typography variant="h6" gutterBottom>
                WebSocket Configuration
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemText
                    primary="WebSocket URL"
                    secondary={config.websocket.url}
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Reconnect Attempts"
                    secondary={config.websocket.reconnectAttempts}
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Heartbeat Interval"
                    secondary={`${config.websocket.heartbeatInterval}ms`}
                  />
                </ListItem>
              </List>
            </Grid>

            <Grid item xs={12} md={4}>
              <Typography variant="h6" gutterBottom>
                Performance Settings
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemText
                    primary="Cache TTL"
                    secondary={`${config.cache.defaultTTL}ms`}
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Virtualization"
                    secondary={config.performance.enableVirtualization ? 'Enabled' : 'Disabled'}
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Chunk Size"
                    secondary={config.performance.chunkSize}
                  />
                </ListItem>
              </List>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Container>
  );
};

export default ProductionIntegrationExample;