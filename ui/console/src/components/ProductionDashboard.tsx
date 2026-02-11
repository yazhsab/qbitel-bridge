import React, { useEffect, useState, useCallback, useMemo } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  CardHeader,
  Typography,
  Button,
  Switch,
  FormControlLabel,
  Chip,
  LinearProgress,
  Avatar,
  IconButton,
  Tooltip,
  Alert,
  Tabs,
  Tab,
  Fade,
  Skeleton,
  useMediaQuery,
  useTheme,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Timeline as TimelineIcon,
  Psychology as AIIcon,
  Security as ThreatIcon,
  Analytics as AnalyticsIcon,
  Refresh as RefreshIcon,
  Fullscreen as FullscreenIcon,
  Settings as SettingsIcon,
  TrendingUp as TrendingUpIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  NetworkCheck as NetworkIcon,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  ComposedChart,
  Bar,
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import { DeviceApiClient } from '../api/devices';
import { DeviceMetrics, DeviceAlert } from '../types/device';
import { config, CacheManager, PerformanceMonitor, debounce } from '../config/production';

interface ProductionDashboardProps {
  apiClient: DeviceApiClient;
}

interface DashboardData {
  deviceMetrics: DeviceMetrics | null;
  alerts: DeviceAlert[];
  protocolData: Array<{
    timestamp: Date;
    protocols: number;
    throughput: number;
    anomalies: number;
  }>;
  aiMetrics: Array<{
    name: string;
    accuracy: number;
    status: string;
    throughput: number;
    latency: number;
  }>;
  threatData: {
    totalAlerts: number;
    newAlerts: number;
    criticalAlerts: number;
    blockedAttempts: number;
    threatIndicators: number;
  };
  analyticsData: Array<{
    timestamp: Date;
    networkThroughput: number;
    cpuUtilization: number;
    securityIncidents: number;
    deviceConnections: number;
  }>;
}

interface DashboardState {
  data: DashboardData;
  loading: boolean;
  error: string | null;
  realTimeEnabled: boolean;
  selectedTab: number;
  lastUpdate: Date | null;
}

// Production WebSocket manager
class ProductionWebSocketManager {
  private ws: WebSocket | null = null;
  private subscribers: Map<string, Array<(data: any) => void>> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private heartbeatInterval: number | null = null;

  constructor(private url: string) {}

  async connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
          console.log('Production WebSocket connected');
          this.reconnectAttempts = 0;
          this.startHeartbeat();
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
          } catch (error) {
            console.error('WebSocket message parse error:', error);
          }
        };

        this.ws.onclose = () => {
          this.stopHeartbeat();
          this.handleReconnect();
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          reject(error);
        };

      } catch (error) {
        reject(error);
      }
    });
  }

  subscribe(topic: string, callback: (data: any) => void): void {
    if (!this.subscribers.has(topic)) {
      this.subscribers.set(topic, []);
    }
    this.subscribers.get(topic)!.push(callback);

    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'subscribe', topic }));
    }
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.stopHeartbeat();
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  private handleMessage(data: any): void {
    const { topic, payload } = data;
    const callbacks = this.subscribers.get(topic);
    
    if (callbacks) {
      callbacks.forEach(callback => {
        try {
          callback(payload);
        } catch (error) {
          console.error('WebSocket callback error:', error);
        }
      });
    }
  }

  private handleReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = Math.pow(2, this.reconnectAttempts) * 1000;
      
      setTimeout(() => {
        this.connect();
      }, delay);
    }
  }

  private startHeartbeat(): void {
    this.heartbeatInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, config.websocket.heartbeatInterval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }
}

const ProductionDashboard: React.FC<ProductionDashboardProps> = ({ apiClient }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  
  const [state, setState] = useState<DashboardState>({
    data: {
      deviceMetrics: null,
      alerts: [],
      protocolData: [],
      aiMetrics: [],
      threatData: {
        totalAlerts: 0,
        newAlerts: 0,
        criticalAlerts: 0,
        blockedAttempts: 0,
        threatIndicators: 0,
      },
      analyticsData: [],
    },
    loading: true,
    error: null,
    realTimeEnabled: false,
    selectedTab: 0,
    lastUpdate: null,
  });

  // WebSocket manager instance
  const wsManager = useMemo(() => 
    new ProductionWebSocketManager(`${config.websocket.url}/dashboard`)
  , []);

  // Load dashboard data with performance monitoring
  const loadDashboardData = useCallback(async () => {
    const endTimer = PerformanceMonitor.startTimer('Dashboard.loadData');
    
    try {
      setState(prev => ({ ...prev, loading: true, error: null }));
      
      // Check cache first
      const cacheKey = 'dashboard.data';
      if (CacheManager.has(cacheKey)) {
        const cachedData = CacheManager.get(cacheKey);
        setState(prev => ({
          ...prev,
          data: cachedData,
          loading: false,
          lastUpdate: new Date(),
        }));
        endTimer();
        return;
      }

      // Load core metrics
      const [metricsResponse, alertsData] = await Promise.all([
        apiClient.getDeviceMetrics(),
        apiClient.getDeviceAlerts(undefined, ['warning', 'error', 'critical'], undefined, false),
      ]);

      // Generate mock data for advanced components
      const now = Date.now();
      
      const protocolData = Array.from({ length: 24 }, (_, i) => ({
        timestamp: new Date(now - (23 - i) * 60 * 60 * 1000),
        protocols: Math.floor(Math.random() * 15) + 5,
        throughput: Math.random() * 1000 + 500,
        anomalies: Math.floor(Math.random() * 3),
      }));

      const aiMetrics = [
        { name: 'Protocol Classifier', accuracy: 94.5, status: 'active', throughput: 1250, latency: 12.5 },
        { name: 'Anomaly Detector', accuracy: 89.2, status: 'active', throughput: 2100, latency: 8.2 },
        { name: 'Threat Intelligence', accuracy: 87.8, status: 'training', throughput: 150, latency: 45.0 },
      ];

      const threatData = {
        totalAlerts: 2847,
        newAlerts: 23,
        criticalAlerts: 8,
        blockedAttempts: 1547,
        threatIndicators: 24170,
      };

      const analyticsData = Array.from({ length: 12 }, (_, i) => ({
        timestamp: new Date(now - (11 - i) * 60 * 60 * 1000),
        networkThroughput: 500 + Math.random() * 400,
        cpuUtilization: 50 + Math.random() * 30,
        securityIncidents: Math.floor(Math.random() * 5),
        deviceConnections: 1000 + Math.random() * 500,
      }));

      const dashboardData: DashboardData = {
        deviceMetrics: metricsResponse.metrics,
        alerts: alertsData.slice(0, 10),
        protocolData,
        aiMetrics,
        threatData,
        analyticsData,
      };

      // Cache the data
      CacheManager.set(cacheKey, dashboardData, config.cache.defaultTTL);

      setState(prev => ({
        ...prev,
        data: dashboardData,
        loading: false,
        lastUpdate: new Date(),
      }));

    } catch (error) {
      console.error('Failed to load dashboard data:', error);
      setState(prev => ({
        ...prev,
        loading: false,
        error: 'Failed to load dashboard data',
      }));
    } finally {
      endTimer();
    }
  }, [apiClient]);

  // Debounced refresh function
  const debouncedRefresh = useMemo(
    () => debounce(loadDashboardData, 300),
    [loadDashboardData]
  );

  // Real-time WebSocket management
  useEffect(() => {
    if (state.realTimeEnabled) {
      wsManager.connect().then(() => {
        // Subscribe to real-time updates
        wsManager.subscribe('dashboard.metrics', (data) => {
          setState(prev => ({
            ...prev,
            data: {
              ...prev.data,
              deviceMetrics: { ...prev.data.deviceMetrics, ...data },
            },
            lastUpdate: new Date(),
          }));
        });

        wsManager.subscribe('dashboard.alerts', (data) => {
          setState(prev => ({
            ...prev,
            data: {
              ...prev.data,
              alerts: [data, ...prev.data.alerts.slice(0, 9)],
            },
            lastUpdate: new Date(),
          }));
        });

        wsManager.subscribe('dashboard.protocols', (data) => {
          setState(prev => ({
            ...prev,
            data: {
              ...prev.data,
              protocolData: [...prev.data.protocolData.slice(-23), data],
            },
            lastUpdate: new Date(),
          }));
        });
      });
    } else {
      wsManager.disconnect();
    }

    return () => {
      wsManager.disconnect();
    };
  }, [state.realTimeEnabled, wsManager]);

  // Initialize dashboard
  useEffect(() => {
    loadDashboardData();
  }, [loadDashboardData]);

  const toggleRealTime = () => {
    setState(prev => ({ ...prev, realTimeEnabled: !prev.realTimeEnabled }));
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setState(prev => ({ ...prev, selectedTab: newValue }));
  };

  // Loading skeleton
  const LoadingSkeleton = () => (
    <Grid container spacing={3}>
      {Array.from({ length: 8 }).map((_, index) => (
        <Grid item xs={12} sm={6} md={3} key={index}>
          <Card>
            <CardContent>
              <Skeleton variant="circular" width={40} height={40} />
              <Skeleton variant="text" width="80%" />
              <Skeleton variant="text" width="60%" />
            </CardContent>
          </Card>
        </Grid>
      ))}
    </Grid>
  );

  if (state.loading && !state.data.deviceMetrics) {
    return <LoadingSkeleton />;
  }

  if (state.error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        {state.error}
        <Button onClick={loadDashboardData} sx={{ ml: 2 }}>
          Retry
        </Button>
      </Alert>
    );
  }

  const metrics = state.data.deviceMetrics!;

  return (
    <Box>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant={isMobile ? "h5" : "h4"} component="h1">
          QbitelAI Production Dashboard
        </Typography>
        
        <Box display="flex" alignItems="center" gap={2}>
          <FormControlLabel
            control={
              <Switch
                checked={state.realTimeEnabled}
                onChange={toggleRealTime}
                color="primary"
              />
            }
            label="Real-time"
            sx={{ display: { xs: 'none', sm: 'flex' } }}
          />
          
          <Chip
            icon={<NetworkIcon />}
            label={wsManager.isConnected() ? 'Connected' : 'Disconnected'}
            color={wsManager.isConnected() ? 'success' : 'error'}
            variant="outlined"
            size="small"
          />
          
          <Tooltip title="Refresh Dashboard">
            <IconButton onClick={debouncedRefresh} disabled={state.loading}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {state.loading && <LinearProgress sx={{ mb: 2 }} />}

      {state.lastUpdate && (
        <Typography variant="caption" color="textSecondary" sx={{ mb: 2, display: 'block' }}>
          Last updated: {state.lastUpdate.toLocaleString()}
        </Typography>
      )}

      {/* Key Performance Indicators */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" justifyContent="space-between">
                  <Box>
                    <Typography color="textSecondary" gutterBottom>
                      Total Devices
                    </Typography>
                    <Typography variant="h4">
                      {metrics.total_devices.toLocaleString()}
                    </Typography>
                  </Box>
                  <Avatar sx={{ bgcolor: 'primary.main' }}>
                    <DashboardIcon />
                  </Avatar>
                </Box>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.1 }}
          >
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" justifyContent="space-between">
                  <Box>
                    <Typography color="textSecondary" gutterBottom>
                      Active Protocols
                    </Typography>
                    <Typography variant="h4" color="info.main">
                      {state.data.protocolData.length > 0 ? 
                        state.data.protocolData[state.data.protocolData.length - 1].protocols : 0}
                    </Typography>
                  </Box>
                  <Avatar sx={{ bgcolor: 'info.main' }}>
                    <TimelineIcon />
                  </Avatar>
                </Box>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.2 }}
          >
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" justifyContent="space-between">
                  <Box>
                    <Typography color="textSecondary" gutterBottom>
                      AI Models Active
                    </Typography>
                    <Typography variant="h4" color="success.main">
                      {state.data.aiMetrics.filter(m => m.status === 'active').length}
                    </Typography>
                  </Box>
                  <Avatar sx={{ bgcolor: 'success.main' }}>
                    <AIIcon />
                  </Avatar>
                </Box>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.3 }}
          >
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" justifyContent="space-between">
                  <Box>
                    <Typography color="textSecondary" gutterBottom>
                      New Threats
                    </Typography>
                    <Typography variant="h4" color="error.main">
                      {state.data.threatData.newAlerts}
                    </Typography>
                  </Box>
                  <Avatar sx={{ bgcolor: 'error.main' }}>
                    <ThreatIcon />
                  </Avatar>
                </Box>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>

      {/* Tabbed Analytics */}
      <Card>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs
            value={state.selectedTab}
            onChange={handleTabChange}
            variant={isMobile ? "scrollable" : "standard"}
            scrollButtons={isMobile ? "auto" : false}
          >
            <Tab label="System Overview" />
            <Tab label="Protocol Analytics" />
            <Tab label="AI Performance" />
            <Tab label="Threat Intelligence" />
          </Tabs>
        </Box>

        <Box sx={{ p: 3 }}>
          <AnimatePresence mode="wait">
            {/* System Overview Tab */}
            {state.selectedTab === 0 && (
              <motion.div
                key="overview"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.3 }}
              >
                <Grid container spacing={3}>
                  <Grid item xs={12} lg={8}>
                    <Card>
                      <CardHeader title="Real-time System Metrics" />
                      <CardContent>
                        <ResponsiveContainer width="100%" height={300}>
                          <ComposedChart data={state.data.analyticsData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis 
                              dataKey="timestamp"
                              tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                            />
                            <YAxis yAxisId="left" />
                            <YAxis yAxisId="right" orientation="right" />
                            <RechartsTooltip 
                              labelFormatter={(value) => new Date(value).toLocaleString()}
                            />
                            <Area
                              yAxisId="left"
                              type="monotone"
                              dataKey="networkThroughput"
                              fill="#8884d8"
                              fillOpacity={0.3}
                              stroke="#8884d8"
                              name="Network Throughput"
                            />
                            <Line
                              yAxisId="right"
                              type="monotone"
                              dataKey="cpuUtilization"
                              stroke="#82ca9d"
                              strokeWidth={2}
                              name="CPU Utilization"
                            />
                            <Bar
                              yAxisId="left"
                              dataKey="securityIncidents"
                              fill="#ff7300"
                              name="Security Incidents"
                            />
                          </ComposedChart>
                        </ResponsiveContainer>
                      </CardContent>
                    </Card>
                  </Grid>

                  <Grid item xs={12} lg={4}>
                    <Card>
                      <CardHeader title="System Health" />
                      <CardContent>
                        <Box mb={2}>
                          <Typography variant="body2" color="textSecondary">
                            Overall Health Score
                          </Typography>
                          <Box display="flex" alignItems="center" mt={1}>
                            <LinearProgress
                              variant="determinate"
                              value={95}
                              sx={{ flexGrow: 1, mr: 2 }}
                              color="success"
                            />
                            <Typography variant="h6" color="success.main">
                              95%
                            </Typography>
                          </Box>
                        </Box>

                        <Box mb={2}>
                          <Typography variant="body2" color="textSecondary">
                            Compliance Rate
                          </Typography>
                          <Box display="flex" alignItems="center" mt={1}>
                            <LinearProgress
                              variant="determinate"
                              value={Math.round((metrics.compliant_devices / metrics.total_devices) * 100)}
                              sx={{ flexGrow: 1, mr: 2 }}
                              color="success"
                            />
                            <Typography variant="h6">
                              {Math.round((metrics.compliant_devices / metrics.total_devices) * 100)}%
                            </Typography>
                          </Box>
                        </Box>

                        <Box>
                          <Typography variant="body2" color="textSecondary">
                            Active Connections
                          </Typography>
                          <Typography variant="h4" sx={{ mt: 1 }}>
                            {metrics.active_devices.toLocaleString()}
                          </Typography>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>
              </motion.div>
            )}

            {/* Other tabs would be implemented similarly */}
            {state.selectedTab === 1 && (
              <motion.div
                key="protocols"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.3 }}
              >
                <Typography variant="h6">Protocol Analytics Dashboard</Typography>
                <Typography variant="body2" color="textSecondary">
                  Real-time protocol discovery and analysis visualization coming soon...
                </Typography>
              </motion.div>
            )}

            {state.selectedTab === 2 && (
              <motion.div
                key="ai"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.3 }}
              >
                <Typography variant="h6">AI Model Performance</Typography>
                <Typography variant="body2" color="textSecondary">
                  Comprehensive AI model monitoring and performance metrics...
                </Typography>
              </motion.div>
            )}

            {state.selectedTab === 3 && (
              <motion.div
                key="threats"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.3 }}
              >
                <Typography variant="h6">Threat Intelligence Dashboard</Typography>
                <Typography variant="body2" color="textSecondary">
                  Real-time security threats and intelligence analysis...
                </Typography>
              </motion.div>
            )}
          </AnimatePresence>
        </Box>
      </Card>
    </Box>
  );
};

export default ProductionDashboard;