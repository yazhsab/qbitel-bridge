import React, { useEffect, useState, useCallback } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  CardHeader,
  Typography,
  CircularProgress,
  Alert,
  Paper,
  IconButton,
  Button,
  Switch,
  FormControlLabel,
  Chip,
  LinearProgress,
  Avatar,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Tabs,
  Tab,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  Devices as DevicesIcon,
  Security as SecurityIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Timeline as TimelineIcon,
  Psychology as AIIcon,
  Shield as ThreatIcon,
  Analytics as AnalyticsIcon,
  NetworkCheck as NetworkIcon,
  Speed as PerformanceIcon,
  Refresh as RefreshIcon,
  Fullscreen as FullscreenIcon,
  ExpandMore as ExpandMoreIcon,
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
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
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  ComposedChart,
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import { DeviceApiClient } from '../api/devices';
import { DeviceMetrics, DeviceAlert } from '../types/device';

interface EnhancedDashboardProps {
  apiClient: DeviceApiClient;
}

interface DashboardState {
  metrics: DeviceMetrics | null;
  alerts: DeviceAlert[];
  protocolData: any[];
  aiMetrics: any[];
  threatMetrics: any[];
  analyticsData: any[];
  loading: boolean;
  error: string | null;
  realTimeEnabled: boolean;
  webSocketConnected: boolean;
}

interface RealTimeDataManager {
  connect(): void;
  disconnect(): void;
  isConnected(): boolean;
  subscribe(topic: string, callback: (data: any) => void): void;
  unsubscribe(topic: string): void;
}

// Production-ready WebSocket manager
class ProductionWebSocketManager implements RealTimeDataManager {
  private ws: WebSocket | null = null;
  private subscribers: Map<string, ((data: any) => void)[]> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private url: string;

  constructor(url: string) {
    this.url = url;
  }

  connect(): void {
    try {
      this.ws = new WebSocket(this.url);
      
      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        this.startHeartbeat();
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.handleMessage(data);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      this.ws.onclose = (event) => {
        console.log('WebSocket disconnected:', event.reason);
        this.stopHeartbeat();
        this.handleReconnect();
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
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

  subscribe(topic: string, callback: (data: any) => void): void {
    if (!this.subscribers.has(topic)) {
      this.subscribers.set(topic, []);
    }
    this.subscribers.get(topic)!.push(callback);

    // Send subscription message
    if (this.isConnected()) {
      this.send({ type: 'subscribe', topic });
    }
  }

  unsubscribe(topic: string): void {
    this.subscribers.delete(topic);
    
    // Send unsubscription message
    if (this.isConnected()) {
      this.send({ type: 'unsubscribe', topic });
    }
  }

  private send(data: any): void {
    if (this.isConnected()) {
      this.ws!.send(JSON.stringify(data));
    }
  }

  private handleMessage(data: any): void {
    const { topic, payload } = data;
    const callbacks = this.subscribers.get(topic);
    
    if (callbacks) {
      callbacks.forEach(callback => {
        try {
          callback(payload);
        } catch (error) {
          console.error('Error in WebSocket callback:', error);
        }
      });
    }
  }

  private handleReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
      
      setTimeout(() => {
        console.log(`Reconnecting WebSocket (attempt ${this.reconnectAttempts})`);
        this.connect();
      }, delay);
    }
  }

  private startHeartbeat(): void {
    this.heartbeatInterval = setInterval(() => {
      if (this.isConnected()) {
        this.send({ type: 'ping' });
      }
    }, 30000); // 30 seconds
  }

  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];

const EnhancedDashboard: React.FC<EnhancedDashboardProps> = ({ apiClient }) => {
  const [state, setState] = useState<DashboardState>({
    metrics: null,
    alerts: [],
    protocolData: [],
    aiMetrics: [],
    threatMetrics: [],
    analyticsData: [],
    loading: true,
    error: null,
    realTimeEnabled: false,
    webSocketConnected: false,
  });

  const [tabValue, setTabValue] = useState(0);
  const [wsManager] = useState(() => new ProductionWebSocketManager(
    `ws://${window.location.host}/api/ws/dashboard`
  ));

  const loadDashboardData = useCallback(async () => {
    try {
      setState(prev => ({ ...prev, loading: true, error: null }));
      
      // Load core metrics
      const [metricsResponse, alertsData] = await Promise.all([
        apiClient.getDeviceMetrics(),
        apiClient.getDeviceAlerts(undefined, ['warning', 'error', 'critical'], undefined, false),
      ]);

      // Generate mock real-time data for advanced components
      const now = Date.now();
      const mockProtocolData = Array.from({ length: 24 }, (_, i) => ({
        timestamp: new Date(now - (23 - i) * 60 * 60 * 1000),
        protocols: Math.floor(Math.random() * 15) + 5,
        throughput: Math.random() * 1000 + 500,
        anomalies: Math.floor(Math.random() * 3),
      }));

      const mockAIMetrics = [
        { name: 'Protocol Classifier', accuracy: 94.5, status: 'active', throughput: 1250 },
        { name: 'Anomaly Detector', accuracy: 89.2, status: 'active', throughput: 2100 },
        { name: 'Threat Intelligence', accuracy: 87.8, status: 'training', throughput: 150 },
      ];

      const mockThreatMetrics = {
        totalAlerts: 2847,
        newAlerts: 23,
        criticalAlerts: 8,
        blockedAttempts: 1547,
        threatIndicators: 24170,
        activeCampaigns: 12,
      };

      const mockAnalyticsData = Array.from({ length: 12 }, (_, i) => ({
        timestamp: new Date(now - (11 - i) * 60 * 60 * 1000),
        networkThroughput: 500 + Math.random() * 400,
        cpuUtilization: 50 + Math.random() * 30,
        securityIncidents: Math.floor(Math.random() * 5),
        deviceConnections: 1000 + Math.random() * 500,
      }));

      setState(prev => ({
        ...prev,
        metrics: metricsResponse.metrics,
        alerts: alertsData.slice(0, 10),
        protocolData: mockProtocolData,
        aiMetrics: mockAIMetrics,
        threatMetrics: mockThreatMetrics,
        analyticsData: mockAnalyticsData,
        loading: false,
      }));

    } catch (error) {
      console.error('Failed to load dashboard data:', error);
      setState(prev => ({
        ...prev,
        loading: false,
        error: 'Failed to load dashboard data',
      }));
    }
  }, [apiClient]);

  // Initialize dashboard data
  useEffect(() => {
    loadDashboardData();
  }, [loadDashboardData]);

  // Real-time WebSocket management
  useEffect(() => {
    if (state.realTimeEnabled) {
      wsManager.connect();

      // Subscribe to real-time updates
      wsManager.subscribe('metrics_update', (data) => {
        setState(prev => ({ ...prev, metrics: { ...prev.metrics, ...data } }));
      });

      wsManager.subscribe('alerts_update', (data) => {
        setState(prev => ({ ...prev, alerts: [data, ...prev.alerts.slice(0, 9)] }));
      });

      wsManager.subscribe('protocol_update', (data) => {
        setState(prev => ({
          ...prev,
          protocolData: [...prev.protocolData.slice(-23), data],
        }));
      });

      wsManager.subscribe('ai_metrics_update', (data) => {
        setState(prev => ({
          ...prev,
          aiMetrics: prev.aiMetrics.map(m => m.name === data.name ? { ...m, ...data } : m),
        }));
      });

      // Check connection status
      const checkConnection = setInterval(() => {
        setState(prev => ({ ...prev, webSocketConnected: wsManager.isConnected() }));
      }, 1000);

      return () => {
        clearInterval(checkConnection);
        wsManager.disconnect();
      };
    }
  }, [state.realTimeEnabled, wsManager]);

  const toggleRealTime = () => {
    setState(prev => ({ ...prev, realTimeEnabled: !prev.realTimeEnabled }));
  };

  const getAlertIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
      case 'error':
        return <ErrorIcon color="error" />;
      case 'warning':
        return <WarningIcon color="warning" />;
      default:
        return <CheckCircleIcon color="info" />;
    }
  };

  const getAlertColor = (severity: string) => {
    switch (severity) {
      case 'critical':
      case 'error':
        return 'error';
      case 'warning':
        return 'warning';
      default:
        return 'info';
    }
  };

  if (state.loading && !state.metrics) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (state.error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        {state.error}
      </Alert>
    );
  }

  const metrics = state.metrics!;
  const complianceRate = Math.round((metrics.compliant_devices / metrics.total_devices) * 100);
  const healthRate = Math.round((metrics.healthy_devices / metrics.total_devices) * 100);

  return (
    <Box>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1" gutterBottom>
          CronosAI Dashboard
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
            label="Real-time Mode"
          />
          
          <Chip
            icon={<NetworkIcon />}
            label={`WebSocket: ${state.webSocketConnected ? 'Connected' : 'Disconnected'}`}
            color={state.webSocketConnected ? 'success' : 'error'}
            variant="outlined"
          />
          
          <Button
            variant="outlined"
            onClick={loadDashboardData}
            startIcon={<RefreshIcon />}
            disabled={state.loading}
          >
            Refresh
          </Button>
        </Box>
      </Box>

      {state.loading && <LinearProgress sx={{ mb: 2 }} />}

      {/* Key Metrics Overview */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    Total Devices
                  </Typography>
                  <Typography variant="h4">
                    {metrics.total_devices}
                  </Typography>
                </Box>
                <DevicesIcon color="primary" sx={{ fontSize: 40 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    Active Protocols
                  </Typography>
                  <Typography variant="h4" color="info.main">
                    {state.protocolData.length > 0 ? state.protocolData[state.protocolData.length - 1].protocols : 0}
                  </Typography>
                </Box>
                <TimelineIcon color="info" sx={{ fontSize: 40 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    AI Models Active
                  </Typography>
                  <Typography variant="h4" color="success.main">
                    {state.aiMetrics.filter(m => m.status === 'active').length}
                  </Typography>
                </Box>
                <AIIcon color="success" sx={{ fontSize: 40 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box>
                  <Typography color="textSecondary" gutterBottom>
                    Threat Alerts
                  </Typography>
                  <Typography variant="h4" color="error.main">
                    {state.threatMetrics.newAlerts}
                  </Typography>
                </Box>
                <ThreatIcon color="error" sx={{ fontSize: 40 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Advanced Dashboard Tabs */}
      <Card>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={(e, newValue) => setTabValue(newValue)}>
            <Tab label="System Overview" />
            <Tab label="Protocol Analytics" />
            <Tab label="AI Model Status" />
            <Tab label="Security Intelligence" />
          </Tabs>
        </Box>

        {/* System Overview Tab */}
        {tabValue === 0 && (
          <Box sx={{ p: 3 }}>
            <Grid container spacing={3}>
              <Grid item xs={12} lg={8}>
                <Card>
                  <CardHeader title="Real-time Analytics Timeline" />
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <ComposedChart data={state.analyticsData}>
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
                          name="Network Throughput (Mbps)"
                        />
                        <Line
                          yAxisId="right"
                          type="monotone"
                          dataKey="cpuUtilization"
                          stroke="#82ca9d"
                          strokeWidth={2}
                          name="CPU Utilization (%)"
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
                  <CardHeader title="Recent Alerts" />
                  <CardContent>
                    {state.alerts.length === 0 ? (
                      <Typography color="textSecondary">
                        No active alerts
                      </Typography>
                    ) : (
                      <List>
                        {state.alerts.slice(0, 5).map((alert) => (
                          <ListItem key={alert.id} divider>
                            <ListItemIcon>
                              {getAlertIcon(alert.severity)}
                            </ListItemIcon>
                            <ListItemText
                              primary={alert.title}
                              secondary={
                                <Box>
                                  <Typography variant="body2" color="textSecondary">
                                    {alert.description}
                                  </Typography>
                                  <Chip
                                    label={alert.severity}
                                    size="small"
                                    color={getAlertColor(alert.severity) as any}
                                    sx={{ mt: 0.5 }}
                                  />
                                </Box>
                              }
                            />
                          </ListItem>
                        ))}
                      </List>
                    )}
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Box>
        )}

        {/* Protocol Analytics Tab */}
        {tabValue === 1 && (
          <Box sx={{ p: 3 }}>
            <Grid container spacing={3}>
              <Grid item xs={12} lg={8}>
                <Card>
                  <CardHeader 
                    title="Protocol Discovery Timeline"
                    action={
                      state.realTimeEnabled && (
                        <motion.div
                          animate={{ scale: [1, 1.2, 1] }}
                          transition={{ repeat: Infinity, duration: 1 }}
                        >
                          <Chip 
                            icon={<PlayIcon />} 
                            label="LIVE" 
                            color="error" 
                            size="small" 
                          />
                        </motion.div>
                      )
                    }
                  />
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={state.protocolData}>
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
                        <Line 
                          yAxisId="left"
                          type="monotone" 
                          dataKey="protocols" 
                          stroke="#8884d8" 
                          strokeWidth={2}
                          name="Active Protocols"
                        />
                        <Line 
                          yAxisId="right"
                          type="monotone" 
                          dataKey="throughput" 
                          stroke="#82ca9d" 
                          strokeWidth={2}
                          name="Throughput (KB/s)"
                        />
                        <Area 
                          yAxisId="left"
                          type="monotone" 
                          dataKey="anomalies" 
                          stackId="1"
                          stroke="#ff7300" 
                          fill="#ff7300"
                          fillOpacity={0.3}
                          name="Anomalies"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} lg={4}>
                <Card>
                  <CardHeader title="Protocol Statistics" />
                  <CardContent>
                    <Box textAlign="center" mb={2}>
                      <Typography variant="h3" color="primary">
                        {state.protocolData.length > 0 ? 
                          state.protocolData[state.protocolData.length - 1].protocols : 0}
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        Protocols Discovered
                      </Typography>
                    </Box>
                    
                    <Box textAlign="center" mb={2}>
                      <Typography variant="h4" color="success.main">
                        {state.protocolData.length > 0 ? 
                          Math.round(state.protocolData[state.protocolData.length - 1].throughput) : 0}
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        Current Throughput (KB/s)
                      </Typography>
                    </Box>

                    <Box textAlign="center">
                      <Typography variant="h4" color="warning.main">
                        {state.protocolData.reduce((sum, d) => sum + d.anomalies, 0)}
                      </Typography>
                      <Typography variant="body2" color="textSecondary">
                        Total Anomalies (24h)
                      </Typography>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Box>
        )}

        {/* AI Model Status Tab */}
        {tabValue === 2 && (
          <Box sx={{ p: 3 }}>
            <Grid container spacing={3}>
              {state.aiMetrics.map((model, index) => (
                <Grid item xs={12} md={4} key={index}>
                  <Card>
                    <CardHeader
                      avatar={
                        <Avatar sx={{ bgcolor: 'primary.main' }}>
                          <AIIcon />
                        </Avatar>
                      }
                      title={model.name}
                      subheader={`Status: ${model.status}`}
                    />
                    <CardContent>
                      <Box mb={2}>
                        <Typography variant="body2" color="textSecondary">
                          Accuracy
                        </Typography>
                        <Box display="flex" alignItems="center">
                          <Typography variant="h6">
                            {model.accuracy.toFixed(1)}%
                          </Typography>
                          <LinearProgress
                            variant="determinate"
                            value={model.accuracy}
                            sx={{ width: '100%', ml: 2 }}
                            color={model.accuracy > 90 ? 'success' : 'warning'}
                          />
                        </Box>
                      </Box>
                      
                      <Box>
                        <Typography variant="body2" color="textSecondary">
                          Throughput
                        </Typography>
                        <Typography variant="h6">
                          {model.throughput.toLocaleString()} req/s
                        </Typography>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Box>
        )}

        {/* Security Intelligence Tab */}
        {tabValue === 3 && (
          <Box sx={{ p: 3 }}>
            <Grid container spacing={3}>
              <Grid item xs={12} sm={6} md={2}>
                <Card>
                  <CardContent>
                    <Box display="flex" alignItems="center">
                      <Avatar sx={{ bgcolor: 'primary.main', mr: 2 }}>
                        <SecurityIcon />
                      </Avatar>
                      <Box>
                        <Typography color="textSecondary" gutterBottom variant="body2">
                          Total Alerts
                        </Typography>
                        <Typography variant="h5">
                          {state.threatMetrics.totalAlerts.toLocaleString()}
                        </Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} sm={6} md={2}>
                <Card>
                  <CardContent>
                    <Box display="flex" alignItems="center">
                      <Avatar sx={{ bgcolor: 'error.main', mr: 2 }}>
                        <ErrorIcon />
                      </Avatar>
                      <Box>
                        <Typography color="textSecondary" gutterBottom variant="body2">
                          Critical
                        </Typography>
                        <Typography variant="h5">
                          {state.threatMetrics.criticalAlerts}
                        </Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} sm={6} md={2}>
                <Card>
                  <CardContent>
                    <Box display="flex" alignItems="center">
                      <Avatar sx={{ bgcolor: 'info.main', mr: 2 }}>
                        <ThreatIcon />
                      </Avatar>
                      <Box>
                        <Typography color="textSecondary" gutterBottom variant="body2">
                          IOCs
                        </Typography>
                        <Typography variant="h5">
                          {state.threatMetrics.threatIndicators.toLocaleString()}
                        </Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} sm={6} md={2}>
                <Card>
                  <CardContent>
                    <Box display="flex" alignItems="center">
                      <Avatar sx={{ bgcolor: 'secondary.main', mr: 2 }}>
                        <WarningIcon />
                      </Avatar>
                      <Box>
                        <Typography color="textSecondary" gutterBottom variant="body2">
                          Blocked
                        </Typography>
                        <Typography variant="h5">
                          {state.threatMetrics.blockedAttempts}
                        </Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} sm={6} md={2}>
                <Card>
                  <CardContent>
                    <Box display="flex" alignItems="center">
                      <Avatar sx={{ bgcolor: 'warning.main', mr: 2 }}>
                        <AnalyticsIcon />
                      </Avatar>
                      <Box>
                        <Typography color="textSecondary" gutterBottom variant="body2">
                          Campaigns
                        </Typography>
                        <Typography variant="h5">
                          {state.threatMetrics.activeCampaigns}
                        </Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} sm={6} md={2}>
                <Card>
                  <CardContent>
                    <Box display="flex" alignItems="center">
                      <Avatar sx={{ bgcolor: 'success.main', mr: 2 }}>
                        <CheckCircleIcon />
                      </Avatar>
                      <Box>
                        <Typography color="textSecondary" gutterBottom variant="body2">
                          New Alerts
                        </Typography>
                        <Typography variant="h5">
                          {state.threatMetrics.newAlerts}
                        </Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </Box>
        )}
      </Card>
    </Box>
  );
};

export default EnhancedDashboard;