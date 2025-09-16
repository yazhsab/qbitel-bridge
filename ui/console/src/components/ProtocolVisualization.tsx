import React, { useEffect, useState, useCallback } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  CardHeader,
  Typography,
  Switch,
  FormControlLabel,
  Button,
  Chip,
  Alert,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Tooltip,
  LinearProgress,
  Avatar,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Divider,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  Refresh as RefreshIcon,
  Timeline as TimelineIcon,
  NetworkCheck as NetworkIcon,
  Security as SecurityIcon,
  Speed as SpeedIcon,
  BugReport as BugIcon,
  Analytics as AnalyticsIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
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
  ScatterChart,
  Scatter,
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import useWebSocket from 'react-use-websocket';
import { DeviceApiClient } from '../api/devices';

interface ProtocolVisualizationProps {
  apiClient: DeviceApiClient;
}

interface ProtocolData {
  id: string;
  name: string;
  type: 'TCP' | 'UDP' | 'HTTP' | 'HTTPS' | 'MQTT' | 'CoAP' | 'Custom';
  status: 'active' | 'discovering' | 'classified' | 'anomaly';
  confidence: number;
  packets: number;
  bytes: number;
  lastSeen: Date;
  source: string;
  destination: string;
  port: number;
  features: Record<string, number>;
  patterns: string[];
}

interface NetworkFlow {
  id: string;
  source: string;
  destination: string;
  protocol: string;
  bandwidth: number;
  latency: number;
  status: 'normal' | 'suspicious' | 'blocked';
  timestamp: Date;
}

interface AnalysisMetrics {
  totalProtocols: number;
  activeFlows: number;
  throughput: number;
  accuracy: number;
  anomalies: number;
  classification_rate: number;
  discovery_rate: number;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];

const ProtocolVisualization: React.FC<ProtocolVisualizationProps> = ({ apiClient }) => {
  const [protocols, setProtocols] = useState<ProtocolData[]>([]);
  const [flows, setFlows] = useState<NetworkFlow[]>([]);
  const [metrics, setMetrics] = useState<AnalysisMetrics>({
    totalProtocols: 0,
    activeFlows: 0,
    throughput: 0,
    accuracy: 95.2,
    anomalies: 0,
    classification_rate: 0,
    discovery_rate: 0,
  });
  const [isLive, setIsLive] = useState(false);
  const [timeRange, setTimeRange] = useState('1h');
  const [selectedProtocol, setSelectedProtocol] = useState<string | null>(null);
  const [historicalData, setHistoricalData] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // WebSocket connection for real-time updates
  const socketUrl = `ws://${window.location.host}/api/ws/protocols`;
  const { lastMessage, connectionStatus, sendMessage } = useWebSocket(
    socketUrl,
    {
      shouldReconnect: (closeEvent) => true,
      reconnectAttempts: 10,
      reconnectInterval: 3000,
    }
  );

  // Handle real-time WebSocket messages
  useEffect(() => {
    if (lastMessage !== null) {
      try {
        const data = JSON.parse(lastMessage.data);
        
        switch (data.type) {
          case 'protocol_update':
            setProtocols(prev => {
              const existing = prev.find(p => p.id === data.protocol.id);
              if (existing) {
                return prev.map(p => p.id === data.protocol.id ? { ...p, ...data.protocol } : p);
              }
              return [...prev, data.protocol];
            });
            break;
          
          case 'flow_update':
            setFlows(prev => {
              const updated = [...prev.slice(-100), data.flow]; // Keep last 100 flows
              return updated;
            });
            break;
          
          case 'metrics_update':
            setMetrics(data.metrics);
            break;
          
          case 'historical_data':
            setHistoricalData(data.data);
            break;
        }
      } catch (err) {
        console.error('Error parsing WebSocket message:', err);
      }
    }
  }, [lastMessage]);

  const loadInitialData = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Load initial protocol data
      // This would typically come from the API
      const mockProtocols: ProtocolData[] = [
        {
          id: '1',
          name: 'HTTP/1.1',
          type: 'HTTP',
          status: 'active',
          confidence: 98.5,
          packets: 1250,
          bytes: 524288,
          lastSeen: new Date(),
          source: '192.168.1.10',
          destination: '10.0.0.1',
          port: 80,
          features: { header_count: 8, method_diversity: 0.6, content_type_variety: 0.4 },
          patterns: ['GET /', 'POST /api', 'User-Agent: Mozilla']
        },
        {
          id: '2',
          name: 'MQTT v3.1.1',
          type: 'MQTT',
          status: 'discovering',
          confidence: 75.2,
          packets: 450,
          bytes: 89432,
          lastSeen: new Date(Date.now() - 5000),
          source: '192.168.1.15',
          destination: '10.0.0.5',
          port: 1883,
          features: { message_frequency: 0.8, topic_diversity: 0.3, qos_distribution: 0.7 },
          patterns: ['CONNECT', 'PUBLISH sensor/temp', 'SUBSCRIBE alerts/#']
        }
      ];
      
      setProtocols(mockProtocols);
      
      // Generate mock historical data
      const now = Date.now();
      const mockHistorical = Array.from({ length: 60 }, (_, i) => ({
        timestamp: new Date(now - (59 - i) * 60000),
        protocols: Math.floor(Math.random() * 10) + 5,
        throughput: Math.random() * 1000 + 500,
        accuracy: 90 + Math.random() * 10,
        anomalies: Math.floor(Math.random() * 5),
      }));
      
      setHistoricalData(mockHistorical);
      
    } catch (err) {
      setError('Failed to load protocol data');
      console.error('Error loading data:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadInitialData();
  }, [loadInitialData]);

  const toggleLiveMode = () => {
    setIsLive(!isLive);
    if (!isLive) {
      sendMessage(JSON.stringify({ action: 'start_live_feed' }));
    } else {
      sendMessage(JSON.stringify({ action: 'stop_live_feed' }));
    }
  };

  const refreshData = () => {
    loadInitialData();
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'success';
      case 'discovering': return 'warning';
      case 'classified': return 'info';
      case 'anomaly': return 'error';
      default: return 'default';
    }
  };

  const getProtocolTypeColor = (type: string) => {
    switch (type) {
      case 'HTTP': return '#4CAF50';
      case 'HTTPS': return '#2196F3';
      case 'MQTT': return '#FF9800';
      case 'TCP': return '#9C27B0';
      case 'UDP': return '#F44336';
      case 'CoAP': return '#607D8B';
      default: return '#795548';
    }
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1" gutterBottom>
          Real-time Protocol Visualization
        </Typography>
        
        <Box display="flex" alignItems="center" gap={2}>
          <FormControlLabel
            control={
              <Switch
                checked={isLive}
                onChange={toggleLiveMode}
                color="primary"
              />
            }
            label="Live Mode"
          />
          
          <Button
            variant="outlined"
            onClick={refreshData}
            startIcon={<RefreshIcon />}
            disabled={loading}
          >
            Refresh
          </Button>
          
          <Chip
            icon={<NetworkIcon />}
            label={`Connected: ${connectionStatus === 'Open' ? 'Yes' : 'No'}`}
            color={connectionStatus === 'Open' ? 'success' : 'error'}
            variant="outlined"
          />
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {loading && <LinearProgress sx={{ mb: 2 }} />}

      {/* Metrics Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={2}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <Avatar sx={{ bgcolor: 'primary.main', mr: 2 }}>
                  <TimelineIcon />
                </Avatar>
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    Protocols
                  </Typography>
                  <Typography variant="h5">
                    {metrics.totalProtocols}
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
                  <NetworkIcon />
                </Avatar>
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    Active Flows
                  </Typography>
                  <Typography variant="h5">
                    {metrics.activeFlows}
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
                  <SpeedIcon />
                </Avatar>
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    Throughput
                  </Typography>
                  <Typography variant="h5">
                    {(metrics.throughput / 1024).toFixed(1)}K
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
                  <AnalyticsIcon />
                </Avatar>
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    Accuracy
                  </Typography>
                  <Typography variant="h5">
                    {metrics.accuracy.toFixed(1)}%
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
                  <BugIcon />
                </Avatar>
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    Anomalies
                  </Typography>
                  <Typography variant="h5">
                    {metrics.anomalies}
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
                  <SecurityIcon />
                </Avatar>
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    Discovery Rate
                  </Typography>
                  <Typography variant="h5">
                    {metrics.discovery_rate.toFixed(1)}/s
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* Real-time Protocol Timeline */}
        <Grid item xs={12} lg={8}>
          <Card>
            <CardHeader 
              title="Protocol Discovery Timeline"
              action={
                <Box display="flex" alignItems="center" gap={1}>
                  {isLive && (
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
                  )}
                </Box>
              }
            />
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={historicalData}>
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
                    name="Protocols"
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

        {/* Protocol Distribution */}
        <Grid item xs={12} lg={4}>
          <Card sx={{ height: 400 }}>
            <CardHeader title="Protocol Distribution" />
            <CardContent>
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={protocols.map(p => ({ name: p.type, value: p.packets }))}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  >
                    {protocols.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={getProtocolTypeColor(entry.type)} />
                    ))}
                  </Pie>
                  <RechartsTooltip />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Live Protocol Table */}
        <Grid item xs={12}>
          <Card>
            <CardHeader 
              title="Discovered Protocols"
              subheader={`${protocols.length} protocols currently tracked`}
            />
            <CardContent sx={{ p: 0 }}>
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Protocol</TableCell>
                      <TableCell>Type</TableCell>
                      <TableCell>Status</TableCell>
                      <TableCell align="right">Confidence</TableCell>
                      <TableCell align="right">Packets</TableCell>
                      <TableCell align="right">Bytes</TableCell>
                      <TableCell>Source → Destination</TableCell>
                      <TableCell>Last Seen</TableCell>
                      <TableCell>Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    <AnimatePresence>
                      {protocols.map((protocol) => (
                        <motion.tr
                          key={protocol.id}
                          initial={{ opacity: 0, y: -10 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: 10 }}
                          transition={{ duration: 0.2 }}
                          component={TableRow}
                          hover
                          onClick={() => setSelectedProtocol(
                            selectedProtocol === protocol.id ? null : protocol.id
                          )}
                          sx={{ cursor: 'pointer' }}
                        >
                          <TableCell>
                            <Box display="flex" alignItems="center">
                              <Avatar 
                                sx={{ 
                                  width: 24, 
                                  height: 24, 
                                  mr: 1,
                                  bgcolor: getProtocolTypeColor(protocol.type)
                                }}
                              >
                                <Typography variant="caption">
                                  {protocol.type.charAt(0)}
                                </Typography>
                              </Avatar>
                              {protocol.name}
                            </Box>
                          </TableCell>
                          <TableCell>
                            <Chip 
                              label={protocol.type} 
                              size="small"
                              sx={{ bgcolor: getProtocolTypeColor(protocol.type), color: 'white' }}
                            />
                          </TableCell>
                          <TableCell>
                            <Chip 
                              label={protocol.status} 
                              size="small"
                              color={getStatusColor(protocol.status) as any}
                            />
                          </TableCell>
                          <TableCell align="right">
                            <Box display="flex" alignItems="center" justifyContent="flex-end">
                              <Typography variant="body2">
                                {protocol.confidence.toFixed(1)}%
                              </Typography>
                              <LinearProgress
                                variant="determinate"
                                value={protocol.confidence}
                                sx={{ width: 50, ml: 1 }}
                                color={protocol.confidence > 90 ? 'success' : protocol.confidence > 70 ? 'warning' : 'error'}
                              />
                            </Box>
                          </TableCell>
                          <TableCell align="right">{protocol.packets.toLocaleString()}</TableCell>
                          <TableCell align="right">{(protocol.bytes / 1024).toFixed(1)}KB</TableCell>
                          <TableCell>
                            <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                              {protocol.source}:{protocol.port} → {protocol.destination}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2" color="textSecondary">
                              {protocol.lastSeen.toLocaleTimeString()}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Tooltip title={selectedProtocol === protocol.id ? "Hide Details" : "Show Details"}>
                              <IconButton size="small">
                                {selectedProtocol === protocol.id ? <VisibilityOffIcon /> : <VisibilityIcon />}
                              </IconButton>
                            </Tooltip>
                          </TableCell>
                        </motion.tr>
                      ))}
                    </AnimatePresence>
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Protocol Details Panel */}
        {selectedProtocol && (
          <Grid item xs={12}>
            <AnimatePresence>
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.3 }}
              >
                {(() => {
                  const protocol = protocols.find(p => p.id === selectedProtocol);
                  if (!protocol) return null;
                  
                  return (
                    <Card>
                      <CardHeader 
                        title={`${protocol.name} Details`}
                        subheader={`Protocol analysis and patterns`}
                      />
                      <CardContent>
                        <Grid container spacing={3}>
                          <Grid item xs={12} md={6}>
                            <Paper sx={{ p: 2 }}>
                              <Typography variant="h6" gutterBottom>
                                Protocol Features
                              </Typography>
                              <List dense>
                                {Object.entries(protocol.features).map(([key, value]) => (
                                  <ListItem key={key}>
                                    <ListItemText
                                      primary={key.replace(/_/g, ' ').toUpperCase()}
                                      secondary={
                                        <LinearProgress
                                          variant="determinate"
                                          value={value * 100}
                                          sx={{ width: '100%' }}
                                        />
                                      }
                                    />
                                    <Typography variant="body2" color="textSecondary" sx={{ ml: 2 }}>
                                      {(value * 100).toFixed(1)}%
                                    </Typography>
                                  </ListItem>
                                ))}
                              </List>
                            </Paper>
                          </Grid>
                          
                          <Grid item xs={12} md={6}>
                            <Paper sx={{ p: 2 }}>
                              <Typography variant="h6" gutterBottom>
                                Detected Patterns
                              </Typography>
                              <List dense>
                                {protocol.patterns.map((pattern, index) => (
                                  <ListItem key={index}>
                                    <ListItemAvatar>
                                      <Avatar sx={{ width: 24, height: 24, bgcolor: 'primary.main' }}>
                                        <Typography variant="caption">
                                          {index + 1}
                                        </Typography>
                                      </Avatar>
                                    </ListItemAvatar>
                                    <ListItemText
                                      primary={
                                        <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                                          {pattern}
                                        </Typography>
                                      }
                                    />
                                  </ListItem>
                                ))}
                              </List>
                            </Paper>
                          </Grid>
                        </Grid>
                      </CardContent>
                    </Card>
                  );
                })()}
              </motion.div>
            </AnimatePresence>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default ProtocolVisualization;