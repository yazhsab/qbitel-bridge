import React, { useEffect, useState, useCallback } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  CardHeader,
  Typography,
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
  LinearProgress,
  Avatar,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Tooltip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
} from '@mui/material';
import {
  Psychology as AIIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Speed as PerformanceIcon,
  Memory as MemoryIcon,
  Storage as StorageIcon,
  Timeline as TimelineIcon,
  Warning as WarningIcon,
  CheckCircle as SuccessIcon,
  Error as ErrorIcon,
  Refresh as RefreshIcon,
  Settings as SettingsIcon,
  ExpandMore as ExpandMoreIcon,
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  Stop as StopIcon,
  Visibility as ViewIcon,
  Download as DownloadIcon,
  CloudUpload as DeployIcon,
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
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ComposedChart,
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import { DeviceApiClient } from '../api/devices';

interface AIModelMonitoringProps {
  apiClient: DeviceApiClient;
}

interface ModelMetrics {
  id: string;
  name: string;
  version: string;
  type: 'classification' | 'regression' | 'clustering' | 'anomaly_detection' | 'nlp' | 'computer_vision';
  status: 'training' | 'deployed' | 'idle' | 'error' | 'updating';
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  auc: number;
  latency: number; // ms
  throughput: number; // requests/sec
  memoryUsage: number; // MB
  cpuUsage: number; // %
  gpuUsage: number; // %
  diskUsage: number; // MB
  lastUpdated: Date;
  trainingTime: number; // hours
  inferenceCount: number;
  errorRate: number;
  datasetSize: number;
  features: number;
  parameters: number;
}

interface ModelPerformanceHistory {
  timestamp: Date;
  accuracy: number;
  latency: number;
  throughput: number;
  memoryUsage: number;
  cpuUsage: number;
  errorRate: number;
}

interface ModelAlert {
  id: string;
  modelId: string;
  type: 'performance_degradation' | 'memory_leak' | 'high_latency' | 'accuracy_drop' | 'resource_exhaustion';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  timestamp: Date;
  acknowledged: boolean;
}

interface DeploymentStatus {
  modelId: string;
  environment: 'development' | 'staging' | 'production';
  replicas: number;
  healthyReplicas: number;
  version: string;
  lastDeployed: Date;
  rolloutStatus: 'in_progress' | 'completed' | 'failed' | 'paused';
}

const AIModelMonitoring: React.FC<AIModelMonitoringProps> = ({ apiClient }) => {
  const [models, setModels] = useState<ModelMetrics[]>([]);
  const [performanceHistory, setPerformanceHistory] = useState<ModelPerformanceHistory[]>([]);
  const [alerts, setAlerts] = useState<ModelAlert[]>([]);
  const [deployments, setDeployments] = useState<DeploymentStatus[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(30); // seconds
  const [settingsOpen, setSettingsOpen] = useState(false);

  const loadModelData = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Mock data - in real implementation, this would come from the AI engine API
      const mockModels: ModelMetrics[] = [
        {
          id: '1',
          name: 'Protocol Classifier v2.1',
          version: '2.1.0',
          type: 'classification',
          status: 'deployed',
          accuracy: 94.5,
          precision: 93.2,
          recall: 95.1,
          f1Score: 94.1,
          auc: 0.97,
          latency: 12.5,
          throughput: 1250,
          memoryUsage: 512,
          cpuUsage: 35.2,
          gpuUsage: 78.5,
          diskUsage: 1024,
          lastUpdated: new Date(),
          trainingTime: 2.5,
          inferenceCount: 125000,
          errorRate: 0.8,
          datasetSize: 50000,
          features: 128,
          parameters: 2500000,
        },
        {
          id: '2',
          name: 'Anomaly Detection Engine',
          version: '1.3.2',
          type: 'anomaly_detection',
          status: 'deployed',
          accuracy: 89.2,
          precision: 87.8,
          recall: 91.5,
          f1Score: 89.6,
          auc: 0.92,
          latency: 8.2,
          throughput: 2100,
          memoryUsage: 256,
          cpuUsage: 22.1,
          gpuUsage: 45.3,
          diskUsage: 512,
          lastUpdated: new Date(Date.now() - 300000),
          trainingTime: 4.2,
          inferenceCount: 87500,
          errorRate: 1.2,
          datasetSize: 75000,
          features: 64,
          parameters: 1200000,
        },
        {
          id: '3',
          name: 'Threat Intelligence NLP',
          version: '3.0.1',
          type: 'nlp',
          status: 'training',
          accuracy: 76.3,
          precision: 74.5,
          recall: 78.9,
          f1Score: 76.6,
          auc: 0.84,
          latency: 45.0,
          throughput: 150,
          memoryUsage: 2048,
          cpuUsage: 85.7,
          gpuUsage: 92.1,
          diskUsage: 4096,
          lastUpdated: new Date(Date.now() - 600000),
          trainingTime: 12.8,
          inferenceCount: 0,
          errorRate: 0.0,
          datasetSize: 120000,
          features: 512,
          parameters: 15000000,
        }
      ];
      
      setModels(mockModels);

      // Mock performance history
      const now = Date.now();
      const mockHistory: ModelPerformanceHistory[] = Array.from({ length: 60 }, (_, i) => ({
        timestamp: new Date(now - (59 - i) * 60000),
        accuracy: 90 + Math.random() * 10,
        latency: 10 + Math.random() * 20,
        throughput: 1000 + Math.random() * 1000,
        memoryUsage: 400 + Math.random() * 200,
        cpuUsage: 20 + Math.random() * 60,
        errorRate: Math.random() * 2,
      }));
      
      setPerformanceHistory(mockHistory);

      // Mock alerts
      const mockAlerts: ModelAlert[] = [
        {
          id: '1',
          modelId: '1',
          type: 'high_latency',
          severity: 'medium',
          message: 'Average latency increased by 25% over the last hour',
          timestamp: new Date(now - 300000),
          acknowledged: false,
        },
        {
          id: '2',
          modelId: '3',
          type: 'performance_degradation',
          severity: 'high',
          message: 'Training accuracy has plateaued for the last 50 epochs',
          timestamp: new Date(now - 900000),
          acknowledged: true,
        }
      ];
      
      setAlerts(mockAlerts);

      // Mock deployment status
      const mockDeployments: DeploymentStatus[] = [
        {
          modelId: '1',
          environment: 'production',
          replicas: 5,
          healthyReplicas: 5,
          version: '2.1.0',
          lastDeployed: new Date(now - 86400000),
          rolloutStatus: 'completed',
        },
        {
          modelId: '2',
          environment: 'production',
          replicas: 3,
          healthyReplicas: 2,
          version: '1.3.2',
          lastDeployed: new Date(now - 172800000),
          rolloutStatus: 'completed',
        }
      ];
      
      setDeployments(mockDeployments);

    } catch (err) {
      setError('Failed to load model data');
      console.error('Error loading model data:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadModelData();
  }, [loadModelData]);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (autoRefresh) {
      interval = setInterval(loadModelData, refreshInterval * 1000);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh, refreshInterval, loadModelData]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'deployed': return 'success';
      case 'training': return 'info';
      case 'idle': return 'warning';
      case 'error': return 'error';
      case 'updating': return 'info';
      default: return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'deployed': return <SuccessIcon />;
      case 'training': return <PlayIcon />;
      case 'idle': return <PauseIcon />;
      case 'error': return <ErrorIcon />;
      case 'updating': return <RefreshIcon />;
      default: return <StopIcon />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'low': return 'info';
      case 'medium': return 'warning';
      case 'high': return 'error';
      case 'critical': return 'error';
      default: return 'default';
    }
  };

  const getModelTypeIcon = (type: string) => {
    switch (type) {
      case 'classification': return '🎯';
      case 'regression': return '📈';
      case 'clustering': return '🔄';
      case 'anomaly_detection': return '⚠️';
      case 'nlp': return '💬';
      case 'computer_vision': return '👁️';
      default: return '🤖';
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} MB`;
    return `${(bytes / 1024).toFixed(1)} GB`;
  };

  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          AI Model Monitoring
        </Typography>
        
        <Box display="flex" alignItems="center" gap={2}>
          <FormControlLabel
            control={
              <Switch
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
              />
            }
            label="Auto Refresh"
          />
          
          <Button
            variant="outlined"
            onClick={loadModelData}
            startIcon={<RefreshIcon />}
            disabled={loading}
          >
            Refresh
          </Button>

          <Button
            variant="outlined"
            onClick={() => setSettingsOpen(true)}
            startIcon={<SettingsIcon />}
          >
            Settings
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {loading && <LinearProgress sx={{ mb: 2 }} />}

      {/* Model Overview Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <Avatar sx={{ bgcolor: 'primary.main', mr: 2 }}>
                  <AIIcon />
                </Avatar>
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    Active Models
                  </Typography>
                  <Typography variant="h5">
                    {models.filter(m => m.status === 'deployed').length}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <Avatar sx={{ bgcolor: 'info.main', mr: 2 }}>
                  <PlayIcon />
                </Avatar>
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    Training
                  </Typography>
                  <Typography variant="h5">
                    {models.filter(m => m.status === 'training').length}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <Avatar sx={{ bgcolor: 'success.main', mr: 2 }}>
                  <PerformanceIcon />
                </Avatar>
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    Avg Accuracy
                  </Typography>
                  <Typography variant="h5">
                    {(models.reduce((sum, m) => sum + m.accuracy, 0) / models.length || 0).toFixed(1)}%
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center">
                <Avatar sx={{ bgcolor: 'warning.main', mr: 2 }}>
                  <WarningIcon />
                </Avatar>
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    Active Alerts
                  </Typography>
                  <Typography variant="h5">
                    {alerts.filter(a => !a.acknowledged).length}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* Performance Timeline */}
        <Grid item xs={12} lg={8}>
          <Card>
            <CardHeader title="Model Performance Timeline" />
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={performanceHistory}>
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
                    dataKey="accuracy"
                    fill="#8884d8"
                    fillOpacity={0.3}
                    stroke="#8884d8"
                    name="Accuracy (%)"
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="latency"
                    stroke="#82ca9d"
                    strokeWidth={2}
                    name="Latency (ms)"
                  />
                  <Bar
                    yAxisId="left"
                    dataKey="errorRate"
                    fill="#ff7300"
                    name="Error Rate (%)"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Resource Usage */}
        <Grid item xs={12} lg={4}>
          <Card sx={{ height: 400 }}>
            <CardHeader title="Resource Usage" />
            <CardContent>
              {selectedModel ? (
                (() => {
                  const model = models.find(m => m.id === selectedModel);
                  if (!model) return <Typography>Model not found</Typography>;
                  
                  const resourceData = [
                    { subject: 'CPU', A: model.cpuUsage, fullMark: 100 },
                    { subject: 'GPU', A: model.gpuUsage, fullMark: 100 },
                    { subject: 'Memory', A: (model.memoryUsage / 1024) * 100, fullMark: 100 },
                    { subject: 'Disk', A: (model.diskUsage / 4096) * 100, fullMark: 100 },
                    { subject: 'Accuracy', A: model.accuracy, fullMark: 100 },
                    { subject: 'Throughput', A: (model.throughput / 3000) * 100, fullMark: 100 },
                  ];
                  
                  return (
                    <ResponsiveContainer width="100%" height={250}>
                      <RadarChart data={resourceData}>
                        <PolarGrid />
                        <PolarAngleAxis dataKey="subject" />
                        <PolarRadiusAxis angle={90} domain={[0, 100]} />
                        <Radar
                          name={model.name}
                          dataKey="A"
                          stroke="#8884d8"
                          fill="#8884d8"
                          fillOpacity={0.3}
                        />
                      </RadarChart>
                    </ResponsiveContainer>
                  );
                })()
              ) : (
                <Box display="flex" alignItems="center" justifyContent="center" height={250}>
                  <Typography color="textSecondary">
                    Select a model to view resource usage
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Model List */}
        <Grid item xs={12}>
          <Card>
            <CardHeader 
              title="AI Models"
              subheader={`${models.length} models registered`}
            />
            <CardContent sx={{ p: 0 }}>
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Model</TableCell>
                      <TableCell>Type</TableCell>
                      <TableCell>Status</TableCell>
                      <TableCell align="right">Accuracy</TableCell>
                      <TableCell align="right">Latency</TableCell>
                      <TableCell align="right">Throughput</TableCell>
                      <TableCell align="right">Memory</TableCell>
                      <TableCell align="right">Inferences</TableCell>
                      <TableCell>Last Updated</TableCell>
                      <TableCell>Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    <AnimatePresence>
                      {models.map((model) => (
                        <motion.tr
                          key={model.id}
                          initial={{ opacity: 0, y: -10 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: 10 }}
                          transition={{ duration: 0.2 }}
                          component={TableRow}
                          hover
                          selected={selectedModel === model.id}
                          onClick={() => setSelectedModel(
                            selectedModel === model.id ? null : model.id
                          )}
                          sx={{ cursor: 'pointer' }}
                        >
                          <TableCell>
                            <Box display="flex" alignItems="center">
                              <Typography sx={{ fontSize: 20, mr: 1 }}>
                                {getModelTypeIcon(model.type)}
                              </Typography>
                              <Box>
                                <Typography variant="body2" fontWeight="medium">
                                  {model.name}
                                </Typography>
                                <Typography variant="caption" color="textSecondary">
                                  v{model.version}
                                </Typography>
                              </Box>
                            </Box>
                          </TableCell>
                          <TableCell>
                            <Chip 
                              label={model.type.replace('_', ' ')} 
                              size="small"
                              variant="outlined"
                            />
                          </TableCell>
                          <TableCell>
                            <Box display="flex" alignItems="center">
                              {getStatusIcon(model.status)}
                              <Chip 
                                label={model.status} 
                                size="small"
                                color={getStatusColor(model.status) as any}
                                sx={{ ml: 1 }}
                              />
                            </Box>
                          </TableCell>
                          <TableCell align="right">
                            <Typography variant="body2">
                              {model.accuracy.toFixed(1)}%
                            </Typography>
                            <LinearProgress
                              variant="determinate"
                              value={model.accuracy}
                              sx={{ width: 50 }}
                              color={model.accuracy > 90 ? 'success' : model.accuracy > 70 ? 'warning' : 'error'}
                            />
                          </TableCell>
                          <TableCell align="right">{model.latency.toFixed(1)}ms</TableCell>
                          <TableCell align="right">{formatNumber(model.throughput)}/s</TableCell>
                          <TableCell align="right">{formatBytes(model.memoryUsage)}</TableCell>
                          <TableCell align="right">{formatNumber(model.inferenceCount)}</TableCell>
                          <TableCell>
                            <Typography variant="body2" color="textSecondary">
                              {model.lastUpdated.toLocaleString()}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Tooltip title="View Details">
                              <IconButton size="small">
                                <ViewIcon />
                              </IconButton>
                            </Tooltip>
                            <Tooltip title="Deploy">
                              <IconButton size="small">
                                <DeployIcon />
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

        {/* Model Details Accordion */}
        {selectedModel && (
          <Grid item xs={12}>
            <AnimatePresence>
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.3 }}
              >
                {(() => {
                  const model = models.find(m => m.id === selectedModel);
                  if (!model) return null;
                  
                  return (
                    <Accordion expanded>
                      <AccordionSummary>
                        <Typography variant="h6">
                          {model.name} - Detailed Metrics
                        </Typography>
                      </AccordionSummary>
                      <AccordionDetails>
                        <Grid container spacing={3}>
                          <Grid item xs={12} md={6}>
                            <Paper sx={{ p: 2 }}>
                              <Typography variant="h6" gutterBottom>
                                Performance Metrics
                              </Typography>
                              <List dense>
                                <ListItem>
                                  <ListItemText primary="Accuracy" secondary={`${model.accuracy.toFixed(2)}%`} />
                                </ListItem>
                                <ListItem>
                                  <ListItemText primary="Precision" secondary={`${model.precision.toFixed(2)}%`} />
                                </ListItem>
                                <ListItem>
                                  <ListItemText primary="Recall" secondary={`${model.recall.toFixed(2)}%`} />
                                </ListItem>
                                <ListItem>
                                  <ListItemText primary="F1 Score" secondary={`${model.f1Score.toFixed(2)}%`} />
                                </ListItem>
                                <ListItem>
                                  <ListItemText primary="AUC" secondary={model.auc.toFixed(3)} />
                                </ListItem>
                              </List>
                            </Paper>
                          </Grid>
                          
                          <Grid item xs={12} md={6}>
                            <Paper sx={{ p: 2 }}>
                              <Typography variant="h6" gutterBottom>
                                System Resources
                              </Typography>
                              <List dense>
                                <ListItem>
                                  <ListItemText 
                                    primary="Memory Usage" 
                                    secondary={
                                      <Box>
                                        <Typography variant="body2">
                                          {formatBytes(model.memoryUsage)}
                                        </Typography>
                                        <LinearProgress
                                          variant="determinate"
                                          value={(model.memoryUsage / 2048) * 100}
                                          sx={{ mt: 0.5 }}
                                        />
                                      </Box>
                                    }
                                  />
                                </ListItem>
                                <ListItem>
                                  <ListItemText 
                                    primary="CPU Usage" 
                                    secondary={
                                      <Box>
                                        <Typography variant="body2">
                                          {model.cpuUsage.toFixed(1)}%
                                        </Typography>
                                        <LinearProgress
                                          variant="determinate"
                                          value={model.cpuUsage}
                                          sx={{ mt: 0.5 }}
                                          color={model.cpuUsage > 80 ? 'error' : 'primary'}
                                        />
                                      </Box>
                                    }
                                  />
                                </ListItem>
                                <ListItem>
                                  <ListItemText 
                                    primary="GPU Usage" 
                                    secondary={
                                      <Box>
                                        <Typography variant="body2">
                                          {model.gpuUsage.toFixed(1)}%
                                        </Typography>
                                        <LinearProgress
                                          variant="determinate"
                                          value={model.gpuUsage}
                                          sx={{ mt: 0.5 }}
                                          color={model.gpuUsage > 90 ? 'warning' : 'success'}
                                        />
                                      </Box>
                                    }
                                  />
                                </ListItem>
                              </List>
                            </Paper>
                          </Grid>
                        </Grid>
                      </AccordionDetails>
                    </Accordion>
                  );
                })()}
              </motion.div>
            </AnimatePresence>
          </Grid>
        )}

        {/* Active Alerts */}
        {alerts.filter(a => !a.acknowledged).length > 0 && (
          <Grid item xs={12}>
            <Card>
              <CardHeader title="Active Model Alerts" />
              <CardContent>
                <List>
                  {alerts.filter(a => !a.acknowledged).map((alert, index) => {
                    const model = models.find(m => m.id === alert.modelId);
                    return (
                      <ListItem key={alert.id} divider={index < alerts.length - 1}>
                        <ListItemIcon>
                          <Avatar sx={{ bgcolor: `${getSeverityColor(alert.severity)}.main` }}>
                            <WarningIcon />
                          </Avatar>
                        </ListItemIcon>
                        <ListItemText
                          primary={
                            <Box display="flex" alignItems="center" gap={1}>
                              <Typography variant="body1">
                                {model?.name}
                              </Typography>
                              <Chip 
                                label={alert.type.replace('_', ' ')} 
                                size="small"
                                color={getSeverityColor(alert.severity) as any}
                              />
                            </Box>
                          }
                          secondary={
                            <Box>
                              <Typography variant="body2">
                                {alert.message}
                              </Typography>
                              <Typography variant="caption" color="textSecondary">
                                {alert.timestamp.toLocaleString()}
                              </Typography>
                            </Box>
                          }
                        />
                      </ListItem>
                    );
                  })}
                </List>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>

      {/* Settings Dialog */}
      <Dialog open={settingsOpen} onClose={() => setSettingsOpen(false)}>
        <DialogTitle>Model Monitoring Settings</DialogTitle>
        <DialogContent>
          <Box sx={{ minWidth: 300, pt: 2 }}>
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Refresh Interval</InputLabel>
              <Select
                value={refreshInterval}
                onChange={(e) => setRefreshInterval(e.target.value as number)}
                label="Refresh Interval"
              >
                <MenuItem value={10}>10 seconds</MenuItem>
                <MenuItem value={30}>30 seconds</MenuItem>
                <MenuItem value={60}>1 minute</MenuItem>
                <MenuItem value={300}>5 minutes</MenuItem>
              </Select>
            </FormControl>
            
            <FormControlLabel
              control={
                <Switch
                  checked={autoRefresh}
                  onChange={(e) => setAutoRefresh(e.target.checked)}
                />
              }
              label="Enable auto-refresh"
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSettingsOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default AIModelMonitoring;