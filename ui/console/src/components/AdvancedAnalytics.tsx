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
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Tabs,
  Tab,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Slider,
  ButtonGroup,
  Menu,
  Checkbox,
  FormGroup,
  RadioGroup,
  Radio,
  FormLabel,
} from '@mui/material';
import {
  Analytics as AnalyticsIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Timeline as TimelineIcon,
  PieChart as PieChartIcon,
  BarChart as BarChartIcon,
  ShowChart as LineChartIcon,
  ScatterPlot as ScatterPlotIcon,
  Insights as InsightsIcon,
  Psychology as AIIcon,
  Speed as PerformanceIcon,
  Refresh as RefreshIcon,
  FilterList as FilterIcon,
  Settings as SettingsIcon,
  Download as DownloadIcon,
  Share as ShareIcon,
  Fullscreen as FullscreenIcon,
  FullscreenExit as FullscreenExitIcon,
  ExpandMore as ExpandMoreIcon,
  ViewModule as DashboardIcon,
  GridView as GridIcon,
  List as ListIcon,
  DateRange as DateRangeIcon,
  Tune as TuneIcon,
  AutoGraph as PredictiveIcon,
  DataUsage as DataIcon,
  Assessment as ReportIcon,
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
  ScatterChart,
  Scatter,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Treemap,
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import { DeviceApiClient } from '../api/devices';

interface AdvancedAnalyticsProps {
  apiClient: DeviceApiClient;
}

interface AnalyticsMetric {
  id: string;
  name: string;
  value: number;
  unit: string;
  trend: 'up' | 'down' | 'stable';
  trendValue: number;
  category: 'performance' | 'security' | 'network' | 'usage';
  timestamp: Date;
  prediction?: number;
  confidence?: number;
}

interface TimeSeriesData {
  timestamp: Date;
  value: number;
  predicted?: number;
  anomaly?: boolean;
  upperBound?: number;
  lowerBound?: number;
}

interface PredictiveModel {
  id: string;
  name: string;
  metric: string;
  algorithm: 'arima' | 'lstm' | 'linear_regression' | 'prophet';
  accuracy: number;
  status: 'active' | 'training' | 'inactive';
  lastTrained: Date;
  nextPrediction: Date;
  horizon: number; // days
}

interface Insight {
  id: string;
  title: string;
  description: string;
  category: 'anomaly' | 'trend' | 'forecast' | 'recommendation';
  severity: 'info' | 'warning' | 'critical';
  confidence: number;
  timestamp: Date;
  metrics: string[];
  actionable: boolean;
}

interface AnalyticsDashboard {
  id: string;
  name: string;
  widgets: DashboardWidget[];
  layout: WidgetLayout[];
  isDefault: boolean;
  createdBy: string;
  createdAt: Date;
}

interface DashboardWidget {
  id: string;
  type: 'line_chart' | 'bar_chart' | 'pie_chart' | 'scatter_plot' | 'radar_chart' | 'metric_card' | 'table' | 'treemap';
  title: string;
  metric: string;
  timeRange: string;
  config: any;
}

interface WidgetLayout {
  widgetId: string;
  x: number;
  y: number;
  w: number;
  h: number;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => (
  <div role="tabpanel" hidden={value !== index}>
    {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
  </div>
);

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];

const AdvancedAnalytics: React.FC<AdvancedAnalyticsProps> = ({ apiClient }) => {
  const [tabValue, setTabValue] = useState(0);
  const [metrics, setMetrics] = useState<AnalyticsMetric[]>([]);
  const [timeSeriesData, setTimeSeriesData] = useState<TimeSeriesData[]>([]);
  const [predictiveModels, setPredictiveModels] = useState<PredictiveModel[]>([]);
  const [insights, setInsights] = useState<Insight[]>([]);
  const [dashboards, setDashboards] = useState<AnalyticsDashboard[]>([]);
  const [selectedMetric, setSelectedMetric] = useState<string>('network_throughput');
  const [timeRange, setTimeRange] = useState('24h');
  const [predictiveHorizon, setPredictiveHorizon] = useState(7);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [fullscreenWidget, setFullscreenWidget] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [viewMode, setViewMode] = useState<'dashboard' | 'grid' | 'list'>('dashboard');

  const loadAnalyticsData = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Mock analytics data
      const mockMetrics: AnalyticsMetric[] = [
        {
          id: 'network_throughput',
          name: 'Network Throughput',
          value: 847.5,
          unit: 'Mbps',
          trend: 'up',
          trendValue: 12.3,
          category: 'performance',
          timestamp: new Date(),
          prediction: 923.2,
          confidence: 87.5,
        },
        {
          id: 'cpu_utilization',
          name: 'CPU Utilization',
          value: 65.8,
          unit: '%',
          trend: 'stable',
          trendValue: -0.2,
          category: 'performance',
          timestamp: new Date(),
          prediction: 68.4,
          confidence: 92.1,
        },
        {
          id: 'security_incidents',
          name: 'Security Incidents',
          value: 23,
          unit: 'count',
          trend: 'down',
          trendValue: -15.7,
          category: 'security',
          timestamp: new Date(),
          prediction: 18,
          confidence: 75.3,
        },
        {
          id: 'device_connections',
          name: 'Device Connections',
          value: 1247,
          unit: 'count',
          trend: 'up',
          trendValue: 8.9,
          category: 'network',
          timestamp: new Date(),
          prediction: 1356,
          confidence: 89.7,
        }
      ];
      
      setMetrics(mockMetrics);

      // Generate time series data with predictions
      const now = Date.now();
      const mockTimeSeries: TimeSeriesData[] = Array.from({ length: 168 }, (_, i) => {
        const timestamp = new Date(now - (167 - i) * 60 * 60 * 1000); // hourly data for 7 days
        const baseValue = 500 + Math.sin(i / 24) * 100 + Math.random() * 50;
        const isAnomaly = Math.random() < 0.05;
        const predicted = i > 140 ? baseValue + Math.sin((i - 140) / 24) * 80 + (Math.random() - 0.5) * 20 : undefined;
        
        return {
          timestamp,
          value: isAnomaly ? baseValue * 1.5 : baseValue,
          predicted,
          anomaly: isAnomaly,
          upperBound: baseValue * 1.2,
          lowerBound: baseValue * 0.8,
        };
      });
      
      setTimeSeriesData(mockTimeSeries);

      const mockModels: PredictiveModel[] = [
        {
          id: '1',
          name: 'Network Throughput Predictor',
          metric: 'network_throughput',
          algorithm: 'lstm',
          accuracy: 89.5,
          status: 'active',
          lastTrained: new Date(now - 86400000),
          nextPrediction: new Date(now + 3600000),
          horizon: 7,
        },
        {
          id: '2',
          name: 'Security Incident Forecaster',
          metric: 'security_incidents',
          algorithm: 'prophet',
          accuracy: 75.2,
          status: 'active',
          lastTrained: new Date(now - 172800000),
          nextPrediction: new Date(now + 7200000),
          horizon: 14,
        }
      ];
      
      setPredictiveModels(mockModels);

      const mockInsights: Insight[] = [
        {
          id: '1',
          title: 'Unusual Network Traffic Pattern',
          description: 'Network throughput shows 35% increase during off-peak hours, suggesting potential security concern or system malfunction.',
          category: 'anomaly',
          severity: 'warning',
          confidence: 87.3,
          timestamp: new Date(now - 3600000),
          metrics: ['network_throughput'],
          actionable: true,
        },
        {
          id: '2',
          title: 'Predicted Resource Shortage',
          description: 'CPU utilization forecast indicates potential resource exhaustion in the next 48 hours based on current growth trend.',
          category: 'forecast',
          severity: 'critical',
          confidence: 92.1,
          timestamp: new Date(now - 1800000),
          metrics: ['cpu_utilization'],
          actionable: true,
        },
        {
          id: '3',
          title: 'Security Improvements Detected',
          description: 'Security incident rate has decreased by 23% over the past week, indicating effective security measures implementation.',
          category: 'trend',
          severity: 'info',
          confidence: 94.7,
          timestamp: new Date(now - 7200000),
          metrics: ['security_incidents'],
          actionable: false,
        }
      ];
      
      setInsights(mockInsights);

    } catch (err) {
      setError('Failed to load analytics data');
      console.error('Error loading analytics data:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadAnalyticsData();
  }, [loadAnalyticsData]);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (autoRefresh) {
      interval = setInterval(loadAnalyticsData, 300000); // Refresh every 5 minutes
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh, loadAnalyticsData]);

  const getTrendIcon = (trend: string, value: number) => {
    if (trend === 'up') return <TrendingUpIcon color={value > 0 ? 'success' : 'error'} />;
    if (trend === 'down') return <TrendingDownIcon color={value < 0 ? 'success' : 'error'} />;
    return <TimelineIcon color="info" />;
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'error';
      case 'warning': return 'warning';
      case 'info': return 'info';
      default: return 'default';
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'performance': return '#2196F3';
      case 'security': return '#F44336';
      case 'network': return '#4CAF50';
      case 'usage': return '#FF9800';
      default: return '#9C27B0';
    }
  };

  const currentData = timeSeriesData.filter(d => {
    const hoursAgo = {
      '1h': 1,
      '6h': 6,
      '24h': 24,
      '7d': 168,
      '30d': 720,
    }[timeRange] || 24;
    
    return d.timestamp > new Date(Date.now() - hoursAgo * 60 * 60 * 1000);
  });

  const formatTimestamp = (timestamp: Date) => {
    if (timeRange === '1h' || timeRange === '6h') {
      return timestamp.toLocaleTimeString();
    }
    if (timeRange === '24h') {
      return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    return timestamp.toLocaleDateString();
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          Advanced Analytics
        </Typography>
        
        <Box display="flex" alignItems="center" gap={2}>
          <ButtonGroup variant="outlined" size="small">
            <Button
              variant={viewMode === 'dashboard' ? 'contained' : 'outlined'}
              onClick={() => setViewMode('dashboard')}
              startIcon={<DashboardIcon />}
            >
              Dashboard
            </Button>
            <Button
              variant={viewMode === 'grid' ? 'contained' : 'outlined'}
              onClick={() => setViewMode('grid')}
              startIcon={<GridIcon />}
            >
              Grid
            </Button>
            <Button
              variant={viewMode === 'list' ? 'contained' : 'outlined'}
              onClick={() => setViewMode('list')}
              startIcon={<ListIcon />}
            >
              List
            </Button>
          </ButtonGroup>
          
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
            onClick={loadAnalyticsData}
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

      {/* Key Metrics Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        {metrics.map((metric) => (
          <Grid item xs={12} sm={6} md={3} key={metric.id}>
            <Card
              sx={{ 
                cursor: 'pointer',
                '&:hover': { elevation: 4 }
              }}
              onClick={() => setSelectedMetric(metric.id)}
            >
              <CardContent>
                <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
                  <Avatar sx={{ bgcolor: getCategoryColor(metric.category) }}>
                    {metric.category === 'performance' && <PerformanceIcon />}
                    {metric.category === 'security' && <InsightsIcon />}
                    {metric.category === 'network' && <TimelineIcon />}
                    {metric.category === 'usage' && <DataIcon />}
                  </Avatar>
                  {getTrendIcon(metric.trend, metric.trendValue)}
                </Box>
                
                <Typography color="textSecondary" gutterBottom variant="body2">
                  {metric.name}
                </Typography>
                
                <Typography variant="h4" component="div">
                  {metric.value.toLocaleString()}
                  <Typography variant="caption" color="textSecondary" sx={{ ml: 1 }}>
                    {metric.unit}
                  </Typography>
                </Typography>
                
                <Box display="flex" justifyContent="space-between" alignItems="center" mt={1}>
                  <Typography 
                    variant="body2" 
                    color={metric.trendValue > 0 ? 'success.main' : metric.trendValue < 0 ? 'error.main' : 'textSecondary'}
                  >
                    {metric.trendValue > 0 ? '+' : ''}{metric.trendValue.toFixed(1)}%
                  </Typography>
                  
                  {metric.prediction && (
                    <Tooltip title={`Predicted: ${metric.prediction} (${metric.confidence}% confidence)`}>
                      <Chip 
                        size="small" 
                        icon={<PredictiveIcon />} 
                        label={`${metric.prediction.toFixed(1)}`}
                        variant="outlined"
                      />
                    </Tooltip>
                  )}
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Tabs for different views */}
      <Card>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={(e, newValue) => setTabValue(newValue)}>
            <Tab label="Time Series Analysis" />
            <Tab label="Predictive Models" />
            <Tab label="Insights & Anomalies" />
            <Tab label="Custom Dashboards" />
          </Tabs>
        </Box>

        {/* Time Series Analysis Tab */}
        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            <Grid item xs={12} lg={9}>
              <Card>
                <CardHeader 
                  title="Time Series Analysis with Predictions"
                  action={
                    <Box display="flex" alignItems="center" gap={2}>
                      <FormControl size="small" sx={{ minWidth: 120 }}>
                        <InputLabel>Metric</InputLabel>
                        <Select
                          value={selectedMetric}
                          onChange={(e) => setSelectedMetric(e.target.value)}
                          label="Metric"
                        >
                          {metrics.map((metric) => (
                            <MenuItem key={metric.id} value={metric.id}>
                              {metric.name}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                      
                      <FormControl size="small" sx={{ minWidth: 100 }}>
                        <InputLabel>Range</InputLabel>
                        <Select
                          value={timeRange}
                          onChange={(e) => setTimeRange(e.target.value)}
                          label="Range"
                        >
                          <MenuItem value="1h">1 Hour</MenuItem>
                          <MenuItem value="6h">6 Hours</MenuItem>
                          <MenuItem value="24h">24 Hours</MenuItem>
                          <MenuItem value="7d">7 Days</MenuItem>
                          <MenuItem value="30d">30 Days</MenuItem>
                        </Select>
                      </FormControl>
                      
                      <IconButton onClick={() => setFullscreenWidget('timeseries')}>
                        <FullscreenIcon />
                      </IconButton>
                    </Box>
                  }
                />
                <CardContent>
                  <ResponsiveContainer width="100%" height={400}>
                    <ComposedChart data={currentData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis 
                        dataKey="timestamp"
                        tickFormatter={formatTimestamp}
                      />
                      <YAxis />
                      <RechartsTooltip 
                        labelFormatter={(value) => new Date(value).toLocaleString()}
                      />
                      
                      {/* Confidence bands */}
                      <Area
                        type="monotone"
                        dataKey="upperBound"
                        stackId="1"
                        stroke="none"
                        fill="#e3f2fd"
                        fillOpacity={0.3}
                      />
                      <Area
                        type="monotone"
                        dataKey="lowerBound"
                        stackId="1"
                        stroke="none"
                        fill="#ffffff"
                        fillOpacity={1}
                      />
                      
                      {/* Actual values */}
                      <Line
                        type="monotone"
                        dataKey="value"
                        stroke="#2196f3"
                        strokeWidth={2}
                        dot={(props) => {
                          const { payload } = props;
                          if (payload?.anomaly) {
                            return <circle {...props} r={4} fill="#f44336" stroke="#f44336" />;
                          }
                          return <circle {...props} r={2} fill="#2196f3" stroke="#2196f3" />;
                        }}
                        name="Actual"
                      />
                      
                      {/* Predictions */}
                      <Line
                        type="monotone"
                        dataKey="predicted"
                        stroke="#ff9800"
                        strokeWidth={2}
                        strokeDasharray="5 5"
                        dot={false}
                        name="Predicted"
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} lg={3}>
              <Card>
                <CardHeader title="Analysis Controls" />
                <CardContent>
                  <Box sx={{ mb: 3 }}>
                    <Typography gutterBottom>
                      Prediction Horizon: {predictiveHorizon} days
                    </Typography>
                    <Slider
                      value={predictiveHorizon}
                      onChange={(e, newValue) => setPredictiveHorizon(newValue as number)}
                      min={1}
                      max={30}
                      marks
                      valueLabelDisplay="auto"
                    />
                  </Box>
                  
                  <Divider sx={{ my: 2 }} />
                  
                  <Box>
                    <Typography variant="h6" gutterBottom>
                      Statistical Summary
                    </Typography>
                    {(() => {
                      const values = currentData.map(d => d.value);
                      const mean = values.reduce((a, b) => a + b, 0) / values.length;
                      const std = Math.sqrt(values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length);
                      const anomalies = currentData.filter(d => d.anomaly).length;
                      
                      return (
                        <List dense>
                          <ListItem>
                            <ListItemText primary="Mean" secondary={mean.toFixed(2)} />
                          </ListItem>
                          <ListItem>
                            <ListItemText primary="Std Dev" secondary={std.toFixed(2)} />
                          </ListItem>
                          <ListItem>
                            <ListItemText primary="Anomalies" secondary={anomalies} />
                          </ListItem>
                          <ListItem>
                            <ListItemText primary="Data Points" secondary={currentData.length} />
                          </ListItem>
                        </List>
                      );
                    })()}
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Predictive Models Tab */}
        <TabPanel value={tabValue} index={1}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Card>
                <CardHeader title="Predictive Models" />
                <CardContent>
                  <TableContainer>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Model</TableCell>
                          <TableCell>Metric</TableCell>
                          <TableCell>Algorithm</TableCell>
                          <TableCell>Accuracy</TableCell>
                          <TableCell>Status</TableCell>
                          <TableCell>Last Trained</TableCell>
                          <TableCell>Horizon</TableCell>
                          <TableCell>Actions</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {predictiveModels.map((model) => (
                          <TableRow key={model.id}>
                            <TableCell>
                              <Box display="flex" alignItems="center">
                                <Avatar sx={{ mr: 2, bgcolor: 'primary.main' }}>
                                  <AIIcon />
                                </Avatar>
                                <Typography variant="body2">
                                  {model.name}
                                </Typography>
                              </Box>
                            </TableCell>
                            <TableCell>
                              <Chip label={model.metric} size="small" />
                            </TableCell>
                            <TableCell>
                              <Chip label={model.algorithm.toUpperCase()} size="small" variant="outlined" />
                            </TableCell>
                            <TableCell>
                              <Box display="flex" alignItems="center">
                                <Typography variant="body2" sx={{ mr: 1 }}>
                                  {model.accuracy.toFixed(1)}%
                                </Typography>
                                <LinearProgress
                                  variant="determinate"
                                  value={model.accuracy}
                                  sx={{ width: 50 }}
                                  color={model.accuracy > 85 ? 'success' : 'warning'}
                                />
                              </Box>
                            </TableCell>
                            <TableCell>
                              <Chip
                                label={model.status}
                                size="small"
                                color={model.status === 'active' ? 'success' : model.status === 'training' ? 'warning' : 'default'}
                              />
                            </TableCell>
                            <TableCell>
                              <Typography variant="body2">
                                {model.lastTrained.toLocaleDateString()}
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Typography variant="body2">
                                {model.horizon} days
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Tooltip title="Retrain Model">
                                <IconButton size="small">
                                  <RefreshIcon />
                                </IconButton>
                              </Tooltip>
                              <Tooltip title="Download Model">
                                <IconButton size="small">
                                  <DownloadIcon />
                                </IconButton>
                              </Tooltip>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Insights Tab */}
        <TabPanel value={tabValue} index={2}>
          <Grid container spacing={3}>
            {insights.map((insight) => (
              <Grid item xs={12} md={6} key={insight.id}>
                <Card>
                  <CardHeader
                    avatar={
                      <Avatar sx={{ bgcolor: `${getSeverityColor(insight.severity)}.main` }}>
                        {insight.category === 'anomaly' && <InsightsIcon />}
                        {insight.category === 'trend' && <TrendingUpIcon />}
                        {insight.category === 'forecast' && <PredictiveIcon />}
                        {insight.category === 'recommendation' && <ReportIcon />}
                      </Avatar>
                    }
                    title={insight.title}
                    subheader={
                      <Box display="flex" alignItems="center" gap={1}>
                        <Chip
                          label={insight.category}
                          size="small"
                          color={getSeverityColor(insight.severity) as any}
                        />
                        <Chip
                          label={`${insight.confidence.toFixed(1)}% confidence`}
                          size="small"
                          variant="outlined"
                        />
                      </Box>
                    }
                  />
                  <CardContent>
                    <Typography variant="body2" paragraph>
                      {insight.description}
                    </Typography>
                    
                    <Box display="flex" flexWrap="wrap" gap={0.5} mb={2}>
                      {insight.metrics.map((metric) => (
                        <Chip key={metric} label={metric} size="small" variant="outlined" />
                      ))}
                    </Box>
                    
                    <Box display="flex" justifyContent="between" alignItems="center">
                      <Typography variant="caption" color="textSecondary">
                        {insight.timestamp.toLocaleString()}
                      </Typography>
                      
                      {insight.actionable && (
                        <Button size="small" variant="outlined">
                          Take Action
                        </Button>
                      )}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </TabPanel>

        {/* Custom Dashboards Tab */}
        <TabPanel value={tabValue} index={3}>
          <Box display="flex" justifyContent="center" alignItems="center" height={400}>
            <Paper sx={{ p: 4, textAlign: 'center' }}>
              <DashboardIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
              <Typography variant="h6" gutterBottom>
                Custom Dashboard Builder
              </Typography>
              <Typography variant="body2" color="textSecondary" paragraph>
                Create and customize your own analytics dashboards with drag-and-drop widgets.
              </Typography>
              <Button variant="contained" startIcon={<DashboardIcon />}>
                Create Dashboard
              </Button>
            </Paper>
          </Box>
        </TabPanel>
      </Card>

      {/* Settings Dialog */}
      <Dialog open={settingsOpen} onClose={() => setSettingsOpen(false)}>
        <DialogTitle>Analytics Settings</DialogTitle>
        <DialogContent>
          <Box sx={{ minWidth: 400, pt: 2 }}>
            <FormControl fullWidth sx={{ mb: 2 }}>
              <FormLabel>Auto Refresh Interval</FormLabel>
              <RadioGroup defaultValue="300" row>
                <FormControlLabel value="60" control={<Radio />} label="1 min" />
                <FormControlLabel value="300" control={<Radio />} label="5 min" />
                <FormControlLabel value="900" control={<Radio />} label="15 min" />
              </RadioGroup>
            </FormControl>
            
            <FormControl fullWidth sx={{ mb: 2 }}>
              <FormLabel>Enabled Features</FormLabel>
              <FormGroup>
                <FormControlLabel control={<Checkbox defaultChecked />} label="Anomaly Detection" />
                <FormControlLabel control={<Checkbox defaultChecked />} label="Predictive Analytics" />
                <FormControlLabel control={<Checkbox defaultChecked />} label="Real-time Insights" />
                <FormControlLabel control={<Checkbox />} label="Advanced Visualizations" />
              </FormGroup>
            </FormControl>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSettingsOpen(false)}>Cancel</Button>
          <Button variant="contained">Save Settings</Button>
        </DialogActions>
      </Dialog>

      {/* Fullscreen Widget Dialog */}
      <Dialog
        open={Boolean(fullscreenWidget)}
        onClose={() => setFullscreenWidget(null)}
        maxWidth="xl"
        fullWidth
        fullScreen
      >
        <DialogTitle>
          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Typography variant="h6">Fullscreen Analytics</Typography>
            <IconButton onClick={() => setFullscreenWidget(null)}>
              <FullscreenExitIcon />
            </IconButton>
          </Box>
        </DialogTitle>
        <DialogContent>
          {fullscreenWidget === 'timeseries' && (
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={currentData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="timestamp"
                  tickFormatter={formatTimestamp}
                />
                <YAxis />
                <RechartsTooltip 
                  labelFormatter={(value) => new Date(value).toLocaleString()}
                />
                <Area
                  type="monotone"
                  dataKey="upperBound"
                  stackId="1"
                  stroke="none"
                  fill="#e3f2fd"
                  fillOpacity={0.3}
                />
                <Area
                  type="monotone"
                  dataKey="lowerBound"
                  stackId="1"
                  stroke="none"
                  fill="#ffffff"
                  fillOpacity={1}
                />
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke="#2196f3"
                  strokeWidth={2}
                  name="Actual"
                />
                <Line
                  type="monotone"
                  dataKey="predicted"
                  stroke="#ff9800"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  dot={false}
                  name="Predicted"
                />
              </ComposedChart>
            </ResponsiveContainer>
          )}
        </DialogContent>
      </Dialog>
    </Box>
  );
};

export default AdvancedAnalytics;