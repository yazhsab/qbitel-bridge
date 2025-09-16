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
  ListItemSecondaryAction,
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
  Badge,
  Tabs,
  Tab,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Breadcrumbs,
  Link,
} from '@mui/material';
import {
  Security as SecurityIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
  Block as BlockIcon,
  CheckCircle as CheckCircleIcon,
  Timeline as TimelineIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
  Refresh as RefreshIcon,
  FilterList as FilterIcon,
  Search as SearchIcon,
  Download as DownloadIcon,
  Upload as UploadIcon,
  PlayArrow as PlayArrowIcon,
  Pause as PauseIcon,
  Stop as StopIcon,
  Settings as SettingsIcon,
  NotificationsActive as AlertIcon,
  Map as MapIcon,
  Language as GlobalIcon,
  Fingerprint as FingerprintIcon,
  Shield as ShieldIcon,
  Gavel as ActionIcon,
  ExpandMore as ExpandMoreIcon,
  NavigateNext as NavigateNextIcon,
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
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import { DeviceApiClient } from '../api/devices';

interface ThreatIntelligenceProps {
  apiClient: DeviceApiClient;
}

interface ThreatIndicator {
  id: string;
  type: 'ip' | 'domain' | 'hash' | 'url' | 'email' | 'file';
  value: string;
  threatType: 'malware' | 'botnet' | 'phishing' | 'c2' | 'exploit' | 'spam' | 'tor';
  severity: 'low' | 'medium' | 'high' | 'critical';
  confidence: number;
  firstSeen: Date;
  lastSeen: Date;
  sources: string[];
  tags: string[];
  description: string;
  status: 'active' | 'inactive' | 'whitelisted' | 'investigating';
  iocs: string[];
  mitre: string[];
}

interface SecurityAlert {
  id: string;
  title: string;
  description: string;
  severity: 'info' | 'low' | 'medium' | 'high' | 'critical';
  category: 'network' | 'endpoint' | 'user' | 'application' | 'data';
  status: 'new' | 'investigating' | 'confirmed' | 'resolved' | 'false_positive';
  assignee?: string;
  source: string;
  sourceIp: string;
  targetIp?: string;
  timestamp: Date;
  indicators: ThreatIndicator[];
  playbook?: string;
  evidence: any[];
  timeline: AlertTimelineEvent[];
}

interface AlertTimelineEvent {
  timestamp: Date;
  action: string;
  user: string;
  details: string;
}

interface ThreatFeed {
  id: string;
  name: string;
  provider: string;
  type: 'commercial' | 'open_source' | 'government' | 'private';
  status: 'active' | 'inactive' | 'error';
  lastUpdate: Date;
  indicatorCount: number;
  accuracy: number;
  coverage: string[];
}

interface SOCMetrics {
  totalAlerts: number;
  newAlerts: number;
  criticalAlerts: number;
  resolvedAlerts: number;
  falsePositives: number;
  meanTimeToDetection: number;
  meanTimeToResponse: number;
  meanTimeToResolution: number;
  threatIndicators: number;
  activeCampaigns: number;
  blockedAttempts: number;
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

const SEVERITY_COLORS = {
  info: '#2196F3',
  low: '#4CAF50',
  medium: '#FF9800',
  high: '#F44336',
  critical: '#9C27B0',
};

const ThreatIntelligence: React.FC<ThreatIntelligenceProps> = ({ apiClient }) => {
  const [tabValue, setTabValue] = useState(0);
  const [alerts, setAlerts] = useState<SecurityAlert[]>([]);
  const [indicators, setIndicators] = useState<ThreatIndicator[]>([]);
  const [feeds, setFeeds] = useState<ThreatFeed[]>([]);
  const [metrics, setMetrics] = useState<SOCMetrics>({
    totalAlerts: 0,
    newAlerts: 0,
    criticalAlerts: 0,
    resolvedAlerts: 0,
    falsePositives: 0,
    meanTimeToDetection: 0,
    meanTimeToResponse: 0,
    meanTimeToResolution: 0,
    threatIndicators: 0,
    activeCampaigns: 0,
    blockedAttempts: 0,
  });
  const [selectedAlert, setSelectedAlert] = useState<string | null>(null);
  const [alertDetailsOpen, setAlertDetailsOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [filterSeverity, setFilterSeverity] = useState<string[]>([]);
  const [searchQuery, setSearchQuery] = useState('');

  const loadThreatData = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Mock threat intelligence data
      const mockAlerts: SecurityAlert[] = [
        {
          id: '1',
          title: 'Suspicious Network Activity Detected',
          description: 'Multiple failed authentication attempts from suspicious IP address',
          severity: 'high',
          category: 'network',
          status: 'investigating',
          assignee: 'security.analyst@company.com',
          source: 'SIEM',
          sourceIp: '192.168.100.45',
          targetIp: '10.0.0.1',
          timestamp: new Date(),
          indicators: [],
          evidence: [],
          timeline: [
            {
              timestamp: new Date(),
              action: 'Alert Created',
              user: 'System',
              details: 'Automated detection triggered by SIEM correlation rules'
            }
          ]
        },
        {
          id: '2',
          title: 'Malware Communication Detected',
          description: 'Device communicating with known C2 server',
          severity: 'critical',
          category: 'endpoint',
          status: 'confirmed',
          assignee: 'incident.response@company.com',
          source: 'EDR',
          sourceIp: '192.168.50.120',
          targetIp: '185.220.101.45',
          timestamp: new Date(Date.now() - 300000),
          indicators: [],
          evidence: [],
          timeline: [
            {
              timestamp: new Date(Date.now() - 300000),
              action: 'Alert Created',
              user: 'System',
              details: 'EDR agent detected suspicious network communication'
            },
            {
              timestamp: new Date(Date.now() - 180000),
              action: 'Investigation Started',
              user: 'John Doe',
              details: 'Assigned to incident response team for investigation'
            }
          ]
        }
      ];
      
      setAlerts(mockAlerts);

      const mockIndicators: ThreatIndicator[] = [
        {
          id: '1',
          type: 'ip',
          value: '185.220.101.45',
          threatType: 'c2',
          severity: 'high',
          confidence: 95,
          firstSeen: new Date(Date.now() - 86400000),
          lastSeen: new Date(),
          sources: ['VirusTotal', 'Shodan', 'AbuseIPDB'],
          tags: ['malware', 'botnet', 'tor'],
          description: 'Known C2 server hosting multiple malware families',
          status: 'active',
          iocs: ['185.220.101.45:8080', '185.220.101.45:443'],
          mitre: ['T1071.001', 'T1095']
        },
        {
          id: '2',
          type: 'domain',
          value: 'malicious-site.example.com',
          threatType: 'phishing',
          severity: 'medium',
          confidence: 78,
          firstSeen: new Date(Date.now() - 172800000),
          lastSeen: new Date(Date.now() - 3600000),
          sources: ['PhishTank', 'URLVoid'],
          tags: ['phishing', 'credential-theft'],
          description: 'Phishing site mimicking popular banking portal',
          status: 'active',
          iocs: ['malicious-site.example.com'],
          mitre: ['T1566.002']
        }
      ];
      
      setIndicators(mockIndicators);

      const mockFeeds: ThreatFeed[] = [
        {
          id: '1',
          name: 'Commercial Threat Feed',
          provider: 'ThreatConnect',
          type: 'commercial',
          status: 'active',
          lastUpdate: new Date(),
          indicatorCount: 15420,
          accuracy: 94.5,
          coverage: ['malware', 'c2', 'phishing']
        },
        {
          id: '2',
          name: 'Open Source Intelligence',
          provider: 'MISP Community',
          type: 'open_source',
          status: 'active',
          lastUpdate: new Date(Date.now() - 1800000),
          indicatorCount: 8750,
          accuracy: 87.2,
          coverage: ['malware', 'botnet', 'exploit']
        }
      ];
      
      setFeeds(mockFeeds);

      const mockMetrics: SOCMetrics = {
        totalAlerts: 2847,
        newAlerts: 23,
        criticalAlerts: 8,
        resolvedAlerts: 2651,
        falsePositives: 165,
        meanTimeToDetection: 4.5,
        meanTimeToResponse: 12.3,
        meanTimeToResolution: 45.7,
        threatIndicators: 24170,
        activeCampaigns: 12,
        blockedAttempts: 1547,
      };
      
      setMetrics(mockMetrics);

    } catch (err) {
      setError('Failed to load threat intelligence data');
      console.error('Error loading threat data:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadThreatData();
  }, [loadThreatData]);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (autoRefresh) {
      interval = setInterval(loadThreatData, 30000); // Refresh every 30 seconds
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh, loadThreatData]);

  const getSeverityColor = (severity: string) => {
    return SEVERITY_COLORS[severity as keyof typeof SEVERITY_COLORS] || SEVERITY_COLORS.info;
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical': return <ErrorIcon />;
      case 'high': return <WarningIcon />;
      case 'medium': return <AlertIcon />;
      case 'low': return <CheckCircleIcon />;
      default: return <SecurityIcon />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'new': return 'error';
      case 'investigating': return 'warning';
      case 'confirmed': return 'error';
      case 'resolved': return 'success';
      case 'false_positive': return 'info';
      default: return 'default';
    }
  };

  const handleAlertAction = (alertId: string, action: string) => {
    // Implementation for alert actions (assign, resolve, etc.)
    console.log(`Action ${action} on alert ${alertId}`);
  };

  const filteredAlerts = alerts.filter(alert => {
    const matchesSeverity = filterSeverity.length === 0 || filterSeverity.includes(alert.severity);
    const matchesSearch = searchQuery === '' || 
      alert.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      alert.description.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesSeverity && matchesSearch;
  });

  return (
    <Box sx={{ flexGrow: 1 }}>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Box>
          <Typography variant="h4" component="h1">
            Threat Intelligence & SOC
          </Typography>
          <Breadcrumbs separator={<NavigateNextIcon fontSize="small" />}>
            <Link color="inherit" href="/dashboard">
              Dashboard
            </Link>
            <Typography color="textPrimary">Threat Intelligence</Typography>
          </Breadcrumbs>
        </Box>
        
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
            onClick={loadThreatData}
            startIcon={<RefreshIcon />}
            disabled={loading}
          >
            Refresh
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {loading && <LinearProgress sx={{ mb: 2 }} />}

      {/* SOC Metrics Dashboard */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
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
                    {metrics.totalAlerts.toLocaleString()}
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
                  <Badge badgeContent={metrics.newAlerts} color="error">
                    <AlertIcon />
                  </Badge>
                </Avatar>
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    New Alerts
                  </Typography>
                  <Typography variant="h5">
                    {metrics.newAlerts}
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
                  <ErrorIcon />
                </Avatar>
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    Critical
                  </Typography>
                  <Typography variant="h5">
                    {metrics.criticalAlerts}
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
                    Resolved
                  </Typography>
                  <Typography variant="h5">
                    {metrics.resolvedAlerts}
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
                  <FingerprintIcon />
                </Avatar>
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    IOCs
                  </Typography>
                  <Typography variant="h5">
                    {metrics.threatIndicators.toLocaleString()}
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
                  <BlockIcon />
                </Avatar>
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    Blocked
                  </Typography>
                  <Typography variant="h5">
                    {metrics.blockedAttempts}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Tabbed Interface */}
      <Card>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={(e, newValue) => setTabValue(newValue)}>
            <Tab label="Security Alerts" />
            <Tab label="Threat Indicators" />
            <Tab label="Threat Feeds" />
            <Tab label="Analytics" />
          </Tabs>
        </Box>

        {/* Security Alerts Tab */}
        <TabPanel value={tabValue} index={0}>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Box display="flex" alignItems="center" gap={2}>
              <TextField
                size="small"
                placeholder="Search alerts..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                InputProps={{
                  startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />,
                }}
              />
              
              <FormControl size="small" sx={{ minWidth: 120 }}>
                <InputLabel>Severity</InputLabel>
                <Select
                  multiple
                  value={filterSeverity}
                  onChange={(e) => setFilterSeverity(e.target.value as string[])}
                  label="Severity"
                >
                  <MenuItem value="critical">Critical</MenuItem>
                  <MenuItem value="high">High</MenuItem>
                  <MenuItem value="medium">Medium</MenuItem>
                  <MenuItem value="low">Low</MenuItem>
                </Select>
              </FormControl>
            </Box>

            <Button
              variant="outlined"
              startIcon={<DownloadIcon />}
            >
              Export
            </Button>
          </Box>

          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Alert</TableCell>
                  <TableCell>Severity</TableCell>
                  <TableCell>Category</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Source</TableCell>
                  <TableCell>Assignee</TableCell>
                  <TableCell>Timestamp</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <AnimatePresence>
                  {filteredAlerts.map((alert) => (
                    <motion.tr
                      key={alert.id}
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: 10 }}
                      component={TableRow}
                      hover
                      onClick={() => {
                        setSelectedAlert(alert.id);
                        setAlertDetailsOpen(true);
                      }}
                      sx={{ cursor: 'pointer' }}
                    >
                      <TableCell>
                        <Box display="flex" alignItems="center">
                          <Avatar
                            sx={{ 
                              bgcolor: getSeverityColor(alert.severity),
                              width: 24,
                              height: 24,
                              mr: 2
                            }}
                          >
                            {getSeverityIcon(alert.severity)}
                          </Avatar>
                          <Box>
                            <Typography variant="body2" fontWeight="medium">
                              {alert.title}
                            </Typography>
                            <Typography variant="caption" color="textSecondary">
                              {alert.description}
                            </Typography>
                          </Box>
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={alert.severity.toUpperCase()}
                          size="small"
                          sx={{ 
                            bgcolor: getSeverityColor(alert.severity),
                            color: 'white',
                            fontWeight: 'bold'
                          }}
                        />
                      </TableCell>
                      <TableCell>
                        <Chip label={alert.category} size="small" variant="outlined" />
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={alert.status.replace('_', ' ')}
                          size="small"
                          color={getStatusColor(alert.status) as any}
                        />
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {alert.source}
                        </Typography>
                        <Typography variant="caption" color="textSecondary">
                          {alert.sourceIp}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {alert.assignee || 'Unassigned'}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {alert.timestamp.toLocaleString()}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Tooltip title="View Details">
                          <IconButton size="small">
                            <VisibilityIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Take Action">
                          <IconButton size="small">
                            <ActionIcon />
                          </IconButton>
                        </Tooltip>
                      </TableCell>
                    </motion.tr>
                  ))}
                </AnimatePresence>
              </TableBody>
            </Table>
          </TableContainer>
        </TabPanel>

        {/* Threat Indicators Tab */}
        <TabPanel value={tabValue} index={1}>
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Indicator</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell>Threat Type</TableCell>
                  <TableCell>Severity</TableCell>
                  <TableCell>Confidence</TableCell>
                  <TableCell>Sources</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Last Seen</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {indicators.map((indicator) => (
                  <TableRow key={indicator.id} hover>
                    <TableCell>
                      <Box>
                        <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                          {indicator.value}
                        </Typography>
                        <Typography variant="caption" color="textSecondary">
                          {indicator.description}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Chip label={indicator.type.toUpperCase()} size="small" />
                    </TableCell>
                    <TableCell>
                      <Chip 
                        label={indicator.threatType} 
                        size="small"
                        color="secondary"
                      />
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={indicator.severity}
                        size="small"
                        sx={{ bgcolor: getSeverityColor(indicator.severity), color: 'white' }}
                      />
                    </TableCell>
                    <TableCell>
                      <Box display="flex" alignItems="center">
                        <Typography variant="body2" sx={{ mr: 1 }}>
                          {indicator.confidence}%
                        </Typography>
                        <LinearProgress
                          variant="determinate"
                          value={indicator.confidence}
                          sx={{ width: 50 }}
                          color={indicator.confidence > 80 ? 'success' : 'warning'}
                        />
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Box display="flex" flexWrap="wrap" gap={0.5}>
                        {indicator.sources.slice(0, 2).map((source) => (
                          <Chip key={source} label={source} size="small" variant="outlined" />
                        ))}
                        {indicator.sources.length > 2 && (
                          <Chip label={`+${indicator.sources.length - 2}`} size="small" />
                        )}
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={indicator.status}
                        size="small"
                        color={indicator.status === 'active' ? 'error' : 'default'}
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2">
                        {indicator.lastSeen.toLocaleString()}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Tooltip title="Block Indicator">
                        <IconButton size="small">
                          <BlockIcon />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Whitelist">
                        <IconButton size="small">
                          <CheckCircleIcon />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </TabPanel>

        {/* Threat Feeds Tab */}
        <TabPanel value={tabValue} index={2}>
          <Grid container spacing={3}>
            {feeds.map((feed) => (
              <Grid item xs={12} md={6} key={feed.id}>
                <Card>
                  <CardHeader
                    title={feed.name}
                    subheader={`Provider: ${feed.provider}`}
                    action={
                      <Chip
                        label={feed.status}
                        size="small"
                        color={feed.status === 'active' ? 'success' : 'error'}
                      />
                    }
                  />
                  <CardContent>
                    <Box display="flex" justifyContent="space-between" mb={2}>
                      <Typography variant="body2" color="textSecondary">
                        Last Update
                      </Typography>
                      <Typography variant="body2">
                        {feed.lastUpdate.toLocaleString()}
                      </Typography>
                    </Box>
                    
                    <Box display="flex" justifyContent="space-between" mb={2}>
                      <Typography variant="body2" color="textSecondary">
                        Indicators
                      </Typography>
                      <Typography variant="body2">
                        {feed.indicatorCount.toLocaleString()}
                      </Typography>
                    </Box>
                    
                    <Box display="flex" justifyContent="space-between" mb={2}>
                      <Typography variant="body2" color="textSecondary">
                        Accuracy
                      </Typography>
                      <Box display="flex" alignItems="center">
                        <Typography variant="body2" sx={{ mr: 1 }}>
                          {feed.accuracy}%
                        </Typography>
                        <LinearProgress
                          variant="determinate"
                          value={feed.accuracy}
                          sx={{ width: 50 }}
                          color={feed.accuracy > 90 ? 'success' : 'warning'}
                        />
                      </Box>
                    </Box>
                    
                    <Box>
                      <Typography variant="body2" color="textSecondary" gutterBottom>
                        Coverage
                      </Typography>
                      <Box display="flex" flexWrap="wrap" gap={0.5}>
                        {feed.coverage.map((category) => (
                          <Chip key={category} label={category} size="small" variant="outlined" />
                        ))}
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </TabPanel>

        {/* Analytics Tab */}
        <TabPanel value={tabValue} index={3}>
          <Grid container spacing={3}>
            <Grid item xs={12} lg={6}>
              <Card>
                <CardHeader title="Alert Trends" />
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <ComposedChart
                      data={Array.from({ length: 7 }, (_, i) => ({
                        day: ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][i],
                        alerts: Math.floor(Math.random() * 50) + 20,
                        resolved: Math.floor(Math.random() * 40) + 15,
                        falsePositives: Math.floor(Math.random() * 10) + 2,
                      }))}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="day" />
                      <YAxis />
                      <RechartsTooltip />
                      <Bar dataKey="alerts" fill="#f44336" name="New Alerts" />
                      <Bar dataKey="resolved" fill="#4caf50" name="Resolved" />
                      <Line type="monotone" dataKey="falsePositives" stroke="#ff9800" name="False Positives" />
                    </ComposedChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12} lg={6}>
              <Card>
                <CardHeader title="Threat Distribution" />
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={[
                          { name: 'Malware', value: 35 },
                          { name: 'Phishing', value: 28 },
                          { name: 'C2', value: 18 },
                          { name: 'Botnet', value: 12 },
                          { name: 'Other', value: 7 },
                        ]}
                        cx="50%"
                        cy="50%"
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      >
                        {[35, 28, 18, 12, 7].map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={Object.values(SEVERITY_COLORS)[index]} />
                        ))}
                      </Pie>
                      <RechartsTooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12}>
              <Card>
                <CardHeader title="SOC Performance Metrics" />
                <CardContent>
                  <Grid container spacing={3}>
                    <Grid item xs={12} md={4}>
                      <Paper sx={{ p: 2, textAlign: 'center' }}>
                        <Typography variant="h4" color="primary">
                          {metrics.meanTimeToDetection}h
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          Mean Time to Detection
                        </Typography>
                      </Paper>
                    </Grid>
                    <Grid item xs={12} md={4}>
                      <Paper sx={{ p: 2, textAlign: 'center' }}>
                        <Typography variant="h4" color="warning.main">
                          {metrics.meanTimeToResponse}h
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          Mean Time to Response
                        </Typography>
                      </Paper>
                    </Grid>
                    <Grid item xs={12} md={4}>
                      <Paper sx={{ p: 2, textAlign: 'center' }}>
                        <Typography variant="h4" color="success.main">
                          {metrics.meanTimeToResolution}h
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          Mean Time to Resolution
                        </Typography>
                      </Paper>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>
      </Card>

      {/* Alert Details Dialog */}
      <Dialog
        open={alertDetailsOpen}
        onClose={() => setAlertDetailsOpen(false)}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>
          Alert Details
        </DialogTitle>
        <DialogContent>
          {selectedAlert && (() => {
            const alert = alerts.find(a => a.id === selectedAlert);
            if (!alert) return null;
            
            return (
              <Box>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={8}>
                    <Typography variant="h6" gutterBottom>
                      {alert.title}
                    </Typography>
                    <Typography variant="body1" paragraph>
                      {alert.description}
                    </Typography>
                    
                    <Accordion>
                      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                        <Typography>Timeline</Typography>
                      </AccordionSummary>
                      <AccordionDetails>
                        <List>
                          {alert.timeline.map((event, index) => (
                            <ListItem key={index}>
                              <ListItemText
                                primary={event.action}
                                secondary={`${event.user} - ${event.timestamp.toLocaleString()}`}
                              />
                            </ListItem>
                          ))}
                        </List>
                      </AccordionDetails>
                    </Accordion>
                  </Grid>
                  
                  <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 2 }}>
                      <Typography variant="h6" gutterBottom>
                        Alert Properties
                      </Typography>
                      <List dense>
                        <ListItem>
                          <ListItemText primary="Severity" secondary={alert.severity} />
                        </ListItem>
                        <ListItem>
                          <ListItemText primary="Status" secondary={alert.status} />
                        </ListItem>
                        <ListItem>
                          <ListItemText primary="Category" secondary={alert.category} />
                        </ListItem>
                        <ListItem>
                          <ListItemText primary="Source" secondary={alert.source} />
                        </ListItem>
                        <ListItem>
                          <ListItemText primary="Source IP" secondary={alert.sourceIp} />
                        </ListItem>
                        {alert.targetIp && (
                          <ListItem>
                            <ListItemText primary="Target IP" secondary={alert.targetIp} />
                          </ListItem>
                        )}
                        <ListItem>
                          <ListItemText primary="Assignee" secondary={alert.assignee || 'Unassigned'} />
                        </ListItem>
                      </List>
                    </Paper>
                  </Grid>
                </Grid>
              </Box>
            );
          })()}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAlertDetailsOpen(false)}>
            Close
          </Button>
          <Button variant="contained" color="primary">
            Take Action
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default ThreatIntelligence;