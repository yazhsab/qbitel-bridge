import React, { useEffect, useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  CircularProgress,
  Alert,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  LinearProgress,
} from '@mui/material';
import {
  Devices as DevicesIcon,
  Security as SecurityIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
} from '@mui/icons-material';
import { DeviceApiClient } from '../api/devices';
import { DeviceMetrics, DeviceAlert } from '../types/device';

interface DashboardProps {
  apiClient: DeviceApiClient;
}

interface DashboardState {
  metrics: DeviceMetrics | null;
  alerts: DeviceAlert[];
  loading: boolean;
  error: string | null;
}

const Dashboard: React.FC<DashboardProps> = ({ apiClient }) => {
  const [state, setState] = useState<DashboardState>({
    metrics: null,
    alerts: [],
    loading: true,
    error: null,
  });

  useEffect(() => {
    loadDashboardData();
    const interval = setInterval(loadDashboardData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const loadDashboardData = async () => {
    try {
      setState(prev => ({ ...prev, loading: true, error: null }));
      
      const [metricsResponse, alertsData] = await Promise.all([
        apiClient.getDeviceMetrics(),
        apiClient.getDeviceAlerts(undefined, ['warning', 'error', 'critical'], undefined, false),
      ]);

      setState(prev => ({
        ...prev,
        metrics: metricsResponse.metrics,
        alerts: alertsData.slice(0, 10), // Show only top 10 alerts
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
  };

  const getAlertIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
        return <ErrorIcon color="error" />;
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
        return 'error';
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
      <Typography variant="h4" component="h1" gutterBottom>
        Dashboard
      </Typography>
      
      {state.loading && (
        <LinearProgress sx={{ mb: 2 }} />
      )}

      <Grid container spacing={3}>
        {/* Device Overview Cards */}
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
                    Active Devices
                  </Typography>
                  <Typography variant="h4" color="success.main">
                    {metrics.active_devices}
                  </Typography>
                </Box>
                <CheckCircleIcon color="success" sx={{ fontSize: 40 }} />
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
                    Compliance Rate
                  </Typography>
                  <Typography variant="h4" color={complianceRate >= 95 ? "success.main" : complianceRate >= 80 ? "warning.main" : "error.main"}>
                    {complianceRate}%
                  </Typography>
                </Box>
                <SecurityIcon color={complianceRate >= 95 ? "success" : complianceRate >= 80 ? "warning" : "error"} sx={{ fontSize: 40 }} />
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
                    Health Rate
                  </Typography>
                  <Typography variant="h4" color={healthRate >= 95 ? "success.main" : healthRate >= 80 ? "warning.main" : "error.main"}>
                    {healthRate}%
                  </Typography>
                </Box>
                {healthRate >= 95 ? (
                  <TrendingUpIcon color="success" sx={{ fontSize: 40 }} />
                ) : (
                  <TrendingDownIcon color="error" sx={{ fontSize: 40 }} />
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Device Status Breakdown */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Device Status Breakdown
              </Typography>
              <Box mt={2}>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                  <Typography variant="body2">Active</Typography>
                  <Typography variant="body2">{metrics.active_devices}</Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={(metrics.active_devices / metrics.total_devices) * 100}
                  color="success"
                  sx={{ mb: 2 }}
                />

                <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                  <Typography variant="body2">Inactive</Typography>
                  <Typography variant="body2">{metrics.inactive_devices}</Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={(metrics.inactive_devices / metrics.total_devices) * 100}
                  color="warning"
                  sx={{ mb: 2 }}
                />

                <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                  <Typography variant="body2">Suspended</Typography>
                  <Typography variant="body2">{metrics.suspended_devices}</Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={(metrics.suspended_devices / metrics.total_devices) * 100}
                  color="error"
                  sx={{ mb: 2 }}
                />

                <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                  <Typography variant="body2">Decommissioned</Typography>
                  <Typography variant="body2">{metrics.decommissioned_devices}</Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={(metrics.decommissioned_devices / metrics.total_devices) * 100}
                  color="info"
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Device Types */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Device Types
              </Typography>
              <Box mt={2}>
                {Object.entries(metrics.devices_by_type).map(([type, count]) => (
                  <Box key={type} display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                    <Typography variant="body2" sx={{ textTransform: 'capitalize' }}>
                      {type}
                    </Typography>
                    <Chip label={count} size="small" color="primary" />
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Alerts */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Alerts
              </Typography>
              {state.alerts.length === 0 ? (
                <Typography color="textSecondary">
                  No active alerts
                </Typography>
              ) : (
                <List>
                  {state.alerts.map((alert) => (
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
                            <Box display="flex" alignItems="center" gap={1} mt={1}>
                              <Chip
                                label={alert.severity}
                                size="small"
                                color={getAlertColor(alert.severity) as any}
                              />
                              <Chip
                                label={alert.alert_type}
                                size="small"
                                variant="outlined"
                              />
                              <Typography variant="caption" color="textSecondary">
                                {new Date(alert.created_at).toLocaleString()}
                              </Typography>
                            </Box>
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

        {/* Quick Stats */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Quick Statistics
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6} md={3}>
                <Box textAlign="center">
                  <Typography variant="h4" color="primary">
                    {metrics.enrollment_rate_24h}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Enrollments (24h)
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Box textAlign="center">
                  <Typography variant="h4" color="success.main">
                    {metrics.compliant_devices}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Compliant Devices
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Box textAlign="center">
                  <Typography variant="h4" color="error.main">
                    {metrics.non_compliant_devices}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Non-Compliant
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Box textAlign="center">
                  <Typography variant="h4" color="success.main">
                    {metrics.healthy_devices}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Healthy Devices
                  </Typography>
                </Box>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;