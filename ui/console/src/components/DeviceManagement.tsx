import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Button,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Grid,
  Alert,
  CircularProgress,
  IconButton,
  Tooltip,
  Badge,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Pause as PauseIcon,
  PlayArrow as PlayIcon,
  Security as SecurityIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
} from '@mui/icons-material';
import { useAuth } from '../auth/AuthContext';
import { deviceApi } from '../api/deviceApi';
import { Device, DeviceStatus, DeviceType, ComplianceStatus, HealthStatus } from '../types/device';

interface DeviceManagementProps {
  organizationId: string;
}

const DeviceManagement: React.FC<DeviceManagementProps> = ({ organizationId }) => {
  const { user, hasPermission } = useAuth();
  const [devices, setDevices] = useState<Device[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedDevice, setSelectedDevice] = useState<Device | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [dialogType, setDialogType] = useState<'view' | 'edit' | 'suspend' | 'decommission'>('view');
  const [suspendReason, setSuspendReason] = useState('');
  const [decommissionReason, setDecommissionReason] = useState('');
  const [filters, setFilters] = useState({
    status: '',
    deviceType: '',
    complianceStatus: '',
    healthStatus: '',
  });

  const loadDevices = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await deviceApi.listDevices(organizationId, filters);
      setDevices(response.devices);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load devices');
    } finally {
      setLoading(false);
    }
  }, [organizationId, filters]);

  useEffect(() => {
    loadDevices();
  }, [loadDevices]);

  const handleDeviceAction = async (device: Device, action: string) => {
    try {
      switch (action) {
        case 'suspend':
          if (!suspendReason.trim()) {
            setError('Suspension reason is required');
            return;
          }
          await deviceApi.suspendDevice(device.id, suspendReason);
          break;
        case 'reactivate':
          await deviceApi.reactivateDevice(device.id);
          break;
        case 'decommission':
          if (!decommissionReason.trim()) {
            setError('Decommission reason is required');
            return;
          }
          await deviceApi.decommissionDevice(device.id, decommissionReason);
          break;
        default:
          throw new Error(`Unknown action: ${action}`);
      }
      
      setDialogOpen(false);
      setSuspendReason('');
      setDecommissionReason('');
      await loadDevices();
    } catch (err) {
      setError(err instanceof Error ? err.message : `Failed to ${action} device`);
    }
  };

  const getStatusColor = (status: DeviceStatus) => {
    switch (status) {
      case 'active': return 'success';
      case 'inactive': return 'warning';
      case 'suspended': return 'error';
      case 'decommissioned': return 'default';
      case 'enrolling': return 'info';
      case 'pending': return 'warning';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  const getComplianceIcon = (status: ComplianceStatus) => {
    switch (status) {
      case 'compliant': return <CheckCircleIcon color="success" />;
      case 'non_compliant': return <ErrorIcon color="error" />;
      case 'checking': return <CircularProgress size={20} />;
      default: return <WarningIcon color="warning" />;
    }
  };

  const getHealthIcon = (status: HealthStatus) => {
    switch (status) {
      case 'healthy': return <CheckCircleIcon color="success" />;
      case 'unhealthy': return <ErrorIcon color="error" />;
      case 'checking': return <CircularProgress size={20} />;
      default: return <WarningIcon color="warning" />;
    }
  };

  const canPerformAction = (action: string) => {
    switch (action) {
      case 'view': return hasPermission('device:read');
      case 'edit': return hasPermission('device:write');
      case 'suspend': return hasPermission('device:suspend');
      case 'reactivate': return hasPermission('device:activate');
      case 'decommission': return hasPermission('device:decommission');
      default: return false;
    }
  };

  const openDialog = (device: Device, type: typeof dialogType) => {
    setSelectedDevice(device);
    setDialogType(type);
    setDialogOpen(true);
  };

  const closeDialog = () => {
    setDialogOpen(false);
    setSelectedDevice(null);
    setSuspendReason('');
    setDecommissionReason('');
    setError(null);
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1">
          Device Management
        </Typography>
        <Box>
          <Button
            startIcon={<RefreshIcon />}
            onClick={loadDevices}
            sx={{ mr: 1 }}
          >
            Refresh
          </Button>
          {canPerformAction('edit') && (
            <Button
              variant="contained"
              startIcon={<AddIcon />}
              onClick={() => {/* Handle add device */}}
            >
              Add Device
            </Button>
          )}
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Filters */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Filters
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Status</InputLabel>
                <Select
                  value={filters.status}
                  label="Status"
                  onChange={(e) => setFilters({ ...filters, status: e.target.value })}
                >
                  <MenuItem value="">All</MenuItem>
                  <MenuItem value="active">Active</MenuItem>
                  <MenuItem value="inactive">Inactive</MenuItem>
                  <MenuItem value="suspended">Suspended</MenuItem>
                  <MenuItem value="decommissioned">Decommissioned</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Device Type</InputLabel>
                <Select
                  value={filters.deviceType}
                  label="Device Type"
                  onChange={(e) => setFilters({ ...filters, deviceType: e.target.value })}
                >
                  <MenuItem value="">All</MenuItem>
                  <MenuItem value="gateway">Gateway</MenuItem>
                  <MenuItem value="endpoint">Endpoint</MenuItem>
                  <MenuItem value="sensor">Sensor</MenuItem>
                  <MenuItem value="actuator">Actuator</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Compliance</InputLabel>
                <Select
                  value={filters.complianceStatus}
                  label="Compliance"
                  onChange={(e) => setFilters({ ...filters, complianceStatus: e.target.value })}
                >
                  <MenuItem value="">All</MenuItem>
                  <MenuItem value="compliant">Compliant</MenuItem>
                  <MenuItem value="non_compliant">Non-Compliant</MenuItem>
                  <MenuItem value="unknown">Unknown</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Health</InputLabel>
                <Select
                  value={filters.healthStatus}
                  label="Health"
                  onChange={(e) => setFilters({ ...filters, healthStatus: e.target.value })}
                >
                  <MenuItem value="">All</MenuItem>
                  <MenuItem value="healthy">Healthy</MenuItem>
                  <MenuItem value="unhealthy">Unhealthy</MenuItem>
                  <MenuItem value="unknown">Unknown</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Device Table */}
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Device Name</TableCell>
              <TableCell>Type</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Compliance</TableCell>
              <TableCell>Health</TableCell>
              <TableCell>Last Seen</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {devices.map((device) => (
              <TableRow key={device.id} hover>
                <TableCell>
                  <Box>
                    <Typography variant="body2" fontWeight="bold">
                      {device.name}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {device.id}
                    </Typography>
                  </Box>
                </TableCell>
                <TableCell>
                  <Chip
                    label={device.device_type}
                    size="small"
                    variant="outlined"
                  />
                </TableCell>
                <TableCell>
                  <Chip
                    label={device.status}
                    color={getStatusColor(device.status)}
                    size="small"
                  />
                </TableCell>
                <TableCell>
                  <Box display="flex" alignItems="center" gap={1}>
                    {getComplianceIcon(device.compliance_status)}
                    <Typography variant="caption">
                      {device.compliance_status}
                    </Typography>
                  </Box>
                </TableCell>
                <TableCell>
                  <Box display="flex" alignItems="center" gap={1}>
                    {getHealthIcon(device.health_status)}
                    <Typography variant="caption">
                      {device.health_status}
                    </Typography>
                  </Box>
                </TableCell>
                <TableCell>
                  <Typography variant="caption">
                    {new Date(device.last_seen).toLocaleString()}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Box display="flex" gap={1}>
                    {canPerformAction('view') && (
                      <Tooltip title="View Details">
                        <IconButton
                          size="small"
                          onClick={() => openDialog(device, 'view')}
                        >
                          <SecurityIcon />
                        </IconButton>
                      </Tooltip>
                    )}
                    {canPerformAction('edit') && (
                      <Tooltip title="Edit">
                        <IconButton
                          size="small"
                          onClick={() => openDialog(device, 'edit')}
                        >
                          <EditIcon />
                        </IconButton>
                      </Tooltip>
                    )}
                    {device.status === 'active' && canPerformAction('suspend') && (
                      <Tooltip title="Suspend">
                        <IconButton
                          size="small"
                          onClick={() => openDialog(device, 'suspend')}
                        >
                          <PauseIcon />
                        </IconButton>
                      </Tooltip>
                    )}
                    {device.status === 'suspended' && canPerformAction('reactivate') && (
                      <Tooltip title="Reactivate">
                        <IconButton
                          size="small"
                          onClick={() => handleDeviceAction(device, 'reactivate')}
                        >
                          <PlayIcon />
                        </IconButton>
                      </Tooltip>
                    )}
                    {canPerformAction('decommission') && device.status !== 'decommissioned' && (
                      <Tooltip title="Decommission">
                        <IconButton
                          size="small"
                          color="error"
                          onClick={() => openDialog(device, 'decommission')}
                        >
                          <DeleteIcon />
                        </IconButton>
                      </Tooltip>
                    )}
                  </Box>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Device Details Dialog */}
      <Dialog
        open={dialogOpen}
        onClose={closeDialog}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          {dialogType === 'view' && 'Device Details'}
          {dialogType === 'edit' && 'Edit Device'}
          {dialogType === 'suspend' && 'Suspend Device'}
          {dialogType === 'decommission' && 'Decommission Device'}
        </DialogTitle>
        <DialogContent>
          {selectedDevice && (
            <Box>
              {dialogType === 'view' && (
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="subtitle2">Device ID</Typography>
                    <Typography variant="body2" gutterBottom>
                      {selectedDevice.id}
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="subtitle2">Name</Typography>
                    <Typography variant="body2" gutterBottom>
                      {selectedDevice.name}
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="subtitle2">Type</Typography>
                    <Typography variant="body2" gutterBottom>
                      {selectedDevice.device_type}
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="subtitle2">Manufacturer</Typography>
                    <Typography variant="body2" gutterBottom>
                      {selectedDevice.manufacturer}
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="subtitle2">Model</Typography>
                    <Typography variant="body2" gutterBottom>
                      {selectedDevice.model}
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="subtitle2">Serial Number</Typography>
                    <Typography variant="body2" gutterBottom>
                      {selectedDevice.serial_number}
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="subtitle2">Firmware Version</Typography>
                    <Typography variant="body2" gutterBottom>
                      {selectedDevice.firmware_version}
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Typography variant="subtitle2">Enrolled At</Typography>
                    <Typography variant="body2" gutterBottom>
                      {new Date(selectedDevice.enrolled_at).toLocaleString()}
                    </Typography>
                  </Grid>
                  <Grid item xs={12}>
                    <Typography variant="subtitle2">Capabilities</Typography>
                    <Box display="flex" gap={1} flexWrap="wrap" mt={1}>
                      {selectedDevice.capabilities.map((capability) => (
                        <Chip
                          key={capability}
                          label={capability}
                          size="small"
                          variant="outlined"
                        />
                      ))}
                    </Box>
                  </Grid>
                </Grid>
              )}

              {dialogType === 'suspend' && (
                <TextField
                  fullWidth
                  multiline
                  rows={3}
                  label="Suspension Reason"
                  value={suspendReason}
                  onChange={(e) => setSuspendReason(e.target.value)}
                  required
                />
              )}

              {dialogType === 'decommission' && (
                <TextField
                  fullWidth
                  multiline
                  rows={3}
                  label="Decommission Reason"
                  value={decommissionReason}
                  onChange={(e) => setDecommissionReason(e.target.value)}
                  required
                />
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={closeDialog}>
            Cancel
          </Button>
          {dialogType === 'suspend' && (
            <Button
              variant="contained"
              color="warning"
              onClick={() => selectedDevice && handleDeviceAction(selectedDevice, 'suspend')}
            >
              Suspend Device
            </Button>
          )}
          {dialogType === 'decommission' && (
            <Button
              variant="contained"
              color="error"
              onClick={() => selectedDevice && handleDeviceAction(selectedDevice, 'decommission')}
            >
              Decommission Device
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default DeviceManagement;