import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  IconButton,
  Menu,
  MenuItem,
  Alert,
  CircularProgress,
  LinearProgress,
} from '@mui/material';
import {
  Add as AddIcon,
  MoreVert as MoreVertIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Visibility as VisibilityIcon,
  TrendingUp as TrendingUpIcon,
  AttachMoney as MoneyIcon,
  Download as DownloadIcon,
  People as PeopleIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { MarketplaceApiClient } from '../../api/marketplace';
import { Protocol, CERTIFICATION_BADGES } from '../../types/marketplace';

interface CreatorDashboardProps {
  apiClient: MarketplaceApiClient;
}

const CreatorDashboard: React.FC<CreatorDashboardProps> = ({ apiClient }) => {
  const navigate = useNavigate();

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [protocols, setProtocols] = useState<Protocol[]>([]);
  const [analytics, setAnalytics] = useState<any>(null);
  const [menuAnchor, setMenuAnchor] = useState<HTMLElement | null>(null);
  const [selectedProtocol, setSelectedProtocol] = useState<Protocol | null>(null);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);

      const [protocolsData, analyticsData] = await Promise.all([
        apiClient.getMyProtocols(),
        apiClient.getCreatorAnalytics(),
      ]);

      setProtocols(protocolsData);
      setAnalytics(analyticsData);
    } catch (err: any) {
      console.error('Failed to load dashboard data:', err);
      setError(err.message || 'Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, protocol: Protocol) => {
    setMenuAnchor(event.currentTarget);
    setSelectedProtocol(protocol);
  };

  const handleMenuClose = () => {
    setMenuAnchor(null);
    setSelectedProtocol(null);
  };

  const handleViewProtocol = () => {
    if (selectedProtocol) {
      navigate(`/marketplace/protocols/${selectedProtocol.protocol_id}`);
    }
    handleMenuClose();
  };

  const handleEditProtocol = () => {
    if (selectedProtocol) {
      navigate(`/marketplace/protocols/${selectedProtocol.protocol_id}/edit`);
    }
    handleMenuClose();
  };

  const handleDeleteProtocol = async () => {
    if (!selectedProtocol) return;

    if (window.confirm(`Are you sure you want to delete ${selectedProtocol.display_name}?`)) {
      try {
        await apiClient.deleteProtocol(selectedProtocol.protocol_id);
        await loadDashboardData();
      } catch (err: any) {
        setError(err.message || 'Failed to delete protocol');
      }
    }
    handleMenuClose();
  };

  if (loading) {
    return (
      <Container maxWidth="xl" sx={{ py: 4 }}>
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
          <CircularProgress size={48} />
        </Box>
      </Container>
    );
  }

  const totalRevenue = analytics?.total_revenue || 0;
  const totalDownloads = analytics?.total_downloads || 0;
  const activeInstallations = analytics?.active_installations || 0;
  const totalProtocols = protocols.length;

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={4}>
        <Box>
          <Typography variant="h4" component="h1" gutterBottom fontWeight="bold">
            Creator Dashboard
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Manage your protocols and track performance
          </Typography>
        </Box>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => navigate('/marketplace/submit')}
          size="large"
        >
          Submit New Protocol
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Metrics Cards */}
      <Grid container spacing={3} mb={4}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={1}>
                <Box
                  sx={{
                    bgcolor: 'primary.light',
                    p: 1,
                    borderRadius: 1,
                    mr: 2,
                  }}
                >
                  <TrendingUpIcon color="primary" />
                </Box>
                <Box>
                  <Typography variant="body2" color="text.secondary">
                    Total Protocols
                  </Typography>
                  <Typography variant="h4" fontWeight="bold">
                    {totalProtocols}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={1}>
                <Box
                  sx={{
                    bgcolor: 'success.light',
                    p: 1,
                    borderRadius: 1,
                    mr: 2,
                  }}
                >
                  <MoneyIcon color="success" />
                </Box>
                <Box>
                  <Typography variant="body2" color="text.secondary">
                    Total Revenue
                  </Typography>
                  <Typography variant="h4" fontWeight="bold">
                    ${totalRevenue.toLocaleString()}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={1}>
                <Box
                  sx={{
                    bgcolor: 'info.light',
                    p: 1,
                    borderRadius: 1,
                    mr: 2,
                  }}
                >
                  <DownloadIcon color="info" />
                </Box>
                <Box>
                  <Typography variant="body2" color="text.secondary">
                    Total Downloads
                  </Typography>
                  <Typography variant="h4" fontWeight="bold">
                    {totalDownloads.toLocaleString()}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={1}>
                <Box
                  sx={{
                    bgcolor: 'warning.light',
                    p: 1,
                    borderRadius: 1,
                    mr: 2,
                  }}
                >
                  <PeopleIcon color="warning" />
                </Box>
                <Box>
                  <Typography variant="body2" color="text.secondary">
                    Active Installations
                  </Typography>
                  <Typography variant="h4" fontWeight="bold">
                    {activeInstallations.toLocaleString()}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Revenue Chart */}
      {analytics?.downloads_trend && analytics.downloads_trend.length > 0 && (
        <Paper sx={{ p: 3, mb: 4 }}>
          <Typography variant="h6" gutterBottom fontWeight="600">
            Downloads Trend (Last 30 Days)
          </Typography>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={analytics.downloads_trend}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="count" stroke="#0066CC" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </Paper>
      )}

      {/* Protocols Table */}
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom fontWeight="600">
          My Protocols ({totalProtocols})
        </Typography>

        {protocols.length === 0 ? (
          <Box textAlign="center" py={4}>
            <Typography variant="body1" color="text.secondary" gutterBottom>
              You haven't submitted any protocols yet
            </Typography>
            <Button
              variant="outlined"
              startIcon={<AddIcon />}
              onClick={() => navigate('/marketplace/submit')}
              sx={{ mt: 2 }}
            >
              Submit Your First Protocol
            </Button>
          </Box>
        ) : (
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Protocol</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell align="right">Downloads</TableCell>
                  <TableCell align="right">Installations</TableCell>
                  <TableCell align="right">Rating</TableCell>
                  <TableCell align="right">Revenue</TableCell>
                  <TableCell align="right">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {protocols.map((protocol) => {
                  const certBadge = CERTIFICATION_BADGES[protocol.quality_metrics.certification_status];
                  const revenue = analytics?.revenue_by_protocol?.[protocol.protocol_id] || 0;

                  return (
                    <TableRow key={protocol.protocol_id} hover>
                      <TableCell>
                        <Box>
                          <Typography variant="body2" fontWeight="600">
                            {protocol.display_name}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            v{protocol.version}
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={certBadge.label}
                          size="small"
                          sx={{
                            bgcolor: `${certBadge.color}20`,
                            color: certBadge.color,
                            fontWeight: 600,
                            borderColor: certBadge.color,
                            border: '1px solid',
                          }}
                        />
                      </TableCell>
                      <TableCell align="right">
                        {protocol.quality_metrics.download_count.toLocaleString()}
                      </TableCell>
                      <TableCell align="right">
                        {protocol.quality_metrics.active_installations.toLocaleString()}
                      </TableCell>
                      <TableCell align="right">
                        <Box display="flex" alignItems="center" justifyContent="flex-end">
                          ⭐ {protocol.quality_metrics.average_rating.toFixed(1)}
                          <Typography variant="caption" color="text.secondary" ml={0.5}>
                            ({protocol.quality_metrics.total_ratings})
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell align="right">
                        <Typography variant="body2" fontWeight="600">
                          ${revenue.toLocaleString()}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        <IconButton
                          size="small"
                          onClick={(e) => handleMenuOpen(e, protocol)}
                        >
                          <MoreVertIcon />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </Paper>

      {/* Context Menu */}
      <Menu
        anchorEl={menuAnchor}
        open={Boolean(menuAnchor)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={handleViewProtocol}>
          <VisibilityIcon fontSize="small" sx={{ mr: 1 }} />
          View Details
        </MenuItem>
        <MenuItem onClick={handleEditProtocol}>
          <EditIcon fontSize="small" sx={{ mr: 1 }} />
          Edit
        </MenuItem>
        <MenuItem onClick={handleDeleteProtocol} sx={{ color: 'error.main' }}>
          <DeleteIcon fontSize="small" sx={{ mr: 1 }} />
          Delete
        </MenuItem>
      </Menu>
    </Container>
  );
};

export default CreatorDashboard;
