import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  Chip,
  Alert,
  CircularProgress,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  Divider,
  LinearProgress,
} from '@mui/material';
import {
  Download as DownloadIcon,
  Refresh as RefreshIcon,
  Info as InfoIcon,
  CloudDownload as CloudDownloadIcon,
  Cancel as CancelIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { MarketplaceApiClient } from '../../api/marketplace';
import { Installation } from '../../types/marketplace';

interface MyProtocolsProps {
  apiClient: MarketplaceApiClient;
}

const MyProtocols: React.FC<MyProtocolsProps> = ({ apiClient }) => {
  const navigate = useNavigate();

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [installations, setInstallations] = useState<Installation[]>([]);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [selectedInstallation, setSelectedInstallation] = useState<Installation | null>(null);

  useEffect(() => {
    loadInstallations();
  }, []);

  const loadInstallations = async () => {
    try {
      setLoading(true);
      setError(null);

      const data = await apiClient.getMyInstallations();
      setInstallations(data);
    } catch (err: any) {
      console.error('Failed to load installations:', err);
      setError(err.message || 'Failed to load installed protocols');
    } finally {
      setLoading(false);
    }
  };

  const handleViewDetails = (installation: Installation) => {
    setSelectedInstallation(installation);
    setDetailsOpen(true);
  };

  const handleDownloadFile = async (url: string, filename: string) => {
    try {
      const response = await fetch(url);
      const blob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = downloadUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(downloadUrl);
    } catch (err) {
      console.error('Download failed:', err);
      alert('Download failed. Please try again.');
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'success';
      case 'suspended':
        return 'warning';
      case 'expired':
        return 'error';
      default:
        return 'default';
    }
  };

  const isExpiringSoon = (expiresAt?: string) => {
    if (!expiresAt) return false;
    const daysUntilExpiry = Math.ceil(
      (new Date(expiresAt).getTime() - Date.now()) / (1000 * 60 * 60 * 24)
    );
    return daysUntilExpiry > 0 && daysUntilExpiry <= 7;
  };

  if (loading) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
          <CircularProgress size={48} />
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={4}>
        <Box>
          <Typography variant="h4" component="h1" gutterBottom fontWeight="bold">
            My Protocols
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Manage your installed protocols and licenses
          </Typography>
        </Box>
        <Box>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={loadInstallations}
            sx={{ mr: 2 }}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            onClick={() => navigate('/marketplace')}
          >
            Browse Marketplace
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {installations.length === 0 ? (
        <Card sx={{ textAlign: 'center', py: 6 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              No Installed Protocols
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              You haven't installed any protocols yet. Browse the marketplace to get started.
            </Typography>
            <Button
              variant="contained"
              onClick={() => navigate('/marketplace')}
              sx={{ mt: 2 }}
            >
              Browse Marketplace
            </Button>
          </CardContent>
        </Card>
      ) : (
        <Grid container spacing={3}>
          {installations.map((installation) => {
            const protocol = installation.protocol;
            const expiringSOon = isExpiringSoon(installation.expires_at);

            return (
              <Grid item xs={12} sm={6} md={4} key={installation.installation_id}>
                <Card
                  sx={{
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    position: 'relative',
                  }}
                >
                  <CardContent sx={{ flexGrow: 1 }}>
                    {/* Status Badge */}
                    <Box position="absolute" top={12} right={12}>
                      <Chip
                        label={installation.status}
                        size="small"
                        color={getStatusColor(installation.status) as any}
                      />
                    </Box>

                    {/* Protocol Info */}
                    <Typography variant="h6" component="div" gutterBottom fontWeight="600">
                      {protocol?.display_name || 'Unknown Protocol'}
                    </Typography>

                    <Typography variant="body2" color="text.secondary" paragraph>
                      Version: {installation.installed_version}
                    </Typography>

                    {/* License Info */}
                    <Box sx={{ mt: 2, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
                      <Typography variant="caption" color="text.secondary" display="block">
                        License Key
                      </Typography>
                      <Typography
                        variant="body2"
                        sx={{
                          fontFamily: 'monospace',
                          fontSize: '0.75rem',
                          wordBreak: 'break-all',
                        }}
                      >
                        {installation.license_key}
                      </Typography>
                    </Box>

                    {/* Expiry Warning */}
                    {expiringSOon && (
                      <Alert severity="warning" sx={{ mt: 2 }}>
                        <Typography variant="caption">
                          License expires in{' '}
                          {Math.ceil(
                            (new Date(installation.expires_at!).getTime() - Date.now()) /
                              (1000 * 60 * 60 * 24)
                          )}{' '}
                          days
                        </Typography>
                      </Alert>
                    )}

                    {/* Usage Stats */}
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="caption" color="text.secondary" display="block">
                        Packets Processed
                      </Typography>
                      <Typography variant="h6" fontWeight="600">
                        {installation.total_packets_processed.toLocaleString()}
                      </Typography>
                    </Box>

                    {installation.last_used_at && (
                      <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
                        Last used: {new Date(installation.last_used_at).toLocaleDateString()}
                      </Typography>
                    )}
                  </CardContent>

                  <CardActions sx={{ justifyContent: 'space-between', px: 2, pb: 2 }}>
                    <Button
                      size="small"
                      startIcon={<InfoIcon />}
                      onClick={() => handleViewDetails(installation)}
                    >
                      Details
                    </Button>
                    <Button
                      size="small"
                      variant="contained"
                      startIcon={<CloudDownloadIcon />}
                      onClick={() => handleViewDetails(installation)}
                    >
                      Files
                    </Button>
                  </CardActions>
                </Card>
              </Grid>
            );
          })}
        </Grid>
      )}

      {/* Details Dialog */}
      <Dialog
        open={detailsOpen}
        onClose={() => setDetailsOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          Protocol Details
          {selectedInstallation && (
            <Chip
              label={selectedInstallation.status}
              size="small"
              color={getStatusColor(selectedInstallation.status) as any}
              sx={{ ml: 2 }}
            />
          )}
        </DialogTitle>
        <DialogContent>
          {selectedInstallation && (
            <Box>
              <Typography variant="h6" gutterBottom>
                {selectedInstallation.protocol?.display_name}
              </Typography>

              <Divider sx={{ my: 2 }} />

              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    Installation ID:
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" fontFamily="monospace" fontSize="0.75rem">
                    {selectedInstallation.installation_id}
                  </Typography>
                </Grid>

                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    License Key:
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" fontFamily="monospace" fontSize="0.75rem">
                    {selectedInstallation.license_key}
                  </Typography>
                </Grid>

                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    Version:
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2">{selectedInstallation.installed_version}</Typography>
                </Grid>

                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    License Type:
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2">{selectedInstallation.license_type}</Typography>
                </Grid>

                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    Installed:
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2">
                    {new Date(selectedInstallation.installation_date).toLocaleDateString()}
                  </Typography>
                </Grid>

                {selectedInstallation.expires_at && (
                  <>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">
                        Expires:
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2">
                        {new Date(selectedInstallation.expires_at).toLocaleDateString()}
                      </Typography>
                    </Grid>
                  </>
                )}

                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    Packets Processed:
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2">
                    {selectedInstallation.total_packets_processed.toLocaleString()}
                  </Typography>
                </Grid>
              </Grid>

              <Divider sx={{ my: 2 }} />

              <Typography variant="subtitle2" gutterBottom fontWeight="600">
                Download Files:
              </Typography>
              <List dense>
                <ListItem
                  button
                  onClick={() =>
                    handleDownloadFile(
                      `${window.location.origin}/api/v1/marketplace/download/${selectedInstallation.installation_id}/spec`,
                      'protocol-spec.yaml'
                    )
                  }
                >
                  <DownloadIcon sx={{ mr: 1 }} fontSize="small" />
                  <ListItemText primary="Protocol Specification" />
                </ListItem>
                <ListItem
                  button
                  onClick={() =>
                    handleDownloadFile(
                      `${window.location.origin}/api/v1/marketplace/download/${selectedInstallation.installation_id}/parser`,
                      'parser.py'
                    )
                  }
                >
                  <DownloadIcon sx={{ mr: 1 }} fontSize="small" />
                  <ListItemText primary="Parser Code" />
                </ListItem>
                <ListItem
                  button
                  onClick={() =>
                    handleDownloadFile(
                      `${window.location.origin}/api/v1/marketplace/download/${selectedInstallation.installation_id}/docs`,
                      'documentation.pdf'
                    )
                  }
                >
                  <DownloadIcon sx={{ mr: 1 }} fontSize="small" />
                  <ListItemText primary="Documentation" />
                </ListItem>
              </List>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDetailsOpen(false)}>Close</Button>
          {selectedInstallation?.protocol && (
            <Button
              variant="contained"
              onClick={() => {
                navigate(`/marketplace/protocols/${selectedInstallation.protocol_id}`);
                setDetailsOpen(false);
              }}
            >
              View in Marketplace
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default MyProtocols;
