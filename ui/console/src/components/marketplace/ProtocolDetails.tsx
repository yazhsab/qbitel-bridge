import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Button,
  Paper,
  Grid,
  Chip,
  Rating,
  Divider,
  Tab,
  Tabs,
  Avatar,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Card,
  CardContent,
  Alert,
  CircularProgress,
  Breadcrumbs,
  Link,
  Skeleton,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  ArrowBack as ArrowBackIcon,
  ShoppingCart as ShoppingCartIcon,
  Download as DownloadIcon,
  Verified as VerifiedIcon,
  CheckCircle as CheckCircleIcon,
  Share as ShareIcon,
  Report as ReportIcon,
  Star as StarIcon,
} from '@mui/icons-material';
import { useParams, useNavigate } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import { MarketplaceApiClient } from '../../api/marketplace';
import { Protocol, Review, CERTIFICATION_BADGES } from '../../types/marketplace';
import PurchaseModal from './PurchaseModal';
import ReviewsSection from './ReviewsSection';

interface ProtocolDetailsProps {
  apiClient: MarketplaceApiClient;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`protocol-tabpanel-${index}`}
      aria-labelledby={`protocol-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

const ProtocolDetails: React.FC<ProtocolDetailsProps> = ({ apiClient }) => {
  const { protocolId } = useParams<{ protocolId: string }>();
  const navigate = useNavigate();

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [protocol, setProtocol] = useState<Protocol | null>(null);
  const [reviews, setReviews] = useState<Review[]>([]);
  const [activeTab, setActiveTab] = useState(0);
  const [purchaseModalOpen, setPurchaseModalOpen] = useState(false);

  useEffect(() => {
    if (protocolId) {
      loadProtocolDetails();
    }
  }, [protocolId]);

  const loadProtocolDetails = async () => {
    try {
      setLoading(true);
      setError(null);

      const data = await apiClient.getProtocol(protocolId!);
      setProtocol(data);

      // Load reviews
      const reviewsData = await apiClient.getProtocolReviews(protocolId!);
      setReviews(reviewsData.reviews);
    } catch (err: any) {
      console.error('Failed to load protocol:', err);
      setError(err.message || 'Failed to load protocol details.');
    } finally {
      setLoading(false);
    }
  };

  const handlePurchase = () => {
    setPurchaseModalOpen(true);
  };

  const handleShare = () => {
    if (navigator.share) {
      navigator.share({
        title: protocol?.display_name,
        text: protocol?.short_description,
        url: window.location.href,
      });
    } else {
      // Fallback: copy to clipboard
      navigator.clipboard.writeText(window.location.href);
      alert('Link copied to clipboard!');
    }
  };

  const handleReport = () => {
    // TODO: Implement report functionality
    console.log('Report protocol');
  };

  if (loading) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Skeleton variant="text" width={200} height={40} />
        <Box sx={{ mt: 4 }}>
          <Grid container spacing={4}>
            <Grid item xs={12} md={8}>
              <Skeleton variant="rectangular" height={400} />
            </Grid>
            <Grid item xs={12} md={4}>
              <Skeleton variant="rectangular" height={400} />
            </Grid>
          </Grid>
        </Box>
      </Container>
    );
  }

  if (error || !protocol) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Alert severity="error">
          {error || 'Protocol not found'}
        </Alert>
        <Button
          startIcon={<ArrowBackIcon />}
          onClick={() => navigate('/marketplace')}
          sx={{ mt: 2 }}
        >
          Back to Marketplace
        </Button>
      </Container>
    );
  }

  const certificationBadge = CERTIFICATION_BADGES[protocol.quality_metrics.certification_status];

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Breadcrumbs */}
      <Breadcrumbs sx={{ mb: 3 }}>
        <Link
          underline="hover"
          color="inherit"
          onClick={() => navigate('/marketplace')}
          sx={{ cursor: 'pointer' }}
        >
          Marketplace
        </Link>
        <Link
          underline="hover"
          color="inherit"
          onClick={() => navigate(`/marketplace?category=${protocol.category}`)}
          sx={{ cursor: 'pointer' }}
        >
          {protocol.category}
        </Link>
        <Typography color="text.primary">{protocol.display_name}</Typography>
      </Breadcrumbs>

      {/* Back Button */}
      <Button
        startIcon={<ArrowBackIcon />}
        onClick={() => navigate('/marketplace')}
        sx={{ mb: 2 }}
      >
        Back to Marketplace
      </Button>

      <Grid container spacing={4}>
        {/* Main Content */}
        <Grid item xs={12} md={8}>
          {/* Hero Section */}
          <Box mb={3}>
            <Box display="flex" alignItems="flex-start" mb={2}>
              <Avatar
                sx={{
                  width: 80,
                  height: 80,
                  bgcolor: 'primary.light',
                  mr: 2,
                  fontSize: '2rem',
                }}
              >
                {protocol.display_name.substring(0, 2).toUpperCase()}
              </Avatar>

              <Box flexGrow={1}>
                <Box display="flex" alignItems="center" mb={1}>
                  <Typography variant="h4" component="h1" fontWeight="bold">
                    {protocol.display_name}
                  </Typography>
                  {(protocol.is_official || protocol.is_featured) && (
                    <Box ml={2}>
                      {protocol.is_official && (
                        <Chip
                          label="Official"
                          size="small"
                          color="primary"
                          sx={{ mr: 0.5 }}
                        />
                      )}
                      {protocol.is_featured && (
                        <Chip
                          label="Featured"
                          size="small"
                          icon={<StarIcon />}
                          color="warning"
                        />
                      )}
                    </Box>
                  )}
                </Box>

                <Typography variant="body1" color="text.secondary" mb={1}>
                  by{' '}
                  <Link
                    component="span"
                    sx={{ cursor: 'pointer', fontWeight: 600 }}
                    onClick={() => navigate(`/marketplace/creator/${protocol.author.user_id}`)}
                  >
                    {protocol.author.organization || protocol.author.username}
                  </Link>
                  {protocol.author.is_verified && (
                    <Tooltip title="Verified Creator">
                      <VerifiedIcon
                        color="primary"
                        sx={{ ml: 0.5, fontSize: 18, verticalAlign: 'middle' }}
                      />
                    </Tooltip>
                  )}
                </Typography>

                <Box display="flex" alignItems="center" gap={2}>
                  <Box display="flex" alignItems="center">
                    <Rating
                      value={protocol.quality_metrics.average_rating}
                      precision={0.1}
                      size="small"
                      readOnly
                    />
                    <Typography variant="body2" color="text.secondary" sx={{ ml: 0.5 }}>
                      {protocol.quality_metrics.average_rating.toFixed(1)} ({protocol.quality_metrics.total_ratings} reviews)
                    </Typography>
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    •
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    <DownloadIcon sx={{ fontSize: 16, verticalAlign: 'middle', mr: 0.5 }} />
                    {protocol.quality_metrics.download_count.toLocaleString()} downloads
                  </Typography>
                </Box>
              </Box>

              {/* Action Icons */}
              <Box>
                <IconButton onClick={handleShare} size="small">
                  <ShareIcon />
                </IconButton>
                <IconButton onClick={handleReport} size="small">
                  <ReportIcon />
                </IconButton>
              </Box>
            </Box>

            <Typography variant="body1" paragraph>
              {protocol.short_description}
            </Typography>

            {/* Tags */}
            <Box display="flex" flexWrap="wrap" gap={1}>
              {protocol.tags.map((tag) => (
                <Chip
                  key={tag}
                  label={tag}
                  size="small"
                  variant="outlined"
                  onClick={() => navigate(`/marketplace?q=${tag}`)}
                />
              ))}
            </Box>
          </Box>

          <Divider sx={{ my: 3 }} />

          {/* Tabs */}
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs value={activeTab} onChange={(e, v) => setActiveTab(v)}>
              <Tab label="Overview" />
              <Tab label="Technical Specs" />
              <Tab label={`Reviews (${protocol.quality_metrics.total_ratings})`} />
              <Tab label="Changelog" />
            </Tabs>
          </Box>

          {/* Overview Tab */}
          <TabPanel value={activeTab} index={0}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom fontWeight="600">
                Description
              </Typography>
              <ReactMarkdown>
                {protocol.long_description || protocol.short_description}
              </ReactMarkdown>

              <Divider sx={{ my: 3 }} />

              <Typography variant="h6" gutterBottom fontWeight="600">
                Features
              </Typography>
              <List>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircleIcon color="success" />
                  </ListItemIcon>
                  <ListItemText primary="Production-ready parser implementation" />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircleIcon color="success" />
                  </ListItemIcon>
                  <ListItemText primary="Comprehensive documentation" />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircleIcon color="success" />
                  </ListItemIcon>
                  <ListItemText primary="Test samples included" />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircleIcon color="success" />
                  </ListItemIcon>
                  <ListItemText primary="Security validated" />
                </ListItem>
                <ListItem>
                  <ListItemIcon>
                    <CheckCircleIcon color="success" />
                  </ListItemIcon>
                  <ListItemText primary="Performance benchmarked" />
                </ListItem>
              </List>
            </Paper>
          </TabPanel>

          {/* Technical Specs Tab */}
          <TabPanel value={activeTab} index={1}>
            <Paper sx={{ p: 3 }}>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">Version</Typography>
                  <Typography variant="body1" fontWeight="600">{protocol.version}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">Protocol Type</Typography>
                  <Typography variant="body1" fontWeight="600">{protocol.protocol_type}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">Category</Typography>
                  <Typography variant="body1" fontWeight="600">{protocol.category}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">Industry</Typography>
                  <Typography variant="body1" fontWeight="600">{protocol.industry || 'N/A'}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">Min QBITEL Version</Typography>
                  <Typography variant="body1" fontWeight="600">
                    {protocol.compatibility?.min_qbitel_version || 'N/A'}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">Last Updated</Typography>
                  <Typography variant="body1" fontWeight="600">
                    {new Date(protocol.updated_at).toLocaleDateString()}
                  </Typography>
                </Grid>
              </Grid>

              {protocol.compatibility && protocol.compatibility.dependencies.length > 0 && (
                <>
                  <Divider sx={{ my: 3 }} />
                  <Typography variant="h6" gutterBottom fontWeight="600">
                    Dependencies
                  </Typography>
                  <List>
                    {protocol.compatibility.dependencies.map((dep, idx) => (
                      <ListItem key={idx}>
                        <ListItemText primary={dep} />
                      </ListItem>
                    ))}
                  </List>
                </>
              )}
            </Paper>
          </TabPanel>

          {/* Reviews Tab */}
          <TabPanel value={activeTab} index={2}>
            <ReviewsSection
              protocolId={protocol.protocol_id}
              apiClient={apiClient}
              initialReviews={reviews}
            />
          </TabPanel>

          {/* Changelog Tab */}
          <TabPanel value={activeTab} index={3}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom fontWeight="600">
                Version {protocol.version}
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Released on {new Date(protocol.updated_at).toLocaleDateString()}
              </Typography>
              <Typography variant="body1">
                Latest stable version with improvements and bug fixes.
              </Typography>
            </Paper>
          </TabPanel>
        </Grid>

        {/* Sidebar - Purchase Card */}
        <Grid item xs={12} md={4}>
          <Paper
            sx={{
              p: 3,
              position: 'sticky',
              top: 24,
            }}
            elevation={3}
          >
            {/* Price */}
            <Box mb={3}>
              {protocol.licensing.license_type === 'free' ? (
                <Typography variant="h3" color="success.main" fontWeight="bold">
                  FREE
                </Typography>
              ) : (
                <>
                  <Typography variant="h3" color="primary.main" fontWeight="bold">
                    ${protocol.licensing.base_price?.toFixed(2)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {protocol.licensing.price_model === 'subscription' ? 'per month' : 'one-time payment'}
                  </Typography>
                </>
              )}
            </Box>

            {/* Purchase Button */}
            <Button
              variant="contained"
              size="large"
              fullWidth
              startIcon={<ShoppingCartIcon />}
              onClick={handlePurchase}
              sx={{ mb: 2, py: 1.5 }}
            >
              {protocol.licensing.license_type === 'free' ? 'Install Now' : 'Purchase Now'}
            </Button>

            {protocol.licensing.license_type !== 'free' && (
              <Typography variant="caption" color="text.secondary" display="block" textAlign="center" mb={2}>
                Free 14-day trial included
              </Typography>
            )}

            <Divider sx={{ my: 2 }} />

            {/* What's Included */}
            <Typography variant="subtitle2" gutterBottom fontWeight="600">
              What's included:
            </Typography>
            <List dense>
              <ListItem>
                <ListItemIcon sx={{ minWidth: 32 }}>
                  <CheckCircleIcon color="success" fontSize="small" />
                </ListItemIcon>
                <ListItemText primary="Protocol specification files" />
              </ListItem>
              <ListItem>
                <ListItemIcon sx={{ minWidth: 32 }}>
                  <CheckCircleIcon color="success" fontSize="small" />
                </ListItemIcon>
                <ListItemText primary="Parser implementation" />
              </ListItem>
              <ListItem>
                <ListItemIcon sx={{ minWidth: 32 }}>
                  <CheckCircleIcon color="success" fontSize="small" />
                </ListItemIcon>
                <ListItemText primary="Test data samples" />
              </ListItem>
              <ListItem>
                <ListItemIcon sx={{ minWidth: 32 }}>
                  <CheckCircleIcon color="success" fontSize="small" />
                </ListItemIcon>
                <ListItemText primary="Documentation" />
              </ListItem>
              <ListItem>
                <ListItemIcon sx={{ minWidth: 32 }}>
                  <CheckCircleIcon color="success" fontSize="small" />
                </ListItemIcon>
                <ListItemText primary="Email support" />
              </ListItem>
            </List>

            <Divider sx={{ my: 2 }} />

            {/* Certification Status */}
            <Box>
              <Typography variant="subtitle2" gutterBottom fontWeight="600">
                Certification:
              </Typography>
              <Chip
                label={certificationBadge.label}
                size="medium"
                sx={{
                  bgcolor: `${certificationBadge.color}20`,
                  color: certificationBadge.color,
                  fontWeight: 600,
                  borderColor: certificationBadge.color,
                  border: '1px solid',
                  width: '100%',
                }}
              />
            </Box>
          </Paper>
        </Grid>
      </Grid>

      {/* Purchase Modal */}
      <PurchaseModal
        open={purchaseModalOpen}
        onClose={() => setPurchaseModalOpen(false)}
        protocol={protocol}
        apiClient={apiClient}
      />
    </Container>
  );
};

export default ProtocolDetails;
