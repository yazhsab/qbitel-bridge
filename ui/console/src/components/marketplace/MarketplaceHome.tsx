import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Container,
  Typography,
  TextField,
  InputAdornment,
  Grid,
  Chip,
  Button,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  Pagination,
  CircularProgress,
  Alert,
  Paper,
  Tabs,
  Tab,
  IconButton,
  Skeleton,
} from '@mui/material';
import {
  Search as SearchIcon,
  FilterList as FilterIcon,
  Star as StarIcon,
  TrendingUp as TrendingUpIcon,
  NewReleases as NewReleasesIcon,
  Add as AddIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { MarketplaceApiClient } from '../../api/marketplace';
import {
  Protocol,
  SearchProtocolsResponse,
  MarketplaceFilters,
  PROTOCOL_CATEGORIES,
  LICENSE_TYPES,
} from '../../types/marketplace';
import ProtocolCard from './ProtocolCard';

interface MarketplaceHomeProps {
  apiClient: MarketplaceApiClient;
}

const MarketplaceHome: React.FC<MarketplaceHomeProps> = ({ apiClient }) => {
  const navigate = useNavigate();

  // State management
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchResults, setSearchResults] = useState<SearchProtocolsResponse | null>(null);
  const [featuredProtocols, setFeaturedProtocols] = useState<Protocol[]>([]);
  const [filters, setFilters] = useState<MarketplaceFilters>({
    searchQuery: '',
    category: '',
    licenseType: '',
    minRating: 0,
    sortBy: 'rating',
  });
  const [page, setPage] = useState(1);
  const [activeTab, setActiveTab] = useState(0);

  // Load initial data
  useEffect(() => {
    loadMarketplaceData();
  }, []);

  // Load protocols when filters or page changes
  useEffect(() => {
    if (!loading) {
      searchProtocols();
    }
  }, [filters, page]);

  const loadMarketplaceData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Load featured protocols
      const featured = await apiClient.getFeaturedProtocols(6);
      setFeaturedProtocols(featured);

      // Load initial search results
      await searchProtocols();
    } catch (err: any) {
      console.error('Failed to load marketplace data:', err);
      setError(err.message || 'Failed to load marketplace. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const searchProtocols = async () => {
    try {
      const results = await apiClient.searchProtocols({
        q: filters.searchQuery || undefined,
        category: filters.category || undefined,
        license_type: filters.licenseType as any || undefined,
        min_rating: filters.minRating || undefined,
        sort: filters.sortBy,
        page,
        limit: 20,
      });

      setSearchResults(results);
    } catch (err: any) {
      console.error('Search failed:', err);
      setError('Search failed. Please try again.');
    }
  };

  const handleSearchChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    setFilters(prev => ({ ...prev, searchQuery: event.target.value }));
    setPage(1); // Reset to first page on search
  }, []);

  const handleCategoryChange = (category: string) => {
    setFilters(prev => ({ ...prev, category: category === prev.category ? '' : category }));
    setPage(1);
  };

  const handleFilterChange = (key: keyof MarketplaceFilters, value: any) => {
    setFilters(prev => ({ ...prev, [key]: value }));
    setPage(1);
  };

  const handlePageChange = (event: React.ChangeEvent<unknown>, value: number) => {
    setPage(value);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleTabChange = async (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
    setLoading(true);

    try {
      let protocols: Protocol[] = [];

      switch (newValue) {
        case 0: // All
          await searchProtocols();
          return;
        case 1: // Featured
          protocols = await apiClient.getFeaturedProtocols(20);
          break;
        case 2: // Popular
          protocols = await apiClient.getPopularProtocols(20);
          break;
        case 3: // Recent
          protocols = await apiClient.getRecentProtocols(20);
          break;
      }

      setSearchResults({
        protocols,
        pagination: {
          total: protocols.length,
          page: 1,
          limit: 20,
          pages: 1,
        },
        facets: {
          categories: {},
          license_types: {},
        },
      });
    } catch (err: any) {
      console.error('Failed to load protocols:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Render loading skeleton
  if (loading && !searchResults) {
    return (
      <Container maxWidth="xl" sx={{ py: 4 }}>
        <Box mb={4}>
          <Skeleton variant="text" width="300px" height={60} />
          <Skeleton variant="text" width="500px" height={30} />
        </Box>
        <Skeleton variant="rectangular" width="100%" height={56} sx={{ mb: 3, borderRadius: 1 }} />
        <Grid container spacing={3}>
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <Grid item xs={12} sm={6} md={4} key={i}>
              <Skeleton variant="rectangular" height={320} sx={{ borderRadius: 2 }} />
            </Grid>
          ))}
        </Grid>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ py: 4 }}>
      {/* Header */}
      <Box mb={4}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Box>
            <Typography variant="h4" component="h1" gutterBottom fontWeight="bold">
              Protocol Marketplace
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Discover, purchase, and deploy protocol parsers in minutes
            </Typography>
          </Box>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            size="large"
            onClick={() => navigate('/marketplace/submit')}
            sx={{ minWidth: 200 }}
          >
            Submit Protocol
          </Button>
        </Box>
      </Box>

      {/* Search Bar */}
      <Paper elevation={2} sx={{ p: 2, mb: 3 }}>
        <TextField
          fullWidth
          placeholder="Search protocols... (e.g., 'HL7', 'Modbus', 'ISO 8583')"
          value={filters.searchQuery}
          onChange={handleSearchChange}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon />
              </InputAdornment>
            ),
          }}
          sx={{
            '& .MuiOutlinedInput-root': {
              borderRadius: 2,
            },
          }}
        />
      </Paper>

      {/* Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
        <Tabs value={activeTab} onChange={handleTabChange}>
          <Tab icon={<SearchIcon />} label="All Protocols" iconPosition="start" />
          <Tab icon={<StarIcon />} label="Featured" iconPosition="start" />
          <Tab icon={<TrendingUpIcon />} label="Popular" iconPosition="start" />
          <Tab icon={<NewReleasesIcon />} label="Recent" iconPosition="start" />
        </Tabs>
      </Box>

      {/* Category Pills */}
      <Box mb={3}>
        <Typography variant="subtitle2" gutterBottom fontWeight="600">
          Categories
        </Typography>
        <Box display="flex" flexWrap="wrap" gap={1}>
          <Chip
            label="All"
            clickable
            color={filters.category === '' ? 'primary' : 'default'}
            onClick={() => handleCategoryChange('')}
          />
          {PROTOCOL_CATEGORIES.map((cat) => (
            <Chip
              key={cat.value}
              label={`${cat.icon} ${cat.label}`}
              clickable
              color={filters.category === cat.value ? 'primary' : 'default'}
              onClick={() => handleCategoryChange(cat.value)}
            />
          ))}
        </Box>
      </Box>

      {/* Filters and Sort */}
      <Box display="flex" gap={2} mb={3} flexWrap="wrap">
        <FormControl size="small" sx={{ minWidth: 150 }}>
          <InputLabel>License Type</InputLabel>
          <Select
            value={filters.licenseType}
            label="License Type"
            onChange={(e) => handleFilterChange('licenseType', e.target.value)}
          >
            <MenuItem value="">All</MenuItem>
            {LICENSE_TYPES.map((type) => (
              <MenuItem key={type.value} value={type.value}>
                {type.label}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <FormControl size="small" sx={{ minWidth: 150 }}>
          <InputLabel>Min Rating</InputLabel>
          <Select
            value={filters.minRating}
            label="Min Rating"
            onChange={(e) => handleFilterChange('minRating', e.target.value)}
          >
            <MenuItem value={0}>All</MenuItem>
            <MenuItem value={3}>3+ Stars</MenuItem>
            <MenuItem value={4}>4+ Stars</MenuItem>
            <MenuItem value={4.5}>4.5+ Stars</MenuItem>
          </Select>
        </FormControl>

        <FormControl size="small" sx={{ minWidth: 150 }}>
          <InputLabel>Sort By</InputLabel>
          <Select
            value={filters.sortBy}
            label="Sort By"
            onChange={(e) => handleFilterChange('sortBy', e.target.value)}
          >
            <MenuItem value="rating">Rating</MenuItem>
            <MenuItem value="downloads">Downloads</MenuItem>
            <MenuItem value="recent">Most Recent</MenuItem>
            <MenuItem value="alphabetical">Alphabetical</MenuItem>
          </Select>
        </FormControl>
      </Box>

      {/* Error Alert */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Protocol Grid */}
      {searchResults && searchResults.protocols.length > 0 ? (
        <>
          <Typography variant="body2" color="text.secondary" mb={2}>
            Showing {searchResults.protocols.length} of {searchResults.pagination.total} protocols
          </Typography>

          <Grid container spacing={3} mb={4}>
            {searchResults.protocols.map((protocol) => (
              <Grid item xs={12} sm={6} md={4} key={protocol.protocol_id}>
                <ProtocolCard protocol={protocol} />
              </Grid>
            ))}
          </Grid>

          {/* Pagination */}
          {searchResults.pagination.pages > 1 && (
            <Box display="flex" justifyContent="center" mt={4}>
              <Pagination
                count={searchResults.pagination.pages}
                page={page}
                onChange={handlePageChange}
                color="primary"
                size="large"
              />
            </Box>
          )}
        </>
      ) : (
        <Paper sx={{ p: 6, textAlign: 'center' }}>
          <Typography variant="h6" color="text.secondary" gutterBottom>
            No protocols found
          </Typography>
          <Typography variant="body2" color="text.secondary" mb={3}>
            Try adjusting your search criteria or filters
          </Typography>
          <Button
            variant="outlined"
            onClick={() => {
              setFilters({
                searchQuery: '',
                category: '',
                licenseType: '',
                minRating: 0,
                sortBy: 'rating',
              });
              setPage(1);
            }}
          >
            Clear Filters
          </Button>
        </Paper>
      )}
    </Container>
  );
};

export default MarketplaceHome;
