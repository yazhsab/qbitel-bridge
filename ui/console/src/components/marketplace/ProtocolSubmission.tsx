import React, { useState } from 'react';
import {
  Box,
  Container,
  Typography,
  Stepper,
  Step,
  StepLabel,
  Button,
  Paper,
  TextField,
  MenuItem,
  Chip,
  FormControl,
  InputLabel,
  Select,
  Grid,
  Alert,
  CircularProgress,
  Divider,
  IconButton,
  OutlinedInput,
  Card,
  CardContent,
  LinearProgress,
} from '@mui/material';
import {
  ArrowBack as ArrowBackIcon,
  CloudUpload as CloudUploadIcon,
  Delete as DeleteIcon,
  CheckCircle as CheckCircleIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { MarketplaceApiClient } from '../../api/marketplace';
import {
  SubmitProtocolRequest,
  SubmissionFlowState,
  PROTOCOL_CATEGORIES,
  LICENSE_TYPES,
} from '../../types/marketplace';

interface ProtocolSubmissionProps {
  apiClient: MarketplaceApiClient;
}

const steps = [
  'Basic Information',
  'Technical Details',
  'Upload Files',
  'Licensing & Pricing',
  'Review & Submit',
];

const ProtocolSubmission: React.FC<ProtocolSubmissionProps> = ({ apiClient }) => {
  const navigate = useNavigate();

  const [state, setState] = useState<SubmissionFlowState>({
    step: 0,
    formData: {
      protocol_name: '',
      display_name: '',
      short_description: '',
      long_description: '',
      category: '',
      subcategory: '',
      tags: [],
      version: '1.0.0',
      protocol_type: 'binary',
      industry: '',
      spec_format: 'yaml',
      spec_file: '',
      parser_code: '',
      test_data: '',
      license_type: 'free',
      price_model: undefined,
      base_price: undefined,
      min_cronos_version: '1.0.0',
    },
    files: {},
    validationErrors: {},
    submitting: false,
  });

  const [tagInput, setTagInput] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const handleInputChange = (field: keyof SubmitProtocolRequest, value: any) => {
    setState(prev => ({
      ...prev,
      formData: {
        ...prev.formData,
        [field]: value,
      },
      validationErrors: {
        ...prev.validationErrors,
        [field]: undefined,
      },
    }));
  };

  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>,
    fileType: 'spec' | 'parser' | 'testData' | 'docs'
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setState(prev => ({
      ...prev,
      files: {
        ...prev.files,
        [fileType]: file,
      },
    }));

    // Convert file to base64
    const reader = new FileReader();
    reader.onload = () => {
      const base64 = reader.result?.toString().split(',')[1] || '';
      handleInputChange(
        fileType === 'spec' ? 'spec_file' :
        fileType === 'parser' ? 'parser_code' :
        'test_data',
        base64
      );
    };
    reader.readAsDataURL(file);
  };

  const handleAddTag = () => {
    if (tagInput.trim() && !state.formData.tags?.includes(tagInput.trim())) {
      handleInputChange('tags', [...(state.formData.tags || []), tagInput.trim()]);
      setTagInput('');
    }
  };

  const handleDeleteTag = (tagToDelete: string) => {
    handleInputChange(
      'tags',
      state.formData.tags?.filter(tag => tag !== tagToDelete) || []
    );
  };

  const validateStep = (): boolean => {
    const errors: Record<string, string> = {};

    switch (state.step) {
      case 0: // Basic Information
        if (!state.formData.protocol_name) errors.protocol_name = 'Protocol name is required';
        if (!state.formData.display_name) errors.display_name = 'Display name is required';
        if (!state.formData.short_description) errors.short_description = 'Short description is required';
        if (!state.formData.category) errors.category = 'Category is required';
        break;

      case 1: // Technical Details
        if (!state.formData.version) errors.version = 'Version is required';
        if (!state.formData.protocol_type) errors.protocol_type = 'Protocol type is required';
        if (!state.formData.min_cronos_version) errors.min_cronos_version = 'Min CRONOS version is required';
        break;

      case 2: // Upload Files
        if (!state.formData.spec_file) errors.spec_file = 'Protocol specification file is required';
        break;

      case 3: // Licensing
        if (!state.formData.license_type) errors.license_type = 'License type is required';
        if (state.formData.license_type !== 'free' && !state.formData.price_model) {
          errors.price_model = 'Price model is required';
        }
        if (state.formData.license_type !== 'free' && !state.formData.base_price) {
          errors.base_price = 'Base price is required';
        }
        break;
    }

    setState(prev => ({ ...prev, validationErrors: errors }));
    return Object.keys(errors).length === 0;
  };

  const handleNext = () => {
    if (validateStep()) {
      if (state.step === steps.length - 1) {
        handleSubmit();
      } else {
        setState(prev => ({ ...prev, step: prev.step + 1 }));
      }
    }
  };

  const handleBack = () => {
    setState(prev => ({ ...prev, step: prev.step - 1 }));
  };

  const handleSubmit = async () => {
    try {
      setState(prev => ({ ...prev, submitting: true }));
      setError(null);

      const result = await apiClient.submitProtocol(state.formData as SubmitProtocolRequest);

      setSuccess(true);
      setTimeout(() => {
        navigate(`/marketplace/protocols/${result.protocol_id}`);
      }, 2000);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Submission failed');
    } finally {
      setState(prev => ({ ...prev, submitting: false }));
    }
  };

  const renderStepContent = () => {
    switch (state.step) {
      case 0: // Basic Information
        return (
          <Box>
            <TextField
              fullWidth
              label="Protocol Name"
              placeholder="e.g., iso8583-v1987"
              value={state.formData.protocol_name}
              onChange={(e) => handleInputChange('protocol_name', e.target.value)}
              error={!!state.validationErrors.protocol_name}
              helperText={state.validationErrors.protocol_name || 'Unique identifier (lowercase, hyphens only)'}
              sx={{ mb: 3 }}
            />

            <TextField
              fullWidth
              label="Display Name"
              placeholder="e.g., ISO 8583:1987 Financial Transaction Protocol"
              value={state.formData.display_name}
              onChange={(e) => handleInputChange('display_name', e.target.value)}
              error={!!state.validationErrors.display_name}
              helperText={state.validationErrors.display_name || 'Human-readable name'}
              sx={{ mb: 3 }}
            />

            <TextField
              fullWidth
              label="Short Description"
              placeholder="Brief description (max 500 characters)"
              value={state.formData.short_description}
              onChange={(e) => handleInputChange('short_description', e.target.value)}
              error={!!state.validationErrors.short_description}
              helperText={state.validationErrors.short_description || `${state.formData.short_description?.length || 0}/500`}
              multiline
              rows={2}
              sx={{ mb: 3 }}
            />

            <TextField
              fullWidth
              label="Long Description (Optional)"
              placeholder="Detailed description (Markdown supported)"
              value={state.formData.long_description}
              onChange={(e) => handleInputChange('long_description', e.target.value)}
              multiline
              rows={4}
              sx={{ mb: 3 }}
            />

            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <FormControl fullWidth error={!!state.validationErrors.category}>
                  <InputLabel>Category</InputLabel>
                  <Select
                    value={state.formData.category}
                    onChange={(e) => handleInputChange('category', e.target.value)}
                    label="Category"
                  >
                    {PROTOCOL_CATEGORIES.map((cat) => (
                      <MenuItem key={cat.value} value={cat.value}>
                        {cat.icon} {cat.label}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Industry (Optional)"
                  value={state.formData.industry}
                  onChange={(e) => handleInputChange('industry', e.target.value)}
                />
              </Grid>
            </Grid>

            <Box sx={{ mt: 3 }}>
              <Typography variant="subtitle2" gutterBottom>
                Tags
              </Typography>
              <Box display="flex" gap={1} mb={1}>
                <TextField
                  size="small"
                  placeholder="Add tag"
                  value={tagInput}
                  onChange={(e) => setTagInput(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleAddTag()}
                />
                <Button variant="outlined" onClick={handleAddTag}>
                  Add
                </Button>
              </Box>
              <Box display="flex" flexWrap="wrap" gap={1}>
                {state.formData.tags?.map((tag) => (
                  <Chip
                    key={tag}
                    label={tag}
                    onDelete={() => handleDeleteTag(tag)}
                    size="small"
                  />
                ))}
              </Box>
            </Box>
          </Box>
        );

      case 1: // Technical Details
        return (
          <Box>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Version"
                  value={state.formData.version}
                  onChange={(e) => handleInputChange('version', e.target.value)}
                  error={!!state.validationErrors.version}
                  helperText={state.validationErrors.version || 'Semantic versioning (e.g., 1.0.0)'}
                  sx={{ mb: 3 }}
                />
              </Grid>

              <Grid item xs={12} md={6}>
                <FormControl fullWidth error={!!state.validationErrors.protocol_type}>
                  <InputLabel>Protocol Type</InputLabel>
                  <Select
                    value={state.formData.protocol_type}
                    onChange={(e) => handleInputChange('protocol_type', e.target.value)}
                    label="Protocol Type"
                  >
                    <MenuItem value="binary">Binary</MenuItem>
                    <MenuItem value="text">Text</MenuItem>
                    <MenuItem value="xml">XML</MenuItem>
                    <MenuItem value="json">JSON</MenuItem>
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12} md={6}>
                <FormControl fullWidth>
                  <InputLabel>Spec Format</InputLabel>
                  <Select
                    value={state.formData.spec_format}
                    onChange={(e) => handleInputChange('spec_format', e.target.value)}
                    label="Spec Format"
                  >
                    <MenuItem value="yaml">YAML</MenuItem>
                    <MenuItem value="json">JSON</MenuItem>
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Min CRONOS Version"
                  value={state.formData.min_cronos_version}
                  onChange={(e) => handleInputChange('min_cronos_version', e.target.value)}
                  error={!!state.validationErrors.min_cronos_version}
                  helperText={state.validationErrors.min_cronos_version}
                />
              </Grid>
            </Grid>
          </Box>
        );

      case 2: // Upload Files
        return (
          <Box>
            <Card variant="outlined" sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="subtitle2" gutterBottom fontWeight="600">
                  Protocol Specification * {state.files.spec && '✓'}
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  YAML or JSON file defining the protocol structure
                </Typography>
                <Button
                  variant="outlined"
                  component="label"
                  startIcon={<CloudUploadIcon />}
                  fullWidth
                >
                  {state.files.spec ? state.files.spec.name : 'Choose File'}
                  <input
                    type="file"
                    hidden
                    accept=".yaml,.yml,.json"
                    onChange={(e) => handleFileUpload(e, 'spec')}
                  />
                </Button>
                {state.validationErrors.spec_file && (
                  <Typography variant="caption" color="error" display="block" sx={{ mt: 1 }}>
                    {state.validationErrors.spec_file}
                  </Typography>
                )}
              </CardContent>
            </Card>

            <Card variant="outlined" sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="subtitle2" gutterBottom fontWeight="600">
                  Parser Code (Optional) {state.files.parser && '✓'}
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  Python parser implementation
                </Typography>
                <Button
                  variant="outlined"
                  component="label"
                  startIcon={<CloudUploadIcon />}
                  fullWidth
                >
                  {state.files.parser ? state.files.parser.name : 'Choose File'}
                  <input
                    type="file"
                    hidden
                    accept=".py"
                    onChange={(e) => handleFileUpload(e, 'parser')}
                  />
                </Button>
              </CardContent>
            </Card>

            <Card variant="outlined">
              <CardContent>
                <Typography variant="subtitle2" gutterBottom fontWeight="600">
                  Test Data (Optional) {state.files.testData && '✓'}
                </Typography>
                <Typography variant="body2" color="text.secondary" paragraph>
                  Sample protocol packets for testing
                </Typography>
                <Button
                  variant="outlined"
                  component="label"
                  startIcon={<CloudUploadIcon />}
                  fullWidth
                >
                  {state.files.testData ? state.files.testData.name : 'Choose File'}
                  <input
                    type="file"
                    hidden
                    onChange={(e) => handleFileUpload(e, 'testData')}
                  />
                </Button>
              </CardContent>
            </Card>
          </Box>
        );

      case 3: // Licensing & Pricing
        return (
          <Box>
            <FormControl fullWidth error={!!state.validationErrors.license_type} sx={{ mb: 3 }}>
              <InputLabel>License Type</InputLabel>
              <Select
                value={state.formData.license_type}
                onChange={(e) => handleInputChange('license_type', e.target.value)}
                label="License Type"
              >
                {LICENSE_TYPES.map((type) => (
                  <MenuItem key={type.value} value={type.value}>
                    <Box>
                      <Typography variant="body1">{type.label}</Typography>
                      <Typography variant="caption" color="text.secondary">
                        {type.description}
                      </Typography>
                    </Box>
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {state.formData.license_type !== 'free' && (
              <>
                <FormControl fullWidth error={!!state.validationErrors.price_model} sx={{ mb: 3 }}>
                  <InputLabel>Price Model</InputLabel>
                  <Select
                    value={state.formData.price_model || ''}
                    onChange={(e) => handleInputChange('price_model', e.target.value)}
                    label="Price Model"
                  >
                    <MenuItem value="one_time">One-time Payment</MenuItem>
                    <MenuItem value="subscription">Monthly Subscription</MenuItem>
                    <MenuItem value="usage_based">Usage-based</MenuItem>
                  </Select>
                </FormControl>

                <TextField
                  fullWidth
                  label="Base Price (USD)"
                  type="number"
                  value={state.formData.base_price || ''}
                  onChange={(e) => handleInputChange('base_price', parseFloat(e.target.value))}
                  error={!!state.validationErrors.base_price}
                  helperText={state.validationErrors.base_price}
                  InputProps={{ startAdornment: '$' }}
                />
              </>
            )}
          </Box>
        );

      case 4: // Review & Submit
        return (
          <Box>
            {success ? (
              <Box textAlign="center" py={4}>
                <CheckCircleIcon color="success" sx={{ fontSize: 64, mb: 2 }} />
                <Typography variant="h5" gutterBottom>
                  Protocol Submitted!
                </Typography>
                <Typography variant="body1" color="text.secondary">
                  Your protocol has been submitted for validation.
                  Redirecting to protocol details...
                </Typography>
              </Box>
            ) : (
              <Box>
                <Typography variant="h6" gutterBottom>
                  Review Your Submission
                </Typography>
                <Divider sx={{ my: 2 }} />

                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">Protocol Name:</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" fontWeight="600">{state.formData.protocol_name}</Typography>
                  </Grid>

                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">Display Name:</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" fontWeight="600">{state.formData.display_name}</Typography>
                  </Grid>

                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">Version:</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" fontWeight="600">{state.formData.version}</Typography>
                  </Grid>

                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">Category:</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" fontWeight="600">{state.formData.category}</Typography>
                  </Grid>

                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">License Type:</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" fontWeight="600">{state.formData.license_type}</Typography>
                  </Grid>

                  {state.formData.license_type !== 'free' && (
                    <>
                      <Grid item xs={6}>
                        <Typography variant="body2" color="text.secondary">Price:</Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2" fontWeight="600">
                          ${state.formData.base_price} ({state.formData.price_model})
                        </Typography>
                      </Grid>
                    </>
                  )}

                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">Files:</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" fontWeight="600">
                      Spec: {state.files.spec ? '✓' : '✗'}
                      {state.files.parser && ', Parser: ✓'}
                      {state.files.testData && ', Test Data: ✓'}
                    </Typography>
                  </Grid>
                </Grid>

                <Alert severity="info" sx={{ mt: 3 }}>
                  Your protocol will go through a validation process that includes syntax validation,
                  parser testing, security scanning, and manual review. This typically takes 2-3 business days.
                </Alert>
              </Box>
            )}
          </Box>
        );

      default:
        return null;
    }
  };

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Button
        startIcon={<ArrowBackIcon />}
        onClick={() => navigate('/marketplace')}
        sx={{ mb: 3 }}
      >
        Back to Marketplace
      </Button>

      <Typography variant="h4" component="h1" gutterBottom fontWeight="bold">
        Submit Protocol
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Share your protocol with the CRONOS AI community
      </Typography>

      <Paper sx={{ p: 3, mt: 3 }}>
        {!success && (
          <Stepper activeStep={state.step} sx={{ mb: 4 }}>
            {steps.map((label) => (
              <Step key={label}>
                <StepLabel>{label}</StepLabel>
              </Step>
            ))}
          </Stepper>
        )}

        {state.submitting && <LinearProgress sx={{ mb: 2 }} />}

        {error && (
          <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {renderStepContent()}

        {!success && (
          <Box display="flex" justifyContent="space-between" mt={4}>
            <Button
              onClick={handleBack}
              disabled={state.step === 0 || state.submitting}
            >
              Back
            </Button>
            <Button
              variant="contained"
              onClick={handleNext}
              disabled={state.submitting}
            >
              {state.submitting ? (
                <CircularProgress size={24} />
              ) : state.step === steps.length - 1 ? (
                'Submit Protocol'
              ) : (
                'Next'
              )}
            </Button>
          </Box>
        )}
      </Paper>
    </Container>
  );
};

export default ProtocolSubmission;
