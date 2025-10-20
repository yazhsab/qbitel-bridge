import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Stepper,
  Step,
  StepLabel,
  TextField,
  RadioGroup,
  Radio,
  FormControlLabel,
  Alert,
  Box,
  Typography,
  CircularProgress,
  Paper,
  List,
  ListItem,
  ListItemText,
  Divider,
  Checkbox,
  FormGroup,
} from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Download as DownloadIcon,
} from '@mui/icons-material';
import { Protocol, PurchaseProtocolRequest, PurchaseProtocolResponse } from '../../types/marketplace';
import { MarketplaceApiClient } from '../../api/marketplace';

interface PurchaseModalProps {
  open: boolean;
  onClose: () => void;
  protocol: Protocol;
  apiClient: MarketplaceApiClient;
}

const PurchaseModal: React.FC<PurchaseModalProps> = ({
  open,
  onClose,
  protocol,
  apiClient,
}) => {
  const [activeStep, setActiveStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [purchaseResult, setPurchaseResult] = useState<PurchaseProtocolResponse | null>(null);
  const [formData, setFormData] = useState<PurchaseProtocolRequest>({
    license_type: 'production',
    payment_method_id: 'pm_card_visa', // Demo payment method
    billing_email: '',
  });
  const [agreedToTerms, setAgreedToTerms] = useState(false);

  const steps = protocol.licensing.license_type === 'free'
    ? ['Confirm', 'Download']
    : ['Select Plan', 'Payment', 'Confirm', 'Complete'];

  const handleClose = () => {
    setActiveStep(0);
    setError(null);
    setPurchaseResult(null);
    setAgreedToTerms(false);
    onClose();
  };

  const handleNext = () => {
    if (activeStep === steps.length - 2) {
      handlePurchase();
    } else {
      setActiveStep(activeStep + 1);
    }
  };

  const handleBack = () => {
    setActiveStep(activeStep - 1);
  };

  const handlePurchase = async () => {
    try {
      setLoading(true);
      setError(null);

      const result = await apiClient.purchaseProtocol(
        protocol.protocol_id,
        formData
      );

      setPurchaseResult(result);
      setActiveStep(activeStep + 1);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Purchase failed');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async (url: string, filename: string) => {
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

  const renderStepContent = () => {
    if (protocol.licensing.license_type === 'free') {
      // Free protocol flow
      if (activeStep === 0) {
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Install Free Protocol
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              This protocol is free and open source. Click "Install" to add it to your account.
            </Typography>
            <Paper variant="outlined" sx={{ p: 2 }}>
              <List dense>
                <ListItem>
                  <ListItemText
                    primary="Protocol specification files"
                    secondary="YAML/JSON format"
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Parser implementation"
                    secondary="Python code"
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Test data samples"
                    secondary="Sample packets for testing"
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="Documentation"
                    secondary="Installation and usage guide"
                  />
                </ListItem>
              </List>
            </Paper>
          </Box>
        );
      } else {
        // Success step
        return (
          <Box textAlign="center">
            <CheckCircleIcon color="success" sx={{ fontSize: 64, mb: 2 }} />
            <Typography variant="h5" gutterBottom>
              Installation Successful!
            </Typography>
            <Typography variant="body1" color="text.secondary" paragraph>
              {protocol.display_name} has been installed to your account.
            </Typography>

            {purchaseResult && (
              <Paper variant="outlined" sx={{ p: 2, mt: 3, textAlign: 'left' }}>
                <Typography variant="subtitle2" gutterBottom fontWeight="600">
                  License Key:
                </Typography>
                <Typography
                  variant="body2"
                  sx={{
                    fontFamily: 'monospace',
                    bgcolor: 'grey.100',
                    p: 1,
                    borderRadius: 1,
                    mb: 2,
                  }}
                >
                  {purchaseResult.license_key}
                </Typography>

                <Typography variant="subtitle2" gutterBottom fontWeight="600">
                  Download Files:
                </Typography>
                <List dense>
                  <ListItem
                    button
                    onClick={() => handleDownload(purchaseResult.download_urls.spec, 'protocol-spec.yaml')}
                  >
                    <DownloadIcon sx={{ mr: 1 }} fontSize="small" />
                    <ListItemText primary="Protocol Specification" />
                  </ListItem>
                  {purchaseResult.download_urls.parser && (
                    <ListItem
                      button
                      onClick={() => handleDownload(purchaseResult.download_urls.parser!, 'parser.py')}
                    >
                      <DownloadIcon sx={{ mr: 1 }} fontSize="small" />
                      <ListItemText primary="Parser Code" />
                    </ListItem>
                  )}
                  {purchaseResult.download_urls.docs && (
                    <ListItem
                      button
                      onClick={() => handleDownload(purchaseResult.download_urls.docs!, 'documentation.pdf')}
                    >
                      <DownloadIcon sx={{ mr: 1 }} fontSize="small" />
                      <ListItemText primary="Documentation" />
                    </ListItem>
                  )}
                </List>
              </Paper>
            )}
          </Box>
        );
      }
    }

    // Paid protocol flow
    switch (activeStep) {
      case 0:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Select Your Plan
            </Typography>
            <RadioGroup
              value={formData.license_type}
              onChange={(e) =>
                setFormData({ ...formData, license_type: e.target.value as any })
              }
            >
              <Paper variant="outlined" sx={{ p: 2, mb: 1 }}>
                <FormControlLabel
                  value="production"
                  control={<Radio />}
                  label={
                    <Box>
                      <Typography variant="subtitle1" fontWeight="600">
                        Production License
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        ${protocol.licensing.base_price}/month • Full production use
                      </Typography>
                    </Box>
                  }
                />
              </Paper>
              <Paper variant="outlined" sx={{ p: 2, mb: 1 }}>
                <FormControlLabel
                  value="development"
                  control={<Radio />}
                  label={
                    <Box>
                      <Typography variant="subtitle1" fontWeight="600">
                        Development License
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        ${(protocol.licensing.base_price || 0) * 0.5}/month • Testing only
                      </Typography>
                    </Box>
                  }
                />
              </Paper>
              <Paper variant="outlined" sx={{ p: 2 }}>
                <FormControlLabel
                  value="enterprise"
                  control={<Radio />}
                  label={
                    <Box>
                      <Typography variant="subtitle1" fontWeight="600">
                        Enterprise License
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Custom pricing • SLA included • Contact sales
                      </Typography>
                    </Box>
                  }
                />
              </Paper>
            </RadioGroup>
          </Box>
        );

      case 1:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Payment Information
            </Typography>
            <TextField
              fullWidth
              label="Billing Email"
              type="email"
              value={formData.billing_email}
              onChange={(e) =>
                setFormData({ ...formData, billing_email: e.target.value })
              }
              required
              sx={{ mb: 2 }}
              helperText="Receipt will be sent to this email"
            />
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Payment Method
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Demo Mode: Payment processing is simulated
              </Typography>
              <Typography variant="caption" color="text.secondary">
                In production, Stripe payment form would appear here
              </Typography>
            </Paper>
          </Box>
        );

      case 2:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Confirm Purchase
            </Typography>
            <Paper variant="outlined" sx={{ p: 2, mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Order Summary
              </Typography>
              <Divider sx={{ my: 1 }} />
              <Box display="flex" justifyContent="space-between" mb={1}>
                <Typography variant="body2">Protocol:</Typography>
                <Typography variant="body2" fontWeight="600">
                  {protocol.display_name}
                </Typography>
              </Box>
              <Box display="flex" justifyContent="space-between" mb={1}>
                <Typography variant="body2">License Type:</Typography>
                <Typography variant="body2" fontWeight="600">
                  {formData.license_type}
                </Typography>
              </Box>
              <Box display="flex" justifyContent="space-between" mb={1}>
                <Typography variant="body2">Billing Email:</Typography>
                <Typography variant="body2" fontWeight="600">
                  {formData.billing_email}
                </Typography>
              </Box>
              <Divider sx={{ my: 1 }} />
              <Box display="flex" justifyContent="space-between">
                <Typography variant="subtitle1" fontWeight="600">
                  Total:
                </Typography>
                <Typography variant="subtitle1" fontWeight="600" color="primary">
                  ${protocol.licensing.base_price}/month
                </Typography>
              </Box>
            </Paper>

            <FormGroup>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={agreedToTerms}
                    onChange={(e) => setAgreedToTerms(e.target.checked)}
                  />
                }
                label={
                  <Typography variant="body2">
                    I agree to the terms and conditions
                  </Typography>
                }
              />
            </FormGroup>
          </Box>
        );

      case 3:
        return (
          <Box textAlign="center">
            <CheckCircleIcon color="success" sx={{ fontSize: 64, mb: 2 }} />
            <Typography variant="h5" gutterBottom>
              Purchase Successful!
            </Typography>
            <Typography variant="body1" color="text.secondary" paragraph>
              Thank you for purchasing {protocol.display_name}!
            </Typography>

            {purchaseResult && (
              <Paper variant="outlined" sx={{ p: 2, mt: 3, textAlign: 'left' }}>
                <Typography variant="subtitle2" gutterBottom fontWeight="600">
                  License Key:
                </Typography>
                <Typography
                  variant="body2"
                  sx={{
                    fontFamily: 'monospace',
                    bgcolor: 'grey.100',
                    p: 1,
                    borderRadius: 1,
                    mb: 2,
                  }}
                >
                  {purchaseResult.license_key}
                </Typography>

                <Typography variant="subtitle2" gutterBottom fontWeight="600">
                  Installation ID:
                </Typography>
                <Typography
                  variant="body2"
                  sx={{
                    fontFamily: 'monospace',
                    bgcolor: 'grey.100',
                    p: 1,
                    borderRadius: 1,
                    mb: 2,
                  }}
                >
                  {purchaseResult.installation_id}
                </Typography>

                <Typography variant="subtitle2" gutterBottom fontWeight="600">
                  Download Files:
                </Typography>
                <List dense>
                  <ListItem
                    button
                    onClick={() => handleDownload(purchaseResult.download_urls.spec, 'protocol-spec.yaml')}
                  >
                    <DownloadIcon sx={{ mr: 1 }} fontSize="small" />
                    <ListItemText primary="Protocol Specification" />
                  </ListItem>
                  {purchaseResult.download_urls.parser && (
                    <ListItem
                      button
                      onClick={() => handleDownload(purchaseResult.download_urls.parser!, 'parser.py')}
                    >
                      <DownloadIcon sx={{ mr: 1 }} fontSize="small" />
                      <ListItemText primary="Parser Code" />
                    </ListItem>
                  )}
                  {purchaseResult.download_urls.docs && (
                    <ListItem
                      button
                      onClick={() => handleDownload(purchaseResult.download_urls.docs!, 'documentation.pdf')}
                    >
                      <DownloadIcon sx={{ mr: 1 }} fontSize="small" />
                      <ListItemText primary="Documentation" />
                    </ListItem>
                  )}
                </List>
              </Paper>
            )}
          </Box>
        );

      default:
        return null;
    }
  };

  const isNextDisabled = () => {
    if (protocol.licensing.license_type === 'free') {
      return false;
    }

    switch (activeStep) {
      case 1:
        return !formData.billing_email;
      case 2:
        return !agreedToTerms;
      default:
        return false;
    }
  };

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="sm" fullWidth>
      <DialogTitle>
        {activeStep === steps.length - 1
          ? 'Purchase Complete'
          : `Purchase ${protocol.display_name}`}
      </DialogTitle>
      <DialogContent>
        {activeStep < steps.length - 1 && (
          <Stepper activeStep={activeStep} sx={{ mb: 3 }}>
            {steps.map((label) => (
              <Step key={label}>
                <StepLabel>{label}</StepLabel>
              </Step>
            ))}
          </Stepper>
        )}

        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {renderStepContent()}
      </DialogContent>
      <DialogActions>
        {activeStep === steps.length - 1 ? (
          <Button variant="contained" onClick={handleClose} fullWidth>
            Close
          </Button>
        ) : (
          <>
            <Button onClick={handleClose}>Cancel</Button>
            {activeStep > 0 && (
              <Button onClick={handleBack} disabled={loading}>
                Back
              </Button>
            )}
            <Button
              variant="contained"
              onClick={handleNext}
              disabled={loading || isNextDisabled()}
            >
              {loading ? (
                <CircularProgress size={24} />
              ) : activeStep === steps.length - 2 ? (
                protocol.licensing.license_type === 'free' ? 'Install' : 'Complete Purchase'
              ) : (
                'Next'
              )}
            </Button>
          </>
        )}
      </DialogActions>
    </Dialog>
  );
};

export default PurchaseModal;
