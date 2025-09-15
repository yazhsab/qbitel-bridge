import React from 'react';
import { Box, Typography, Card, CardContent } from '@mui/material';
import { DeviceApiClient } from '../api/devices';

interface SystemSettingsProps {
  apiClient: DeviceApiClient;
}

const SystemSettings: React.FC<SystemSettingsProps> = ({ apiClient }) => {
  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        System Settings
      </Typography>
      <Card>
        <CardContent>
          <Typography variant="body1">
            System configuration and settings interface will be implemented here.
            This will include QSLB system configuration, user management, and administrative settings.
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
};

export default SystemSettings;