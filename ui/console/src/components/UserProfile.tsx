import React from 'react';
import { Box, Typography, Card, CardContent } from '@mui/material';
import { User } from '../types/auth';
import { OidcAuthService } from '../auth/oidc';

interface UserProfileProps {
  user: User;
  authService: OidcAuthService;
}

const UserProfile: React.FC<UserProfileProps> = ({ user, authService }) => {
  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        User Profile
      </Typography>
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            {user.profile.name || user.profile.email}
          </Typography>
          <Typography variant="body1">
            User profile and account settings will be implemented here.
            This will include profile information, security settings, and preferences.
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
};

export default UserProfile;