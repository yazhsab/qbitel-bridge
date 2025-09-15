# QSLB Admin Console

Enterprise-grade management interface for the Quantum-Safe Load Balancer (QSLB) system.

## Overview

The QSLB Admin Console is a React-based web application that provides comprehensive management capabilities for the QSLB infrastructure. It features device lifecycle management, policy administration, compliance monitoring, security oversight, and system configuration.

## Features

### 🔐 Authentication & Authorization
- **OIDC Integration**: Secure authentication with OpenID Connect
- **Multi-Factor Authentication**: Support for TOTP, SMS, and WebAuthn
- **Role-Based Access Control**: Granular permissions and role management
- **Session Management**: Automatic token renewal and session timeout
- **Audit Logging**: Comprehensive audit trail for all user actions

### 📱 Device Management
- **Device Lifecycle**: Complete device enrollment, provisioning, and decommissioning
- **TPM Attestation**: Hardware-based device identity verification
- **Certificate Management**: Automated certificate issuance, renewal, and revocation
- **Compliance Monitoring**: Real-time compliance status and violation tracking
- **Health Monitoring**: Device health checks and performance metrics

### 📋 Policy Management
- **Policy Creation**: Define and manage device policies
- **Policy Deployment**: Automated policy distribution to devices
- **Policy Compliance**: Monitor policy adherence and violations
- **Policy Versioning**: Track policy changes and rollback capabilities

### 📊 Monitoring & Analytics
- **Real-time Dashboard**: Live system metrics and status overview
- **Security Monitoring**: Threat detection and incident response
- **Compliance Reporting**: Automated compliance reports and attestations
- **Performance Analytics**: System performance metrics and trends

### ⚙️ System Administration
- **User Management**: User accounts, roles, and permissions
- **System Configuration**: QSLB system settings and parameters
- **Integration Management**: External system integrations and APIs
- **Backup & Recovery**: System backup and disaster recovery

## Technology Stack

- **Frontend**: React 18 with TypeScript
- **UI Framework**: Material-UI (MUI) v5
- **State Management**: React Hooks and Context API
- **Authentication**: OIDC Client TS
- **Build Tool**: Vite
- **Styling**: CSS-in-JS with MUI theming
- **Charts**: Recharts for data visualization
- **HTTP Client**: Fetch API with custom error handling

## Prerequisites

- Node.js 18.0.0 or higher
- npm 9.0.0 or higher
- Modern web browser with ES2020 support

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd qslb/ui/console
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Start development server**:
   ```bash
   npm run dev
   ```

5. **Build for production**:
   ```bash
   npm run build
   ```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_BASE_URL` | QSLB API base URL | `https://api.qslb.local` |
| `VITE_OIDC_AUTHORITY` | OIDC provider URL | `https://auth.qslb.local` |
| `VITE_OIDC_CLIENT_ID` | OIDC client identifier | `qslb-console` |
| `VITE_ENABLE_MFA` | Enable multi-factor authentication | `true` |
| `VITE_SESSION_TIMEOUT` | Session timeout in milliseconds | `28800000` (8 hours) |
| `VITE_AUDIT_ENABLED` | Enable audit logging | `true` |

### OIDC Configuration

The console requires an OIDC provider configured with:
- **Client Type**: Public (SPA)
- **Grant Types**: Authorization Code with PKCE
- **Redirect URIs**: `https://console.qslb.local/auth/callback`
- **Post Logout URIs**: `https://console.qslb.local/`
- **Scopes**: `openid profile email roles permissions organization`

## Development

### Project Structure

```
src/
├── components/          # React components
│   ├── DeviceManagement.tsx
│   ├── Dashboard.tsx
│   ├── PolicyManagement.tsx
│   ├── ComplianceReporting.tsx
│   ├── SecurityMonitoring.tsx
│   ├── SystemSettings.tsx
│   └── UserProfile.tsx
├── auth/               # Authentication services
│   └── oidc.ts
├── api/                # API clients
│   └── devices.ts
├── types/              # TypeScript type definitions
│   ├── auth.ts
│   ├── device.ts
│   └── env.d.ts
├── App.tsx             # Main application component
├── main.tsx            # Application entry point
└── index.css           # Global styles
```

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint
- `npm run lint:fix` - Fix ESLint issues
- `npm run type-check` - Run TypeScript type checking

### Code Style

- **TypeScript**: Strict mode enabled with comprehensive type checking
- **ESLint**: Configured with React and TypeScript rules
- **Prettier**: Code formatting (configure in your editor)
- **Material-UI**: Follow MUI design system guidelines

## Security

### Security Features

- **Content Security Policy**: Strict CSP headers
- **HTTPS Only**: All communications over HTTPS
- **Token Security**: Secure token storage and automatic renewal
- **Input Validation**: Client-side and server-side validation
- **XSS Protection**: React's built-in XSS protection
- **CSRF Protection**: CSRF tokens for state-changing operations

### Security Best Practices

1. **Environment Variables**: Never commit sensitive data to version control
2. **Dependencies**: Regularly update dependencies and scan for vulnerabilities
3. **Authentication**: Always verify user authentication and authorization
4. **API Security**: Use proper authentication headers and validate responses
5. **Error Handling**: Don't expose sensitive information in error messages

## Deployment

### Production Build

```bash
# Build the application
npm run build

# The build artifacts will be in the 'dist' directory
# Serve these files with a web server like nginx or Apache
```

### Docker Deployment

```dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Environment-Specific Configuration

- **Development**: Use `.env.development`
- **Staging**: Use `.env.staging`
- **Production**: Use `.env.production`

## API Integration

The console integrates with the QSLB Management API:

- **Base URL**: Configured via `VITE_API_BASE_URL`
- **Authentication**: Bearer token from OIDC provider
- **Error Handling**: Comprehensive error handling with user-friendly messages
- **Retry Logic**: Automatic retry for transient failures
- **Rate Limiting**: Respect API rate limits

## Monitoring

### Application Monitoring

- **Error Tracking**: Automatic error reporting and logging
- **Performance Monitoring**: Core Web Vitals and performance metrics
- **User Analytics**: User interaction tracking (if enabled)
- **Health Checks**: Application health monitoring endpoints

### Logging

- **Client-side Logging**: Structured logging with different levels
- **Audit Logging**: User action audit trail
- **Error Logging**: Comprehensive error logging with stack traces
- **Performance Logging**: Performance metrics and timing data

## Troubleshooting

### Common Issues

1. **Authentication Failures**:
   - Verify OIDC configuration
   - Check network connectivity to auth provider
   - Validate client ID and redirect URIs

2. **API Connection Issues**:
   - Verify API base URL configuration
   - Check CORS settings on the API server
   - Validate authentication tokens

3. **Build Issues**:
   - Clear node_modules and reinstall dependencies
   - Check Node.js and npm versions
   - Verify TypeScript configuration

### Debug Mode

Enable debug logging by setting `VITE_DEBUG_LOGGING=true` in your environment.

## Contributing

1. Follow the existing code style and patterns
2. Add TypeScript types for all new code
3. Include comprehensive error handling
4. Add appropriate logging and monitoring
5. Update documentation for new features
6. Test thoroughly before submitting changes

## License

This project is part of the QSLB system and follows the same licensing terms.

## Support

For support and questions:
- **Documentation**: https://docs.qslb.local
- **Support Email**: support@qslb.local
- **Issue Tracker**: Internal issue tracking system