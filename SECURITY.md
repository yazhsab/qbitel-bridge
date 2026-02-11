# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x     | :white_check_mark: |

## Reporting a Vulnerability

The QBITEL Bridge team takes security vulnerabilities seriously. We appreciate your efforts to responsibly disclose any issues you find.

**Please do NOT report security vulnerabilities through public GitHub issues.**

### How to Report

Send an email to **security@qbitel.com** with the following information:

- A description of the vulnerability
- Steps to reproduce the issue
- Affected versions
- Any potential impact you have identified
- (Optional) Suggested fix or mitigation

### What to Expect

- **Acknowledgment**: We will acknowledge receipt of your report within **48 hours**.
- **Assessment**: We will investigate and provide an initial assessment within **5 business days**.
- **Resolution**: We aim to release a fix for confirmed vulnerabilities within **30 days**, depending on complexity.
- **Disclosure**: We will coordinate with you on public disclosure timing. We follow a 90-day disclosure policy.

### Safe Harbor

We consider security research conducted in good faith to be authorized. We will not pursue legal action against researchers who:

- Make a good faith effort to avoid privacy violations, data destruction, or service disruption
- Only interact with accounts they own or with explicit permission
- Report vulnerabilities promptly and do not exploit them beyond what is necessary to confirm the issue

## Security Best Practices for Deployment

- Always use TLS 1.3 in production (`tls_min_version: "TLSv1.3"`)
- Set all secrets via environment variables, never in config files
- Enable audit logging (`audit_logging_enabled: true`)
- Use the `verify-full` SSL mode for database connections in production
- Rotate JWT secrets and encryption keys regularly
- Enable post-quantum cryptography for forward secrecy
- Review the [deployment guide](DEPLOYMENT.md) for hardening recommendations
