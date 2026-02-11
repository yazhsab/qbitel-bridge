export interface SidebarSection {
  title: string;
  items: { label: string; href: string }[];
}

export const docsSidebar: SidebarSection[] = [
  {
    title: 'Getting Started',
    items: [
      { label: 'Quick Start', href: '/docs/getting-started/quickstart' },
      { label: 'Installation', href: '/docs/getting-started/installation' },
      { label: 'Configuration', href: '/docs/getting-started/configuration' },
      { label: 'First Discovery', href: '/docs/getting-started/first-discovery' },
    ],
  },
  {
    title: 'Architecture',
    items: [
      { label: 'System Overview', href: '/docs/architecture/overview' },
      { label: 'AI Engine', href: '/docs/architecture/ai-engine' },
      { label: 'Rust Data Plane', href: '/docs/architecture/data-plane' },
      { label: 'Go Control Plane', href: '/docs/architecture/control-plane' },
      { label: 'UI Console', href: '/docs/architecture/ui-console' },
    ],
  },
  {
    title: 'Deployment',
    items: [
      { label: 'Docker Compose', href: '/docs/deployment/docker' },
      { label: 'Kubernetes', href: '/docs/deployment/kubernetes' },
      { label: 'Helm Charts', href: '/docs/deployment/helm' },
      { label: 'Air-Gapped', href: '/docs/deployment/air-gapped' },
      { label: 'Production Checklist', href: '/docs/deployment/production' },
    ],
  },
  {
    title: 'Development',
    items: [
      { label: 'Python', href: '/docs/development/python' },
      { label: 'Rust', href: '/docs/development/rust' },
      { label: 'Go', href: '/docs/development/go' },
      { label: 'React/TypeScript', href: '/docs/development/react' },
      { label: 'Testing', href: '/docs/development/testing' },
    ],
  },
  {
    title: 'API Reference',
    items: [
      { label: 'REST API', href: '/docs/api/rest-api' },
      { label: 'gRPC API', href: '/docs/api/grpc-api' },
      { label: 'Authentication', href: '/docs/api/authentication' },
    ],
  },
  {
    title: 'Security',
    items: [
      { label: 'PQC Algorithms', href: '/docs/security/pqc-algorithms' },
      { label: 'Zero-Trust Architecture', href: '/docs/security/zero-trust' },
      { label: 'Compliance Frameworks', href: '/docs/security/compliance-frameworks' },
    ],
  },
  {
    title: 'Operations',
    items: [
      { label: 'Monitoring', href: '/docs/operations/monitoring' },
      { label: 'Troubleshooting', href: '/docs/operations/troubleshooting' },
    ],
  },
  {
    title: 'Reference',
    items: [
      { label: 'Environment Variables', href: '/docs/reference/environment-variables' },
      { label: 'Changelog', href: '/docs/reference/changelog' },
    ],
  },
];
