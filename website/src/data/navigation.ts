export interface NavItem {
  label: string;
  href: string;
  children?: NavItem[];
  description?: string;
  icon?: string;
  badge?: string;
}

export const mainNavigation: NavItem[] = [
  {
    label: 'Platform',
    href: '/products',
    children: [
      { label: 'Platform Overview', href: '/products', description: 'The focused five-module product surface', icon: 'layers' },
      { label: 'Discover', href: '/products/protocol-discovery', description: 'Inventory legacy protocols and build the protocol graph', icon: 'search' },
      { label: 'Understand', href: '/products/legacy-whisperer', description: 'Explain behavior, risk, fraud, and failure patterns', icon: 'brain' },
      { label: 'Modernize', href: '/products/translation-studio', description: 'Generate specs, adapters, APIs, and replay harnesses', icon: 'code' },
      { label: 'Protect', href: '/products/post-quantum-crypto', description: 'Plan and apply governed post-quantum protection', icon: 'lock' },
      { label: 'Prove', href: '/products/enterprise-compliance', description: 'Bundle evidence, approvals, lineage, and attestation', icon: 'check' },
    ],
  },
  {
    label: 'Industries',
    href: '/industries',
    children: [
      { label: 'BPO / Call Center', href: '/industries/bpo-call-center', description: 'Voice-channel fraud, PCI evidence, and SIP/RTP protection' },
      { label: 'Banking & Finance', href: '/industries/banking', description: 'Mainframe modernization and payment-rail protection' },
      { label: 'IoT', href: '/industries/iot', description: 'Device trust, crypto posture, and protocol inventory' },
      { label: 'Defense', href: '/industries/defense', description: 'Air-gapped modernization and CNSA 2.0 readiness' },
      { label: 'Critical Infrastructure', href: '/industries/critical-infrastructure', description: 'OT protocol inventory and governed overlay rollout' },
      { label: 'Healthcare', href: '/industries/healthcare', description: 'Legacy device interoperability with low-change protection' },
      { label: 'Telecommunications', href: '/industries/telecommunications', description: 'Signaling modernization at network scale' },
    ],
  },
  { label: 'Demo Lab', href: '/demo-lab', badge: 'Interactive' },
  {
    label: 'Docs',
    href: '/docs',
    children: [
      { label: 'Quick Start', href: '/docs/getting-started/quickstart', description: 'Deploy in under 5 minutes' },
      { label: 'Architecture', href: '/docs/architecture/overview', description: '4-layer system design' },
      { label: 'PQC Algorithms', href: '/docs/security/pqc-algorithms', description: 'Algorithm families and implementation layers' },
      { label: 'API Reference', href: '/docs/api/rest-api', description: 'REST & gRPC API documentation' },
      { label: 'Deployment', href: '/docs/deployment/docker', description: 'Production deployment guides' },
      { label: 'Development', href: '/docs/development/python', description: 'Contributing and local development setup' },
    ],
  },
  { label: 'Roadmap', href: '/roadmap' },
  { label: 'PQC Audit', href: '/pqc-audit' },
  { label: 'About', href: '/about' },
];

export const footerNavigation = {
  platform: [
    { label: 'Platform Overview', href: '/products' },
    { label: 'Discover', href: '/products/protocol-discovery' },
    { label: 'Understand', href: '/products/legacy-whisperer' },
    { label: 'Modernize', href: '/products/translation-studio' },
    { label: 'Protect', href: '/products/post-quantum-crypto' },
    { label: 'Prove', href: '/products/enterprise-compliance' },
  ],
  resources: [
    { label: 'Interactive Demo Lab', href: '/demo-lab' },
    { label: 'Documentation', href: '/docs' },
    { label: 'Quick Start', href: '/docs/getting-started/quickstart' },
    { label: 'PQC Audit', href: '/pqc-audit' },
    { label: 'API Reference', href: '/docs/api/rest-api' },
    { label: 'Roadmap', href: '/roadmap' },
    { label: 'Infographics & Media', href: '/resources' },
  ],
  company: [
    { label: 'About', href: '/about' },
    { label: 'Enterprise', href: '/enterprise' },
    { label: 'Security Policy', href: '/security' },
    { label: 'License (Apache 2.0)', href: 'https://github.com/yazhsab/qbitel-bridge/blob/main/LICENSE' },
  ],
  community: [
    { label: 'GitHub', href: 'https://github.com/yazhsab/qbitel-bridge' },
    { label: 'Contributing', href: '/community/contributing' },
    { label: 'Code of Conduct', href: '/community/code-of-conduct' },
    { label: 'Discussions', href: 'https://github.com/yazhsab/qbitel-bridge/discussions' },
  ],
};
