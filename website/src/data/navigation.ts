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
    label: 'Products',
    href: '/products',
    children: [
      { label: 'AI Protocol Discovery', href: '/products/protocol-discovery', description: '5-phase automated discovery pipeline', icon: 'search' },
      { label: 'Translation Studio', href: '/products/translation-studio', description: 'Auto-generate APIs & SDKs', icon: 'code' },
      { label: 'Protocol Marketplace', href: '/products/protocol-marketplace', description: 'Community protocol sharing', icon: 'store' },
      { label: 'Zero-Touch Security', href: '/products/zero-touch-security', description: 'LLM-powered autonomous response', icon: 'shield' },
      { label: 'Post-Quantum Crypto', href: '/products/post-quantum-crypto', description: 'NIST Level 5 PQC', icon: 'lock' },
      { label: 'Cloud-Native Security', href: '/products/cloud-native-security', description: 'Kubernetes & service mesh', icon: 'cloud' },
      { label: 'Enterprise Compliance', href: '/products/enterprise-compliance', description: '9 regulatory frameworks', icon: 'check' },
      { label: 'Legacy Whisperer', href: '/products/legacy-whisperer', description: 'COBOL & mainframe analysis', icon: 'terminal' },
      { label: 'Threat Intelligence', href: '/products/threat-intelligence', description: 'MITRE ATT&CK integration', icon: 'eye' },
      { label: 'IAM Monitoring', href: '/products/iam-monitoring', description: 'Identity & access monitoring', icon: 'users' },
    ],
  },
  {
    label: 'Industries',
    href: '/industries',
    children: [
      { label: 'Banking & Finance', href: '/industries/banking', description: 'ISO 8583, SWIFT, FIX' },
      { label: 'Healthcare', href: '/industries/healthcare', description: 'HL7, FHIR, DICOM' },
      { label: 'Critical Infrastructure', href: '/industries/critical-infrastructure', description: 'Modbus, DNP3, IEC 61850' },
      { label: 'Automotive', href: '/industries/automotive', description: 'V2X, CAN, IEEE 1609.2' },
      { label: 'Aviation', href: '/industries/aviation', description: 'ADS-B, ACARS, ARINC 429' },
      { label: 'Telecommunications', href: '/industries/telecommunications', description: 'Diameter, SS7, SIP' },
    ],
  },
  {
    label: 'Docs',
    href: '/docs',
    children: [
      { label: 'Quick Start', href: '/docs/getting-started/quickstart', description: 'Get running in 10 minutes' },
      { label: 'Installation', href: '/docs/getting-started/installation', description: 'Full setup guide' },
      { label: 'Architecture', href: '/docs/architecture/overview', description: 'System design & components' },
      { label: 'API Reference', href: '/docs/api/rest-api', description: 'REST & gRPC APIs' },
      { label: 'Deployment', href: '/docs/deployment/docker', description: 'Docker, K8s, Helm' },
      { label: 'Development', href: '/docs/development/python', description: 'Contributing guide' },
    ],
  },
  { label: 'Tutorials', href: '/tutorials' },
  { label: 'Community', href: '/community' },
  { label: 'Blog', href: '/blog' },
];

export const footerNavigation = {
  products: [
    { label: 'Protocol Discovery', href: '/products/protocol-discovery' },
    { label: 'Translation Studio', href: '/products/translation-studio' },
    { label: 'Zero-Touch Security', href: '/products/zero-touch-security' },
    { label: 'Post-Quantum Crypto', href: '/products/post-quantum-crypto' },
    { label: 'Protocol Marketplace', href: '/products/protocol-marketplace' },
    { label: 'Enterprise Compliance', href: '/products/enterprise-compliance' },
  ],
  resources: [
    { label: 'Documentation', href: '/docs' },
    { label: 'Quick Start', href: '/docs/getting-started/quickstart' },
    { label: 'Tutorials', href: '/tutorials' },
    { label: 'API Reference', href: '/docs/api/rest-api' },
    { label: 'Blog', href: '/blog' },
    { label: 'Roadmap', href: '/roadmap' },
  ],
  community: [
    { label: 'GitHub', href: 'https://github.com/yazhsab/qbitel-bridge' },
    { label: 'Contributing', href: '/community/contributing' },
    { label: 'Code of Conduct', href: '/community/code-of-conduct' },
    { label: 'Security Policy', href: '/security' },
    { label: 'Discussions', href: 'https://github.com/yazhsab/qbitel-bridge/discussions' },
  ],
  company: [
    { label: 'About', href: '/about' },
    { label: 'Enterprise', href: '/enterprise' },
    { label: 'License (Apache 2.0)', href: 'https://github.com/yazhsab/qbitel-bridge/blob/main/LICENSE' },
  ],
};
