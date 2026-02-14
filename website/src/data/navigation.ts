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
      { label: 'AI Protocol Discovery', href: '/products/protocol-discovery', description: '5-phase AI pipeline (89%+ accuracy)', icon: 'search' },
      { label: 'Translation Studio', href: '/products/translation-studio', description: 'Auto-generate APIs & SDKs in 6 languages', icon: 'code' },
      { label: 'Protocol Marketplace', href: '/products/protocol-marketplace', description: '1,000+ community protocol definitions', icon: 'store' },
      { label: 'Zero-Touch Security', href: '/products/zero-touch-security', description: 'LLM-powered autonomous SOC (<1s response)', icon: 'shield' },
      { label: 'Post-Quantum Crypto', href: '/products/post-quantum-crypto', description: 'NIST FIPS 203/204 Level 5 PQC', icon: 'lock' },
      { label: 'Cloud-Native Security', href: '/products/cloud-native-security', description: 'Quantum-safe K8s & service mesh', icon: 'cloud' },
      { label: 'Enterprise Compliance', href: '/products/enterprise-compliance', description: '9 frameworks, audit-ready in minutes', icon: 'check' },
      { label: 'Legacy Whisperer', href: '/products/legacy-whisperer', description: 'COBOL & mainframe modernization', icon: 'terminal' },
      { label: 'Threat Intelligence', href: '/products/threat-intelligence', description: 'MITRE ATT&CK + AI threat hunting', icon: 'eye' },
      { label: 'IAM Monitoring', href: '/products/iam-monitoring', description: 'UEBA & multi-cloud identity', icon: 'users' },
    ],
  },
  {
    label: 'Industries',
    href: '/industries',
    children: [
      { label: 'Banking & Finance', href: '/industries/banking', description: 'ISO 8583, SWIFT, FIX protocol security' },
      { label: 'Healthcare', href: '/industries/healthcare', description: 'HL7, FHIR, DICOM device protection' },
      { label: 'Critical Infrastructure', href: '/industries/critical-infrastructure', description: 'Modbus, DNP3, IEC 61850 for SCADA' },
      { label: 'Automotive', href: '/industries/automotive', description: 'V2X, CAN bus quantum-safe comms' },
      { label: 'Aviation', href: '/industries/aviation', description: 'ADS-B, ACARS, ARINC 429 protection' },
      { label: 'Telecommunications', href: '/industries/telecommunications', description: 'Diameter, SS7, SIP core security' },
    ],
  },
  {
    label: 'Docs',
    href: '/docs',
    children: [
      { label: 'Quick Start', href: '/docs/getting-started/quickstart', description: 'Deploy in under 5 minutes' },
      { label: 'Installation', href: '/docs/getting-started/installation', description: 'Docker, K8s, Helm, air-gapped' },
      { label: 'Architecture', href: '/docs/architecture/overview', description: '4-layer system design' },
      { label: 'API Reference', href: '/docs/api/rest-api', description: 'REST & gRPC API documentation' },
      { label: 'Deployment', href: '/docs/deployment/docker', description: 'Production deployment guides' },
      { label: 'Development', href: '/docs/development/python', description: 'Contributing & dev setup' },
    ],
  },
  { label: 'Tutorials', href: '/tutorials' },
  { label: 'Resources', href: '/resources' },
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
    { label: 'Cloud-Native Security', href: '/products/cloud-native-security' },
    { label: 'Legacy Whisperer', href: '/products/legacy-whisperer' },
  ],
  resources: [
    { label: 'Documentation', href: '/docs' },
    { label: 'Quick Start', href: '/docs/getting-started/quickstart' },
    { label: 'Tutorials', href: '/tutorials' },
    { label: 'API Reference', href: '/docs/api/rest-api' },
    { label: 'Blog', href: '/blog' },
    { label: 'Roadmap', href: '/roadmap' },
    { label: 'Infographics & Media', href: '/resources' },
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
