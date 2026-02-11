export interface Product {
  id: string;
  title: string;
  tagline: string;
  description: string;
  icon: string;
  href: string;
  color: string;
  features: string[];
  metrics?: { label: string; value: string }[];
}

export const products: Product[] = [
  {
    id: 'protocol-discovery',
    title: 'AI Protocol Discovery',
    tagline: 'Discover undocumented protocols in hours, not months',
    description: 'A 5-phase automated pipeline using statistical analysis, ML classification (CNN + BiLSTM), grammar learning (PCFG + transformers), parser generation, and adaptive learning to reverse-engineer any protocol.',
    icon: 'search',
    href: '/products/protocol-discovery',
    color: 'from-blue-500 to-cyan-500',
    features: ['5-phase AI pipeline', '89%+ classification accuracy', '50K+ msg/sec parser throughput', '150+ protocol types supported'],
    metrics: [
      { label: 'Accuracy', value: '89%+' },
      { label: 'Discovery Time', value: '2-4 hrs' },
      { label: 'Parser Throughput', value: '50K msg/s' },
    ],
  },
  {
    id: 'translation-studio',
    title: 'Translation Studio',
    tagline: 'Auto-generate APIs and SDKs from discovered protocols',
    description: 'Instantly generate OpenAPI 3.0 specifications and production-ready SDKs in Python, TypeScript, Go, Java, Rust, and C# from any discovered protocol grammar.',
    icon: 'code',
    href: '/products/translation-studio',
    color: 'from-purple-500 to-pink-500',
    features: ['OpenAPI 3.0 generation', '6 language SDKs', '10K+ req/sec bridge throughput', 'API gateway integration'],
    metrics: [
      { label: 'API Gen Time', value: '<2s' },
      { label: 'SDK Languages', value: '6' },
      { label: 'Bridge Throughput', value: '10K req/s' },
    ],
  },
  {
    id: 'protocol-marketplace',
    title: 'Protocol Marketplace',
    tagline: 'Community-driven protocol definition sharing',
    description: 'A two-sided marketplace for pre-built protocol definitions covering banking, healthcare, industrial, telecom, IoT, and legacy systems. Publish, discover, and monetize protocol parsers.',
    icon: 'store',
    href: '/products/protocol-marketplace',
    color: 'from-green-500 to-emerald-500',
    features: ['1,000+ protocol definitions', '4-step validation pipeline', 'Stripe Connect monetization', '7 industry categories'],
  },
  {
    id: 'zero-touch-security',
    title: 'Zero-Touch Security Engine',
    tagline: 'LLM-powered autonomous security operations',
    description: 'Reduce SOC response time from 65 minutes to under 1 second with an AI-driven decision engine. On-premise Ollama support ensures data never leaves your network.',
    icon: 'shield',
    href: '/products/zero-touch-security',
    color: 'from-red-500 to-orange-500',
    features: ['78% auto-execute rate', '<1 second decision time', 'On-premise LLM (Ollama)', 'MITRE ATT&CK mapping'],
    metrics: [
      { label: 'Autonomy', value: '78%' },
      { label: 'Decision Time', value: '<1s' },
      { label: 'Accuracy', value: '94%' },
    ],
  },
  {
    id: 'post-quantum-crypto',
    title: 'Post-Quantum Cryptography',
    tagline: 'NIST Level 5 quantum-safe encryption for every industry',
    description: 'Protect against "harvest now, decrypt later" attacks with ML-KEM (Kyber-1024), ML-DSA (Dilithium-5), Falcon-1024, and SLH-DSA. Domain-specific profiles optimize PQC for banking, healthcare, automotive, aviation, and industrial environments.',
    icon: 'lock',
    href: '/products/post-quantum-crypto',
    color: 'from-cyan-500 to-blue-500',
    features: ['ML-KEM + ML-DSA + Falcon + SLH-DSA', 'Domain-specific PQC profiles', 'Hybrid classical + quantum modes', '<1ms encryption overhead'],
    metrics: [
      { label: 'Security Level', value: 'NIST 5' },
      { label: 'Latency', value: '<1ms' },
      { label: 'Throughput', value: '10K msg/s' },
    ],
  },
  {
    id: 'cloud-native-security',
    title: 'Cloud-Native Security',
    tagline: 'Quantum-safe Kubernetes and service mesh protection',
    description: 'Comprehensive cloud-native security with quantum-safe mTLS for Istio/Envoy, eBPF runtime monitoring, container vulnerability scanning, and secure Kafka event streaming.',
    icon: 'cloud',
    href: '/products/cloud-native-security',
    color: 'from-indigo-500 to-violet-500',
    features: ['Quantum-safe service mesh', 'eBPF runtime monitoring', 'Container image scanning', '100K+ msg/sec Kafka'],
    metrics: [
      { label: 'eBPF Overhead', value: '<1% CPU' },
      { label: 'Containers', value: '10K+/node' },
      { label: 'Kafka', value: '100K msg/s' },
    ],
  },
  {
    id: 'enterprise-compliance',
    title: 'Enterprise Compliance',
    tagline: 'Automated continuous compliance for 9 frameworks',
    description: 'Continuous compliance monitoring with automated evidence collection for SOC 2, GDPR, HIPAA, PCI-DSS, ISO 27001, NIST CSF, NERC CIP, FedRAMP, and CMMC. Generate audit-ready reports in under 10 minutes.',
    icon: 'check',
    href: '/products/enterprise-compliance',
    color: 'from-emerald-500 to-teal-500',
    features: ['9 regulatory frameworks', '80-90% automation rate', '<10 min audit reports', 'Continuous evidence collection'],
  },
  {
    id: 'legacy-whisperer',
    title: 'Legacy System Whisperer',
    tagline: 'Deep COBOL and mainframe analysis',
    description: 'Analyze COBOL copybooks, JCL jobs, and mainframe datasets. Extract business rules, map data flows, predict hardware failures, and generate modern adapters in 6 languages.',
    icon: 'terminal',
    href: '/products/legacy-whisperer',
    color: 'from-amber-500 to-yellow-500',
    features: ['COBOL copybook parser', 'JCL analyzer', 'Business rule extraction', 'Predictive failure analysis'],
  },
  {
    id: 'threat-intelligence',
    title: 'Threat Intelligence Platform',
    tagline: 'Automated threat detection and hunting',
    description: 'MITRE ATT&CK integration with STIX/TAXII feeds, YARA/Sigma rule execution, automated threat hunting, and IOC lifecycle management.',
    icon: 'eye',
    href: '/products/threat-intelligence',
    color: 'from-rose-500 to-red-500',
    features: ['MITRE ATT&CK mapping', 'STIX/TAXII feeds', 'YARA + Sigma rules', 'Automated hunting'],
  },
  {
    id: 'iam-monitoring',
    title: 'Enterprise IAM Monitoring',
    tagline: 'Identity and access management intelligence',
    description: 'Monitor and secure identity infrastructure with behavioral analytics, privilege escalation detection, session management, and multi-cloud IAM integration.',
    icon: 'users',
    href: '/products/iam-monitoring',
    color: 'from-sky-500 to-blue-500',
    features: ['Behavioral analytics', 'Privilege escalation detection', 'Multi-cloud IAM', 'Session monitoring'],
  },
];
