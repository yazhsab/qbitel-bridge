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
    tagline: 'Months of reverse engineering in hours',
    description: 'A 5-phase automated pipeline using statistical analysis, ML classification (CNN + BiLSTM + Transformer ensemble), PCFG grammar inference, parser generation, and adaptive learning to reverse-engineer any protocol — documented or not.',
    icon: 'search',
    href: '/products/protocol-discovery',
    color: 'from-blue-500 to-cyan-500',
    features: ['5-phase AI pipeline with adaptive learning', '89%+ classification accuracy across 150+ protocols', '50K+ msg/sec auto-generated parser throughput', 'Works with zero documentation'],
    metrics: [
      { label: 'Accuracy', value: '89%+' },
      { label: 'Discovery', value: '2-4 hrs' },
      { label: 'Throughput', value: '50K msg/s' },
    ],
  },
  {
    id: 'translation-studio',
    title: 'Translation Studio',
    tagline: 'Legacy to modern APIs without code rewrites',
    description: 'Instantly generate OpenAPI 3.0 specifications and production-ready SDKs in Python, TypeScript, Go, Java, Rust, and C# from any discovered protocol grammar. Deploy as API gateways with built-in observability.',
    icon: 'code',
    href: '/products/translation-studio',
    color: 'from-purple-500 to-pink-500',
    features: ['OpenAPI 3.0 + gRPC generation', '6 language SDK generation', '10K+ req/sec bridge throughput', 'Prometheus + Grafana observability'],
    metrics: [
      { label: 'API Gen', value: '<2s' },
      { label: 'Languages', value: '6' },
      { label: 'Throughput', value: '10K req/s' },
    ],
  },
  {
    id: 'protocol-marketplace',
    title: 'Protocol Marketplace',
    tagline: 'Community-powered protocol intelligence',
    description: 'A two-sided marketplace for pre-built protocol definitions covering banking, healthcare, industrial, telecom, IoT, and legacy systems. Publish, discover, rate, and monetize protocol parsers with a 4-step validation pipeline.',
    icon: 'store',
    href: '/products/protocol-marketplace',
    color: 'from-green-500 to-emerald-500',
    features: ['1,000+ protocol definitions', '4-step validation pipeline', 'Stripe Connect monetization', '7 industry categories'],
  },
  {
    id: 'zero-touch-security',
    title: 'Zero-Touch Security Engine',
    tagline: 'From 65 minutes to under 1 second',
    description: 'AI-driven autonomous security operations with LLM-powered decision engine. Reduces SOC response time from 65 minutes to sub-second. On-premise Ollama support ensures sensitive data never leaves your network perimeter.',
    icon: 'shield',
    href: '/products/zero-touch-security',
    color: 'from-red-500 to-orange-500',
    features: ['78% auto-execute rate with confidence scoring', '<1 second decision time (900x faster)', 'On-premise LLM inference (Ollama)', 'Full MITRE ATT&CK technique mapping'],
    metrics: [
      { label: 'Autonomy', value: '78%' },
      { label: 'Response', value: '<1s' },
      { label: 'Accuracy', value: '94%' },
    ],
  },
  {
    id: 'post-quantum-crypto',
    title: 'Post-Quantum Cryptography',
    tagline: 'NIST-standard quantum-safe encryption',
    description: 'Protect against harvest-now-decrypt-later attacks with NIST FIPS 203/204 compliant algorithms: ML-KEM (Kyber-1024), ML-DSA (Dilithium-5), Falcon-1024, and SLH-DSA. Industry-optimized PQC profiles for banking, healthcare, automotive, aviation, and industrial environments.',
    icon: 'lock',
    href: '/products/post-quantum-crypto',
    color: 'from-cyan-500 to-blue-500',
    features: ['ML-KEM + ML-DSA + Falcon + SLH-DSA', 'Industry-specific PQC optimization profiles', 'Hybrid classical + quantum transition modes', '<1ms encryption overhead at wire speed'],
    metrics: [
      { label: 'Security', value: 'NIST 5' },
      { label: 'Latency', value: '<1ms' },
      { label: 'Throughput', value: '10K msg/s' },
    ],
  },
  {
    id: 'cloud-native-security',
    title: 'Cloud-Native Security',
    tagline: 'Quantum-safe Kubernetes at scale',
    description: 'End-to-end cloud-native security with quantum-safe mTLS for Istio/Envoy service mesh, eBPF kernel-level runtime monitoring, container vulnerability scanning, and encrypted Kafka event streaming at 100K+ messages per second.',
    icon: 'cloud',
    href: '/products/cloud-native-security',
    color: 'from-indigo-500 to-violet-500',
    features: ['Quantum-safe Istio/Envoy service mesh', 'eBPF runtime monitoring (<1% CPU)', 'Container image vulnerability scanning', 'AES-256-GCM Kafka at 100K+ msg/sec'],
    metrics: [
      { label: 'eBPF Overhead', value: '<1% CPU' },
      { label: 'Scale', value: '10K+/node' },
      { label: 'Streaming', value: '100K msg/s' },
    ],
  },
  {
    id: 'enterprise-compliance',
    title: 'Enterprise Compliance',
    tagline: 'Audit-ready in minutes, not months',
    description: 'Continuous compliance monitoring with automated evidence collection across SOC 2, GDPR, HIPAA, PCI-DSS, ISO 27001, NIST CSF, NERC CIP, FedRAMP, and CMMC. Generate audit-ready reports with cryptographic proof in under 10 minutes.',
    icon: 'check',
    href: '/products/enterprise-compliance',
    color: 'from-emerald-500 to-teal-500',
    features: ['9 regulatory frameworks supported', '80-90% compliance automation rate', '<10 min audit-ready report generation', 'Continuous evidence collection with crypto proof'],
  },
  {
    id: 'legacy-whisperer',
    title: 'Legacy System Whisperer',
    tagline: 'Decode COBOL, modernize mainframes',
    description: 'Deep analysis of COBOL copybooks, JCL jobs, and mainframe datasets. Extract business rules, map data dependencies, predict hardware failures with ML, and auto-generate modern adapters in 6 languages.',
    icon: 'terminal',
    href: '/products/legacy-whisperer',
    color: 'from-amber-500 to-yellow-500',
    features: ['COBOL copybook + JCL parsing', 'Business rule extraction engine', 'ML-powered predictive failure analysis', 'Modern adapter generation (6 languages)'],
  },
  {
    id: 'threat-intelligence',
    title: 'Threat Intelligence Platform',
    tagline: 'Proactive threat hunting with AI',
    description: 'Automated threat detection with MITRE ATT&CK integration, STIX/TAXII intelligence feeds, YARA + Sigma rule execution, and AI-driven threat hunting with behavioral analysis.',
    icon: 'eye',
    href: '/products/threat-intelligence',
    color: 'from-rose-500 to-red-500',
    features: ['Full MITRE ATT&CK technique mapping', 'STIX/TAXII threat intelligence feeds', 'YARA + Sigma rule engine', 'AI-powered automated threat hunting'],
  },
  {
    id: 'iam-monitoring',
    title: 'Enterprise IAM Monitoring',
    tagline: 'Identity intelligence across your stack',
    description: 'Monitor and secure identity infrastructure with UEBA behavioral analytics, privilege escalation detection, session anomaly monitoring, and multi-cloud IAM integration across AWS, Azure, and GCP.',
    icon: 'users',
    href: '/products/iam-monitoring',
    color: 'from-sky-500 to-blue-500',
    features: ['UEBA behavioral analytics engine', 'Privilege escalation detection', 'Multi-cloud IAM (AWS, Azure, GCP)', 'Real-time session anomaly monitoring'],
  },
];
