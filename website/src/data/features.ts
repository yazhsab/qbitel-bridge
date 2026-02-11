export interface Feature {
  title: string;
  description: string;
  icon: string;
  href: string;
  badge?: string;
}

export const coreFeatures: Feature[] = [
  {
    title: 'AI Protocol Discovery',
    description: 'Automatically discover and reverse-engineer undocumented protocols using a 5-phase AI pipeline. Reduce months of manual work to hours.',
    icon: 'M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z',
    href: '/products/protocol-discovery',
    badge: '89%+ Accuracy',
  },
  {
    title: 'Post-Quantum Cryptography',
    description: 'NIST Level 5 quantum-safe encryption with ML-KEM (Kyber-1024), ML-DSA (Dilithium-5), and domain-specific profiles for every industry.',
    icon: 'M16.5 10.5V6.75a4.5 4.5 0 10-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 002.25-2.25v-6.75a2.25 2.25 0 00-2.25-2.25H6.75a2.25 2.25 0 00-2.25 2.25v6.75a2.25 2.25 0 002.25 2.25z',
    href: '/products/post-quantum-crypto',
    badge: 'NIST Level 5',
  },
  {
    title: 'Zero-Touch Security',
    description: 'LLM-powered autonomous security decisions with 78% auto-execution rate. On-premise Ollama support for air-gapped environments.',
    icon: 'M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z',
    href: '/products/zero-touch-security',
    badge: '78% Autonomous',
  },
  {
    title: 'Translation Studio',
    description: 'Auto-generate OpenAPI specs and production-ready SDKs in 6 languages from discovered protocols. Bridge legacy to modern APIs instantly.',
    icon: 'M17.25 6.75L22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3l-4.5 16.5',
    href: '/products/translation-studio',
  },
  {
    title: 'Protocol Marketplace',
    description: 'Community-driven marketplace for pre-built protocol definitions. Share, discover, and monetize protocol parsers and adapters.',
    icon: 'M13.5 21v-7.5a.75.75 0 01.75-.75h3a.75.75 0 01.75.75V21m-4.5 0H2.36m11.14 0H18m0 0h3.64m-1.39 0V9.349m-16.5 11.65V9.35m0 0a3.001 3.001 0 003.75-.615A2.993 2.993 0 009.75 9.75c.896 0 1.7-.393 2.25-1.016a2.993 2.993 0 002.25 1.016c.896 0 1.7-.393 2.25-1.016A3.001 3.001 0 0021 9.349m-18 0a2.999 2.999 0 002.25-1.033 2.999 2.999 0 002.25 1.033 2.999 2.999 0 002.25-1.033A2.999 2.999 0 0012 9.35',
    href: '/products/protocol-marketplace',
  },
  {
    title: 'Enterprise Compliance',
    description: 'Automated continuous compliance for 9 frameworks: SOC2, GDPR, HIPAA, PCI-DSS, ISO 27001, NIST 800-53, NERC CIP, FedRAMP, CMMC.',
    icon: 'M9 12.75L11.25 15 15 9.75M21 12c0 1.268-.63 2.39-1.593 3.068a3.745 3.745 0 01-1.043 3.296 3.745 3.745 0 01-3.296 1.043A3.745 3.745 0 0112 21c-1.268 0-2.39-.63-3.068-1.593a3.746 3.746 0 01-3.296-1.043 3.745 3.745 0 01-1.043-3.296A3.745 3.745 0 013 12c0-1.268.63-2.39 1.593-3.068a3.745 3.745 0 011.043-3.296 3.746 3.746 0 013.296-1.043A3.746 3.746 0 0112 3c1.268 0 2.39.63 3.068 1.593a3.746 3.746 0 013.296 1.043 3.746 3.746 0 011.043 3.296A3.745 3.745 0 0121 12z',
    href: '/products/enterprise-compliance',
    badge: '9 Frameworks',
  },
  {
    title: 'Cloud-Native Security',
    description: 'Quantum-safe service mesh with Istio/Envoy, eBPF runtime monitoring, container scanning, and secure Kafka streaming.',
    icon: 'M2.25 15a4.5 4.5 0 004.5 4.5H18a3.75 3.75 0 001.332-7.257 3 3 0 00-3.758-3.848 5.25 5.25 0 00-10.233 2.33A4.502 4.502 0 002.25 15z',
    href: '/products/cloud-native-security',
  },
  {
    title: 'Legacy Whisperer',
    description: 'Deep analysis of COBOL copybooks, JCL jobs, and mainframe systems. Extract business rules and generate modern adapters.',
    icon: 'M6.75 7.5l3 2.25-3 2.25m4.5 0h3m-9 8.25h13.5A2.25 2.25 0 0021 18V6a2.25 2.25 0 00-2.25-2.25H5.25A2.25 2.25 0 003 6v12a2.25 2.25 0 002.25 2.25z',
    href: '/products/legacy-whisperer',
  },
];

export const howItWorks = [
  { step: '01', title: 'Discover', description: 'AI analyzes network traffic and extracts protocol grammar using PCFG inference and transformers.' },
  { step: '02', title: 'Protect', description: 'Apply NIST Level 5 post-quantum encryption with domain-specific profiles for your industry.' },
  { step: '03', title: 'Translate', description: 'Auto-generate REST APIs, gRPC services, and SDKs in 6 languages from discovered protocols.' },
  { step: '04', title: 'Comply', description: 'Continuous compliance monitoring across 9 regulatory frameworks with automated evidence collection.' },
  { step: '05', title: 'Operate', description: 'Zero-touch autonomous security with LLM-powered decisions and complete audit trails.' },
];
