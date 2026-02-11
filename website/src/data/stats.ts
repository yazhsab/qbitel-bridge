export interface Stat {
  value: string;
  label: string;
  description?: string;
}

export const heroStats: Stat[] = [
  { value: '89%+', label: 'Discovery Accuracy', description: 'CNN + BiLSTM + Transformer ensemble' },
  { value: 'Level 5', label: 'NIST PQC', description: '256-bit quantum-safe (ML-KEM, ML-DSA)' },
  { value: '78%', label: 'Autonomous Response', description: 'Zero-touch security decisions via LLM' },
  { value: '<1ms', label: 'Encryption Overhead', description: 'Hybrid PQC + classical processing' },
  { value: '9', label: 'Compliance Frameworks', description: 'SOC2, GDPR, HIPAA, PCI-DSS & more' },
  { value: '100K+', label: 'Events/sec', description: 'Encrypted Kafka streaming (AES-256-GCM)' },
];

export const performanceStats: Stat[] = [
  { value: '2-4 hrs', label: 'Protocol Discovery', description: 'vs 6-12 months manual reverse engineering' },
  { value: '50K+', label: 'msg/sec Parsing', description: 'Auto-generated parser throughput' },
  { value: '<1s', label: 'Threat Response', description: '900x faster than manual SOC triage' },
  { value: '10K+', label: 'Containers/Node', description: 'eBPF monitoring with <1% CPU overhead' },
  { value: '150ms', label: 'P95 Latency', description: 'End-to-end protocol discovery request' },
  { value: '6', label: 'SDK Languages', description: 'Python, TypeScript, Go, Java, Rust, C#' },
];

export const enterpriseStats: Stat[] = [
  { value: '$2-10M', label: 'Traditional Cost', description: 'Manual protocol reverse engineering' },
  { value: '$200K', label: 'With QBitel', description: 'AI-automated protocol discovery' },
  { value: '95%', label: 'Cost Reduction', description: 'On protocol discovery projects' },
  { value: '24/7', label: 'Autonomous Ops', description: 'Zero-touch security with audit trails' },
];
