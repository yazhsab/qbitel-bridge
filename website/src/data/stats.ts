export interface Stat {
  value: string;
  label: string;
  description?: string;
}

export const heroStats: Stat[] = [
  { value: '89%+', label: 'Discovery Accuracy', description: 'Protocol classification with CNN + BiLSTM' },
  { value: 'Level 5', label: 'NIST PQC', description: '256-bit quantum security (ML-KEM, ML-DSA)' },
  { value: '78%', label: 'Autonomous Response', description: 'Zero-touch security decisions' },
  { value: '<1ms', label: 'Encryption Latency', description: 'Hybrid PQC + classical overhead' },
  { value: '9', label: 'Compliance Frameworks', description: 'SOC2, GDPR, HIPAA, PCI-DSS, and more' },
  { value: '100K+', label: 'msg/sec Throughput', description: 'Kafka event streaming with AES-256-GCM' },
];

export const performanceStats: Stat[] = [
  { value: '2-4 hrs', label: 'Discovery Time', description: 'vs 6-12 months manual reverse engineering' },
  { value: '50K+', label: 'msg/sec Parsing', description: 'Generated parser throughput' },
  { value: '<1s', label: 'Decision Time', description: '900x faster than manual SOC response' },
  { value: '10K+', label: 'Containers', description: 'eBPF monitoring with <1% CPU overhead' },
  { value: '150ms', label: 'P95 Latency', description: 'Protocol discovery request latency' },
  { value: '6', label: 'SDK Languages', description: 'Python, TypeScript, Go, Java, Rust, C#' },
];
