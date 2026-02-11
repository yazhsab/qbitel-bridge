export interface TechCategory {
  category: string;
  items: { name: string; description: string }[];
}

export const techStack: TechCategory[] = [
  {
    category: 'Languages',
    items: [
      { name: 'Python 3.11+', description: 'AI Engine, discovery, compliance' },
      { name: 'Rust 1.75+', description: 'Data plane, PQC-TLS, wire-speed' },
      { name: 'Go 1.22+', description: 'Control plane, gRPC API, agents' },
      { name: 'TypeScript 5.x', description: 'Admin console, React dashboard' },
    ],
  },
  {
    category: 'AI & ML',
    items: [
      { name: 'PyTorch 2.x', description: 'CNN, BiLSTM, Transformer models' },
      { name: 'Transformers', description: 'Protocol grammar learning' },
      { name: 'Ollama', description: 'On-premise LLM inference' },
      { name: 'ChromaDB', description: 'Vector embeddings & RAG pipeline' },
    ],
  },
  {
    category: 'Cryptography',
    items: [
      { name: 'ML-KEM (Kyber-1024)', description: 'NIST FIPS 203 key encapsulation' },
      { name: 'ML-DSA (Dilithium-5)', description: 'NIST FIPS 204 digital signatures' },
      { name: 'liboqs / oqs-rs', description: 'Open Quantum Safe (Rust + Python)' },
      { name: 'AES-256-GCM', description: 'Authenticated symmetric encryption' },
    ],
  },
  {
    category: 'Infrastructure',
    items: [
      { name: 'Kubernetes', description: 'Container orchestration & Helm' },
      { name: 'Istio + Envoy', description: 'Quantum-safe service mesh mTLS' },
      { name: 'eBPF', description: 'Kernel-level runtime monitoring' },
      { name: 'Apache Kafka', description: 'Encrypted streaming (100K+ msg/s)' },
    ],
  },
  {
    category: 'Observability',
    items: [
      { name: 'Prometheus', description: 'Metrics collection & alerting' },
      { name: 'Grafana', description: 'Dashboards & visualization' },
      { name: 'OpenTelemetry', description: 'Distributed tracing & spans' },
      { name: 'Jaeger', description: 'Request trace visualization' },
    ],
  },
];
