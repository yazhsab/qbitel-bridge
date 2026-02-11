export interface TechCategory {
  category: string;
  items: { name: string; description: string }[];
}

export const techStack: TechCategory[] = [
  {
    category: 'Languages',
    items: [
      { name: 'Python 3.10+', description: 'AI Engine, compliance, discovery' },
      { name: 'Rust 1.70+', description: 'Data plane, PQC-TLS, wire-speed' },
      { name: 'Go 1.21+', description: 'Control plane, mgmt API, agents' },
      { name: 'TypeScript/React', description: 'Admin console, dashboard' },
    ],
  },
  {
    category: 'AI & ML',
    items: [
      { name: 'PyTorch', description: 'Deep learning models' },
      { name: 'Transformers', description: 'Protocol grammar learning' },
      { name: 'Ollama', description: 'On-premise LLM inference' },
      { name: 'ChromaDB', description: 'Vector embeddings & RAG' },
    ],
  },
  {
    category: 'Cryptography',
    items: [
      { name: 'ML-KEM (Kyber-1024)', description: 'Key encapsulation' },
      { name: 'ML-DSA (Dilithium-5)', description: 'Digital signatures' },
      { name: 'liboqs / oqs-rs', description: 'PQC library (Rust & Python)' },
      { name: 'AES-256-GCM', description: 'Symmetric encryption' },
    ],
  },
  {
    category: 'Infrastructure',
    items: [
      { name: 'Kubernetes', description: 'Container orchestration' },
      { name: 'Istio + Envoy', description: 'Quantum-safe service mesh' },
      { name: 'eBPF', description: 'Runtime monitoring' },
      { name: 'Kafka', description: 'Event streaming (100K+ msg/s)' },
    ],
  },
  {
    category: 'Observability',
    items: [
      { name: 'Prometheus', description: 'Metrics collection' },
      { name: 'Grafana', description: 'Dashboards & visualization' },
      { name: 'OpenTelemetry', description: 'Distributed tracing' },
      { name: 'Jaeger', description: 'Trace visualization' },
    ],
  },
];
