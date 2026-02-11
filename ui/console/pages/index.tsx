import React from 'react';

export default function Home() {
  return (
    <main style={{padding: 24}}>
      <h1>QBITEL Admin Console</h1>
      <p>OIDC-ready Next.js shell. Wire Keycloak and RBAC here.</p>
      <ul>
        <li>Fleet → Devices</li>
        <li>Policies → OPA bundles</li>
        <li>Observability → Grafana links</li>
        <li>OTA → A/B bundles</li>
      </ul>
    </main>
  );
}
