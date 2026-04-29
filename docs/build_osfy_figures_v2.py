"""Generate professional high-resolution figures for OSFY article using Playwright."""

from playwright.sync_api import sync_playwright
import os

OUT = "/Users/prabakarankannan/qbitel/docs"

# ═══════════════════════════════════════════════════════════════════
# Figure 1: Protocol discovery pipeline
# ═══════════════════════════════════════════════════════════════════
FIGURE1_HTML = """<!DOCTYPE html>
<html><head>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Inter', -apple-system, sans-serif;
    background: #FFFFFF;
    width: 1400px; height: 580px;
    padding: 36px 40px 30px;
  }
  .title { text-align: center; font-size: 21px; font-weight: 700; color: #1a1a2e; margin-bottom: 3px; }
  .subtitle { text-align: center; font-size: 12.5px; color: #6b7280; margin-bottom: 28px; }
  .pipeline { display: flex; align-items: center; justify-content: center; }
  .stage-box {
    width: 162px; height: 112px; border-radius: 14px;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    position: relative;
    box-shadow: 0 4px 14px rgba(0,0,0,0.07), 0 1px 3px rgba(0,0,0,0.05);
    border: 1.5px solid;
  }
  .stage-box .icon { font-size: 25px; margin-bottom: 5px; }
  .stage-box .label { font-size: 11.5px; font-weight: 600; text-align: center; line-height: 1.35; color: #1a1a2e; }
  .stage-box .sublabel { font-size: 9.5px; color: #6b7280; text-align: center; margin-top: 3px; }
  .s0 { background: linear-gradient(145deg, #f8f9fa, #e9ecef); border-color: #ced4da; }
  .s1 { background: linear-gradient(145deg, #eaf5fd, #d0e8f7); border-color: #7cb8e4; }
  .s2 { background: linear-gradient(145deg, #e6f0fa, #c5ddee); border-color: #5a9fcf; }
  .s3 { background: linear-gradient(145deg, #e4f5eb, #c8e6d5); border-color: #5baa7c; }
  .s4 { background: linear-gradient(145deg, #efebfd, #ddd4f7); border-color: #8b6fc0; }
  .s5 { background: linear-gradient(145deg, #fef5e6, #fde4bf); border-color: #e09422; }
  .arr { padding: 0 4px; display: flex; align-items: center; }
  .arr svg { width: 34px; height: 18px; }
  .out-badge {
    background: linear-gradient(135deg, #059669, #047857); color: white;
    padding: 11px 20px; border-radius: 28px; font-size: 12px; font-weight: 600;
    box-shadow: 0 4px 14px rgba(5,150,105,0.28); white-space: nowrap; margin-left: 6px;
  }
  .conf { position: absolute; top: -13px; right: 6px; background: #8b6fc0; color: white;
    font-size: 8.5px; font-weight: 600; padding: 2px 8px; border-radius: 10px; }
  .step-row { display: flex; justify-content: center; margin-bottom: 6px; padding-right: 80px; }
  .step-n { width: 162px; text-align: center; margin: 0 21px; }
  .step-n span { display: inline-flex; align-items: center; justify-content: center;
    width: 21px; height: 21px; border-radius: 50%;
    background: #e5e7eb; color: #4b5563; font-size: 10px; font-weight: 600; }
  .metrics {
    display: flex; justify-content: center; gap: 48px;
    margin-top: 22px; padding: 16px 28px;
    background: linear-gradient(145deg, #f9fafb, #f1f3f5);
    border-radius: 14px; border: 1px solid #e5e7eb;
  }
  .met .v { font-size: 19px; font-weight: 700; color: #1a1a2e; text-align: center; }
  .met .d { font-size: 9.5px; color: #6b7280; text-align: center; margin-top: 1px; }
</style></head>
<body>
  <div class="title">QBITEL Bridge protocol discovery pipeline</div>
  <div class="subtitle">From raw traffic to validated parsers — five-stage ML-driven protocol learning</div>
  <div class="step-row">
    <div class="step-n"><span>1</span></div><div class="step-n"><span>2</span></div>
    <div class="step-n"><span>3</span></div><div class="step-n"><span>4</span></div>
    <div class="step-n"><span>5</span></div>
  </div>
  <div class="pipeline">
    <div class="stage-box s0"><div class="icon">📡</div><div class="label">Raw Traffic<br>Capture</div><div class="sublabel">Packet streams</div></div>
    <div class="arr"><svg viewBox="0 0 34 18"><line x1="2" y1="9" x2="24" y2="9" stroke="#7cb8e4" stroke-width="2.2"/><polygon points="24,4.5 33,9 24,13.5" fill="#7cb8e4"/></svg></div>
    <div class="stage-box s1"><div class="icon">📊</div><div class="label">Statistical<br>Analysis</div><div class="sublabel">Entropy &amp; byte patterns</div></div>
    <div class="arr"><svg viewBox="0 0 34 18"><line x1="2" y1="9" x2="24" y2="9" stroke="#5a9fcf" stroke-width="2.2"/><polygon points="24,4.5 33,9 24,13.5" fill="#5a9fcf"/></svg></div>
    <div class="stage-box s2"><div class="icon">🧠</div><div class="label">PCFG Grammar<br>Learning</div><div class="sublabel">EM algorithm inference</div></div>
    <div class="arr"><svg viewBox="0 0 34 18"><line x1="2" y1="9" x2="24" y2="9" stroke="#5baa7c" stroke-width="2.2"/><polygon points="24,4.5 33,9 24,13.5" fill="#5baa7c"/></svg></div>
    <div class="stage-box s3"><div class="icon">⚙️</div><div class="label">Dynamic Parser<br>Generation</div><div class="sublabel">Runtime compilers</div></div>
    <div class="arr"><svg viewBox="0 0 34 18"><line x1="2" y1="9" x2="24" y2="9" stroke="#8b6fc0" stroke-width="2.2"/><polygon points="24,4.5 33,9 24,13.5" fill="#8b6fc0"/></svg></div>
    <div class="stage-box s4" style="position:relative"><div class="conf">threshold: 0.7</div><div class="icon">🤖</div><div class="label">Ensemble ML<br>Classification</div><div class="sublabel">CNN + LSTM + RF</div></div>
    <div class="arr"><svg viewBox="0 0 34 18"><line x1="2" y1="9" x2="24" y2="9" stroke="#e09422" stroke-width="2.2"/><polygon points="24,4.5 33,9 24,13.5" fill="#e09422"/></svg></div>
    <div class="stage-box s5"><div class="icon">🛡️</div><div class="label">Compliance<br>Validation</div><div class="sublabel">Anomaly detection</div></div>
    <div class="arr"><svg viewBox="0 0 34 18"><line x1="2" y1="9" x2="24" y2="9" stroke="#059669" stroke-width="2.5"/><polygon points="24,4.5 33,9 24,13.5" fill="#059669"/></svg></div>
    <div class="out-badge">✅ Protected Perimeter</div>
  </div>
  <div class="metrics">
    <div class="met"><div class="v">89%+</div><div class="d">First-pass accuracy</div></div>
    <div class="met"><div class="v">2–4 hrs</div><div class="d">Discovery time</div></div>
    <div class="met"><div class="v">&lt;10ms</div><div class="d">Classification latency</div></div>
    <div class="met"><div class="v">30+</div><div class="d">Protocol types tested</div></div>
    <div class="met"><div class="v">Zero</div><div class="d">Manual configuration</div></div>
  </div>
</body></html>"""


# ═══════════════════════════════════════════════════════════════════
# Figure 2: PQC algorithm selection by deployment domain
# ═══════════════════════════════════════════════════════════════════
FIGURE2_HTML = """<!DOCTYPE html>
<html><head>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Inter', -apple-system, sans-serif;
    background: #FFFFFF;
    width: 1400px; height: 750px;
    padding: 36px 40px 30px;
  }
  .title { text-align: center; font-size: 21px; font-weight: 700; color: #1a1a2e; margin-bottom: 3px; }
  .subtitle { text-align: center; font-size: 12.5px; color: #6b7280; margin-bottom: 30px; }

  .domains { display: flex; justify-content: center; gap: 14px; margin-bottom: 0; position: relative; z-index: 2; }
  .domain {
    width: 155px; padding: 14px 10px; border-radius: 12px; text-align: center;
    box-shadow: 0 3px 12px rgba(0,0,0,0.06); border: 1.5px solid; position: relative;
  }
  .domain .icon { font-size: 28px; margin-bottom: 6px; }
  .domain .name { font-size: 12px; font-weight: 700; color: #1a1a2e; }
  .domain .algo { font-size: 10px; font-weight: 500; color: #4b5563; margin-top: 4px; line-height: 1.4; }
  .domain .constraint { font-size: 9px; color: #9ca3af; margin-top: 3px; font-style: italic; }
  .d-health { background: linear-gradient(145deg, #ecfdf5, #d1fae5); border-color: #6ee7b7; }
  .d-auto { background: linear-gradient(145deg, #fef3e2, #fde4bf); border-color: #f59e0b; }
  .d-bank { background: linear-gradient(145deg, #eef2ff, #e0e7ff); border-color: #818cf8; }
  .d-avia { background: linear-gradient(145deg, #fdf2f8, #fce7f3); border-color: #f472b6; }
  .d-indus { background: linear-gradient(145deg, #fff7ed, #ffedd5); border-color: #fb923c; }
  .d-bpo { background: linear-gradient(145deg, #f0fdf4, #dcfce7); border-color: #4ade80; }

  .connector-area {
    display: flex; justify-content: center; align-items: center;
    height: 80px; position: relative;
  }
  .connector-area svg { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }

  .engine-box {
    width: 420px; margin: 0 auto; padding: 18px 28px;
    background: linear-gradient(145deg, #1e3a5f, #1a2d4a);
    border-radius: 16px; text-align: center; position: relative; z-index: 2;
    box-shadow: 0 8px 28px rgba(26,45,74,0.25);
  }
  .engine-box .eng-title { font-size: 16px; font-weight: 700; color: #ffffff; }
  .engine-box .eng-sub { font-size: 11px; color: #93c5fd; margin-top: 4px; }
  .engine-box .eng-note { font-size: 9.5px; color: #94a3b8; margin-top: 6px; font-style: italic; }

  .algo-row { display: flex; justify-content: center; gap: 18px; margin-top: 0; position: relative; z-index: 2; }
  .algo-box {
    width: 190px; padding: 16px 14px; border-radius: 12px; text-align: center;
    box-shadow: 0 3px 12px rgba(0,0,0,0.06); border: 1.5px solid;
  }
  .algo-box .a-name { font-size: 13px; font-weight: 700; color: #1a1a2e; }
  .algo-box .a-fips { font-size: 10px; font-weight: 600; color: #6b7280; margin-top: 2px; }
  .algo-box .a-desc { font-size: 9.5px; color: #9ca3af; margin-top: 4px; }
  .a-kem { background: linear-gradient(145deg, #eaf5fd, #d4edfc); border-color: #60a5fa; }
  .a-dsa { background: linear-gradient(145deg, #eef2ff, #e0e7ff); border-color: #818cf8; }
  .a-fal { background: linear-gradient(145deg, #ecfdf5, #d1fae5); border-color: #34d399; }
  .a-slh { background: linear-gradient(145deg, #fefce8, #fef9c3); border-color: #facc15; }
  .a-fal .a-fips { color: #059669; }

  .conn2-area { display: flex; justify-content: center; height: 55px; position: relative; }
  .conn2-area svg { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }
</style></head>
<body>
  <div class="title">PQC algorithm selection by deployment domain</div>
  <div class="subtitle">Domain-aware cryptographic engine — automatic algorithm selection based on constraints</div>

  <div class="domains">
    <div class="domain d-health"><div class="icon">🏥</div><div class="name">Healthcare</div><div class="algo">ML-KEM-512</div><div class="constraint">64 KB RAM devices</div></div>
    <div class="domain d-auto"><div class="icon">🚗</div><div class="name">Automotive</div><div class="algo">Falcon-512</div><div class="constraint">&lt;1ms real-time V2X</div></div>
    <div class="domain d-bank"><div class="icon">🏦</div><div class="name">Banking</div><div class="algo">ML-KEM-1024 +<br>ML-DSA-87</div><div class="constraint">Maximum security</div></div>
    <div class="domain d-avia"><div class="icon">✈️</div><div class="name">Aviation</div><div class="algo">Aggregate signatures</div><div class="constraint">ARINC 429/629</div></div>
    <div class="domain d-indus"><div class="icon">🏭</div><div class="name">Industrial</div><div class="algo">IEC 61850 auth</div><div class="constraint">Power grid GOOSE</div></div>
    <div class="domain d-bpo"><div class="icon">📞</div><div class="name">BPO / Call Centre</div><div class="algo">SIP/RTP PQC</div><div class="constraint">DTMF masking</div></div>
  </div>

  <div class="connector-area">
    <svg viewBox="0 0 1400 80" preserveAspectRatio="none">
      <defs><marker id="ah" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0,0 8,3 0,6" fill="#94a3b8"/></marker></defs>
      <line x1="178" y1="5" x2="560" y2="70" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#ah)"/>
      <line x1="348" y1="5" x2="620" y2="70" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#ah)"/>
      <line x1="518" y1="5" x2="680" y2="70" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#ah)"/>
      <line x1="688" y1="5" x2="720" y2="70" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#ah)"/>
      <line x1="858" y1="5" x2="780" y2="70" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#ah)"/>
      <line x1="1028" y1="5" x2="840" y2="70" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#ah)"/>
    </svg>
  </div>

  <div class="engine-box">
    <div class="eng-title">PQCEngine</div>
    <div class="eng-sub">Domain-aware unified cryptographic interface</div>
    <div class="eng-note">Hybrid mode: X25519 / P-384 ECDH + ML-KEM for TLS compatibility</div>
  </div>

  <div class="conn2-area">
    <svg viewBox="0 0 1400 55" preserveAspectRatio="none">
      <defs><marker id="ah2" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0,0 8,3 0,6" fill="#94a3b8"/></marker></defs>
      <line x1="560" y1="5" x2="310" y2="48" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#ah2)"/>
      <line x1="630" y1="5" x2="530" y2="48" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#ah2)"/>
      <line x1="750" y1="5" x2="740" y2="48" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#ah2)"/>
      <line x1="840" y1="5" x2="960" y2="48" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#ah2)"/>
    </svg>
  </div>

  <div class="algo-row">
    <div class="algo-box a-kem"><div class="a-name">ML-KEM (Kyber)</div><div class="a-fips">FIPS 203</div><div class="a-desc">Key encapsulation for session keys<br>512 / 768 / 1024 variants</div></div>
    <div class="algo-box a-dsa"><div class="a-name">ML-DSA (Dilithium)</div><div class="a-fips">FIPS 204</div><div class="a-desc">Digital signatures<br>44 / 65 / 87 security levels</div></div>
    <div class="algo-box a-fal"><div class="a-name">Falcon</div><div class="a-fips">NIST Alternate (not FIPS)</div><div class="a-desc">Compact signatures<br>Bandwidth-constrained channels</div></div>
    <div class="algo-box a-slh"><div class="a-name">SLH-DSA (SPHINCS+)</div><div class="a-fips">FIPS 205</div><div class="a-desc">Stateless hash-based signatures<br>High-assurance contexts</div></div>
  </div>
</body></html>"""


# ═══════════════════════════════════════════════════════════════════
# Figure 3: System architecture
# ═══════════════════════════════════════════════════════════════════
FIGURE3_HTML = """<!DOCTYPE html>
<html><head>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Inter', -apple-system, sans-serif;
    background: #FFFFFF;
    width: 1400px; height: 780px;
    padding: 36px 40px 30px;
  }
  .title { text-align: center; font-size: 21px; font-weight: 700; color: #1a1a2e; margin-bottom: 3px; }
  .subtitle { text-align: center; font-size: 12.5px; color: #6b7280; margin-bottom: 30px; }

  .arch { display: flex; gap: 28px; justify-content: center; align-items: stretch; }

  .panel {
    flex: 1; max-width: 520px; border-radius: 18px; padding: 22px 20px 18px;
    position: relative; border: 2px solid;
  }
  .panel-py { background: linear-gradient(170deg, #f8faff, #eef3fb); border-color: #3b82f6; }
  .panel-rs { background: linear-gradient(170deg, #f0fdf4, #ecfdf5); border-color: #22c55e; }

  .panel-header {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid rgba(0,0,0,0.08);
  }
  .panel-header .badge {
    padding: 4px 12px; border-radius: 20px; font-size: 10px; font-weight: 600; color: white;
  }
  .badge-py { background: #3b82f6; }
  .badge-rs { background: #22c55e; }
  .panel-header .p-title { font-size: 15px; font-weight: 700; color: #1a1a2e; }
  .panel-header .p-sub { font-size: 10.5px; color: #6b7280; margin-left: auto; font-style: italic; }

  .comp-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .comp {
    padding: 14px 12px; border-radius: 10px; text-align: center;
    border: 1px solid; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
  }
  .comp .c-icon { font-size: 22px; margin-bottom: 4px; }
  .comp .c-name { font-size: 11px; font-weight: 600; color: #1a1a2e; }
  .comp .c-desc { font-size: 9px; color: #6b7280; margin-top: 2px; }
  .c-py { background: rgba(59,130,246,0.06); border-color: rgba(59,130,246,0.2); }
  .c-rs { background: rgba(34,197,94,0.06); border-color: rgba(34,197,94,0.2); }

  .tech-stack {
    margin-top: 12px; padding: 8px 12px; border-radius: 8px;
    background: rgba(0,0,0,0.03); text-align: center;
    font-size: 10px; color: #6b7280;
  }
  .tech-stack strong { color: #4b5563; font-weight: 600; }

  .perf-badge {
    margin-top: 12px; padding: 10px 14px; border-radius: 10px;
    text-align: center; border: 1.5px solid #22c55e;
    background: rgba(34,197,94,0.08);
  }
  .perf-badge .perf-row { display: flex; justify-content: center; gap: 28px; }
  .perf-badge .perf-item .pv { font-size: 16px; font-weight: 700; color: #15803d; }
  .perf-badge .perf-item .pd { font-size: 9px; color: #6b7280; }

  /* gRPC connector */
  .grpc-connector {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    width: 80px; position: relative;
  }
  .grpc-label {
    background: linear-gradient(135deg, #f59e0b, #d97706); color: white;
    padding: 6px 14px; border-radius: 20px; font-size: 11px; font-weight: 700;
    box-shadow: 0 3px 10px rgba(245,158,11,0.3); margin-bottom: 6px; white-space: nowrap;
  }
  .grpc-arrows { display: flex; flex-direction: column; gap: 4px; align-items: center; }
  .grpc-arrows svg { width: 50px; height: 14px; }

  /* Bottom bar */
  .bottom-bar {
    display: flex; justify-content: center; gap: 20px; margin-top: 22px;
  }
  .bottom-box {
    padding: 14px 20px; border-radius: 12px; text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05); border: 1.5px solid;
    display: flex; align-items: center; gap: 10px;
  }
  .bb-net { background: linear-gradient(145deg, #fef3e2, #fde4bf); border-color: #f59e0b; flex: 1; max-width: 440px; }
  .bb-ops { background: linear-gradient(145deg, #f8f9fa, #e9ecef); border-color: #9ca3af; flex: 1; max-width: 440px; }
  .bottom-box .bb-icon { font-size: 24px; }
  .bottom-box .bb-text { text-align: left; }
  .bottom-box .bb-title { font-size: 12px; font-weight: 600; color: #1a1a2e; }
  .bottom-box .bb-desc { font-size: 9.5px; color: #6b7280; margin-top: 1px; }

  .conn-arrows { display: flex; justify-content: center; gap: 280px; margin-top: 8px; margin-bottom: 4px; }
  .conn-arrows svg { width: 20px; height: 30px; }
</style></head>
<body>
  <div class="title">QBITEL Bridge system architecture</div>
  <div class="subtitle">Dual-runtime design — Python AI engine (cold path) and Rust dataplane (hot path)</div>

  <div class="arch">
    <!-- Python AI Engine -->
    <div class="panel panel-py">
      <div class="panel-header">
        <span class="badge badge-py">PYTHON</span>
        <span class="p-title">AI Engine</span>
        <span class="p-sub">Cold path — learning &amp; reasoning</span>
      </div>
      <div class="comp-grid">
        <div class="comp c-py"><div class="c-icon">🔍</div><div class="c-name">ML Protocol Discovery</div><div class="c-desc">PCFG inference &amp; classification</div></div>
        <div class="comp c-py"><div class="c-icon">🤖</div><div class="c-name">Agent Framework</div><div class="c-desc">Multi-agent orchestration</div></div>
        <div class="comp c-py"><div class="c-icon">💬</div><div class="c-name">LLM Integration</div><div class="c-desc">Ollama / cloud models</div></div>
        <div class="comp c-py"><div class="c-icon">🔐</div><div class="c-name">PQC Crypto Layer</div><div class="c-desc">FIPS 203/204/205</div></div>
        <div class="comp c-py"><div class="c-icon">🌐</div><div class="c-name">REST API</div><div class="c-desc">FastAPI endpoints</div></div>
        <div class="comp c-py"><div class="c-icon">📈</div><div class="c-name">Observability</div><div class="c-desc">Prometheus metrics</div></div>
      </div>
      <div class="tech-stack"><strong>Stack:</strong> FastAPI · PyTorch · LangGraph · Redis</div>
    </div>

    <!-- gRPC connector -->
    <div class="grpc-connector">
      <div class="grpc-label">gRPC</div>
      <div class="grpc-arrows">
        <svg viewBox="0 0 50 14"><line x1="0" y1="7" x2="38" y2="7" stroke="#f59e0b" stroke-width="2.5"/><polygon points="38,3 48,7 38,11" fill="#f59e0b"/></svg>
        <svg viewBox="0 0 50 14"><polygon points="12,3 2,7 12,11" fill="#f59e0b"/><line x1="12" y1="7" x2="50" y2="7" stroke="#f59e0b" stroke-width="2.5"/></svg>
      </div>
      <div style="font-size:8px;color:#92400e;text-align:center;margin-top:4px;font-weight:500;">Typed<br>interface</div>
    </div>

    <!-- Rust Dataplane -->
    <div class="panel panel-rs">
      <div class="panel-header">
        <span class="badge badge-rs">RUST</span>
        <span class="p-title">Dataplane</span>
        <span class="p-sub">Hot path — forwarding &amp; encryption</span>
      </div>
      <div class="comp-grid">
        <div class="comp c-rs"><div class="c-icon">📦</div><div class="c-name">Packet Processing</div><div class="c-desc">High-throughput forwarding</div></div>
        <div class="comp c-rs"><div class="c-icon">🔒</div><div class="c-name">PQC-TLS Termination</div><div class="c-desc">Quantum-safe handshakes</div></div>
        <div class="comp c-rs"><div class="c-icon">🔄</div><div class="c-name">Protocol Proxying</div><div class="c-desc">Transparent interception</div></div>
        <div class="comp c-rs"><div class="c-icon">📋</div><div class="c-name">Session Management</div><div class="c-desc">Connection tracking</div></div>
      </div>
      <div class="perf-badge">
        <div class="perf-row">
          <div class="perf-item"><div class="pv">10M+</div><div class="pd">packets/sec</div></div>
          <div class="perf-item"><div class="pv">&lt;10ms</div><div class="pd">latency (p95)</div></div>
          <div class="perf-item"><div class="pv">&lt;2ms</div><div class="pd">PQC overhead</div></div>
        </div>
      </div>
      <div class="tech-stack"><strong>Stack:</strong> Tokio async runtime · GPU acceleration</div>
    </div>
  </div>

  <div class="conn-arrows">
    <svg viewBox="0 0 20 30"><line x1="10" y1="0" x2="10" y2="22" stroke="#9ca3af" stroke-width="2"/><polygon points="5,22 10,30 15,22" fill="#9ca3af"/></svg>
    <svg viewBox="0 0 20 30"><polygon points="5,8 10,0 15,8" fill="#f59e0b"/><line x1="10" y1="8" x2="10" y2="30" stroke="#f59e0b" stroke-width="2"/></svg>
  </div>

  <div class="bottom-bar">
    <div class="bottom-box bb-ops">
      <div class="bb-icon">👁️</div>
      <div class="bb-text"><div class="bb-title">Operators · Grafana · Alerting</div><div class="bb-desc">Dashboards, alerts (CPU &gt;90%, memory &gt;85%), incident reports</div></div>
    </div>
    <div class="bottom-box bb-net">
      <div class="bb-icon">🌍</div>
      <div class="bb-text"><div class="bb-title">Network Traffic</div><div class="bb-desc">Legacy + modern protocols — transparent bump-in-the-wire</div></div>
    </div>
  </div>
</body></html>"""


def render_figures():
    with sync_playwright() as p:
        browser = p.chromium.launch()

        figures = [
            ("figure1_discovery_pipeline.png", FIGURE1_HTML, 1400, 580),
            ("figure2_pqc_algorithm_selection.png", FIGURE2_HTML, 1400, 750),
            ("figure3_system_architecture.png", FIGURE3_HTML, 1400, 780),
        ]

        for filename, html, w, h in figures:
            page = browser.new_page(viewport={"width": w, "height": h}, device_scale_factor=2)
            page.set_content(html, wait_until="networkidle")
            page.wait_for_timeout(1500)  # wait for font loading
            path = os.path.join(OUT, filename)
            page.screenshot(path=path, full_page=False)
            page.close()
            size_kb = os.path.getsize(path) // 1024
            print(f"Created {filename} ({size_kb} KB, {w*2}x{h*2} px)")

        browser.close()


if __name__ == "__main__":
    render_figures()
