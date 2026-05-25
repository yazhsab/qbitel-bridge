/* QBITEL Banking Security Dashboard - Vanilla JS */

// ─── Tab Switching ─────────────────────────────────────────────────────────────

document.querySelectorAll('.tab-btn').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(s => s.classList.remove('active'));
    tab.classList.add('active');
    const target = document.getElementById('tab-' + tab.dataset.tab);
    if (target) target.classList.add('active');
  });
});

// ─── Utility Functions ─────────────────────────────────────────────────────────

function showLoading(btn) { btn.classList.add('loading'); btn.disabled = true; }
function hideLoading(btn) { btn.classList.remove('loading'); btn.disabled = false; }

function formatHex(hex, len = 32) {
  if (!hex) return '';
  return hex.length <= len ? hex : hex.substring(0, len) + '...';
}

function formatMs(ms) { return (ms || 0).toFixed(1) + ' ms'; }

async function apiCall(url, body = {}) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({ error: response.statusText }));
    throw new Error(err.error || `HTTP ${response.status}`);
  }
  return response.json();
}

function riskColor(level) {
  return { critical: '#ff1744', high: '#ff9100', moderate: '#ffea00', low: '#00e676' }[level] || '#8b949e';
}

// ─── UC1: SWIFT Proxy Re-Encryption Chain ──────────────────────────────────────

const swiftRunBtn = document.getElementById('swift-run');
if (swiftRunBtn) swiftRunBtn.addEventListener('click', runSwiftChain);

async function runSwiftChain() {
  const btn = document.getElementById('swift-run');
  const resultsDiv = document.getElementById('swift-results');
  const metricsDiv = document.getElementById('swift-metrics');

  showLoading(btn);
  if (resultsDiv) resultsDiv.textContent = 'Running SWIFT chain...';

  // Reset flow diagram
  document.querySelectorAll('.bank-node').forEach(n => n.classList.remove('active'));
  document.querySelectorAll('.flow-arrow').forEach(a => a.classList.remove('active'));

  try {
    const data = await apiCall('/api/swift/full-chain', {});

    // Animate flow diagram
    const nodes = document.querySelectorAll('.bank-node');
    const arrows = document.querySelectorAll('.flow-arrow');

    if (nodes[0]) nodes[0].classList.add('active');
    setTimeout(() => { if (arrows[0]) arrows[0].classList.add('active'); }, 400);
    setTimeout(() => { if (nodes[1]) nodes[1].classList.add('active'); }, 800);
    setTimeout(() => { if (arrows[1]) arrows[1].classList.add('active'); }, 1200);
    setTimeout(() => { if (nodes[2]) nodes[2].classList.add('active'); }, 1600);

    // Render audit trail / chain info
    if (resultsDiv) {
      const output = {
        status: data.status,
        message_id: data.message_id,
        chain: data.chain,
        hops: data.hops,
        re_encryption_keys: data.re_encryption_keys,
        audit_entries: (data.audit || []).length,
      };
      resultsDiv.textContent = JSON.stringify(output, null, 2);
    }

    // Render metrics
    if (metricsDiv && data.metrics) {
      const m = data.metrics;
      const cards = metricsDiv.querySelectorAll('.metric-card .metric-value');
      if (cards[0]) cards[0].textContent = formatMs(m.total_latency_ms || 0);
      if (cards[1]) cards[1].textContent = formatMs(m.total_latency_ms ? m.total_latency_ms * 0.3 : 0);
      if (cards[2]) cards[2].textContent = formatMs(m.total_latency_ms ? m.total_latency_ms * 0.35 : 0);
      if (cards[3]) cards[3].textContent = formatMs(m.total_latency_ms ? m.total_latency_ms * 0.35 : 0);
    }
  } catch (error) {
    if (resultsDiv) resultsDiv.textContent = 'Error: ' + error.message;
  } finally {
    hideLoading(btn);
  }
}

// ─── UC2: Multi-Authority Threshold Signing ────────────────────────────────────

const thresholdState = { requestId: null, collected: 0, threshold: 3, signedAuthorities: new Set() };

const thresholdInitBtn = document.getElementById('threshold-init');
if (thresholdInitBtn) thresholdInitBtn.addEventListener('click', initiateThreshold);

document.querySelectorAll('.sign-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const authorityId = btn.dataset.authorityId;
    if (authorityId) signAsAuthority(authorityId, btn);
  });
});

async function initiateThreshold() {
  const btn = document.getElementById('threshold-init');
  const resultDiv = document.getElementById('threshold-result');
  const progressFill = document.querySelector('.progress-fill');
  const progressText = document.querySelector('.progress-text');

  showLoading(btn);
  thresholdState.requestId = null;
  thresholdState.collected = 0;
  thresholdState.signedAuthorities.clear();

  document.querySelectorAll('.authority-card').forEach(c => c.classList.remove('signed'));
  document.querySelectorAll('.sign-btn').forEach(b => { b.disabled = true; b.textContent = 'Sign'; });
  if (progressFill) progressFill.style.width = '0%';
  if (progressText) progressText.textContent = '0 of 3 signatures collected';
  if (resultDiv) resultDiv.textContent = 'Initiating signing request...';

  try {
    const data = await apiCall('/api/threshold/initiate', { amount: 50000000 });
    thresholdState.requestId = data.request_id;
    thresholdState.threshold = data.quorum ? data.quorum.threshold : 3;

    if (progressText) progressText.textContent = `0 of ${thresholdState.threshold} signatures collected`;
    document.querySelectorAll('.sign-btn').forEach(b => { b.disabled = false; });

    if (resultDiv) {
      resultDiv.textContent = JSON.stringify({
        request_id: data.request_id,
        tier: data.tier,
        quorum: data.quorum,
        status: 'pending - click authority cards to sign'
      }, null, 2);
    }
  } catch (error) {
    if (resultDiv) resultDiv.textContent = 'Error: ' + error.message;
  } finally {
    hideLoading(btn);
  }
}

async function signAsAuthority(authorityId, signBtn) {
  if (!thresholdState.requestId || thresholdState.signedAuthorities.has(authorityId)) return;

  const card = signBtn.closest('.authority-card');
  const progressFill = document.querySelector('.progress-fill');
  const progressText = document.querySelector('.progress-text');

  showLoading(signBtn);

  try {
    await apiCall('/api/threshold/sign', {
      request_id: thresholdState.requestId,
      authority_id: authorityId
    });

    thresholdState.signedAuthorities.add(authorityId);
    thresholdState.collected++;
    if (card) card.classList.add('signed');
    signBtn.textContent = 'Signed';

    const pct = (thresholdState.collected / thresholdState.threshold) * 100;
    if (progressFill) progressFill.style.width = pct + '%';
    if (progressText) progressText.textContent = `${thresholdState.collected} of ${thresholdState.threshold} signatures collected`;

    if (thresholdState.collected >= thresholdState.threshold) {
      await combineSignatures();
    }
  } catch (error) {
    signBtn.textContent = 'Failed';
  }
}

async function combineSignatures() {
  const resultDiv = document.getElementById('threshold-result');

  try {
    const data = await apiCall('/api/threshold/combine', { request_id: thresholdState.requestId });
    if (resultDiv) {
      const lines = [
        '=== Multi-Authority Signature Combined ===',
        '',
        'Combined Signature: ' + formatHex(data.combined_signature || '', 64),
        '',
        data.verified ? '\u2705 Verification: PASSED' : '\u274c Verification: FAILED',
        '',
        'Contributing Authorities:'
      ];
      (data.contributing_authorities || []).forEach(a => lines.push('  - ' + a));
      if (data.latency_ms) lines.push('', 'Latency: ' + formatMs(data.latency_ms));
      resultDiv.textContent = lines.join('\n');
    }

    // Disable remaining sign buttons
    document.querySelectorAll('.sign-btn').forEach(b => { b.disabled = true; });
  } catch (error) {
    if (resultDiv) resultDiv.textContent += '\nCombine error: ' + error.message;
  }
}

// ─── UC3: Zero-Knowledge Proofs ────────────────────────────────────────────────

['balance', 'capital', 'aml'].forEach(type => {
  const btn = document.getElementById('zkp-' + type + '-btn');
  const apiType = { balance: 'balance-range', capital: 'capital-adequacy', aml: 'aml-screening' }[type];
  if (btn) btn.addEventListener('click', () => runZKP(apiType, btn));
});

async function runZKP(type, btn) {
  const card = btn.closest('.zkp-card');
  const resultDiv = card ? card.querySelector('.zkp-result') : null;

  showLoading(btn);
  if (resultDiv) resultDiv.innerHTML = '<div class="spinner"></div>';

  try {
    const data = await apiCall('/api/zkp/' + type, {});

    if (resultDiv) {
      const verified = data.verification && data.verification.valid;
      resultDiv.innerHTML = [
        `<div><strong>Type:</strong> ${data.proof_type || type}</div>`,
        `<div><strong>Framework:</strong> ${data.framework || 'N/A'}</div>`,
        data.proof ? `<div><strong>Proof ID:</strong> ${formatHex(data.proof.proof_id || '', 24)}</div>` : '',
        `<div style="margin-top:0.5rem" class="${verified ? 'verified' : ''}">${verified ? '\u2705 Verified' : '\u274c Failed'}</div>`,
        data.demo_note ? `<div style="margin-top:0.5rem;color:var(--text-muted);font-size:0.8rem">${data.demo_note}</div>` : '',
        data.latency_ms != null ? `<div style="margin-top:0.5rem"><span class="metric-badge">Latency: ${formatMs(data.latency_ms)}</span></div>` : ''
      ].filter(Boolean).join('');
    }
  } catch (error) {
    if (resultDiv) resultDiv.innerHTML = '<div style="color:#ff1744">Error: ' + error.message + '</div>';
  } finally {
    hideLoading(btn);
  }
}

// ─── UC4: Quantum Threat Assessment ────────────────────────────────────────────

const threatScanBtn = document.getElementById('threat-scan');
if (threatScanBtn) threatScanBtn.addEventListener('click', runThreatScan);

async function runThreatScan() {
  const btn = document.getElementById('threat-scan');
  const summaryDiv = document.getElementById('portfolio-summary');
  const tableEl = document.getElementById('asset-table');

  showLoading(btn);

  try {
    const data = await apiCall('/api/threat/scan', {});
    const summary = data.portfolio_summary || {};

    // Render summary cards
    if (summaryDiv) {
      const levels = [
        { key: 'critical_count', label: 'Critical', cls: 'critical' },
        { key: 'high_count', label: 'High', cls: 'high' },
        { key: 'moderate_count', label: 'Moderate', cls: 'moderate' },
        { key: 'low_count', label: 'Low', cls: 'low' },
      ];
      summaryDiv.innerHTML = levels.map(l =>
        `<div class="summary-card ${l.cls}">
          <div class="count">${summary[l.key] || 0}</div>
          <div class="label">${l.label}</div>
        </div>`
      ).join('');
    }

    // Render asset table
    if (tableEl && data.assessments) {
      let tbody = tableEl.querySelector('tbody');
      if (!tbody) {
        tableEl.innerHTML = `<thead><tr>
          <th>Asset</th><th>Algorithm</th><th>Risk Score</th><th>Level</th><th>HNDL</th><th>Action</th>
        </tr></thead><tbody></tbody>`;
        tbody = tableEl.querySelector('tbody');
      }
      tbody.innerHTML = data.assessments.map(a => {
        const score = Math.round(a.quantum_risk_score || 0);
        const level = (a.risk_level || 'low').toLowerCase();
        const color = riskColor(level);
        return `<tr>
          <td>${a.asset_id || ''}</td>
          <td>${a.algorithm || ''}</td>
          <td><div class="risk-bar"><div class="risk-fill" style="width:${score}%;background:${color}"></div></div> ${score}</td>
          <td><span class="risk-badge ${level}">${level}</span></td>
          <td>${a.harvest_now_risk ? '<span style="color:#ff1744">Yes</span>' : '<span style="color:#00e676">No</span>'}</td>
          <td style="font-size:0.8rem">${a.recommended_action || ''}</td>
        </tr>`;
      }).join('');
    }
  } catch (error) {
    console.error('Threat scan error:', error);
  } finally {
    hideLoading(btn);
  }
}
