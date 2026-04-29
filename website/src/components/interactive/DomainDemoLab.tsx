import { useMemo, useState } from 'react';

type DemoId =
  | 'bpo'
  | 'iot'
  | 'defense'
  | 'banking'
  | 'industrial'
  | 'healthcare'
  | 'telecom'
  | 'mobility';

type Finding = {
  label: string;
  detail: string;
  severity: 'low' | 'medium' | 'high';
};

type DemoConfig = {
  id: DemoId;
  label: string;
  solution: string;
  inputLabel: string;
  placeholder: string;
  protocols: string[];
  keywords: string[];
  riskTerms: string[];
  explainers: string[];
  evidence: string[];
  sample: string;
  accent: string;
};

const demos: DemoConfig[] = [
  {
    id: 'bpo',
    label: 'BPO / Call Center',
    solution: 'Voice Shield',
    inputLabel: 'Paste SIP/RTP, IVR, call log, or payment-call metadata',
    placeholder: 'INVITE sip:+14085550198@carrier.example SIP/2.0...',
    protocols: ['SIP', 'RTP/SRTP', 'WebRTC', 'IVR', 'PCI voice capture'],
    keywords: ['INVITE', 'SIP/2.0', 'RTP', 'call-id', 'ani', 'dnis', 'dtmf', 'recording'],
    riskTerms: ['premium', 'international', 'dtmf', 'card', 'cvv', 'recording', 'failed auth'],
    explainers: [
      'Maps voice call paths across SIP trunks, IVR, agents, recording, and CRM systems.',
      'Flags toll fraud, payment leakage, voice phishing, and client-audit exposure.',
      'Produces PCI and client-assurance evidence for the protected voice path.',
    ],
    evidence: ['Call-flow map', 'Fraud indicators', 'PCI exposure summary', 'Client evidence pack'],
    sample:
      'INVITE sip:+442079460958@carrier.example SIP/2.0\nCall-ID: bpo-7719\nFrom: <sip:+14085550100@contact-center>\nTo: <sip:+442079460958@carrier.example>\nX-IVR-Step: collect-card\nX-DTMF-Observed: 16 digits\nRTP-Recording: enabled\nDisposition: failed-auth retry=4 premium-route=true',
    accent: 'from-teal-500 to-cyan-400',
  },
  {
    id: 'iot',
    label: 'IoT',
    solution: 'Device Trust',
    inputLabel: 'Paste MQTT, CoAP, BLE, gateway, or device telemetry',
    placeholder: 'topic=factory/line7/temp payload={...}',
    protocols: ['MQTT', 'CoAP', 'BLE', 'Zigbee', 'LoRaWAN', 'Proprietary binary'],
    keywords: ['mqtt', 'topic', 'coap', 'ble', 'zigbee', 'lorawan', 'firmware', 'device_id'],
    riskTerms: ['default password', 'telnet', 'rsa-1024', 'expired', 'unknown firmware', 'unencrypted'],
    explainers: [
      'Builds a device trust graph from observed traffic, firmware hints, and gateway logs.',
      'Scores crypto posture and weak update paths before device replacement is possible.',
      'Recommends gateway-first containment for constrained or unmanaged devices.',
    ],
    evidence: ['Device inventory', 'Crypto posture', 'Firmware risk', 'Gateway control plan'],
    sample:
      'mqtt topic=factory/line7/pump42/status\npayload={"device_id":"pump42","firmware":"1.0.3","auth":"default password","crypto":"rsa-1024","transport":"unencrypted","temp":86}\ncoap /sensor/vibration device_id=pump42 uptime=91023',
    accent: 'from-lime-500 to-emerald-400',
  },
  {
    id: 'defense',
    label: 'Defense',
    solution: 'Mission Bridge',
    inputLabel: 'Paste mission protocol, syslog, serial, or tactical message samples',
    placeholder: 'MSG-TYPE=TRACK-UPDATE CLASS=SECRET...',
    protocols: ['Tactical data links', 'Serial links', 'Legacy IP', 'MIL-STD messages'],
    keywords: ['mission', 'tactical', 'track', 'classification', 'serial', 'mil-std', 'link'],
    riskTerms: ['rsa', 'ecc', 'cleartext', 'unsigned', 'unknown', 'classified', 'legacy key'],
    explainers: [
      'Runs as an offline discovery and assurance workflow for mission systems.',
      'Maps protocol fields, trust boundaries, approval gates, and CNSA 2.0 migration needs.',
      'Creates signed evidence for ATO and security-review packages.',
    ],
    evidence: ['Mission flow map', 'CNSA 2.0 readiness', 'Approval gates', 'Signed evidence pack'],
    sample:
      'MSG-TYPE=TRACK-UPDATE\nCLASSIFICATION=SECRET\nLINK=LEGACY-IP\nTRACK-ID=A17\nCOORD=12.973,77.594\nAUTH=unsigned\nKEY-EXCHANGE=ECC-P256\nROUTE=mission-gateway-alpha',
    accent: 'from-slate-500 to-zinc-300',
  },
  {
    id: 'banking',
    label: 'Banking',
    solution: 'Mainframe Shield',
    inputLabel: 'Paste ISO 8583, SWIFT, TN3270e, COBOL, or payment samples',
    placeholder: 'MTI=0200 PAN=411111******1111...',
    protocols: ['ISO 8583', 'SWIFT MT/MX', 'TN3270e', 'COBOL/JCL', 'FIX'],
    keywords: ['mti', 'iso8583', 'swift', 'mt103', 'pan', 'tn3270', 'cobol', 'jcl'],
    riskTerms: ['pan', 'pin', 'rsa-1024', 'des', 'clear', 'settlement', 'wire'],
    explainers: [
      'Discovers payment message structure and mainframe dependency paths.',
      'Generates adapter and API candidates without forcing core-system replacement.',
      'Maps crypto posture and evidence for PCI, DORA, and board risk reviews.',
    ],
    evidence: ['Payment flow graph', 'Protocol asset record', 'PQC migration plan', 'PCI/DORA evidence'],
    sample:
      'MTI=0200 PAN=411111******1111 PROC=000000 AMT=000000250000 TERM=ATM019\nKEY_EXCHANGE=RSA-1024 MAC=3DES\nSWIFT MT103 FIELD20=REF7781 FIELD32A=260427USD2500,00',
    accent: 'from-blue-500 to-indigo-400',
  },
  {
    id: 'industrial',
    label: 'OT / Critical Infra',
    solution: 'Industrial Shield',
    inputLabel: 'Paste Modbus, DNP3, IEC 61850, OPC UA, or historian samples',
    placeholder: 'modbus unit=7 function=write register=40001...',
    protocols: ['Modbus', 'DNP3', 'IEC 61850', 'OPC UA', 'Historian logs'],
    keywords: ['modbus', 'dnp3', 'iec 61850', 'opc', 'register', 'coil', 'scada'],
    riskTerms: ['write', 'trip', 'override', 'unauthenticated', 'cleartext', 'safety'],
    explainers: [
      'Builds passive OT asset and zone/conduit visibility without interrupting operations.',
      'Separates observation from active protection so safety constraints remain explicit.',
      'Creates NERC CIP and operations evidence for each protected flow.',
    ],
    evidence: ['Zone/conduit map', 'Critical flow list', 'Safety gate plan', 'NERC CIP evidence'],
    sample:
      'modbus tcp src=10.4.2.18 dst=10.4.2.44 unit=7 function=write_single_register register=40001 value=1\nsafety_override=false auth=none transport=cleartext',
    accent: 'from-amber-500 to-orange-400',
  },
  {
    id: 'healthcare',
    label: 'Healthcare',
    solution: 'MedTech Bridge',
    inputLabel: 'Paste HL7, FHIR, DICOM, device, or EHR integration samples',
    placeholder: 'MSH|^~\\&|MONITOR|ICU...',
    protocols: ['HL7 v2', 'FHIR', 'DICOM', 'X12', 'Medical device telemetry'],
    keywords: ['MSH|', 'PID|', 'OBR|', 'FHIR', 'DICOM', 'patient', 'device'],
    riskTerms: ['patient', 'dob', 'mrn', 'unencrypted', 'legacy tls', 'phi'],
    explainers: [
      'Discovers medical device and EHR interoperability paths without firmware rewrites.',
      'Identifies PHI exposure and integration points that need gateway protection.',
      'Produces HIPAA and FDA cybersecurity evidence tied to observed flows.',
    ],
    evidence: ['Device-flow inventory', 'PHI exposure map', 'Integration bridge plan', 'HIPAA evidence'],
    sample:
      'MSH|^~\\&|MONITOR|ICU|EHR|HOSP|202604271330||ORU^R01|A771|P|2.5\nPID|1||MRN77821||DOE^JANE||19791209\nOBX|1|NM|HR||118|bpm\ntransport=legacy tls patient_phi=true',
    accent: 'from-emerald-500 to-green-400',
  },
  {
    id: 'telecom',
    label: 'Telecom',
    solution: 'Signaling Shield',
    inputLabel: 'Paste SIP, SS7, Diameter, GTP, SMPP, or mediation samples',
    placeholder: 'diameter cmd=Credit-Control app=4...',
    protocols: ['SIP', 'SS7/SIGTRAN', 'Diameter', 'GTP', 'SMPP', 'RADIUS'],
    keywords: ['diameter', 'gtp', 'sip', 'ss7', 'sigtran', 'sccp', 'smpp', 'imsi'],
    riskTerms: ['roaming', 'imsi', 'location', 'spoof', 'cleartext', 'fraud'],
    explainers: [
      'Maps signaling and mediation flows at carrier scale.',
      'Finds fraud, spoofing, and sensitive subscriber-data exposure.',
      'Prioritizes hybrid protection where latency and interoperability allow it.',
    ],
    evidence: ['Signaling graph', 'Subscriber risk map', 'Mediation adapter plan', 'Operator evidence'],
    sample:
      'diameter cmd=Credit-Control app=4 imsi=404991234567890 roaming=true result=limited\nsip fraud-score=high route=international trunk=gw-7\ntransport=cleartext',
    accent: 'from-violet-500 to-purple-400',
  },
  {
    id: 'mobility',
    label: 'Aviation / Automotive',
    solution: 'Mobility Shield',
    inputLabel: 'Paste ADS-B, ACARS, CAN, V2X, or gateway samples',
    placeholder: 'CAN id=0x18FF50E5 data=...',
    protocols: ['ADS-B', 'ACARS', 'ARINC 429', 'CAN', 'V2X', 'IEEE 1609.2'],
    keywords: ['ads-b', 'acars', 'arinc', 'can', 'v2x', 'bsm', 'vin'],
    riskTerms: ['spoof', 'unsigned', 'replay', 'cleartext', 'safety', 'firmware'],
    explainers: [
      'Validates long-life mobility protocols while preserving safety and certification boundaries.',
      'Builds gateway protection and crypto-agility plans for constrained links.',
      'Produces evidence for security review, certification support, and rollout control.',
    ],
    evidence: ['Protocol validation', 'Safety boundary map', 'Crypto agility plan', 'Certification evidence'],
    sample:
      'CAN id=0x18FF50E5 data=02 FF 01 7A 00 00 11 09 safety=true\nV2X BSM id=veh-77 speed=83 heading=270 signature=unsigned replay_window=old',
    accent: 'from-sky-500 to-cyan-400',
  },
];

export default function DomainDemoLab() {
  const [activeId, setActiveId] = useState<DemoId>('bpo');
  const activeDemo = demos.find((demo) => demo.id === activeId) ?? demos[0];
  const [inputByDemo, setInputByDemo] = useState<Record<DemoId, string>>(() =>
    Object.fromEntries(demos.map((demo) => [demo.id, demo.sample])) as Record<DemoId, string>,
  );

  const input = inputByDemo[activeDemo.id];
  const analysis = useMemo(() => analyzeInput(activeDemo, input), [activeDemo, input]);

  const setInput = (value: string) => {
    setInputByDemo((current) => ({ ...current, [activeDemo.id]: value }));
  };

  return (
    <div className="relative overflow-hidden rounded-[1.75rem] border border-surface-600/70 bg-surface-900/80 shadow-2xl shadow-quantum-950/40">
      <div className="pointer-events-none absolute inset-0 opacity-40">
        <div className="demo-scan-line" />
      </div>
      <div className="relative border-b border-surface-600/60 bg-surface-800/60 p-5 sm:p-6">
        <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.32em] text-neon-cyan/80">
              Interactive Demo Lab
            </p>
            <h3 className="mt-3 text-3xl font-bold text-white">
              Test the vertical story with your own sample data.
            </h3>
            <p className="mt-3 max-w-3xl text-sm leading-7 text-slate-300">
              Pick a domain, paste representative traffic or logs, and see an explainable
              QBITEL-style output: discovered protocols, risk signals, modernization path,
              protection controls, and evidence artifacts.
            </p>
          </div>

          <div className="flex min-w-[11rem] items-center gap-3 rounded-2xl border border-neon-green/30 bg-neon-green/10 px-4 py-3">
            <span className="relative flex h-3 w-3">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-neon-green opacity-60" />
              <span className="relative inline-flex h-3 w-3 rounded-full bg-neon-green" />
            </span>
            <span className="text-sm font-semibold text-neon-green">Local browser analysis</span>
          </div>
        </div>
      </div>

      <div className="relative grid gap-0 lg:grid-cols-[18rem_1fr]">
        <aside className="border-b border-surface-600/60 bg-surface-950/40 p-4 lg:border-b-0 lg:border-r">
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-1">
            {demos.map((demo) => {
              const active = demo.id === activeDemo.id;
              return (
                <button
                  key={demo.id}
                  type="button"
                  onClick={() => setActiveId(demo.id)}
                  className={`group rounded-2xl border p-4 text-left transition-all duration-300 ${
                    active
                      ? 'border-neon-cyan/70 bg-neon-cyan/10 shadow-lg shadow-neon-cyan/10'
                      : 'border-surface-600/60 bg-surface-900/60 hover:border-neon-cyan/40 hover:bg-surface-800/80'
                  }`}
                >
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-sm font-semibold text-white">{demo.label}</span>
                    <span
                      className={`h-2.5 w-2.5 rounded-full transition-colors ${
                        active ? 'bg-neon-cyan' : 'bg-surface-500 group-hover:bg-neon-cyan/60'
                      }`}
                    />
                  </div>
                  <div className="mt-2 text-xs uppercase tracking-[0.2em] text-neon-copper/80">
                    {demo.solution}
                  </div>
                </button>
              );
            })}
          </div>
        </aside>

        <main className="grid gap-0 xl:grid-cols-[0.95fr_1.05fr]">
          <section className="border-b border-surface-600/60 p-5 sm:p-6 xl:border-b-0 xl:border-r">
            <div className="flex flex-col gap-4">
              <div>
                <div className={`inline-flex rounded-full bg-gradient-to-r px-3 py-1 text-xs font-semibold uppercase tracking-[0.22em] text-white ${activeDemo.accent}`}>
                  {activeDemo.solution}
                </div>
                <h4 className="mt-4 text-2xl font-semibold text-white">{activeDemo.label}</h4>
                <p className="mt-2 text-sm leading-6 text-slate-300">{activeDemo.inputLabel}</p>
              </div>

              <div className="grid gap-2">
                <label htmlFor="demo-input" className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">
                  Input data
                </label>
                <textarea
                  id="demo-input"
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  placeholder={activeDemo.placeholder}
                  spellCheck={false}
                  className="min-h-[18rem] resize-y rounded-2xl border border-surface-600/70 bg-surface-950/80 p-4 font-mono text-sm leading-6 text-slate-100 outline-none transition-colors placeholder:text-slate-600 focus:border-neon-cyan/70"
                />
              </div>

              <div className="flex flex-wrap gap-2">
                <button type="button" onClick={() => setInput(activeDemo.sample)} className="btn-secondary px-4 py-2 text-sm">
                  Load sample
                </button>
                <button type="button" onClick={() => setInput('')} className="btn-ghost px-4 py-2 text-sm">
                  Clear
                </button>
              </div>

              <div className="rounded-3xl border border-surface-600/70 bg-surface-800/70 p-4">
                <div className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">
                  What this demo proves
                </div>
                <ul className="mt-3 space-y-3 text-sm text-slate-200">
                  {activeDemo.explainers.map((item) => (
                    <li key={item} className="flex items-start gap-3">
                      <span className="mt-1 h-2 w-2 rounded-full bg-neon-green" />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </section>

          <section className="p-5 sm:p-6">
            <div className="grid gap-5">
              <div className="grid gap-4 md:grid-cols-3">
                <SignalCard label="Protocol confidence" value={`${analysis.confidence}%`} tone="cyan" />
                <SignalCard label="Risk score" value={`${analysis.risk}%`} tone={analysis.risk > 70 ? 'red' : analysis.risk > 42 ? 'copper' : 'green'} />
                <SignalCard label="Evidence readiness" value={`${analysis.evidenceReadiness}%`} tone="green" />
              </div>

              <div className="rounded-3xl border border-surface-600/70 bg-surface-800/70 p-5">
                <div className="mb-5 flex items-center justify-between gap-4">
                  <div>
                    <div className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">
                      QBITEL workflow
                    </div>
                    <h5 className="mt-2 text-xl font-semibold text-white">Explainable output path</h5>
                  </div>
                  <div className="hidden h-10 w-10 animate-pulse-slow rounded-full border border-neon-cyan/50 bg-neon-cyan/10 md:block" />
                </div>

                <div className="grid gap-3">
                  {analysis.steps.map((step, index) => (
                    <div key={step.label} className="relative overflow-hidden rounded-2xl border border-surface-600/60 bg-surface-900/70 p-4">
                      <div
                        className={`absolute left-0 top-0 h-full bg-gradient-to-r ${activeDemo.accent} opacity-10 transition-all duration-700`}
                        style={{ width: `${Math.min(100, step.score)}%` }}
                      />
                      <div className="relative flex items-start gap-4">
                        <div className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-xl border border-neon-cyan/30 bg-neon-cyan/10 text-sm font-bold text-neon-cyan">
                          {index + 1}
                        </div>
                        <div>
                          <h6 className="text-sm font-semibold text-white">{step.label}</h6>
                          <p className="mt-1 text-sm leading-6 text-slate-300">{step.detail}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="grid gap-5 lg:grid-cols-2">
                <div className="rounded-3xl border border-surface-600/70 bg-surface-800/70 p-5">
                  <div className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">
                    Detected protocols and artifacts
                  </div>
                  <div className="mt-4 flex flex-wrap gap-2">
                    {analysis.detectedProtocols.map((item) => (
                      <span key={item} className="rounded-full border border-neon-cyan/20 bg-neon-cyan/10 px-3 py-1 text-xs text-neon-cyan">
                        {item}
                      </span>
                    ))}
                  </div>
                </div>

                <div className="rounded-3xl border border-surface-600/70 bg-surface-800/70 p-5">
                  <div className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">
                    Evidence pack preview
                  </div>
                  <div className="mt-4 grid gap-2">
                    {activeDemo.evidence.map((item) => (
                      <div key={item} className="rounded-xl border border-surface-600/60 bg-surface-900/70 px-3 py-2 text-sm text-slate-200">
                        {item}
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="rounded-3xl border border-surface-600/70 bg-surface-800/70 p-5">
                <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
                  <div>
                    <div className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">
                      Findings
                    </div>
                    <h5 className="mt-2 text-xl font-semibold text-white">{analysis.findings.length} explainable signals found</h5>
                  </div>
                  <div className="rounded-full border border-surface-500/70 bg-surface-900/70 px-3 py-1 text-xs uppercase tracking-[0.2em] text-slate-300">
                    {analysis.inputBytes} chars analyzed
                  </div>
                </div>

                <div className="mt-5 grid gap-3">
                  {analysis.findings.map((finding) => (
                    <div key={`${finding.label}-${finding.detail}`} className="rounded-2xl border border-surface-600/70 bg-surface-900/70 p-4">
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <h6 className="text-sm font-semibold text-white">{finding.label}</h6>
                        <span className={`rounded-full border px-2.5 py-1 text-[0.68rem] uppercase tracking-[0.18em] ${severityClass(finding.severity)}`}>
                          {finding.severity}
                        </span>
                      </div>
                      <p className="mt-2 text-sm leading-6 text-slate-300">{finding.detail}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}

function analyzeInput(demo: DemoConfig, input: string) {
  const normalized = input.toLowerCase();
  const inputBytes = input.length;
  const matchedKeywords = demo.keywords.filter((keyword) => normalized.includes(keyword.toLowerCase()));
  const matchedRisks = demo.riskTerms.filter((term) => normalized.includes(term.toLowerCase()));

  const detectedProtocols = demo.protocols.filter((protocol) => {
    const protocolKey = protocol.toLowerCase().split(/[ /()]+/)[0];
    return normalized.includes(protocolKey) || matchedKeywords.length > 0;
  });

  const confidence = clamp(42 + matchedKeywords.length * 9 + Math.min(18, Math.floor(inputBytes / 80)), 35, 98);
  const risk = clamp(24 + matchedRisks.length * 13 + (inputBytes > 0 ? 8 : 0), 10, 96);
  const evidenceReadiness = clamp(38 + detectedProtocols.length * 8 + matchedKeywords.length * 5, 30, 94);

  const findings: Finding[] = [];
  if (matchedKeywords.length) {
    findings.push({
      label: 'Protocol signals detected',
      detail: `Matched ${matchedKeywords.slice(0, 5).join(', ')}. QBITEL would use these to seed the protocol asset record and confidence score.`,
      severity: 'low',
    });
  } else {
    findings.push({
      label: 'Low protocol signal',
      detail: 'The sample is sparse. Add headers, message fields, device IDs, call IDs, or transaction fields to improve discovery confidence.',
      severity: 'medium',
    });
  }

  if (matchedRisks.length) {
    findings.push({
      label: 'Risk indicators detected',
      detail: `Found ${matchedRisks.slice(0, 5).join(', ')}. These terms drive risk scoring, protection recommendations, and evidence-pack priority.`,
      severity: risk > 70 ? 'high' : 'medium',
    });
  }

  if (inputBytes > 240) {
    findings.push({
      label: 'Enough sample structure for replay',
      detail: 'The input has enough structure to generate a replay fixture, validation checklist, and first-pass field map.',
      severity: 'low',
    });
  }

  if (findings.length < 3) {
    findings.push({
      label: 'Recommended next input',
      detail: `Add two or three more ${demo.label} samples with normal and abnormal cases so QBITEL can compare behavior and improve field confidence.`,
      severity: 'low',
    });
  }

  return {
    confidence,
    risk,
    evidenceReadiness,
    inputBytes,
    detectedProtocols: detectedProtocols.length ? detectedProtocols : demo.protocols.slice(0, 3),
    findings,
    steps: [
      {
        label: 'Discover',
        detail: `Identifies ${demo.label.toLowerCase()} protocols, systems, flows, and ownership hints from the submitted sample.`,
        score: confidence,
      },
      {
        label: 'Understand',
        detail: `Explains behavior and risk using ${matchedRisks.length || 'baseline'} risk signals and domain-specific context.`,
        score: risk,
      },
      {
        label: 'Modernize',
        detail: 'Prepares a protocol asset record, field map, replay fixture, and adapter/API candidate for review.',
        score: Math.max(48, confidence - 8),
      },
      {
        label: 'Protect',
        detail: 'Recommends policy controls, gateway protection, fraud/threat rules, and PQC posture where operationally appropriate.',
        score: Math.max(42, risk),
      },
      {
        label: 'Prove',
        detail: `Packages ${demo.evidence.slice(0, 2).join(' and ').toLowerCase()} with lineage, approvals, and evidence.`,
        score: evidenceReadiness,
      },
    ],
  };
}

function SignalCard({ label, value, tone }: { label: string; value: string; tone: 'cyan' | 'green' | 'copper' | 'red' }) {
  const toneClass = {
    cyan: 'text-neon-cyan border-neon-cyan/30 bg-neon-cyan/10',
    green: 'text-neon-green border-neon-green/30 bg-neon-green/10',
    copper: 'text-neon-copper border-neon-copper/30 bg-neon-copper/10',
    red: 'text-red-300 border-red-400/30 bg-red-500/10',
  }[tone];

  return (
    <div className={`rounded-3xl border p-5 ${toneClass}`}>
      <div className="text-xs font-semibold uppercase tracking-[0.22em] opacity-80">{label}</div>
      <div className="mt-3 text-3xl font-bold text-white">{value}</div>
    </div>
  );
}

function severityClass(severity: Finding['severity']) {
  switch (severity) {
    case 'low':
      return 'border-neon-green/30 bg-neon-green/10 text-neon-green';
    case 'medium':
      return 'border-neon-copper/30 bg-neon-copper/10 text-neon-copper';
    case 'high':
      return 'border-red-400/30 bg-red-500/10 text-red-300';
  }
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}
