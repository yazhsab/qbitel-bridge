import { useEffect, useMemo, useState } from 'react';

type UseCaseId =
  | 'bpo'
  | 'iot'
  | 'defense'
  | 'banking'
  | 'industrial'
  | 'healthcare'
  | 'telecom'
  | 'mobility';

type UseCase = {
  id: UseCaseId;
  label: string;
  pack: string;
  buyer: string;
  plainProblem: string;
  sampleLabel: string;
  sample: string;
  protocols: string[];
  risks: string[];
  outputs: string[];
  outcome: string;
  accent: string;
  glow: string;
};

const steps = [
  {
    label: 'Discover',
    title: 'Map the real systems',
    detail: 'QBITEL identifies protocols, assets, dependencies, owners, and message structure from samples, logs, or captures.',
  },
  {
    label: 'Understand',
    title: 'Explain what matters',
    detail: 'The platform translates technical signals into fraud, privacy, uptime, compliance, and crypto-risk language.',
  },
  {
    label: 'Modernize',
    title: 'Create the bridge',
    detail: 'Reviewed field maps, replay fixtures, gateway plans, and adapter candidates make change safer.',
  },
  {
    label: 'Protect',
    title: 'Apply governed control',
    detail: 'Policy, segmentation, fraud rules, crypto-agility, and fail-closed controls are applied around the existing system.',
  },
  {
    label: 'Prove',
    title: 'Package the evidence',
    detail: 'Executives, auditors, clients, and engineering teams receive a usable evidence pack with lineage and approvals.',
  },
];

const useCases: UseCase[] = [
  {
    id: 'bpo',
    label: 'BPO / Call Center',
    pack: 'Voice Shield',
    buyer: 'CX leaders, CISOs, client assurance teams',
    plainProblem:
      'Call centers carry fraud, PCI, PII, IVR, recording, and client-audit risk across many voice systems.',
    sampleLabel: 'SIP + IVR + recording metadata',
    sample:
      'INVITE sip:+442079460958@carrier.example SIP/2.0\nX-IVR-Step: collect-card\nX-DTMF-Observed: 16 digits\nRTP-Recording: enabled\nDisposition: failed-auth retry=4 premium-route=true',
    protocols: ['SIP', 'RTP/SRTP', 'WebRTC', 'IVR', 'PCI voice'],
    risks: ['Toll fraud', 'Payment leakage', 'Voice phishing', 'Client evidence gaps'],
    outputs: ['Call-flow map', 'Fraud indicators', 'PCI exposure summary', 'Client evidence pack'],
    outcome: 'Turn voice security into measurable client trust.',
    accent: '#22d3ee',
    glow: 'shadow-cyan-500/20',
  },
  {
    id: 'iot',
    label: 'IoT',
    pack: 'Device Trust',
    buyer: 'IoT platform, product security, operations',
    plainProblem:
      'Device fleets grow faster than governance, leaving weak firmware, identity, telemetry, and crypto paths hidden.',
    sampleLabel: 'MQTT + gateway telemetry',
    sample:
      'mqtt topic=factory/line7/pump42/status\npayload={"device_id":"pump42","firmware":"1.0.3","auth":"default password","crypto":"rsa-1024","transport":"unencrypted"}',
    protocols: ['MQTT', 'CoAP', 'BLE', 'Zigbee', 'LoRaWAN'],
    risks: ['Weak crypto', 'Firmware drift', 'Unknown devices', 'Unsafe telemetry'],
    outputs: ['Device trust graph', 'Firmware exposure', 'Protocol inventory', 'Gateway control plan'],
    outcome: 'Convert unmanaged device sprawl into governed device trust.',
    accent: '#34d399',
    glow: 'shadow-emerald-500/20',
  },
  {
    id: 'defense',
    label: 'Defense',
    pack: 'Mission Bridge',
    buyer: 'Mission modernization and high-assurance teams',
    plainProblem:
      'Mission systems need modernization without breaking air-gap, approval, classification, or fail-closed boundaries.',
    sampleLabel: 'Offline mission message',
    sample:
      'MSG-TYPE=TRACK-UPDATE\nCLASSIFICATION=SECRET\nLINK=LEGACY-IP\nAUTH=unsigned\nKEY-EXCHANGE=ECC-P256\nROUTE=mission-gateway-alpha',
    protocols: ['Tactical links', 'Serial', 'Legacy IP', 'MIL-STD', 'Offline captures'],
    risks: ['Unknown dependencies', 'Air-gap transfer risk', 'Crypto transition', 'Approval gaps'],
    outputs: ['Mission flow map', 'Approval trail', 'Crypto posture', 'Signed evidence pack'],
    outcome: 'Build explainable control around mission infrastructure.',
    accent: '#93c5fd',
    glow: 'shadow-blue-500/20',
  },
  {
    id: 'banking',
    label: 'Banking',
    pack: 'Mainframe Shield',
    buyer: 'CIOs, CISOs, payments, mainframe teams',
    plainProblem:
      'Payment and mainframe flows cannot be safely changed until business meaning and hidden dependencies are known.',
    sampleLabel: 'Payment + mainframe trace',
    sample:
      'MTI=0200 PAN=411111******1111 PROC=000000 AMT=000000250000\nKEY_EXCHANGE=RSA-1024 MAC=3DES\nSWIFT MT103 FIELD20=REF7781',
    protocols: ['ISO 8583', 'SWIFT', 'TN3270e', 'ACH', 'COBOL'],
    risks: ['Payment opacity', 'PCI/DORA gaps', 'Crypto agility pressure', 'Fragile integrations'],
    outputs: ['Payment graph', 'Adapter package', 'PCI/DORA evidence', 'Crypto migration plan'],
    outcome: 'Modernize financial infrastructure without breaking trust.',
    accent: '#60a5fa',
    glow: 'shadow-indigo-500/20',
  },
  {
    id: 'industrial',
    label: 'OT / Critical Infra',
    pack: 'Industrial Shield',
    buyer: 'OT security, plant operations, utilities',
    plainProblem:
      'Safety-critical systems need visibility first and disruption last across SCADA, PLC, substation, and historian flows.',
    sampleLabel: 'Modbus / PLC event',
    sample:
      'modbus tcp src=10.4.2.18 dst=10.4.2.44 unit=7 function=write_single_register register=40001 value=1\nauth=none transport=cleartext',
    protocols: ['Modbus', 'DNP3', 'IEC 61850', 'OPC UA', 'Historian'],
    risks: ['Unsafe automation', 'Protocol blind spots', 'Uptime pressure', 'NERC CIP evidence'],
    outputs: ['Zone map', 'Critical flow list', 'Safety gate plan', 'NERC CIP pack'],
    outcome: 'Protect critical infrastructure without blind automation.',
    accent: '#f59e0b',
    glow: 'shadow-amber-500/20',
  },
  {
    id: 'healthcare',
    label: 'Healthcare',
    pack: 'MedTech Bridge',
    buyer: 'Healthcare CISOs, biomedical engineering, compliance',
    plainProblem:
      'Patient-care workflows depend on devices and integrations that must be secured without interrupting care.',
    sampleLabel: 'HL7 + device telemetry',
    sample:
      'MSH|^~\\&|MONITOR|ICU|EHR|HOSP|202604271330||ORU^R01|A771|P|2.5\nPID|1||MRN77821||DOE^JANE\ntransport=legacy tls patient_phi=true',
    protocols: ['HL7', 'FHIR', 'DICOM', 'X12', 'Medical telemetry'],
    risks: ['PHI exposure', 'Legacy device risk', 'Integration fragility', 'Audit pressure'],
    outputs: ['Device inventory', 'PHI flow map', 'Integration spec', 'HIPAA evidence'],
    outcome: 'Secure clinical integration without forcing replacement.',
    accent: '#2dd4bf',
    glow: 'shadow-teal-500/20',
  },
  {
    id: 'telecom',
    label: 'Telecom',
    pack: 'Signaling Shield',
    buyer: 'Telecom security, network engineering, fraud teams',
    plainProblem:
      'Operators need protocol intelligence across signaling, mediation, voice, routing, and subscriber-risk surfaces.',
    sampleLabel: 'Diameter + SIP fraud event',
    sample:
      'diameter cmd=Credit-Control app=4 imsi=404991234567890 roaming=true result=limited\nsip fraud-score=high route=international trunk=gw-7',
    protocols: ['SIP', 'SS7', 'Diameter', 'GTP', 'Mediation'],
    risks: ['Signaling abuse', 'Voice fraud', 'Subscriber exposure', 'Mediation blind spots'],
    outputs: ['Signaling map', 'Fraud pattern report', 'Protocol inventory', 'Operator evidence'],
    outcome: 'Make network behavior explainable and provable.',
    accent: '#a78bfa',
    glow: 'shadow-violet-500/20',
  },
  {
    id: 'mobility',
    label: 'Aviation / Automotive',
    pack: 'Mobility Shield',
    buyer: 'Mobility security, platform engineering, certification',
    plainProblem:
      'Connected mobility systems need long-lifecycle trust while preserving latency, safety, and certification boundaries.',
    sampleLabel: 'CAN + V2X sample',
    sample:
      'CAN id=0x18FF50E5 data=02 FF 01 7A 00 00 11 09 safety=true\nV2X BSM id=veh-77 speed=83 heading=270 signature=unsigned replay_window=old',
    protocols: ['V2X', 'CAN', 'ARINC', 'ADS-B', 'ACARS'],
    risks: ['Spoofing', 'Replay risk', 'Lifecycle crypto', 'Certification evidence'],
    outputs: ['Protocol map', 'Latency profile', 'Crypto lifecycle plan', 'Certification evidence'],
    outcome: 'Build trust for connected mobility without disrupting safety.',
    accent: '#fb7185',
    glow: 'shadow-rose-500/20',
  },
];

export default function IndustryUseCaseAnimator() {
  const [activeId, setActiveId] = useState<UseCaseId>('bpo');
  const [activeStep, setActiveStep] = useState(0);
  const [inputByCase, setInputByCase] = useState<Record<UseCaseId, string>>(
    () =>
      Object.fromEntries(useCases.map((item) => [item.id, item.sample])) as Record<
        UseCaseId,
        string
      >,
  );

  const active = useCases.find((item) => item.id === activeId) ?? useCases[0];
  const input = inputByCase[active.id];
  const analysis = useMemo(() => scoreInput(active, input), [active, input]);
  const step = steps[activeStep];

  useEffect(() => {
    const timer = window.setInterval(() => {
      setActiveStep((current) => (current + 1) % steps.length);
    }, 3200);
    return () => window.clearInterval(timer);
  }, []);

  const setInput = (value: string) => {
    setInputByCase((current) => ({ ...current, [active.id]: value }));
  };

  return (
    <section
      data-testid="industry-usecase-animator"
      className="relative overflow-hidden rounded-[1.75rem] border border-surface-600/70 bg-surface-950/70 shadow-2xl shadow-quantum-950/40"
    >
      <div className="pointer-events-none absolute inset-0 bg-grid-pattern bg-grid opacity-10" />
      <div
        className="pointer-events-none absolute right-[-12rem] top-[-12rem] h-[30rem] w-[30rem] rounded-full blur-3xl"
        style={{ backgroundColor: `${active.accent}22` }}
      />

      <div className="relative border-b border-surface-600/60 p-5 sm:p-7">
        <div className="grid gap-6 xl:grid-cols-[1fr_0.72fr] xl:items-end">
          <div>
            <div className="inline-flex rounded-full border border-neon-cyan/30 bg-neon-cyan/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.22em] text-neon-cyan">
              Animated industry walkthrough
            </div>
            <h2 className="mt-4 max-w-4xl text-3xl font-bold leading-tight text-white sm:text-4xl">
              See how QBITEL explains each industry use case in business language.
            </h2>
            <p className="mt-4 max-w-3xl text-sm leading-7 text-slate-300 sm:text-base">
              Each story moves from sample data to discovered systems, risk, modernization,
              protection, and the evidence a buyer can actually use.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3 rounded-3xl border border-surface-600/70 bg-surface-900/70 p-4 sm:grid-cols-4">
            <Metric label="Signals" value={analysis.signals} color={active.accent} />
            <Metric label="Risk" value={`${analysis.risk}%`} color={analysis.risk > 70 ? '#fb7185' : active.accent} />
            <Metric label="Evidence" value={`${analysis.evidence}%`} color="#28f0a4" />
            <Metric label="Steps" value="5" color="#4fd8ff" />
          </div>
        </div>
      </div>

      <div className="relative grid gap-0 xl:grid-cols-[18rem_1fr]">
        <div className="border-b border-surface-600/60 bg-surface-950/50 p-4 xl:border-b-0 xl:border-r">
          <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-1">
            {useCases.map((item) => {
              const selected = item.id === active.id;
              return (
                <button
                  key={item.id}
                  data-testid={`industry-story-${item.id}`}
                  type="button"
                  onClick={() => {
                    setActiveId(item.id);
                    setActiveStep(0);
                  }}
                  className={`rounded-2xl border p-4 text-left transition duration-300 ${
                    selected
                      ? 'bg-surface-800 text-white shadow-lg'
                      : 'border-surface-600/60 bg-surface-900/60 text-slate-300 hover:border-neon-cyan/40 hover:bg-surface-800/70'
                  }`}
                  style={{ borderColor: selected ? `${item.accent}aa` : undefined }}
                >
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-sm font-semibold">{item.label}</span>
                    <span
                      className="h-2.5 w-2.5 rounded-full"
                      style={{ backgroundColor: selected ? item.accent : '#64748b' }}
                    />
                  </div>
                  <div className="mt-2 text-xs uppercase tracking-[0.2em]" style={{ color: item.accent }}>
                    {item.pack}
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        <div className="grid gap-0 lg:grid-cols-[0.92fr_1.08fr]">
          <div className="border-b border-surface-600/60 p-5 sm:p-6 lg:border-b-0 lg:border-r">
            <div className="flex flex-col gap-5">
              <div>
                <div
                  className="inline-flex rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-[0.22em] text-surface-950"
                  style={{ backgroundColor: active.accent }}
                >
                  {active.pack}
                </div>
                <h3 className="mt-4 text-2xl font-semibold text-white">{active.label}</h3>
                <p className="mt-3 text-sm leading-7 text-slate-300">{active.plainProblem}</p>
              </div>

              <div className="rounded-3xl border border-surface-600/70 bg-surface-900/80 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <div className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">
                      Visitor input
                    </div>
                    <div className="mt-1 text-sm font-semibold text-white">{active.sampleLabel}</div>
                  </div>
                  <button
                    type="button"
                    onClick={() => setInput(active.sample)}
                    className="rounded-lg border border-surface-500/70 px-3 py-2 text-xs font-semibold text-slate-200 transition hover:border-neon-cyan/60 hover:text-neon-cyan"
                  >
                    Load sample
                  </button>
                </div>
                <textarea
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  spellCheck={false}
                  className="mt-4 min-h-[13rem] w-full resize-y rounded-2xl border border-surface-600/70 bg-surface-950/85 p-4 font-mono text-xs leading-6 text-slate-100 outline-none transition focus:border-neon-cyan/70 sm:text-sm"
                />
              </div>

              <div className="rounded-3xl border border-surface-600/70 bg-surface-900/80 p-5">
                <div className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">
                  Buyer
                </div>
                <p className="mt-2 text-sm leading-6 text-slate-200">{active.buyer}</p>
              </div>
            </div>
          </div>

          <div className="p-5 sm:p-6">
            <div className="grid gap-5">
              <div className="rounded-3xl border border-surface-600/70 bg-surface-900/80 p-5">
                <div className="mb-5 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                  <div>
                    <div className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">
                      Animated explanation
                    </div>
                    <h4 className="mt-2 text-2xl font-semibold text-white">{step.title}</h4>
                    <p className="mt-2 text-sm leading-6 text-slate-300">{step.detail}</p>
                  </div>
                  <div
                    className={`flex h-16 w-16 shrink-0 items-center justify-center rounded-2xl border bg-surface-950 text-xl font-bold ${active.glow}`}
                    style={{ borderColor: `${active.accent}66`, color: active.accent }}
                  >
                    {activeStep + 1}
                  </div>
                </div>

                <div className="relative min-h-[18rem] overflow-hidden rounded-3xl border border-surface-600/60 bg-surface-950/80 p-5">
                  <div className="absolute inset-0 bg-grid-pattern bg-grid opacity-10" />
                  <div className="relative grid gap-5 lg:grid-cols-[0.95fr_1.05fr] lg:items-center">
                    <ProtocolColumn active={active} />
                    <FlowStage active={active} activeStep={activeStep} />
                  </div>
                </div>
              </div>

              <div className="grid gap-5 lg:grid-cols-2">
                <div className="rounded-3xl border border-surface-600/70 bg-surface-900/80 p-5">
                  <div className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">
                    Risk signals explained
                  </div>
                  <div className="mt-4 grid gap-3">
                    {active.risks.map((risk, index) => (
                      <div
                        key={risk}
                        className="flex items-center gap-3 rounded-2xl border border-surface-600/60 bg-surface-950/70 p-3 text-sm text-slate-200"
                      >
                        <span
                          className="h-2.5 w-2.5 rounded-full"
                          style={{ backgroundColor: index === 0 ? '#fb7185' : active.accent }}
                        />
                        <span>{risk}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="rounded-3xl border border-surface-600/70 bg-surface-900/80 p-5">
                  <div className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">
                    Output buyer receives
                  </div>
                  <div className="mt-4 grid gap-3">
                    {active.outputs.map((output) => (
                      <div
                        key={output}
                        className="rounded-2xl border border-surface-600/60 bg-surface-950/70 p-3 text-sm font-medium text-slate-100"
                      >
                        {output}
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div
                className="rounded-3xl border bg-surface-900/80 p-5"
                style={{ borderColor: `${active.accent}66` }}
              >
                <div className="text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">
                  Simple takeaway
                </div>
                <p className="mt-3 text-xl font-semibold leading-8 text-white">{active.outcome}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function ProtocolColumn({ active }: { active: UseCase }) {
  return (
    <div className="grid gap-3">
      {active.protocols.map((protocol, index) => (
        <div
          key={protocol}
          className="relative overflow-hidden rounded-2xl border border-surface-600/70 bg-surface-900/85 p-3"
        >
          <div
            className="absolute inset-y-0 left-0 w-1"
            style={{ backgroundColor: active.accent, opacity: 0.55 + index * 0.07 }}
          />
          <div className="pl-3 text-sm font-semibold text-white">{protocol}</div>
          <div
            className="mt-2 h-1 overflow-hidden rounded-full bg-surface-700"
            style={{ animationDelay: `${index * 140}ms` }}
          >
            <div
              className="h-full rounded-full usecase-stream"
              style={{ backgroundColor: active.accent, animationDelay: `${index * 220}ms` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function FlowStage({ active, activeStep }: { active: UseCase; activeStep: number }) {
  return (
    <div className="grid gap-3">
      {steps.map((item, index) => {
        const selected = index === activeStep;
        const complete = index < activeStep;
        return (
          <button
            key={item.label}
            type="button"
            className={`grid grid-cols-[2.75rem_1fr] items-center gap-3 rounded-2xl border p-3 text-left transition duration-300 ${
              selected
                ? 'bg-surface-800 shadow-lg'
                : complete
                  ? 'bg-surface-900/80'
                  : 'bg-surface-950/70'
            }`}
            style={{ borderColor: selected || complete ? `${active.accent}77` : 'rgba(71, 85, 105, 0.65)' }}
          >
            <span
              className="flex h-11 w-11 items-center justify-center rounded-xl text-sm font-bold text-surface-950"
              style={{ backgroundColor: selected || complete ? active.accent : '#64748b' }}
            >
              {index + 1}
            </span>
            <span>
              <span className="block text-sm font-semibold text-white">{item.label}</span>
              <span className="mt-1 block text-xs leading-5 text-slate-400">{item.title}</span>
            </span>
          </button>
        );
      })}
    </div>
  );
}

function Metric({ label, value, color }: { label: string; value: string | number; color: string }) {
  return (
    <div className="rounded-2xl border border-surface-600/60 bg-surface-950/60 p-3">
      <div className="text-[0.68rem] font-semibold uppercase tracking-[0.18em] text-slate-400">
        {label}
      </div>
      <div className="mt-2 text-2xl font-bold text-white" style={{ color }}>
        {value}
      </div>
    </div>
  );
}

function scoreInput(active: UseCase, input: string) {
  const text = input.toLowerCase();
  const matchedProtocols = active.protocols.filter((item) =>
    text.includes(item.toLowerCase().split(/[ /]+/)[0]),
  );
  const matchedRisks = active.risks.filter((item) =>
    item
      .toLowerCase()
      .split(/\s+/)
      .some((word) => word.length > 4 && text.includes(word)),
  );

  return {
    signals: Math.max(3, matchedProtocols.length + matchedRisks.length + Math.floor(input.length / 110)),
    risk: Math.min(96, 34 + matchedRisks.length * 14 + (input.length > 80 ? 12 : 0)),
    evidence: Math.min(95, 44 + matchedProtocols.length * 9 + Math.floor(input.length / 130) * 6),
  };
}
