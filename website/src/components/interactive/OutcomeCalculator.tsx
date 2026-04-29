import { useState } from 'react';

type ProgramMode = 'inventory' | 'modernization' | 'pqc';

const modeLabels: Record<ProgramMode, string> = {
  inventory: 'Protocol inventory',
  modernization: 'Modernization',
  pqc: 'PQC migration',
};

const modeMultipliers: Record<ProgramMode, number> = {
  inventory: 0.82,
  modernization: 0.88,
  pqc: 0.93,
};

export default function OutcomeCalculator() {
  const [protocolCount, setProtocolCount] = useState(8);
  const [costPerProtocolK, setCostPerProtocolK] = useState(850);
  const [monthsPerProtocol, setMonthsPerProtocol] = useState(7);
  const [mode, setMode] = useState<ProgramMode>('modernization');

  const legacyCostK = protocolCount * costPerProtocolK;
  const qbitelCostK = Math.round(legacyCostK * 0.14);
  const savingsK = Math.max(0, legacyCostK - qbitelCostK);

  const legacyMonths = protocolCount * monthsPerProtocol;
  const qbitelMonths = Number((Math.max(1, protocolCount * 0.35)).toFixed(1));
  const monthsRecovered = Math.max(0, Number((legacyMonths - qbitelMonths).toFixed(1)));

  const evidenceDays = Math.max(2, Math.round(protocolCount * 0.7));
  const riskReduction = Math.round(modeMultipliers[mode] * 100);

  const legacyWidth = 100;
  const qbitelWidth = Math.max(14, Math.round((qbitelCostK / legacyCostK) * 100));
  const timeWidth = Math.max(10, Math.round((qbitelMonths / legacyMonths) * 100));

  return (
    <div className="grid gap-8 lg:grid-cols-[0.9fr_1.1fr]">
      <section className="card p-6">
        <div className="text-xs font-semibold uppercase tracking-[0.32em] text-neon-cyan/80">
          Outcome Model
        </div>
        <h3 className="mt-3 text-2xl font-bold text-white">
          Compare the legacy program against a QBITEL-style motion.
        </h3>
        <p className="mt-3 text-sm leading-7 text-slate-300">
          Use realistic ranges from your program and see what changes when protocol discovery,
          generated integration assets, and PQC planning are part of one workflow.
        </p>

        <div className="mt-6 space-y-6">
          <label className="block">
            <div className="mb-2 flex items-center justify-between text-sm text-slate-200">
              <span>Protocols or interfaces in scope</span>
              <span className="font-semibold text-neon-cyan">{protocolCount}</span>
            </div>
            <input
              type="range"
              min="1"
              max="20"
              value={protocolCount}
              onChange={(event) => setProtocolCount(Number(event.target.value))}
              className="w-full accent-cyan-400"
            />
          </label>

          <label className="block">
            <div className="mb-2 flex items-center justify-between text-sm text-slate-200">
              <span>Current cost per protocol</span>
              <span className="font-semibold text-neon-cyan">${costPerProtocolK}K</span>
            </div>
            <input
              type="range"
              min="150"
              max="2000"
              step="50"
              value={costPerProtocolK}
              onChange={(event) => setCostPerProtocolK(Number(event.target.value))}
              className="w-full accent-cyan-400"
            />
          </label>

          <label className="block">
            <div className="mb-2 flex items-center justify-between text-sm text-slate-200">
              <span>Months per protocol today</span>
              <span className="font-semibold text-neon-cyan">{monthsPerProtocol} mo</span>
            </div>
            <input
              type="range"
              min="2"
              max="12"
              value={monthsPerProtocol}
              onChange={(event) => setMonthsPerProtocol(Number(event.target.value))}
              className="w-full accent-cyan-400"
            />
          </label>

          <div>
            <div className="mb-3 text-sm text-slate-200">Primary program objective</div>
            <div className="flex flex-wrap gap-2">
              {(['inventory', 'modernization', 'pqc'] as ProgramMode[]).map((value) => (
                <button
                  key={value}
                  type="button"
                  onClick={() => setMode(value)}
                  className={`rounded-full border px-3 py-2 text-sm transition-colors ${
                    mode === value
                      ? 'border-neon-cyan bg-neon-cyan/10 text-neon-cyan'
                      : 'border-surface-600/70 bg-surface-900/60 text-slate-300 hover:border-neon-cyan/40 hover:text-white'
                  }`}
                >
                  {modeLabels[value]}
                </button>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className="card overflow-hidden">
        <div className="border-b border-surface-600/60 p-6">
          <div className="text-xs font-semibold uppercase tracking-[0.32em] text-neon-copper/80">
            Estimated Program Shift
          </div>
          <div className="mt-4 grid gap-4 md:grid-cols-3">
            <MetricCard label="Cost avoided" value={`$${savingsK.toLocaleString()}K`} note="vs manual reverse engineering and one-off integration work" />
            <MetricCard label="Timeline recovered" value={`${monthsRecovered} mo`} note="across discovery, handoff, and validation cycles" />
            <MetricCard label="Control uplift" value={`${riskReduction}%`} note={`for a ${modeLabels[mode].toLowerCase()}-led program`} />
          </div>
        </div>

        <div className="grid gap-6 p-6">
          <ComparisonBar
            title="Program cost"
            legacyLabel={`Legacy: $${legacyCostK.toLocaleString()}K`}
            qbitelLabel={`QBITEL-led: $${qbitelCostK.toLocaleString()}K`}
            legacyWidth={legacyWidth}
            qbitelWidth={qbitelWidth}
          />

          <ComparisonBar
            title="Delivery time"
            legacyLabel={`Legacy: ${legacyMonths.toFixed(1)} months`}
            qbitelLabel={`QBITEL-led: ${qbitelMonths.toFixed(1)} months`}
            legacyWidth={legacyWidth}
            qbitelWidth={timeWidth}
          />

          <div className="rounded-3xl border border-surface-600/70 bg-surface-900/70 p-5">
            <div className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">
              What ships in the first wave
            </div>
            <div className="mt-4 grid gap-3 md:grid-cols-2">
              <OutcomeChip
                title="Day 1-2"
                text="Capture traffic, classify systems, and produce a first-pass protocol inventory."
              />
              <OutcomeChip
                title={`By day ${evidenceDays}`}
                text="Produce reviewed specs, evidence artifacts, and the first set of generated integration assets."
              />
              <OutcomeChip
                title="Pilot stage"
                text="Apply policy-driven rollout gates, replay validation, and a migration sequence for the first production path."
              />
              <OutcomeChip
                title="Board / audit view"
                text="Bundle crypto posture, deployment scope, and evidence chain into a single executive-facing package."
              />
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

function MetricCard({
  label,
  value,
  note,
}: {
  label: string;
  value: string;
  note: string;
}) {
  return (
    <div className="rounded-3xl border border-surface-600/70 bg-surface-900/70 p-5">
      <div className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">{label}</div>
      <div className="mt-3 text-3xl font-bold text-white">{value}</div>
      <p className="mt-2 text-xs leading-6 text-slate-400">{note}</p>
    </div>
  );
}

function ComparisonBar({
  title,
  legacyLabel,
  qbitelLabel,
  legacyWidth,
  qbitelWidth,
}: {
  title: string;
  legacyLabel: string;
  qbitelLabel: string;
  legacyWidth: number;
  qbitelWidth: number;
}) {
  return (
    <div className="rounded-3xl border border-surface-600/70 bg-surface-900/70 p-5">
      <div className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">{title}</div>
      <div className="mt-4 space-y-4">
        <div>
          <div className="mb-2 flex items-center justify-between text-sm text-slate-200">
            <span>{legacyLabel}</span>
            <span>Baseline</span>
          </div>
          <div className="h-3 rounded-full bg-surface-700">
            <div
              className="h-3 rounded-full bg-gradient-to-r from-neon-copper to-red-400"
              style={{ width: `${legacyWidth}%` }}
            />
          </div>
        </div>
        <div>
          <div className="mb-2 flex items-center justify-between text-sm text-slate-200">
            <span>{qbitelLabel}</span>
            <span>After consolidation</span>
          </div>
          <div className="h-3 rounded-full bg-surface-700">
            <div
              className="h-3 rounded-full bg-gradient-to-r from-neon-cyan to-neon-green"
              style={{ width: `${qbitelWidth}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function OutcomeChip({ title, text }: { title: string; text: string }) {
  return (
    <div className="rounded-2xl border border-surface-600/70 bg-surface-800/80 p-4">
      <div className="text-xs font-semibold uppercase tracking-[0.24em] text-neon-cyan/80">{title}</div>
      <p className="mt-2 text-sm leading-6 text-slate-200">{text}</p>
    </div>
  );
}
