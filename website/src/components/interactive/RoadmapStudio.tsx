import { useState } from 'react';
import { roadmapPhases } from '@/data/strategy';

type PhaseStatus = 'all' | 'active' | 'next' | 'planned' | 'scale';

const statusLabel: Record<Exclude<PhaseStatus, 'all'>, string> = {
  active: 'Active',
  next: 'Next',
  planned: 'Planned',
  scale: 'Scale',
};

export default function RoadmapStudio() {
  const [status, setStatus] = useState<PhaseStatus>('all');

  const visiblePhases =
    status === 'all'
      ? roadmapPhases
      : roadmapPhases.filter((phase) => phase.status === status);

  return (
    <div className="card overflow-hidden">
      <div className="border-b border-surface-600/60 p-6">
        <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.32em] text-neon-cyan/80">
              Product Roadmap
            </p>
            <h3 className="mt-3 text-3xl font-bold text-white">
              A sharper 24-month sequence for QBITEL.
            </h3>
            <p className="mt-3 max-w-3xl text-sm leading-7 text-slate-300">
              The roadmap below is designed around one buyer journey: critical system discovery,
              behavior intelligence, modernization asset generation, governed protection, and
              evidence-led proof.
            </p>
          </div>

          <div className="flex flex-wrap gap-2">
            {(['all', 'active', 'next', 'planned', 'scale'] as PhaseStatus[]).map((value) => (
              <button
                key={value}
                type="button"
                onClick={() => setStatus(value)}
                className={`rounded-full border px-3 py-2 text-sm transition-colors ${
                  status === value
                    ? 'border-neon-cyan bg-neon-cyan/10 text-neon-cyan'
                    : 'border-surface-600/70 bg-surface-900/60 text-slate-300 hover:border-neon-cyan/40 hover:text-white'
                }`}
              >
                {value === 'all' ? 'All phases' : statusLabel[value]}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="grid gap-5 p-6">
        {visiblePhases.map((phase, index) => (
          <article
            key={phase.id}
            className="rounded-3xl border border-surface-600/70 bg-surface-900/70 p-6"
          >
            <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
              <div className="flex items-start gap-4">
                <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-2xl border border-surface-500/80 bg-surface-800 text-lg font-bold text-white">
                  {index + 1}
                </div>
                <div>
                  <div className="text-xs font-semibold uppercase tracking-[0.24em] text-neon-copper/80">
                    {phase.window}
                  </div>
                  <h4 className="mt-2 text-2xl font-semibold text-white">{phase.label}</h4>
                  <p className="mt-3 max-w-3xl text-sm leading-7 text-slate-300">
                    {phase.objective}
                  </p>
                </div>
              </div>

              <span className={`inline-flex rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.24em] ${badgeClass(phase.status)}`}>
                {statusLabel[phase.status]}
              </span>
            </div>

            <div className="mt-6 grid gap-3 md:grid-cols-2">
              {phase.deliverables.map((item) => (
                <div
                  key={item}
                  className="rounded-2xl border border-surface-600/70 bg-surface-800/80 p-4"
                >
                  <div className="flex items-start gap-3">
                    <span className="mt-1 h-2 w-2 rounded-full bg-neon-green" />
                    <p className="text-sm leading-6 text-slate-200">{item}</p>
                  </div>
                </div>
              ))}
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}

function badgeClass(status: Exclude<PhaseStatus, 'all'>) {
  switch (status) {
    case 'active':
      return 'border-neon-green/40 bg-neon-green/10 text-neon-green';
    case 'next':
      return 'border-neon-cyan/40 bg-neon-cyan/10 text-neon-cyan';
    case 'planned':
      return 'border-neon-copper/40 bg-neon-copper/10 text-neon-copper';
    case 'scale':
      return 'border-quantum-400/40 bg-quantum-500/10 text-quantum-300';
  }
}
