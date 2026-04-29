import { useMemo, useState } from 'react';
import {
  auditStatusCopy,
  maturitySummary,
  pqcAuditItems,
  type AuditDomain,
  type AuditStatus,
} from '@/data/pqcAudit';

type StatusFilter = 'all' | AuditStatus;
type DomainFilter = 'all' | AuditDomain;

const domainLabels: Record<AuditDomain, string> = {
  platform: 'Platform core',
  'stateful-signatures': 'State management',
  automotive: 'Automotive',
  industrial: 'Industrial',
  healthcare: 'Healthcare',
  'rust-dataplane': 'Rust dataplane',
};

export default function PqcAuditExplorer() {
  const [status, setStatus] = useState<StatusFilter>('all');
  const [domain, setDomain] = useState<DomainFilter>('all');

  const visibleItems = useMemo(() => {
    return pqcAuditItems.filter((item) => {
      const statusMatch = status === 'all' || item.status === status;
      const domainMatch = domain === 'all' || item.domain === domain;
      return statusMatch && domainMatch;
    });
  }, [domain, status]);

  return (
    <div className="card overflow-hidden">
      <div className="border-b border-surface-600/60 p-6">
        <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.32em] text-neon-cyan/80">
              PQC Implementation Review
            </p>
            <h3 className="mt-3 text-3xl font-bold text-white">
              Filter the current maturity of each custom and domain-specific module.
            </h3>
            <p className="mt-3 max-w-3xl text-sm leading-7 text-slate-300">
              This view reflects the code as reviewed in the repository: which parts are strong,
              which are guarded by strict configuration, and which still rely on simplified or
              placeholder cryptography.
            </p>
          </div>

          <div className="flex flex-wrap gap-2">
            {(['all', 'strong', 'guarded', 'prototype', 'unsafe'] as StatusFilter[]).map(
              (value) => (
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
                  {value === 'all' ? 'All status' : auditStatusCopy[value].label}
                </button>
              ),
            )}
          </div>
        </div>
      </div>

      <div className="grid gap-6 p-6">
        <div className="grid gap-4 md:grid-cols-3">
          {maturitySummary.map((item) => (
            <div
              key={item.label}
              className="rounded-3xl border border-surface-600/70 bg-surface-900/70 p-5"
            >
              <div className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">
                {item.label}
              </div>
              <div className="mt-3 text-3xl font-bold text-white">{item.value}</div>
              <p className="mt-2 text-sm leading-6 text-slate-300">{item.detail}</p>
            </div>
          ))}
        </div>

        <div className="flex flex-wrap gap-2">
          {(['all', ...Object.keys(domainLabels)] as DomainFilter[]).map((value) => (
            <button
              key={value}
              type="button"
              onClick={() => setDomain(value)}
              className={`rounded-full border px-3 py-2 text-sm transition-colors ${
                domain === value
                  ? 'border-neon-copper bg-neon-copper/10 text-neon-copper'
                  : 'border-surface-600/70 bg-surface-900/60 text-slate-300 hover:border-neon-copper/40 hover:text-white'
              }`}
            >
              {value === 'all' ? 'All domains' : domainLabels[value]}
            </button>
          ))}
        </div>

        <div className="grid gap-5">
          {visibleItems.map((item) => (
            <article
              key={item.id}
              className="rounded-3xl border border-surface-600/70 bg-surface-900/70 p-6"
            >
              <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                <div>
                  <div className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">
                    {domainLabels[item.domain]}
                  </div>
                  <h4 className="mt-2 text-2xl font-semibold text-white">{item.label}</h4>
                  <p className="mt-3 max-w-3xl text-sm leading-7 text-slate-300">
                    {item.summary}
                  </p>
                </div>

                <div className={`inline-flex rounded-full border px-3 py-1 text-xs font-semibold uppercase tracking-[0.24em] ${statusBadgeClass(item.status)}`}>
                  {auditStatusCopy[item.status].label}
                </div>
              </div>

              <div className="mt-5 rounded-2xl border border-surface-600/70 bg-surface-800/80 p-4">
                <div className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">
                  Code reference
                </div>
                <p className="mt-2 font-mono text-sm text-neon-cyan">
                  {item.file}:{item.lines}
                </p>
              </div>

              <div className="mt-5 grid gap-4 lg:grid-cols-2">
                <div>
                  <div className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">
                    What the code currently shows
                  </div>
                  <ul className="mt-3 space-y-3 text-sm text-slate-200">
                    {item.highlights.map((highlight) => (
                      <li key={highlight} className="flex items-start gap-3">
                        <span className="mt-1 h-2 w-2 rounded-full bg-neon-cyan" />
                        <span>{highlight}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div>
                  <div className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">
                    Recommended next step
                  </div>
                  <ul className="mt-3 space-y-3 text-sm text-slate-200">
                    {item.recommendations.map((recommendation) => (
                      <li key={recommendation} className="flex items-start gap-3">
                        <span className="mt-1 h-2 w-2 rounded-full bg-neon-green" />
                        <span>{recommendation}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </article>
          ))}
        </div>
      </div>
    </div>
  );
}

function statusBadgeClass(status: AuditStatus) {
  switch (status) {
    case 'strong':
      return 'border-neon-green/40 bg-neon-green/10 text-neon-green';
    case 'guarded':
      return 'border-neon-cyan/40 bg-neon-cyan/10 text-neon-cyan';
    case 'prototype':
      return 'border-neon-copper/40 bg-neon-copper/10 text-neon-copper';
    case 'unsafe':
      return 'border-red-400/40 bg-red-500/10 text-red-300';
  }
}
