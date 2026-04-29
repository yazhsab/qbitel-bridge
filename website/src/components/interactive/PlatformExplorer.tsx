import { useState } from 'react';
import { productModules, verticalStories } from '@/data/strategy';

export default function PlatformExplorer() {
  const [activeModuleId, setActiveModuleId] = useState(productModules[0].id);
  const [activeVerticalId, setActiveVerticalId] = useState(verticalStories[0].id);

  const activeModule =
    productModules.find((module) => module.id === activeModuleId) ?? productModules[0];
  const activeVertical =
    verticalStories.find((vertical) => vertical.id === activeVerticalId) ?? verticalStories[0];

  return (
    <div className="grid gap-8 lg:grid-cols-[1.2fr_0.8fr]">
      <section className="card overflow-hidden">
        <div className="border-b border-surface-600/60 p-6">
          <p className="text-xs font-semibold uppercase tracking-[0.32em] text-neon-cyan/80">
            Platform Modules
          </p>
          <h3 className="mt-3 text-2xl font-bold text-white">
            One product surface. Five linked workflows.
          </h3>
          <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-300">
            Shift the story from a bundle of disconnected features to an operating sequence:
            discover the estate, understand behavior and risk, generate modernization assets,
            apply quantum-safe protection, and prove control with evidence.
          </p>
        </div>

        <div className="grid gap-5 p-6">
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
            {productModules.map((module, index) => {
              const active = module.id === activeModule.id;

              return (
                <button
                  key={module.id}
                  type="button"
                  onClick={() => setActiveModuleId(module.id)}
                  className={`rounded-2xl border p-4 text-left transition-all duration-300 ${
                    active
                      ? 'border-neon-cyan/70 bg-neon-cyan/10 shadow-lg shadow-neon-cyan/10'
                      : 'border-surface-600/70 bg-surface-900/60 hover:border-neon-cyan/30 hover:bg-surface-800/80'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold uppercase tracking-[0.28em] text-slate-400">
                      0{index + 1}
                    </span>
                    <span
                      className={`h-2.5 w-2.5 rounded-full ${
                        active ? 'bg-neon-cyan' : 'bg-surface-500'
                      }`}
                    />
                  </div>
                  <div className="mt-4 text-lg font-semibold text-white">{module.label}</div>
                  <div className="mt-1 text-xs uppercase tracking-[0.2em] text-neon-copper/80">
                    {module.eyebrow}
                  </div>
                  <p className="mt-3 text-sm leading-6 text-slate-300">{module.headline}</p>
                </button>
              );
            })}
          </div>

          <div className="rounded-3xl border border-surface-600/70 bg-surface-900/70 p-6">
            <div className="flex flex-wrap items-center gap-3">
              <span className={`inline-flex rounded-full bg-gradient-to-r px-3 py-1 text-xs font-semibold uppercase tracking-[0.24em] text-white ${activeModule.color}`}>
                {activeModule.label}
              </span>
              <a
                href={activeModule.href}
                className="text-sm font-medium text-neon-cyan transition-colors hover:text-white"
              >
                Open module page
              </a>
            </div>

            <h4 className="mt-5 text-2xl font-semibold text-white">{activeModule.headline}</h4>
            <p className="mt-3 max-w-2xl text-sm leading-7 text-slate-300">
              {activeModule.summary}
            </p>

            <div className="mt-6 grid gap-6 lg:grid-cols-2">
              <div>
                <div className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">
                  Operator outcomes
                </div>
                <ul className="mt-4 space-y-3 text-sm text-slate-200">
                  {activeModule.outcomes.map((item) => (
                    <li key={item} className="flex items-start gap-3">
                      <span className="mt-1 h-2 w-2 rounded-full bg-neon-green" />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <div className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">
                  Evidence produced
                </div>
                <div className="mt-4 flex flex-wrap gap-2">
                  {activeModule.evidence.map((item) => (
                    <span
                      key={item}
                      className="rounded-full border border-surface-500/70 bg-surface-800/90 px-3 py-1 text-xs text-slate-200"
                    >
                      {item}
                    </span>
                  ))}
                </div>

                <div className="mt-6 text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">
                  Protocols and artifacts in scope
                </div>
                <div className="mt-4 flex flex-wrap gap-2">
                  {activeModule.protocols.map((item) => (
                    <span
                      key={item}
                      className="rounded-full border border-neon-cyan/20 bg-neon-cyan/10 px-3 py-1 text-xs text-neon-cyan"
                    >
                      {item}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <aside className="card overflow-hidden">
        <div className="border-b border-surface-600/60 p-6">
          <p className="text-xs font-semibold uppercase tracking-[0.32em] text-neon-cyan/80">
            Domain Fit
          </p>
          <h3 className="mt-3 text-2xl font-bold text-white">
            Tune the message by industry instead of selling generic AI.
          </h3>
        </div>

        <div className="grid gap-4 p-6">
          <div className="flex flex-wrap gap-2">
            {verticalStories.map((vertical) => {
              const active = vertical.id === activeVertical.id;
              return (
                <button
                  key={vertical.id}
                  type="button"
                  onClick={() => setActiveVerticalId(vertical.id)}
                  className={`rounded-full border px-3 py-2 text-sm transition-colors ${
                    active
                      ? 'border-neon-cyan bg-neon-cyan/10 text-neon-cyan'
                      : 'border-surface-600/70 bg-surface-900/60 text-slate-300 hover:border-neon-cyan/40 hover:text-white'
                  }`}
                >
                  {vertical.label}
                </button>
              );
            })}
          </div>

          <div className="rounded-3xl border border-surface-600/70 bg-surface-900/70 p-5">
            <div className="text-xs font-semibold uppercase tracking-[0.24em] text-neon-copper/80">
              {activeVertical.label}
            </div>
            <h4 className="mt-3 text-xl font-semibold text-white">{activeVertical.title}</h4>
            <p className="mt-3 text-sm leading-7 text-slate-300">{activeVertical.challenge}</p>

            <div className="mt-5 rounded-2xl border border-surface-600/70 bg-surface-800/80 p-4">
              <div className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">
                Why QBITEL fits
              </div>
              <p className="mt-3 text-sm leading-6 text-slate-200">{activeVertical.qbitelFit}</p>
            </div>

            <div className="mt-5 grid gap-4">
              <div>
                <div className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">
                  Recommended PQC profile
                </div>
                <p className="mt-2 text-sm leading-6 text-slate-200">{activeVertical.pqcProfile}</p>
              </div>

              <div>
                <div className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">
                  Deployment motion
                </div>
                <p className="mt-2 text-sm leading-6 text-slate-200">{activeVertical.deployment}</p>
              </div>

              <div>
                <div className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">
                  Program priorities
                </div>
                <div className="mt-3 flex flex-wrap gap-2">
                  {activeVertical.priorities.map((item) => (
                    <span
                      key={item}
                      className="rounded-full border border-surface-500/80 bg-surface-800 px-3 py-1 text-xs text-slate-200"
                    >
                      {item}
                    </span>
                  ))}
                </div>
              </div>

              <div>
                <div className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">
                  Common protocols
                </div>
                <div className="mt-3 flex flex-wrap gap-2">
                  {activeVertical.protocols.map((item) => (
                    <span
                      key={item}
                      className="rounded-full border border-neon-cyan/20 bg-neon-cyan/10 px-3 py-1 text-xs text-neon-cyan"
                    >
                      {item}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </aside>
    </div>
  );
}
