/**
 * components/TopBar.tsx
 *
 * Why this file exists:
 *   Matches the reference UI's top bar: model name, framework badge,
 *   confidence badge, and (visual-only in Phase 1) Download/Share/Export
 *   buttons.
 *
 * How it connects:
 *   Rendered by App.tsx, receives the currently loaded UniversalGraph.
 */

import type { UniversalGraph } from "../types/graph";

interface TopBarProps {
  graph: UniversalGraph | null;
}

export default function TopBar({ graph }: TopBarProps) {
  return (
    <header className="h-16 shrink-0 border-b border-white/5 flex items-center justify-between px-6 bg-panel">
      <div className="flex items-center gap-3">
        <h1 className="text-white font-semibold text-lg">
          {graph ? graph.model_name : "No model loaded"}
        </h1>
        {graph && (
          <>
            <span className="text-xs px-2 py-1 rounded-md bg-blue-500/20 text-blue-300 capitalize">
              {graph.meta.framework}
            </span>
            <span
              className={`text-xs px-2 py-1 rounded-md capitalize ${
                graph.meta.confidence === "traced"
                  ? "bg-green-500/20 text-green-300"
                  : "bg-yellow-500/20 text-yellow-300"
              }`}
            >
              {graph.meta.confidence === "traced" ? "Traced" : "Static Analysis"}
            </span>
          </>
        )}
      </div>

      <div className="flex items-center gap-2">
        <button className="text-sm text-gray-300 border border-white/10 px-3 py-1.5 rounded-lg hover:bg-white/5" disabled>
          ↓ Download
        </button>
        <button className="text-sm text-gray-300 border border-white/10 px-3 py-1.5 rounded-lg hover:bg-white/5" disabled>
          ⇧ Share
        </button>
        <button className="text-sm bg-accent text-white px-3 py-1.5 rounded-lg hover:bg-accent/90" disabled>
          Export
        </button>
      </div>
    </header>
  );
}
