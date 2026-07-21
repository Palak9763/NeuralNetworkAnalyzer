/**
 * components/ModelSummary.tsx
 *
 * Why this file exists:
 *   Matches the reference UI's bottom "Model Summary" card. In Phase 1/2,
 *   only fields the backend actually returns are shown (total layers,
 *   total params, framework). FLOPs and model size are left as "—" until
 *   Phase 5 (torchinfo/fvcore integration) populates them.
 *
 * How it connects:
 *   Rendered by App.tsx, receives the loaded UniversalGraph.
 */

import type { UniversalGraph } from "../types/graph";

interface ModelSummaryProps {
  graph: UniversalGraph;
}

function StatRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex justify-between py-1.5 text-sm border-b border-white/5 last:border-0">
      <span className="text-gray-500">{label}</span>
      <span className="text-gray-200 font-medium">{value}</span>
    </div>
  );
}

export default function ModelSummary({ graph }: ModelSummaryProps) {
  return (
    <div className="bg-panel rounded-xl p-5 h-full">
      <h3 className="text-white font-semibold mb-3">Model Summary</h3>
      <StatRow label="Total Layers" value={graph.meta.total_layers} />
      <StatRow label="Total Parameters" value={graph.meta.total_params.toLocaleString()} />
      <StatRow label="Framework" value={graph.meta.framework} />
      <StatRow label="Confidence" value={graph.meta.confidence} />
      <StatRow label="FLOPs" value={graph.meta.flops ?? "— (Phase 5)"} />
      {graph.meta.warnings.length > 0 && (
        <div className="mt-3 text-xs text-yellow-300 bg-yellow-500/10 rounded-lg p-2">
          {graph.meta.warnings[0]}
        </div>
      )}
    </div>
  );
}
