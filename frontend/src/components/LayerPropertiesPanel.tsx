/**
 * components/LayerPropertiesPanel.tsx
 *
 * Why this file exists:
 *   Matches the reference UI's right-hand "Layer Properties" panel.
 *   Shows full metadata for whichever node was last clicked in
 *   GraphCanvas, plus a static legend.
 *
 * How it connects:
 *   Rendered by App.tsx. Receives the selected GraphNode as a prop,
 *   updated via GraphCanvas's onNodeClick callback.
 */

import type { GraphNode } from "../types/graph";

interface LayerPropertiesPanelProps {
  node: GraphNode | null;
}

const LEGEND = [
  { label: "Convolution", color: "#2563eb" },
  { label: "Normalization", color: "#eab308" },
  { label: "Activation", color: "#22c55e" },
  { label: "Pooling", color: "#a855f7" },
  { label: "Fully Connected", color: "#f43f5e" },
  { label: "Other", color: "#334155" },
];

function Row({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex justify-between py-1.5 text-sm">
      <span className="text-gray-500">{label}</span>
      <span className="text-gray-200 font-medium">{value}</span>
    </div>
  );
}

export default function LayerPropertiesPanel({ node }: LayerPropertiesPanelProps) {
  return (
    <aside className="w-80 shrink-0 border-l border-white/5 bg-panel h-full overflow-y-auto p-5 text-gray-300">
      <h2 className="text-white font-semibold mb-4">Layer Properties</h2>

      {node ? (
        <div className="mb-6">
          <div className="text-white font-semibold text-base mb-3">{node.type}</div>
          <Row label="Label" value={node.label} />
          <Row label="Params" value={node.params.toLocaleString()} />
          <Row label="Input Shape" value={node.input_shape?.join(" × ") ?? "—"} />
          <Row label="Output Shape" value={node.output_shape?.join(" × ") ?? "—"} />
        </div>
      ) : (
        <p className="text-sm text-gray-500 mb-6">Click a node in the diagram to see its details here.</p>
      )}

      <h3 className="text-white font-semibold mb-3 text-sm">Legend</h3>
      <div className="grid grid-cols-2 gap-2">
        {LEGEND.map((item) => (
          <div key={item.label} className="flex items-center gap-2 text-xs">
            <span className="w-2.5 h-2.5 rounded-sm inline-block" style={{ background: item.color }} />
            {item.label}
          </div>
        ))}
      </div>
    </aside>
  );
}
