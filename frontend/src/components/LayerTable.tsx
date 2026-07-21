/**
 * components/LayerTable.tsx
 *
 * Why this file exists:
 *   Matches the reference UI's bottom "Layer Table" panel - a plain
 *   sortable-in-spirit list of every node in the graph. Sorting/filtering
 *   interactivity can be added in Phase 5 without changing the data shape.
 *
 * How it connects:
 *   Rendered by App.tsx, receives the loaded UniversalGraph's nodes.
 */

import type { UniversalGraph } from "../types/graph";

interface LayerTableProps {
  graph: UniversalGraph;
}

export default function LayerTable({ graph }: LayerTableProps) {
  return (
    <div className="bg-panel rounded-xl p-5 h-full overflow-y-auto">
      <h3 className="text-white font-semibold mb-3">Layer Table</h3>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-gray-500 border-b border-white/5">
            <th className="pb-2 font-normal">#</th>
            <th className="pb-2 font-normal">Layer</th>
            <th className="pb-2 font-normal">Type</th>
            <th className="pb-2 font-normal">Output Shape</th>
            <th className="pb-2 font-normal">Params</th>
          </tr>
        </thead>
        <tbody>
          {graph.nodes.map((n, i) => (
            <tr key={n.id} className="border-b border-white/5 last:border-0 text-gray-300">
              <td className="py-2">{i + 1}</td>
              <td className="py-2">{n.label}</td>
              <td className="py-2">
                <span className="text-xs px-2 py-0.5 rounded bg-white/5">{n.type}</span>
              </td>
              <td className="py-2 text-gray-400">{n.output_shape?.join(" × ") ?? "—"}</td>
              <td className="py-2 text-gray-400">{n.params.toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
