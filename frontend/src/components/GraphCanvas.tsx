/**
 * components/GraphCanvas.tsx
 *
 * Why this file exists:
 *   This is Phase 2's core deliverable - a "dumb renderer" that takes the
 *   backend's UniversalGraph JSON and draws it with React Flow. No
 *   grouping/layout intelligence lives here (that's Phase 3/4) - this
 *   component only knows how to turn {nodes, edges} into boxes and
 *   arrows, using a simple layered auto-layout as a placeholder until
 *   Dagre.js/ELK.js are integrated in Phase 4.
 *
 * What it does:
 *   - Converts UniversalGraph nodes into React Flow nodes, computing a
 *     simple top-to-bottom position based on graph depth (BFS from
 *     inputs) since no dedicated layout engine is wired up yet
 *   - Converts edges, styling skip-connections differently
 *   - Calls onNodeClick when a node is clicked, so the parent can show
 *     Layer Properties in the side panel
 *
 * How it connects:
 *   Rendered by App.tsx with the graph fetched via api/client.fetchGraph.
 *   Clicking a node feeds LayerPropertiesPanel.tsx via App.tsx's state.
 */

import { useMemo } from "react";
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  type Edge,
  type Node,
} from "reactflow";
import "reactflow/dist/style.css";
import type { GraphNode, UniversalGraph } from "../types/graph";

interface GraphCanvasProps {
  graph: UniversalGraph;
  onNodeClick: (node: GraphNode) => void;
}

const LAYER_TYPE_COLORS: Record<string, string> = {
  Conv2d: "#2563eb",
  Linear: "#f43f5e",
  BatchNorm2d: "#eab308",
  ReLU: "#22c55e",
  MaxPool2d: "#a855f7",
  Flatten: "#64748b",
};

function computeDepths(graph: UniversalGraph): Map<string, number> {
  const incoming = new Map<string, string[]>();
  graph.nodes.forEach((n) => incoming.set(n.id, []));
  graph.edges.forEach((e) => incoming.get(e.target)?.push(e.source));

  const depths = new Map<string, number>();
  const roots = graph.nodes.filter((n) => (incoming.get(n.id) ?? []).length === 0);
  const queue: [string, number][] = roots.map((r) => [r.id, 0]);

  while (queue.length > 0) {
    const [id, depth] = queue.shift()!;
    if (depths.has(id) && depths.get(id)! >= depth) continue;
    depths.set(id, depth);
    graph.edges
      .filter((e) => e.source === id)
      .forEach((e) => queue.push([e.target, depth + 1]));
  }

  // Any node with no computed depth (disconnected) gets appended at the end.
  graph.nodes.forEach((n, i) => {
    if (!depths.has(n.id)) depths.set(n.id, i);
  });

  return depths;
}

export default function GraphCanvas({ graph, onNodeClick }: GraphCanvasProps) {
  const { nodes, edges } = useMemo(() => {
    const depths = computeDepths(graph);
    const countPerDepth = new Map<number, number>();

    const rfNodes: Node[] = graph.nodes.map((n) => {
      const depth = depths.get(n.id) ?? 0;
      const col = countPerDepth.get(depth) ?? 0;
      countPerDepth.set(depth, col + 1);

      return {
        id: n.id,
        position: { x: col * 220, y: depth * 130 },
        data: {
          label: (
            <div>
              <div className="font-semibold text-sm">{n.type}</div>
              <div className="text-[10px] opacity-70">{n.label}</div>
              {n.output_shape && (
                <div className="text-[10px] opacity-60 mt-1">
                  → {n.output_shape.join("×")}
                </div>
              )}
            </div>
          ),
          raw: n,
        },
        style: {
          background: LAYER_TYPE_COLORS[n.type] ?? "#334155",
          color: "white",
          borderRadius: 10,
          padding: 10,
          width: 170,
          border: "1px solid rgba(255,255,255,0.15)",
        },
      };
    });

    const rfEdges: Edge[] = graph.edges.map((e, i) => ({
      id: `edge-${i}`,
      source: e.source,
      target: e.target,
      animated: e.is_skip_connection,
      style: e.is_skip_connection
        ? { stroke: "#f43f5e", strokeDasharray: "4 2" }
        : { stroke: "#64748b" },
    }));

    return { nodes: rfNodes, edges: rfEdges };
  }, [graph]);

  return (
    <div id="reactflow-wrapper" style={{ width: "100%", height: "100%" }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodeClick={(_, node) => onNodeClick(node.data.raw as GraphNode)}
        fitView
        proOptions={{ hideAttribution: true }}
      >
        <Background color="#1f2230" gap={20} />
        <Controls />
        <MiniMap
          nodeColor={(n) => (n.style?.background as string) ?? "#334155"}
          maskColor="rgba(13,14,20,0.8)"
          style={{ background: "#12141c" }}
        />
      </ReactFlow>
    </div>
  );
}
