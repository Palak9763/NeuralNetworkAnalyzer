/**
 * components/GraphCanvas.tsx
 *
 * Why this file exists:
 *   Renders the backend's UniversalGraph JSON with React Flow. As of
 *   Phase 4, all position/sizing math (including Phase 3's group
 *   container nesting) lives in lib/layoutGraph.tsx, which uses Dagre for
 *   real graph layout instead of a hand-rolled single-column stack. This
 *   component is now a thin rendering layer: fetch layout, hand it to
 *   ReactFlow, wire up the click handler.
 *
 * How it connects:
 *   Rendered by App.tsx with the graph fetched via api/client.fetchGraph.
 *   Delegates to lib/layoutGraph.layoutGraph() for all node/edge
 *   positioning. Clicking a leaf node calls onNodeClick; clicking a group
 *   container does nothing (group nodes are non-selectable).
 */

import { useMemo } from "react";
import ReactFlow, { Background, Controls, MiniMap } from "reactflow";
import "reactflow/dist/style.css";
import { layoutGraph } from "../lib/layoutGraph";
import type { GraphNode, UniversalGraph } from "../types/graph";

interface GraphCanvasProps {
  graph: UniversalGraph;
  onNodeClick: (node: GraphNode) => void;
}

export default function GraphCanvas({ graph, onNodeClick }: GraphCanvasProps) {
  const { nodes, edges } = useMemo(() => layoutGraph(graph), [graph]);

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodeClick={(_, node) => node.data.raw && onNodeClick(node.data.raw as GraphNode)}
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
  );
}