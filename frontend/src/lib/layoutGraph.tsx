/**
 * lib/layoutGraph.tsx
 *
 * Why this file exists:
 *   Turns the backend's UniversalGraph JSON into positioned React Flow
 *   nodes/edges using Dagre's compound-graph layout. This version brings
 *   the diagram closer to the reference UI in two ways:
 *
 *   1. Horizontal flow (rankdir "LR" instead of "TB"). A long chain of
 *      layers reads left-to-right, like the reference's stem
 *      (Input -> Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d) instead of
 *      stacking into one long vertical column.
 *
 *   2. Explicit residual-merge nodes. Previously a skip connection was
 *      just a dashed line running behind other boxes, which is hard to
 *      read. Now, wherever a node receives both a normal edge and a
 *      skip-connection edge (i.e. a residual add), we insert a small "+"
 *      circle node at that merge point - both incoming paths point INTO
 *      the circle, and a single edge continues from the circle into the
 *      original target. This mirrors the "⊕" merge indicator in the
 *      reference UI's ResNet diagram.
 *
 * How it connects:
 *   Called by components/GraphCanvas.tsx. Takes the raw UniversalGraph
 *   and returns ready-to-render React Flow nodes (with parentNode/extent
 *   already set) and edges - GraphCanvas just renders what this returns.
 */

import dagre from "dagre";
import type { Node, Edge } from "reactflow";
import type { GraphGroup, UniversalGraph } from "../types/graph";

export const NODE_WIDTH = 170;
export const NODE_HEIGHT = 64;
const MERGE_NODE_SIZE = 30;

const GROUP_LABEL_HEIGHT = 28;
const STAGE_LABEL_HEIGHT = 48;
const GROUP_PADDING = 20;

const LAYER_TYPE_COLORS: Record<string, string> = {
  Conv2d: "#2563eb",
  Linear: "#f43f5e",
  BatchNorm2d: "#eab308",
  ReLU: "#22c55e",
  MaxPool2d: "#a855f7",
  AdaptiveAvgPool2d: "#a855f7",
  Flatten: "#64748b",
};

const GROUP_STYLES: Record<GraphGroup["type"], { border: string; background: string }> = {
  conv_block: { border: "#2563eb88", background: "rgba(37,99,235,0.06)" },
  residual_block: { border: "#a855f788", background: "rgba(168,85,247,0.06)" },
  stage: { border: "#f59e0b88", background: "rgba(245,158,11,0.04)" },
};

interface LayoutResult {
  nodes: Node[];
  edges: Edge[];
}

interface ReroutedEdge {
  source: string;
  target: string;
  isSkip: boolean;
}

/**
 * Finds every node that receives both a normal edge and a skip-connection
 * edge (a residual merge point), and rewrites the edge list so both
 * incoming paths point into a new synthetic "+" merge node instead of
 * directly into the original target - with one edge continuing from the
 * merge node into that target. Nodes with no such merge are left as-is.
 */
function insertMergeNodes(graph: UniversalGraph): {
  mergeNodes: { id: string; parentGroupId?: string }[];
  edges: ReroutedEdge[];
} {
  const edgesByTarget = new Map<string, typeof graph.edges>();
  graph.edges.forEach((e) => {
    const list = edgesByTarget.get(e.target) ?? [];
    list.push(e);
    edgesByTarget.set(e.target, list);
  });

  const nodeGroupId = new Map(graph.nodes.map((n) => [n.id, n.group_id ?? undefined]));
  const mergeNodes: { id: string; parentGroupId?: string }[] = [];
  const edges: ReroutedEdge[] = [];
  let mergeCounter = 0;

  for (const [target, incoming] of edgesByTarget) {
    const hasSkip = incoming.some((e) => e.is_skip_connection);
    if (!hasSkip || incoming.length < 2) {
      incoming.forEach((e) => edges.push({ source: e.source, target: e.target, isSkip: e.is_skip_connection }));
      continue;
    }

    mergeCounter += 1;
    const mergeId = `merge_${mergeCounter}`;
    mergeNodes.push({ id: mergeId, parentGroupId: nodeGroupId.get(target) });

    incoming.forEach((e) => edges.push({ source: e.source, target: mergeId, isSkip: e.is_skip_connection }));
    edges.push({ source: mergeId, target, isSkip: false });
  }

  return { mergeNodes, edges };
}

/**
 * Lays out the graph using Dagre's compound-graph mode with a
 * left-to-right flow direction, and explicit residual-merge nodes.
 */
export function layoutGraph(graph: UniversalGraph): LayoutResult {
  const { mergeNodes, edges: workingEdges } = insertMergeNodes(graph);

  const g = new dagre.graphlib.Graph({ compound: true });
  g.setGraph({ rankdir: "LR", nodesep: 40, ranksep: 90, marginx: 20, marginy: 20 });
  g.setDefaultEdgeLabel(() => ({}));

  const groupById = new Map(graph.groups.map((grp) => [grp.id, grp]));
  const leafGroups = graph.groups.filter((grp) => grp.type !== "stage");
  const stageGroups = graph.groups.filter((grp) => grp.type === "stage");

  stageGroups.forEach((stage) => {
    g.setNode(stage.id, { label: stage.label, width: 1, height: 1 });
  });
  leafGroups.forEach((grp) => {
    g.setNode(grp.id, { label: grp.label, width: 1, height: 1 });
    if (grp.parent_group_id) g.setParent(grp.id, grp.parent_group_id);
  });

  graph.nodes.forEach((n) => {
    g.setNode(n.id, { width: NODE_WIDTH, height: NODE_HEIGHT });
    if (n.group_id) g.setParent(n.id, n.group_id);
  });

  mergeNodes.forEach((m) => {
    g.setNode(m.id, { width: MERGE_NODE_SIZE, height: MERGE_NODE_SIZE });
    if (m.parentGroupId) g.setParent(m.id, m.parentGroupId);
  });

  workingEdges.forEach((e) => {
    g.setEdge(e.source, e.target);
  });

  dagre.layout(g);

  // See Phase 3/4 notes: Dagre auto-sizes cluster containers tightly
  // around their children with no room for our label text, so every
  // group's box is inflated after layout and children are positioned
  // relative to that inflated origin, not Dagre's tight one.
  //
  // Known limitation (unchanged from the previous version): this
  // inflation happens after Dagre has already spaced sibling nodes at
  // the same rank, so two groups landing side-by-side at the same rank
  // could visually overlap slightly. Sequential stage architectures
  // (the overwhelming majority of real CNNs) are unaffected.
  const rfNodes: Node[] = [];

  const dagreNodeAbs = (id: string) => {
    const d = g.node(id);
    return { left: d.x - d.width / 2, top: d.y - d.height / 2, width: d.width, height: d.height };
  };

  const inflatedBox = new Map<string, { left: number; top: number; width: number; height: number }>();

  stageGroups.forEach((stage) => {
    const tight = dagreNodeAbs(stage.id);
    inflatedBox.set(stage.id, {
      left: tight.left - GROUP_PADDING,
      top: tight.top - STAGE_LABEL_HEIGHT,
      width: tight.width + GROUP_PADDING * 2,
      height: tight.height + STAGE_LABEL_HEIGHT + GROUP_PADDING,
    });
  });
  leafGroups.forEach((grp) => {
    const tight = dagreNodeAbs(grp.id);
    inflatedBox.set(grp.id, {
      left: tight.left - GROUP_PADDING,
      top: tight.top - GROUP_LABEL_HEIGHT,
      width: tight.width + GROUP_PADDING * 2,
      height: tight.height + GROUP_LABEL_HEIGHT + GROUP_PADDING,
    });
  });

  stageGroups.forEach((stage) => {
    const box = inflatedBox.get(stage.id)!;
    const style = GROUP_STYLES.stage;
    rfNodes.push({
      id: stage.id,
      type: "group",
      position: { x: box.left, y: box.top },
      style: {
        width: box.width,
        height: box.height,
        background: style.background,
        border: `1.5px dashed ${style.border}`,
        borderRadius: 14,
      },
      data: { label: null },
      selectable: false,
      draggable: false,
      zIndex: 0,
    });
    rfNodes.push({
      id: `${stage.id}__label`,
      position: { x: box.left + 12, y: box.top + 6 },
      data: { label: <div className="text-amber-300 text-xs font-semibold">{stage.label}</div> },
      style: { background: "transparent", border: "none", width: "auto", padding: 0 },
      selectable: false,
      draggable: false,
      zIndex: 5,
    });
  });

  leafGroups.forEach((grp) => {
    const box = inflatedBox.get(grp.id)!;
    const style = GROUP_STYLES[grp.type];
    const parentBox = grp.parent_group_id ? inflatedBox.get(grp.parent_group_id) : null;
    const relX = parentBox ? box.left - parentBox.left : box.left;
    const relY = parentBox ? box.top - parentBox.top : box.top;

    rfNodes.push({
      id: grp.id,
      type: "group",
      position: { x: relX, y: relY },
      parentNode: grp.parent_group_id ?? undefined,
      extent: grp.parent_group_id ? "parent" : undefined,
      style: {
        width: box.width,
        height: box.height,
        background: style.background,
        border: `1.5px solid ${style.border}`,
        borderRadius: 10,
      },
      data: { label: null },
      selectable: false,
      draggable: false,
      zIndex: 1,
    });
    rfNodes.push({
      id: `${grp.id}__label`,
      position: { x: relX + 10, y: relY + 4 },
      parentNode: grp.parent_group_id ?? undefined,
      extent: grp.parent_group_id ? "parent" : undefined,
      data: { label: <div className="text-xs font-semibold" style={{ color: style.border }}>{grp.label}</div> },
      style: { background: "transparent", border: "none", width: "auto", padding: 0 },
      selectable: false,
      draggable: false,
      zIndex: 5,
    });
  });

  const nodeRelativePosition = (id: string, groupId?: string) => {
    const abs = dagreNodeAbs(id);
    const parentBox = groupId ? inflatedBox.get(groupId) : null;
    return {
      x: parentBox ? abs.left - parentBox.left : abs.left,
      y: parentBox ? abs.top - parentBox.top : abs.top,
    };
  };

  graph.nodes.forEach((n) => {
    const leafGroup = n.group_id ? groupById.get(n.group_id) : undefined;
    const position = nodeRelativePosition(n.id, leafGroup?.id);

    rfNodes.push({
      id: n.id,
      position,
      parentNode: leafGroup?.id,
      extent: leafGroup ? "parent" : undefined,
      data: {
        label: (
          <div>
            <div className="font-semibold text-sm">{n.type}</div>
            <div className="text-[10px] opacity-70">{n.label}</div>
            {n.output_shape && (
              <div className="text-[10px] opacity-60 mt-1">→ {n.output_shape.join("×")}</div>
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
        width: NODE_WIDTH,
        border: "1px solid rgba(255,255,255,0.15)",
      },
      zIndex: 10,
    });
  });

  // Explicit "+" merge circles - one per residual-add point, matching the
  // reference UI's "⊕" indicator instead of a bare dashed line.
  mergeNodes.forEach((m) => {
    const position = nodeRelativePosition(m.id, m.parentGroupId);
    rfNodes.push({
      id: m.id,
      position,
      parentNode: m.parentGroupId,
      extent: m.parentGroupId ? "parent" : undefined,
      data: { label: <span className="text-white text-sm font-bold">+</span> },
      style: {
        width: MERGE_NODE_SIZE,
        height: MERGE_NODE_SIZE,
        borderRadius: "50%",
        background: "#a855f7",
        border: "1.5px solid #e9d5ff",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 0,
      },
      selectable: false,
      draggable: false,
      zIndex: 10,
    });
  });

  const rfEdges: Edge[] = workingEdges.map((e, i) => ({
    id: `edge-${i}`,
    source: e.source,
    target: e.target,
    animated: e.isSkip,
    style: e.isSkip
      ? { stroke: "#f43f5e", strokeDasharray: "4 2" }
      : { stroke: "#64748b" },
    zIndex: 8,
  }));

  return { nodes: rfNodes, edges: rfEdges };
}

// Re-exported so consumers don't need to import group-related constants
// from two different places.
export const LAYOUT_CONSTANTS = { GROUP_LABEL_HEIGHT, STAGE_LABEL_HEIGHT, GROUP_PADDING };