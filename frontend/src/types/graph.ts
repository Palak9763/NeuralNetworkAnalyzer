/**
 * types/graph.ts
 *
 * Why this file exists:
 *   Mirrors backend/app/schemas/graph.py exactly. This is the frontend
 *   half of the Universal Graph contract - keeping both sides in sync
 *   manually (rather than auto-generating) is fine at this project size,
 *   but if the backend schema changes, this file must change with it.
 *
 * How it connects:
 *   Used by api/client.ts (typed fetch responses) and every component
 *   that renders graph data (GraphCanvas, LayerPropertiesPanel, etc).
 */

export type Framework = "pytorch" | "tensorflow" | "jax" | "unknown";
export type Confidence = "traced" | "static" | "partial";
export type GroupType = "conv_block" | "residual_block" | "stage";

export interface GraphGroup {
  id: string;
  label: string;
  type: GroupType;
  member_node_ids: string[];
  parent_group_id: string | null;
  repeat_count: number;
}

export interface GraphNode {
  id: string;
  type: string;
  label: string;
  input_shape: number[] | null;
  output_shape: number[] | null;
  params: number;
  group_id: string | null;
}

export interface GraphEdge {
  source: string;
  target: string;
  is_skip_connection: boolean;
}

export interface GraphMeta {
  framework: Framework;
  confidence: Confidence;
  total_params: number;
  total_layers: number;
  flops: number | null;
  warnings: string[];
}

export interface UniversalGraph {
  job_id: string;
  model_name: string;
  meta: GraphMeta;
  nodes: GraphNode[];
  edges: GraphEdge[];
  groups: GraphGroup[];
}

export interface UploadResponse {
  job_id: string;
  filename: string;
  status: string;
}
