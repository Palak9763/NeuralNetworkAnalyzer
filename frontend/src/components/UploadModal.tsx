/**
 * components/UploadModal.tsx
 *
 * Why this file exists:
 *   Handles the actual file picking + upload + polling workflow. Kept
 *   separate from App.tsx so upload-specific state (loading, error,
 *   drag state) doesn't clutter the main layout component.
 *
 * What it does:
 *   - Lets the user pick a .py or .zip file
 *   - Calls uploadProject(), then fetchGraph() once
 *   - Reports success (the graph) or failure (error message) to the parent
 *
 * How it connects:
 *   Rendered by App.tsx when the user clicks "Upload Project" in the
 *   sidebar. Uses api/client.ts for both HTTP calls.
 */

import { useState } from "react";
import { fetchGraph, uploadProject } from "../api/client";
import type { UniversalGraph } from "../types/graph";

interface UploadModalProps {
  onClose: () => void;
  onGraphReady: (graph: UniversalGraph) => void;
}

export default function UploadModal({ onClose, onGraphReady }: UploadModalProps) {
  const [status, setStatus] = useState<"idle" | "uploading" | "parsing" | "error">("idle");
  const [error, setError] = useState<string | null>(null);

  async function handleFile(file: File) {
    setStatus("uploading");
    setError(null);
    try {
      const uploadRes = await uploadProject(file);
      setStatus("parsing");
      const graph = await fetchGraph(uploadRes.job_id);
      onGraphReady(graph);
      onClose();
    } catch (err: any) {
      setStatus("error");
      const detail = err?.response?.data?.detail ?? err.message ?? "Unknown error";
      setError(detail);
    }
  }

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
      <div className="bg-panel rounded-xl p-6 w-[420px] border border-white/10">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-white font-semibold">Upload Project</h2>
          <button onClick={onClose} className="text-gray-500 hover:text-white">
            ✕
          </button>
        </div>

        <label className="block border-2 border-dashed border-white/10 rounded-lg p-8 text-center cursor-pointer hover:border-accent/50 transition">
          <input
            type="file"
            accept=".py,.zip"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
          />
          <div className="text-gray-400 text-sm">
            {status === "idle" && "Click to select a .py file or .zip project"}
            {status === "uploading" && "Uploading…"}
            {status === "parsing" && "Parsing model architecture…"}
          </div>
        </label>

        {status === "error" && (
          <div className="mt-4 text-sm text-red-300 bg-red-500/10 rounded-lg p-3">{error}</div>
        )}

        <p className="text-xs text-gray-600 mt-4">
          Phase 1 supports PyTorch models only (torch.fx tracing with AST fallback).
        </p>
      </div>
    </div>
  );
}
