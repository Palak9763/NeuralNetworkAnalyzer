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
  const handleDownload = () => {
    if (!graph) return;
    const dataStr = JSON.stringify(graph, null, 2);
    const dataBlob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${graph.model_name || "graph"}-architecture.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleShare = async () => {
    if (!graph) return;
    // Prefer sharing a job-specific link so recipients can load the same graph
    const jobId = (graph as any).job_id || (graph.job_id as unknown as string);
    const base = window.location.origin + window.location.pathname;
    const shareUrl = jobId ? `${base}?job=${encodeURIComponent(jobId)}` : window.location.href;

    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(shareUrl);
        alert("Share link copied to clipboard");
        return;
      }
    } catch (e) {
      // fallthrough to copying graph JSON
    }

    try {
      await navigator.clipboard.writeText(JSON.stringify(graph));
      alert("Graph JSON copied to clipboard");
    } catch (e) {
      alert("Unable to copy to clipboard");
    }
  };

  const handleExport = async () => {
    if (!graph) return;
    const wrapper = document.getElementById("reactflow-wrapper");
    if (!wrapper) {
      alert("Unable to find graph area for export.");
      return;
    }

    const rect = wrapper.getBoundingClientRect();
    const width = Math.max(wrapper.scrollWidth, rect.width);
    const height = Math.max(wrapper.scrollHeight, rect.height);
    const origWidth = wrapper.style.width;
    const origHeight = wrapper.style.height;
    const origOverflow = wrapper.style.overflow;

    try {
      wrapper.style.width = `${width}px`;
      wrapper.style.height = `${height}px`;
      wrapper.style.overflow = "visible";

      const htmlToImage = await import("html-to-image");
      const dataUrl = await htmlToImage.toPng(wrapper as HTMLElement, {
        width,
        height,
        backgroundColor: "#0b0f17",
        pixelRatio: window.devicePixelRatio || 2,
        cacheBust: true,
      });

      const link = document.createElement("a");
      link.href = dataUrl;
      link.download = `${graph.model_name || "graph"}-graph.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      return;
    } catch (e) {
      console.error("Graph export failed:", e);
      alert("Export failed. Please try again or use Download to save JSON.");
      return;
    } finally {
      wrapper.style.width = origWidth;
      wrapper.style.height = origHeight;
      wrapper.style.overflow = origOverflow;
    }
  };

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
        <button 
          onClick={handleDownload}
          disabled={!graph}
          className="text-sm text-gray-300 border border-white/10 px-3 py-1.5 rounded-lg hover:bg-white/5 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          ↓ Download
        </button>
        <button
          onClick={handleShare}
          className="text-sm text-gray-300 border border-white/10 px-3 py-1.5 rounded-lg hover:bg-white/5"
        >
          ⇧ Share
        </button>
        <button
          onClick={handleExport}
          className="text-sm bg-accent text-white px-3 py-1.5 rounded-lg hover:bg-accent/90"
        >
          Export
        </button>
      </div>
    </header>
  );
}
