/**
 * components/HistoryPage.tsx
 *
 * Full-page upload history view. Shows all previously uploaded models
 * with the ability to load any past upload into the visualizer.
 */

import { useEffect, useState } from "react";
import { fetchUploads, fetchGraph } from "../api/client";
import type { UniversalGraph } from "../types/graph";

interface UploadItem {
  job_id: string;
  filename: string;
  uploaded_at: string;
}

interface HistoryPageProps {
  currentGraph: UniversalGraph | null;
  onLoadGraph: (g: UniversalGraph) => void;
}

export default function HistoryPage({ currentGraph, onLoadGraph }: HistoryPageProps) {
  const [uploads, setUploads] = useState<UploadItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingJob, setLoadingJob] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");

  useEffect(() => {
    fetchUploads()
      .then(setUploads)
      .catch(() => setUploads([]))
      .finally(() => setLoading(false));
  }, []);

  const handleLoad = async (jobId: string) => {
    setLoadingJob(jobId);
    try {
      const g = await fetchGraph(jobId);
      onLoadGraph(g);
    } catch {
      alert("Failed to load graph for job " + jobId);
    } finally {
      setLoadingJob(null);
    }
  };

  const filtered = uploads.filter((u) =>
    (u.filename || "").toLowerCase().includes(searchQuery.toLowerCase())
  );

  const formatRelativeTime = (dateStr: string) => {
    const diff = Date.now() - new Date(dateStr).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return "Just now";
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    const days = Math.floor(hrs / 24);
    if (days < 7) return `${days}d ago`;
    return new Date(dateStr).toLocaleDateString();
  };

  return (
    <div className="h-full overflow-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <h2 className="text-white font-bold text-xl flex items-center gap-2">
            <span className="text-lg">🕘</span> Upload History
          </h2>
          <p className="text-gray-400 text-sm mt-1">
            {uploads.length} model{uploads.length !== 1 ? "s" : ""} uploaded
          </p>
        </div>

        {/* Search */}
        <div className="relative">
          <input
            type="text"
            placeholder="Search uploads…"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="bg-[#0a0c12] border border-white/10 rounded-xl pl-10 pr-4 py-2.5 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-accent/50 transition w-64"
          />
          <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>
      </div>

      {/* Loading state */}
      {loading && (
        <div className="flex items-center justify-center py-20">
          <div className="flex flex-col items-center gap-3">
            <div className="w-8 h-8 border-2 border-accent/30 border-t-accent rounded-full animate-spin" />
            <p className="text-gray-400 text-sm">Loading upload history…</p>
          </div>
        </div>
      )}

      {/* Empty state */}
      {!loading && uploads.length === 0 && (
        <div className="flex flex-col items-center justify-center py-20 gap-4">
          <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-accent/20 to-purple-500/10 flex items-center justify-center text-4xl border border-white/5">
            📂
          </div>
          <h3 className="text-white text-lg font-semibold">No uploads yet</h3>
          <p className="text-gray-400 text-sm text-center max-w-sm">
            Upload your first PyTorch model to see it appear here. You can load any past upload to analyze it again.
          </p>
        </div>
      )}

      {/* No search results */}
      {!loading && uploads.length > 0 && filtered.length === 0 && (
        <div className="text-center py-16">
          <p className="text-gray-400 text-sm">No uploads match "{searchQuery}"</p>
        </div>
      )}

      {/* Upload list */}
      {!loading && filtered.length > 0 && (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {filtered.map((u) => {
            const isActive = currentGraph?.job_id === u.job_id;
            const isLoading = loadingJob === u.job_id;

            return (
              <div
                key={u.job_id}
                className={`group relative rounded-xl border p-4 transition-all duration-200 ${
                  isActive
                    ? "bg-gradient-to-br from-accent/10 to-purple-500/5 border-accent/40 shadow-lg shadow-accent/5"
                    : "bg-panel border-white/5 hover:border-white/15 hover:bg-white/[0.02]"
                }`}
              >
                {/* Active indicator */}
                {isActive && (
                  <div className="absolute top-3 right-3">
                    <span className="flex h-2.5 w-2.5">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent opacity-75" />
                      <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-accent" />
                    </span>
                  </div>
                )}

                {/* File icon + info */}
                <div className="flex items-start gap-3 mb-3">
                  <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500/20 to-indigo-500/10 flex items-center justify-center text-lg border border-white/5 shrink-0">
                    🧠
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="text-sm text-white font-medium truncate">
                      {u.filename || "(no filename)"}
                    </div>
                    <div className="text-xs text-gray-500 mt-0.5">
                      {new Date(u.uploaded_at).toLocaleString()}
                    </div>
                  </div>
                </div>

                {/* Footer */}
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-500">
                    {formatRelativeTime(u.uploaded_at)}
                  </span>
                  <button
                    onClick={() => handleLoad(u.job_id)}
                    disabled={isLoading}
                    className={`text-xs px-4 py-1.5 rounded-full font-medium transition-all duration-200 ${
                      isActive
                        ? "bg-accent text-white cursor-default"
                        : isLoading
                        ? "bg-accent/30 text-accent cursor-wait"
                        : "bg-white/5 text-gray-300 hover:bg-accent hover:text-white"
                    }`}
                  >
                    {isActive ? "Active" : isLoading ? "Loading…" : "Load"}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
