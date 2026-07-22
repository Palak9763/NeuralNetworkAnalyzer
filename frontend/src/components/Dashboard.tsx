import { useEffect, useState } from "react";
import { fetchUploads, fetchGraph } from "../api/client";
import type { UniversalGraph } from "../types/graph";

interface UploadItem {
  job_id: string;
  filename: string;
  uploaded_at: string;
}

export default function Dashboard({ onLoadGraph }: { onLoadGraph: (g: UniversalGraph) => void }) {
  const [uploads, setUploads] = useState<UploadItem[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    fetchUploads()
      .then((list) => setUploads(list))
      .catch(() => setUploads([]))
      .finally(() => setLoading(false));
  }, []);

  const handleLoad = async (jobId: string) => {
    try {
      const g = await fetchGraph(jobId);
      onLoadGraph(g);
    } catch (e) {
      alert("Failed to load graph for job " + jobId);
    }
  };

  const handleShare = (jobId: string) => {
    const base = window.location.origin + window.location.pathname;
    const shareUrl = `${base}?job=${encodeURIComponent(jobId)}`;
    navigator.clipboard?.writeText(shareUrl).then(() => alert("Share link copied"));
  };

  return (
    <div className="p-6 overflow-auto">
      <h2 className="text-white font-semibold text-lg mb-4">Dashboard</h2>
      {loading && <div className="text-gray-400">Loading uploads…</div>}

      {!loading && uploads.length === 0 && (
        <div className="text-gray-500">No recent uploads found.</div>
      )}

      <div className="grid gap-3">
        {uploads.map((u) => (
          <div key={u.job_id} className="bg-panel p-3 rounded-lg border border-white/5 flex items-center justify-between">
            <div>
              <div className="text-sm text-white">{u.filename || "(no filename)"}</div>
              <div className="text-xs text-gray-400">Job: {u.job_id} • {new Date(u.uploaded_at).toLocaleString()}</div>
            </div>
            <div className="flex gap-2">
              <button onClick={() => handleLoad(u.job_id)} className="text-sm px-3 py-1.5 rounded bg-accent text-white">Load</button>
              <button onClick={() => handleShare(u.job_id)} className="text-sm px-3 py-1.5 rounded border border-white/10 text-gray-300">Share</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
