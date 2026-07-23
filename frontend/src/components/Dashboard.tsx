import { useEffect, useState } from "react";
import { fetchSource, fetchUploads, fetchGraph } from "../api/client";
import type { SourceResponse, UniversalGraph, GraphNode } from "../types/graph";

interface UploadItem {
  job_id: string;
  filename: string;
  uploaded_at: string;
}

type TabKey = "overview" | "architecture" | "layers" | "code";

const TABS: { key: TabKey; label: string; icon: string }[] = [
  { key: "overview", label: "Overview", icon: "📊" },
  { key: "architecture", label: "Architecture Type", icon: "🏗️" },
  { key: "layers", label: "Layer Type Breakdown", icon: "📋" },
  { key: "code", label: "Code Preview", icon: "💻" },
];

/* ── helpers ─────────────────────────────────────────── */

const formatShape = (shape: number[] | null) => (shape ? shape.join(" × ") : "—");

const deriveArchitectureType = (g: UniversalGraph) => {
  const labels = g.groups.map((gr) => gr.label.toLowerCase()).join(" ");
  const hasResidual = labels.includes("residual") || g.nodes.some((n) => /add|skip|shortcut/i.test(n.label));
  const hasConv = g.nodes.some((n) => /conv/i.test(n.type));
  if (hasResidual) return "ResNet (Residual Network)";
  if (hasConv) return "CNN (Convolutional Neural Network)";
  return "MLP / Feedforward";
};

const deriveTaskType = (g: UniversalGraph) => {
  if (g.nodes.some((n) => /conv|pool|batchnorm/i.test(n.type))) return "Image Classification";
  if (g.nodes.some((n) => /lstm|gru|rnn/i.test(n.type))) return "Sequence Modeling";
  return "General Purpose";
};

const deriveCharacteristics = (g: UniversalGraph) => {
  const chars: string[] = [];
  const hasSkip = g.edges.some((e) => e.is_skip_connection) || g.nodes.some((n) => /add|skip/i.test(n.label));
  if (hasSkip) chars.push("Residual Connections (out + identity)");
  if (g.nodes.some((n) => /conv/i.test(n.type))) chars.push("Convolutional Feature Learning");
  if (g.nodes.some((n) => /batchnorm|norm/i.test(n.type))) chars.push("Batch Normalization");
  if (g.nodes.some((n) => /pool/i.test(n.type))) chars.push("Global Average Pooling");
  if (g.nodes.some((n) => /linear|fc|dense/i.test(n.type))) chars.push("Fully Connected Classifier");
  if (g.nodes.some((n) => /relu|gelu|sigmoid/i.test(n.type))) chars.push("Non-linear Activations");
  return chars;
};

type LayerCategory = "Conv2d" | "BatchNorm2d" | "ReLU" | "Pooling" | "Linear" | "Other";

const categorize = (type: string): LayerCategory => {
  const t = type.toLowerCase();
  if (/conv/i.test(t)) return "Conv2d";
  if (/batchnorm|norm/i.test(t)) return "BatchNorm2d";
  if (/relu|gelu|sigmoid|tanh|activation/i.test(t)) return "ReLU";
  if (/pool/i.test(t)) return "Pooling";
  if (/linear|fc|dense/i.test(t)) return "Linear";
  return "Other";
};

const CATEGORY_COLORS: Record<LayerCategory, string> = {
  Conv2d: "#4285f4",
  BatchNorm2d: "#fbbc04",
  ReLU: "#34a853",
  Pooling: "#a855f7",
  Linear: "#ea4335",
  Other: "#94a3b8",
};

/* ── sub-components ──────────────────────────────────── */

function StatRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex justify-between py-2 border-b border-white/5 last:border-0">
      <span className="text-gray-400 text-sm">{label}</span>
      <span className="text-white text-sm font-medium">{String(value)}</span>
    </div>
  );
}

function SectionHeader({ icon, title }: { icon: string; title: string }) {
  return (
    <div className="flex items-center gap-2 mb-4">
      <span className="text-lg">{icon}</span>
      <h3 className="text-white font-semibold uppercase tracking-wider text-sm">{title}</h3>
    </div>
  );
}

function DonutChart({ data }: { data: { label: string; value: number; color: string }[] }) {
  const total = data.reduce((s, d) => s + d.value, 0);
  if (total === 0) return null;
  let cumulative = 0;
  const size = 160;
  const cx = size / 2, cy = size / 2, r = 55, strokeW = 20;

  return (
    <div className="flex items-center gap-6">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {data.filter((d) => d.value > 0).map((d) => {
          const pct = d.value / total;
          const circumference = 2 * Math.PI * r;
          const dashLen = pct * circumference;
          const dashOff = -cumulative * circumference;
          cumulative += pct;
          return (
            <circle key={d.label} cx={cx} cy={cy} r={r} fill="none" stroke={d.color}
              strokeWidth={strokeW} strokeDasharray={`${dashLen} ${circumference - dashLen}`}
              strokeDashoffset={dashOff} transform={`rotate(-90 ${cx} ${cy})`}
              style={{ transition: "stroke-dasharray 0.6s ease" }} />
          );
        })}
        <text x={cx} y={cy - 6} textAnchor="middle" className="fill-white text-lg font-bold">{total}</text>
        <text x={cx} y={cy + 12} textAnchor="middle" className="fill-gray-400 text-[10px]">layers</text>
      </svg>
      <div className="space-y-1.5">
        {data.filter((d) => d.value > 0).map((d) => (
          <div key={d.label} className="flex items-center gap-2 text-sm">
            <span className="w-3 h-3 rounded-sm shrink-0" style={{ background: d.color }} />
            <span className="text-gray-300">{d.label}: {d.value} ({Math.round((d.value / total) * 100)}%)</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── main component ──────────────────────────────────── */

export default function Dashboard({
  graph: externalGraph,
  onLoadGraph,
}: {
  graph: UniversalGraph | null;
  onLoadGraph: (g: UniversalGraph) => void;
}) {
  const [graph, setGraph] = useState<UniversalGraph | null>(externalGraph);
  const [uploads, setUploads] = useState<UploadItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [source, setSource] = useState<SourceResponse | null>(null);
  const [sourceLoading, setSourceLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<TabKey>("overview");

  useEffect(() => { setGraph(externalGraph); }, [externalGraph]);

  useEffect(() => {
    setLoading(true);
    fetchUploads().then(setUploads).catch(() => setUploads([])).finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!graph) { setSource(null); return; }
    setSourceLoading(true);
    fetchSource(graph.job_id).then(setSource).catch(() => setSource(null)).finally(() => setSourceLoading(false));
  }, [graph]);

  const handleLoad = async (jobId: string) => {
    try {
      const g = await fetchGraph(jobId);
      setGraph(g);
      onLoadGraph(g);
    } catch { alert("Failed to load graph for job " + jobId); }
  };

  /* ── derived values ── */
  const archType = graph ? deriveArchitectureType(graph) : "—";
  const taskType = graph ? deriveTaskType(graph) : "—";
  const characteristics = graph ? deriveCharacteristics(graph) : [];
  const inputShape = graph ? formatShape(graph.nodes.find((n) => n.input_shape)?.input_shape ?? null) : "—";
  const outputClasses = graph
    ? graph.nodes.slice().reverse().find((n) => n.output_shape)?.output_shape?.[0] ?? "—"
    : "—";
  const sourceFilename = graph
    ? uploads.find((u) => u.job_id === graph.job_id)?.filename ?? graph.model_name
    : "—";

  const layerCounts = graph
    ? (Object.keys(CATEGORY_COLORS) as LayerCategory[]).map((cat) => ({
        label: cat,
        value: graph.nodes.filter((n) => categorize(n.type) === cat).length,
        color: CATEGORY_COLORS[cat],
      }))
    : [];

  const archFlow = graph
    ? (() => {
        const seen = new Set<string>();
        return graph.nodes.reduce<string[]>((acc, n) => {
          const cat = categorize(n.type);
          const key = cat === "Other" ? n.type : cat;
          if (!seen.has(key)) { seen.add(key); acc.push(n.type); }
          return acc;
        }, []);
      })()
    : [];

  const blockSummary = graph
    ? graph.groups.map((g) => {
        const members = graph.nodes.filter((n) => g.member_node_ids.includes(n.id));
        const outCh = members.slice().reverse().find((n) => n.output_shape)?.output_shape?.[1] ?? "—";
        return { block: g.label, type: g.type.replace(/_/g, " "), repetitions: g.repeat_count, outputChannels: outCh };
      })
    : [];

  /* ── no model state ── */
  if (!graph) {
    return (
      <div className="h-full flex flex-col items-center justify-center gap-6 p-8">
        <div className="bg-panel rounded-2xl p-10 text-center max-w-lg border border-white/5">
          <div className="text-5xl mb-4">🧠</div>
          <h3 className="text-white text-xl font-semibold mb-2">No model loaded yet</h3>
          <p className="text-gray-400 mb-6">Upload a model or select a recent project to see the analysis report.</p>
          {loading && <p className="text-gray-500 text-sm">Loading recent uploads…</p>}
          {!loading && uploads.length > 0 && (
            <div className="space-y-2 text-left">
              <p className="text-xs uppercase tracking-widest text-gray-500 mb-2">Recent Uploads</p>
              {uploads.slice(0, 5).map((u) => (
                <button key={u.job_id} onClick={() => handleLoad(u.job_id)}
                  className="w-full flex items-center justify-between p-3 rounded-xl bg-[#0f1320] border border-white/5 hover:border-accent/50 transition group">
                  <div>
                    <div className="text-sm text-white">{u.filename || "(no filename)"}</div>
                    <div className="text-xs text-gray-500">{new Date(u.uploaded_at).toLocaleString()}</div>
                  </div>
                  <span className="text-xs px-3 py-1 rounded-full bg-accent/20 text-accent group-hover:bg-accent group-hover:text-white transition">Load</span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  }

  /* ── tab content ── */
  const renderTab = () => {
    switch (activeTab) {
      case "overview":
        return (
          <div className="grid gap-5 lg:grid-cols-2">
            {/* Model Info */}
            <div className="bg-panel rounded-xl p-5 border border-white/5">
              <SectionHeader icon="📊" title="Overview" />
              <StatRow label="Framework" value={`🔥 ${graph.meta.framework.charAt(0).toUpperCase() + graph.meta.framework.slice(1)}`} />
              <StatRow label="Detected Model" value={graph.model_name} />
              <StatRow label="Model Type" value={archType} />
              <StatRow label="Task Type" value={taskType} />
              <StatRow label="Output Classes" value={outputClasses} />
              <StatRow label="Input Shape" value={inputShape} />
              <StatRow label="File Name" value={sourceFilename} />
              <StatRow label="Total Parameters" value={graph.meta.total_params.toLocaleString()} />
              <StatRow label="Confidence" value={graph.meta.confidence} />
              <StatRow label="FLOPs" value={graph.meta.flops?.toLocaleString() ?? "— (Phase 5)"} />
            </div>
            {/* Quick Stats Cards */}
            <div className="space-y-5">
              <div className="grid grid-cols-2 gap-4">
                {[
                  { label: "Total Layers", value: graph.meta.total_layers, color: "from-blue-500/20 to-blue-600/5" },
                  { label: "Parameters", value: graph.meta.total_params.toLocaleString(), color: "from-purple-500/20 to-purple-600/5" },
                  { label: "Groups", value: graph.groups.length, color: "from-emerald-500/20 to-emerald-600/5" },
                  { label: "Connections", value: graph.edges.length, color: "from-amber-500/20 to-amber-600/5" },
                ].map((s) => (
                  <div key={s.label} className={`bg-gradient-to-br ${s.color} rounded-xl p-4 border border-white/5`}>
                    <div className="text-xs uppercase tracking-widest text-gray-400 mb-2">{s.label}</div>
                    <div className="text-2xl text-white font-bold">{s.value}</div>
                  </div>
                ))}
              </div>
              {/* Additional Info */}
              <div className="bg-panel rounded-xl p-5 border border-white/5">
                <SectionHeader icon="ℹ️" title="Additional Info" />
                <StatRow label="Skip Connections" value={graph.edges.some((e) => e.is_skip_connection) ? "Yes (Identity)" : "No"} />
                <StatRow label="Normalization" value={graph.nodes.some((n) => /norm/i.test(n.type)) ? "BatchNorm2d" : "None"} />
                <StatRow label="Activation" value={graph.nodes.find((n) => /relu|gelu|sigmoid/i.test(n.type))?.type ?? "None"} />
                <StatRow label="Classifier" value={graph.nodes.some((n) => /linear|fc/i.test(n.type)) ? "Linear (Fully Connected)" : "—"} />
                <StatRow label="Total Blocks" value={graph.groups.length} />
              </div>
            </div>
            {/* Recent Uploads */}
            <div className="bg-panel rounded-xl p-5 border border-white/5 lg:col-span-2">
              <SectionHeader icon="📁" title="Recent Uploads" />
              {loading && <p className="text-gray-400 text-sm">Loading…</p>}
              {!loading && uploads.length === 0 && <p className="text-gray-500 text-sm">No recent uploads found.</p>}
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {uploads.slice(0, 6).map((u) => (
                  <div key={u.job_id} className={`rounded-xl border p-3 bg-[#0f1320] flex items-center justify-between gap-3 ${u.job_id === graph.job_id ? "border-accent/60" : "border-white/5"}`}>
                    <div className="min-w-0">
                      <div className="text-sm text-white truncate">{u.filename || "(no filename)"}</div>
                      <div className="text-xs text-gray-500">{new Date(u.uploaded_at).toLocaleString()}</div>
                    </div>
                    <button onClick={() => handleLoad(u.job_id)}
                      className={`shrink-0 text-xs px-3 py-1 rounded-full transition ${u.job_id === graph.job_id ? "bg-accent text-white" : "bg-white/5 text-gray-300 hover:bg-accent/30"}`}>
                      {u.job_id === graph.job_id ? "Active" : "Load"}
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </div>
        );

      case "architecture":
        return (
          <div className="grid gap-5 lg:grid-cols-[1fr_1fr]">
            {/* Architecture Type */}
            <div className="bg-panel rounded-xl p-5 border border-white/5">
              <SectionHeader icon="🏗️" title="Architecture Type" />
              <div className="bg-gradient-to-r from-emerald-500/10 to-teal-500/5 rounded-xl p-4 mb-5 border border-emerald-500/20">
                <h4 className="text-emerald-300 text-lg font-bold">{archType}</h4>
                <p className="text-gray-400 text-sm mt-1">
                  {archType.includes("ResNet") && "This is a ResNet-style architecture using BasicBlock with identity skip connections."}
                  {archType.includes("CNN") && "A convolutional neural network with sequential feature extraction layers."}
                  {archType.includes("MLP") && "A multi-layer perceptron with fully connected layers."}
                </p>
              </div>
              <h4 className="text-white font-semibold text-sm mb-3">Key Characteristics</h4>
              <div className="space-y-2">
                {characteristics.map((c) => (
                  <div key={c} className="flex items-center gap-2">
                    <span className="text-emerald-400 text-sm">✓</span>
                    <span className="text-gray-300 text-sm">{c}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Residual Connection + Block Summary */}
            <div className="space-y-5">
              {graph.edges.some((e) => e.is_skip_connection) && (
                <div className="bg-panel rounded-xl p-5 border border-white/5">
                  <SectionHeader icon="🔗" title="Residual Connection Detected" />
                  <p className="text-gray-400 text-sm mb-3">Pattern Detected: out = out + identity<br />(This is the core of ResNet)</p>
                  <div className="bg-[#090b11] rounded-lg p-4 font-mono text-sm text-gray-300 whitespace-pre">
{`identity = x
...
out = self.bn2(out)
out = out + identity  # residual add
out = self.relu2(out)`}
                  </div>
                </div>
              )}
              <div className="bg-panel rounded-xl p-5 border border-white/5">
                <SectionHeader icon="🧱" title="Block Summary" />
                {blockSummary.length > 0 ? (
                  <div className="overflow-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="text-gray-500 text-xs uppercase border-b border-white/5">
                          <th className="text-left py-2 pr-3">Block</th>
                          <th className="text-left py-2 pr-3">Type</th>
                          <th className="text-center py-2 pr-3">Reps</th>
                          <th className="text-right py-2">Out Channels</th>
                        </tr>
                      </thead>
                      <tbody>
                        {blockSummary.map((b, i) => (
                          <tr key={i} className="border-b border-white/5 last:border-0">
                            <td className="py-2 pr-3 text-white">{b.block}</td>
                            <td className="py-2 pr-3 text-gray-300">{b.type}</td>
                            <td className="py-2 pr-3 text-center text-gray-300">{b.repetitions}</td>
                            <td className="py-2 text-right text-gray-300">{b.outputChannels}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p className="text-gray-500 text-sm">No grouped blocks detected.</p>
                )}
              </div>
            </div>

            {/* Architecture Flow */}
            <div className="bg-panel rounded-xl p-5 border border-white/5 lg:col-span-2">
              <SectionHeader icon="➡️" title="Architecture Flow (Summary)" />
              <div className="flex flex-wrap items-center gap-2">
                <span className="bg-blue-500/15 text-blue-300 px-3 py-1.5 rounded-lg text-sm font-medium">Input</span>
                {archFlow.map((step, i) => (
                  <span key={i} className="flex items-center gap-2">
                    <span className="text-gray-600">→</span>
                    <span className="px-3 py-1.5 rounded-lg text-sm font-medium" style={{
                      background: `${CATEGORY_COLORS[categorize(step)]}20`,
                      color: CATEGORY_COLORS[categorize(step)],
                    }}>{step}</span>
                  </span>
                ))}
                <span className="text-gray-600">→</span>
                <span className="bg-emerald-500/15 text-emerald-300 px-3 py-1.5 rounded-lg text-sm font-medium">Output</span>
              </div>
            </div>
          </div>
        );

      case "layers":
        return (
          <div className="grid gap-5 lg:grid-cols-[1fr_340px]">
            {/* Layers Table */}
            <div className="bg-panel rounded-xl p-5 border border-white/5">
              <SectionHeader icon="📋" title="Layers Detected (In Order)" />
              <div className="overflow-auto max-h-[500px]">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-panel">
                    <tr className="text-gray-500 text-xs uppercase border-b border-white/10">
                      <th className="text-left py-2 pr-3 w-8">#</th>
                      <th className="text-left py-2 pr-3">Layer</th>
                      <th className="text-left py-2 pr-3">Type</th>
                      <th className="text-right py-2 pr-3">Output Shape</th>
                      <th className="text-right py-2">Params</th>
                    </tr>
                  </thead>
                  <tbody>
                    {graph.nodes.map((n, i) => (
                      <tr key={n.id} className="border-b border-white/5 last:border-0 hover:bg-white/[0.02] transition">
                        <td className="py-2 pr-3 text-gray-500">{i + 1}</td>
                        <td className="py-2 pr-3 text-white font-medium">{n.label}</td>
                        <td className="py-2 pr-3">
                          <span className="px-2 py-0.5 rounded text-xs font-medium" style={{
                            background: `${CATEGORY_COLORS[categorize(n.type)]}20`,
                            color: CATEGORY_COLORS[categorize(n.type)],
                          }}>{n.type}</span>
                        </td>
                        <td className="py-2 pr-3 text-right text-gray-300 font-mono text-xs">{formatShape(n.output_shape)}</td>
                        <td className="py-2 text-right text-gray-300">{n.params.toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Donut + Legend */}
            <div className="space-y-5">
              <div className="bg-panel rounded-xl p-5 border border-white/5">
                <SectionHeader icon="🍩" title="Layer Type Breakdown" />
                <DonutChart data={layerCounts} />
              </div>
              {/* Legend */}
              <div className="bg-panel rounded-xl p-5 border border-white/5">
                <SectionHeader icon="🎨" title="Legend" />
                <div className="grid grid-cols-2 gap-2">
                  {Object.entries(CATEGORY_COLORS).map(([label, color]) => (
                    <div key={label} className="flex items-center gap-2">
                      <span className="w-3 h-3 rounded-sm shrink-0" style={{ background: color }} />
                      <span className="text-gray-300 text-sm">{label}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        );

      case "code":
        return (
          <div className="bg-panel rounded-xl p-5 border border-white/5">
            <SectionHeader icon="💻" title="Source Code Preview" />
            <div className="bg-[#090b11] rounded-xl p-5 min-h-[400px] overflow-auto text-sm text-gray-200 font-mono whitespace-pre-wrap">
              {sourceLoading && <div className="text-gray-400">Loading source preview…</div>}
              {!sourceLoading && source && <code>{source.code}</code>}
              {!sourceLoading && !source && <div className="text-gray-500">Source preview is not available for this model.</div>}
            </div>
          </div>
        );
    }
  };

  return (
    <div className="h-full overflow-auto p-6 space-y-5">
      {/* Page Title */}
      <div>
        <h2 className="text-white font-bold text-xl">Neural Network Analysis Report</h2>
        <p className="text-gray-400 text-sm mt-1">
          {graph.model_name} • {graph.meta.framework} • {graph.meta.total_params.toLocaleString()} parameters
        </p>
      </div>

      {/* Tab Bar */}
      <div className="flex gap-1 bg-[#0a0c12] rounded-xl p-1 border border-white/5 w-fit">
        {TABS.map((tab) => (
          <button key={tab.key} onClick={() => setActiveTab(tab.key)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              activeTab === tab.key
                ? "bg-accent text-white shadow-lg shadow-accent/20"
                : "text-gray-400 hover:text-white hover:bg-white/5"
            }`}>
            <span>{tab.icon}</span>
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {renderTab()}
    </div>
  );
}
