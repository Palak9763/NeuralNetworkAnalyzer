import { useState } from "react";

function SectionHeader({ icon, title }: { icon: string; title: string }) {
  return (
    <div className="flex items-center gap-2 mb-4">
      <span className="text-lg">{icon}</span>
      <h3 className="text-white font-semibold text-sm uppercase tracking-wider">{title}</h3>
    </div>
  );
}

function Toggle({ label, description, checked, onChange }: { label: string; description: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <div className="flex items-center justify-between py-3 border-b border-white/5 last:border-0">
      <div>
        <div className="text-white text-sm font-medium">{label}</div>
        <div className="text-gray-500 text-xs mt-0.5">{description}</div>
      </div>
      <button onClick={() => onChange(!checked)}
        className={`relative w-11 h-6 rounded-full transition-colors ${checked ? "bg-accent" : "bg-white/10"}`}>
        <span className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow transition-transform ${checked ? "translate-x-5" : ""}`} />
      </button>
    </div>
  );
}

export default function SettingsPage() {
  const [darkMode, setDarkMode] = useState(true);
  const [autoLayout, setAutoLayout] = useState(true);
  const [showParams, setShowParams] = useState(true);
  const [showShapes, setShowShapes] = useState(true);
  const [animations, setAnimations] = useState(true);
  const [skipConnections, setSkipConnections] = useState(true);
  const [apiBase, setApiBase] = useState(import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000");

  return (
    <div className="h-full overflow-auto p-6 space-y-6">
      <div>
        <h2 className="text-white font-bold text-xl">Settings</h2>
        <p className="text-gray-400 text-sm mt-1">Configure the Neural Network Analyzer to your preferences.</p>
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        {/* Appearance */}
        <div className="bg-panel rounded-xl p-5 border border-white/5">
          <SectionHeader icon="🎨" title="Appearance" />
          <Toggle label="Dark Mode" description="Use dark theme throughout the app" checked={darkMode} onChange={setDarkMode} />
          <Toggle label="Animations" description="Enable smooth transitions and micro-animations" checked={animations} onChange={setAnimations} />
        </div>

        {/* Visualizer */}
        <div className="bg-panel rounded-xl p-5 border border-white/5">
          <SectionHeader icon="🔬" title="Visualizer" />
          <Toggle label="Auto Layout" description="Automatically arrange nodes when loading a model" checked={autoLayout} onChange={setAutoLayout} />
          <Toggle label="Show Parameters" description="Display parameter counts on graph nodes" checked={showParams} onChange={setShowParams} />
          <Toggle label="Show Shapes" description="Display tensor shapes on graph nodes" checked={showShapes} onChange={setShowShapes} />
          <Toggle label="Skip Connections" description="Highlight residual / skip connections" checked={skipConnections} onChange={setSkipConnections} />
        </div>

        {/* API Configuration */}
        <div className="bg-panel rounded-xl p-5 border border-white/5">
          <SectionHeader icon="🔗" title="API Configuration" />
          <div className="space-y-3">
            <div>
              <label className="text-sm text-gray-400 block mb-1">Backend URL</label>
              <input type="text" value={apiBase} onChange={(e) => setApiBase(e.target.value)}
                className="w-full bg-[#0a0c12] border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-accent/50 transition" />
            </div>
            <div className="flex items-center gap-2 text-sm">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-emerald-400">Connected</span>
              <span className="text-gray-500">• Port 8000</span>
            </div>
          </div>
        </div>

        {/* Export Defaults */}
        <div className="bg-panel rounded-xl p-5 border border-white/5">
          <SectionHeader icon="📤" title="Export Defaults" />
          <div className="space-y-3">
            <div>
              <label className="text-sm text-gray-400 block mb-1">Default Export Format</label>
              <select className="w-full bg-[#0a0c12] border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-accent/50 transition appearance-none">
                <option value="png">PNG Image</option>
                <option value="svg">SVG Vector</option>
                <option value="json">JSON Data</option>
                <option value="pdf">PDF Report</option>
              </select>
            </div>
            <div>
              <label className="text-sm text-gray-400 block mb-1">Image Resolution</label>
              <select className="w-full bg-[#0a0c12] border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-accent/50 transition appearance-none">
                <option value="1x">1x (Standard)</option>
                <option value="2x">2x (Retina)</option>
                <option value="4x">4x (Print Quality)</option>
              </select>
            </div>
          </div>
        </div>

        {/* About */}
        <div className="bg-panel rounded-xl p-5 border border-white/5 lg:col-span-2">
          <SectionHeader icon="ℹ️" title="About" />
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            {[
              { label: "Version", value: "1.0.0" },
              { label: "Framework", value: "React + Vite" },
              { label: "Backend", value: "FastAPI" },
              { label: "Parser", value: "PyTorch Tracer" },
            ].map((item) => (
              <div key={item.label} className="bg-[#0a0c12] rounded-xl p-3 border border-white/5">
                <div className="text-xs text-gray-500 uppercase tracking-widest">{item.label}</div>
                <div className="text-white font-medium text-sm mt-1">{item.value}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
