/**
 * components/Sidebar.tsx
 *
 * Why this file exists:
 *   Matches the left navigation from the reference UI screenshot
 *   (logo, Upload Project button, nav items, user profile footer).
 *   In Phase 1, only "Visualizer" is functional - the rest are visual
 *   placeholders until Phase 6 (auth/persistence) wires them to real data.
 *
 * How it connects:
 *   Rendered by App.tsx. onUploadClick is passed down from App to open
 *   the upload flow regardless of which "page" is active.
 */

interface SidebarProps {
  onUploadClick: () => void;
  currentPage: string;
  onNavigate: (page: string) => void;
  userEmail?: string | null;
  onLogout?: () => void;
}

const NAV_ITEMS = [
  { key: "dashboard", label: "Dashboard", enabled: true },
  { key: "visualizer", label: "Visualizer", enabled: true },
  { label: "Projects", enabled: false },
  { label: "Saved Graphs", enabled: false },
  { label: "History", enabled: false },
  { label: "Examples", enabled: false },
  { key: "settings", label: "Settings", enabled: true },
  { key: "help", label: "Help", enabled: true },
];

export default function Sidebar({ onUploadClick, currentPage, onNavigate, userEmail, onLogout }: SidebarProps) {
  return (
    <aside className="w-56 shrink-0 bg-panel border-r border-white/5 flex flex-col h-screen text-gray-300">
      <div className="px-5 py-6 flex items-center gap-2 text-white font-semibold text-lg">
        <span className="w-3 h-3 rounded-full bg-accent inline-block" />
        NeuralNetworks
      </div>

      <div className="px-4 mb-4">
        <button
          onClick={onUploadClick}
          className="w-full bg-accent hover:bg-accent/90 text-white text-sm font-medium py-2.5 rounded-lg transition"
        >
          + Upload Project
        </button>
      </div>

      <nav className="flex-1 px-2 space-y-1">
        {NAV_ITEMS.map((item: any) => (
          <div
            key={item.label || item.key}
            onClick={() => item.enabled && onNavigate(item.key || item.label)}
            className={`px-3 py-2 rounded-lg text-sm cursor-pointer ${
              currentPage === (item.key || item.label)
                ? "bg-white/5 text-white"
                : item.enabled
                ? "hover:bg-white/5"
                : "opacity-40 cursor-not-allowed"
            }`}
            title={item.enabled ? "" : "Available in a later phase"}
          >
            {item.label}
          </div>
        ))}
      </nav>

      <div className="px-4 py-4 border-t border-white/5 flex items-center gap-3">
        <div className="w-8 h-8 rounded-full bg-accent/30 flex items-center justify-center text-xs font-semibold text-white uppercase">
          {userEmail ? userEmail.charAt(0) : "U"}
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-white text-xs truncate">{userEmail ?? "Guest User"}</div>
          {onLogout && (
            <button onClick={onLogout} className="text-gray-500 hover:text-red-400 text-xs transition">
              Sign out
            </button>
          )}
        </div>
      </div>
    </aside>
  );
}

