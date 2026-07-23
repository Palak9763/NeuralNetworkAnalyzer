/**
 * App.tsx
 *
 * Why this file exists:
 *   Top-level layout component. Owns the app's core state (currently
 *   loaded graph, currently selected node, upload modal visibility) and
 *   arranges the reference UI's structure: Sidebar | (TopBar + Canvas +
 *   bottom cards) | LayerPropertiesPanel.
 *
 * How it connects:
 *   Composes every component in components/*. Passes the loaded
 *   UniversalGraph down to GraphCanvas, ModelSummary, and LayerTable.
 */

import { useState, useEffect } from "react";
import Sidebar from "./components/Sidebar";
import Dashboard from "./components/Dashboard";
import SettingsPage from "./components/SettingsPage";
import HelpPage from "./components/HelpPage";
import TopBar from "./components/TopBar";
import GraphCanvas from "./components/GraphCanvas";
import ThemeToggle from "./components/ThemeToggle";
import AuthPage from "./components/AuthPage";

import ModelSummary from "./components/ModelSummary";
import LayerTable from "./components/LayerTable";
import UploadModal from "./components/UploadModal";
import { fetchGraph } from "./api/client";
import type { GraphNode, UniversalGraph } from "./types/graph";

export default function App() {
  const [authToken, setAuthToken] = useState<string | null>(() => localStorage.getItem("nna-token"));
  const [userEmail, setUserEmail] = useState<string | null>(() => localStorage.getItem("nna-email"));
  const [graph, setGraph] = useState<UniversalGraph | null>(null);
  const [page, setPage] = useState<string>("visualizer");
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [showUpload, setShowUpload] = useState(false);

  function handleLoginSuccess(token: string, email: string) {
    localStorage.setItem("nna-token", token);
    localStorage.setItem("nna-email", email);
    setAuthToken(token);
    setUserEmail(email);
  }

  function handleLogout() {
    localStorage.removeItem("nna-token");
    localStorage.removeItem("nna-email");
    setAuthToken(null);
    setUserEmail(null);
    setGraph(null);
  }

  useEffect(() => {
    if (!authToken) return;
    // If a shared link includes ?job=<job_id>, fetch that graph on load
    const params = new URLSearchParams(window.location.search);
    const job = params.get("job");
    if (job) {
      (async () => {
        try {
          const g = await fetchGraph(job);
          setGraph(g);
        } catch (e) {
          // ignore - user can upload instead
        }
      })();
    }
  }, [authToken]);

  // Show auth page if not logged in
  if (!authToken) {
    return (
      <>
        <AuthPage onLoginSuccess={handleLoginSuccess} />
        <ThemeToggle />
      </>
    );
  }

  return (
    <div className="flex h-screen w-screen overflow-hidden" style={{ background: 'var(--color-bg)' }}>
      <Sidebar onUploadClick={() => setShowUpload(true)} currentPage={page} onNavigate={setPage} userEmail={userEmail} onLogout={handleLogout} />

      <div className="flex-1 flex flex-col min-w-0">
        <TopBar graph={graph} />

        <div className="flex-1 flex min-h-0">
          <main className="flex-1 flex flex-col min-w-0">
            <div className="flex-1 min-h-0">
              {page === "dashboard" ? (
                <Dashboard
                  graph={graph}
                  onLoadGraph={(g: UniversalGraph) => {
                    setGraph(g);
                    setSelectedNode(null);
                    setPage("visualizer");
                  }}
                />
              ) : page === "visualizer" ? (
                graph ? (
                  <div id="reactflow-wrapper" className="h-full min-h-0">
                    <GraphCanvas graph={graph} onNodeClick={setSelectedNode} />
                  </div>
                ) : (
                  <div className="h-full flex items-center justify-center text-gray-600 text-sm">
                    Upload a PyTorch project to see its architecture diagram here.
                  </div>
                )
              ) : page === "settings" ? (
                <SettingsPage />
              ) : page === "help" ? (
                <HelpPage />
              ) : (
                <div className="h-full flex items-center justify-center text-gray-600 text-sm">Page not implemented</div>
              )}
            </div>

            {graph && page === "visualizer" && (
              <div className="h-56 shrink-0 grid grid-cols-2 gap-4 p-4 border-t border-white/5">
                <ModelSummary graph={graph} />
                <LayerTable graph={graph} />
              </div>
            )}
          </main>


        </div>
      </div>

      {showUpload && (
        <UploadModal
          onClose={() => setShowUpload(false)}
          onGraphReady={(g) => {
            setGraph(g);
            setSelectedNode(null);
          }}
        />
      )}
      <ThemeToggle />
    </div>
  );
}
