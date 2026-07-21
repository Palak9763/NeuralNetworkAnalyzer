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

import { useState } from "react";
import Sidebar from "./components/Sidebar";
import TopBar from "./components/TopBar";
import GraphCanvas from "./components/GraphCanvas";
import LayerPropertiesPanel from "./components/LayerPropertiesPanel";
import ModelSummary from "./components/ModelSummary";
import LayerTable from "./components/LayerTable";
import UploadModal from "./components/UploadModal";
import type { GraphNode, UniversalGraph } from "./types/graph";

export default function App() {
  const [graph, setGraph] = useState<UniversalGraph | null>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [showUpload, setShowUpload] = useState(false);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#0d0e14]">
      <Sidebar onUploadClick={() => setShowUpload(true)} />

      <div className="flex-1 flex flex-col min-w-0">
        <TopBar graph={graph} />

        <div className="flex-1 flex min-h-0">
          <main className="flex-1 flex flex-col min-w-0">
            <div className="flex-1 min-h-0">
              {graph ? (
                <GraphCanvas graph={graph} onNodeClick={setSelectedNode} />
              ) : (
                <div className="h-full flex items-center justify-center text-gray-600 text-sm">
                  Upload a PyTorch project to see its architecture diagram here.
                </div>
              )}
            </div>

            {graph && (
              <div className="h-56 shrink-0 grid grid-cols-2 gap-4 p-4 border-t border-white/5">
                <ModelSummary graph={graph} />
                <LayerTable graph={graph} />
              </div>
            )}
          </main>

          <LayerPropertiesPanel node={selectedNode} />
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
    </div>
  );
}
