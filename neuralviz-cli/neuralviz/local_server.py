"""
neuralviz/local_server.py

Starts a minimal local FastAPI server that serves:
  1. GET /neuralviz/graph  → the already-parsed UniversalGraph JSON
  2. GET /  and static files → the bundled frontend_dist/ React build

Then opens the user's default browser at:
  http://127.0.0.1:{port}/?mode=local

And blocks until Ctrl+C, then exits cleanly.

Design notes:
    - The graph is already in memory when this module is called — no
      re-parsing, no job_id, no Celery, no Redis, no database.
    - The local server only needs to serve one user, one graph, synchronously.
    - We use a random available port by default (OS assigns one when we bind
      to port 0), avoiding conflicts with the user's running dev server.
    - The `?mode=local` query parameter tells the frontend (via client.ts) to
      skip the normal upload flow and fetch from /neuralviz/graph instead.
    - Static files are served via starlette.staticfiles from frontend_dist/.
    - Ctrl+C is caught by uvicorn's signal handling — we just add a clean
      shutdown message via the lifespan context manager.
"""

from __future__ import annotations

import json
import socket
import sys
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuralviz._vendored.schemas.graph import UniversalGraph

FRONTEND_DIST = Path(__file__).parent / "frontend_dist"


def _find_free_port() -> int:
    """Ask the OS to assign us a free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(graph: "UniversalGraph", port: int = 0) -> None:
    """
    Start the local FastAPI server and open the browser.
    Blocks until Ctrl+C.
    """
    import uvicorn
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    # Resolve port
    actual_port = port if port > 0 else _find_free_port()
    url = f"http://127.0.0.1:{actual_port}/?mode=local"

    # Serialise graph once
    graph_json = json.loads(graph.model_dump_json())

    # ── Lifespan: open browser after startup ─────────────────────────────────
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Print before yielding (startup phase)
        print()
        print(f"  neuralviz local server running at {url}")
        print("  Press Ctrl+C to stop.")
        print()
        webbrowser.open(url)
        yield
        # Shutdown phase — nothing extra needed

    app = FastAPI(title="neuralviz local server", lifespan=lifespan)

    # Allow any origin so the page can call the API regardless of how it loaded
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    # ── API endpoint ─────────────────────────────────────────────────────────
    @app.get("/neuralviz/graph")
    async def get_graph() -> JSONResponse:
        """Return the in-memory UniversalGraph as JSON."""
        return JSONResponse(content=graph_json)

    # ── Static file serving ───────────────────────────────────────────────────
    if FRONTEND_DIST.exists() and FRONTEND_DIST.is_dir():
        # Mount /assets and other sub-paths from the Vite build
        assets_dir = FRONTEND_DIST / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        @app.get("/", include_in_schema=False)
        async def serve_index():
            index = FRONTEND_DIST / "index.html"
            if index.exists():
                return FileResponse(str(index))
            return JSONResponse({"error": "index.html not found in frontend_dist"}, status_code=404)

        # Catch-all for any other static files (favicon, etc.)
        @app.get("/{path:path}", include_in_schema=False)
        async def serve_static(path: str):
            target = FRONTEND_DIST / path
            if target.exists() and target.is_file():
                return FileResponse(str(target))
            # SPA fallback: return index.html so React Router handles the path
            index = FRONTEND_DIST / "index.html"
            if index.exists():
                return FileResponse(str(index))
            return JSONResponse({"error": f"Not found: {path}"}, status_code=404)
    else:
        # frontend_dist not bundled — serve a minimal JSON-only response
        @app.get("/", include_in_schema=False)
        async def serve_no_frontend():
            return JSONResponse({
                "message": "neuralviz local server is running, but the frontend was not bundled.",
                "graph_url": f"http://127.0.0.1:{actual_port}/neuralviz/graph",
                "hint": "Run build_frontend.ps1 and reinstall to get the visual diagram.",
            })

        print(
            "  [Warning] frontend_dist/ was not found in the package.\n"
            "  Browser mode will serve raw JSON only.\n"
            "  Run build_frontend.ps1 and reinstall to bundle the frontend.\n",
            file=sys.stderr,
        )

    # ── Start server ──────────────────────────────────────────────────────────
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=actual_port,
        log_level="warning",   # suppress uvicorn's per-request access logs
        access_log=False,
    )
    server = uvicorn.Server(config)
    try:
        server.run()
    except KeyboardInterrupt:
        print("\n  neuralviz: server stopped.")
