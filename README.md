# NeuralNetworkAnalyzer

Arya Gandhi
<<<<<<< HEAD

Upload a PyTorch model → get an interactive, auto-generated architecture
diagram (React Flow), with real layer shapes, parameter counts, and a
click-to-inspect properties panel.

This repo contains **Phase 1 + Phase 2** of the full project roadmap:
a working, tested, end-to-end slice (upload → detect → parse → render),
built with no database, no auth, and no job queue — those are deliberately
deferred to later phases (see [Roadmap](#roadmap) below).

---

## What's included right now

**Backend (FastAPI)**
- `POST /api/v1/upload` — accepts `.py` or `.zip`, validates size/type
- `GET /api/v1/graph/{job_id}` — runs the parsing chain, returns the
  Universal Graph JSON
- Framework Detector — reads imports via Python's `ast` module
- **Tier 1 parser: torch.fx** — traces real execution, captures real
  input/output shapes and parameter counts per layer
- **Tier 2 parser: AST fallback** — static source analysis, works even
  when a model can't actually be instantiated/run (missing weights,
  constructor args that can't be inferred, etc.)
- Every response follows one fixed JSON contract regardless of which
  tier produced it (see `backend/app/schemas/graph.py`)

**Frontend (React + TypeScript + Vite + Tailwind + React Flow)**
- Sidebar, top bar with framework/confidence badges, upload modal
- Interactive diagram: color-coded layer types, skip-connection styling,
  zoom/pan, minimap
- Click a node → Layer Properties panel (type, shape, params)
- Model Summary card + sortable Layer Table

**Verified working end-to-end** — a real `SimpleCNN` was uploaded and
parsed via torch.fx (correct shapes/params extracted), and a model with
required constructor args was uploaded to confirm the AST fallback
triggers correctly when tracing fails.

---

## Running it locally

### Backend

On Windows PowerShell:

```powershell
cd backend
py -3 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

If PowerShell blocks script execution, run this once first:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Visit `http://localhost:8000/docs` for interactive API docs.

### Frontend

```powershell
cd frontend
npm install
Copy-Item .env.example .env      # defaults to http://localhost:8000
npm run dev
```

Visit `http://localhost:5173`.

### Try it

1. Open the frontend, click **"+ Upload Project"**
2. Upload any `.py` file containing a `torch.nn.Module` subclass with a
   no-argument constructor (see `backend/sample_models/simple_cnn.py`
   for a working example)
3. The diagram renders automatically; click any node to see its details

---

## Project structure

```
backend/
  app/
    api/routes/          # HTTP layer only — no business logic here
      upload.py
      graph.py
      health.py
    core/
      config.py           # environment-driven settings
      exceptions.py        # domain exception hierarchy
    engines/
      detector/            # framework detection (ast-based)
      pytorch/
        fx_parser.py        # Tier 1: torch.fx tracing
        ast_parser.py        # Tier 2: static AST fallback
      graph/
        universal_graph.py   # normalizes any parser's output -> fixed schema
    schemas/
      graph.py              # the Universal Graph JSON contract (pydantic)
    services/
      upload_service.py      # upload orchestration
      parser_service.py       # detect -> parse chain -> fallback orchestration
    utils/
      file_handler.py          # pure filesystem helpers
    main.py                     # FastAPI app entrypoint
  requirements.txt
  storage/uploads/               # local temp storage (gitignored)

frontend/
  src/
    api/client.ts                # typed wrapper over backend endpoints
    types/graph.ts                # TS mirror of backend schemas/graph.py
    components/
      Sidebar.tsx
      TopBar.tsx
      GraphCanvas.tsx              # React Flow rendering
      LayerPropertiesPanel.tsx
      ModelSummary.tsx
      LayerTable.tsx
      UploadModal.tsx
    App.tsx
    main.tsx
  package.json
```

---

## The Universal Graph contract

Every parser (PyTorch now, TensorFlow/JAX/custom later) must produce this
exact shape. It's what lets the frontend, and every stage after parsing,
stay identical no matter which framework or tier produced the data.

```json
{
  "job_id": "1e2929812e1b",
  "model_name": "SimpleCNN",
  "meta": {
    "framework": "pytorch",
    "confidence": "traced",
    "total_params": 1008618,
    "total_layers": 8,
    "flops": null,
    "warnings": []
  },
  "nodes": [
    {
      "id": "node_1",
      "type": "Conv2d",
      "label": "conv1",
      "input_shape": [1, 3, 224, 224],
      "output_shape": [1, 16, 224, 224],
      "params": 448,
      "group_id": null
    }
  ],
  "edges": [
    { "source": "node_1", "target": "node_2", "is_skip_connection": false }
  ]
}
```

`confidence` is `"traced"` when torch.fx successfully ran the model, or
`"static"` when it fell back to AST-only analysis. `group_id` stays
`null` until Phase 3 (block grouping) is built — the field already
exists in the contract so nothing downstream needs to change later.

---

## Known limitations of this phase (by design)

- **PyTorch only.** TensorFlow/JAX/custom-code uploads are detected and
  rejected with a clear message, not silently mishandled.
- **No-argument model constructors only** for the torch.fx tier. Models
  requiring constructor arguments (e.g. `num_classes`) automatically fall
  back to the AST tier, which has no such requirement but produces less
  detail (no confirmed shapes/params).
- **No grouping yet.** Every individual op (Conv2d, ReLU, BatchNorm2d...)
  is its own node — deep models like ResNet50 will render as a long flat
  chain rather than grouped "Stage 1 - Conv2_x" blocks. That's Phase 3.
- **Simple layered auto-layout**, not Dagre.js/ELK.js. Works fine for
  linear/simple-branching models; will get replaced in Phase 4.
- **No persistence.** Uploads live in `backend/storage/uploads/{job_id}/`
  and are never cleaned up automatically yet — fine for local dev, not
  for production.
- **Synchronous parsing.** Large models will block the request thread.
  Acceptable until Phase 7 adds Celery/Redis.

---

## Roadmap

| Phase | What it adds |
|---|---|
| 1 ✅ | Core parsing engine: upload, framework detection, torch.fx + AST chain, Universal Graph JSON |
| 2 ✅ | React Flow rendering, Layer Properties panel, Model Summary, Layer Table |
| 3 | Grouping engine: collapse Conv+BN+ReLU into blocks, detect skip connections, detect repeated blocks (`Block × N`) |
| 4 | Real layout engine (Dagre.js/ELK.js) replacing the simple depth-based layout |
| 5 | FLOPs + param breakdown (torchinfo/fvcore), Code Preview tab |
| 6 | PostgreSQL + SQLAlchemy models, JWT auth, Projects/Saved Graphs/History pages |
| 7 | Redis + Celery for background job processing (once uploads are demonstrably slow) |
| 8 | TensorFlow/Keras parser chain, ONNX as a secondary universal fallback, JAX support, best-effort custom/raw-code parsing via AST pattern-matching |

Build strictly in this order — each phase should be working and tested
before the next one starts.
=======
#Team Members :Sarthak Darandale,Palak Deshmukh
NeuralNetworkAnalyzer is a full-stack AI platform that automatically detects deep learning frameworks, parses neural network architectures from uploaded projects or model files, and generates interactive architecture diagrams. Supports PyTorch, TensorFlow, JAX, and custom models with a universal graph visualization engine.
>>>>>>> e9ef88b6d7a7fb1b04db89ad55657cd0af718e86
