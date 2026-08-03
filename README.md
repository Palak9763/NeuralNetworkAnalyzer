# NeuralNetworkAnalyzer & NeuralViz CLI

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![React Flow](https://img.shields.io/badge/Frontend-React%20Flow-ff007f.svg)](https://reactflow.dev/)
[![CLI Package](https://img.shields.io/badge/PyPI-neuralviz--0.2.0-orange.svg)](https://pypi.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> NeuralNetworkAnalyzer is an enterprise-grade AI platform and CLI tool (`neuralviz`) that automatically inspects deep learning model code, detects framework usage (PyTorch, Hugging Face Transformers, TensorFlow/Keras, JAX/Flax, ONNX), and builds interactive, visual architecture diagrams with layer properties, parameter counts, and skip-connection topology.

---

## 💡 Project Understanding & Architecture Vision

Deep learning models are often defined across complex Python source files, Hugging Face hub checkpoints, or serialized ONNX binaries. Understanding, reviewing, and debugging these architectures manually requires tedious code reading or heavy execution setup.

**NeuralNetworkAnalyzer** solves this with a **Zero-Weight-Loading, Tiered Parsing Engine**:

1. **Framework-Agnostic Contract**: Converts any deep learning architecture into a single fixed JSON schema (`UniversalGraph`).
2. **Zero-Memory Hugging Face Parsing**: Parses models like TrOCR, ViT, LLaMA, GPT-2, and BLIP by fetching **only lightweight `config.json` (~3 KB)**, preventing system memory spikes and `OSError 1455` (Windows paging file limits).
3. **Multi-Tier Execution Fallback**: Combines dynamic execution tracing (`torch.fx`, Keras tracing), ONNX graph export, and static AST code parsing (`ast.NodeVisitor`) so even unrunnable, incomplete code can be visualized.
4. **Dual Interfaces**:
   - **Web Application**: Interactive React Flow canvas with zoom, pan, minimap, group collapsing, and layer property inspection.
   - **CLI Tool (`neuralviz`)**: Zero-config terminal launcher supporting interactive browser mode, terminal ASCII rendering, and JSON dumping.

---

## 🔄 Interactive System Workflows

### 1. High-Level Ingestion & Dual Output Flow

```mermaid
graph TD
    User(["User / Developer"]) -->|"Upload .py / .zip"| Web["Web App / FastAPI"]
    User -->|"Run CLI command"| CLI["NeuralViz CLI Tool"]

    subgraph Core Engine Layer
        FD["Framework Detector"]
        PS["Parser Service Orchestrator"]
        GE["Grouping Engine"]
        UG["Universal Graph Normalizer"]

        FD -->|"Detect Framework"| PS
        PS -->|"Raw Parse Result"| UG
        UG --> GE
    end

    Web --> FD
    CLI --> FD

    GE -->|"UniversalGraph Contract"| WebOut["React Flow Web Canvas"]
    GE -->|"UniversalGraph Contract"| CLIOut["Terminal ASCII / Local Server / JSON"]
```

---

### 2. Multi-Tier Parsing Fallback Chain

```mermaid
flowchart TD
    Start(["Input Source Code / Model File"]) --> FrameworkCheck{"Detect Framework"}

    %% PyTorch & HF Branch
    FrameworkCheck -->|"PyTorch / Hugging Face"| HasPretrained{"Contains from_pretrained?"}
    HasPretrained -->|"Yes"| Tier0["Tier 0: HF Config-Only Parser (config.json only, NO weights)"]
    Tier0 -->|"Success"| Normalizer
    Tier0 -->|"Network / Auth Error"| Tier2

    HasPretrained -->|"No"| Tier1["Tier 1: torch.fx Tracing (Full execution tracing & shapes)"]
    Tier1 -->|"Success"| Normalizer
    Tier1 -->|"Tracing Failed"| Tier15["Tier 1.5: ONNX Export Fallback"]
    Tier15 -->|"Success"| Normalizer
    Tier15 -->|"Export Failed"| Tier2["Tier 2: Source AST Visitor (Static source analysis via NodeVisitor)"]
    Tier2 --> Normalizer

    %% TensorFlow Branch
    FrameworkCheck -->|"TensorFlow / Keras"| TF1["Keras Tracing Engine"]
    TF1 -->|"Success"| Normalizer
    TF1 -->|"Failed"| TF2["tf2onnx Export Fallback"]
    TF2 --> Normalizer

    %% JAX / Flax Branch
    FrameworkCheck -->|"JAX / Flax"| JAX1["Flax / Haiku Inspection"]
    JAX1 -->|"Success"| Normalizer
    JAX1 -->|"Failed"| JAX2["JAX AST Code Parser"]
    JAX2 --> Normalizer

    %% ONNX Branch
    FrameworkCheck -->|"ONNX Binary / Extension"| ONNX1["Direct ONNX Graph Reader"]
    ONNX1 --> Normalizer

    %% Raw Code Fallback
    FrameworkCheck -->|"Unknown / Framework-Free"| RawAST["Custom Raw-Code AST Pattern Matcher"]
    RawAST --> Normalizer

    Normalizer["Universal Graph Normalizer"] --> Grouping["Grouping Engine (ConvBlocks, Residual Skip-Connections, Stages)"]
    Grouping --> End(["Final UniversalGraph Contract"])
```

---

### 3. Detailed Sequence Diagram: Hugging Face Config-Only Pipeline

```mermaid
sequenceDiagram
    autonumber
    actor Developer
    participant CLI as NeuralViz CLI / Web Backend
    participant Detector as Framework Detector
    participant Service as Parser Service
    participant HFParser as HF Config Parser
    participant Hub as Hugging Face Hub API
    participant Grouping as Grouping Engine

    Developer->>CLI: neuralviz ocr_pipeline.py
    CLI->>Detector: detect_framework("ocr_pipeline.py")
    Detector-->>CLI: Framework.PYTORCH (via 'transformers' import)
    CLI->>Service: parse_project(job_id, path)
    Service->>Service: has_pretrained_call() -> True
    Service->>HFParser: run_hf_config_parser(path)
    HFParser->>Hub: AutoConfig.from_pretrained("microsoft/trocr-base-handwritten")
    Hub-->>HFParser: Return config.json (<10 KB)
    HFParser->>HFParser: Execute VisionEncoderDecoder Graph Builder
    HFParser-->>Service: RawParseResult (Canonical Layer Graph)
    Service->>Grouping: build_universal_graph() + build_groups()
    Grouping-->>CLI: UniversalGraph JSON
    CLI-->>Developer: Render Interactive Browser UI / ASCII Terminal Diagram
```

---

## ⚡ Framework Support Matrix

| Framework / Ecosystem         | Detection Import / Extension          | Parsing Strategy                                      |    Weight Loading    |
| ----------------------------- | ------------------------------------- | ----------------------------------------------------- | :------------------: |
| **Hugging Face Transformers** | `transformers`, `diffusers`           | **Tier 0 Config-Only** (`AutoConfig.from_pretrained`) |   ❌ **No (0 MB)**   |
| **PyTorch (`nn.Module`)**     | `torch`, `timm`, `peft`, `accelerate` | `torch.fx` → ONNX Export → AST Visitor                | ❌ Static / Optional |
| **TensorFlow / Keras**        | `tensorflow`, `keras`, `tf_keras`     | Keras Functional/Sequential → `tf2onnx`               | ❌ Static / Optional |
| **JAX / Flax / Haiku**        | `jax`, `flax`, `haiku`, `optax`       | Flax Module tabulate → JAX AST Engine                 | ❌ Static / Optional |
| **ONNX**                      | `.onnx` extension, `onnxruntime`      | Direct ONNX Protobuf Protostructure Engine            |        ❌ No         |
| **Framework-Free Code**       | NumPy / Raw Math Operations           | Custom AST Pattern Matching Engine                    |        ❌ No         |

---

## 🚀 Quick Start: Web Application (`NeuralNetworkAnalyzer`)

### 1. Run Backend Server (FastAPI)

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

_API docs available at:_ `http://localhost:8000/docs`

### 2. Run Frontend Client (React + Vite)

```powershell
cd frontend
npm install
npm run dev
```

_Open web interface at:_ `http://localhost:5173`

---

## 📦 Quick Start: CLI Tool (`neuralviz`)

The CLI package can be run directly against any Python file or project folder without setting up the full backend web server.

### Installation

```bash
pip install neuralviz
```

### CLI Command Options

```bash
# 1. Open interactive React Flow diagram in browser (default)
neuralviz my_model.py

# 2. Print ASCII text diagram directly in terminal
neuralviz my_model.py --text

# 3. Dump raw UniversalGraph JSON to stdout (pipe-friendly)
neuralviz my_model.py --json | jq .

# 4. Run completely offline using local Hugging Face cache (~/.cache/huggingface/)
neuralviz ocr_pipeline.py --offline

# 5. Point to a project directory — auto-detects the best model file
neuralviz my_project_directory/

# 6. Specify custom port for local browser mode
neuralviz my_model.py --port 8842
```

---

## 📁 Repository Structure

```
NeuralNetworkAnalyzer/
├── README.md                           # Main repository documentation & architecture guide
├── backend/                            # FastAPI Web Backend
│   ├── app/
│   │   ├── api/routes/                 # Upload, Graph, and Health routes
│   │   ├── core/                       # Settings & structured domain exceptions
│   │   ├── engines/
│   │   │   ├── detector/               # Framework detection engine
│   │   │   ├── pytorch/                # FX parser, AST parser, Pretrained parser
│   │   │   ├── tensorflow/             # Keras parser
│   │   │   ├── jax/                    # Flax / Haiku parser engine
│   │   │   ├── onnx/                   # Direct ONNX parser
│   │   │   └── graph/                  # UniversalGraph normalizer & Grouping engine
│   │   ├── schemas/                    # Pydantic UniversalGraph contract
│   │   ├── services/                   # Orchestrator & parsing chain service
│   │   └── main.py                     # FastAPI application entrypoint
│   └── requirements.txt
│
├── frontend/                           # React + Vite + React Flow Frontend
│   ├── src/
│   │   ├── components/                 # GraphCanvas, Sidebar, TopBar, LayerProperties
│   │   ├── types/                      # TypeScript mirror of UniversalGraph
│   │   └── App.tsx
│   └── package.json
│
└── neuralviz-cli/                      # Standalone CLI Package (PyPI: neuralviz)
    ├── pyproject.toml                  # Hatchling build & entrypoint configuration
    └── neuralviz/
        ├── cli.py                      # Terminal entrypoint & argument parser
        ├── local_server.py             # Embedded browser server
        ├── text_render.py              # Terminal ASCII tree renderer
        └── _vendored/                  # Standalone CLI-vendored parser engine
```

---

## 👥 Team Members & Credits

- **Sarthak Darandale**
- **Palak Deshmukh**

---

## 📄 License

This project is licensed under the **MIT License**.
