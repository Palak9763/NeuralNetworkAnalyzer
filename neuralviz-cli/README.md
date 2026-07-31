# neuralviz

**Instantly visualize your PyTorch / TensorFlow model's architecture — right from your terminal.**

```
pip install neuralviz
neuralviz path/to/my_model.py
```

No server setup. No upload. No account. Just point it at your model file.

---

## Features

- **Browser mode** (default): spawns a local server and opens the full interactive diagram in your browser — the same React/ReactFlow UI as the hosted app
- **Text mode** (`--text`): prints a clean ASCII architecture diagram directly in your terminal, with group brackets and param counts
- **JSON mode** (`--json`): dumps the raw `UniversalGraph` JSON to stdout — pipe it into `jq`, `python -m json.tool`, or any other tool
- **Directory support**: pass a folder and `neuralviz` finds the most likely model file automatically
- **Tiered parsing**: tries `torch.fx` (highest fidelity) → ONNX export → AST static analysis, falling back gracefully

## Supported frameworks

| Framework | Status |
|-----------|--------|
| PyTorch (`nn.Module`) | ✅ Full support |
| TensorFlow / Keras | ✅ Full support |
| Framework-free (NumPy only) | ✅ Pattern matching |
| Pretrained hub models | ✅ Auto-detected |

## Requirements

- Python 3.9+
- Your model's framework already installed in the same environment:
  - PyTorch: `pip install torch`
  - TensorFlow: `pip install tensorflow` or `tensorflow-cpu`

`neuralviz` does **not** install `torch` or `tensorflow` for you — it uses whatever is already in your environment, to avoid version conflicts.

## Usage

```bash
# Open interactive browser diagram (default)
neuralviz path/to/my_model.py

# Print ASCII text diagram in terminal
neuralviz path/to/my_model.py --text

# Use a specific port instead of a random one
neuralviz path/to/my_model.py --port 8842

# Dump raw JSON (pipe-friendly)
neuralviz path/to/my_model.py --json | python -m json.tool

# Point at a directory — auto-finds the best candidate
neuralviz path/to/my_project/

# Show version
neuralviz --version
```

## Example text output

```
Model: SimpleCNN  (pytorch, traced)
────────────────────────────────────────────────────────────────
[ConvBlock]
  conv1    Conv2d        →  16×224×224   (448 params)
  relu     ReLU          →  16×224×224
  pool     MaxPool2d     →  16×112×112
[ConvBlock]
  conv2    Conv2d        →  32×112×112   (4,640 params)
  relu     ReLU          →  32×112×112
  pool     MaxPool2d     →  32×56×56
flatten    Flatten       →  100352
fc         Linear        →  10           (1,003,530 params)
────────────────────────────────────────────────────────────────
Total layers: 8   Total params: 1,008,618   FLOPs: 163.98M
```

## Building for development

```bash
# 1. Build the frontend (requires Node.js)
cd frontend
npm install
npm run build
cd ..

# 2. Copy the build output into the CLI package
.\neuralviz-cli\build_frontend.ps1   # Windows
# or
sh neuralviz-cli/build_frontend.sh   # Linux/macOS

# 3. Install in editable mode
pip install -e neuralviz-cli/
```

## Publishing to PyPI

```bash
cd neuralviz-cli
pip install build twine
python -m build
twine upload --repository testpypi dist/*   # test first
twine upload dist/*                          # publish for real
```

## License

MIT
