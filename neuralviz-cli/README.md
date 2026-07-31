# neuralviz

**Instantly visualize your PyTorch, Hugging Face, TensorFlow, and JAX model architecture — right from your terminal.**

[![PyPI](https://img.shields.io/pypi/v/neuralviz.svg)](https://pypi.org/project/neuralviz/)
[![Python Version](https://img.shields.io/pypi/pyversions/neuralviz.svg)](https://pypi.org/project/neuralviz/)
[![License](https://img.shields.io/github/license/sarth/NeuralNetworkAnalyzer.svg)](LICENSE)

```bash
pip install neuralviz
neuralviz path/to/my_model.py
```

No server setup. No file uploads. No account required. Point it at your model file or directory.

---

## ⚡ Key Features

- 🌐 **Browser mode** (default): Spawns a local embedded server and opens an interactive React Flow diagram in your browser.
- 📺 **Text mode** (`--text`): Prints a clean ASCII architecture diagram directly in your terminal with group brackets and parameter counts.
- 📄 **JSON mode** (`--json`): Dumps the raw `UniversalGraph` JSON contract to stdout (pipe-friendly for `jq` or custom tools).
- 📴 **Offline mode** (`--offline`): Disables all network access and parses Hugging Face models using only the local cache (`~/.cache/huggingface/`).
- ⚡ **Zero-Weight Hugging Face Config Parser**: Parses models like TrOCR, ViT, LLaMA, GPT-2, and BLIP using **only `config.json` (~3 KB)** without downloading weights or exhausting system RAM (`OSError 1455`).
- 📁 **Directory Auto-Detection**: Pass a project folder and `neuralviz` automatically selects the primary model file.

---

## 🔄 Supported Frameworks

| Framework / Ecosystem | Parsing Tier Strategy | Status |
|-----------------------|-----------------------|:------:|
| **Hugging Face Transformers** | Tier 0: Config-Only (`AutoConfig.from_pretrained`) | ✅ Full Support |
| **PyTorch (`nn.Module`)** | `torch.fx` → ONNX Export → AST Visitor | ✅ Full Support |
| **TensorFlow / Keras** | Keras Functional/Sequential → `tf2onnx` | ✅ Full Support |
| **JAX / Flax / Haiku** | Flax Module Inspection → JAX AST Engine | ✅ Full Support |
| **ONNX** | Direct `.onnx` Graph Reader | ✅ Full Support |
| **Framework-Free Math** | Custom AST Pattern Matcher | ✅ Full Support |

---

## 🚀 Usage Examples

```bash
# Open interactive browser diagram (default)
neuralviz path/to/my_model.py

# Run in terminal text mode (ASCII rendering)
neuralviz path/to/my_model.py --text

# Use offline mode (reads local Hugging Face cache only)
neuralviz ocr_pipeline.py --offline

# Specify custom port for local browser mode
neuralviz path/to/my_model.py --port 8842

# Dump raw JSON (pipe to jq or python json.tool)
neuralviz path/to/my_model.py --json | python -m json.tool

# Pass a directory — auto-finds candidate model file
neuralviz path/to/my_project/

# Check installed version
neuralviz --version
```

---

## 📺 ASCII Terminal Output Example (`--text`)

```
Model: VisionEncoderDecoderModel (microsoft/trocr-base-handwritten) (pytorch, static)
────────────────────────────────────────────────────────────────────────
  [encoder.embeddings]          VisionEmbeddings     →  PatchEmbeddings
  [encoder.layer.0]             ViTLayer             →  TransformerBlock
  [encoder.layer.1]             ViTLayer             →  TransformerBlock
  ... (10 layers)
  [decoder.embeddings]          TrOCREmbeddings      →  TextEmbeddings
  [decoder.layer.0]             TrOCRDecoderLayer    →  DecoderBlock
  [decoder.layer.1]             TrOCRDecoderLayer    →  DecoderBlock
  ... (10 layers)
  [lm_head]                     LMHead               →  LinearProjections
────────────────────────────────────────────────────────────────────────
Architecture built from config.json only (model_type='vision-encoder-decoder').
No weights loaded. System RAM protected.
```

---

## 👥 Authors & Credits

Developed as part of the **NeuralNetworkAnalyzer** platform by:
- **Sarthak Darandale**
- **Palak Deshmukh**

---

## 📄 License

MIT License
