# OCR Text Extraction System

Extracts text from photos/scans of documents or handwriting — detection +
recognition + confidence scores, similar to Google's document scanner.

**Hybrid approach:**
- **PaddleOCR** finds *where* the text lines are (detection) — good at
  handling skewed photos, uneven lighting, messy notebook pages.
- **TrOCR** (Microsoft, fine-tuned on the IAM Handwriting Database) reads
  *what* each line says (recognition) — much better at cursive handwriting
  than PaddleOCR's built-in recognizer, which is trained mostly on printed
  and scene text.

So: PaddleOCR draws the boxes, TrOCR reads what's inside each box.

## Setup

```bash
pip install -r requirements.txt
```

First run downloads two sets of model weights:
- PaddleOCR detection model (~5MB)
- TrOCR handwriting model (~1.3GB for the default `base` model)

Both are cached locally after the first run — no internet needed after that.

If the TrOCR download is too slow/large, swap to the smaller model in
`src/ocr_pipeline.py` (`get_recognizer` call in `run_ocr`):
- `"microsoft/trocr-small-handwritten"` — faster, smaller, less accurate
- `"microsoft/trocr-base-handwritten"` — default, good balance
- `"microsoft/trocr-large-handwritten"` — best accuracy, slow, ~2.5GB

## Usage

**Command line (single image):**
```bash
cd src
python ocr_pipeline.py /path/to/image.jpg
```

**Web UI (drag-and-drop):**
```bash
streamlit run web/app.py
```
Opens at http://localhost:8501 — upload an image, see extracted text,
per-line confidence bars, and export to JSON/CSV/TXT.

**REST API:**
```bash
uvicorn api.app:app --reload --port 8000
```
- `POST /predict` — single image → JSON with text + confidence
- `POST /predict-batch` — multiple images → list of results
- `GET /models` — info about the OCR engine
- `GET /health` — liveness check
- Interactive docs at http://localhost:8000/docs

## Running tests

```bash
pytest tests/ -v                      # unit tests (no network needed)
pytest tests/ -v -m integration       # + integration test (needs model download)
```

## Project structure

```
ocr-project/
├── src/
│   ├── preprocessing.py    # denoise, contrast enhance, deskew
│   └── ocr_pipeline.py     # PaddleOCR (detect) + TrOCR (recognize)
├── api/
│   └── app.py               # FastAPI REST endpoints
├── web/
│   └── app.py               # Streamlit UI
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── pytest.ini
```

## If accuracy still isn't good enough

1. Try the `large` TrOCR checkpoint (see Setup above) for a real accuracy bump
2. Take clearer, well-lit, flat photos — TrOCR is sensitive to blur and
   extreme skew even with preprocessing
3. **Fine-tune on your own handwriting** — see `finetune/README.md` for the
   full workflow. This is the real fix if a specific person's handwriting
   style is consistently misread: auto-crop lines from your own photos,
   label them, and fine-tune TrOCR to your exact handwriting.

## Notes

- Preprocessing (deskew/denoise/contrast) runs before detection and helps
  most on phone photos of documents — less necessary for clean flat scans.
- `min_confidence` (default 0.60) flags low-confidence lines for manual
  review instead of silently returning wrong text. TrOCR confidence scores
  run lower than PaddleOCR's did — this is expected, not a bug.
- Running on CPU works but is slow per line (transformer model); a GPU
  (if available) is used automatically via `torch.cuda.is_available()`.
