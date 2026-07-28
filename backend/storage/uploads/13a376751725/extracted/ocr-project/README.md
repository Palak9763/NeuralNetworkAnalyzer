# OCR Text Extraction System

Extracts text from photos/scans of documents or handwriting — detection +
recognition + confidence scores, similar to Google's document scanner.

Built on **PaddleOCR** (pretrained detection + recognition models) instead
of training digit-classifiers from scratch, so it reads full lines of real
text (letters, numbers, punctuation) rather than isolated single digits.

## Setup

```bash
pip install -r requirements.txt
```

First run will download PaddleOCR's model weights (~10-15MB), so it needs
internet access once. After that it's cached locally.

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
│   └── ocr_pipeline.py     # PaddleOCR wrapper, single + batch inference
├── api/
│   └── app.py               # FastAPI REST endpoints
├── web/
│   └── app.py               # Streamlit UI
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── pytest.ini
```

## If accuracy isn't good enough out of the box

1. Collect ~500-1000 labeled samples of your specific handwriting/document type
2. Fine-tune PaddleOCR's recognition model (training scripts in the
   [PaddleOCR repo](https://github.com/PaddlePaddle/PaddleOCR)), or
3. Fine-tune **TrOCR** instead via HuggingFace `Trainer` — generally easier
   to set up than PaddleOCR's own training pipeline if you're used to
   HuggingFace tooling.

## Notes

- Preprocessing (deskew/denoise/contrast) runs before PaddleOCR and helps
  most on phone photos of documents — less necessary for clean flat scans.
- `min_confidence` (default 0.70) flags low-confidence lines for manual
  review instead of silently returning wrong text.
- Swap `lang="en"` for other PaddleOCR-supported languages if needed.
