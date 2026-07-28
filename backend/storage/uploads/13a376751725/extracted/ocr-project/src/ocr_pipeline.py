"""
ocr_pipeline.py
End-to-end OCR pipeline: preprocessing -> PaddleOCR (detection + recognition)
-> confidence filtering -> structured output.

This replaces the old "train 3 CNNs on MNIST digits" approach with a
pretrained text detection + recognition model that reads full lines of
real text (letters, digits, punctuation), the way Google's scanner does.
"""

import time
import os
from typing import Optional
import numpy as np
import cv2

from preprocessing import preprocess_image

# PaddleOCR is initialized once and reused (loading it per-request is slow)
_OCR_ENGINE = None


def get_ocr_engine(lang: str = "en"):
    """Lazily loads PaddleOCR (downloads model weights on first run)."""
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        from paddleocr import PaddleOCR
        _OCR_ENGINE = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
    return _OCR_ENGINE


def run_ocr(
    image_path: str,
    apply_preprocessing: bool = True,
    min_confidence: float = 0.70,
    lang: str = "en",
) -> dict:
    """
    Runs the full pipeline on a single image and returns structured results.

    Returns:
        {
          "text": "combined text, one line per detected text region",
          "lines": [
              {"text": "...", "confidence": 0.98, "box": [[x,y], ...]}
          ],
          "low_confidence_lines": [...],   # flagged for manual review
          "processing_time_ms": 245,
          "engine": "paddleocr"
        }
    """
    start = time.time()

    if apply_preprocessing:
        cleaned = preprocess_image(image_path)
        tmp_path = "_tmp_preprocessed.jpg"
        cv2.imwrite(tmp_path, cleaned)
        target_path = tmp_path
    else:
        target_path = image_path

    engine = get_ocr_engine(lang=lang)
    raw_result = engine.ocr(target_path, cls=True)

    if apply_preprocessing and os.path.exists("_tmp_preprocessed.jpg"):
        os.remove("_tmp_preprocessed.jpg")

    lines = []
    low_confidence_lines = []
    combined_text_parts = []

    # raw_result is a list (one entry per image); each entry is a list of
    # [box, (text, confidence)] for every detected line
    detections = raw_result[0] if raw_result and raw_result[0] else []

    for box, (text, confidence) in detections:
        entry = {"text": text, "confidence": round(float(confidence), 4), "box": box}
        lines.append(entry)
        combined_text_parts.append(text)
        if confidence < min_confidence:
            low_confidence_lines.append(entry)

    elapsed_ms = round((time.time() - start) * 1000, 1)

    return {
        "text": "\n".join(combined_text_parts),
        "lines": lines,
        "low_confidence_lines": low_confidence_lines,
        "processing_time_ms": elapsed_ms,
        "engine": "paddleocr",
    }


def run_ocr_batch(image_paths: list, **kwargs) -> list:
    """Runs run_ocr over a list of image paths, capturing per-image errors
    so one bad file doesn't kill the whole batch."""
    results = []
    for path in image_paths:
        try:
            result = run_ocr(path, **kwargs)
            result["file"] = path
            result["status"] = "ok"
        except Exception as e:
            result = {"file": path, "status": "error", "error": str(e)}
        results.append(result)
    return results


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python ocr_pipeline.py <image_path>")
        sys.exit(1)

    output = run_ocr(sys.argv[1])
    print(json.dumps(output, indent=2))
