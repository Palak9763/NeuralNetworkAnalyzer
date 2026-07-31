# VENDORED COPY — adapted from backend/app/engines/detector/framework_detector.py
# Part of the neuralviz CLI package. Sync manually if upstream changes.
# NOTE: All `app.*` imports have been rewritten to `neuralviz._vendored.*`.

"""
engines/detector/framework_detector.py

Detects the deep learning framework of a Python file from its imports.
Also recognizes .onnx files by extension.

Priority order: ONNX > PYTORCH (torch/transformers/diffusers/timm/peft) > TENSORFLOW > JAX
"""

import ast
import logging
from pathlib import Path

from neuralviz._vendored.schemas.graph import Framework

logger = logging.getLogger(__name__)

_FRAMEWORK_IMPORT_MAP: dict[str, Framework] = {
    # PyTorch ecosystem
    "torch":          Framework.PYTORCH,
    "transformers":   Framework.PYTORCH,
    "diffusers":      Framework.PYTORCH,
    "timm":           Framework.PYTORCH,
    "peft":           Framework.PYTORCH,
    "trl":            Framework.PYTORCH,
    "accelerate":     Framework.PYTORCH,
    "bitsandbytes":   Framework.PYTORCH,
    # TensorFlow / Keras ecosystem
    "tensorflow":     Framework.TENSORFLOW,
    "keras":          Framework.TENSORFLOW,
    "tf_keras":       Framework.TENSORFLOW,
    # JAX ecosystem
    "jax":            Framework.JAX,
    "flax":           Framework.JAX,
    "haiku":          Framework.JAX,
    "optax":          Framework.JAX,
    "orbax":          Framework.JAX,
    # ONNX
    "onnxruntime":    Framework.ONNX,
    "onnx":           Framework.ONNX,
}

_FRAMEWORK_PRIORITY: list[Framework] = [
    Framework.ONNX,
    Framework.PYTORCH,
    Framework.TENSORFLOW,
    Framework.JAX,
]


def _extract_imported_module_names(tree: ast.AST) -> list[str]:
    """Walk an AST and collect every top-level module name that was imported."""
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module.split(".")[0])
    return modules


def detect_framework(file_path: Path) -> Framework:
    """
    Determine the deep learning framework used in a file without executing it.

    1. Returns Framework.ONNX immediately for .onnx files.
    2. Otherwise parses import statements and returns the highest-priority
       framework detected.
    3. Returns Framework.UNKNOWN if no recognized import is found or the file
       cannot be parsed.
    """
    if file_path.suffix.lower() == ".onnx":
        logger.info("Detected framework=ONNX for %s (file extension)", file_path)
        return Framework.ONNX

    try:
        source = file_path.read_text(errors="ignore")
        tree = ast.parse(source)
    except (SyntaxError, OSError) as exc:
        logger.warning("Could not parse %s for framework detection: %s", file_path, exc)
        return Framework.UNKNOWN

    imported_modules = set(_extract_imported_module_names(tree))

    detected: set[Framework] = set()
    for module_name in imported_modules:
        fw = _FRAMEWORK_IMPORT_MAP.get(module_name)
        if fw is not None:
            detected.add(fw)
            logger.debug("Framework candidate %s for %s (import '%s')", fw, file_path, module_name)

    if not detected:
        logger.info("No known framework import found in %s", file_path)
        return Framework.UNKNOWN

    for fw in _FRAMEWORK_PRIORITY:
        if fw in detected:
            logger.info("Detected framework=%s for %s", fw, file_path)
            return fw

    return next(iter(detected))
