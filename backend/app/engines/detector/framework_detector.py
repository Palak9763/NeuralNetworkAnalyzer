"""
engines/detector/framework_detector.py

Why this file exists:
    Before any parsing can happen, the system needs to know which deep
    learning framework a file was built with, so it can route the file
    to the correct parser chain (PyTorch chain, TensorFlow chain, etc).
    This is a cheap, static check that runs before any heavier parsing.

What it does:
    Parses the target file's import statements using Python's built-in
    `ast` module (never executes the code) and matches them against
    known framework import names.

How it connects:
    Called by services/parser_service.py immediately after a candidate
    model file has been located by utils/file_handler.py. Its result
    (a Framework enum) decides which engine (engines/pytorch/*, etc.)
    handles the rest of the pipeline.
"""

import ast
import logging
from pathlib import Path

from app.schemas.graph import Framework

logger = logging.getLogger(__name__)

_FRAMEWORK_IMPORT_MAP: dict[str, Framework] = {
    "torch": Framework.PYTORCH,
    "tensorflow": Framework.TENSORFLOW,
    "keras": Framework.TENSORFLOW,
    "jax": Framework.JAX,
    "flax": Framework.JAX,
}


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
    Read a Python file's source and determine which framework it uses,
    based purely on its import statements. The file is never executed.

    Returns Framework.UNKNOWN if no recognized import is found, or if the
    file cannot be parsed as valid Python (e.g. syntax errors).
    """
    try:
        source = file_path.read_text(errors="ignore")
        tree = ast.parse(source)
    except (SyntaxError, OSError) as exc:
        logger.warning("Could not parse %s for framework detection: %s", file_path, exc)
        return Framework.UNKNOWN

    modules = _extract_imported_module_names(tree)

    for module_name in modules:
        if module_name in _FRAMEWORK_IMPORT_MAP:
            framework = _FRAMEWORK_IMPORT_MAP[module_name]
            logger.info("Detected framework=%s for %s (import '%s')", framework, file_path, module_name)
            return framework

    logger.info("No known framework import found in %s", file_path)
    return Framework.UNKNOWN
