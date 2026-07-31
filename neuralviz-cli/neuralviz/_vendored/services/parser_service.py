# VENDORED COPY — adapted from backend/app/services/parser_service.py
# Part of the neuralviz CLI package. Sync manually if upstream changes.
# NOTE: All `app.*` imports have been rewritten to `neuralviz._vendored.*`.
# The `settings` dependency from app.core.config has been removed — the CLI
# does not need a database, upload directory, or Celery worker.

"""
services/parser_service.py

Orchestrates the full parsing chain for the CLI:
  framework detection → tiered parser fallback → UniversalGraph → grouping.

Tier order — PyTorch / HuggingFace files containing from_pretrained():
  Tier 0  HF config parser   — downloads only config.json (~3 KB, NO weights)
  Tier 2  AST                — static source analysis, last resort

Tier order — locally-defined nn.Module files:
  Tier 1    torch.fx         — full tracing, highest confidence
  Tier 1.5  ONNX export      — runs model, names may be mangled
  Tier 2    AST              — static source analysis, lowest confidence

TensorFlow:
  Tier A/B  Keras parser
  Tier C    ONNX / tf2onnx

JAX / Flax / Haiku:
  Flax parser (was previously dead code — now properly wired)

ONNX (.onnx files):
  Direct ONNX parser

Unknown framework:
  Custom / raw-code AST → FrameworkNotSupportedError if that fails
"""

import logging
from pathlib import Path

from neuralviz._vendored.core.exceptions import (
    FrameworkNotSupportedError,
    ModelParsingError,
    ParseChainError,
    ParsingFailure,
)
from neuralviz._vendored.engines.detector.framework_detector import detect_framework
from neuralviz._vendored.engines.graph.grouping_engine import build_groups
from neuralviz._vendored.engines.graph.universal_graph import build_universal_graph
from neuralviz._vendored.engines.onnx.onnx_parser import run_onnx_from_pytorch, run_onnx_from_tensorflow
from neuralviz._vendored.engines.pytorch.ast_parser import parse_with_ast
from neuralviz._vendored.engines.pytorch.fx_parser import run_torch_fx
from neuralviz._vendored.engines.pytorch.pretrained_parser import has_pretrained_call
from neuralviz._vendored.engines.tensorflow.keras_parser import run_keras_parser
from neuralviz._vendored.schemas.graph import Confidence, Framework, UniversalGraph

logger = logging.getLogger(__name__)


# ── PyTorch / HuggingFace ────────────────────────────────────────────────────

def _parse_pytorch_file(job_id: str, model_file: Path):
    """
    Tiered parsing chain for PyTorch and HuggingFace files.

    Files with from_pretrained():
      Tier 0 → HF config parser (config.json only, ~3 KB, NO weights)
      Tier 2 → AST fallback

    Files with locally-defined nn.Module:
      Tier 1   → torch.fx
      Tier 1.5 → ONNX export
      Tier 2   → AST
    """
    failures: list[ParsingFailure] = []

    # ── HuggingFace hub-loading scripts ──────────────────────────────────────
    if has_pretrained_call(model_file):

        # Tier 0: download only config.json — zero weight loading, zero RAM spike.
        try:
            from neuralviz._vendored.engines.pytorch.hf_config_parser import run_hf_config_parser
            raw = run_hf_config_parser(model_file)
            logger.info("job_id=%s parsed via HF config parser (Tier 0)", job_id)
            return raw, Confidence.STATIC
        except ModelParsingError as config_err:
            logger.warning(
                "job_id=%s HF config parser failed (%s), falling back to AST",
                job_id, config_err,
            )
            failures.append(ParsingFailure(
                tier="hf_config",
                error=str(config_err),
                suggestion=(
                    "Check your internet connection, verify the checkpoint name "
                    "on huggingface.co, or ensure 'transformers' is installed "
                    "(pip install transformers)."
                ),
            ))

        # Tier 2 fallback: AST — no network, no weights required.
        try:
            raw = parse_with_ast(model_file)
            raw.warnings.insert(
                0,
                f"HF config parser failed ({failures[0].error}); "
                "architecture inferred from static source analysis only.",
            )
            logger.info("job_id=%s parsed via AST fallback (Tier 2)", job_id)
            return raw, Confidence.STATIC
        except ModelParsingError as ast_err:
            failures.append(ParsingFailure(
                tier="ast",
                error=str(ast_err),
                suggestion=(
                    "Define your model as a class inheriting from nn.Module, "
                    "or use a recognized HuggingFace class (BERT, ViT, T5, etc.)."
                ),
                is_fatal=True,
            ))
            raise ParseChainError(failures) from ast_err

    # ── Locally-defined nn.Module files ──────────────────────────────────────

    # Tier 1: torch.fx (requires torch installed)
    try:
        raw = run_torch_fx(model_file)
        logger.info("job_id=%s parsed via torch.fx (Tier 1)", job_id)
        return raw, Confidence.TRACED
    except ModelParsingError as fx_error:
        logger.warning(
            "job_id=%s torch.fx failed (%s), trying ONNX export fallback",
            job_id, fx_error,
        )
        failures.append(ParsingFailure(
            tier="torch_fx",
            error=str(fx_error),
            suggestion=(
                "Simplify dynamic control flow in forward() or add type "
                "annotations so torch.fx can trace the model."
            ),
        ))

    # Tier 1.5: ONNX export
    try:
        raw = run_onnx_from_pytorch(model_file)
        logger.info("job_id=%s parsed via ONNX export (Tier 1.5 fallback)", job_id)
        return raw, Confidence.STATIC
    except ModelParsingError as onnx_error:
        logger.warning(
            "job_id=%s ONNX export also failed (%s), falling back to AST",
            job_id, onnx_error,
        )
        failures.append(ParsingFailure(
            tier="onnx_export",
            error=str(onnx_error),
            suggestion=(
                "Ensure 'onnx' is installed (pip install onnx) and the model "
                "has no custom autograd ops."
            ),
        ))

    # Tier 2: AST static analysis
    try:
        raw = parse_with_ast(model_file)
        raw.warnings.insert(
            0,
            "torch.fx tracing and ONNX export both failed; results are from "
            "static source analysis only and may be incomplete.",
        )
        logger.info("job_id=%s parsed via AST (Tier 2 final fallback)", job_id)
        return raw, Confidence.STATIC
    except ModelParsingError as ast_error:
        logger.error("job_id=%s all three PyTorch tiers failed", job_id)
        failures.append(ParsingFailure(
            tier="ast",
            error=str(ast_error),
            suggestion=(
                "Ensure your model class inherits from nn.Module and defines "
                "__init__ with layer assignments and a forward() method."
            ),
            is_fatal=True,
        ))
        raise ParseChainError(failures) from ast_error


# ── TensorFlow / Keras ───────────────────────────────────────────────────────

def _parse_tensorflow_file(job_id: str, model_file: Path):
    """Keras (Tier A/B) → ONNX/tf2onnx (Tier C)."""
    failures: list[ParsingFailure] = []

    try:
        raw = run_keras_parser(model_file)
        logger.info("job_id=%s parsed via Keras parser (Tier A/B)", job_id)
        return raw, Confidence.TRACED
    except ModelParsingError as keras_error:
        logger.warning(
            "job_id=%s Keras parsing failed (%s), trying ONNX/tf2onnx fallback",
            job_id, keras_error,
        )
        failures.append(ParsingFailure(
            tier="keras",
            error=str(keras_error),
            suggestion="Ensure your model uses the Keras functional or sequential API.",
        ))

    try:
        raw = run_onnx_from_tensorflow(model_file)
        logger.info("job_id=%s parsed via ONNX/tf2onnx (Tier C fallback)", job_id)
        return raw, Confidence.STATIC
    except ModelParsingError as onnx_error:
        logger.error("job_id=%s both TF tiers failed", job_id)
        failures.append(ParsingFailure(
            tier="onnx_tf",
            error=str(onnx_error),
            suggestion="Install tf2onnx: pip install tf2onnx",
            is_fatal=True,
        ))
        raise ParseChainError(failures) from onnx_error


# ── JAX / Flax / Haiku ──────────────────────────────────────────────────────

def _parse_jax_file(job_id: str, model_file: Path):
    """
    JAX/Flax/Haiku parser.

    Previously this was dead code: parser_service.py raised
    FrameworkNotSupportedError for any JAX-detected file instead of calling
    flax_parser.  That bug is now fixed — JAX files reach run_jax_parser().
    """
    failures: list[ParsingFailure] = []
    try:
        from neuralviz._vendored.engines.jax.flax_parser import run_jax_parser
        raw, confidence = run_jax_parser(model_file)
        logger.info("job_id=%s parsed via JAX/Flax parser", job_id)
        return raw, confidence
    except ImportError as imp_err:
        # flax_parser module not present in this vendored build
        failures.append(ParsingFailure(
            tier="jax_flax",
            error=str(imp_err),
            suggestion="JAX support requires flax: pip install flax",
            is_fatal=True,
        ))
        raise ParseChainError(failures) from imp_err
    except ModelParsingError as jax_error:
        logger.error("job_id=%s JAX parser failed: %s", job_id, jax_error)
        failures.append(ParsingFailure(
            tier="jax_flax",
            error=str(jax_error),
            suggestion="Ensure jax and flax (or haiku) are installed.",
            is_fatal=True,
        ))
        raise ParseChainError(failures) from jax_error


# ── ONNX ────────────────────────────────────────────────────────────────────

def _parse_onnx_file(job_id: str, model_file: Path):
    """Direct ONNX file parser (for .onnx extension files)."""
    failures: list[ParsingFailure] = []
    try:
        # Attempt the direct ONNX reader if it exists in this build
        from neuralviz._vendored.engines.onnx.onnx_parser import run_onnx_direct  # type: ignore[attr-defined]
        raw = run_onnx_direct(model_file)
        logger.info("job_id=%s parsed via direct ONNX parser", job_id)
        return raw, Confidence.TRACED
    except (ImportError, AttributeError):
        pass  # run_onnx_direct may not exist yet — fall through to helper
    except ModelParsingError as onnx_err:
        failures.append(ParsingFailure(
            tier="onnx_direct",
            error=str(onnx_err),
            suggestion="Ensure 'onnx' is installed: pip install onnx",
            is_fatal=True,
        ))
        raise ParseChainError(failures) from onnx_err

    # Fallback: run_onnx_from_pytorch can read a bare .onnx file too
    try:
        raw = run_onnx_from_pytorch(model_file)
        logger.info("job_id=%s parsed via ONNX helper fallback", job_id)
        return raw, Confidence.STATIC
    except ModelParsingError as onnx_err2:
        failures.append(ParsingFailure(
            tier="onnx_direct",
            error=str(onnx_err2),
            suggestion="Ensure 'onnx' is installed: pip install onnx",
            is_fatal=True,
        ))
        raise ParseChainError(failures) from onnx_err2


# ── Orchestrator ─────────────────────────────────────────────────────────────

def parse_project(job_id: str, model_file: Path) -> UniversalGraph:
    """
    Run the full parsing chain on a candidate model file and
    return the resulting UniversalGraph.

    Raises:
        FrameworkNotSupportedError — no parser for framework; custom fallback
            also failed.
        ParseChainError — every tier for the framework failed; carries
            per-tier ParsingFailure records with actionable suggestions.
    """
    framework = detect_framework(model_file)
    logger.info(
        "job_id=%s detected framework=%s for %s",
        job_id, framework, model_file.name,
    )

    if framework == Framework.PYTORCH:
        raw, confidence = _parse_pytorch_file(job_id, model_file)

    elif framework == Framework.TENSORFLOW:
        raw, confidence = _parse_tensorflow_file(job_id, model_file)

    elif framework == Framework.JAX:
        raw, confidence = _parse_jax_file(job_id, model_file)

    elif framework == Framework.ONNX:
        raw, confidence = _parse_onnx_file(job_id, model_file)

    elif framework == Framework.UNKNOWN:
        try:
            from neuralviz._vendored.engines.custom.raw_code_parser import parse_raw_code
            raw = parse_raw_code(model_file)
            confidence = Confidence.STATIC
            logger.info("job_id=%s parsed via custom/raw-code AST parser", job_id)
        except ModelParsingError as raw_error:
            raise FrameworkNotSupportedError(
                f"No recognized framework found in '{model_file.name}', and "
                "custom pattern matching also found no recognizable model "
                f"structure. Raw-code parser error: {raw_error}"
            ) from raw_error

    else:
        raise FrameworkNotSupportedError(
            f"Detected framework '{framework.value}' in '{model_file.name}' "
            "has no parser implementation in this version."
        )

    graph = build_universal_graph(
        job_id=job_id,
        raw=raw,
        framework=framework,
        confidence=confidence,
    )
    return build_groups(graph)
