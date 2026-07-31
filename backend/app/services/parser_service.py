"""
services/parser_service.py

Why this file exists:
    This is the orchestrator for the entire parsing chain. It ties together
    the framework detector and all parser tiers to produce the final
    UniversalGraph. This is the ONE place that implements the tiered
    fallback strategy - if a new framework or tier is added later it
    plugs in here without changing the API layer or the graph schema.

What it does:
    1. Detects the framework of the candidate file
    2. Routes to the appropriate tier chain:

       PyTorch:
         a. If file contains from_pretrained(...), try pretrained loader first
            (torch.fx and AST are both guaranteed to fail on hub-loading
            scripts with no locally-defined instantiable nn.Module class)
         b. Otherwise: torch.fx (Tier 1) → ONNX export (Tier 1.5) → AST (Tier 2)
         c. If pretrained path fails, falls back to AST

       TensorFlow:
         Keras parser (Tier A/B) → ONNX/tf2onnx fallback (Tier C)

       Unknown framework:
         Custom/Raw-Code AST parser → FrameworkNotSupportedError if that fails

    3. Converts whichever raw result succeeded into a UniversalGraph,
       then runs the grouping engine as the final step.

How it connects:
    Called by api/routes/graph.py. Depends on engines/detector,
    engines/pytorch/{fx_parser,ast_parser,pretrained_parser},
    engines/tensorflow/keras_parser, engines/onnx/onnx_parser,
    engines/custom/raw_code_parser, engines/graph/universal_graph,
    and engines/graph/grouping_engine.
"""

import logging
from pathlib import Path

from app.core.exceptions import FrameworkNotSupportedError, ModelParsingError
from app.engines.detector.framework_detector import detect_framework
from app.engines.graph.grouping_engine import build_groups
from app.engines.graph.universal_graph import build_universal_graph
from app.engines.onnx.onnx_parser import run_onnx_from_pytorch, run_onnx_from_tensorflow
from app.engines.pytorch.ast_parser import parse_with_ast
from app.engines.pytorch.fx_parser import run_torch_fx
from app.engines.pytorch.pretrained_parser import has_pretrained_call, run_pretrained_loader
from app.engines.tensorflow.keras_parser import run_keras_parser
from app.schemas.graph import Confidence, Framework, UniversalGraph

logger = logging.getLogger(__name__)


def _parse_pytorch_file(job_id: str, model_file: Path):
    """Runs the ordered PyTorch tier chain and returns (raw, confidence).

    Tier order:
      1. Pretrained loader (only when a from_pretrained call is detected)
      2. torch.fx  (Tier 1 - full tracing, highest confidence)
      3. ONNX export  (Tier 1.5 - runs model but names may be mangled)
      4. AST  (Tier 2 - static source analysis, lowest confidence)
    """

    # Route straight to the pretrained loader when the file clearly loads
    # a model from a hub rather than defining its own architecture -
    # torch.fx would only waste time failing on it (no locally-defined,
    # no-arg-constructible nn.Module class exists in such files).
    if has_pretrained_call(model_file):
        try:
            raw = run_pretrained_loader(model_file)
            logger.info("job_id=%s parsed via pretrained loader", job_id)
            return raw, Confidence.STATIC
        except ModelParsingError as pretrained_error:
            logger.warning(
                "job_id=%s pretrained loader failed (%s), falling back to AST",
                job_id, pretrained_error,
            )
            try:
                raw = parse_with_ast(model_file)
                raw.warnings.insert(
                    0,
                    f"Pretrained model loading failed ({pretrained_error}); "
                    f"falling back to static source analysis.",
                )
                return raw, Confidence.STATIC
            except ModelParsingError as ast_error:
                raise ModelParsingError(
                    f"Could not parse model. Pretrained loader error: "
                    f"{pretrained_error}. AST error: {ast_error}."
                ) from ast_error

    # --- Tier 1: torch.fx ---
    try:
        raw = run_torch_fx(model_file)
        logger.info("job_id=%s parsed via torch.fx (Tier 1)", job_id)
        return raw, Confidence.TRACED
    except ModelParsingError as fx_error:
        logger.warning(
            "job_id=%s torch.fx failed (%s), trying ONNX export fallback",
            job_id, fx_error,
        )

    # --- Tier 1.5: ONNX export fallback ---
    try:
        raw = run_onnx_from_pytorch(model_file)
        logger.info("job_id=%s parsed via ONNX export (Tier 1.5 fallback)", job_id)
        return raw, Confidence.STATIC
    except ModelParsingError as onnx_error:
        logger.warning(
            "job_id=%s ONNX export also failed (%s), falling back to AST",
            job_id, onnx_error,
        )

    # --- Tier 2: AST static analysis ---
    try:
        raw = parse_with_ast(model_file)
        raw.warnings.insert(
            0,
            f"torch.fx tracing and ONNX export both failed; results are from "
            f"static source analysis only and may be incomplete.",
        )
        logger.info("job_id=%s parsed via AST (Tier 2 final fallback)", job_id)
        return raw, Confidence.STATIC
    except ModelParsingError as ast_error:
        logger.error("job_id=%s all three PyTorch tiers failed", job_id)
        raise ModelParsingError(
            f"Could not parse model. All tiers failed. "
            f"AST error: {ast_error}."
        ) from ast_error


def _parse_tensorflow_file(job_id: str, model_file: Path):
    """Runs the ordered TensorFlow tier chain and returns (raw, confidence).

    Tier order:
      A/B. Keras parser (Functional/Sequential or Subclassed tracing)
      C.   ONNX export via tf2onnx (fallback when Keras fails)
    """
    try:
        raw = run_keras_parser(model_file)
        logger.info("job_id=%s parsed via Keras parser (Tier A/B)", job_id)
        return raw, Confidence.TRACED
    except ModelParsingError as keras_error:
        logger.warning(
            "job_id=%s Keras parsing failed (%s), trying ONNX/tf2onnx fallback",
            job_id, keras_error,
        )

    try:
        raw = run_onnx_from_tensorflow(model_file)
        logger.info("job_id=%s parsed via ONNX/tf2onnx (Tier C fallback)", job_id)
        return raw, Confidence.STATIC
    except ModelParsingError as onnx_error:
        logger.error("job_id=%s both TF tiers failed", job_id)
        raise ModelParsingError(
            f"Could not parse TensorFlow model. "
            f"Keras error: {keras_error}. ONNX error: {onnx_error}."
        ) from onnx_error


def _parse_jax_file(job_id: str, model_file: Path):
    """Runs the JAX/Flax parsing chain."""
    try:
        from app.engines.jax.flax_parser import run_jax_parser
        raw = run_jax_parser(model_file)
        
        # run_jax_parser mixes dynamic and AST. If AST was used, it adds a warning.
        is_static = any("static source analysis" in w for w in raw.warnings)
        confidence = Confidence.STATIC if is_static else Confidence.TRACED
        
        logger.info("job_id=%s parsed via JAX parser", job_id)
        return raw, confidence
    except ModelParsingError as jax_error:
        logger.error("job_id=%s JAX parsing failed", job_id)
        raise ModelParsingError(
            f"Could not parse JAX model. Error: {jax_error}."
        ) from jax_error



def parse_project(job_id: str, model_file: Path) -> UniversalGraph:
    """
    Run the full parsing chain on a candidate model file and
    return the resulting UniversalGraph.

    Raises:
        FrameworkNotSupportedError: detected framework has no parser and
            raw-code pattern matching also failed to find structure
        ModelParsingError: every available tier for the framework failed
    """
    framework = detect_framework(model_file)

    if framework == Framework.PYTORCH:
        raw, confidence = _parse_pytorch_file(job_id, model_file)

    elif framework == Framework.TENSORFLOW:
        raw, confidence = _parse_tensorflow_file(job_id, model_file)

    elif framework == Framework.JAX:
        raw, confidence = _parse_jax_file(job_id, model_file)

    elif framework == Framework.UNKNOWN:
        # Route to custom/raw-code AST parser before giving up entirely.
        # This handles hand-written NumPy models and other framework-free
        # educational implementations.
        try:
            from app.engines.custom.raw_code_parser import parse_raw_code
            raw = parse_raw_code(model_file)
            confidence = Confidence.STATIC
            logger.info("job_id=%s parsed via custom/raw-code AST parser", job_id)
        except ModelParsingError as raw_error:
            raise FrameworkNotSupportedError(
                f"No recognized framework found in '{model_file.name}', and "
                f"custom pattern matching also found no recognizable model "
                f"structure. Raw-code parser error: {raw_error}"
            ) from raw_error

    else:
        # Any future framework without a parser
        readable = framework.value
        raise FrameworkNotSupportedError(
            f"Detected {readable} in '{model_file.name}'. "
            f"Support for this framework is planned for a later phase."
        )

    graph = build_universal_graph(
        job_id=job_id,
        raw=raw,
        framework=framework,
        confidence=confidence,
    )
    return build_groups(graph)