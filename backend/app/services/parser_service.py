"""
services/parser_service.py

Why this file exists:
    This is the orchestrator for Phase 1's entire parsing chain. It ties
    together the framework detector and the PyTorch parser tiers and
    produces the final UniversalGraph. This is the ONE place that
    implements the "tiered fallback" strategy discussed throughout the
    project design - if a new framework or tier is added later
    (TensorFlow, ONNX), it plugs in here without changing the API layer
    or the graph schema.

What it does:
    1. Detects the framework of the candidate file
    2. If unsupported -> raises FrameworkNotSupportedError with a clear
       message (Phase 1 only supports PyTorch)
    3. If PyTorch:
       a. If the file contains a from_pretrained(...) call (e.g. a
          HuggingFace fine-tuning script with no locally-defined
          architecture), try the pretrained loader FIRST - torch.fx and
          AST are both guaranteed to fail on such files, since there's no
          locally-defined, instantiable nn.Module class to find.
       b. Otherwise, try torch.fx (Tier 1), falling back to AST (Tier 2)
          on any failure.
       c. If the pretrained path itself fails, it also falls back to AST
          as a last resort (in case the file happens to ALSO define a
          class alongside the from_pretrained call).
    4. Converts whichever raw result succeeded into a UniversalGraph

How it connects:
    Called by api/routes/graph.py. Depends on engines/detector,
    engines/pytorch/{fx_parser,ast_parser,pretrained_parser}, and
    engines/graph/universal_graph.
"""

import logging
from pathlib import Path

from app.core.exceptions import FrameworkNotSupportedError, ModelParsingError
from app.engines.detector.framework_detector import detect_framework
from app.engines.graph.universal_graph import build_universal_graph
from app.engines.pytorch.ast_parser import parse_with_ast
from app.engines.pytorch.fx_parser import run_torch_fx
from app.engines.pytorch.pretrained_parser import has_pretrained_call, run_pretrained_loader
from app.schemas.graph import Confidence, Framework, UniversalGraph

logger = logging.getLogger(__name__)

_SUPPORTED_IN_PHASE_1 = (Framework.PYTORCH,)


def _parse_pytorch_file(job_id: str, model_file: Path):
    """Runs the ordered PyTorch tier chain and returns (raw, confidence)."""

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

    try:
        raw = run_torch_fx(model_file)
        logger.info("job_id=%s parsed via torch.fx (Tier 1)", job_id)
        return raw, Confidence.TRACED
    except ModelParsingError as fx_error:
        logger.warning("job_id=%s torch.fx failed (%s), falling back to AST", job_id, fx_error)
        try:
            raw = parse_with_ast(model_file)
            raw.warnings.insert(
                0,
                f"torch.fx tracing failed ({fx_error}); results are from static "
                f"source analysis only and may be incomplete.",
            )
            logger.info("job_id=%s parsed via AST (Tier 2 fallback)", job_id)
            return raw, Confidence.STATIC
        except ModelParsingError as ast_error:
            logger.error("job_id=%s both torch.fx and AST failed", job_id)
            raise ModelParsingError(
                f"Could not parse model. torch.fx error: {fx_error}. AST error: {ast_error}."
            ) from ast_error


def parse_project(job_id: str, model_file: Path) -> UniversalGraph:
    """
    Run the full Phase 1 parsing chain on a candidate model file and
    return the resulting UniversalGraph.

    Raises:
        FrameworkNotSupportedError: framework detected but not implemented yet
        ModelParsingError: framework is PyTorch but every available tier failed
    """
    framework = detect_framework(model_file)

    if framework not in _SUPPORTED_IN_PHASE_1:
        readable = framework.value if framework != Framework.UNKNOWN else "an unrecognized framework"
        raise FrameworkNotSupportedError(
            f"Detected {readable} in '{model_file.name}'. "
            f"Phase 1 currently supports PyTorch only - support for other "
            f"frameworks is planned for a later phase."
        )

    raw, confidence = _parse_pytorch_file(job_id, model_file)
    return build_universal_graph(job_id=job_id, raw=raw, framework=framework, confidence=confidence)