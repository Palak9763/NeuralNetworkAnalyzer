"""
services/parser_service.py

Why this file exists:
    This is the orchestrator for Phase 1's entire parsing chain. It ties
    together the framework detector and the two PyTorch parser tiers
    (torch.fx first, AST fallback second) and produces the final
    UniversalGraph. This is the ONE place that implements the "tiered
    fallback" strategy discussed throughout the project design - if a new
    framework or tier is added later (TensorFlow, ONNX), it plugs in here
    without changing the API layer or the graph schema.

What it does:
    1. Detects the framework of the candidate file
    2. If unsupported -> raises FrameworkNotSupportedError with a clear
       message (Phase 1 only supports PyTorch)
    3. If PyTorch -> tries torch.fx (Tier 1), falls back to AST (Tier 2)
       on any failure, tagging the result's confidence accordingly
    4. Converts whichever raw result succeeded into a UniversalGraph

How it connects:
    Called by api/routes/graph.py. Depends on engines/detector,
    engines/pytorch/{fx_parser,ast_parser}, and engines/graph/universal_graph.
"""

import logging
from pathlib import Path

from app.core.exceptions import FrameworkNotSupportedError, ModelParsingError
from app.engines.detector.framework_detector import detect_framework
from app.engines.graph.universal_graph import build_universal_graph
from app.engines.pytorch.ast_parser import parse_with_ast
from app.engines.pytorch.fx_parser import run_torch_fx
from app.schemas.graph import Confidence, Framework, UniversalGraph

logger = logging.getLogger(__name__)

_SUPPORTED_IN_PHASE_1 = (Framework.PYTORCH,)


def parse_project(job_id: str, model_file: Path) -> UniversalGraph:
    """
    Run the full Phase 1 parsing chain on a candidate model file and
    return the resulting UniversalGraph.

    Raises:
        FrameworkNotSupportedError: framework detected but not implemented yet
        ModelParsingError: framework is PyTorch but both tiers failed
    """
    framework = detect_framework(model_file)

    if framework not in _SUPPORTED_IN_PHASE_1:
        readable = framework.value if framework != Framework.UNKNOWN else "an unrecognized framework"
        raise FrameworkNotSupportedError(
            f"Detected {readable} in '{model_file.name}'. "
            f"Phase 1 currently supports PyTorch only - support for other "
            f"frameworks is planned for a later phase."
        )

    try:
        raw = run_torch_fx(model_file)
        confidence = Confidence.TRACED
        logger.info("job_id=%s parsed via torch.fx (Tier 1)", job_id)
    except ModelParsingError as fx_error:
        logger.warning("job_id=%s torch.fx failed (%s), falling back to AST", job_id, fx_error)
        try:
            raw = parse_with_ast(model_file)
            confidence = Confidence.STATIC
            raw.warnings.insert(
                0,
                f"torch.fx tracing failed ({fx_error}); results are from static "
                f"source analysis only and may be incomplete.",
            )
            logger.info("job_id=%s parsed via AST (Tier 2 fallback)", job_id)
        except ModelParsingError as ast_error:
            logger.error("job_id=%s both torch.fx and AST failed", job_id)
            raise ModelParsingError(
                f"Could not parse model. torch.fx error: {fx_error}. AST error: {ast_error}."
            ) from ast_error

    return build_universal_graph(job_id=job_id, raw=raw, framework=framework, confidence=confidence)
