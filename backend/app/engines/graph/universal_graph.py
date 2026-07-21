"""
engines/graph/universal_graph.py

Why this file exists:
    This is the single point where any parser's output (torch.fx, AST,
    and in later phases: TensorFlow, ONNX) gets converted into the fixed
    Universal Graph contract defined in schemas/graph.py. Keeping this
    conversion in one place means every downstream consumer (API
    responses, frontend) only ever has to understand one shape.

What it does:
    Takes a RawParseResult (from either fx_parser or ast_parser) plus
    metadata about how it was produced, and returns a UniversalGraph.

How it connects:
    Called by services/parser_service.py as the final step before
    returning a response from POST /upload -> GET /graph/{job_id}.
"""

from app.engines.pytorch.ast_parser import RawParseResult
from app.schemas.graph import (
    Confidence,
    Framework,
    GraphEdge,
    GraphMeta,
    GraphNode,
    UniversalGraph,
)


def build_universal_graph(
    job_id: str,
    raw: RawParseResult,
    framework: Framework,
    confidence: Confidence,
) -> UniversalGraph:
    nodes = [
        GraphNode(
            id=n.id,
            type=n.type,
            label=n.label,
            input_shape=n.input_shape,
            output_shape=n.output_shape,
            params=n.params,
            group_id=None,  # populated by the grouping engine in Phase 3
        )
        for n in raw.nodes
    ]

    edges = [
        GraphEdge(source=e.source, target=e.target, is_skip_connection=e.is_skip_connection)
        for e in raw.edges
    ]

    total_params = sum(n.params for n in nodes)

    meta = GraphMeta(
        framework=framework,
        confidence=confidence,
        total_params=total_params,
        total_layers=len(nodes),
        flops=None,  # populated in Phase 5
        warnings=raw.warnings,
    )

    return UniversalGraph(
        job_id=job_id,
        model_name=raw.model_name,
        meta=meta,
        nodes=nodes,
        edges=edges,
    )
