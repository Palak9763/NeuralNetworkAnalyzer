"""
schemas/graph.py

Why this file exists:
    Defines the Universal Graph JSON contract as Pydantic models. This is
    the single, fixed data shape that every framework parser (PyTorch,
    TensorFlow, JAX, custom) must produce, and that the frontend consumes.
    Fixing this contract early means Stages 5-8 (grouping, layout,
    rendering) never need to change when a new framework is added later.

What it does:
    - GraphNode: one layer/operation in the model
    - GraphEdge: one connection between two layers
    - GraphMeta: metadata about how the graph was produced (framework,
      confidence level, totals)
    - UniversalGraph: the full response returned by GET /graph/{job_id}

How it connects:
    Built by services/parser_service.py after a parser engine runs.
    Returned directly by api/routes/graph.py. Consumed by the frontend's
    TypeScript types (frontend/src/types/graph.ts mirrors this shape).
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Framework(str, Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"
    UNKNOWN = "unknown"


class Confidence(str, Enum):
    TRACED = "traced"           # produced by executing the model (torch.fx)
    STATIC = "static"            # produced by reading source only (AST)
    PARTIAL = "partial"          # some nodes could not be resolved


class GraphNode(BaseModel):
    id: str = Field(..., description="Unique node identifier, e.g. 'node_3'")
    type: str = Field(..., description="Operation/layer type, e.g. 'Conv2d'")
    label: str = Field(..., description="Human-readable name, e.g. 'conv1'")
    input_shape: Optional[list[int]] = None
    output_shape: Optional[list[int]] = None
    params: int = 0
    group_id: Optional[str] = Field(
        default=None, description="Set by the grouping engine (Phase 3), null in Phase 1"
    )


class GraphEdge(BaseModel):
    source: str
    target: str
    is_skip_connection: bool = False


class GraphMeta(BaseModel):
    framework: Framework
    confidence: Confidence
    total_params: int = 0
    total_layers: int = 0
    flops: Optional[int] = Field(
        default=None, description="Populated in Phase 5 via fvcore/torchinfo"
    )
    warnings: list[str] = Field(default_factory=list)


class UniversalGraph(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    job_id: str
    model_name: str
    meta: GraphMeta
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class UploadResponse(BaseModel):
    job_id: str
    filename: str
    status: str = "uploaded"


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # uploaded | detecting | parsing | done | failed
    detail: Optional[str] = None
