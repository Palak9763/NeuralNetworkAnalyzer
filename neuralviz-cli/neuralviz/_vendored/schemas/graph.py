# VENDORED COPY — synced from backend/app/schemas/graph.py
# This file is part of the neuralviz CLI package's vendored engine layer.
# If you update backend/app/schemas/graph.py, manually sync changes here.
# See neuralviz-cli/README.md for the sync procedure.

"""
schemas/graph.py

Defines the Universal Graph JSON contract as Pydantic models. This is
the single, fixed data shape that every framework parser (PyTorch,
TensorFlow, JAX, custom) must produce, and that the frontend consumes.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Framework(str, Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"
    ONNX = "onnx"
    UNKNOWN = "unknown"


class Confidence(str, Enum):
    TRACED = "traced"           # produced by executing the model (torch.fx)
    STATIC = "static"           # produced by reading source only (AST)
    PARTIAL = "partial"         # some nodes could not be resolved


class GraphNode(BaseModel):
    id: str = Field(..., description="Unique node identifier, e.g. 'node_3'")
    type: str = Field(..., description="Operation/layer type, e.g. 'Conv2d'")
    label: str = Field(..., description="Human-readable name, e.g. 'conv1'")
    input_shape: Optional[list[int]] = None
    output_shape: Optional[list[int]] = None
    params: int = 0
    flops: Optional[int] = Field(default=None, description="FLOP count for this node/layer")
    line_number: Optional[int] = Field(default=None, description="Source line number")
    group_id: Optional[str] = Field(default=None, description="Set by grouping engine")


class GraphEdge(BaseModel):
    source: str
    target: str
    is_skip_connection: bool = False


class GroupType(str, Enum):
    CONV_BLOCK = "conv_block"
    RESIDUAL_BLOCK = "residual_block"
    STAGE = "stage"


class GraphGroup(BaseModel):
    id: str = Field(..., description="Unique group identifier, e.g. 'group_3'")
    label: str = Field(..., description="Display label, e.g. 'ConvBlock'")
    type: GroupType
    member_node_ids: list[str] = Field(default_factory=list)
    parent_group_id: Optional[str] = Field(default=None)
    repeat_count: int = Field(default=1)


class GraphMeta(BaseModel):
    framework: Framework
    confidence: Confidence
    total_params: int = 0
    total_layers: int = 0
    flops: Optional[int] = Field(default=None)
    warnings: list[str] = Field(default_factory=list)


class UniversalGraph(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    job_id: str
    model_name: str
    meta: GraphMeta
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    groups: list[GraphGroup] = Field(default_factory=list)


class UploadResponse(BaseModel):
    job_id: str
    filename: str
    status: str = "uploaded"


class SourceResponse(BaseModel):
    job_id: str
    filename: str
    code: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    detail: Optional[str] = None


class ProjectMetadata(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    detected_framework: str
    primary_model_file: str
    files_scanned: int
    model_candidates: list[str]
    dependencies: list[str]
    warnings: list[str]


class ProjectAnalysisResponse(BaseModel):
    metadata: ProjectMetadata
    graph: UniversalGraph
