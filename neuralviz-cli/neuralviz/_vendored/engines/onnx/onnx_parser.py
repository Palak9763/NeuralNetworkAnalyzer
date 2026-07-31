# VENDORED COPY — synced from backend/app/engines\onnx\onnx_parser.py
# Part of the neuralviz CLI package. Sync manually if upstream changes.

"""
engines/onnx/onnx_parser.py

Why this file exists:
    This is the ONNX universal fallback tier - positioned between native
    tracers and the existing static AST tiers. When a PyTorch model fails
    torch.fx tracing (e.g. dynamic control flow, custom ops), or a
    TensorFlow model fails Keras parsing, this tier attempts to export
    the model to ONNX format and parse the resulting computation graph.

    ONNX is used as a *fallback*, not a primary path, because:
    - Native tracers preserve real layer names; ONNX export frequently
      mangles them into generic paths like "/encoder/block1/Conv_0"
    - ONNX export itself can fail on truly exotic ops (custom autograd
      Functions, non-standard CUDA kernels) - those cases fall through
      to the next lower tier (AST for PyTorch)

    Confidence is tagged as STATIC (not TRACED) despite ONNX export
    running the model, because the naming lossiness and potential export
    gaps mean it cannot claim the same fidelity as native tracing.

What it does:
    - run_onnx_from_pytorch: loads a PyTorch nn.Module file, exports to
      ONNX via torch.onnx.export (in-memory BytesIO buffer), parses the
      resulting graph
    - run_onnx_from_tensorflow: loads a Keras model file, converts to
      ONNX via tf2onnx, parses the resulting graph
    - _parse_onnx_model: shared graph-walking logic for both paths -
      builds nodes from ONNX NodeProtos, edges via named-tensor matching,
      skip connections via merge-op detection, params via initializer
      tensor sizes

How it connects:
    Called by services/parser_service.py:
    - PyTorch chain: torch.fx → ONNX (this module) → AST
    - TensorFlow chain: Keras parser → ONNX (this module)
    Both callers catch ModelParsingError and fall through to the next tier.
"""

import io
import logging
from pathlib import Path

import numpy as np

from neuralviz._vendored.core.exceptions import ModelParsingError
from neuralviz._vendored.engines.pytorch.ast_parser import RawEdge, RawNode, RawParseResult

logger = logging.getLogger(__name__)

# op_types that act as merge/add points - a node with 2+ distinct input
# sources at one of these ops signals a skip/residual connection
_MERGE_OP_TYPES = frozenset({"Add", "Sum", "Concat", "Average", "Max", "Min"})

_ONNX_FALLBACK_WARNING = (
    "Parsed via ONNX export fallback - layer names may not match the "
    "original source code."
)


def _onnx_dim_to_int(dim) -> int:
    """Convert an ONNX TensorShapeProto.Dimension to int.

    Mirrors the behaviour of _shape_to_int_list() in keras_parser.py:
    symbolic/dynamic dimensions (dim_param) become -1 instead of raising.
    """
    if dim.HasField("dim_value"):
        return int(dim.dim_value)
    # dim_param is a string like "batch_size" or "N" - treat as dynamic
    return -1


def _onnx_type_to_shape(type_proto) -> list[int] | None:
    """Extract a shape list from an ONNX TypeProto, or None if unavailable."""
    try:
        tensor_type = type_proto.tensor_type
        if not tensor_type.HasField("shape"):
            return None
        return [_onnx_dim_to_int(d) for d in tensor_type.shape.dim]
    except Exception:
        return None


def _parse_onnx_model(onnx_model, model_name: str) -> RawParseResult:
    """
    Shared graph-walking logic for a loaded (and shape-inferred) ONNX model.

    Steps:
    1. Build a set of initializer names (weights/biases stored in the graph)
    2. Build a shape map: tensor_name → list[int] from value_info + inputs/outputs
    3. Build a reverse map: output_tensor_name → node_id
    4. Walk graph.node to create RawNodes and RawEdges
    5. Detect skip connections at merge ops
    """
    try:
        import onnx
        import onnx.shape_inference
    except ImportError as exc:
        raise ModelParsingError("onnx package is not installed.") from exc

    # --- Run shape inference to populate value_info ---
    try:
        onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    except Exception as exc:
        logger.warning("ONNX shape inference failed (%s); shapes may be missing", exc)

    graph = onnx_model.graph

    # --- 1. Initializer name set (learnable weights stored in the graph) ---
    initializer_names: set[str] = {init.name for init in graph.initializer}
    initializer_by_name: dict[str, object] = {init.name: init for init in graph.initializer}

    # --- 2. Shape map: tensor name → shape list ---
    shape_map: dict[str, list[int] | None] = {}
    for vi in graph.input:
        shape_map[vi.name] = _onnx_type_to_shape(vi.type)
    for vi in graph.value_info:
        shape_map[vi.name] = _onnx_type_to_shape(vi.type)
    for vi in graph.output:
        shape_map[vi.name] = _onnx_type_to_shape(vi.type)

    # --- 3. Reverse map: output tensor name → node_id (built incrementally) ---
    tensor_to_node_id: dict[str, str] = {}

    nodes: list[RawNode] = []
    edges: list[RawEdge] = []
    counter = 0

    for onnx_node in graph.node:
        counter += 1
        node_id = f"node_{counter}"

        # --- Op label: prefer the node's own .name, fall back to a
        # synthesised "<op_type>_<counter>" so every node has a useful label ---
        raw_label = onnx_node.name.strip() if onnx_node.name.strip() else f"{onnx_node.op_type}_{counter}"
        # Strip leading "/" for cleaner display (ONNX often prefixes paths with /)
        label = raw_label.lstrip("/")

        # --- Param count: sum elements of all initializer inputs ---
        params = 0
        for inp_name in onnx_node.input:
            if inp_name in initializer_by_name:
                init = initializer_by_name[inp_name]
                # dims is a repeated int64 field; prod gives total elements
                dims = list(init.dims)
                params += int(np.prod(dims)) if dims else 1

        # --- Input/output shapes: use first non-initializer tensor for in_shape ---
        in_shape: list[int] | None = None
        for inp_name in onnx_node.input:
            if inp_name and inp_name not in initializer_names:
                in_shape = shape_map.get(inp_name)
                break

        out_shape: list[int] | None = None
        if onnx_node.output:
            out_shape = shape_map.get(onnx_node.output[0])

        nodes.append(RawNode(
            id=node_id,
            type=onnx_node.op_type,
            label=label,
            params=params,
            input_shape=in_shape,
            output_shape=out_shape,
            flops=0,
            line_number=None,
        ))

        # Register this node's output tensors in the reverse map
        for out_name in onnx_node.output:
            if out_name:
                tensor_to_node_id[out_name] = node_id

    # --- 4. Build edges using the reverse map ---
    # We need a second pass because we need all nodes' output registrations
    # to be complete before we can look up sources.  Re-walk graph.node.
    counter2 = 0
    for onnx_node in graph.node:
        counter2 += 1
        target_id = f"node_{counter2}"

        # Collect source node IDs from non-initializer inputs
        source_ids: list[str] = []
        for inp_name in onnx_node.input:
            if not inp_name:
                continue
            if inp_name in initializer_names:
                continue
            src_id = tensor_to_node_id.get(inp_name)
            if src_id and src_id != target_id and src_id not in source_ids:
                source_ids.append(src_id)

        # --- 5. Skip-connection detection ---
        is_merge = (
            onnx_node.op_type in _MERGE_OP_TYPES
            and len(source_ids) >= 2
        )

        for i, src_id in enumerate(source_ids):
            edges.append(RawEdge(
                source=src_id,
                target=target_id,
                is_skip_connection=(is_merge and i > 0),
            ))

    if not nodes:
        raise ModelParsingError("ONNX graph contained no nodes after parsing.")

    logger.info(
        "ONNX parse of model '%s' found %d nodes, %d edges",
        model_name, len(nodes), len(edges),
    )

    return RawParseResult(
        nodes=nodes,
        edges=edges,
        model_name=model_name,
        total_flops=None,
        warnings=[_ONNX_FALLBACK_WARNING],
    )


def run_onnx_from_pytorch(file_path: Path) -> RawParseResult:
    """
    Export a PyTorch nn.Module file to ONNX (in-memory) and parse the graph.

    Raises ModelParsingError for any failure so the caller (parser_service)
    can catch it and fall through to the AST tier.
    """
    try:
        import onnx
        import torch
    except ImportError as exc:
        raise ModelParsingError(f"Required package not available: {exc}") from exc

    # Reuse the exact same load+instantiate helpers from fx_parser.py to
    # avoid duplicating the module-loading logic (same isolation guarantees)
    from neuralviz._vendored.engines.pytorch.fx_parser import (
        DEFAULT_DUMMY_INPUT_SHAPE,
        _find_and_instantiate_model,
        _load_module_from_path,
    )

    try:
        module = _load_module_from_path(file_path)
    except ModelParsingError:
        raise
    except Exception as exc:
        raise ModelParsingError(f"Could not load module from {file_path}: {exc}") from exc

    try:
        model, model_name = _find_and_instantiate_model(module)
    except ModelParsingError:
        raise

    # Export to ONNX in an in-memory buffer (no temp files, no disk I/O)
    buffer = io.BytesIO()
    dummy_input = torch.randn(*DEFAULT_DUMMY_INPUT_SHAPE)

    try:
        # Suppress verbose warnings and redirect stdout/stderr to avoid Windows cp1252 emoji print crashes in PyTorch 2.6
        import sys
        import warnings
        stdout_trap = io.StringIO()
        stderr_trap = io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = stdout_trap
            sys.stderr = stderr_trap
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.onnx.export(
                    model,
                    dummy_input,
                    buffer,
                    opset_version=17,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                )
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
    except Exception as exc:
        raise ModelParsingError(f"torch.onnx.export failed: {exc}") from exc

    buffer.seek(0)
    try:
        onnx_model = onnx.load_from_string(buffer.read())
    except Exception as exc:
        raise ModelParsingError(f"Could not load exported ONNX model: {exc}") from exc

    logger.info("job file=%s exported to ONNX successfully, parsing graph", file_path.name)
    result = _parse_onnx_model(onnx_model, model_name)

    # Prepend a PyTorch-specific context note alongside the standard warning
    result.warnings.insert(0, f"torch.fx tracing failed; fell back to ONNX export for '{model_name}'.")
    return result


def run_onnx_from_tensorflow(file_path: Path) -> RawParseResult:
    """
    Convert a TensorFlow/Keras model to ONNX via tf2onnx and parse the graph.

    Raises ModelParsingError for any failure so the caller (parser_service)
    can catch it and propagate the combined error.
    """
    try:
        import onnx
        import tensorflow as tf
    except ImportError as exc:
        raise ModelParsingError(f"Required package not available: {exc}") from exc

    try:
        import tf2onnx
    except ImportError as exc:
        raise ModelParsingError(
            "tf2onnx is not installed - TensorFlow→ONNX fallback unavailable."
        ) from exc

    # Reuse Keras model loading helpers from keras_parser.py
    from neuralviz._vendored.engines.tensorflow.keras_parser import _find_keras_model, _load_module_from_path

    try:
        module = _load_module_from_path(file_path)
    except ModelParsingError:
        raise
    except Exception as exc:
        raise ModelParsingError(f"Could not load module from {file_path}: {exc}") from exc

    try:
        model, _is_functional = _find_keras_model(module)
    except ModelParsingError:
        raise

    model_name = getattr(model, "name", None) or type(model).__name__

    # Build the model by calling it once with a dummy input so that
    # tf2onnx can inspect a fully-built graph
    try:
        dummy_input = tf.zeros((1, 224, 224, 3))
        model(dummy_input)
    except Exception as exc:
        logger.warning("Could not run TF model with default dummy input (%s); trying anyway", exc)

    # Convert to ONNX proto using tf2onnx
    try:
        import tf2onnx.convert as tf2onnx_convert
        # Provide the input signature so tf2onnx knows the expected shape
        input_signature = [tf.TensorSpec([None, 224, 224, 3], tf.float32, name="input")]
        onnx_model_proto, _external_tensor_storage = tf2onnx_convert.from_keras(
            model,
            input_signature=input_signature,
            opset=17,
        )
    except Exception as exc:
        raise ModelParsingError(f"tf2onnx conversion failed: {exc}") from exc

    logger.info("job file=%s converted to ONNX via tf2onnx, parsing graph", file_path.name)
    result = _parse_onnx_model(onnx_model_proto, model_name)
    result.warnings.insert(0, f"Keras parsing failed; fell back to ONNX/tf2onnx export for '{model_name}'.")
    return result
