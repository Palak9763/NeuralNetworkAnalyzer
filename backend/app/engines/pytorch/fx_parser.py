"""
engines/pytorch/fx_parser.py

Why this file exists:
    This is Tier 1 of the PyTorch parsing chain - the most accurate method.
    Instead of just reading source code, it actually instantiates the model
    and traces a dummy input through it, recording every real operation in
    the exact order it executes. This catches things static analysis can
    miss (e.g. layers built inside loops, conditionally constructed layers).

What it does:
    - Dynamically imports the uploaded .py file as a Python module
    - Finds the first nn.Module subclass and instantiates it (no-arg only
      in this version - constructors requiring arguments raise, which
      triggers the AST fallback in parser_service.py)
    - Runs torch.fx.symbolic_trace on it
    - Attaches forward hooks to every leaf module to capture real
      input/output tensor shapes and parameter counts
    - Emits the same RawParseResult shape that ast_parser.py emits

How it connects:
    Called first by services/parser_service.py. If anything here raises
    (unrunnable model, missing weights, incompatible constructor), the
    service catches it and falls back to ast_parser.parse_with_ast().
"""

import importlib.util
import logging
import sys
import uuid
from pathlib import Path

from app.core.exceptions import ModelParsingError
from app.engines.pytorch.ast_parser import RawEdge, RawNode, RawParseResult

logger = logging.getLogger(__name__)

DEFAULT_DUMMY_INPUT_SHAPE = (1, 3, 224, 224)


def _load_module_from_path(file_path: Path):
    """Dynamically import a .py file as a standalone module so its classes
    can be inspected/instantiated, without polluting sys.modules permanently."""
    module_name = f"uploaded_model_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ModelParsingError(f"Could not load {file_path} as a Python module.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - genuinely any error is possible in user code
        raise ModelParsingError(f"Uploaded file raised an error on import: {exc}") from exc
    finally:
        sys.modules.pop(module_name, None)

    return module


def _find_and_instantiate_model(module):
    import torch.nn as nn

    model_classes = [
        obj for name, obj in vars(module).items()
        if isinstance(obj, type) and issubclass(obj, nn.Module) and obj is not nn.Module
    ]
    if not model_classes:
        raise ModelParsingError("No nn.Module subclass found in uploaded file.")

    # Prefer the LAST-defined class first: helper/building-block classes
    # (e.g. a BasicBlock or ResidualBlock used inside a bigger model) are
    # conventionally defined before the top-level model that composes
    # them. Try each candidate with no constructor args, most-likely-main
    # first, falling back through the rest before giving up.
    last_error: TypeError | None = None
    for model_class in reversed(model_classes):
        try:
            model = model_class()
        except TypeError as exc:
            last_error = exc
            continue
        model.eval()
        return model, model_class.__name__

    raise ModelParsingError(
        f"None of the {len(model_classes)} nn.Module subclass(es) found could be "
        f"instantiated without constructor arguments. Last error: {last_error}"
    )


def run_torch_fx(file_path: Path) -> RawParseResult:
    """
    Trace a PyTorch model file with torch.fx and return a RawParseResult.
    Raises ModelParsingError for any failure, which the caller should
    catch and fall back to AST parsing.
    """
    try:
        import torch
        import torch.fx as fx
    except ImportError as exc:
        raise ModelParsingError("PyTorch is not installed in this environment.") from exc

    module = _load_module_from_path(file_path)
    model, model_name = _find_and_instantiate_model(module)

    try:
        traced = fx.symbolic_trace(model)
    except Exception as exc:  # noqa: BLE001
        raise ModelParsingError(f"torch.fx tracing failed: {exc}") from exc

    # Capture real shapes/param counts by hooking every leaf submodule and
    # running one dummy forward pass.
    shapes: dict[str, tuple[list[int], list[int]]] = {}
    param_counts: dict[str, int] = {}
    hooks = []

    def make_hook(name: str):
        def hook(mod, inputs, output):
            in_shape = list(inputs[0].shape) if inputs else None
            out_shape = list(output.shape) if hasattr(output, "shape") else None
            shapes[name] = (in_shape, out_shape)
            param_counts[name] = sum(p.numel() for p in mod.parameters(recurse=False))
        return hook

    for name, submodule in model.named_modules():
        if name == "":
            continue
        hooks.append(submodule.register_forward_hook(make_hook(name)))

    try:
        dummy_input = torch.randn(*DEFAULT_DUMMY_INPUT_SHAPE)
        with torch.no_grad():
            model(dummy_input)
    except Exception as exc:  # noqa: BLE001
        for h in hooks:
            h.remove()
        raise ModelParsingError(
            f"Model raised an error when run with a dummy {DEFAULT_DUMMY_INPUT_SHAPE} input: {exc}"
        ) from exc
    finally:
        for h in hooks:
            h.remove()

    nodes: list[RawNode] = []
    edges: list[RawEdge] = []
    node_id_by_target: dict[str, str] = {}
    counter = 0

    _ADD_LIKE_TARGETS = {"add", "add_", "iadd", "__add__", "__iadd__"}

    def _is_add_like(fx_op_node) -> bool:
        if fx_op_node.op == "call_method" and fx_op_node.target in _ADD_LIKE_TARGETS:
            return True
        if fx_op_node.op == "call_function":
            target_name = getattr(fx_op_node.target, "__name__", str(fx_op_node.target))
            if target_name in _ADD_LIKE_TARGETS:
                return True
        return False

    def _resolve_module_predecessors(arg, via_add=False, _depth=0):
        """
        Walk backwards through non-call_module fx nodes (add, cat, view,
        flatten, etc.) until reaching real call_module nodes, so that an
        intermediate op like a residual `x + shortcut` doesn't silently
        break the edge between the two real layers on either side of it.
        Returns a list of (module_target, passed_through_add) tuples.
        Depth-limited as a safety guard against pathological graphs.
        """
        if _depth > 25 or not hasattr(arg, "op"):
            return []
        if arg.op == "call_module":
            return [(arg.target, via_add)]

        is_add = via_add or _is_add_like(arg)
        results = []
        for sub_arg in arg.args:
            results.extend(_resolve_module_predecessors(sub_arg, via_add=is_add, _depth=_depth + 1))
        return results

    for fx_node in traced.graph.nodes:
        if fx_node.op != "call_module":
            continue
        counter += 1
        node_id = f"node_{counter}"
        node_id_by_target[fx_node.target] = node_id

        in_shape, out_shape = shapes.get(fx_node.target, (None, None))
        module_type = type(dict(model.named_modules())[fx_node.target]).__name__

        nodes.append(RawNode(
            id=node_id,
            type=module_type,
            label=fx_node.target,
            params=param_counts.get(fx_node.target, 0),
            input_shape=in_shape,
            output_shape=out_shape,
        ))

        # Resolve predecessors through any intermediate non-module ops
        # (residual adds, torch.cat, .view, .flatten, etc.) rather than
        # only looking at the immediate arg - this is what lets skip/
        # residual connections survive into the graph at all.
        predecessors: list[tuple[str, bool]] = []
        for arg in fx_node.args:
            predecessors.extend(_resolve_module_predecessors(arg))

        # De-duplicate while preserving first-seen order (a node can
        # appear once even if referenced through multiple argument paths).
        seen: set[str] = set()
        unique_predecessors: list[tuple[str, bool]] = []
        for target, via_add in predecessors:
            if target not in seen:
                seen.add(target)
                unique_predecessors.append((target, via_add))

        # A residual merge point looks like: 2+ distinct module
        # predecessors reaching this node, at least one of them arriving
        # through an add-like op. Heuristic for which branch is the
        # "skip": whichever predecessor is NOT the most recently traced
        # module (i.e. not immediately preceding in execution order) is
        # the shortcut/identity branch - the main path is the one that
        # was just processed right before this merge.
        is_residual_merge = len(unique_predecessors) >= 2 and any(v for _, v in unique_predecessors)
        most_recent_target = nodes[-2].label if len(nodes) >= 2 else None

        for target, _via_add in unique_predecessors:
            if target not in node_id_by_target:
                continue
            is_skip = is_residual_merge and target != most_recent_target
            edges.append(RawEdge(
                source=node_id_by_target[target],
                target=node_id,
                is_skip_connection=is_skip,
            ))

    if not nodes:
        raise ModelParsingError("torch.fx trace produced no call_module nodes.")

    logger.info("torch.fx parse of %s (%s) found %d nodes, %d edges", file_path, model_name, len(nodes), len(edges))

    return RawParseResult(nodes=nodes, edges=edges, model_name=model_name, warnings=[])