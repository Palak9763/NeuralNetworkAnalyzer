# VENDORED COPY — synced from backend/app/engines\custom\raw_code_parser.py
# Part of the neuralviz CLI package. Sync manually if upstream changes.

"""
engines/custom/raw_code_parser.py

Why this file exists:
    Some model uploads have no framework at all - raw NumPy neural nets,
    manual backprop scripts, or educational implementations common among
    students and researchers. Since there is no framework tracer or ONNX
    exporter to hook into, this module performs best-effort AST pattern
    matching on the source code to infer the network structure.

What it does:
    1. Scans the file's AST for class definitions containing an __init__
       method and at least one forward-like method (forward, call, predict, run)
    2. Inside __init__: detects attribute assignments that look like weight
       initializations (e.g. self.W1 = np.random.randn(784, 128)) and
       extracts attribute names + shapes (if specified as literal args)
    3. Inside the forward-like method: detects mathematical operations
       (np.dot, np.matmul, @ operator, np.maximum, sigmoid, softmax, etc.)
       in source call order
    4. Emits a RawParseResult tagged with Confidence.STATIC and a clear
       warning that the structure is inferred from pattern matching

How it connects:
    Called by services/parser_service.py when detect_framework() returns
    Framework.UNKNOWN. If this parser also finds no recognizable structure,
    it raises ModelParsingError, which parser_service catches to raise
    FrameworkNotSupportedError.
"""

import ast
import logging
from pathlib import Path

import numpy as np

from neuralviz._vendored.core.exceptions import ModelParsingError
from neuralviz._vendored.engines.pytorch.ast_parser import RawEdge, RawNode, RawParseResult

logger = logging.getLogger(__name__)

# Forward-like method names to search for in model classes
_FORWARD_METHOD_NAMES = ("forward", "call", "predict", "run")

# Known weight initialization function names
_INIT_FUNC_NAMES = {
    "randn", "rand", "zeros", "ones", "empty", "eye", "normal",
    "uniform", "kaiming_uniform", "xavier_uniform", "zeros_like", "ones_like",
}

# Known weight attribute name patterns (if assigned any call/expression)
_WEIGHT_ATTR_PREFIXES = ("w", "b", "weight", "bias", "param", "kernel")

_RAW_CODE_WARNING = (
    "This appears to be a custom/framework-free implementation. Structure "
    "is inferred from source code patterns and may be incomplete - shapes "
    "and parameter counts are not confirmed."
)


def _extract_shape_from_args(args: list[ast.expr]) -> list[int] | None:
    """Try to extract a tuple or list of literal integer dimensions from call args."""
    if not args:
        return None

    # Case 1: single tuple or list arg, e.g. np.zeros((784, 128))
    first = args[0]
    if isinstance(first, (ast.Tuple, ast.List)):
        dims = []
        for elt in first.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                dims.append(elt.value)
            elif isinstance(elt, ast.UnaryOp) and isinstance(elt.op, ast.USub) and isinstance(elt.operand, ast.Constant):
                dims.append(-int(elt.operand.value))
            else:
                return None
        return dims if dims else None

    # Case 2: positional args, e.g. np.random.randn(784, 128)
    dims = []
    for arg in args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
            dims.append(arg.value)
        else:
            break
    return dims if dims else None


def _is_weight_init_call(call: ast.Call, attr_name: str) -> bool:
    """Return True if a Call AST node represents a weight initialization."""
    func_name = ""
    if isinstance(call.func, ast.Attribute):
        func_name = call.func.attr
    elif isinstance(call.func, ast.Name):
        func_name = call.func.id

    if func_name.lower() in _INIT_FUNC_NAMES:
        return True

    # Fall back to attribute name convention (e.g. self.W1 = ...)
    attr_lower = attr_name.lower()
    return any(attr_lower.startswith(p) for p in _WEIGHT_ATTR_PREFIXES)


def _find_candidate_classes(tree: ast.AST) -> list[ast.ClassDef]:
    """Find all ClassDef nodes that define both __init__ and a forward-like method."""
    candidates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        method_names = {
            n.name for n in node.body
            if isinstance(n, ast.FunctionDef)
        }

        has_init = "__init__" in method_names
        has_forward = any(m in method_names for m in _FORWARD_METHOD_NAMES)

        if has_init and has_forward:
            candidates.append(node)

    return candidates


def _extract_weights_and_ops(model_class: ast.ClassDef) -> tuple[list[RawNode], list[RawEdge]]:
    """
    Extract weight initializations and forward-pass operations from a class.

    Returns (nodes, edges) or ([], []) if nothing recognizable is found.
    """
    init_method = next(
        (n for n in model_class.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
        None,
    )
    forward_method = next(
        (
            n for n in model_class.body
            if isinstance(n, ast.FunctionDef) and n.name in _FORWARD_METHOD_NAMES
        ),
        None,
    )

    if not init_method:
        return [], []

    nodes: list[RawNode] = []
    node_id_map: dict[str, str] = {}
    counter = 0

    # 1. Scan __init__ for weight initializations
    weight_attrs: set[str] = set()
    for stmt in ast.walk(init_method):
        if not isinstance(stmt, ast.Assign):
            continue
        for target in stmt.targets:
            if not (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                continue

            attr_name = target.attr
            shape = None
            params = 0

            if isinstance(stmt.value, ast.Call) and _is_weight_init_call(stmt.value, attr_name):
                weight_attrs.add(attr_name)
                shape = _extract_shape_from_args(stmt.value.args)
                if shape:
                    params = int(np.prod(shape))

                counter += 1
                node_id = f"node_{counter}"
                node_id_map[attr_name] = node_id

                nodes.append(RawNode(
                    id=node_id,
                    type="WeightInit",
                    label=f"self.{attr_name}",
                    params=params,
                    output_shape=shape,
                    line_number=stmt.lineno,
                ))
            elif any(attr_name.lower().startswith(p) for p in _WEIGHT_ATTR_PREFIXES):
                weight_attrs.add(attr_name)
                counter += 1
                node_id = f"node_{counter}"
                node_id_map[attr_name] = node_id

                nodes.append(RawNode(
                    id=node_id,
                    type="WeightInit",
                    label=f"self.{attr_name}",
                    params=0,
                    line_number=stmt.lineno,
                ))

    # 2. Scan forward-like method for mathematical operations
    if forward_method:
        for stmt in ast.walk(forward_method):
            op_type = None
            op_label = None

            if isinstance(stmt, ast.Call):
                func = stmt.func
                func_name = ""
                if isinstance(func, ast.Attribute):
                    func_name = func.attr
                elif isinstance(func, ast.Name):
                    func_name = func.id

                func_lower = func_name.lower()
                if func_lower in ("dot", "matmul"):
                    op_type = "MatMul"
                    op_label = func_name
                elif func_lower in ("relu", "maximum"):
                    op_type = "ReLU"
                    op_label = func_name
                elif func_lower in ("sigmoid", "softmax", "tanh", "exp", "log"):
                    op_type = "Activation"
                    op_label = func_name

            elif isinstance(stmt, ast.BinOp) and isinstance(stmt.op, ast.MatMult):
                op_type = "MatMul"
                op_label = "@"

            if op_type:
                counter += 1
                node_id = f"node_{counter}"
                nodes.append(RawNode(
                    id=node_id,
                    type=op_type,
                    label=op_label or op_type,
                    line_number=getattr(stmt, "lineno", None),
                ))

    if not nodes:
        return [], []

    # 3. Connect nodes sequentially in AST appearance order
    edges: list[RawEdge] = []
    for i in range(len(nodes) - 1):
        edges.append(RawEdge(
            source=nodes[i].id,
            target=nodes[i + 1].id,
            is_skip_connection=False,
        ))

    return nodes, edges


def parse_raw_code(file_path: Path) -> RawParseResult:
    """
    Statically inspect a Python file for custom/framework-free model patterns.

    Raises ModelParsingError if no recognizable model structure is found.
    """
    source = file_path.read_text(errors="ignore")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ModelParsingError(f"File is not valid Python: {exc}") from exc

    candidates = _find_candidate_classes(tree)
    if not candidates:
        raise ModelParsingError(
            "Could not identify a recognizable model structure - no class "
            "with __init__ and a forward/call/predict/run method was found."
        )

    # Try candidate classes in reverse order (conventionally main model is defined last)
    for model_class in reversed(candidates):
        nodes, edges = _extract_weights_and_ops(model_class)
        if nodes:
            logger.info(
                "Raw code parse of %s found class=%s, %d nodes, %d edges",
                file_path, model_class.name, len(nodes), len(edges),
            )
            return RawParseResult(
                nodes=nodes,
                edges=edges,
                model_name=model_class.name,
                total_flops=None,
                warnings=[_RAW_CODE_WARNING],
            )

    raise ModelParsingError(
        "Could not identify a recognizable model structure - class was found "
        "but contained no weight initializations or recognized operations."
    )
