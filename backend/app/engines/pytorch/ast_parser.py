"""
engines/pytorch/ast_parser.py

Why this file exists:
    torch.fx (fx_parser.py) requires the model to actually be instantiated
    and run with a dummy input. Many real-world uploads won't satisfy that
    (missing weight files, custom CUDA ops, broken imports, dynamic control
    flow). This module is the fallback tier: it reads the model's source
    code as plain text structure, without ever executing it, so it works
    even on completely unrunnable projects.

What it does:
    - Finds the first class that subclasses nn.Module
    - Reads its __init__ method to find layer definitions
      (self.conv1 = nn.Conv2d(3, 64, 3))
    - Reads its forward() method to find the call order
      (self.conv1(x) -> self.relu(x) -> ...)
    - Emits the same (raw_nodes, raw_edges) shape that fx_parser.py emits,
      so downstream code (graph/universal_graph.py) doesn't need to know
      or care which parser tier produced the data.

How it connects:
    Called by services/parser_service.py as the fallback when
    fx_parser.run_torch_fx() raises. Its output feeds into
    engines/graph/universal_graph.py exactly like the fx parser's output.
"""

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.core.exceptions import ModelParsingError

logger = logging.getLogger(__name__)


@dataclass
class RawNode:
    id: str
    type: str
    label: str
    params: int = 0
    input_shape: list[int] | None = None
    output_shape: list[int] | None = None
    flops: int = 0
    line_number: int | None = None


@dataclass
class RawEdge:
    source: str
    target: str
    is_skip_connection: bool = False


@dataclass
class RawParseResult:
    nodes: list[RawNode] = field(default_factory=list)
    edges: list[RawEdge] = field(default_factory=list)
    model_name: str = "UnknownModel"
    total_flops: list[str] | int | None = None
    warnings: list[str] = field(default_factory=list)


def _base_is_torch_module(base: ast.expr) -> bool:
    """
    True if a base-class expression refers to torch's Module, regardless of
    how it was imported: nn.Module, torch.nn.Module, Module (bare, from
    `from torch.nn import Module`), or any other alias ending in '.Module'.
    """
    base_name = ast.unparse(base) if hasattr(ast, "unparse") else ""
    return base_name.split(".")[-1] == "Module"


def _is_nn_module_class(node: ast.ClassDef, class_map: dict[str, ast.ClassDef], _seen: set[str] | None = None) -> bool:
    """
    True if this class inherits from torch's Module, either directly
    (class Net(nn.Module)) or transitively through a locally-defined
    custom base class (class Base(nn.Module) -> class Net(Base)).
    _seen guards against infinite recursion on circular/self-referential
    definitions.
    """
    _seen = _seen or set()
    if node.name in _seen:
        return False
    _seen.add(node.name)

    for base in node.bases:
        if _base_is_torch_module(base):
            return True
        base_name = ast.unparse(base) if hasattr(ast, "unparse") else ""
        base_class_name = base_name.split(".")[-1]
        if base_class_name in class_map and _is_nn_module_class(class_map[base_class_name], class_map, _seen):
            return True
    return False


def _find_model_class(tree: ast.Module) -> ast.ClassDef | None:
    all_classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    class_map = {c.name: c for c in all_classes}

    candidates = [n for n in all_classes if _is_nn_module_class(n, class_map)]
    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0]

    # Multiple nn.Module subclasses (common: a GAN's Generator+Discriminator,
    # or a custom BaseModel plus its subclasses). Prefer the "leaf" model:
    # one that (a) has a forward() method, and (b) is not itself used as a
    # base class by another candidate - i.e. the main model, not a shared
    # base class. Falls back to the last-defined candidate (main model is
    # conventionally defined last, after its building blocks).
    base_names_in_use = {
        ast.unparse(base).split(".")[-1]
        for c in candidates
        for base in c.bases
    }
    leaf_candidates = [c for c in candidates if c.name not in base_names_in_use]
    pool = leaf_candidates or candidates

    with_forward = [c for c in pool if any(isinstance(n, ast.FunctionDef) and n.name == "forward" for n in c.body)]
    pool = with_forward or pool

    return pool[-1]


def _extract_layer_definitions(init_method: ast.FunctionDef) -> dict[str, RawNode]:
    """
    Scan __init__ for lines like: self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
    Returns a mapping of attribute name ("conv1") -> RawNode.
    """
    layers: dict[str, RawNode] = {}
    counter = 0

    for stmt in ast.walk(init_method):
        if not isinstance(stmt, ast.Assign):
            continue
        target = stmt.targets[0]
        if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self"):
            continue
        if not isinstance(stmt.value, ast.Call):
            continue

        attr_name = target.attr
        call = stmt.value
        layer_type = ast.unparse(call.func) if hasattr(ast, "unparse") else "Unknown"
        layer_type = layer_type.split(".")[-1]  # nn.Conv2d -> Conv2d

        counter += 1
        layers[attr_name] = RawNode(
            id=f"node_{counter}",
            type=layer_type,
            label=attr_name,
            line_number=stmt.lineno,
        )

    return layers


def _extract_call_order(forward_method: ast.FunctionDef, layers: dict[str, RawNode]) -> list[str]:
    """
    Scan forward() for calls like self.conv1(x), in source order, and
    return the sequence of attribute names as they are invoked.
    """
    order: list[str] = []
    for stmt in ast.walk(forward_method):
        if not isinstance(stmt, ast.Call):
            continue
        func = stmt.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
            if func.attr in layers:
                order.append(func.attr)
    return order


def parse_with_ast(file_path: Path) -> RawParseResult:
    """
    Statically parse a PyTorch model file without executing it.
    Raises ModelParsingError if no nn.Module subclass could be found.
    """
    source = file_path.read_text(errors="ignore")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ModelParsingError(f"File is not valid Python: {exc}") from exc

    model_class = _find_model_class(tree)
    if model_class is None:
        raise ModelParsingError("No nn.Module subclass found in this file.")

    init_method = next(
        (n for n in model_class.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
        None,
    )
    forward_method = next(
        (n for n in model_class.body if isinstance(n, ast.FunctionDef) and n.name == "forward"),
        None,
    )

    warnings: list[str] = []
    layers = _extract_layer_definitions(init_method) if init_method else {}
    if not layers:
        warnings.append("No layer definitions found in __init__.")

    call_order = _extract_call_order(forward_method, layers) if forward_method else []
    if not call_order:
        warnings.append(
            "Could not determine execution order from forward(); "
            "falling back to declaration order."
        )
        call_order = list(layers.keys())

    nodes = [layers[name] for name in call_order if name in layers]
    edges = [
        RawEdge(source=layers[call_order[i]].id, target=layers[call_order[i + 1]].id)
        for i in range(len(call_order) - 1)
    ]

    logger.info(
        "AST parse of %s found class=%s, %d layers, %d edges",
        file_path, model_class.name, len(nodes), len(edges),
    )

    return RawParseResult(
        nodes=nodes,
        edges=edges,
        model_name=model_class.name,
        warnings=warnings,
    )