"""
engines/pytorch/pretrained_parser.py

Why this file exists:
    Many real-world PyTorch scripts (fine-tuning scripts especially) don't
    define their own nn.Module architecture at all - they load an already-
    built model from HuggingFace's hub via `SomeClass.from_pretrained(...)`.
    Neither torch.fx (needs a locally-defined, instantiable class) nor the
    AST parser (needs a `class X(nn.Module):` definition) can do anything
    with a file like that - there is no architecture written in the source.

    This module handles that case: it statically finds the from_pretrained
    call, resolves which class/module it came from, dynamically loads the
    REAL model object from HuggingFace's hub, and introspects that live
    object's structure directly.

What it does:
    - Scans the file's imports and top-level from_pretrained(...) calls
    - Resolves the class name to its source module (e.g. "transformers")
    - Dynamically imports that class and calls .from_pretrained() with the
      same (literal) arguments the script used
    - Walks the resulting real model with named_modules() to build nodes
    - Builds edges from declaration order (NOT traced execution order -
      this is a real limitation, see module docstring below)

Important limitation (be upfront about this, don't hide it):
    Because the model is never actually run with a forward pass, edges
    here represent module *declaration* order, not confirmed data flow.
    For simple sequential architectures this usually matches. For complex
    encoder-decoder or branching architectures (like TrOCR), it may not
    perfectly reflect real execution order. The result is tagged with
    confidence="static" and a clear warning explaining this.

    This also requires internet access (to download the pretrained
    weights/config from HuggingFace's hub) and the relevant library
    (e.g. `transformers`) to be installed.

How it connects:
    Tried by services/parser_service.py BEFORE torch.fx, whenever the file
    is statically detected to contain a from_pretrained(...) call - since
    torch.fx/AST are guaranteed to fail on such files anyway (no locally
    defined, no-arg-constructible nn.Module class exists in them).
"""

import ast
import importlib
import logging
from pathlib import Path

from app.core.exceptions import ModelParsingError
from app.engines.pytorch.ast_parser import RawEdge, RawNode, RawParseResult

logger = logging.getLogger(__name__)


def _build_import_map(tree: ast.Module) -> dict[str, str]:
    """Map a locally-used name -> the module it was imported from.
    e.g. `from transformers import VisionEncoderDecoderModel`
         -> {"VisionEncoderDecoderModel": "transformers"}
    """
    import_map: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                local_name = alias.asname or alias.name
                import_map[local_name] = node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name.split(".")[0]
                import_map[local_name] = alias.name
    return import_map


def _build_variable_literals(tree: ast.Module) -> dict[str, object]:
    """
    Track simple `variable = <literal>` assignments anywhere in the file,
    e.g. `model_name = "microsoft/trocr-base-handwritten"`. This is a
    very common pattern - people rarely inline the checkpoint string
    directly into from_pretrained(...), they usually assign it to a
    variable first. If a variable is assigned more than once, the last
    assignment found wins (simple last-write heuristic, not true control
    flow analysis).
    """
    literals: dict[str, object] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        value = _literal_value(node.value)
        if value is None:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                literals[target.id] = value
    return literals


def _literal_value(node: ast.expr, variable_literals: dict[str, object] | None = None):
    """Best-effort extraction of a literal Python value from an AST node.
    If the node is a bare variable reference (e.g. `model_name`) and that
    variable was previously assigned a literal elsewhere in the file, that
    resolved value is used. Returns None if no literal value can be
    determined at all - those cases genuinely can't be resolved without
    running the whole script, which we deliberately avoid."""
    if variable_literals and isinstance(node, ast.Name) and node.id in variable_literals:
        return variable_literals[node.id]
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


# Classes matching these substrings load auxiliary objects (tokenizers,
# feature extractors, configs) rather than a neural network architecture.
# A from_pretrained() call on one of these should never be selected as
# "the model" even though it's syntactically identical to a real one.
_NON_MODEL_CLASS_HINTS = (
    "Tokenizer", "Processor", "FeatureExtractor", "Config",
    "ImageProcessor", "Extractor",
)


def _resolve_candidate(
    node: ast.Call, import_map: dict[str, str], variable_literals: dict[str, object]
) -> tuple[str, str, list, dict] | None:
    """Extract (class_name, module_name, args, kwargs) from a single
    `SomeClass.from_pretrained(...)` call node, or None if unresolvable."""
    if not isinstance(node.func.value, ast.Name):
        return None

    class_name = node.func.value.id
    module_name = import_map.get(class_name)
    if module_name is None:
        return None

    args = [_literal_value(a, variable_literals) for a in node.args]
    if any(a is None for a in args):
        return None  # non-literal positional arg, can't resolve safely

    kwargs = {}
    for kw in node.keywords:
        if kw.arg is None:
            return None
        value = _literal_value(kw.value, variable_literals)
        if value is None:
            return None
        kwargs[kw.arg] = value

    return class_name, module_name, args, kwargs


def find_pretrained_call(tree: ast.Module, import_map: dict[str, str]) -> tuple[str, str, list, dict] | None:
    """
    Search for `SomeClass.from_pretrained(...)` calls where SomeClass is a
    locally-imported name, and return the one most likely to be the actual
    model architecture (as opposed to a tokenizer/processor/config, which
    share the exact same from_pretrained(...) call pattern).

    Preference order:
      1. A call assigned to a variable literally named "model"
      2. A call whose class name doesn't match a known non-model pattern
         (Tokenizer/Processor/Config/etc.)
      3. Otherwise, the first resolvable call found (last resort)

    Returns (class_name, module_name, args, kwargs) or None if no
    resolvable from_pretrained call exists at all.
    """
    resolvable: list[tuple[ast.Call, ast.Assign | None, tuple]] = []
    variable_literals = _build_variable_literals(tree)

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "from_pretrained":
            continue
        candidate = _resolve_candidate(node, import_map, variable_literals)
        if candidate is None:
            continue

        # Find the enclosing assignment (if any) so we can check the
        # target variable's name, e.g. `model = X.from_pretrained(...)`.
        assign_target = None
        for parent in ast.walk(tree):
            if isinstance(parent, ast.Assign) and parent.value is node:
                assign_target = parent
                break

        resolvable.append((node, assign_target, candidate))

    if not resolvable:
        return None

    def is_non_model(class_name: str) -> bool:
        return any(hint in class_name for hint in _NON_MODEL_CLASS_HINTS)

    def assigned_to_model_var(assign: ast.Assign | None) -> bool:
        if assign is None:
            return False
        return any(
            isinstance(t, ast.Name) and t.id.lower() == "model"
            for t in assign.targets
        )

    # Preference 1: explicitly assigned to a variable named "model".
    for _, assign, candidate in resolvable:
        if assigned_to_model_var(assign) and not is_non_model(candidate[0]):
            return candidate

    # Preference 2: not a known non-model class.
    for _, _, candidate in resolvable:
        if not is_non_model(candidate[0]):
            return candidate

    # Last resort: whatever was found first.
    return resolvable[0][2]


def has_pretrained_call(file_path: Path) -> bool:
    """Cheap upfront check used by parser_service to decide routing,
    without needing full resolution to succeed."""
    try:
        tree = ast.parse(file_path.read_text(errors="ignore"))
    except SyntaxError:
        return False
    return any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "from_pretrained"
        for n in ast.walk(tree)
    )


def run_pretrained_loader(file_path: Path) -> RawParseResult:
    """
    Load a real pretrained model referenced in file_path via its
    .from_pretrained(...) call, and introspect its actual module tree.
    Raises ModelParsingError on any failure (unresolvable call, import
    error, network/hub error, etc.) so the caller can fall back to AST.
    """
    source = file_path.read_text(errors="ignore")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ModelParsingError(f"File is not valid Python: {exc}") from exc

    import_map = _build_import_map(tree)
    resolved = find_pretrained_call(tree, import_map)
    if resolved is None:
        raise ModelParsingError(
            "Found a from_pretrained(...) call but could not resolve it "
            "statically (non-literal arguments or unrecognized import)."
        )

    class_name, module_name, args, kwargs = resolved
    logger.info("Attempting pretrained load: %s.%s(*%s, **%s)", module_name, class_name, args, kwargs)

    try:
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise ModelParsingError(
            f"Could not import '{class_name}' from '{module_name}': {exc}"
        ) from exc

    try:
        model = model_class.from_pretrained(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - network errors, hub errors, etc. all possible
        raise ModelParsingError(
            f"{class_name}.from_pretrained() failed (requires internet access "
            f"to download from HuggingFace's hub the first time): {exc}"
        ) from exc

    return introspect_loaded_model(model, display_name=f"{class_name} (pretrained)")


def introspect_loaded_model(model, display_name: str) -> RawParseResult:
    """
    Walk an already-instantiated nn.Module's structure via named_modules()
    and build a RawParseResult from it. Separated from run_pretrained_loader
    so this logic can be unit-tested against any in-memory model object,
    independent of the network call that produced it.
    """
    model.eval()

    nodes: list[RawNode] = []
    edges: list[RawEdge] = []

    # Only walk LEAF modules (no children of their own) - matching the
    # granularity fx_parser and ast_parser use, so downstream grouping
    # logic (Phase 3) can treat all three tiers' output consistently.
    for name, submodule in model.named_modules():
        if name == "" or any(True for _ in submodule.children()):
            continue
        params = sum(p.numel() for p in submodule.parameters(recurse=False))
        nodes.append(RawNode(
            id=f"node_{len(nodes) + 1}",
            type=type(submodule).__name__,
            label=name,
            params=params,
            input_shape=None,   # not available - see module docstring
            output_shape=None,  # not available - see module docstring
        ))

    # Edges reflect declaration order only (see limitation in docstring).
    for i in range(len(nodes) - 1):
        edges.append(RawEdge(source=nodes[i].id, target=nodes[i + 1].id))

    if not nodes:
        raise ModelParsingError("Loaded model has no leaf submodules to display.")

    logger.info(
        "Introspected %s: found %d leaf modules (declaration-order edges only)",
        display_name, len(nodes),
    )

    return RawParseResult(
        nodes=nodes,
        edges=edges,
        model_name=display_name,
        warnings=[
            "This model was loaded via from_pretrained() rather than traced. "
            "Layer order reflects declaration order, not confirmed execution "
            "order, and input/output shapes are not available.",
        ],
    )