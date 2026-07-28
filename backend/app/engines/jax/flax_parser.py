"""
engines/jax/flax_parser.py

Why this file exists:
    This is the JAX/Flax equivalent of the PyTorch and TensorFlow parsing
    chains.  It supports:
      - Tier A: Flax nn.Module models (via `nn.tabulate` or manual inspection)
      - Tier B: Haiku models (hk.transform)
      - Tier C: AST-based fallback for raw JAX functional code or when
                dynamic loading fails

    All paths produce a RawParseResult that feeds into the Universal Graph
    pipeline exactly like PyTorch/TF parsers do.

How it connects:
    Called by services/parser_service.py when framework == Framework.JAX.
"""

import ast
import importlib.util
import logging
import sys
import uuid
from pathlib import Path

from app.core.exceptions import ModelParsingError
from app.engines.pytorch.ast_parser import RawEdge, RawNode, RawParseResult

logger = logging.getLogger(__name__)


# ── Module loading ───────────────────────────────────────────


def _load_module_from_path(file_path: Path):
    """Dynamically import a .py file as a standalone module."""
    module_name = f"uploaded_jax_model_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ModelParsingError(f"Could not load {file_path} as a Python module.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        raise ModelParsingError(f"Uploaded file raised an error on import: {exc}") from exc
    finally:
        sys.modules.pop(module_name, None)

    return module


# ── Shape helpers ────────────────────────────────────────────


def _shape_to_int_list(shape) -> list[int] | None:
    """Convert JAX/Flax shapes to list[int], mapping None to -1."""
    if shape is None:
        return None
    if hasattr(shape, "shape"):
        shape = shape.shape
    if not isinstance(shape, (list, tuple)):
        try:
            shape = list(shape)
        except Exception:
            return None
    return [int(d) if d is not None else -1 for d in shape]


# ── Flax detection ───────────────────────────────────────────


def _is_flax_module_class(obj) -> bool:
    """Check if obj is a subclass of flax.linen.Module."""
    if not isinstance(obj, type):
        return False
    try:
        import flax.linen as nn
        if issubclass(obj, nn.Module) and obj is not nn.Module:
            return True
    except Exception:
        pass
    # Fallback: check MRO names
    try:
        for base in obj.__mro__:
            if base.__name__ == "Module" and "flax" in getattr(base, "__module__", ""):
                return True
    except Exception:
        pass
    return False


def _find_flax_modules(module):
    """Find all Flax nn.Module subclasses defined in the uploaded module."""
    return [
        obj for name, obj in vars(module).items()
        if _is_flax_module_class(obj)
    ]


# ── Tier A: Flax nn.Module parser ────────────────────────────


def _parse_flax_module(model_class, file_path: Path) -> RawParseResult:
    """
    Instantiate a Flax module, initialize it with a dummy input,
    then walk the parameter tree and module structure to extract nodes.
    """
    import jax
    import jax.numpy as jnp

    try:
        model = model_class()
    except Exception as exc:
        raise ModelParsingError(
            f"Could not instantiate Flax module '{model_class.__name__}' "
            f"without constructor arguments: {exc}"
        ) from exc

    # Try common input shapes for initialization
    dummy_inputs = [
        jnp.ones((1, 28, 28, 1)),    # MNIST-like (NHWC)
        jnp.ones((1, 32, 32, 3)),    # CIFAR-like (NHWC)
        jnp.ones((1, 224, 224, 3)),  # ImageNet-like (NHWC)
        jnp.ones((1, 784)),           # Flattened MNIST
        jnp.ones((1, 128)),           # Small vector
        jnp.ones((1, 10)),            # Tiny vector
    ]

    params = None
    successful_input = None
    init_error = None

    rng = jax.random.PRNGKey(0)

    for dummy in dummy_inputs:
        try:
            params = model.init(rng, dummy)
            successful_input = dummy
            break
        except Exception as exc:
            init_error = exc
            continue

    if params is None:
        raise ModelParsingError(
            f"Could not initialize Flax module '{model_class.__name__}' "
            f"with any standard input shape. Last error: {init_error}"
        )

    # Try to use nn.tabulate for structured info
    nodes = []
    edges = []
    node_map = {}

    try:
        result = _extract_from_tabulate(model, params, successful_input, rng)
        if result:
            return result
    except Exception as tab_err:
        logger.debug("nn.tabulate extraction failed: %s, falling back to param tree walk", tab_err)

    # Fallback: walk the parameter tree to extract layer info
    _walk_param_tree(params, model_class.__name__, nodes, edges, node_map)

    # Try to get output shape via apply
    try:
        output = model.apply(params, successful_input)
        if hasattr(output, "shape") and nodes:
            nodes[-1] = RawNode(
                id=nodes[-1].id,
                type=nodes[-1].type,
                label=nodes[-1].label,
                params=nodes[-1].params,
                input_shape=nodes[-1].input_shape,
                output_shape=_shape_to_int_list(output.shape),
                flops=0,
                line_number=None,
            )
    except Exception:
        pass

    if not nodes:
        raise ModelParsingError("Flax module produced no parseable layers.")

    return RawParseResult(
        nodes=nodes,
        edges=edges,
        model_name=model_class.__name__,
        total_flops=None,
        warnings=[],
    )


def _extract_from_tabulate(model, params, dummy_input, rng) -> RawParseResult | None:
    """Try to use flax.linen.tabulate to get structured layer info."""
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    import re

    try:
        # Use nn.tabulate to get a structured table
        table_fn = nn.tabulate(model, rng, console_kwargs={"width": 200, "no_color": True})
        table_str = table_fn(dummy_input)
    except Exception:
        return None

    if not table_str or len(table_str) < 10:
        return None

    # Parse the tabulate output to extract layer information
    lines = table_str.strip().split("\n")

    nodes = []
    edges = []
    node_idx = 0

    # Pattern to match layer lines from tabulate output
    # Typical format: "  ModuleName    LayerType    output_shape    param_count"
    for line in lines:
        line = line.strip()
        if not line or line.startswith("─") or line.startswith("=") or line.startswith("Total"):
            continue

        # Try to extract meaningful layer info from each non-header line
        parts = [p.strip() for p in re.split(r'\s{2,}', line) if p.strip()]
        if len(parts) < 2:
            continue

        # Skip header rows
        if any(h in line.lower() for h in ["path", "module", "inputs", "outputs"]):
            continue

        layer_name = parts[0] if parts else f"layer_{node_idx}"
        layer_type = parts[1] if len(parts) > 1 else "Unknown"

        # Try to parse output shape
        out_shape = None
        for part in parts:
            if "(" in part and "," in part:
                try:
                    shape_str = part.strip("() ")
                    dims = [int(d.strip()) for d in shape_str.split(",") if d.strip().lstrip("-").isdigit()]
                    if dims:
                        out_shape = dims
                        break
                except Exception:
                    pass

        # Try to parse param count
        param_count = 0
        for part in reversed(parts):
            cleaned = part.replace(",", "").strip()
            if cleaned.isdigit():
                param_count = int(cleaned)
                break

        node_id = f"node_{node_idx + 1}"
        nodes.append(RawNode(
            id=node_id,
            type=layer_type,
            label=layer_name,
            params=param_count,
            input_shape=None,
            output_shape=out_shape,
            flops=0,
            line_number=None,
        ))

        if node_idx > 0:
            edges.append(RawEdge(
                source=f"node_{node_idx}",
                target=node_id,
                is_skip_connection=False,
            ))

        node_idx += 1

    if not nodes:
        return None

    return RawParseResult(
        nodes=nodes,
        edges=edges,
        model_name=type(model).__name__,
        total_flops=None,
        warnings=[],
    )


def _walk_param_tree(
    params: dict,
    model_name: str,
    nodes: list,
    edges: list,
    node_map: dict,
    prefix: str = "",
    depth: int = 0,
):
    """Recursively walk the Flax parameter tree to build graph nodes."""
    if not isinstance(params, dict):
        return

    # Flax params are typically nested as {"params": {"Dense_0": {"kernel": ..., "bias": ...}}}
    param_dict = params
    if "params" in param_dict and isinstance(param_dict["params"], dict):
        param_dict = param_dict["params"]
    if "batch_stats" in params:
        # Merge batch_stats into the tree for BatchNorm layers
        pass

    for key, value in param_dict.items():
        full_path = f"{prefix}/{key}" if prefix else key

        if isinstance(value, dict):
            # Check if this is a leaf layer (has kernel/bias/scale etc.)
            param_keys = set(value.keys())
            leaf_indicators = {"kernel", "bias", "scale", "embedding"}

            if param_keys & leaf_indicators:
                # This is a leaf layer - extract info
                node_idx = len(nodes) + 1
                node_id = f"node_{node_idx}"

                # Determine layer type from the key name
                layer_type = _infer_layer_type(key, value)

                # Count parameters
                total_params = _count_params(value)

                # Extract shapes
                out_shape = None
                in_shape = None
                if "kernel" in value:
                    kernel = value["kernel"]
                    if hasattr(kernel, "shape"):
                        k_shape = list(kernel.shape)
                        if len(k_shape) == 2:
                            # Dense layer: (in, out)
                            in_shape = [k_shape[0]]
                            out_shape = [k_shape[1]]
                        elif len(k_shape) == 4:
                            # Conv layer: (H, W, C_in, C_out)
                            in_shape = [k_shape[2]]
                            out_shape = [k_shape[3]]
                elif "embedding" in value:
                    emb = value["embedding"]
                    if hasattr(emb, "shape"):
                        e_shape = list(emb.shape)
                        if len(e_shape) == 2:
                            out_shape = [e_shape[1]]

                nodes.append(RawNode(
                    id=node_id,
                    type=layer_type,
                    label=key,
                    params=total_params,
                    input_shape=in_shape,
                    output_shape=out_shape,
                    flops=0,
                    line_number=None,
                ))
                node_map[full_path] = node_id

                # Connect to previous node
                if len(nodes) > 1:
                    edges.append(RawEdge(
                        source=f"node_{node_idx - 1}",
                        target=node_id,
                        is_skip_connection=False,
                    ))
            else:
                # Recurse into nested modules
                _walk_param_tree(value, model_name, nodes, edges, node_map, full_path, depth + 1)


def _infer_layer_type(key: str, params: dict) -> str:
    """Infer the Flax layer type from the parameter key name and structure."""
    key_lower = key.lower()

    if "dense" in key_lower or "linear" in key_lower:
        return "Dense"
    if "conv" in key_lower:
        if "kernel" in params and hasattr(params["kernel"], "shape"):
            ndim = len(params["kernel"].shape)
            if ndim == 4:
                return "Conv"
            elif ndim == 5:
                return "Conv3D"
        return "Conv"
    if "batchnorm" in key_lower or "bn" in key_lower:
        return "BatchNorm"
    if "layernorm" in key_lower or "ln" in key_lower:
        return "LayerNorm"
    if "groupnorm" in key_lower:
        return "GroupNorm"
    if "embed" in key_lower:
        return "Embed"
    if "attention" in key_lower or "attn" in key_lower:
        if "self" in key_lower:
            return "SelfAttention"
        if "multi" in key_lower:
            return "MultiHeadDotProductAttention"
        return "Attention"
    if "dropout" in key_lower:
        return "Dropout"

    # Infer from parameter shapes
    if "kernel" in params:
        kernel = params["kernel"]
        if hasattr(kernel, "shape"):
            if len(kernel.shape) == 2:
                return "Dense"
            elif len(kernel.shape) >= 3:
                return "Conv"

    if "scale" in params and "bias" in params and "kernel" not in params:
        return "LayerNorm"

    if "embedding" in params:
        return "Embed"

    return key.split("_")[0] if "_" in key else key


def _count_params(param_dict: dict) -> int:
    """Count total parameters in a leaf parameter dictionary."""
    total = 0
    for value in param_dict.values():
        if hasattr(value, "size"):
            total += int(value.size)
        elif hasattr(value, "shape"):
            import functools
            import operator
            total += functools.reduce(operator.mul, value.shape, 1)
    return total


# ── Tier B: AST fallback for raw JAX code ────────────────────


def _parse_jax_ast(file_path: Path) -> RawParseResult:
    """
    AST-based fallback parser for JAX/Flax files.
    Extracts layer definitions from class bodies and function calls.
    """
    try:
        source = file_path.read_text(errors="ignore")
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ModelParsingError(f"Syntax error in {file_path.name}: {exc}") from exc

    nodes = []
    edges = []
    model_name = "JAXModel"

    # Known Flax/JAX layer call patterns
    known_layers = {
        "nn.Dense": "Dense",
        "nn.Conv": "Conv",
        "nn.Conv2d": "Conv2D",
        "nn.BatchNorm": "BatchNorm",
        "nn.LayerNorm": "LayerNorm",
        "nn.GroupNorm": "GroupNorm",
        "nn.Dropout": "Dropout",
        "nn.Embed": "Embed",
        "nn.relu": "ReLU",
        "nn.gelu": "GELU",
        "nn.sigmoid": "Sigmoid",
        "nn.tanh": "Tanh",
        "nn.softmax": "Softmax",
        "nn.log_softmax": "LogSoftmax",
        "nn.max_pool": "MaxPool",
        "nn.avg_pool": "AvgPool",
        "nn.SelfAttention": "SelfAttention",
        "nn.MultiHeadDotProductAttention": "MultiHeadDotProductAttention",
        "nn.Sequential": "Sequential",
        # JAX functional
        "jax.nn.relu": "ReLU",
        "jax.nn.gelu": "GELU",
        "jax.nn.sigmoid": "Sigmoid",
        "jax.nn.softmax": "Softmax",
        "jnp.dot": "MatMul",
        "jnp.matmul": "MatMul",
    }

    # Find class definitions that inherit from nn.Module
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Check if it's a Flax module
            is_flax = any(
                _ast_name_matches(base, ("nn.Module", "Module"))
                for base in node.bases
            )
            if is_flax:
                model_name = node.name
                # Extract layers from the class body
                _extract_layers_from_class(node, nodes, edges, known_layers)
                break

    # If no class found, look for top-level layer calls
    if not nodes:
        _extract_layers_from_module(tree, nodes, edges, known_layers)

    if not nodes:
        raise ModelParsingError(
            "No JAX/Flax layers could be extracted from the source file. "
            "Ensure the file defines a Flax nn.Module or uses recognizable JAX operations."
        )

    return RawParseResult(
        nodes=nodes,
        edges=edges,
        model_name=model_name,
        total_flops=None,
        warnings=["Parsed via static analysis (AST). Shapes and parameter counts may be incomplete."],
    )


def _ast_name_matches(node, targets: tuple) -> bool:
    """Check if an AST node matches any of the target dotted names."""
    name = _get_ast_name(node)
    return name in targets


def _get_ast_name(node) -> str:
    """Extract the full dotted name from an AST node."""
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        parent = _get_ast_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    elif isinstance(node, ast.Call):
        return _get_ast_name(node.func)
    return ""


def _extract_layers_from_class(
    class_node: ast.ClassDef,
    nodes: list,
    edges: list,
    known_layers: dict,
):
    """Extract layer definitions from a Flax nn.Module class body."""
    node_idx = 0

    for item in ast.walk(class_node):
        # Look for self.xxx = nn.Dense(...) style assignments
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Attribute) and isinstance(item.value, ast.Call):
                    layer_name = target.attr
                    call_name = _get_ast_name(item.value.func)
                    layer_type = known_layers.get(call_name, call_name.split(".")[-1] if "." in call_name else call_name)

                    if layer_type and layer_type not in ("self",):
                        node_idx += 1
                        node_id = f"node_{node_idx}"
                        params = _extract_ast_layer_params(item.value)

                        nodes.append(RawNode(
                            id=node_id,
                            type=layer_type,
                            label=layer_name,
                            params=0,
                            input_shape=None,
                            output_shape=params.get("features", None),
                            flops=0,
                            line_number=item.lineno,
                        ))

                        if node_idx > 1:
                            edges.append(RawEdge(
                                source=f"node_{node_idx - 1}",
                                target=node_id,
                                is_skip_connection=False,
                            ))

        # Look for nn.Dense(...)(...) style calls in @nn.compact methods
        if isinstance(item, ast.Call):
            call_name = _get_ast_name(item.func)
            if call_name in known_layers:
                layer_type = known_layers[call_name]
                node_idx += 1
                node_id = f"node_{node_idx}"
                
                params_node = item.func if isinstance(item.func, ast.Call) else item
                params = _extract_ast_layer_params(params_node)

                nodes.append(RawNode(
                    id=node_id,
                    type=layer_type,
                    label=f"{layer_type.lower()}_{node_idx}",
                    params=0,
                    input_shape=None,
                    output_shape=params.get("features", None),
                    flops=0,
                    line_number=item.lineno,
                ))

                if node_idx > 1:
                    edges.append(RawEdge(
                        source=f"node_{node_idx - 1}",
                        target=node_id,
                        is_skip_connection=False,
                    ))


def _extract_layers_from_module(
    tree: ast.AST,
    nodes: list,
    edges: list,
    known_layers: dict,
):
    """Extract layers from top-level function definitions (functional JAX style)."""
    node_idx = 0

    for item in ast.walk(tree):
        if isinstance(item, ast.Call):
            call_name = _get_ast_name(item.func)
            if call_name in known_layers:
                layer_type = known_layers[call_name]
                node_idx += 1
                node_id = f"node_{node_idx}"

                nodes.append(RawNode(
                    id=node_id,
                    type=layer_type,
                    label=f"{layer_type.lower()}_{node_idx}",
                    params=0,
                    input_shape=None,
                    output_shape=None,
                    flops=0,
                    line_number=item.lineno,
                ))

                if node_idx > 1:
                    edges.append(RawEdge(
                        source=f"node_{node_idx - 1}",
                        target=node_id,
                        is_skip_connection=False,
                    ))


def _extract_ast_layer_params(call_node: ast.Call) -> dict:
    """Extract keyword arguments from a layer constructor call."""
    params = {}
    for kw in call_node.keywords:
        if kw.arg == "features" and isinstance(kw.value, ast.Constant):
            params["features"] = [int(kw.value.value)]
        elif kw.arg == "kernel_size" and isinstance(kw.value, ast.Tuple):
            params["kernel_size"] = [
                elt.value for elt in kw.value.elts if isinstance(elt, ast.Constant)
            ]
    # Also check positional args (e.g. nn.Dense(256))
    if call_node.args and isinstance(call_node.args[0], ast.Constant):
        params.setdefault("features", [int(call_node.args[0].value)])
    return params


# ── Public entry point ───────────────────────────────────────


def run_jax_parser(file_path: Path) -> RawParseResult:
    """
    Parse a JAX/Flax model file and return a RawParseResult.

    Strategy:
      1. Try dynamic loading + Flax module inspection (Tier A)
      2. Fall back to AST-based parsing (Tier B)
    """
    # Tier A: Dynamic loading
    try:
        try:
            import jax  # noqa: F401
            import flax.linen  # noqa: F401
            jax_available = True
        except ImportError:
            jax_available = False

        if jax_available:
            module = _load_module_from_path(file_path)
            flax_classes = _find_flax_modules(module)

            if flax_classes:
                last_error = None
                for model_class in reversed(flax_classes):
                    try:
                        result = _parse_flax_module(model_class, file_path)
                        logger.info("Parsed JAX/Flax model '%s' via dynamic inspection", model_class.__name__)
                        return result
                    except ModelParsingError as exc:
                        last_error = exc
                        continue

                logger.warning(
                    "All %d Flax modules failed dynamic parsing, falling back to AST. Last: %s",
                    len(flax_classes), last_error,
                )
    except ModelParsingError:
        logger.warning("Dynamic loading failed for %s, falling back to AST", file_path)
    except Exception as exc:
        logger.warning("Unexpected error during dynamic JAX parsing: %s, falling back to AST", exc)

    # Tier B: AST fallback
    try:
        result = _parse_jax_ast(file_path)
        result.warnings.insert(0, "JAX/Flax dynamic parsing was not available; results are from static source analysis.")
        logger.info("Parsed JAX file '%s' via AST fallback", file_path.name)
        return result
    except ModelParsingError:
        raise
    except Exception as exc:
        raise ModelParsingError(f"JAX AST parser failed: {exc}") from exc
