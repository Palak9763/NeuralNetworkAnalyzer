"""
engines/tensorflow/keras_parser.py

Why this file exists:
    This is the TensorFlow/Keras equivalent of the PyTorch parsing chain.
    It supports both Functional/Sequential API models (Tier A) and Subclassed models (Tier B),
    mapping both to the Universal Graph JSON contract by outputting RawParseResult.
"""
import importlib.util
import logging
import sys
import types
import uuid
from pathlib import Path

from neuralviz._vendored.core.exceptions import ModelParsingError
from neuralviz._vendored.engines.pytorch.ast_parser import RawEdge, RawNode, RawParseResult

logger = logging.getLogger(__name__)

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
    except Exception as exc:  # noqa: BLE001
        raise ModelParsingError(f"Uploaded file raised an error on import: {exc}") from exc
    finally:
        sys.modules.pop(module_name, None)

    return module

def _is_keras_model_class(obj) -> bool:
    if not isinstance(obj, type):
        return False
    try:
        import tensorflow as tf
        if issubclass(obj, tf.keras.Model) and obj is not tf.keras.Model:
            return True
    except Exception:
        pass
    try:
        import keras
        if issubclass(obj, keras.Model) and obj is not keras.Model:
            return True
    except Exception:
        pass
    try:
        for base in obj.__mro__:
            if base.__name__ == "Model" and base.__module__.startswith(("keras", "tensorflow.keras")):
                return True
    except Exception:
        pass
    return False

def _is_keras_model_instance(obj) -> bool:
    try:
        import tensorflow as tf
        if isinstance(obj, tf.keras.Model):
            return True
    except Exception:
        pass
    try:
        import keras
        if isinstance(obj, keras.Model):
            return True
    except Exception:
        pass
    return False

def _find_keras_model(module):
    """Search module variables for (a) already built Keras model instances,
    or (b) Keras Model subclasses (instantiated in reverse order)."""
    # 1. Look for instances first
    model_instances = [
        obj for name, obj in vars(module).items()
        if _is_keras_model_instance(obj)
    ]
    if model_instances:
        for model in reversed(model_instances):
            is_functional = type(model).__name__ in ("Functional", "Sequential")
            return model, is_functional

    # 2. Look for subclass classes
    model_classes = [
        obj for name, obj in vars(module).items()
        if _is_keras_model_class(obj) and obj.__name__ not in ("Functional", "Sequential")
    ]
    if not model_classes:
        raise ModelParsingError("No tf.keras.Model instances or subclasses found in uploaded file.")

    last_error = None
    for model_class in reversed(model_classes):
        try:
            model = model_class()
            is_functional = type(model).__name__ in ("Functional", "Sequential")
            return model, is_functional
        except Exception as exc:
            last_error = exc
            continue

    raise ModelParsingError(
        f"None of the {len(model_classes)} tf.keras.Model subclasses found could be "
        f"instantiated without constructor arguments. Last error: {last_error}"
    )

def _shape_to_int_list(shape):
    """Convert Keras shapes to list[int], mapping None/unknown to -1 to avoid Pydantic crash."""
    if shape is None:
        return None

    # Handle lists/tuples of shapes (take first shape)
    if isinstance(shape, list) and len(shape) > 0 and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    elif hasattr(shape, "as_list"):
        # For TensorFlow TensorShape objects
        try:
            shape = shape.as_list()
        except Exception:
            pass

    if not isinstance(shape, (list, tuple)):
        return None

    out = []
    for dim in shape:
        if dim is None:
            out.append(-1)
        elif hasattr(dim, "value"):
            val = dim.value
            out.append(val if val is not None else -1)
        elif isinstance(dim, int):
            out.append(dim)
        else:
            try:
                out.append(int(dim))
            except (ValueError, TypeError):
                out.append(-1)
    return out

def _collect_leaf_layers(layer, seen=None):
    """Recursively collect leaf layers, descending into nested models/layers.

    In Keras 3, tf.keras.Model subclasses expose .layers (a public list),
    but plain tf.keras.layers.Layer subclasses (e.g. a custom ResBlock defined
    as `class ResBlock(layers.Layer)`) only expose ._layers (a private list).
    We must check both to recurse into any nested composite layer.
    """
    if seen is None:
        seen = set()

    if id(layer) in seen:
        return []
    seen.add(id(layer))

    # --- Try .layers first (tf.keras.Model subclasses) ---
    children = getattr(layer, "layers", None)
    if children:
        leaves = []
        for sub_layer in children:
            leaves.extend(_collect_leaf_layers(sub_layer, seen))
        return leaves

    private_children = getattr(layer, "_layers", None)
    if private_children:
        real_sub_layers = [
            c for c in private_children
            if hasattr(c, "call") and c is not layer
        ]
        if real_sub_layers:
            leaves = []
            for sub_layer in real_sub_layers:
                leaves.extend(_collect_leaf_layers(sub_layer, seen))
            return leaves

    return [layer]

def _parse_functional_model(model) -> RawParseResult:
    """Tier A extraction directly from Keras Functional/Sequential graph connectivity."""
    nodes = []
    edges = []
    node_id_by_layer_name = {}

    for idx, layer in enumerate(model.layers):
        node_id = f"node_{idx+1}"
        node_id_by_layer_name[layer.name] = node_id

        out_shape = None
        try:
            if hasattr(layer, "output") and hasattr(layer.output, "shape"):
                out_shape = _shape_to_int_list(layer.output.shape)
            elif hasattr(layer, "output_shape"):
                out_shape = _shape_to_int_list(layer.output_shape)
        except Exception:
            pass

        in_shape = None
        try:
            if hasattr(layer, "input") and hasattr(layer.input, "shape"):
                in_shape = _shape_to_int_list(layer.input.shape)
            elif hasattr(layer, "input_shape"):
                in_shape = _shape_to_int_list(layer.input_shape)
        except Exception:
            pass

        try:
            params = layer.count_params()
        except Exception:
            params = 0

        nodes.append(RawNode(
            id=node_id,
            type=type(layer).__name__,
            label=layer.name,
            params=params,
            input_shape=in_shape,
            output_shape=out_shape,
            flops=0,
            line_number=None,
        ))

    # 2. Extract connectivity and detect skip connections
    for layer in model.layers:
        target_name = layer.name
        target_id = node_id_by_layer_name[target_name]

        # Extract parent operation names
        parent_names = []
        if hasattr(layer, "_inbound_nodes") and layer._inbound_nodes:
            node = layer._inbound_nodes[0]
            if hasattr(node, "parent_nodes") and node.parent_nodes:
                for p in node.parent_nodes:
                    if hasattr(p, "operation") and p.operation:
                        parent_names.append(p.operation.name)
                    elif hasattr(p, "node_layer"):
                        parent_names.append(p.node_layer.name)

        # De-duplicate while preserving order
        valid_parents = [p for p in parent_names if p in node_id_by_layer_name]
        unique_parents = []
        for p in valid_parents:
            if p not in unique_parents:
                unique_parents.append(p)

        # Check for skip connection merge point
        is_merge = False
        if type(layer).__name__ in ("Add", "Concatenate", "Average", "Maximum", "Minimum") and len(unique_parents) >= 2:
            is_merge = True

        for i, p_name in enumerate(unique_parents):
            source_id = node_id_by_layer_name[p_name]
            is_skip = is_merge and (i > 0)
            edges.append(RawEdge(
                source=source_id,
                target=target_id,
                is_skip_connection=is_skip,
            ))

    if not nodes:
        raise ModelParsingError("Keras Functional/Sequential trace produced no nodes.")

    model_name = getattr(model, "name", "FunctionalModel")
    if model_name in ("model", "sequential", "functional"):
        model_name = type(model).__name__

    return RawParseResult(
        nodes=nodes,
        edges=edges,
        model_name=model_name,
        total_flops=None,
        warnings=[],
    )

def _parse_subclassed_model(model) -> RawParseResult:
    """Tier B extraction via monkey-patching leaf layer calls and tracking tensors."""
    import tensorflow as tf

    leaf_layers = _collect_leaf_layers(model)
    if not leaf_layers:
        raise ModelParsingError("No leaf layers found in subclassed model.")

    original_calls = {}
    call_log = []
    captured_tensors = []  # Keep-alive list for Python GC safety

    def make_patched_call(layer, layer_index):
        orig_call = layer.call

        def patched_call(self, *args, **kwargs):
            output = orig_call(*args, **kwargs)

            # Record input tensor ids and keep alive
            input_tensor_ids = []

            def collect_tensor_ids(obj):
                if tf.is_tensor(obj) or (hasattr(obj, "shape") and hasattr(obj, "dtype")):
                    input_tensor_ids.append(id(obj))
                    captured_tensors.append(obj)
                elif isinstance(obj, (list, tuple)):
                    for item in obj:
                        collect_tensor_ids(item)
                elif isinstance(obj, dict):
                    for item in obj.values():
                        collect_tensor_ids(item)

            for arg in args:
                collect_tensor_ids(arg)
            for kwarg_val in kwargs.values():
                collect_tensor_ids(kwarg_val)

            # Extract output shapes
            out_shape = None
            try:
                if hasattr(output, "shape"):
                    out_shape = _shape_to_int_list(output.shape)
                elif isinstance(output, (list, tuple)) and output and hasattr(output[0], "shape"):
                    out_shape = _shape_to_int_list(output[0].shape)
            except Exception:
                pass

            # Record output tensor ids and keep alive
            output_tensor_ids = []
            if isinstance(output, (list, tuple)):
                for out_item in output:
                    if tf.is_tensor(out_item) or (hasattr(out_item, "shape") and hasattr(out_item, "dtype")):
                        output_tensor_ids.append(id(out_item))
                        captured_tensors.append(out_item)
            else:
                if tf.is_tensor(output) or (hasattr(output, "shape") and hasattr(output, "dtype")):
                    output_tensor_ids.append(id(output))
                    captured_tensors.append(output)

            # Extract input shapes
            in_shape = None
            try:
                if args and (tf.is_tensor(args[0]) or (hasattr(args[0], "shape") and hasattr(args[0], "dtype"))):
                    in_shape = _shape_to_int_list(args[0].shape)
                elif args and isinstance(args[0], (list, tuple)) and args[0] and hasattr(args[0][0], "shape"):
                    in_shape = _shape_to_int_list(args[0][0].shape)
            except Exception:
                pass

            try:
                params = layer.count_params()
            except Exception:
                params = 0

            call_log.append({
                "layer_name": layer.name,
                "layer_type": type(layer).__name__,
                "input_tensor_ids": input_tensor_ids,
                "output_tensor_ids": output_tensor_ids,
                "input_shape": in_shape,
                "output_shape": out_shape,
                "params": params,
            })
            return output

        return types.MethodType(patched_call, layer), orig_call

    # Monkey-patch every leaf layer
    for idx, layer in enumerate(leaf_layers):
        patched, orig = make_patched_call(layer, idx)
        original_calls[layer] = orig
        layer.call = patched

    # Force eager execution to get stable Python object ids for tensors
    orig_eager = tf.config.functions_run_eagerly()
    tf.config.run_functions_eagerly(True)

    try:
        dummy_input = tf.zeros((1, 224, 224, 3))
        dummy_input_id = id(dummy_input)
        captured_tensors.append(dummy_input)

        # Run forward pass
        _ = model(dummy_input)
    except Exception as exc:
        raise ModelParsingError(f"Failed during subclassed model dummy forward pass: {exc}") from exc
    finally:
        # Restore original functions eager configuration
        tf.config.run_functions_eagerly(orig_eager)
        # Restore original calls
        for layer, orig in original_calls.items():
            layer.call = orig

    # Keep only the first occurrence of each layer name (handling retracing / duplicate passes)
    unique_calls = []
    seen_names = set()
    for call in call_log:
        name = call["layer_name"]
        if name not in seen_names:
            seen_names.add(name)
            unique_calls.append(call)

    if not unique_calls:
        raise ModelParsingError("Model execution traced no leaf layer calls.")

    # Reconstruct nodes
    nodes = []
    node_id_by_layer_name = {}
    for idx, call in enumerate(unique_calls):
        node_id = f"node_{idx+1}"
        node_id_by_layer_name[call["layer_name"]] = node_id

        nodes.append(RawNode(
            id=node_id,
            type=call["layer_type"],
            label=call["layer_name"],
            params=call["params"],
            input_shape=call["input_shape"],
            output_shape=call["output_shape"],
            flops=0,
            line_number=None,
        ))

    # Reconstruct edges
    tensor_source_map = {}
    tensor_source_map[dummy_input_id] = "__INPUT__"

    for call in call_log:
        name = call["layer_name"]
        for out_id in call["output_tensor_ids"]:
            tensor_source_map[out_id] = name

    parents_map = {}
    for call in unique_calls:
        target_name = call["layer_name"]
        parents_map[target_name] = []
        for inp_id in call["input_tensor_ids"]:
            source_name = tensor_source_map.get(inp_id)
            if source_name and source_name != "__INPUT__" and source_name != target_name:
                if source_name not in parents_map[target_name]:
                    parents_map[target_name].append(source_name)

    edges = []
    for call in unique_calls:
        target_name = call["layer_name"]
        target_id = node_id_by_layer_name[target_name]

        parent_names = parents_map.get(target_name, [])
        valid_parents = [p for p in parent_names if p in node_id_by_layer_name]

        is_merge = False
        if call["layer_type"] in ("Add", "Concatenate", "Average", "Maximum", "Minimum") and len(valid_parents) >= 2:
            is_merge = True

        for i, p_name in enumerate(valid_parents):
            source_id = node_id_by_layer_name[p_name]
            is_skip = is_merge and (i > 0)
            edges.append(RawEdge(
                source=source_id,
                target=target_id,
                is_skip_connection=is_skip,
            ))

    model_name = type(model).__name__
    return RawParseResult(
        nodes=nodes,
        edges=edges,
        model_name=model_name,
        total_flops=None,
        warnings=[],
    )

def run_keras_parser(file_path: Path) -> RawParseResult:
    """Trace a TensorFlow/Keras model file and return a RawParseResult."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ModelParsingError("TensorFlow is not installed in this environment.") from exc

    module = _load_module_from_path(file_path)
    model, is_functional = _find_keras_model(module)

    if is_functional:
        try:
            return _parse_functional_model(model)
        except Exception as exc:
            raise ModelParsingError(f"Keras Functional parser failed: {exc}") from exc
    else:
        try:
            return _parse_subclassed_model(model)
        except Exception as exc:
            raise ModelParsingError(f"Keras Subclassed parser failed: {exc}") from exc
