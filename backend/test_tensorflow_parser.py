# backend/test_tensorflow_parser.py
import sys
import tempfile
from pathlib import Path

# Setup path so we can import app modules
sys.path.append(str(Path(__file__).parent))

from app.services.parser_service import parse_project
from app.schemas.graph import Framework, Confidence


# 1. Helper to write code text to a temporary file and parse it
def parse_code_content(code_text: str, filename: str = "model.py"):
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_file = Path(tmpdir) / filename
        temp_file.write_text(code_text)
        graph = parse_project(job_id="test_job", model_file=temp_file)
        return graph


def test_functional_model():
    print("\n=== Running Test 1: Functional API Model ===")
    code = """
import tensorflow as tf
from tensorflow.keras import layers

def create_model():
    inputs = layers.Input(shape=(224, 224, 3), name="input_1")
    x = layers.Conv2D(32, 3, padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    
    # Branch
    branch = layers.Conv2D(32, 3, padding="same", name="conv_branch")(x)
    branch = layers.BatchNormalization(name="bn_branch")(branch)
    
    # Residual merge
    x = layers.Add(name="add1")([x, branch])
    x = layers.ReLU(name="relu2")(x)
    
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    outputs = layers.Dense(10, name="fc")(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="functional_test_model")

model = create_model()
"""
    graph = parse_code_content(code)
    
    # Print nodes
    print("Nodes found:")
    for node in graph.nodes:
        print(f"  - ID: {node.id}, Type: {node.type}, Label: {node.label}, InShape: {node.input_shape}, OutShape: {node.output_shape}, Params: {node.params}")
        
    print("Edges found:")
    for edge in graph.edges:
        print(f"  - {edge.source} -> {edge.target} (is_skip: {edge.is_skip_connection})")

    # Assertions
    labels = [n.label for n in graph.nodes]
    assert "input_1" in labels, "input_1 missing"
    assert "conv1" in labels, "conv1 missing"
    assert "bn1" in labels, "bn1 missing"
    assert "relu1" in labels, "relu1 missing"
    assert "conv_branch" in labels, "conv_branch missing"
    assert "bn_branch" in labels, "bn_branch missing"
    assert "add1" in labels, "add1 missing"
    assert "relu2" in labels, "relu2 missing"
    assert "gap" in labels, "gap missing"
    assert "fc" in labels, "fc missing"
    
    # Verify shape conversion None -> -1
    for node in graph.nodes:
        if node.input_shape:
            assert None not in node.input_shape, f"None shape element found in {node.label} input_shape"
        if node.output_shape:
            assert None not in node.output_shape, f"None shape element found in {node.label} output_shape"
            
    # Find add1 node and its incoming edges
    add_node = next(n for n in graph.nodes if n.label == "add1")
    incoming = [e for e in graph.edges if e.target == add_node.id]
    assert len(incoming) == 2, f"Add layer should have 2 incoming edges, found {len(incoming)}"
    
    # One edge must be a skip connection (the second one)
    skips = [e for e in incoming if e.is_skip_connection]
    non_skips = [e for e in incoming if not e.is_skip_connection]
    assert len(skips) == 1, "Should have exactly one skip connection"
    assert len(non_skips) == 1, "Should have exactly one non-skip connection"
    print("Test 1 PASSED!")


def test_flat_subclassed_model():
    print("\n=== Running Test 2: Flat Subclassed Model (>= 8 layers) ===")
    code = """
import tensorflow as tf
from tensorflow.keras import layers

class FlatModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = layers.Conv2D(32, 3, padding="same", name="conv1")
        self.bn1 = layers.BatchNormalization(name="bn1")
        self.relu1 = layers.ReLU(name="relu1")
        
        self.conv2 = layers.Conv2D(32, 3, padding="same", name="conv2")
        self.bn2 = layers.BatchNormalization(name="bn2")
        self.relu2 = layers.ReLU(name="relu2")
        
        self.conv3 = layers.Conv2D(32, 3, padding="same", name="conv3")
        self.bn3 = layers.BatchNormalization(name="bn3")
        
        self.add = layers.Add(name="add")
        self.relu3 = layers.ReLU(name="relu3")
        
        self.gap = layers.GlobalAveragePooling2D(name="gap")
        self.fc = layers.Dense(10, name="fc")
        
    def call(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        
        # branch starts at x1
        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)
        
        x2 = self.conv3(x2)
        x2 = self.bn3(x2)
        
        # residual merge x1 and x2
        x3 = self.add([x1, x2])
        x3 = self.relu3(x3)
        
        x3 = self.gap(x3)
        return self.fc(x3)

model = FlatModel()
"""
    graph = parse_code_content(code)
    
    # Print nodes and edges
    print("Nodes found:")
    for node in graph.nodes:
        print(f"  - ID: {node.id}, Type: {node.type}, Label: {node.label}, InShape: {node.input_shape}, OutShape: {node.output_shape}, Params: {node.params}")
        
    print("Edges found:")
    for edge in graph.edges:
        print(f"  - {edge.source} -> {edge.target} (is_skip: {edge.is_skip_connection})")

    # Assertions
    labels = [n.label for n in graph.nodes]
    # Check that at least 8 layers are registered
    print(f"Total layers: {len(labels)}")
    assert len(labels) >= 8, f"Should have at least 8 layers, found {len(labels)}"
    
    # Check connectivity order is not scrambled
    # Expected sequential order: conv1 -> bn1 -> relu1 -> conv2 -> bn2 -> relu2 -> conv3 -> bn3 -> add -> relu3 -> gap -> fc
    # Let's map label -> node ID
    node_id_map = {n.label: n.id for n in graph.nodes}
    
    # Verify add has incoming from relu1 and bn3
    add_id = node_id_map["add"]
    relu1_id = node_id_map["relu1"]
    bn3_id = node_id_map["bn3"]
    
    incoming = [e for e in graph.edges if e.target == add_id]
    sources = [e.source for e in incoming]
    assert relu1_id in sources, "relu1 -> add edge missing"
    assert bn3_id in sources, "bn3 -> add edge missing"
    
    # Verify skip connection is marked
    # Whichever comes second in trace (typically bn3, or whichever is marked as skip)
    # The first parent (e.g. relu1 or bn3) is non-skip, the other is skip.
    skips = [e for e in incoming if e.is_skip_connection]
    assert len(skips) == 1, "Should have exactly one skip connection incoming to add"
    print("Test 2 PASSED!")


def test_nested_subclassed_model():
    print("\n=== Running Test 3: Nested Subclassed Model (Composed Block) ===")
    code = """
import tensorflow as tf
from tensorflow.keras import layers

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, filters, name=None):
        super().__init__(name=name)
        self.conv1 = layers.Conv2D(filters, 3, padding="same", name="conv1")
        self.bn1 = layers.BatchNormalization(name="bn1")
        self.relu = layers.ReLU(name="relu")
        self.conv2 = layers.Conv2D(filters, 3, padding="same", name="conv2")
        self.bn2 = layers.BatchNormalization(name="bn2")
        self.add = layers.Add(name="add")
        
    def call(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.add([shortcut, x])

class NestedModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.prep = layers.Conv2D(32, 3, padding="same", name="prep")
        self.res = ResBlock(32, name="res_block")
        self.classifier = layers.Dense(10, name="classifier")
        
    def call(self, x):
        x = self.prep(x)
        x = self.res(x)
        x = layers.GlobalAveragePooling2D()(x)
        return self.classifier(x)

model = NestedModel()
"""
    graph = parse_code_content(code)
    
    # Print nodes and edges
    print("Nodes found:")
    for node in graph.nodes:
        print(f"  - ID: {node.id}, Type: {node.type}, Label: {node.label}, InShape: {node.input_shape}, OutShape: {node.output_shape}, Params: {node.params}")
        
    print("Edges found:")
    for edge in graph.edges:
        print(f"  - {edge.source} -> {edge.target} (is_skip: {edge.is_skip_connection})")

    # Assertions
    labels = [n.label for n in graph.nodes]
    print("Exposed layers:", labels)
    
    # Confirm sub-model layers are exposed individually, e.g. "conv1" of ResBlock should be exposed
    # Keras name prefixing normally makes these like "res_block/conv1" or "conv1"
    sub_layers = [l for l in labels if "conv1" in l or "bn1" in l or "relu" in l or "conv2" in l or "bn2" in l]
    assert len(sub_layers) >= 5, f"Sub-model's internal layers should be exposed individually, found: {sub_layers}"
    
    # Confirm we have the custom block layers in the nodes list and connected
    # We should have prep -> res_block's first layer -> ... -> res_block's add -> global average pooling -> classifier
    # Check that the grouping engine detected residual blocks or groups
    print("Groups found:")
    for grp in graph.groups:
        print(f"  - Group ID: {grp.id}, Label: {grp.label}, Type: {grp.type}, Members: {grp.member_node_ids}")
        
    assert len(graph.groups) > 0, "Grouping engine should have created groups (e.g. residual_block or conv_block)"
    print("Test 3 PASSED!")


def test_pytorch_regression():
    print("\n=== Running Test 4: PyTorch Regression (SimpleCNN) ===")
    pytorch_cnn_path = Path(__file__).parent / "sample_models" / "simple_cnn.py"
    if not pytorch_cnn_path.exists():
        print("Skipping PyTorch regression: sample file simple_cnn.py not found.")
        return
        
    graph = parse_project(job_id="test_pytorch_job", model_file=pytorch_cnn_path)
    print("PyTorch model parsed successfully!")
    print(f"Total nodes: {len(graph.nodes)}")
    print(f"Total edges: {len(graph.edges)}")
    print(f"Total groups: {len(graph.groups)}")
    assert len(graph.nodes) > 0, "Should find nodes in PyTorch model"
    assert graph.meta.framework == Framework.PYTORCH, "Framework should be PyTorch"
    print("Test 4 PASSED!")


if __name__ == "__main__":
    try:
        test_functional_model()
        test_flat_subclassed_model()
        test_nested_subclassed_model()
        test_pytorch_regression()
        print("\nALL TESTS PASSED SUCCESSFULLY! No regressions detected.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nTEST SUITE FAILED: {e}")
        sys.exit(1)
