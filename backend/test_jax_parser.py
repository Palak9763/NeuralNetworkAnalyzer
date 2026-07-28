"""Test JAX/Flax parser with a sample model."""
import sys
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, ".")

from pathlib import Path
from app.engines.jax.flax_parser import run_jax_parser
from app.engines.detector.framework_detector import detect_framework

# Create a test Flax model file
test_model = '''
import jax
import jax.numpy as jnp
import flax.linen as nn

class SimpleCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x

model = SimpleCNN()
'''

test_file = Path("test_jax_model.py")
test_file.write_text(test_model)

try:
    # Test framework detection
    fw = detect_framework(test_file)
    print(f"Detected framework: {fw}")

    # Test parsing
    result = run_jax_parser(test_file)
    print(f"Model: {result.model_name}")
    print(f"Nodes: {len(result.nodes)}")
    print(f"Edges: {len(result.edges)}")
    print(f"Warnings: {len(result.warnings)}")
    print("Layers:")
    for n in result.nodes:
        # Sanitize label for Windows console
        label = n.label.encode("ascii", errors="replace").decode()
        ntype = n.type.encode("ascii", errors="replace").decode()
        print(f"  {n.id}: {ntype} | {label} | params={n.params} out={n.output_shape}")
    print(f"Edges: {len(result.edges)}")
    print("PASSED - JAX parser test")
except Exception as exc:
    print(f"FAILED - JAX parser test: {exc}")
    import traceback
    traceback.print_exc()
finally:
    test_file.unlink(missing_ok=True)
