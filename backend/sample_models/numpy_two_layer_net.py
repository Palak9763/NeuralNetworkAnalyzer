"""
numpy_two_layer_net.py

A hand-written 2-layer NumPy neural network with no deep learning framework
imports. Used to verify Feature 2 (Custom / Raw-Code AST Pattern-Matching).

Expected parse path:
  detect_framework() -> Framework.UNKNOWN
  parser_service -> parse_raw_code() -> SUCCEEDS (finds W1, b1, W2, b2 + MatMul/Activation ops)
"""
import numpy as np


class TwoLayerNumPyNet:
    """A from-scratch 2-layer neural network using NumPy only."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 128, output_dim: int = 10):
        self.W1 = np.random.randn(784, 128)
        self.b1 = np.zeros(128)
        self.W2 = np.random.randn(128, 10)
        self.b2 = np.zeros(10)

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        # Layer 1
        h1 = np.dot(x, self.W1) + self.b1
        a1 = np.maximum(0, h1)  # ReLU
        # Layer 2
        logits = a1 @ self.W2 + self.b2
        probs = self._softmax(logits)
        return probs
