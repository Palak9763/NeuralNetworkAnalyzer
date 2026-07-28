"""
dynamic_control_flow_net.py

A PyTorch model whose forward() contains a tensor-shape-dependent if-branch
that makes torch.fx symbolic_trace fail (FX requires all control flow to be
traceable as graph nodes, which Tensor-dependent Python conditionals are not).

This file is used to verify that the ONNX fallback tier activates and
succeeds where torch.fx fails.

Expected parse path:  torch.fx FAILS  →  ONNX export SUCCEEDS
"""
import torch
import torch.nn as nn


class DynamicControlFlowNet(nn.Module):
    """A small network with a tensor-dependent if-branch in forward().

    torch.fx.symbolic_trace raises TraceError when evaluating `if x.size(2) > 100:`
    because x.size(2) is a Proxy object during symbolic tracing.
    ONNX export handles this by running the model with a concrete input.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2a = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # branch A
        self.conv2b = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # branch B
        self.conv3 = nn.Conv2d(32, 10, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        # Tensor-dependent control flow: FX symbolic_trace cannot evaluate
        # a Proxy object in a Python if statement, raising TraceError.
        if x.size(2) > 100:
            x = self.relu(self.conv2a(x))
        else:
            x = self.relu(self.conv2b(x))
        return self.conv3(x)
