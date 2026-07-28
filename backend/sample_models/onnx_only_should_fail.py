"""
onnx_only_should_fail.py

A PyTorch model that uses a custom torch.autograd.Function whose backward
pass has no ONNX equivalent. This is used to verify the final fallback:

  torch.fx FAILS  →  ONNX export FAILS  →  AST SUCCEEDS

The custom autograd Function is untraceable by torch.fx AND cannot be
exported to ONNX (ONNX has no way to represent arbitrary Python backward
pass logic). The AST parser is the last resort and should still extract
the nn.Module layer definitions from __init__.
"""
import torch
import torch.nn as nn


class _ExoticOp(torch.autograd.Function):
    """A custom autograd Function with no ONNX kernel registration.

    torch.onnx.export raises a RuntimeError / torch.onnx.errors.UnsupportedOperatorError
    for ops that have no registered ONNX symbolic or fallback, which is the
    case for completely custom autograd Functions.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        # A nonsense operation - what matters is that this is a custom op
        return x * torch.sigmoid(x) + x.abs()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        return grad_output * (torch.sigmoid(x) + x)


class OnnxShouldFailNet(nn.Module):
    """Small model that wraps the exotic custom op.

    torch.fx: fails (symbolic trace cannot handle the custom Function dispatch)
    ONNX export: fails (no ONNX opset representation for _ExoticOp)
    AST: succeeds (reads __init__ and sees fc1, fc2 as nn.Linear layers)
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = _ExoticOp.apply(x)   # <-- This breaks both FX and ONNX
        return self.fc2(x)
