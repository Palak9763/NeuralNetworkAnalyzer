"""
neuralviz/text_render.py

Renders a UniversalGraph as a clean ASCII diagram to stdout.

Output structure:
    Model: SimpleCNN  (pytorch, traced)
    ─────────────────────────────────────────────────────────
    [ConvBlock]
      conv1    Conv2d        →  16×224×224    (448 params)
      relu     ReLU          →  16×224×224
      pool     MaxPool2d     →  16×112×112
    [ConvBlock]
      conv2    Conv2d        →  32×112×112    (4,640 params)
      relu     ReLU          →  32×112×112
      pool     MaxPool2d     →  32×56×56
    flatten    Flatten       →  100352
    fc         Linear        →  10            (1,003,530 params)
    ─────────────────────────────────────────────────────────
    Total layers: 8   Total params: 1,008,618   FLOPs: 163.98M

Design choices:
    - Colors via colorama if available; degrades to plain text silently.
    - Skip connections shown as "(+ residual from <label>)" annotation at
      the merge node, since ASCII can't draw curved arrows.
    - Groups from graph.groups are shown as bracketed headers with member
      nodes indented underneath.
    - Stages (parent_group_id set) are shown with an outer Stage header
      wrapping the repeated block headers.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuralviz._vendored.schemas.graph import UniversalGraph


# ── Safe character helpers ─────────────────────────────────────────────────────

def _can_encode(ch: str) -> bool:
    """Return True if stdout's encoding can represent the given character."""
    # colorama on Windows wraps sys.stdout; check the underlying buffer too
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        buf = getattr(sys.stdout, "buffer", None)
        if buf:
            enc = getattr(buf, "encoding", enc) or enc
    except Exception:
        pass
    try:
        ch.encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


# Pre-compute safe chars based on actual terminal capability
_SEP_CHAR  = "\u2500" if _can_encode("\u2500") else "-"   # ─ or -
_SHAPE_SEP = "\u00d7" if _can_encode("\u00d7") else "x"   # × or x
_ARROW     = "\u2192" if _can_encode("\u2192") else "->"  # → or ->

# ── Terminal width ────────────────────────────────────────────────────────────

def _term_width() -> int:
    try:
        return min(os.get_terminal_size().columns, 100)
    except OSError:
        return 72


def _separator(width: int) -> str:
    return _SEP_CHAR * width


# ── Color support ─────────────────────────────────────────────────────────────

def _supports_color() -> bool:
    """Return True if stdout looks like a colour-capable terminal."""
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    if os.environ.get("TERM") in ("dumb", ""):
        return False
    if os.environ.get("NO_COLOR"):
        return False
    return True


class _Colors:
    RESET = ""
    BOLD = ""
    DIM = ""
    # Layer-type colours (matching the web frontend's legend)
    CONV = ""       # blue
    NORM = ""       # yellow
    ACTIVATION = "" # green
    POOL = ""       # purple
    LINEAR = ""     # red
    SKIP = ""       # cyan (for skip-connection annotations)
    HEADER = ""     # bright white
    SEPARATOR = ""  # dark grey

    def __init__(self):
        if not _supports_color():
            return
        try:
            import colorama
            colorama.init()
            F = colorama.Fore
            S = colorama.Style
            B = colorama.Back
            self.RESET = S.RESET_ALL
            self.BOLD = S.BRIGHT
            self.DIM = S.DIM
            self.CONV = F.BLUE
            self.NORM = F.YELLOW
            self.ACTIVATION = F.GREEN
            self.POOL = F.MAGENTA
            self.LINEAR = F.RED
            self.SKIP = F.CYAN
            self.HEADER = S.BRIGHT
            self.SEPARATOR = F.WHITE + S.DIM
        except ImportError:
            pass  # no color — all fields stay ""


C = _Colors()

# ── Layer type → colour mapping ───────────────────────────────────────────────

_TYPE_COLOR: dict[str, str] = {}

def _color_for_type(layer_type: str) -> str:
    t = layer_type.lower()
    if any(k in t for k in ("conv",)):
        return C.CONV
    if any(k in t for k in ("batchnorm", "layernorm", "groupnorm", "norm")):
        return C.NORM
    if any(k in t for k in ("relu", "gelu", "selu", "elu", "sigmoid", "tanh",
                              "activation", "softmax", "hardswish", "silu")):
        return C.ACTIVATION
    if any(k in t for k in ("pool",)):
        return C.POOL
    if any(k in t for k in ("linear", "dense", "fc", "fullyconnected",
                              "embedding", "classifier")):
        return C.LINEAR
    return ""


# ── Shape formatting ──────────────────────────────────────────────────────────


def _fmt_shape(shape: list[int] | None) -> str:
    if not shape:
        return ""
    dims = shape
    if dims and dims[0] == 1:
        dims = dims[1:]
    if len(dims) == 1:
        return str(dims[0])
    return _SHAPE_SEP.join(str(d) for d in dims)


def _fmt_params(p: int) -> str:
    if p == 0:
        return ""
    if p >= 1_000_000:
        return f"({p / 1e6:,.2f}M params)"
    if p >= 1_000:
        return f"({p:,} params)"
    return f"({p} params)"


def _fmt_flops(f: int | None) -> str:
    if f is None:
        return ""
    if f >= 1_000_000_000:
        return f"{f / 1e9:.2f}G"
    if f >= 1_000_000:
        return f"{f / 1e6:.2f}M"
    if f >= 1_000:
        return f"{f / 1e3:.2f}k"
    return str(f)


# ── Core render logic ─────────────────────────────────────────────────────────

def render(graph: "UniversalGraph") -> None:
    """Print the graph as a styled ASCII diagram to stdout."""
    width = _term_width()
    sep = C.SEPARATOR + _separator(width) + C.RESET

    # Header
    fw = graph.meta.framework.value if hasattr(graph.meta.framework, "value") else str(graph.meta.framework)
    conf = graph.meta.confidence.value if hasattr(graph.meta.confidence, "value") else str(graph.meta.confidence)
    print()
    print(f"{C.BOLD}Model: {graph.model_name}{C.RESET}  "
          f"{C.DIM}({fw}, {conf}){C.RESET}")
    print(sep)

    # Build lookup structures
    node_by_id = {n.id: n for n in graph.nodes}
    group_by_id = {g.id: g for g in graph.groups}

    # Map node → group, group → members
    node_to_group: dict[str, str] = {n.id: n.group_id for n in graph.nodes if n.group_id}
    # Top-level groups (no parent) in node order
    node_order = {n.id: i for i, n in enumerate(graph.nodes)}

    def group_first_node_order(g) -> int:
        if not g.member_node_ids:
            return 9999
        return node_order.get(g.member_node_ids[0], 9999)

    # Groups that are direct children of a Stage are rendered inside the Stage header
    groups_with_parent = {g.id for g in graph.groups if g.parent_group_id}

    # Collect stage groups
    stage_groups = [g for g in graph.groups if g.type.value == "stage"]
    top_groups = sorted(
        [g for g in graph.groups if g.id not in groups_with_parent and g.type.value != "stage"],
        key=group_first_node_order,
    )

    # Build skip-connection source labels for annotation
    skip_sources: dict[str, list[str]] = {}  # target_id -> [source_label, ...]
    for edge in graph.edges:
        if edge.is_skip_connection:
            src_label = node_by_id[edge.source].label if edge.source in node_by_id else edge.source
            skip_sources.setdefault(edge.target, []).append(src_label)

    # Track which nodes have been printed
    printed: set[str] = set()

    def print_node(node, indent: str = "") -> None:
        if node.id in printed:
            return
        printed.add(node.id)

        type_color = _color_for_type(node.type)
        label_col = 14
        type_col = 20
        shape_col = 16

        label_str = node.label.ljust(label_col)
        type_str = (type_color + node.type + C.RESET).ljust(type_col + len(type_color) + len(C.RESET))
        shape_str = ""
        if node.output_shape:
            shape_str = (_ARROW + "  " + _fmt_shape(node.output_shape)).ljust(shape_col)

        params_str = ""
        if node.params:
            params_str = f"  {C.DIM}{_fmt_params(node.params)}{C.RESET}"

        skip_str = ""
        if node.id in skip_sources:
            sources = ", ".join(skip_sources[node.id])
            skip_str = f"  {C.SKIP}(+ residual from {sources}){C.RESET}"

        print(f"{indent}{label_str}  {type_str}  {shape_str}{params_str}{skip_str}")

    def print_group_header(g, extra_indent: str = "") -> None:
        group_type = g.type.value if hasattr(g.type, "value") else str(g.type)
        print(f"{extra_indent}{C.HEADER}[{g.label}]{C.RESET}")

    # Iterate through nodes in order, grouping them
    rendered_groups: set[str] = set()
    i = 0
    nodes_in_order = list(graph.nodes)

    # Build set of nodes belonging to each top-level group
    group_members: dict[str, set[str]] = {}
    for g in graph.groups:
        group_members[g.id] = set(g.member_node_ids)

    for node in nodes_in_order:
        gid = node_to_group.get(node.id)

        if gid and gid not in rendered_groups:
            g = group_by_id.get(gid)
            if not g:
                print_node(node)
                continue

            # Check if this group is inside a Stage
            stage_indent = ""
            if g.parent_group_id and g.parent_group_id in group_by_id:
                stage_g = group_by_id[g.parent_group_id]
                if stage_g.id not in rendered_groups:
                    # Print Stage header once
                    print(f"{C.BOLD}[{stage_g.label}]{C.RESET}")
                    rendered_groups.add(stage_g.id)
                stage_indent = "  "

            print_group_header(g, extra_indent=stage_indent)
            rendered_groups.add(gid)

            # Print all members of this group
            for member_id in g.member_node_ids:
                if member_id in node_by_id:
                    print_node(node_by_id[member_id], indent=stage_indent + "  ")

        elif not gid:
            print_node(node)

    print(sep)

    # Footer
    total_params = graph.meta.total_params
    total_layers = graph.meta.total_layers
    flops_str = _fmt_flops(graph.meta.flops)

    footer_parts = [
        f"Total layers: {C.BOLD}{total_layers}{C.RESET}",
        f"Total params: {C.BOLD}{total_params:,}{C.RESET}",
    ]
    if flops_str:
        footer_parts.append(f"FLOPs: {C.BOLD}{flops_str}{C.RESET}")

    print("  ".join(footer_parts))
    print()
