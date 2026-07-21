"""
engines/graph/grouping_engine.py

Why this file exists:
    This is Phase 3 - the single most valuable and hardest stage in the
    whole pipeline. Without it, every individual op (Conv2d, BatchNorm2d,
    ReLU...) renders as its own separate box, so a real model like
    ResNet50 shows as a long flat chain of 100+ boxes instead of a clean
    diagram with "Stage 1 (3 blocks)" containers like the reference UI.

    Deliberately operates on the UniversalGraph (nodes + edges), not on
    any framework-specific data - this is what lets grouping work
    identically regardless of which tier (torch.fx, AST, pretrained
    loader, or a future TensorFlow/JAX parser) produced the graph.

What it does, in order:
    1. Residual block detection: uses is_skip_connection edges (see
       fx_parser.py's predecessor resolution) to find each skip's
       "merge point", then walks the main-path chain between the skip's
       source and that merge point to find the block's full membership.
    2. Simple sequential grouping: any remaining un-grouped run of nodes
       with in-degree==1/out-degree==1 (a straight, unbranched chain -
       e.g. Conv2d -> BatchNorm2d -> ReLU) gets collapsed into one
       "ConvBlock".
    3. Stage detection: consecutive residual blocks with an identical
       member-type signature (e.g. three BasicBlocks in a row) get
       wrapped in a single "Stage" container group with repeat_count set,
       matching the "Stage 1 (3 blocks)" pattern from the reference UI.

Known limitation (real, not hidden):
    Step 1's main-path walk requires each intermediate node to have
    exactly one normal (non-skip) outgoing edge. This correctly handles
    standard single-branch residual blocks (BasicBlock, Bottleneck-style).
    Models with more exotic branching inside a single block (e.g. multiple
    parallel skip paths, Inception-style multi-branch blocks) will not be
    grouped by step 1 and their nodes are left ungrouped rather than
    grouped incorrectly - a conservative, honest fallback.

How it connects:
    Called by services/parser_service.py as the final step before
    returning a response, right after engines/graph/universal_graph.py
    builds the initial (ungrouped) UniversalGraph.
"""

import logging
from collections import defaultdict

from app.schemas.graph import GraphGroup, GroupType, UniversalGraph

logger = logging.getLogger(__name__)

_MIN_RUN_LENGTH_TO_GROUP = 2  # don't bother grouping a "run" of just 1 node
_MIN_REPEATS_TO_FORM_STAGE = 2
_MAX_MAIN_PATH_WALK_STEPS = 100


def _walk_main_path(start: str, target: str, normal_out: dict[str, list[str]]) -> list[str] | None:
    """
    Walk forward from `start` following normal (non-skip) edges until
    reaching `target`. Returns the full path (inclusive of both ends) if
    every intermediate node has exactly one normal successor (a clean,
    unbranched chain), or None if the path is ambiguous/branching/too
    long - in which case the caller should leave those nodes ungrouped
    rather than guess.
    """
    path = [start]
    current = start
    steps = 0
    while current != target:
        steps += 1
        if steps > _MAX_MAIN_PATH_WALK_STEPS:
            return None
        successors = normal_out.get(current, [])
        if len(successors) != 1:
            return None
        current = successors[0]
        path.append(current)
    return path


def _find_residual_blocks(
    graph: UniversalGraph, normal_out: dict[str, list[str]]
) -> list[list[str]]:
    """Returns a list of member-node-id lists, one per detected residual block."""
    blocks: list[list[str]] = []
    for edge in graph.edges:
        if not edge.is_skip_connection:
            continue
        skip_source, merge_node = edge.source, edge.target

        # The skip source itself is the block's "input", not a member of
        # the block - find its main-path successor(s) and try each until
        # one of them reaches the merge node cleanly.
        for candidate_start in normal_out.get(skip_source, []):
            path = _walk_main_path(candidate_start, merge_node, normal_out)
            if path is not None:
                blocks.append(path)
                break
    return blocks


def _group_sequential_runs(
    ungrouped_ids: list[str],
    normal_out: dict[str, list[str]],
    normal_in: dict[str, list[str]],
    already_grouped: set[str],
) -> list[list[str]]:
    """
    Collapses contiguous, unbranched runs of ungrouped nodes (in-degree==1
    and out-degree==1 at every step) into simple sequential groups.
    Stops a run as soon as it would enter an already-grouped node (e.g.
    a residual block) or hit a branch point.
    """
    visited: set[str] = set()
    runs: list[list[str]] = []

    for node_id in ungrouped_ids:
        if node_id in visited:
            continue
        run = [node_id]
        visited.add(node_id)
        current = node_id
        while True:
            successors = normal_out.get(current, [])
            if len(successors) != 1:
                break
            next_id = successors[0]
            if next_id in already_grouped or next_id in visited:
                break
            if len(normal_in.get(next_id, [])) != 1:
                break  # next_id is a merge point - don't silently absorb it
            run.append(next_id)
            visited.add(next_id)
            current = next_id
        if len(run) >= _MIN_RUN_LENGTH_TO_GROUP:
            runs.append(run)

    return runs


def _label_for_sequential_run(member_types: list[str]) -> str:
    """Picks a more meaningful label than a generic 'Block' when possible."""
    if any("Conv" in t for t in member_types):
        return "ConvBlock"
    if any("Linear" in t for t in member_types):
        return "DenseBlock"
    if any("Pool" in t for t in member_types):
        return "PoolBlock"
    return "Block"


def build_groups(graph: UniversalGraph) -> UniversalGraph:
    """
    Runs the full Phase 3 grouping pipeline on an already-built
    UniversalGraph and returns a new UniversalGraph with node.group_id
    populated and graph.groups filled in. Does not mutate the input.
    """
    node_by_id = {n.id: n for n in graph.nodes}
    node_order_index = {n.id: idx for idx, n in enumerate(graph.nodes)}

    normal_out: dict[str, list[str]] = defaultdict(list)
    normal_in: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        if not edge.is_skip_connection:
            normal_out[edge.source].append(edge.target)
            normal_in[edge.target].append(edge.source)

    groups: list[GraphGroup] = []
    node_group_id: dict[str, str] = {}
    group_counter = 0

    # --- Step 1: residual blocks ---
    residual_member_lists = _find_residual_blocks(graph, normal_out)
    for members in residual_member_lists:
        group_counter += 1
        gid = f"group_{group_counter}"
        for node_id in members:
            node_group_id[node_id] = gid
        groups.append(GraphGroup(
            id=gid,
            label="ResidualBlock",
            type=GroupType.RESIDUAL_BLOCK,
            member_node_ids=members,
        ))

    # --- Step 2: simple sequential runs among whatever wasn't grouped above ---
    ungrouped_ids = [n.id for n in graph.nodes if n.id not in node_group_id]
    already_grouped = set(node_group_id.keys())
    sequential_runs = _group_sequential_runs(ungrouped_ids, normal_out, normal_in, already_grouped)
    for run in sequential_runs:
        group_counter += 1
        gid = f"group_{group_counter}"
        for node_id in run:
            node_group_id[node_id] = gid
        label = _label_for_sequential_run([node_by_id[nid].type for nid in run])
        groups.append(GraphGroup(
            id=gid,
            label=label,
            type=GroupType.CONV_BLOCK,
            member_node_ids=run,
        ))

    # --- Step 3: Stage detection - merge consecutive identical residual blocks ---
    def signature(members: list[str]) -> tuple:
        return tuple(node_by_id[m].type for m in members)

    residual_groups = [g for g in groups if g.type == GroupType.RESIDUAL_BLOCK]
    residual_groups.sort(key=lambda g: node_order_index[g.member_node_ids[0]])

    stage_counter = 0
    i = 0
    while i < len(residual_groups):
        j = i
        sig = signature(residual_groups[i].member_node_ids)
        while j + 1 < len(residual_groups) and signature(residual_groups[j + 1].member_node_ids) == sig:
            j += 1
        run_length = j - i + 1
        if run_length >= _MIN_REPEATS_TO_FORM_STAGE:
            stage_counter += 1
            stage_gid = f"stage_{stage_counter}"
            for g in residual_groups[i:j + 1]:
                g.parent_group_id = stage_gid
            groups.append(GraphGroup(
                id=stage_gid,
                label=f"Stage {stage_counter} ({run_length} blocks)",
                type=GroupType.STAGE,
                member_node_ids=[],  # a Stage's members are groups, not raw nodes
                repeat_count=run_length,
            ))
        i = j + 1

    new_nodes = [
        n.model_copy(update={"group_id": node_group_id.get(n.id)})
        for n in graph.nodes
    ]

    logger.info(
        "Grouping produced %d group(s) (%d residual, %d sequential, %d stage) covering %d/%d nodes",
        len(groups),
        sum(1 for g in groups if g.type == GroupType.RESIDUAL_BLOCK),
        sum(1 for g in groups if g.type == GroupType.CONV_BLOCK),
        sum(1 for g in groups if g.type == GroupType.STAGE),
        len(node_group_id),
        len(graph.nodes),
    )

    return graph.model_copy(update={"nodes": new_nodes, "groups": groups})