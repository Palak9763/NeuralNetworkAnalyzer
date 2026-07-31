# VENDORED COPY — adapted from backend/app/engines/pytorch/hf_config_parser.py
# Part of the neuralviz CLI package. Sync manually if upstream changes.
# NOTE: All `app.*` imports have been rewritten to `neuralviz._vendored.*`.

"""
engines/pytorch/hf_config_parser.py

Parses Hugging Face model architecture from config.json ONLY.
Never downloads model weights. Requires only the `transformers` package.
Downloads typically < 10 KB (config.json only).

See backend/app/engines/pytorch/hf_config_parser.py for full documentation.
"""

import logging
from pathlib import Path
from typing import Callable

from neuralviz._vendored.core.exceptions import ModelParsingError
from neuralviz._vendored.engines.pytorch.ast_parser import RawEdge, RawNode, RawParseResult
from neuralviz._vendored.engines.pytorch.pretrained_parser import (
    _build_import_map,
    find_pretrained_call,
)

logger = logging.getLogger(__name__)


# ── Shared node / edge helpers ────────────────────────────────────────────────

def _make_node(idx: int, layer_type: str, label: str, params: int = 0) -> RawNode:
    return RawNode(
        id=f"node_{idx}",
        type=layer_type,
        label=label,
        params=params,
        input_shape=None,
        output_shape=None,
    )


def _seq_edges(nodes: list[RawNode]) -> list[RawEdge]:
    return [
        RawEdge(source=nodes[i].id, target=nodes[i + 1].id)
        for i in range(len(nodes) - 1)
    ]


# ── Per-family graph builders ─────────────────────────────────────────────────

def _build_encoder_only_graph(config) -> tuple[list[RawNode], list[RawEdge]]:
    n = getattr(config, "num_hidden_layers", 12)
    nodes: list[RawNode] = [_make_node(1, "Embeddings", "embeddings")]
    for i in range(n):
        nodes.append(_make_node(len(nodes) + 1, "TransformerLayer", f"encoder.layer.{i}"))
    nodes.append(_make_node(len(nodes) + 1, "Pooler", "pooler"))
    return nodes, _seq_edges(nodes)


def _build_vit_graph(config) -> tuple[list[RawNode], list[RawEdge]]:
    n = getattr(config, "num_hidden_layers", 12)
    nodes: list[RawNode] = [
        _make_node(1, "PatchEmbeddings", "embeddings.patch_embeddings"),
    ]
    for i in range(n):
        nodes.append(_make_node(len(nodes) + 1, "ViTLayer", f"encoder.layer.{i}"))
    nodes.append(_make_node(len(nodes) + 1, "LayerNorm", "layernorm"))
    nodes.append(_make_node(len(nodes) + 1, "Pooler", "pooler"))
    return nodes, _seq_edges(nodes)


def _build_swin_graph(config) -> tuple[list[RawNode], list[RawEdge]]:
    depths = getattr(config, "depths", [2, 2, 6, 2])
    nodes: list[RawNode] = [
        _make_node(1, "PatchEmbeddings", "embeddings.patch_embeddings"),
    ]
    for stage_idx, depth in enumerate(depths):
        for layer_idx in range(depth):
            nodes.append(_make_node(
                len(nodes) + 1,
                "SwinLayer",
                f"encoder.layers.{stage_idx}.blocks.{layer_idx}",
            ))
    nodes.append(_make_node(len(nodes) + 1, "LayerNorm", "layernorm"))
    nodes.append(_make_node(len(nodes) + 1, "Pooler", "pooler"))
    return nodes, _seq_edges(nodes)


def _build_encoder_decoder_graph(config) -> tuple[list[RawNode], list[RawEdge]]:
    enc_cfg = getattr(config, "encoder", config)
    dec_cfg = getattr(config, "decoder", config)
    n_enc = getattr(enc_cfg, "num_hidden_layers",
                    getattr(enc_cfg, "num_layers", 6))
    n_dec = getattr(dec_cfg, "num_hidden_layers",
                    getattr(dec_cfg, "num_layers", 6))
    enc_type = getattr(enc_cfg, "model_type", "encoder").title().replace("-", "")
    dec_type = getattr(dec_cfg, "model_type", "decoder").title().replace("-", "")

    nodes: list[RawNode] = []
    nodes.append(_make_node(1, f"{enc_type}Embeddings", "encoder.embeddings"))
    for i in range(n_enc):
        nodes.append(_make_node(len(nodes) + 1, f"{enc_type}Layer", f"encoder.layer.{i}"))
    nodes.append(_make_node(len(nodes) + 1, f"{dec_type}Embeddings", "decoder.embeddings"))
    for i in range(n_dec):
        nodes.append(_make_node(len(nodes) + 1, f"{dec_type}Layer", f"decoder.layer.{i}"))
    nodes.append(_make_node(len(nodes) + 1, "LMHead", "lm_head"))
    return nodes, _seq_edges(nodes)


def _build_gpt_graph(config) -> tuple[list[RawNode], list[RawEdge]]:
    n = getattr(config, "n_layer",
                getattr(config, "num_hidden_layers",
                        getattr(config, "num_layers", 12)))
    nodes: list[RawNode] = [
        _make_node(1, "TokenEmbedding", "wte"),
        _make_node(2, "PositionEmbedding", "wpe"),
    ]
    for i in range(n):
        nodes.append(_make_node(len(nodes) + 1, "CausalBlock", f"h.{i}"))
    nodes.append(_make_node(len(nodes) + 1, "LayerNorm", "ln_f"))
    nodes.append(_make_node(len(nodes) + 1, "LMHead", "lm_head"))
    return nodes, _seq_edges(nodes)


def _build_llama_graph(config) -> tuple[list[RawNode], list[RawEdge]]:
    n = getattr(config, "num_hidden_layers", 32)
    nodes: list[RawNode] = [_make_node(1, "Embeddings", "model.embed_tokens")]
    for i in range(n):
        nodes.append(_make_node(len(nodes) + 1, "DecoderLayer", f"model.layers.{i}"))
    nodes.append(_make_node(len(nodes) + 1, "RMSNorm", "model.norm"))
    nodes.append(_make_node(len(nodes) + 1, "LMHead", "lm_head"))
    return nodes, _seq_edges(nodes)


def _build_clip_graph(config) -> tuple[list[RawNode], list[RawEdge]]:
    vis_cfg = getattr(config, "vision_config", config)
    txt_cfg = getattr(config, "text_config", config)
    n_vis = getattr(vis_cfg, "num_hidden_layers", 12)
    n_txt = getattr(txt_cfg, "num_hidden_layers", 12)

    nodes: list[RawNode] = []
    nodes.append(_make_node(1, "PatchEmbeddings", "vision_model.embeddings"))
    for i in range(n_vis):
        nodes.append(_make_node(len(nodes) + 1, "CLIPEncoderLayer",
                                f"vision_model.encoder.layers.{i}"))
    nodes.append(_make_node(len(nodes) + 1, "VisualProjection", "visual_projection"))
    vis_tower_end = len(nodes) - 1

    txt_start = len(nodes)
    nodes.append(_make_node(len(nodes) + 1, "TextEmbeddings", "text_model.embeddings"))
    for i in range(n_txt):
        nodes.append(_make_node(len(nodes) + 1, "CLIPEncoderLayer",
                                f"text_model.encoder.layers.{i}"))
    nodes.append(_make_node(len(nodes) + 1, "TextProjection", "text_projection"))

    edges = _seq_edges(nodes[:vis_tower_end + 1])
    edges += _seq_edges(nodes[txt_start:])
    return nodes, edges


def _build_blip_graph(config) -> tuple[list[RawNode], list[RawEdge]]:
    vis_cfg = getattr(config, "vision_config", config)
    n_vis = getattr(vis_cfg, "num_hidden_layers", 12)
    n_text = getattr(config, "num_hidden_layers", 6)
    nodes: list[RawNode] = [
        _make_node(1, "PatchEmbeddings", "vision_model.embeddings"),
    ]
    for i in range(n_vis):
        nodes.append(_make_node(len(nodes) + 1, "ViTLayer",
                                f"vision_model.encoder.layers.{i}"))
    nodes.append(_make_node(len(nodes) + 1, "TextEmbeddings", "text_decoder.bert.embeddings"))
    for i in range(n_text):
        nodes.append(_make_node(len(nodes) + 1, "CrossAttentionLayer",
                                f"text_decoder.bert.encoder.layer.{i}"))
    nodes.append(_make_node(len(nodes) + 1, "LMHead", "text_decoder.cls"))
    return nodes, _seq_edges(nodes)


def _build_blip2_graph(config) -> tuple[list[RawNode], list[RawEdge]]:
    vis_cfg = getattr(config, "vision_config", config)
    n_vis = getattr(vis_cfg, "num_hidden_layers", 39)
    qformer_cfg = getattr(config, "qformer_config", config)
    n_qformer = getattr(qformer_cfg, "num_hidden_layers", 12)
    nodes: list[RawNode] = [
        _make_node(1, "PatchEmbeddings", "vision_model.embeddings"),
    ]
    for i in range(n_vis):
        nodes.append(_make_node(len(nodes) + 1, "ViTLayer",
                                f"vision_model.encoder.layers.{i}"))
    nodes.append(_make_node(len(nodes) + 1, "QueryTokens", "query_tokens"))
    for i in range(n_qformer):
        nodes.append(_make_node(len(nodes) + 1, "QFormerLayer",
                                f"qformer.encoder.layer.{i}"))
    nodes.append(_make_node(len(nodes) + 1, "LanguageProjection", "language_projection"))
    nodes.append(_make_node(len(nodes) + 1, "LMDecoder", "language_model"))
    return nodes, _seq_edges(nodes)


def _build_whisper_graph(config) -> tuple[list[RawNode], list[RawEdge]]:
    n_enc = getattr(config, "encoder_layers", 6)
    n_dec = getattr(config, "decoder_layers", 6)
    nodes: list[RawNode] = [
        _make_node(1, "AudioConv1", "model.encoder.conv1"),
        _make_node(2, "AudioConv2", "model.encoder.conv2"),
        _make_node(3, "PositionalEmbedding", "model.encoder.embed_positions"),
    ]
    for i in range(n_enc):
        nodes.append(_make_node(len(nodes) + 1, "EncoderLayer",
                                f"model.encoder.layers.{i}"))
    nodes.append(_make_node(len(nodes) + 1, "EncoderLayerNorm", "model.encoder.layer_norm"))
    nodes.append(_make_node(len(nodes) + 1, "DecoderEmbeddings", "model.decoder.embed_tokens"))
    for i in range(n_dec):
        nodes.append(_make_node(len(nodes) + 1, "DecoderLayer",
                                f"model.decoder.layers.{i}"))
    nodes.append(_make_node(len(nodes) + 1, "DecoderLayerNorm", "model.decoder.layer_norm"))
    nodes.append(_make_node(len(nodes) + 1, "LMHead", "proj_out"))
    return nodes, _seq_edges(nodes)


_MODEL_TYPE_BUILDERS: dict[str, Callable] = {
    "bert":                   _build_encoder_only_graph,
    "roberta":                _build_encoder_only_graph,
    "distilbert":             _build_encoder_only_graph,
    "albert":                 _build_encoder_only_graph,
    "electra":                _build_encoder_only_graph,
    "deberta":                _build_encoder_only_graph,
    "deberta-v2":             _build_encoder_only_graph,
    "camembert":              _build_encoder_only_graph,
    "xlm-roberta":            _build_encoder_only_graph,
    "xlnet":                  _build_encoder_only_graph,
    "vit":                    _build_vit_graph,
    "deit":                   _build_vit_graph,
    "beit":                   _build_vit_graph,
    "swin":                   _build_swin_graph,
    "swin-v2":                _build_swin_graph,
    "convnext":               _build_vit_graph,
    "t5":                     _build_encoder_decoder_graph,
    "mt5":                    _build_encoder_decoder_graph,
    "bart":                   _build_encoder_decoder_graph,
    "mbart":                  _build_encoder_decoder_graph,
    "pegasus":                _build_encoder_decoder_graph,
    "marian":                 _build_encoder_decoder_graph,
    "led":                    _build_encoder_decoder_graph,
    "vision-encoder-decoder": _build_encoder_decoder_graph,
    "donut-swin":             _build_swin_graph,
    "gpt2":                   _build_gpt_graph,
    "gpt_neo":                _build_gpt_graph,
    "gpt_neox":               _build_gpt_graph,
    "gpt_j":                  _build_gpt_graph,
    "gpt-sw3":                _build_gpt_graph,
    "opt":                    _build_gpt_graph,
    "llama":                  _build_llama_graph,
    "mistral":                _build_llama_graph,
    "mixtral":                _build_llama_graph,
    "falcon":                 _build_llama_graph,
    "phi":                    _build_llama_graph,
    "phi3":                   _build_llama_graph,
    "gemma":                  _build_llama_graph,
    "gemma2":                 _build_llama_graph,
    "qwen2":                  _build_llama_graph,
    "clip":                   _build_clip_graph,
    "blip":                   _build_blip_graph,
    "blip-2":                 _build_blip2_graph,
    "whisper":                _build_whisper_graph,
}


def _extract_hf_metadata(config, checkpoint: str) -> list[str]:
    structural_attrs = [
        "hidden_size", "num_hidden_layers", "num_attention_heads",
        "intermediate_size", "vocab_size", "max_position_embeddings",
        "patch_size", "image_size", "num_channels",
    ]
    lines = []
    for attr in structural_attrs:
        val = getattr(config, attr, None)
        if val is not None:
            lines.append(f"  {attr}: {val}")
    if not lines:
        return []
    return [f"Model config for '{checkpoint}':"] + lines


def run_hf_config_parser(file_path: Path, *, local_files_only: bool = False) -> RawParseResult:
    """
    Parse a Hugging Face model script by downloading ONLY config.json.
    No model weights are ever downloaded or loaded into memory.

    Parameters
    ----------
    file_path : Path
        Path to the Python script containing a from_pretrained() call.
    local_files_only : bool
        If True, disables all network requests and uses only the local
        HuggingFace cache. Pass True for --offline mode.
        Also auto-enabled when the env var NEURALVIZ_HF_LOCAL_ONLY=1 is set.

    Raises
    ------
    ModelParsingError
        If transformers is not installed, no from_pretrained() call can
        be resolved, AutoConfig fails, or the model_type is not registered.
    """
    import os
    # Honour the --offline CLI flag forwarded via environment variable
    if os.environ.get("NEURALVIZ_HF_LOCAL_ONLY") == "1":
        local_files_only = True

    try:
        source = file_path.read_text(errors="ignore")
    except OSError as exc:
        raise ModelParsingError(f"Cannot read file {file_path}: {exc}") from exc

    try:
        import ast as _ast
        tree = _ast.parse(source)
    except SyntaxError as exc:
        raise ModelParsingError(f"File is not valid Python: {exc}") from exc

    import_map = _build_import_map(tree)
    resolved = find_pretrained_call(tree, import_map)
    if resolved is None:
        raise ModelParsingError(
            "Could not statically resolve a from_pretrained() call "
            "(non-literal arguments or unrecognized import)."
        )

    class_name, module_name, args, kwargs = resolved
    if not args:
        raise ModelParsingError(
            f"{class_name}.from_pretrained() was called with no positional "
            "arguments — cannot determine checkpoint name statically."
        )

    checkpoint: str = str(args[0])
    logger.info(
        "HF config parser: fetching config for checkpoint='%s' (local_files_only=%s)",
        checkpoint, local_files_only,
    )

    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise ModelParsingError(
            "The 'transformers' package is not installed in this environment. "
            "Install it with: pip install transformers"
        ) from exc

    try:
        config = AutoConfig.from_pretrained(
            checkpoint,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
    except OSError as exc:
        raise ModelParsingError(
            f"AutoConfig.from_pretrained('{checkpoint}') failed: {exc}\n"
            "  • Check your internet connection.\n"
            "  • Verify the checkpoint name is correct on huggingface.co.\n"
            "  • If the model is private, ensure HF_TOKEN is set."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise ModelParsingError(
            f"AutoConfig.from_pretrained('{checkpoint}') raised an unexpected "
            f"error: {exc}"
        ) from exc

    model_type: str = getattr(config, "model_type", "unknown")
    builder = _MODEL_TYPE_BUILDERS.get(model_type)

    if builder is None:
        known = sorted(_MODEL_TYPE_BUILDERS.keys())
        raise ModelParsingError(
            f"No config-based graph builder is registered for "
            f"model_type='{model_type}'. "
            f"Registered types: {known}. "
            "Falling back to AST parser."
        )

    try:
        nodes, edges = builder(config)
    except Exception as exc:  # noqa: BLE001
        raise ModelParsingError(
            f"Config builder for model_type='{model_type}' raised an error: {exc}"
        ) from exc

    if not nodes:
        raise ModelParsingError(
            f"Config builder for model_type='{model_type}' produced no nodes."
        )

    metadata_lines = _extract_hf_metadata(config, checkpoint)
    warnings = [
        f"Architecture built from config.json only (model_type='{model_type}'). "
        "No model weights were downloaded. "
        "Layer order is canonical for this architecture family; "
        "tensor shapes are not available (no forward pass was run).",
    ]
    if metadata_lines:
        warnings.append("\n".join(metadata_lines))

    logger.info(
        "HF config parser: built %d nodes, %d edges for '%s' (model_type=%s)",
        len(nodes), len(edges), checkpoint, model_type,
    )

    return RawParseResult(
        nodes=nodes,
        edges=edges,
        model_name=f"{class_name} ({checkpoint})",
        warnings=warnings,
    )
