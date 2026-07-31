"""
engines/pytorch/hf_config_parser.py

Why this file exists:
    The previous approach (pretrained_parser.py) called
    SomeClass.from_pretrained() to download and load the full model weights
    (hundreds of MB to several GB) just to call named_modules() on the result.
    On Windows this triggered OSError 1455 (paging file too small) for large
    models such as TrOCR, Donut, BLIP, and GPT-2.

    The complete architectural description of any Hugging Face model is already
    available in config.json — a tiny JSON file (typically < 10 KB) that is
    always the first file downloaded by from_pretrained(). This module fetches
    ONLY that file via AutoConfig.from_pretrained() and builds the visualization
    graph from it, without ever touching the model weights.

What it does:
    1. Re-uses the static from_pretrained() call resolver from
       pretrained_parser.py to find the checkpoint name in the source file.
    2. Calls AutoConfig.from_pretrained(checkpoint) — downloads config.json
       only (< 10 KB). Network-free if the config is already cached.
    3. Routes to a registered builder function keyed on config.model_type.
    4. Each builder constructs a RawParseResult from the config's structural
       fields (num_hidden_layers, num_attention_heads, etc.) — no weights,
       no GPU, no RAM spike.
    5. Supports 30+ model families: BERT, RoBERTa, ViT, TrOCR, T5, GPT-2,
       LLaMA, CLIP, BLIP, BLIP-2, Donut, Swin, DeiT, and many more.

How it connects:
    Called by services/parser_service.py as Tier 0 in the HuggingFace path
    (before torch.fx or AST, both of which cannot handle hub-loading scripts).
    On failure it raises ModelParsingError so the caller can fall back to the
    AST parser.

Limitations:
    - Input/output tensor shapes are not available (no forward pass is run).
    - Edge order reflects the canonical sub-module declaration order for each
      architecture family, not a traced execution order.
    - model_type values not in _MODEL_TYPE_BUILDERS raise ModelParsingError,
      triggering the AST fallback.
    - Requires an internet connection the first time a checkpoint is accessed
      (subsequent runs use the HuggingFace cache, typically in ~/.cache/huggingface/).
    - Use --offline / local_files_only=True to disable all network access.
"""

import logging
from pathlib import Path
from typing import Callable

from app.core.exceptions import ModelParsingError
from app.engines.pytorch.ast_parser import RawEdge, RawNode, RawParseResult
from app.engines.pytorch.pretrained_parser import (
    _build_import_map,
    find_pretrained_call,
)

logger = logging.getLogger(__name__)


# ── Shared node / edge helpers ────────────────────────────────────────────────

def _make_node(idx: int, layer_type: str, label: str, params: int = 0) -> RawNode:
    """Create a RawNode with a consistent id scheme."""
    return RawNode(
        id=f"node_{idx}",
        type=layer_type,
        label=label,
        params=params,
        input_shape=None,
        output_shape=None,
    )


def _seq_edges(nodes: list[RawNode]) -> list[RawEdge]:
    """Build a simple sequential edge list connecting nodes in order."""
    return [
        RawEdge(source=nodes[i].id, target=nodes[i + 1].id)
        for i in range(len(nodes) - 1)
    ]


# ── Per-family graph builders ─────────────────────────────────────────────────

def _build_encoder_only_graph(config) -> tuple[list[RawNode], list[RawEdge]]:
    """
    BERT, RoBERTa, DistilBERT, ALBERT, Electra, DeBERTa, etc.
    Structure: Embeddings → [TransformerLayer × N] → Pooler
    """
    n = getattr(config, "num_hidden_layers", 12)
    nodes: list[RawNode] = [_make_node(1, "Embeddings", "embeddings")]
    for i in range(n):
        nodes.append(_make_node(len(nodes) + 1, "TransformerLayer", f"encoder.layer.{i}"))
    nodes.append(_make_node(len(nodes) + 1, "Pooler", "pooler"))
    return nodes, _seq_edges(nodes)


def _build_vit_graph(config) -> tuple[list[RawNode], list[RawEdge]]:
    """
    ViT, DeiT, BEiT, etc.
    Structure: PatchEmbeddings → [ViTLayer × N] → LayerNorm → Pooler
    """
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
    """
    Swin Transformer.
    Structure: PatchEmbeddings → [SwinStage × N_stages] → LayerNorm → Pooler
    """
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
    """
    T5, BART, mBART, Pegasus, TrOCR (VisionEncoderDecoder), Donut, etc.
    Structure:
        EncoderEmbedding → [EncoderLayer × N_enc]
        → DecoderEmbedding → [DecoderLayer × N_dec] → LMHead
    """
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
    """
    GPT-2, GPT-Neo, GPT-J, GPT-NeoX, LLaMA, Mistral, Falcon, etc.
    Structure: TokenEmbedding + PosEmbedding → [CausalBlock × N] → LayerNorm → LMHead
    """
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
    """
    LLaMA, Mistral, Mixtral, Phi, Gemma, etc. (RoPE-based decoder-only).
    Structure: Embeddings → [DecoderLayer × N] → LayerNorm → LMHead
    """
    n = getattr(config, "num_hidden_layers", 32)
    nodes: list[RawNode] = [_make_node(1, "Embeddings", "model.embed_tokens")]
    for i in range(n):
        nodes.append(_make_node(len(nodes) + 1, "DecoderLayer", f"model.layers.{i}"))
    nodes.append(_make_node(len(nodes) + 1, "RMSNorm", "model.norm"))
    nodes.append(_make_node(len(nodes) + 1, "LMHead", "lm_head"))
    return nodes, _seq_edges(nodes)


def _build_clip_graph(config) -> tuple[list[RawNode], list[RawEdge]]:
    """
    CLIP, ALIGN, SigLIP — dual-encoder vision-language models.
    Structure: Two independent towers (vision + text), both linear.
    """
    vis_cfg = getattr(config, "vision_config", config)
    txt_cfg = getattr(config, "text_config", config)
    n_vis = getattr(vis_cfg, "num_hidden_layers", 12)
    n_txt = getattr(txt_cfg, "num_hidden_layers", 12)

    nodes: list[RawNode] = []

    # Vision tower
    nodes.append(_make_node(1, "PatchEmbeddings", "vision_model.embeddings"))
    for i in range(n_vis):
        nodes.append(_make_node(len(nodes) + 1, "CLIPEncoderLayer",
                                f"vision_model.encoder.layers.{i}"))
    vis_proj = _make_node(len(nodes) + 1, "VisualProjection", "visual_projection")
    nodes.append(vis_proj)
    vis_tower_end = len(nodes) - 1  # 0-indexed last vision node

    # Text tower
    txt_start = len(nodes)
    nodes.append(_make_node(len(nodes) + 1, "TextEmbeddings", "text_model.embeddings"))
    for i in range(n_txt):
        nodes.append(_make_node(len(nodes) + 1, "CLIPEncoderLayer",
                                f"text_model.encoder.layers.{i}"))
    nodes.append(_make_node(len(nodes) + 1, "TextProjection", "text_projection"))

    # Two independent sequential towers
    edges = _seq_edges(nodes[:vis_tower_end + 1])       # vision tower
    edges += _seq_edges(nodes[txt_start:])               # text tower
    return nodes, edges


def _build_blip_graph(config) -> tuple[list[RawNode], list[RawEdge]]:
    """
    BLIP (Salesforce/blip-image-captioning-*).
    Structure: ViT encoder → QFormer text-image fusion → LMHead
    """
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
    """
    BLIP-2 (Salesforce/blip2-*).
    Structure: ViT → QFormer → LanguageModelProjection → LLM decoder
    """
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
    """
    Whisper (openai/whisper-*).
    Structure: AudioEncoder → TextDecoder
    """
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


# ── Model-type → builder registry ────────────────────────────────────────────

_MODEL_TYPE_BUILDERS: dict[str, Callable] = {
    # Encoder-only (BERT family)
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
    # Vision encoder-only
    "vit":                    _build_vit_graph,
    "deit":                   _build_vit_graph,
    "beit":                   _build_vit_graph,
    "swin":                   _build_swin_graph,
    "swin-v2":                _build_swin_graph,
    "convnext":               _build_vit_graph,    # similar structure
    # Encoder-decoder
    "t5":                     _build_encoder_decoder_graph,
    "mt5":                    _build_encoder_decoder_graph,
    "bart":                   _build_encoder_decoder_graph,
    "mbart":                  _build_encoder_decoder_graph,
    "pegasus":                _build_encoder_decoder_graph,
    "marian":                 _build_encoder_decoder_graph,
    "led":                    _build_encoder_decoder_graph,
    "vision-encoder-decoder": _build_encoder_decoder_graph,  # TrOCR, Donut
    "donut-swin":             _build_swin_graph,              # encoder half of Donut
    # Decoder-only (GPT family)
    "gpt2":                   _build_gpt_graph,
    "gpt_neo":                _build_gpt_graph,
    "gpt_neox":               _build_gpt_graph,
    "gpt_j":                  _build_gpt_graph,
    "gpt-sw3":                _build_gpt_graph,
    "opt":                    _build_gpt_graph,
    # Decoder-only (LLaMA / RoPE family)
    "llama":                  _build_llama_graph,
    "mistral":                _build_llama_graph,
    "mixtral":                _build_llama_graph,
    "falcon":                 _build_llama_graph,
    "phi":                    _build_llama_graph,
    "phi3":                   _build_llama_graph,
    "gemma":                  _build_llama_graph,
    "gemma2":                 _build_llama_graph,
    "qwen2":                  _build_llama_graph,
    # Multimodal
    "clip":                   _build_clip_graph,
    "blip":                   _build_blip_graph,
    "blip-2":                 _build_blip2_graph,
    # Speech
    "whisper":                _build_whisper_graph,
}


# ── Metadata helper ───────────────────────────────────────────────────────────

def _extract_hf_metadata(config, checkpoint: str) -> list[str]:
    """Collect human-readable structural metadata from a HF config object."""
    structural_attrs = [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "vocab_size",
        "max_position_embeddings",
        "patch_size",
        "image_size",
        "num_channels",
    ]
    lines = []
    for attr in structural_attrs:
        val = getattr(config, attr, None)
        if val is not None:
            lines.append(f"  {attr}: {val}")
    if not lines:
        return []
    return [f"Model config for '{checkpoint}':"] + lines


# ── Public API ────────────────────────────────────────────────────────────────

def run_hf_config_parser(file_path: Path, *, local_files_only: bool = False) -> RawParseResult:
    """
    Parse a Hugging Face model script by downloading ONLY config.json.
    No model weights are ever downloaded or loaded into memory.

    Parameters
    ----------
    file_path:
        Path to the Python script containing a from_pretrained() call.
    local_files_only:
        If True, disables all network requests and uses only the local
        HuggingFace cache (typically ~/.cache/huggingface/). Pass True
        when the user requests --offline mode.

    Raises
    ------
    ModelParsingError
        If transformers is not installed, no from_pretrained() call can
        be resolved, AutoConfig fails, or the detected model_type is not
        in the builder registry. The caller should fall back to ast_parser.
    """
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
        # Covers: model not found, no internet, HuggingFace 404, etc.
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
