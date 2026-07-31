"""
neuralviz/cli.py

Main entrypoint for the `neuralviz` command.

Usage:
    neuralviz <path>                # open browser diagram (default)
    neuralviz <path> --text         # print ASCII diagram to terminal
    neuralviz <path> --json         # dump UniversalGraph JSON to stdout
    neuralviz <path> --port 8842    # use a specific port
    neuralviz <path> --offline      # skip network; use cached HF configs only
    neuralviz --version             # print version and exit

Design:
    - Uses stdlib `argparse` — no heavy external CLI framework needed.
    - Calls the vendored parse_project() directly as a Python function;
      no HTTP, no job upload, no polling.
    - HuggingFace models are parsed from config.json ONLY (no weights
      downloaded) via hf_config_parser. Use --offline to disable all
      network access and rely on the local HuggingFace cache.
    - Catches FrameworkNotSupportedError / ParseChainError and prints
      a clean, coloured per-tier error — never a raw Python traceback.
    - For directory inputs, reuses find_candidate_model_files() from
      the vendored file_handler, identical logic to the web app's zip flow.
"""

import argparse
import json
import logging
import os
import sys
import uuid
from pathlib import Path


# ── Color helpers (degrade gracefully if colorama is absent) ─────────────────

def _init_colors():
    """Try to initialise colorama; return (RED, YELLOW, CYAN, BOLD, RESET)."""
    try:
        import colorama
        colorama.init()
        return (
            colorama.Fore.RED,
            colorama.Fore.YELLOW,
            colorama.Fore.CYAN,
            colorama.Style.BRIGHT,
            colorama.Style.RESET_ALL,
        )
    except ImportError:
        return ("", "", "", "", "")


RED, YELLOW, CYAN, BOLD, RESET = _init_colors()


def _err(msg: str) -> None:
    print(f"{RED}{BOLD}Error:{RESET} {msg}", file=sys.stderr)


def _warn(msg: str) -> None:
    print(f"{YELLOW}Warning:{RESET} {msg}", file=sys.stderr)


def _info(msg: str) -> None:
    print(f"{CYAN}{msg}{RESET}", file=sys.stderr)


# ── Path resolution ──────────────────────────────────────────────────────────

def _resolve_model_file(raw_path: str) -> Path:
    """
    Given a path string from the CLI argument:
      - If it's a .py file, validate it exists and return it.
      - If it's a directory, run find_candidate_model_files() on it and
        return the best candidate (same logic as the web app's zip upload).
    Exits with a clean error message if nothing usable is found.
    """
    path = Path(raw_path).resolve()

    if not path.exists():
        _err(f"Path does not exist: {path}")
        sys.exit(1)

    if path.is_file():
        if path.suffix != ".py":
            _err(f"Expected a Python (.py) file, got: {path.name}")
            sys.exit(1)
        return path

    # Directory — scan for best candidate
    if path.is_dir():
        try:
            from neuralviz._vendored.utils.file_handler import find_candidate_model_files
        except ImportError as exc:
            _err(f"Internal import error: {exc}")
            sys.exit(1)

        candidates = find_candidate_model_files(path)
        if not candidates:
            _err(
                f"No Python model files found inside '{path}'.\n"
                "  Make sure the directory contains a .py file with a "
                "PyTorch / TensorFlow model definition."
            )
            sys.exit(1)

        chosen = candidates[0]
        _info(f"Auto-selected model file: {chosen.relative_to(path.parent)}")
        return chosen

    _err(f"Path is neither a file nor a directory: {path}")
    sys.exit(1)


# ── Parsing ──────────────────────────────────────────────────────────────────

def _parse_model(model_file: Path, offline: bool = False):
    """
    Run the full parsing chain. Returns a UniversalGraph on success.
    Exits cleanly with an error message on expected failure modes.

    Parameters
    ----------
    model_file : Path
        Resolved path to the .py file to parse.
    offline : bool
        If True, passes local_files_only=True to the HuggingFace config
        parser — no network requests will be made. The local HuggingFace
        cache (~/.cache/huggingface/) must already contain the config.
    """
    try:
        from neuralviz._vendored.services.parser_service import parse_project
        from neuralviz._vendored.core.exceptions import (
            FrameworkNotSupportedError,
            ModelParsingError,
            ParseChainError,
        )
    except ImportError as exc:
        _err(
            f"Could not import parser engines: {exc}\n\n"
            "  Make sure the framework you need is installed:\n"
            "    PyTorch / HuggingFace:  pip install torch transformers\n"
            "    TensorFlow:            pip install tensorflow-cpu\n"
            "    JAX / Flax:            pip install jax flax"
        )
        sys.exit(1)

    job_id = uuid.uuid4().hex[:12]

    # Propagate --offline to hf_config_parser via environment variable.
    # The parser reads NEURALVIZ_HF_LOCAL_ONLY at call time.
    if offline:
        os.environ["NEURALVIZ_HF_LOCAL_ONLY"] = "1"
    else:
        os.environ.pop("NEURALVIZ_HF_LOCAL_ONLY", None)

    try:
        return parse_project(job_id, model_file)

    except ParseChainError as exc:
        # Show each tier's failure with its actionable suggestion
        lines = [f"{RED}{BOLD}All parsing strategies failed:{RESET}"]
        for f in exc.failures:
            lines.append(f"  {BOLD}[{f.tier}]{RESET} {f.error}")
            if f.suggestion:
                lines.append(f"    {YELLOW}→ {f.suggestion}{RESET}")
        print("\n".join(lines), file=sys.stderr)
        sys.exit(1)

    except FrameworkNotSupportedError as exc:
        _err(
            f"Framework not supported.\n\n"
            f"  {exc}\n\n"
            "  neuralviz supports: PyTorch/HuggingFace (torch/transformers), "
            "TensorFlow/Keras, JAX/Flax, and framework-free NumPy models.\n"
            "  If your model uses one of those, check that the right "
            "package is installed in this environment."
        )
        sys.exit(1)

    except ModelParsingError as exc:
        _err(
            f"Could not parse model.\n\n"
            f"  {exc}\n\n"
            "  Try adding type annotations or simplifying dynamic control "
            "flow in your forward() method."
        )
        sys.exit(1)

    except Exception as exc:  # noqa: BLE001
        _err(f"Unexpected error during parsing: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    from neuralviz import __version__

    parser = argparse.ArgumentParser(
        prog="neuralviz",
        description="Visualize your PyTorch / TensorFlow model's architecture.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  neuralviz my_model.py                    # open browser diagram
  neuralviz my_model.py --text             # ASCII diagram in terminal
  neuralviz my_model.py --json             # raw JSON to stdout
  neuralviz my_project/                    # auto-finds model file in directory
  neuralviz my_model.py --port 8842        # use specific port for browser mode
  neuralviz ocr_pipeline.py --offline      # use cached HuggingFace config (no network)
        """,
    )

    parser.add_argument(
        "path",
        nargs="?",
        help="Path to a Python model file (.py) or a project directory.",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        default=False,
        help="Print an ASCII diagram to the terminal instead of opening a browser.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Dump raw UniversalGraph JSON to stdout (pipe-friendly).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        metavar="PORT",
        help="Port for the local browser server (default: random available port).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"neuralviz {__version__}",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=False,
        help=(
            "Disable all network requests. For HuggingFace models, uses only "
            "the local cache (~/.cache/huggingface/). Raises an error if the "
            "config is not already cached."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging output.",
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    # Require <path> unless --version was already handled by argparse
    if not args.path:
        parser.print_help()
        sys.exit(0)

    # 1. Resolve model file
    model_file = _resolve_model_file(args.path)

    # 2. Parse
    if args.offline:
        _info(f"Parsing {model_file.name} (offline mode — using local HF cache) ...")
    else:
        _info(f"Parsing {model_file.name} ...")
    graph = _parse_model(model_file, offline=args.offline)

    # Show any parser warnings
    for w in graph.meta.warnings:
        _warn(w)

    # 3. Dispatch output mode
    if args.json:
        # Raw JSON to stdout — pipe-friendly
        print(json.dumps(graph.model_dump(), indent=2))

    elif args.text:
        from neuralviz.text_render import render
        render(graph)

    else:
        # Default: browser mode
        from neuralviz.local_server import serve
        serve(graph, port=args.port)


if __name__ == "__main__":
    main()
