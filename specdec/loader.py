"""Load Baguette checkpoints while keeping model code and weights external."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch


def load_baguette(source: Path, checkpoint: Path, device: torch.device):
    """Use the architecture from Baguette's model.py.

    Exported weight-only checkpoints are supported. Training checkpoints also
    contain NumPy RNG state and must first be exported with Baguette's script.
    """
    source = source.resolve()
    if not (source / "model.py").is_file():
        raise FileNotFoundError(f"no model.py in {source}")
    sys.path.insert(0, str(source))
    from model import ModelConfig, build_model

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model_cfg" not in payload or "model" not in payload:
        raise ValueError("expected Baguette checkpoint with model_cfg and model")
    config = ModelConfig.from_dict(payload["model_cfg"])
    model = build_model(config)
    model.load_state_dict(payload["model"])
    return model.to(device).eval()


def assert_tokenizer_compatible(target_path: Path, draft_path: Path) -> None:
    """Require identical serialized tokenizers, not merely equal vocab size."""
    def digest(path: Path) -> bytes:
        return hashlib.sha256(path.read_bytes()).digest()

    if digest(target_path) != digest(draft_path):
        raise ValueError("target and draft tokenizer files differ")
