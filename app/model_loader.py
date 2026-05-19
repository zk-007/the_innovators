"""Ensure checkpoint exists locally (for Hugging Face Spaces first boot)."""

from __future__ import annotations

from pathlib import Path

from .config import DEFAULT_CHECKPOINT, MODELS_DIR

# After you create the HF Model repo, set this (see docs/DEPLOY_FOR_JUDGES.md)
HF_MODEL_REPO = "zk-007/the-innovators"
HF_MODEL_FILE = "models/convnext_tiny_fb_in22k_ft_in1k_fold0_best.pth"


def ensure_checkpoint() -> Path:
    if DEFAULT_CHECKPOINT.exists():
        return DEFAULT_CHECKPOINT

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("Checkpoint not found locally. Downloading from Hugging Face Hub…")

    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=HF_MODEL_FILE,
            repo_type="model",
            local_dir=str(MODELS_DIR.parent),
        )
        return Path(path)
    except Exception as exc:
        raise FileNotFoundError(
            f"Model weights missing at {DEFAULT_CHECKPOINT}. "
            f"Upload the .pth to Hugging Face model repo `{HF_MODEL_REPO}` "
            f"or include it in the Space files. Details: docs/DEPLOY_FOR_JUDGES.md"
        ) from exc
