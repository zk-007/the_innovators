"""
Upload model checkpoint to Hugging Face Hub (one-time).

Usage:
  pip install huggingface_hub
  huggingface-cli login
  python scripts/upload_weights_to_hf.py --repo-id YOUR_USERNAME/the-innovators
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "models" / "convnext_tiny_fb_in22k_ft_in1k_fold0_best.pth"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="e.g. zk-007/the-innovators")
    args = parser.parse_args()

    if not CKPT.exists():
        raise FileNotFoundError(f"Missing {CKPT}")

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(CKPT),
        path_in_repo="models/convnext_tiny_fb_in22k_ft_in1k_fold0_best.pth",
        repo_id=args.repo_id,
        repo_type="model",
    )
    print("Uploaded to:", f"https://huggingface.co/{args.repo_id}")
    print("Set HF_MODEL_REPO in app/model_loader.py to:", args.repo_id)


if __name__ == "__main__":
    main()
