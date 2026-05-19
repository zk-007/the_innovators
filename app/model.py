from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import timm
import torch
from PIL import Image, ImageEnhance, ImageOps

from .config import CLASSES, DEFAULT_CHECKPOINT
from .model_loader import ensure_checkpoint
from .transforms import get_eval_transform


class SignLanguageModel:
    def __init__(self, checkpoint_path: Path | str | None = None, device: str | None = None):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else ensure_checkpoint()
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.classes = CLASSES
        self._load()

    def _load(self) -> None:
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model_name = checkpoint.get("model_name", "convnext_tiny.fb_in22k_ft_in1k")
        self.img_size = int(checkpoint.get("img_size", 224))
        self.fold = checkpoint.get("fold", 0)
        self.best_acc = checkpoint.get("best_acc")
        self.epoch = checkpoint.get("epoch")

        self.model = timm.create_model(self.model_name, pretrained=False, num_classes=len(self.classes))
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.to(self.device)
        self.model.eval()
        self.transform = get_eval_transform(self.img_size)

    def _forward_probs(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.amp.autocast(device_type="cuda", enabled=self.device.type == "cuda"):
            logits = self.model(tensor)
            return torch.softmax(logits, dim=1)[0].cpu().numpy()

    def _augment_views(self, image: Image.Image) -> list[Image.Image]:
        image = image.convert("RGB")
        views = [
            image,
            ImageOps.mirror(image),
            ImageEnhance.Brightness(image).enhance(0.88),
            ImageEnhance.Brightness(image).enhance(1.12),
            ImageEnhance.Contrast(image).enhance(1.15),
        ]
        return views

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image | np.ndarray,
        *,
        use_tta: bool = True,
    ) -> Tuple[str, float, dict[str, float]]:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")

        if not use_tta:
            probs = self._forward_probs(image)
        else:
            views = self._augment_views(image)
            acc = None
            for view in views:
                p = self._forward_probs(view)
                acc = p if acc is None else acc + p
            probs = acc / len(views)

        idx = int(probs.argmax())
        label = self.classes[idx]
        confidence = float(probs[idx])
        prob_map = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
        return label, confidence, prob_map

    def metadata(self) -> dict:
        return {
            "model_name": self.model_name,
            "img_size": self.img_size,
            "fold": self.fold,
            "epoch": self.epoch,
            "best_val_acc": self.best_acc,
            "checkpoint": str(self.checkpoint_path),
            "device": str(self.device),
        }
