"""Temporal smoothing + stable webcam prediction."""

from __future__ import annotations

from collections import deque

import numpy as np

from .config import CLASSES, MIN_MARGIN, MIN_VOTE_CONFIDENCE, TEMPORAL_FRAMES
from .model import SignLanguageModel
from .preprocess import prepare_webcam_frame


class WebcamPredictor:
    def __init__(self, model: SignLanguageModel):
        self.model = model
        self._prob_history: deque[np.ndarray] = deque(maxlen=TEMPORAL_FRAMES)

    def reset(self) -> None:
        self._prob_history.clear()

    def _smooth_probs(self) -> np.ndarray:
        if not self._prob_history:
            return np.ones(len(CLASSES)) / len(CLASSES)
        stacked = np.stack(list(self._prob_history), axis=0)
        return stacked.mean(axis=0)

    def predict_from_webcam(
        self,
        image: np.ndarray,
        *,
        mirror: bool = True,
        use_hand_crop: bool = True,
    ) -> tuple[str, float, dict[str, float], object, str]:
        pil_crop, note = prepare_webcam_frame(
            image,
            mirror=mirror,
            use_hand_crop=use_hand_crop,
            gray_background=True,
            match_dataset_style=True,
        )
        _, _, prob_map = self.model.predict(pil_crop, use_tta=True)
        probs = np.array([prob_map[c] for c in CLASSES], dtype=np.float64)
        self._prob_history.append(probs)

        avg = self._smooth_probs()
        idx = int(avg.argmax())
        label = CLASSES[idx]
        confidence = float(avg[idx])

        sorted_p = np.sort(avg)[::-1]
        margin = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else confidence

        prob_map_smooth = {CLASSES[i]: float(avg[i]) for i in range(len(CLASSES))}
        return label, confidence, prob_map_smooth, pil_crop, note, margin
