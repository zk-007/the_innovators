"""Hold-to-commit spelling logic for Phase 2 word builder."""

from __future__ import annotations

import time
from collections import deque

from .config import HOLD_SECONDS, MIN_VOTE_CONFIDENCE, SPELLABLE_LABELS

# After committing a letter, wait this long before the same letter can commit again.
# Stops one long hold becoming "AA", but allows ALL / BEE (pause, then hold same letter again).
SAME_LETTER_GAP_SEC = 0.45


class SpellingState:
    def __init__(self, hold_seconds: float = HOLD_SECONDS):
        self.hold_seconds = hold_seconds
        self.sentence = ""
        self.recent: deque[tuple[str, float]] = deque(maxlen=12)
        self._hold_label: str | None = None
        self._hold_start: float | None = None
        self.last_committed = ""
        self._repeat_cooldown_until = 0.0

    def reset_sentence(self) -> str:
        self.sentence = ""
        self.recent.clear()
        self._reset_hold()
        self._repeat_cooldown_until = 0.0
        return self.sentence

    def _reset_hold(self) -> None:
        self._hold_label = None
        self._hold_start = None

    def smooth_label(self, label: str, confidence: float, min_conf: float = MIN_VOTE_CONFIDENCE) -> str:
        if confidence >= min_conf:
            self.recent.append((label, confidence))
        if not self.recent:
            return label
        scores: dict[str, float] = {}
        for lbl, conf in self.recent:
            scores[lbl] = scores.get(lbl, 0.0) + conf
        return max(scores, key=scores.get)

    def _same_letter_on_cooldown(self, label: str) -> bool:
        return (
            label == self.last_committed
            and label in SPELLABLE_LABELS
            and time.time() < self._repeat_cooldown_until
        )

    def update(
        self,
        label: str,
        confidence: float,
        threshold: float,
        *,
        margin: float = 0.0,
        min_margin: float = 0.10,
    ) -> tuple[str, str]:
        if margin < min_margin and label in SPELLABLE_LABELS:
            self._reset_hold()
            return (
                self.sentence,
                f"Uncertain (gap {margin:.0%}) — adjust hand; need clearer {label}",
            )

        if label == "del":
            if self.sentence:
                self.sentence = self.sentence[:-1]
            self._reset_hold()
            self._repeat_cooldown_until = 0.0
            return self.sentence, "Deleted last character"

        if label == "space":
            if self.sentence and not self.sentence.endswith(" "):
                self.sentence += " "
            self._reset_hold()
            self._repeat_cooldown_until = 0.0
            return self.sentence, "Added space"

        if label == "nothing" or label not in SPELLABLE_LABELS:
            self._reset_hold()
            return self.sentence, f"Showing: {label} ({confidence:.0%})"

        if confidence < threshold:
            self._reset_hold()
            return self.sentence, f"Low confidence ({confidence:.0%}) — plain wall, hand closer"

        now = time.time()

        if label != self._hold_label:
            if self._same_letter_on_cooldown(label):
                wait = self._repeat_cooldown_until - now
                return (
                    self.sentence,
                    f"Pause ~{wait:.1f}s (or show another letter), then hold {label} again",
                )
            self._hold_label = label
            self._hold_start = now
            return self.sentence, f"Holding {label}… keep steady ({confidence:.0%})"

        assert self._hold_start is not None
        elapsed = now - self._hold_start
        remaining = max(0.0, self.hold_seconds - elapsed)
        if elapsed < self.hold_seconds:
            return self.sentence, f"Holding {label}… {remaining:.1f}s to commit"

        # Commit (repeat letters allowed: ALL, BEE, etc.)
        repeat = bool(self.sentence) and self.sentence[-1] == label
        self.sentence += label
        self.last_committed = label
        self._reset_hold()
        # Short pause before same letter can be committed again (prevents one long hold → AA)
        self._repeat_cooldown_until = now + SAME_LETTER_GAP_SEC

        msg = f"Committed: {label}"
        if repeat:
            msg += " (again)"
        return self.sentence, msg
