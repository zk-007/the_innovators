"""Hold-to-commit spelling logic for Phase 2 word builder."""

from __future__ import annotations

import time
from collections import deque

from .config import HOLD_SECONDS, SPELLABLE_LABELS


class SpellingState:
    def __init__(self, hold_seconds: float = HOLD_SECONDS):
        self.hold_seconds = hold_seconds
        self.sentence = ""
        self.recent: deque[str] = deque(maxlen=8)
        self._hold_label: str | None = None
        self._hold_start: float | None = None
        self.last_committed = ""

    def reset_sentence(self) -> str:
        self.sentence = ""
        self._reset_hold()
        return self.sentence

    def _reset_hold(self) -> None:
        self._hold_label = None
        self._hold_start = None

    def update(self, label: str, confidence: float, threshold: float) -> tuple[str, str]:
        """Returns (sentence, status_message)."""
        if label == "del":
            if self.sentence:
                self.sentence = self.sentence[:-1]
            self._reset_hold()
            return self.sentence, "Deleted last character"

        if label == "space":
            if self.sentence and not self.sentence.endswith(" "):
                self.sentence += " "
            self._reset_hold()
            return self.sentence, "Added space"

        if label == "nothing" or label not in SPELLABLE_LABELS:
            self._reset_hold()
            return self.sentence, f"Showing: {label} ({confidence:.0%}) — hold A–Z to spell"

        if confidence < threshold:
            self._reset_hold()
            return self.sentence, f"Low confidence ({confidence:.0%}) — clearer sign / lighting"

        now = time.time()
        if label != self._hold_label:
            self._hold_label = label
            self._hold_start = now
            return self.sentence, f"Holding {label}… keep steady ({confidence:.0%})"

        assert self._hold_start is not None
        elapsed = now - self._hold_start
        remaining = max(0.0, self.hold_seconds - elapsed)
        if elapsed < self.hold_seconds:
            return self.sentence, f"Holding {label}… {remaining:.1f}s to commit"

        if not self.sentence or self.sentence[-1] != label:
            self.sentence += label
            self.last_committed = label
            self._reset_hold()
            return self.sentence, f"Committed: {label}"

        self._reset_hold()
        return self.sentence, f"Already added {label}"

    def smooth_label(self, label: str) -> str:
        self.recent.append(label)
        counts: dict[str, int] = {}
        for item in self.recent:
            counts[item] = counts.get(item, 0) + 1
        return max(counts, key=counts.get)
