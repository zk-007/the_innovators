from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DEFAULT_CHECKPOINT = MODELS_DIR / "convnext_tiny_fb_in22k_ft_in1k_fold0_best.pth"
REF_DIR = ROOT / "assets" / "dataset_reference"

CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "space", "del", "nothing",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Live demo tuning
CONFIDENCE_THRESHOLD = 0.55
DISPLAY_MIN_CONFIDENCE = 0.42
MIN_MARGIN = 0.08
HOLD_SECONDS = 1.0
SMOOTHING_WINDOW = 15
MIN_VOTE_CONFIDENCE = 0.35
TEMPORAL_FRAMES = 10
SPELLABLE_LABELS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Letters that score well on raw dataset images (use first in demo)
STRONG_LETTERS = frozenset({"B", "E", "F", "K", "M", "N", "W", "Y", "del", "nothing"})
