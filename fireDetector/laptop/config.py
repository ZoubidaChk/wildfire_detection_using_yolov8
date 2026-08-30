"""
Settings for the laptop fire detector.
Tweak numbers here without touching the rest of the code.
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', '..', 'models')
DATA_DIR   = os.path.join(BASE_DIR, '..', 'data')

# Where to save snapshots / webcam recordings
DETECTIONS_DIR = os.path.join(DATA_DIR, 'detections')

# Model filename (placed inside MODELS_DIR)
MODEL_FILENAME = 'best.pt'

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
# Input size the model sees. 640 = same as training, most accurate.
# Drop to 416 or 320 on a slow CPU for more FPS.
INPUT_SIZE = 640

# How sure the model must be before drawing a box.
# Lower -> more detections, more false positives.
# Higher -> fewer false positives, may miss real fire.
CONFIDENCE_THRESHOLD = 0.45

# Non-max suppression threshold. Rarely needs changing.
IOU_THRESHOLD = 0.45

# Class index for "fire" — single-class dataset uses 0.
FIRE_CLASS_ID = 0

# 'auto' picks the best available: cuda -> mps -> cpu.
# You can force a specific one: 'cpu', 'cuda', 'cuda:0', 'mps'.
DEVICE = 'auto'

# Use 16-bit floats on NVIDIA GPUs. Big speedup, no accuracy loss
# for inference. Ignored on CPU and Apple MPS.
HALF_PRECISION = True

# Run inference on every Nth frame (1 = every frame).
# Display still updates every frame using the most recent boxes.
# Raise to 2 or 3 if CPU can't keep up.
FRAME_SKIP = 1

# ---------------------------------------------------------------------------
# Webcam capture settings (only used when source is a webcam index)
# ---------------------------------------------------------------------------
WEBCAM_WIDTH  = 1280
WEBCAM_HEIGHT = 720

# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------
# Seconds to wait before alerting again after a detection.
ALERT_COOLDOWN_SEC = 5.0

# Play a system beep when fire is detected.
PLAY_SOUND = True

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
# Show a live preview window. Disable with --no-display on the command line.
SHOW_VIDEO = True

# Save an annotated copy of video / webcam input alongside the original.
SAVE_ANNOTATED_VIDEO = True

# Also save a JPG snapshot every time fire is detected (in addition to video).
SAVE_DETECTION_SNAPSHOTS = False

# Window title
WINDOW_NAME = "Fire Detection"
