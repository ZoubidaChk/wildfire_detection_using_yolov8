"""
Model loading and device selection — kept separate so detect_fire.py
stays focused on the main loop.
"""
import os
import numpy as np
from ultralytics import YOLO

import config


def resolve_device(requested: str) -> str:
    """
    Turn 'auto' into a real device string.
    Any non-'auto' value is passed through unchanged so users can force
    a specific device with --device.
    """
    if requested != 'auto':
        return requested

    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda:0'
        mps = getattr(torch.backends, 'mps', None)
        if mps is not None and mps.is_available():
            return 'mps'
    except ImportError:
        # No torch installed — ultralytics will fail later with a clearer error
        pass

    return 'cpu'


def load_model() -> YOLO:
    """Load best.pt from the models/ folder. Loud, clear error if missing."""
    model_path = os.path.join(config.MODELS_DIR, config.MODEL_FILENAME)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Could not find {config.MODEL_FILENAME} at:\n"
            f"  {os.path.abspath(model_path)}\n\n"
            f"Train the model on Kaggle (see kaggle/fire_detection_train.ipynb),\n"
            f"download best.pt, and drop it into the models/ folder."
        )
    print(f"Loading model: {model_path}")
    return YOLO(model_path)


def warmup(model: YOLO, imgsz: int, device: str, half: bool) -> None:
    """
    Run one prediction on a dummy frame so the first real frame
    isn't slow (PyTorch lazily initialises CUDA on first use).
    """
    print("Warming up...")
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    model.predict(
        source=dummy,
        imgsz=imgsz,
        device=device,
        half=half,
        verbose=False,
    )
    print("Ready.")
