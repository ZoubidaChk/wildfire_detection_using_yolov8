"""
Frame source (webcam / video file / image folder) and annotated-video writer.
"""
import os
import time
from pathlib import Path

import cv2

import config

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# Source kinds
KIND_WEBCAM   = 'webcam'
KIND_VIDEO    = 'video'
KIND_IMAGES   = 'images'


class FrameSource:
    """
    One class, three kinds of input.
    Use `kind` to tell which one you got.
    """

    def __init__(self, source: str):
        self.source = source
        self._cap = None
        self._image_paths = None
        self._image_idx   = 0

        self.kind = None
        self.fps    = 30.0
        self.width  = config.WEBCAM_WIDTH
        self.height = config.WEBCAM_HEIGHT
        self.total_frames = 0   # 0 for live / unknown

        if source.isdigit():
            self._open_webcam(int(source))
        elif os.path.isdir(source):
            self._open_image_folder(source)
        elif os.path.isfile(source):
            self._open_video_file(source)
        else:
            raise FileNotFoundError(
                f"Source not understood: {source!r}\n"
                f"Expected a webcam index (e.g. 0), a path to a video file, "
                f"or a path to a folder of images."
            )

    # ------------------------------------------------------------------ open

    def _open_webcam(self, index: int):
        backend = cv2.CAP_DSHOW if os.name == 'nt' else 0
        self._cap = cv2.VideoCapture(index, backend)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam {index}. Is another app using it "
                f"(Zoom, Teams, Discord)? Try a different --source number."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.WEBCAM_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WEBCAM_HEIGHT)
        # Actual settings the camera accepted
        self.width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or config.WEBCAM_WIDTH
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or config.WEBCAM_HEIGHT
        self.fps    = float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.kind   = KIND_WEBCAM
        print(f"Webcam {index} opened: {self.width}x{self.height} @ {self.fps:.0f} fps")

    def _open_video_file(self, path: str):
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video file: {path}")
        self.width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps    = float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.kind   = KIND_VIDEO
        print(f"Video opened: {path} — {self.width}x{self.height} @ {self.fps:.0f} fps, "
              f"{self.total_frames} frames")

    def _open_image_folder(self, folder: str):
        files = sorted(Path(folder).iterdir())
        self._image_paths = [str(p) for p in files if p.suffix.lower() in IMAGE_EXTS]
        if not self._image_paths:
            raise FileNotFoundError(f"No images ({sorted(IMAGE_EXTS)}) in {folder}")
        # Probe first image for size
        first = cv2.imread(self._image_paths[0])
        if first is None:
            raise RuntimeError(f"Could not read first image: {self._image_paths[0]}")
        self.height, self.width = first.shape[:2]
        self.total_frames = len(self._image_paths)
        self.kind = KIND_IMAGES
        print(f"Image folder opened: {folder} — {self.total_frames} images, "
              f"first is {self.width}x{self.height}")

    # ------------------------------------------------------------------ read

    def read(self):
        """Returns (ok, frame) — same shape as cv2.VideoCapture.read()."""
        if self.kind == KIND_IMAGES:
            if self._image_idx >= len(self._image_paths):
                return False, None
            path = self._image_paths[self._image_idx]
            self._image_idx += 1
            frame = cv2.imread(path)
            if frame is not None and frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame is not None, frame
        ok, frame = self._cap.read()
        if ok and frame is not None and frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return ok, frame

    def current_image_path(self):
        """Path of the most recently returned image (image-folder mode only)."""
        if self.kind != KIND_IMAGES or self._image_idx == 0:
            return None
        return self._image_paths[self._image_idx - 1]

    def release(self):
        if self._cap is not None:
            self._cap.release()


# ---------------------------------------------------------------------------
# Annotated video writer
# ---------------------------------------------------------------------------

class AnnotatedWriter:
    """
    Wraps cv2.VideoWriter. Picks a sensible output path based on the source.
    Use `write(frame)` per frame; call `release()` at the end.
    """

    def __init__(self, source: 'FrameSource'):
        self._writer = None
        self.path    = None

        if source.kind == KIND_IMAGES:
            # No video output for image folders — snapshots go into DETECTIONS_DIR.
            return

        if source.kind == KIND_VIDEO:
            # Save next to input as <name>.annotated.mp4
            inp = Path(source.source)
            self.path = str(inp.with_suffix('')) + '.annotated.mp4'
        else:  # webcam
            os.makedirs(config.DATA_DIR, exist_ok=True)
            ts = time.strftime('%Y%m%d_%H%M%S')
            self.path = os.path.join(config.DATA_DIR, f'webcam_{ts}.annotated.mp4')

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._writer = cv2.VideoWriter(
            self.path, fourcc, source.fps,
            (source.width, source.height),
        )
        if not self._writer.isOpened():
            print(f"WARNING: could not open video writer for {self.path} — "
                  f"annotated video will not be saved.")
            self._writer = None
            self.path = None
        else:
            print(f"Saving annotated video to: {self.path}")

    def write(self, frame):
        if self._writer is not None:
            self._writer.write(frame)

    def release(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
