"""
Run fire detection on a video file, webcam, or folder of images.

Examples:
    python detect_fire.py                            # default webcam (index 0)
    python detect_fire.py --source 1                 # second webcam
    python detect_fire.py --source clip.mp4          # video file
    python detect_fire.py --source images/           # folder of images
    python detect_fire.py --device cpu               # force CPU
    python detect_fire.py --device cuda:0            # force a specific GPU
    python detect_fire.py --no-display               # headless (still saves)
    python detect_fire.py --no-save                  # don't write annotated video

Press 'q' in the preview window to quit.
"""
import argparse
import datetime
import os
import time

import cv2
import numpy as np

import config
from alert_system import AlertSystem
from model_loader import load_model, resolve_device, warmup
from video_io import AnnotatedWriter, FrameSource, KIND_IMAGES


class _NoopWriter:
    """Stand-in writer when --no-save is set."""
    path = None
    def write(self, frame): pass
    def release(self): pass


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_boxes(frame, result, fire_class_id):
    """Draw bounding boxes + labels on `frame`. Returns (frame, fire_detected)."""
    fire_detected = False
    if result is None or result.boxes is None:
        return frame, False

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        label  = result.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls_id == fire_class_id or label.lower() == 'fire':
            colour = (0, 0, 255)         # red
            fire_detected = True
        elif label.lower() == 'smoke':
            colour = (0, 128, 255)       # orange
        else:
            colour = (0, 255, 0)         # green

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(
            frame, f"{label} {conf:.2f}",
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2,
        )

    return frame, fire_detected


def draw_fire_banner(frame):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (0, 0, 200), -1)
    cv2.putText(
        frame, "FIRE DETECTED",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2,
    )


def draw_fps(frame, fps):
    cv2.putText(
        frame, f"FPS: {fps:.1f}",
        (frame.shape[1] - 140, 26),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
    )


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class FireDetector:
    def __init__(self, source: str, device: str, imgsz: int, conf_thr: float,
                 show_video: bool, save_video: bool):
        self.imgsz    = imgsz
        self.conf_thr = conf_thr
        self.show     = show_video

        # Resolve device + load model
        self.device = resolve_device(device)
        print(f"Device: {self.device}")
        self.model  = load_model()
        self.use_half = config.HALF_PRECISION and self.device.startswith('cuda')
        warmup(self.model, self.imgsz, self.device, self.use_half)

        # Inputs / outputs
        self.frames = FrameSource(source)
        self.writer = AnnotatedWriter(self.frames) if save_video else _NoopWriter()
        self.alert  = AlertSystem()

        os.makedirs(config.DETECTIONS_DIR, exist_ok=True)

        # Counters
        self._frame_count       = 0
        self._fire_frame_count  = 0
        self._fps_counter       = 0
        self._fps_timer         = time.time()
        self._current_fps       = 0.0
        self._start_time        = None

    # ------------------------------------------------------------------

    def _update_fps(self):
        self._fps_counter += 1
        elapsed = time.time() - self._fps_timer
        if elapsed >= 1.0:
            self._current_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_timer   = time.time()

    def _infer(self, frame):
        return self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf_thr,
            iou=config.IOU_THRESHOLD,
            device=self.device,
            half=self.use_half,
            verbose=False,
        )[0]

    def _save_snapshot(self, frame):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(config.DETECTIONS_DIR, f"fire_{ts}.jpg")
        cv2.imwrite(path, frame)

    def _save_image_result(self, frame, src_path):
        """For image-folder mode: write annotated image next to detections dir."""
        stem = os.path.splitext(os.path.basename(src_path))[0]
        ts   = datetime.datetime.now().strftime("%H%M%S_%f")
        path = os.path.join(config.DETECTIONS_DIR, f"{stem}_annotated_{ts}.jpg")
        cv2.imwrite(path, frame)

    # ------------------------------------------------------------------

    def run(self):
        print("\nRunning. Press 'q' in the window to quit.\n")
        last_result = None
        self._start_time = time.time()

        try:
            while True:
                ok, frame = self.frames.read()
                if not ok:
                    print("End of source.")
                    break

                self._frame_count += 1

                # Inference (with optional frame skipping, but never skip for image folders)
                run_now = (
                    self.frames.kind == KIND_IMAGES
                    or self._frame_count % config.FRAME_SKIP == 0
                )
                if run_now:
                    last_result = self._infer(frame)
                    self._update_fps()

                # Annotate
                annotated, fire_now = draw_boxes(frame.copy(), last_result, config.FIRE_CLASS_ID)
                if fire_now:
                    self._fire_frame_count += 1
                    draw_fire_banner(annotated)
                    self.alert.trigger()
                    if config.SAVE_DETECTION_SNAPSHOTS:
                        self._save_snapshot(annotated)

                draw_fps(annotated, self._current_fps)

                # Output
                self.writer.write(annotated)
                if self.frames.kind == KIND_IMAGES:
                    self._save_image_result(annotated, self.frames.current_image_path())

                if self.show:
                    cv2.imshow(config.WINDOW_NAME, annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Quit requested.")
                        break

        except KeyboardInterrupt:
            print("\nStopped by Ctrl+C.")
        finally:
            self._cleanup_and_summarise()

    def _cleanup_and_summarise(self):
        elapsed = time.time() - (self._start_time or time.time())
        avg_fps = self._frame_count / elapsed if elapsed > 0 else 0.0

        self.frames.release()
        self.writer.release()
        self.alert.cleanup()
        cv2.destroyAllWindows()

        print("\n" + "=" * 55)
        print(f"Frames processed   : {self._frame_count}")
        print(f"Frames with fire   : {self._fire_frame_count}")
        print(f"Elapsed            : {elapsed:.1f} s")
        print(f"Average FPS        : {avg_fps:.1f}")
        if self.writer.path:
            print(f"Annotated video    : {self.writer.path}")
        if self.frames.kind == KIND_IMAGES:
            print(f"Annotated images   : {config.DETECTIONS_DIR}")
        print("=" * 55)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Fire detection on laptop (CPU or GPU).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--source', default='0',
                   help='Webcam index (e.g. 0), path to a video file, or path to a folder of images. Default: 0')
    p.add_argument('--device', default=config.DEVICE,
                   help="'auto', 'cpu', 'cuda', 'cuda:0', or 'mps'. Default: auto")
    p.add_argument('--imgsz', type=int, default=config.INPUT_SIZE,
                   help=f"Inference input size. Default: {config.INPUT_SIZE}")
    p.add_argument('--conf', type=float, default=config.CONFIDENCE_THRESHOLD,
                   help=f"Confidence threshold (0-1). Default: {config.CONFIDENCE_THRESHOLD}")
    p.add_argument('--no-display', action='store_true',
                   help='Run without a preview window (still saves annotated video).')
    p.add_argument('--no-save', action='store_true',
                   help='Do not save an annotated video copy.')
    return p.parse_args()


def main():
    args = parse_args()
    detector = FireDetector(
        source=args.source,
        device=args.device,
        imgsz=args.imgsz,
        conf_thr=args.conf,
        show_video=not args.no_display and config.SHOW_VIDEO,
        save_video=not args.no_save and config.SAVE_ANNOTATED_VIDEO,
    )
    detector.run()


if __name__ == '__main__':
    main()
