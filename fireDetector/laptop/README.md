# Laptop inference

This application runs the trained wildfire detector on a webcam, video file, or directory of images. It supports CPU inference and can use an NVIDIA CUDA or Apple MPS device when the local PyTorch installation supports it.

## Requirements

Use Python 3.10 or 3.11. Create a virtual environment from the repository root:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

Install PyTorch first using the command appropriate for the operating system and GPU from [the official PyTorch selector](https://pytorch.org/get-started/locally/). Then install the remaining dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r fireDetector/laptop/requirements.txt
```

## Model weights

The application expects a trained Ultralytics model at:

```text
fireDetector/models/best.pt
```

The `models/` directory and model binaries are ignored by Git. Generate `best.pt` by running the training notebook in `fireDetector/kaggle/`, or provide a compatible model through an approved private artifact-storage workflow.

## Commands

Run commands from `fireDetector/laptop/`:

```bash
# Webcam 0
python detect_fire.py

# Video file
python detect_fire.py --source path/to/video.mp4

# Image directory
python detect_fire.py --source path/to/images

# CPU, no preview window
python detect_fire.py --source path/to/video.mp4 --device cpu --no-display

# Do not save an annotated video
python detect_fire.py --source path/to/video.mp4 --no-save
```

Press **`q`** in the preview window to stop an interactive run. Annotated videos and image results are written to the local `fireDetector/data/` directory, which is intentionally ignored by Git.

## Configuration

Adjust defaults in `config.py` or override the most common options from the command line:

| Option | Purpose | Default |
|---|---|---:|
| `--device` | `auto`, `cpu`, `cuda:0`, or `mps` | `auto` |
| `--imgsz` | Inference image size | `640` |
| `--conf` | Detection confidence threshold | `0.45` |
| `--no-display` | Disable the OpenCV preview | off |
| `--no-save` | Disable annotated-video output | off |

For slower hardware, reduce `--imgsz` and increase `FRAME_SKIP` in `config.py`. For safety-critical use, do not rely on a single detection; validate the model on representative data and keep a human in the decision loop.

## Troubleshooting

If `best.pt` cannot be found, check that it exists exactly at `fireDetector/models/best.pt`. If `ultralytics` or `cv2` cannot be imported, activate the virtual environment and rerun the dependency installation. If a CUDA device is not detected, verify that the installed PyTorch build matches the machine's CUDA support; CPU mode remains available with `--device cpu`. If a webcam is unavailable, close other applications using it and try another index such as `--source 1`.
