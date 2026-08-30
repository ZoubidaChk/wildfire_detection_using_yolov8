# AI-Based Wildfire Detection System

An end-to-end graduation project for detecting **fire and smoke in images and video** with a YOLOv8-based computer-vision pipeline. The repository contains the laptop inference application, the Kaggle training notebook, and a small set of selected result visualizations.

> **Publication status:** This is a local release candidate. It has been prepared for a future public GitHub repository, but it has **not** been published, committed, or pushed.

## What is included

| Component | Location | Purpose |
|---|---|---|
| Training notebook | [`fireDetector/kaggle/fire_detection_train.ipynb`](fireDetector/kaggle/fire_detection_train.ipynb) | Train a YOLOv8 model in a Kaggle notebook using a compatible YOLO-format dataset. |
| Laptop inference application | [`fireDetector/laptop/`](fireDetector/laptop/) | Run detection on a webcam, video file, or image directory. |
| Selected results | [`results/`](results/) | Shareable confusion matrix, labels visualization, and representative detection outputs. |
| Runtime documentation | [`fireDetector/laptop/README.md`](fireDetector/laptop/README.md) | Installation, commands, configuration, and troubleshooting. |

The trained model weights, original datasets, test videos, generated detection folders, private project files, and editor metadata are intentionally excluded from this release candidate. The model can be recreated by running the training notebook and placing the resulting `best.pt` file in a local `fireDetector/models/` directory.

## Project workflow

```text
YOLO-format dataset
        │
        ▼
Kaggle training notebook ──► best.pt
                                  │
                                  ▼
                    Laptop inference application
                       ├── images
                       ├── video files
                       └── webcam
                                  │
                                  ▼
                  annotated output + fire alerts
```

## Quick start

### 1. Clone and enter the repository

```bash
git clone <your-repository-url>
cd wildfiregraduation-public
```

### 2. Create an isolated Python environment

Python **3.10 or 3.11** is recommended.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

Install PyTorch using the command appropriate for the target machine, then install the application dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r fireDetector/laptop/requirements.txt
```

### 3. Add the trained weights locally

Run the Kaggle notebook, download the resulting `best.pt`, and place it at:

```text
fireDetector/models/best.pt
```

The weights directory is ignored by Git because trained binaries are large and are normally distributed separately. Do not commit model weights unless you have confirmed the repository's storage and licensing requirements.

### 4. Run inference

From `fireDetector/laptop/`:

```bash
python detect_fire.py --source path/to/video.mp4
python detect_fire.py --source path/to/image_directory
python detect_fire.py --source 0
```

Useful options include `--device cpu`, `--device cuda:0`, `--no-display`, and `--no-save`. Press **`q`** in the preview window to stop an interactive run.

## Reproducibility

The notebook documents the training workflow, but the exact dataset version, split, random seed, hardware, and training configuration should be recorded before making scientific performance claims. The files in `results/` are representative project outputs and should not be interpreted as a complete benchmark without the corresponding evaluation protocol and dataset details.

For a reproducible experiment, record the following in the notebook or a separate experiment log:

- Dataset name, version, source URL, license, and train/validation/test split.
- Python, PyTorch, Ultralytics, CUDA, and GPU versions.
- Model architecture, image size, epochs, batch size, augmentation, and random seed.
- Precision, confidence threshold, IoU threshold, and evaluation metrics.
- The exact commit or release used to produce the result files.

## Responsible use

This project is an educational prototype and should not be treated as a certified fire-safety system. Detection errors can occur because of lighting, smoke appearance, camera quality, occlusion, weather, domain shift, or dataset bias. Any real-world deployment would require extensive validation, human oversight, false-alarm analysis, and compliance with applicable safety requirements.

## Repository layout

```text
.
├── fireDetector/
│   ├── kaggle/
│   │   └── fire_detection_train.ipynb
│   └── laptop/
│       ├── alert_system.py
│       ├── config.py
│       ├── detect_fire.py
│       ├── model_loader.py
│       ├── requirements.txt
│       ├── video_io.py
│       └── README.md
├── results/
├── .gitignore
└── README.md
```

## Acknowledgements and citations

Before publishing, complete the acknowledgements in the training notebook and cite the dataset, Ultralytics/YOLOv8, and any source images or videos used in the project. Keep each dataset's original license and attribution requirements with the public repository.

## License

No license has been selected for this release candidate. Before making the repository public, choose a license that you are authorized to apply and add the corresponding license text as `LICENSE`. Until then, all rights remain reserved by the copyright holder(s).
