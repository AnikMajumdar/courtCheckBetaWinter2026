# CourtCheck: Tennis Court Keypoint Detection

A Python package for detecting tennis court keypoints using Detectron2. This project provides tools for training and inference on tennis court images.

## Features

- **Keypoint Detection**: Detect 17 keypoints on tennis courts
- **Visualization**: Overlay keypoints and court lines on images
- **Video Processing**: Create videos with keypoint overlays from frame sequences
- **Training Support**: Train custom models on your own datasets

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended) or CPU
- pip

### Setup

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd courtCheckBetaWinter2026
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Detectron2** (varies by CUDA version):
   
   For CUDA 11.8:
   ```bash
   pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
   ```
   
   For CUDA 11.7:
   ```bash
   pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu117/torch2.0/index.html
   ```
   
   For CPU only:
   ```bash
   pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html
   ```
   
   See [Detectron2 installation guide](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md) for other versions.

5. **Install the package in development mode**:
   ```bash
   pip install -e .
   ```
   
   Or add the `src` directory to your Python path:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

## Usage

### Inference on a Single Image

```bash
python scripts/infer.py \
    --config assets/configs/keypoint_rcnn_R_50_FPN_3x.yaml \
    --weights model_tennis_court_det.pt \
    --input path/to/image.jpg \
    --output output/ \
    --save-overlay \
    --save-json
```

### Inference on a Directory of Images

```bash
python scripts/infer.py \
    --config assets/configs/keypoint_rcnn_R_50_FPN_3x.yaml \
    --weights model_tennis_court_det.pt \
    --input path/to/frames/ \
    --output output/ \
    --save-overlay \
    --save-json
```

### Command Line Options

**Inference (`scripts/infer.py`)**:
- `--config`: Path to Detectron2 config YAML file (required)
- `--weights`: Path to model weights file (.pth) (required)
- `--input`: Input image file or directory (required)
- `--output`: Output directory for results (default: `output`)
- `--device`: Device to run on: `cuda` or `cpu` (default: `cuda`)
- `--score-thresh`: Score threshold for predictions (default: 0.5)
- `--save-overlay`: Save images with keypoint overlays
- `--save-json`: Save predictions as JSON files
- `--num-classes`: Number of classes in model (default: 11)

**Training (`scripts/train.py`)**:
- `--config`: Path to Detectron2 config YAML file (required)
- `--data-root`: Root directory containing dataset and annotations (required)
- `--output-dir`: Output directory for checkpoints (required)
- `--train-games`: List of game numbers for training (e.g., `1 2 3`)
- `--val-games`: List of game numbers for validation (e.g., `4 5`)
- `--device`: Device to train on: `cuda` or `cpu` (default: `cuda`)
- `--max-iter`: Maximum training iterations (default: 50000)
- `--resume`: Resume from last checkpoint
- `--verify`: Verify dataset integrity before training

### Using as a Python Package

```python
from courtcheck.court.infer import load_predictor, predict_image
from courtcheck.court.visualize import visualize_predictions_with_lines
import cv2

# Load predictor
predictor = load_predictor(
    config_path="assets/configs/keypoint_rcnn_R_50_FPN_3x.yaml",
    weights_path="model_tennis_court_det.pt",
    device="cuda",
    score_thresh=0.5,
)

# Run inference
img = cv2.imread("path/to/image.jpg")
pred = predict_image(predictor, "path/to/image.jpg")

# Visualize
overlay = visualize_predictions_with_lines(img, predictor)
cv2.imwrite("output.jpg", overlay)
```

## Project Structure

```
courtCheckBetaWinter2026/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── courtcheck/
│       ├── __init__.py
│       ├── config.py
│       └── court/
│           ├── __init__.py
│           ├── metadata.py      # Keypoint names, flip map, court lines
│           ├── infer.py          # Inference functions
│           ├── visualize.py      # Visualization functions
│           └── video.py          # Video creation utilities
│       └── training/
│           ├── __init__.py
│           ├── dataset.py        # Dataset registration
│           └── train.py          # Training utilities
├── scripts/
│   ├── infer.py                  # CLI for inference
│   └── train.py                  # CLI for training
└── assets/
    └── configs/                  # Detectron2 config files
```

## Keypoint Metadata

The model detects 17 keypoints on a tennis court:
- `BTL`, `BTLI`, `BTRI`, `BTR`: Top boundary points
- `BBR`, `BBRI`, `BBL`, `BBLI`: Bottom boundary points
- `ITL`, `ITM`, `ITR`: Inner top points
- `IBL`, `IBM`, `IBR`: Inner bottom points
- `NL`, `NM`, `NR`: Net points

See `src/courtcheck/court/metadata.py` for full definitions and court line connections.

## Dataset Format

Training requires COCO-format JSON annotation files and corresponding image directories. The expected structure is:

```
data_root/
├── annotations/
│   └── model_annotations/
│       └── games/
│           └── all_games/
│               ├── game1.json
│               ├── game2.json
│               └── ...
└── dataset/
    ├── game1/
    │   └── game1_images/
    │       ├── frame001.jpg
    │       └── ...
    └── game2/
        └── game2_images/
            └── ...
```

## Troubleshooting

### Detectron2 Installation Issues

If you encounter issues installing Detectron2:
1. Check your PyTorch and CUDA versions: `python -c "import torch; print(torch.__version__, torch.version.cuda)"`
2. Visit the [Detectron2 installation page](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md) for the correct wheel URL
3. For CPU-only systems, use the CPU wheel

### Import Errors

If you get import errors:
- Make sure you've added `src` to your `PYTHONPATH`, or
- Install the package in development mode: `pip install -e .`

### CUDA Out of Memory

If you run out of GPU memory:
- Reduce `--ims-per-batch` in training
- Process images in smaller batches during inference
- Use `--device cpu` for CPU inference (slower)

## License

[Add your license information here]

## Contact

For questions, please contact: corypham1@gmail.com

