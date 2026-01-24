# modal/app.py
import base64
import io
from typing import Any, Dict

import modal

app = modal.App("courtcheck-inference")

# Persistent storage for weights
weights_vol = modal.Volume.from_name("tennis-weights")
WEIGHTS_PATH = "/weights/model_tennis_court_det.pt"
CONFIG_PATH = "/app/assets/configs/Base-Keypoint-RCNN-FPN.yaml"
NUM_CLASSES = 11

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")  # cv2 runtime deps
    .pip_install(
        "numpy",
        "opencv-python",
        "pillow",
        "matplotlib",
        "fastapi",
        "pydantic",
        "torch",
        "torchvision",
    )
    # Reliable install: build detectron2 from source
    .pip_install("git+https://github.com/facebookresearch/detectron2.git")
    # Copy your repo code into the container
    .add_local_dir("src", remote_path="/app/src")
    .add_local_dir("assets", remote_path="/app/assets")
)

# Keep predictor warm per container
_predictor = None

def _load_predictor():
    global _predictor
    if _predictor is not None:
        return _predictor

    import sys
    sys.path.append("/app/src")

    from courtcheck.court.infer import load_predictor  # your refactored function

    _predictor = load_predictor(
        CONFIG_PATH,
        WEIGHTS_PATH,
        device="cuda",
        score_thresh=0.5,
        num_classes=NUM_CLASSES,
        )

    return _predictor


@app.function(
    image=image,
    gpu="L4",
    cpu=2,
    memory=8192,  # MiB (8 GiB)
    timeout=60 * 10,
    volumes={"/weights": weights_vol.read_only()},
)
@modal.fastapi_endpoint(method="POST")
def predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    payload: {"image_base64": "<base64-encoded image bytes>"}
    returns: predictions as JSON-serializable dict
    """
    import numpy as np
    import cv2

    predictor = _load_predictor()

    b64 = payload["image_base64"]
    img_bytes = base64.b64decode(b64)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    outputs = predictor(img_bgr)

    # Convert Detectron2 Instances to something JSON-friendly.
    # Keep it minimal: keypoints + scores + boxes if present.
    inst = outputs["instances"].to("cpu")

    result: Dict[str, Any] = {
        "num_instances": int(len(inst)),
        "scores": inst.scores.tolist() if inst.has("scores") else [],
        "pred_boxes": inst.pred_boxes.tensor.tolist() if inst.has("pred_boxes") else [],
    }

    if inst.has("pred_keypoints"):
        # shape: [N, K, 3] -> (x, y, prob)
        result["pred_keypoints"] = inst.pred_keypoints.tolist()

    return result
