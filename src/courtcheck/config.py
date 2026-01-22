"""Configuration utilities for CourtCheck."""

import os
from pathlib import Path


def get_default_config_path():
    """Get default Detectron2 config path if available."""
    # Check for common locations
    config_paths = [
        "assets/configs/keypoint_rcnn_R_50_FPN_3x.yaml",
        os.path.expanduser("~/.courtcheck/configs/keypoint_rcnn_R_50_FPN_3x.yaml"),
    ]
    for path in config_paths:
        if os.path.exists(path):
            return path
    return None


def get_default_weights_path():
    """Get default model weights path if available."""
    weights_paths = [
        "model_tennis_court_det.pt",
        "weights/model_tennis_court_det.pt",
        os.path.expanduser("~/.courtcheck/weights/model_tennis_court_det.pt"),
    ]
    for path in weights_paths:
        if os.path.exists(path):
            return path
    return None

