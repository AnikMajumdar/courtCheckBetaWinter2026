"""Inference functions for court keypoint detection."""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

from courtcheck.court.metadata import (
    KEYPOINT_NAMES,
    KEYPOINT_FLIP_MAP,
    SKELETON,
)

logger = logging.getLogger(__name__)


def load_predictor(
    config_path: str,
    weights_path: str,
    device: str = "cuda",
    score_thresh: float = 0.5,
    num_classes: int = 11,
) -> DefaultPredictor:
    """
    Load a trained Detectron2 predictor.

    Args:
        config_path: Path to Detectron2 config YAML file
        weights_path: Path to model weights (.pth file)
        device: Device to run inference on ('cuda' or 'cpu')
        score_thresh: Score threshold for predictions
        num_classes: Number of classes in the model

    Returns:
        DefaultPredictor instance
    """
    cfg = get_cfg()
    
    # Try to load from file, otherwise use model zoo
    if os.path.exists(config_path):
        cfg.merge_from_file(config_path)
    else:
        # Fallback to model zoo config
        logger.warning(f"Config file not found at {config_path}, using model zoo config")
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.KEYPOINT_ON = True
    cfg.MODEL.DEVICE = device

    # Register metadata
    dataset_name = "tennis_court_inference"
    MetadataCatalog.get(dataset_name).keypoint_names = KEYPOINT_NAMES
    MetadataCatalog.get(dataset_name).keypoint_flip_map = KEYPOINT_FLIP_MAP
    MetadataCatalog.get(dataset_name).keypoint_connection_rules = SKELETON

    predictor = DefaultPredictor(cfg)
    logger.info(f"Loaded predictor from {weights_path} on {device}")
    return predictor


def predict_image(
    predictor: DefaultPredictor,
    image_path: Union[str, Path],
) -> Dict[str, Any]:
    """
    Run inference on a single image.

    Args:
        predictor: Loaded DefaultPredictor instance
        image_path: Path to input image

    Returns:
        Dictionary with predictions containing:
        - 'instances': Detectron2 Instances object
        - 'keypoints': numpy array of keypoints (N, 17, 3)
        - 'scores': confidence scores
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    outputs = predictor(img)
    instances = outputs["instances"].to("cpu")

    # Extract keypoints and scores
    keypoints = None
    scores = None
    if len(instances) > 0:
        # Get most confident instance
        max_conf_idx = instances.scores.argmax()
        instances = instances[max_conf_idx : max_conf_idx + 1]
        keypoints = instances.pred_keypoints.numpy()[0] if instances.has("pred_keypoints") else None
        scores = instances.scores.numpy()[0] if instances.has("scores") else None

    return {
        "instances": instances,
        "keypoints": keypoints,
        "scores": scores,
        "image_shape": img.shape,
    }


def predict_frames_dir(
    predictor: DefaultPredictor,
    frames_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    save_json: bool = False,
    save_overlay: bool = False,
    visualize_fn=None,
) -> List[Dict[str, Any]]:
    """
    Run inference on all images in a directory.

    Args:
        predictor: Loaded DefaultPredictor instance
        frames_dir: Directory containing input images
        output_dir: Optional output directory for results
        save_json: Whether to save predictions as JSON
        save_overlay: Whether to save images with overlays
        visualize_fn: Optional visualization function to apply

    Returns:
        List of prediction dictionaries
    """
    frames_dir = Path(frames_dir)
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    # Find all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted([
        f for f in frames_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    if len(image_files) == 0:
        logger.warning(f"No images found in {frames_dir}")
        return []

    results = []
    output_dir = Path(output_dir) if output_dir else None

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        if save_json:
            (output_dir / "json").mkdir(exist_ok=True)
        if save_overlay:
            (output_dir / "overlays").mkdir(exist_ok=True)

    for image_file in image_files:
        try:
            pred = predict_image(predictor, image_file)
            pred["image_path"] = str(image_file)
            results.append(pred)

            # Save JSON if requested
            if save_json and output_dir:
                json_path = output_dir / "json" / f"{image_file.stem}.json"
                save_prediction_json(pred, json_path)

            # Save overlay if requested
            if save_overlay and output_dir and visualize_fn:
                img = cv2.imread(str(image_file))
                if img is not None:
                    overlay = visualize_fn(img, predictor)
                    overlay_path = output_dir / "overlays" / image_file.name
                    cv2.imwrite(str(overlay_path), overlay)

        except Exception as e:
            logger.error(f"Error processing {image_file}: {e}")
            continue

    logger.info(f"Processed {len(results)} images from {frames_dir}")
    return results


def save_prediction_json(prediction: Dict[str, Any], output_path: Union[str, Path]):
    """Save prediction results as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable format
    result = {
        "image_shape": prediction.get("image_shape", []),
        "num_instances": len(prediction.get("instances", [])),
    }

    if prediction.get("keypoints") is not None:
        result["keypoints"] = prediction["keypoints"].tolist()

    if prediction.get("scores") is not None:
        result["score"] = float(prediction["scores"])

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

