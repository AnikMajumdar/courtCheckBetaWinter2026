"""Visualization functions for court keypoint predictions."""

import logging
from collections import deque
from typing import Optional

import cv2
import numpy as np
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode

from courtcheck.court.metadata import (
    KEYPOINT_NAMES,
    COURT_LINES,
    LINE_COLORS,
)

logger = logging.getLogger(__name__)

# Keypoint history for stabilization (global state)
_keypoint_history = {name: deque(maxlen=10) for name in KEYPOINT_NAMES}


def reset_keypoint_history():
    """Reset the keypoint stabilization history."""
    global _keypoint_history
    _keypoint_history = {name: deque(maxlen=10) for name in KEYPOINT_NAMES}


def stabilize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    Stabilize keypoints using temporal smoothing.

    Args:
        keypoints: Array of keypoints (N, 3) where last dim is (x, y, visibility)

    Returns:
        Stabilized keypoints array
    """
    stabilized = []
    for i, keypoint in enumerate(keypoints):
        _keypoint_history[KEYPOINT_NAMES[i]].append(keypoint[:2])
        if len(_keypoint_history[KEYPOINT_NAMES[i]]) > 0:
            stabilized.append(np.mean(_keypoint_history[KEYPOINT_NAMES[i]], axis=0))
        else:
            stabilized.append(keypoint[:2])
    return np.array(stabilized)


def visualize_predictions_with_lines(
    img: np.ndarray,
    predictor,
    keypoint_names: Optional[list] = None,
    lines: Optional[list] = None,
    dataset_name: str = "tennis_court_inference",
    stabilize: bool = True,
) -> np.ndarray:
    """
    Visualize predictions with keypoint labels and court lines.

    Args:
        img: Input image (BGR format)
        predictor: Detectron2 predictor
        keypoint_names: Optional list of keypoint names (uses default if None)
        lines: Optional list of line connections (uses default if None)
        dataset_name: Dataset name for metadata
        stabilize: Whether to apply keypoint stabilization

    Returns:
        Image with overlays (BGR format)
    """
    if keypoint_names is None:
        keypoint_names = KEYPOINT_NAMES
    if lines is None:
        lines = COURT_LINES

    outputs = predictor(img)
    v = Visualizer(
        img[:, :, ::-1],
        metadata=MetadataCatalog.get(dataset_name),
        scale=0.8,
        instance_mode=ColorMode.IMAGE,
    )
    instances = outputs["instances"].to("cpu")

    if len(instances) > 0:
        max_conf_idx = instances.scores.argmax()
        instances = instances[max_conf_idx : max_conf_idx + 1]

    out = v.draw_instance_predictions(instances)
    keypoints = instances.pred_keypoints.numpy()[0] if len(instances) > 0 else None
    scores = instances.scores if instances.has("scores") else [1.0]

    if keypoints is None:
        return img

    label_offset_x = 5
    label_offset_y = -10

    img_copy = img.copy()

    if stabilize:
        stabilized_keypoints = stabilize_keypoints(keypoints)
    else:
        stabilized_keypoints = keypoints[:, :2]

    for idx, (keypoints_per_instance, score) in enumerate(zip([stabilized_keypoints], scores)):
        average_kp_score = 0
        for j, keypoint in enumerate(keypoints_per_instance):
            x, y = keypoint
            kp_score = keypoints[j, 2] if len(keypoints[j]) > 2 else 1.0
            label = keypoint_names[j]
            kp_score = max(0, min(1, kp_score))
            average_kp_score += kp_score
            if kp_score > 0:
                cv2.putText(
                    img_copy,
                    label,
                    (int(x) + label_offset_x, int(y) + label_offset_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.circle(img_copy, (int(x), int(y)), 3, (0, 0, 255), -1)

        average_kp_score /= len(keypoints_per_instance)
        average_kp_score = max(0, min(1, average_kp_score)) * 100
        cv2.putText(
            img_copy,
            f"Confidence: {average_kp_score:.2f}%",
            (10, 30 + 30 * idx),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # Draw court lines
        for (start, end), color in zip(lines, LINE_COLORS):
            try:
                start_idx = keypoint_names.index(start)
                end_idx = keypoint_names.index(end)
                if start_idx < len(stabilized_keypoints) and end_idx < len(stabilized_keypoints):
                    cv2.line(
                        img_copy,
                        (
                            int(stabilized_keypoints[start_idx][0]),
                            int(stabilized_keypoints[start_idx][1]),
                        ),
                        (
                            int(stabilized_keypoints[end_idx][0]),
                            int(stabilized_keypoints[end_idx][1]),
                        ),
                        color,
                        2,
                    )
            except (ValueError, IndexError):
                continue

    return img_copy


def visualize_predictions_with_labels(
    img: np.ndarray,
    predictor,
    keypoint_names: Optional[list] = None,
    dataset_name: str = "tennis_court_inference",
) -> np.ndarray:
    """
    Visualize predictions with keypoint labels and confidence scores.

    Args:
        img: Input image (BGR format)
        predictor: Detectron2 predictor
        keypoint_names: Optional list of keypoint names (uses default if None)
        dataset_name: Dataset name for metadata

    Returns:
        Image with overlays (BGR format)
    """
    if keypoint_names is None:
        keypoint_names = KEYPOINT_NAMES

    outputs = predictor(img)
    v = Visualizer(
        img[:, :, ::-1],
        metadata=MetadataCatalog.get(dataset_name),
        scale=0.8,
        instance_mode=ColorMode.IMAGE,
    )
    instances = outputs["instances"].to("cpu")

    # Filter to keep only the most confident instance
    if len(instances) > 0:
        max_conf_idx = instances.scores.argmax()
        instances = instances[max_conf_idx : max_conf_idx + 1]

    out = v.draw_instance_predictions(instances)

    # Draw keypoints and labels
    keypoints = instances.pred_keypoints if len(instances) > 0 else None
    scores = instances.scores if instances.has("scores") else [1.0] * len(instances)

    if keypoints is None:
        return img

    label_offset_x = 10
    label_offset_y = -20
    percentage_offset_y = -5

    for idx, (keypoints_per_instance, score) in enumerate(zip(keypoints, scores)):
        average_kp_score = 0
        for j, keypoint in enumerate(keypoints_per_instance):
            x, y, kp_score = keypoint
            label = keypoint_names[j]
            kp_score = max(0, min(1, kp_score))
            average_kp_score += kp_score
            if kp_score > 0:
                # Draw background box for label
                label_text = f"{label}"
                label_size, _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                label_bg_tl = (
                    int(x) + label_offset_x,
                    int(y) + label_offset_y - label_size[1],
                )
                label_bg_br = (
                    int(x) + label_offset_x + label_size[0],
                    int(y) + label_offset_y + 4,
                )
                cv2.rectangle(img, label_bg_tl, label_bg_br, (0, 0, 0), cv2.FILLED)
                cv2.putText(
                    img,
                    label_text,
                    (int(x) + label_offset_x, int(y) + label_offset_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # Draw background box for percentage
                percentage_text = f"({kp_score * 100:.2f}%)"
                percentage_size, _ = cv2.getTextSize(
                    percentage_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                )
                percentage_bg_tl = (
                    int(x) + label_offset_x,
                    int(y) + percentage_offset_y - percentage_size[1],
                )
                percentage_bg_br = (
                    int(x) + label_offset_x + percentage_size[0],
                    int(y) + percentage_offset_y + 2,
                )
                cv2.rectangle(
                    img, percentage_bg_tl, percentage_bg_br, (0, 0, 0), cv2.FILLED
                )
                cv2.putText(
                    img,
                    percentage_text,
                    (int(x) + label_offset_x, int(y) + percentage_offset_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)

        # Compute average keypoint score
        average_kp_score /= len(keypoints_per_instance)
        cv2.putText(
            img,
            f"Confidence: {average_kp_score * 100:.2f}%",
            (10, 30 + 30 * idx),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return img

