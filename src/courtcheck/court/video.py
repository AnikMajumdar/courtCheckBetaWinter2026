"""Video creation utilities for court keypoint visualization."""

import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from courtcheck.court.visualize import visualize_predictions_with_lines
from courtcheck.court.metadata import KEYPOINT_NAMES

logger = logging.getLogger(__name__)


def create_video_from_frames(
    output_dir: Union[str, Path],
    dataset_path: Union[str, Path],
    predictor,
    keypoint_names: Optional[list] = None,
    lines: Optional[list] = None,
    dataset_name: str = "tennis_court_inference",
    video_name: Optional[str] = None,
    fps: int = 30,
) -> Path:
    """
    Create a video from visualized frames.

    Args:
        output_dir: Directory to save the video
        dataset_path: Directory containing input images
        predictor: Detectron2 predictor
        keypoint_names: Optional list of keypoint names
        lines: Optional list of line connections
        dataset_name: Dataset name for metadata
        video_name: Optional video filename (auto-generated if None)
        fps: Frames per second for output video

    Returns:
        Path to created video file
    """
    output_dir = Path(output_dir)
    dataset_path = Path(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if keypoint_names is None:
        keypoint_names = KEYPOINT_NAMES

    # Find all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted([
        f for f in dataset_path.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    if len(image_files) == 0:
        raise ValueError(f"No images found in {dataset_path}")

    # Generate video name if not provided
    if video_name is None:
        video_name = f"court_detection_{dataset_path.name}.mp4"

    video_path = output_dir / video_name

    # Get dimensions from first image
    first_image_path = dataset_path / image_files[0]
    first_image = cv2.imread(str(first_image_path))
    if first_image is None:
        raise ValueError(f"Could not read first image: {first_image_path}")

    height, width, _ = first_image.shape

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    if not video.isOpened():
        raise RuntimeError(f"Could not open video writer for {video_path}")

    # Process each frame
    for i, image_file in enumerate(image_files):
        img_path = dataset_path / image_file
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Could not read image: {img_path}")
            continue

        visualized_img = visualize_predictions_with_lines(
            img, predictor, keypoint_names, lines, dataset_name
        )
        video.write(visualized_img)

        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(image_files)} frames")

    video.release()
    logger.info(f"Video saved to {video_path}")
    return video_path


def transform_keypoints_to_2d(keypoints: np.ndarray, keypoint_names: list) -> np.ndarray:
    """
    Transform keypoints to a fixed 2D plane using perspective transformation.

    Args:
        keypoints: Array of keypoints (N, 3) with (x, y, visibility)
        keypoint_names: List of keypoint names

    Returns:
        Transformed keypoints in 2D plane
    """
    keypoint_dict = {keypoint_names[i]: keypoints[i, :2] for i in range(len(keypoint_names))}

    src_points = np.array(
        [
            keypoint_dict["BTL"],
            keypoint_dict["BTR"],
            keypoint_dict["BBL"],
            keypoint_dict["BBR"],
        ],
        dtype=np.float32,
    )

    dst_points = np.array(
        [[50, 50], [350, 50], [50, 550], [350, 550]], dtype=np.float32
    )

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed_keypoints = cv2.perspectiveTransform(
        keypoints[None, :, :2], matrix
    )[0]

    return transformed_keypoints


def visualize_2d_court_skeleton(
    transformed_keypoints: np.ndarray,
    lines: list,
    keypoint_names: list,
    image_size: tuple = (600, 400),
) -> np.ndarray:
    """
    Visualize 2D court skeleton from transformed keypoints.

    Args:
        transformed_keypoints: Transformed keypoints array
        lines: List of line connections
        keypoint_names: List of keypoint names
        image_size: Output image size (height, width)

    Returns:
        Blank image with court skeleton drawn
    """
    blank_image = np.zeros((image_size[0], image_size[1], 3), np.uint8)

    # Draw lines between keypoints
    for start, end in lines:
        try:
            start_idx = keypoint_names.index(start)
            end_idx = keypoint_names.index(end)
            start_point = tuple(map(int, transformed_keypoints[start_idx]))
            end_point = tuple(map(int, transformed_keypoints[end_idx]))
            cv2.line(blank_image, start_point, end_point, (255, 255, 255), 2)
        except (ValueError, IndexError):
            continue

    # Draw keypoints
    for point in transformed_keypoints:
        point = tuple(map(int, point))
        cv2.circle(blank_image, point, 5, (0, 0, 255), -1)

    return blank_image


def create_2d_video_from_frames(
    output_dir: Union[str, Path],
    dataset_path: Union[str, Path],
    predictor,
    keypoint_names: Optional[list] = None,
    lines: Optional[list] = None,
    dataset_name: str = "tennis_court_inference",
    video_name: Optional[str] = None,
    fps: int = 30,
    image_size: tuple = (600, 400),
) -> Path:
    """
    Create a 2D court skeleton video from frames.

    Args:
        output_dir: Directory to save the video
        dataset_path: Directory containing input images
        predictor: Detectron2 predictor
        keypoint_names: Optional list of keypoint names
        lines: Optional list of line connections
        dataset_name: Dataset name for metadata
        video_name: Optional video filename
        fps: Frames per second for output video
        image_size: Output video size (height, width)

    Returns:
        Path to created video file
    """
    from courtcheck.court.metadata import COURT_LINES

    output_dir = Path(output_dir)
    dataset_path = Path(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if keypoint_names is None:
        keypoint_names = KEYPOINT_NAMES
    if lines is None:
        lines = COURT_LINES

    # Find all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted([
        f for f in dataset_path.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    if len(image_files) == 0:
        raise ValueError(f"No images found in {dataset_path}")

    # Generate video name if not provided
    if video_name is None:
        video_name = f"2d_court_skeleton_{dataset_path.name}.mp4"

    video_path = output_dir / video_name

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(
        str(video_path), fourcc, fps, (image_size[1], image_size[0])
    )

    if not video.isOpened():
        raise RuntimeError(f"Could not open video writer for {video_path}")

    # Process each frame
    for i, image_file in enumerate(image_files):
        img_path = dataset_path / image_file
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Could not read image: {img_path}")
            continue

        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")

        if len(instances) > 0:
            max_conf_idx = instances.scores.argmax()
            keypoints = instances.pred_keypoints.numpy()[max_conf_idx]
            transformed_keypoints = transform_keypoints_to_2d(keypoints, keypoint_names)
            court_skeleton = visualize_2d_court_skeleton(
                transformed_keypoints, lines, keypoint_names, image_size
            )
            video.write(court_skeleton)

        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(image_files)} frames")

    video.release()
    logger.info(f"2D video saved to {video_path}")
    return video_path

