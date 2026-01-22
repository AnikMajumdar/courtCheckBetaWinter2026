"""Dataset registration and verification utilities."""

import os
import json
import logging
from typing import List, Tuple

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

from courtcheck.court.metadata import (
    KEYPOINT_NAMES,
    KEYPOINT_FLIP_MAP,
    SKELETON,
)

logger = logging.getLogger(__name__)


def unregister_dataset(dataset_name: str):
    """Unregister a dataset if it already exists."""
    if dataset_name in DatasetCatalog.list():
        DatasetCatalog.pop(dataset_name)
        MetadataCatalog.pop(dataset_name)
        logger.info(f"Unregistered dataset: {dataset_name}")


def register_datasets(
    json_files: List[str],
    image_dirs: List[str],
    base_dataset_name: str = "tennis_game",
) -> List[str]:
    """
    Register COCO-format datasets for training/validation.

    Args:
        json_files: List of paths to COCO JSON annotation files
        image_dirs: List of paths to image directories
        base_dataset_name: Base name for dataset registration

    Returns:
        List of registered dataset names
    """
    registered_names = []

    for json_file, image_dir in zip(json_files, image_dirs):
        dataset_name = os.path.basename(json_file).split(".")[0]
        unregister_dataset(dataset_name)

        if not os.path.exists(json_file):
            logger.warning(f"JSON file not found: {json_file}, skipping")
            continue

        if not os.path.exists(image_dir):
            logger.warning(f"Image directory not found: {image_dir}, skipping")
            continue

        register_coco_instances(dataset_name, {}, json_file, image_dir)
        MetadataCatalog.get(dataset_name).keypoint_names = KEYPOINT_NAMES
        MetadataCatalog.get(dataset_name).keypoint_flip_map = KEYPOINT_FLIP_MAP
        MetadataCatalog.get(dataset_name).keypoint_connection_rules = SKELETON

        registered_names.append(dataset_name)
        logger.info(
            f"Registered dataset {dataset_name} with {json_file} and {image_dir}"
        )

    return registered_names


def verify_dataset(json_files: List[str], image_dirs: List[str]) -> bool:
    """
    Verify dataset integrity by checking that all images referenced in JSON
    files exist in the image directories.

    Args:
        json_files: List of paths to COCO JSON annotation files
        image_dirs: List of paths to image directories

    Returns:
        True if all datasets are valid, False otherwise
    """
    all_valid = True

    for json_file, image_dir in zip(json_files, image_dirs):
        if not os.path.exists(json_file):
            logger.warning(f"JSON file not found: {json_file}")
            all_valid = False
            continue

        if not os.path.exists(image_dir):
            logger.warning(f"Image directory not found: {image_dir}")
            all_valid = False
            continue

        with open(json_file) as f:
            data = json.load(f)

        image_files = {img["file_name"] for img in data["images"]}
        all_images = set(os.listdir(image_dir))
        missing_images = image_files - all_images
        extra_images = all_images - image_files

        if len(missing_images) > 0 or len(extra_images) > 0:
            logger.warning(
                f"{json_file}: {len(missing_images)} missing images, "
                f"{len(extra_images)} extra images"
            )
            if missing_images:
                logger.debug(f"Missing images: {list(missing_images)[:10]}")
            if extra_images:
                logger.debug(f"Extra images: {list(extra_images)[:10]}")
            all_valid = False
        else:
            logger.info(f"{json_file}: All images verified")

    return all_valid

