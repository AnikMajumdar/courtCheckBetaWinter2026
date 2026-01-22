"""Training utilities for court detection model."""

import os
import logging
from typing import Optional

from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

from courtcheck.training.dataset import register_datasets

logger = logging.getLogger(__name__)


class TrainerWithEval(DefaultTrainer):
    """Custom trainer with COCO evaluation support."""

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


def train_model(
    config_path: str,
    train_json_files: list,
    train_image_dirs: list,
    val_json_files: list,
    val_image_dirs: list,
    output_dir: str,
    max_iter: int = 50000,
    resume: bool = False,
    base_lr: float = 0.0001,
    ims_per_batch: int = 4,
    num_workers: int = 4,
    num_classes: int = 11,
    checkpoint_period: int = 20000,
) -> TrainerWithEval:
    """
    Train a Detectron2 model for court keypoint detection.

    Args:
        config_path: Path to Detectron2 config YAML file
        train_json_files: List of training JSON annotation files
        train_image_dirs: List of training image directories
        val_json_files: List of validation JSON annotation files
        val_image_dirs: List of validation image directories
        output_dir: Output directory for checkpoints and logs
        max_iter: Maximum number of training iterations
        resume: Whether to resume from last checkpoint
        base_lr: Base learning rate
        ims_per_batch: Images per batch
        num_workers: Number of data loading workers
        num_classes: Number of classes
        checkpoint_period: Iterations between checkpoints

    Returns:
        Trained TrainerWithEval instance
    """
    # Register datasets
    train_names = register_datasets(train_json_files, train_image_dirs, "tennis_game_train")
    val_names = register_datasets(val_json_files, val_image_dirs, "tennis_game_val")

    if not train_names:
        raise ValueError("No training datasets registered")
    if not val_names:
        raise ValueError("No validation datasets registered")

    # Setup config
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.DATASETS.TRAIN = tuple(train_names)
    cfg.DATASETS.TEST = tuple(val_names)
    cfg.DATALOADER.NUM_WORKERS = num_workers
    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = [int(max_iter * 0.75), int(max_iter * 0.875)]
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = TrainerWithEval(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()

    logger.info(f"Training completed. Model saved to {output_dir}")
    return trainer

