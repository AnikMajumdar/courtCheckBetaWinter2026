#!/usr/bin/env python
"""CLI entrypoint for court detection model training."""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from courtcheck.training.train import train_model
from courtcheck.training.dataset import verify_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train court keypoint detection model"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to Detectron2 config YAML file",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory containing dataset and annotations",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to train on (default: cuda)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=50000,
        help="Maximum number of training iterations (default: 50000)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint",
    )
    parser.add_argument(
        "--train-games",
        type=int,
        nargs="+",
        help="List of game numbers for training (e.g., --train-games 1 2 3)",
    )
    parser.add_argument(
        "--val-games",
        type=int,
        nargs="+",
        help="List of game numbers for validation (e.g., --val-games 4 5)",
    )
    parser.add_argument(
        "--base-lr",
        type=float,
        default=0.0001,
        help="Base learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--ims-per-batch",
        type=int,
        default=4,
        help="Images per batch (default: 4)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=11,
        help="Number of classes (default: 11)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify dataset integrity before training",
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    data_root = Path(args.data_root)
    if not data_root.exists():
        logger.error(f"Data root directory not found: {data_root}")
        sys.exit(1)

    # Build paths
    annotations_dir = data_root / "annotations" / "model_annotations" / "games" / "all_games"
    dataset_dir = data_root / "dataset"

    if not annotations_dir.exists():
        logger.error(f"Annotations directory not found: {annotations_dir}")
        sys.exit(1)

    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    # Build file lists
    if args.train_games is None or args.val_games is None:
        logger.error("--train-games and --val-games are required")
        sys.exit(1)

    train_json_files = [
        annotations_dir / f"game{num}.json" for num in args.train_games
    ]
    train_image_dirs = [
        dataset_dir / f"game{num}" / f"game{num}_images" for num in args.train_games
    ]

    val_json_files = [
        annotations_dir / f"game{num}.json" for num in args.val_games
    ]
    val_image_dirs = [
        dataset_dir / f"game{num}" / f"game{num}_images" for num in args.val_games
    ]

    # Verify datasets if requested
    if args.verify:
        logger.info("Verifying datasets...")
        train_valid = verify_dataset(
            [str(f) for f in train_json_files],
            [str(d) for d in train_image_dirs],
        )
        val_valid = verify_dataset(
            [str(f) for f in val_json_files],
            [str(d) for d in val_image_dirs],
        )
        if not (train_valid and val_valid):
            logger.warning("Dataset verification found issues, but continuing...")

    # Start training
    logger.info("Starting training...")
    try:
        train_model(
            config_path=args.config,
            train_json_files=[str(f) for f in train_json_files],
            train_image_dirs=[str(d) for d in train_image_dirs],
            val_json_files=[str(f) for f in val_json_files],
            val_image_dirs=[str(d) for d in val_image_dirs],
            output_dir=args.output_dir,
            max_iter=args.max_iter,
            resume=args.resume,
            base_lr=args.base_lr,
            ims_per_batch=args.ims_per_batch,
            num_workers=args.num_workers,
            num_classes=args.num_classes,
        )
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

