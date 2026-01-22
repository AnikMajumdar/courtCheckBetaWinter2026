#!/usr/bin/env python
"""CLI entrypoint for court keypoint inference."""

import argparse
import logging
import sys
from pathlib import Path

import cv2

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from courtcheck.court.infer import load_predictor, predict_image, predict_frames_dir
from courtcheck.court.visualize import visualize_predictions_with_lines

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run court keypoint detection inference on images"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to Detectron2 config YAML file",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to model weights file (.pth)",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input image file or directory containing images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for results (default: output)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on (default: cuda)",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.5,
        help="Score threshold for predictions (default: 0.5)",
    )
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        help="Save images with keypoint overlays",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save predictions as JSON files",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=11,
        help="Number of classes in the model (default: 11)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    if not Path(args.weights).exists():
        logger.error(f"Weights file not found: {args.weights}")
        sys.exit(1)

    if not Path(args.input).exists():
        logger.error(f"Input path not found: {args.input}")
        sys.exit(1)

    # Load predictor
    logger.info("Loading predictor...")
    try:
        predictor = load_predictor(
            args.config,
            args.weights,
            device=args.device,
            score_thresh=args.score_thresh,
            num_classes=args.num_classes,
        )
    except Exception as e:
        logger.error(f"Failed to load predictor: {e}")
        sys.exit(1)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine if input is a file or directory
    if input_path.is_file():
        # Single image inference
        logger.info(f"Processing single image: {input_path}")
        try:
            pred = predict_image(predictor, input_path)

            if args.save_json:
                json_path = output_path / f"{input_path.stem}.json"
                from courtcheck.court.infer import save_prediction_json

                save_prediction_json(pred, json_path)
                logger.info(f"Saved JSON to {json_path}")

            if args.save_overlay:
                img = cv2.imread(str(input_path))
                overlay = visualize_predictions_with_lines(img, predictor)
                overlay_path = output_path / f"{input_path.stem}_overlay.jpg"
                cv2.imwrite(str(overlay_path), overlay)
                logger.info(f"Saved overlay to {overlay_path}")

            logger.info("Inference completed successfully")
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            sys.exit(1)

    elif input_path.is_dir():
        # Directory inference
        logger.info(f"Processing directory: {input_path}")
        try:
            # Create visualization function
            def vis_fn(img, pred):
                return visualize_predictions_with_lines(img, pred)

            results = predict_frames_dir(
                predictor,
                input_path,
                output_dir=output_path,
                save_json=args.save_json,
                save_overlay=args.save_overlay,
                visualize_fn=vis_fn if args.save_overlay else None,
            )
            logger.info(f"Processed {len(results)} images")
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            sys.exit(1)
    else:
        logger.error(f"Input path is neither a file nor directory: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()

