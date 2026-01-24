#!/usr/bin/env python
"""Helper script to upload model weights to Modal volume."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import modal

app = modal.App("upload-weights")
vol = modal.Volume.from_name("tennis-weights", create_if_missing=True)


@app.function(volumes={"/weights": vol})
def upload_weights(local_path: str = "model_tennis_court_det.pt"):
    """Upload weights file to Modal volume."""
    import shutil
    from pathlib import Path

    local_file = Path(local_path)
    if not local_file.exists():
        raise FileNotFoundError(f"Weights file not found: {local_path}")

    # Copy to volume
    vol_path = f"/weights/{local_file.name}"
    shutil.copy(str(local_file), vol_path)
    
    # Commit the changes
    vol.commit()
    print(f"✅ Successfully uploaded {local_path} to Modal volume at {vol_path}")
    return vol_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload model weights to Modal")
    parser.add_argument(
        "--weights",
        type=str,
        default="model_tennis_court_det.pt",
        help="Path to weights file (default: model_tennis_court_det.pt)",
    )

    args = parser.parse_args()

    with app.run():
        upload_weights(args.weights)
