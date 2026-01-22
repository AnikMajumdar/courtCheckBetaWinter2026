# Detectron2 Configuration Files

Place your Detectron2 config YAML files here.

## Default Config

The default config used is:
- `COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml`

You can download this from the Detectron2 model zoo or use your own custom config.

## Getting the Default Config

The default config can be obtained from Detectron2's model zoo:

```python
from detectron2 import model_zoo
config_path = model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
```

Or download it directly from:
https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml

Save it to this directory as `keypoint_rcnn_R_50_FPN_3x.yaml` for local use.

