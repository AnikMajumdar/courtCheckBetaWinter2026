# Modal Deployment Setup Guide

This guide will help you set up and deploy the CourtCheck inference service on Modal.

## Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com) if you haven't already
2. **Modal CLI**: Install the Modal Python package
3. **Model Weights**: You need your trained model weights file (`model_tennis_court_det.pt`)

## Step 1: Install Modal

```bash
pip install modal
```

Or add it to your requirements:
```bash
pip install -r requirements.txt
```

## Step 2: Authenticate with Modal

```bash
modal token new
```

This will open a browser window for you to authenticate. After authentication, you'll be ready to use Modal.

## Step 3: Create Modal Volume and Upload Weights

Modal uses volumes for persistent storage. You need to create a volume and upload your model weights:

```bash
# Create the volume (if it doesn't exist)
modal volume create tennis-weights

# Upload your model weights to the volume
# First, let's create a helper script to upload the weights
```

Create a temporary upload script `upload_weights.py`:

```python
import modal

app = modal.App("upload-weights")
vol = modal.Volume.from_name("tennis-weights", create_if_missing=True)

@app.function(volumes={"/weights": vol})
def upload():
    import shutil
    # Copy local weights to volume
    shutil.copy("model_tennis_court_det.pt", "/weights/model_tennis_court_det.pt")
    vol.commit()
    print("Weights uploaded successfully!")

if __name__ == "__main__":
    with app.run():
        upload()
```

Then run:
```bash
modal run upload_weights.py
```

**Alternative: Use Modal CLI to upload directly**

```bash
# Mount the volume and copy files
modal volume put tennis-weights model_tennis_court_det.pt /model_tennis_court_det.pt
```

## Step 4: Verify Config File Exists

Make sure your config file exists at `assets/configs/court_keypoints.yaml`. If you're using a different config, update the `CONFIG_PATH` in `modal/app.py`.

## Step 5: Test Locally (Optional)

Before deploying, you can test the app locally:

```bash
modal serve modal/app.py
```

This will:
- Build the container image
- Start a local development server
- Give you a URL to test the endpoint

**Note**: The first run will take several minutes as it builds the image with Detectron2.

## Step 6: Deploy to Production

Once everything works locally, deploy to production:

```bash
modal deploy modal/app.py
```

This creates a persistent deployment that will:
- Keep the endpoint running
- Auto-scale based on traffic
- Provide a stable URL

After deployment, Modal will give you an endpoint URL like:
```
https://<your-username>--courtcheck-inference-predict.modal.run
```

## Step 7: Test the Deployed Endpoint

You can test the endpoint with a simple Python script:

```python
import requests
import base64

# Read and encode an image
with open("test_image.jpg", "rb") as f:
    image_bytes = f.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

# Send request
response = requests.post(
    "https://<your-username>--courtcheck-inference-predict.modal.run",
    json={"image_base64": image_b64}
)

print(response.json())
```

Or use curl:

```bash
# Encode image to base64
IMAGE_B64=$(base64 -i test_image.jpg)

# Send request
curl -X POST \
  https://<your-username>--courtcheck-inference-predict.modal.run \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"$IMAGE_B64\"}"
```

## Troubleshooting

### Issue: "modal serve" command not found

**Solution**: Make sure Modal is installed:
```bash
pip install modal
```

### Issue: Authentication errors

**Solution**: Re-authenticate:
```bash
modal token new
```

### Issue: Volume not found

**Solution**: Create the volume first:
```bash
modal volume create tennis-weights
```

Then upload your weights (see Step 3).

### Issue: Config file not found

**Solution**: 
1. Check that `assets/configs/court_keypoints.yaml` exists
2. Or update `CONFIG_PATH` in `modal/app.py` to point to your config file
3. Make sure the config file is in the `assets` directory (it gets copied to the container)

### Issue: Weights file not found

**Solution**: 
1. Make sure you've uploaded weights to the volume (Step 3)
2. Verify the volume name matches in `modal/app.py`: `modal.Volume.from_name("tennis-weights")`
3. Check the path: `/weights/model_tennis_court_det.pt`

### Issue: Import errors in container

**Solution**: 
- The code should be automatically copied via `.add_local_dir("src", remote_path="/app/src")`
- Make sure your local `src/` directory structure matches what's expected
- Check that all imports in `src/courtcheck/court/infer.py` are correct

### Issue: GPU not available / Out of memory

**Solution**: 
- Change GPU type in `modal/app.py`: `gpu="L4"` to `gpu="T4"` (cheaper) or `gpu="A10G"` (more powerful)
- Or remove GPU requirement for testing: remove `gpu="L4"` line (will use CPU, slower)

### Issue: Build takes too long

**Solution**: 
- First build always takes 10-15 minutes (building Detectron2 from source)
- Subsequent builds are cached and much faster
- Consider using a pre-built Detectron2 wheel if available

## Monitoring and Logs

View logs from your deployment:

```bash
modal app logs courtcheck-inference
```

Or in the Modal dashboard: https://modal.com/apps

## Cost Considerations

- **GPU (L4)**: ~$0.50-1.00/hour when running
- **Storage (Volume)**: ~$0.10/GB/month
- **Idle time**: No cost when not receiving requests (with proper configuration)

Modal charges only for actual compute time, so if your endpoint is idle, you don't pay.

## Next Steps

1. Set up a custom domain (optional)
2. Add authentication/API keys if needed
3. Set up monitoring and alerts
4. Optimize the container image size if needed

For more information, see the [Modal documentation](https://modal.com/docs).
