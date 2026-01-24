# Quick Start: Modal Deployment

## Why `modal serve modal/app.py` Might Not Work

Common issues and solutions:

### 1. **Modal Not Installed**
```bash
pip install modal
```

### 2. **Not Authenticated**
```bash
modal token new
```
This opens a browser to authenticate your account.

### 3. **Typo in Code** 
Check line 78 in `modal/app.py` - it should be:
```python
img_array = np.frombuffer(img_bytes, dtype=np.uint8)  # ✅ Correct
```
NOT:
```python
img_array = np.frombuffer(img_bytes, dtype=np.uint8)  # ❌ Wrong
```

### 4. **Missing Model Weights**
You need to upload your weights file to a Modal volume first.

### 5. **Missing Config File**
Ensure `assets/configs/court_keypoints.yaml` exists.

## Quick Setup (5 Steps)

### Step 1: Install & Authenticate
```bash
pip install modal
modal token new
```

### Step 2: Upload Weights
```bash
# Make sure model_tennis_court_det.pt is in the repo root
modal run scripts/upload_weights_to_modal.py --weights model_tennis_court_det.pt
```

Or manually:
```bash
modal volume create tennis-weights
# Then use the upload script or Modal dashboard
```

### Step 3: Verify Config Exists
```bash
ls assets/configs/court_keypoints.yaml
```

### Step 4: Fix Any Code Issues
Check `modal/app.py` line 78 for the `np.frombuffer` typo.

### Step 5: Run
```bash
modal serve modal/app.py
```

## Expected Output

When `modal serve` works, you'll see:
```
✓ Created objects.
├── 🔨 Created image.
├── 💾 Created volume tennis-weights.
└── 🔌 Created endpoint.
   └── https://<username>--courtcheck-inference-predict.modal.run
```

## Deploy to Production

Once `modal serve` works:
```bash
modal deploy modal/app.py
```

## Test the Endpoint

```python
import requests
import base64

with open("test_image.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "https://<username>--courtcheck-inference-predict.modal.run",
    json={"image_base64": img_b64}
)
print(response.json())
```

## Common Error Messages

| Error | Solution |
|-------|----------|
| `modal: command not found` | `pip install modal` |
| `Authentication required` | `modal token new` |
| `Volume not found` | Create volume first (see Step 2) |
| `NameError: name 'np' is not defined` | Check imports in function |
| `FileNotFoundError: model weights` | Upload weights to volume |
| `ModuleNotFoundError: courtcheck` | Check `src/` directory structure |

For detailed setup, see `MODAL_SETUP.md`.
