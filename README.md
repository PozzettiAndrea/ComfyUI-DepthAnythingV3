# ComfyUI Depth Anything V3

Custom nodes for [Depth Anything V3](https://github.com/ByteDance-Seed/Depth-Anything-3) integration with ComfyUI.

![Simple Workflow](docs/simple_workflow.png)

![Advanced Workflow](docs/advanced_workflow.png)

![Advanced 3D Workflow](docs/advanced_3d_workflow.png)

## Demo Videos


https://github.com/user-attachments/assets/24a1c640-4dee-48fe-a35b-34bec6bc6347


https://github.com/user-attachments/assets/0ca86877-836a-41c8-8851-5a92b7f2a028


https://github.com/user-attachments/assets/9e95e18a-21ba-4206-bce1-dbee46f3716f



## Description

Depth Anything V3 is the latest depth estimation model that predicts spatially consistent geometry from visual inputs.

**Published**: November 14, 2025
**Paper**: [Depth Anything 3: Recovering the Visual Space from Any Views](https://arxiv.org/abs/2511.10647)

## Nodes

### Model Loading
**Download And Load Depth Anything V3 Model** - Downloads and loads model from HuggingFace
- Inputs: `model` (variant), `precision` (auto/fp16/fp32/bf16)
- Output: `da3_model`
- Models auto-download to `ComfyUI/models/depthanything3/`

**DA3 Enable Tiled Processing** - Configure model for high-resolution images
- Inputs: `da3_model`, `tile_size`, `overlap`
- Output: `da3_model` (with tiled config)
- Use for 4K+ images to avoid OOM errors

### Inference Nodes
**Depth Anything V3** - Basic depth estimation
- Inputs: `da3_model`, `images`, `camera_params` (optional), `resize_method`, `invert_depth`
- Output: `depth` (normalized 0-1, grayscale)

**Depth Anything V3 (3D/Raw)** - Optimized for 3D reconstruction
- Inputs: `da3_model`, `images`, `camera_params` (optional), `resize_method`, `invert_depth`
- Outputs: `depth_raw` (metric), `confidence`, `intrinsics`, `sky_mask` (MASK type)

**Depth Anything V3 (Advanced)** - All available outputs
- Inputs: `da3_model`, `images`, `camera_params` (optional), `resize_method`, `invert_depth`
- Outputs: `depth`, `confidence`, `ray_origin`, `ray_direction`, `extrinsics`, `intrinsics`, `sky_mask`

**Depth Anything V3 (Multi-View)** - Process multiple images with cross-view attention
- Inputs: `da3_model`, `images` (batch), `resize_method`, `invert_depth`
- Outputs: `depth`, `confidence`
- Use for video frames or multiple angles of same scene

### 3D Processing
**DA3 to Point Cloud** - Convert depth to point cloud
**DA3 Save Point Cloud** - Export to PLY format
**DA3 to 3D Gaussians** - Extract 3D Gaussian splats (placeholder)
**DA3 Save 3D Gaussians** - Export Gaussians to PLY

### Camera Utilities
**DA3 Create Camera Parameters** - Create camera conditioning input
**DA3 Parse Camera Pose** - Parse camera parameters from JSON

## Model Variants

| Model | Size | Features |
|-------|------|----------|
| DA3-Small | 80M | Fast, good quality |
| DA3-Base | 220M | Balanced quality and speed |
| DA3-Large | 350M | High quality, balanced |
| DA3-Giant | 1.15B | Best quality, slower |
| DA3Mono-Large | 350M | Optimized for monocular depth |
| DA3Metric-Large | 350M | Metric depth estimation |
| DA3Nested-Giant-Large | 1.4B | Combined model with metric scaling |

## Model Capabilities

Different models support different features:

| Feature | Small/Base/Large/Giant | Mono-Large | Metric-Large | Nested |
|---------|------------------------|------------|--------------|--------|
| **Sky Segmentation** | ❌ | ✅ | ✅ | ✅ |
| **Camera Conditioning** | ✅ | ❌ | ❌ | ✅ |
| **Multi-View Attention** | ✅ | ⚠️ | ⚠️ | ✅ |
| **3D Gaussians** | ✅* | ❌ | ❌ | ✅* |
| **Ray Maps** | ✅ | ❌ | ❌ | ✅ |

- ✅ = Fully supported
- ❌ = Not available (returns zeros/ignored)
- ⚠️ = Works but no cross-view attention benefit (images processed independently)
- ✅* = Requires fine-tuned model weights (placeholder in current release)

**Choose your model based on needs:**
- Need sky masks? → Use Mono/Metric/Nested
- Need camera conditioning? → Use Main series or Nested
- Processing video/multi-view? → Use Main series or Nested for consistency
- Single images only? → Any model works

## Credits

- **Original Paper**: Haotong Lin, Sili Chen, Jun Hao Liew, et al. (ByteDance Seed Team)
- **Implementation**: Based on the official [Depth Anything 3 repository](https://github.com/ByteDance-Seed/Depth-Anything-3)
- **Inspiration**: [kijai's ComfyUI-Depth-Anything-V2](https://github.com/kijai/ComfyUI-DepthAnythingV2)

## License

Model architecture files based on Depth Anything 3 (Apache 2.0 / CC BY-NC 4.0 depending on model).

Note: Some models (Giant, Nested) use CC BY-NC 4.0 license (non-commercial use only).
