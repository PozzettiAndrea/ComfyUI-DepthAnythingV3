# ComfyUI Depth Anything V3

Custom nodes for [Depth Anything V3](https://github.com/ByteDance-Seed/Depth-Anything-3) integration with ComfyUI.

![Simple Workflow](docs/simple_workflow.png)

![Advanced Workflow](docs/advanced_workflow.png)

![Advanced 3D Workflow](docs/advanced_3d_workflow.png)

## Description

Depth Anything V3 is the latest depth estimation model that predicts spatially consistent geometry from visual inputs. This package provides ComfyUI nodes for easy integration.

**Published**: November 14, 2025
**Paper**: [Depth Anything 3: Recovering the Visual Space from Any Views](https://arxiv.org/abs/2511.10647)

## Installation

1. Clone or copy this directory to your ComfyUI custom_nodes folder:
```bash
cd ComfyUI/custom_nodes/
git clone <your-repo-url> ComfyUI-DepthAnythingV3
```

2. Install dependencies:
```bash
cd ComfyUI-DepthAnythingV3
pip install -r requirements.txt
```

3. Restart ComfyUI

## Nodes

### 1. Download And Load Depth Anything V3 Model

Downloads and loads a Depth Anything V3 model from HuggingFace.

**Inputs**:
- `model`: Model variant to load
  - `da3_small.safetensors` - Small model (80M params)
  - `da3_large.safetensors` - Large model (350M params)
  - `da3_giant.safetensors` - Giant model (1.15B params)
  - `da3mono_large.safetensors` - Monocular depth specialized (350M params)
  - `da3metric_large.safetensors` - Metric depth estimation (350M params)
  - `da3nested_giant_large.safetensors` - Nested model (1.4B params)
- `precision`: Inference precision (auto/fp16/fp32/bf16)

**Outputs**:
- `da3_model`: Loaded model object

**Note**: Models are auto-downloaded to `ComfyUI/models/depthanything3/` on first use.

### 2. Depth Anything V3

Runs depth estimation on input images.

**Inputs**:
- `da3_model`: Model from the loader node
- `images`: Input image batch

**Outputs**:
- `depth`: Normalized depth maps (0-1 range, visualized as grayscale)

## Model Variants

| Model | Size | Features |
|-------|------|----------|
| DA3-Small | 80M | Fast, good quality |
| DA3-Large | 350M | High quality, balanced |
| DA3-Giant | 1.15B | Best quality, slower |
| DA3Mono-Large | 350M | Optimized for monocular depth |
| DA3Metric-Large | 350M | Metric depth estimation |
| DA3Nested-Giant-Large | 1.4B | Combined model with metric scaling |

## Usage Example

1. Add "Download And Load Depth Anything V3 Model" node
2. Select your desired model variant
3. Add "Depth Anything V3" node
4. Connect the model output to the inference node
5. Connect your images to the inference node
6. The depth output can be viewed directly or used in other nodes

## Differences from V2

Depth Anything V3 uses:
- Simpler architecture (vanilla DINO encoder)
- Unified depth-ray representation
- Better performance on monocular depth estimation
- Support for multi-view depth and camera pose estimation (future update)

## Credits

- **Original Paper**: Haotong Lin, Sili Chen, Jun Hao Liew, et al. (ByteDance Seed Team)
- **Implementation**: Based on the official [Depth Anything 3 repository](https://github.com/ByteDance-Seed/Depth-Anything-3)
- **ComfyUI Integration**: Custom nodes for ComfyUI compatibility

## License

The model architecture files are based on Depth Anything 3 (Apache 2.0 / CC BY-NC 4.0 depending on model).

Note: Some models (Giant, Nested) use CC BY-NC 4.0 license (non-commercial use only).
