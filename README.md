# ComfyUI Depth Anything V3

Custom nodes for [Depth Anything V3](https://github.com/ByteDance-Seed/Depth-Anything-3) integration with ComfyUI.

![Simple Workflow](docs/simple_workflow.png)

![Advanced Workflow](docs/advanced_workflow.png)

![Advanced 3D Workflow](docs/advanced_3d_workflow.png)

## Description

Depth Anything V3 is the latest depth estimation model that predicts spatially consistent geometry from visual inputs.

**Published**: November 14, 2025
**Paper**: [Depth Anything 3: Recovering the Visual Space from Any Views](https://arxiv.org/abs/2511.10647)

## Nodes

**Download And Load Depth Anything V3 Model** - Downloads and loads model from HuggingFace
- Inputs: `model` (variant), `precision` (auto/fp16/fp32/bf16)
- Output: `da3_model`
- Models auto-download to `ComfyUI/models/depthanything3/`

**Depth Anything V3** - Runs depth estimation
- Inputs: `da3_model`, `images`
- Output: `depth` (normalized 0-1, grayscale)

## Model Variants

| Model | Size | Features |
|-------|------|----------|
| DA3-Small | 80M | Fast, good quality |
| DA3-Large | 350M | High quality, balanced |
| DA3-Giant | 1.15B | Best quality, slower |
| DA3Mono-Large | 350M | Optimized for monocular depth |
| DA3Metric-Large | 350M | Metric depth estimation |
| DA3Nested-Giant-Large | 1.4B | Combined model with metric scaling |

## Credits

- **Original Paper**: Haotong Lin, Sili Chen, Jun Hao Liew, et al. (ByteDance Seed Team)
- **Implementation**: Based on the official [Depth Anything 3 repository](https://github.com/ByteDance-Seed/Depth-Anything-3)
- **Inspiration**: [kijai's ComfyUI-Depth-Anything-V2](https://github.com/kijai/ComfyUI-DepthAnythingV2)

## License

Model architecture files based on Depth Anything 3 (Apache 2.0 / CC BY-NC 4.0 depending on model).

Note: Some models (Giant, Nested) use CC BY-NC 4.0 license (non-commercial use only).
