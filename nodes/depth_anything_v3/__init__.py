# Depth Anything V3 for ComfyUI
#
# Backward-compatible imports for code that imported from the old model/ subpackage.

from .model import (
    DepthAnything3Net,
    NestedDepthAnything3Net,
    DinoV2,
    DinoVisionTransformer,
    DPT,
    DualDPT,
    vit_small,
    vit_base,
    vit_large,
    vit_giant2,
)
from .camera import CameraEnc, CameraDec
from .gs import GSDPT, GaussianAdapter, Gaussians
