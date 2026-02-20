"""
ComfyUI-DepthAnythingV3: Depth Anything V3 nodes for ComfyUI
"""
from .load_model import (
    DownloadAndLoadDepthAnythingV3Model,
    DA3_EnableTiledProcessing,
    DA3_DownloadModel,
    LoadSALADModel,
)

from .nodes_inference import DepthAnything_V3

from .nodes_3d import (
    DA3_ToPointCloud,
    DA3_SavePointCloud,
    DA3_FilterGaussians,
    DA3_ToMesh,
)

from .nodes_camera import (
    DA3_CreateCameraParams,
    DA3_ParseCameraPose,
)

from .nodes_multiview import (
    DepthAnythingV3_MultiView,
    DA3_MultiViewPointCloud,
)

from .streaming import DepthAnythingV3_Streaming

from .preview_nodes import DA3_PreviewPointCloud

NODE_CLASSES = [
    # Loaders
    DownloadAndLoadDepthAnythingV3Model,
    DA3_EnableTiledProcessing,
    DA3_DownloadModel,
    LoadSALADModel,
    # Inference
    DepthAnything_V3,
    # Multi-view
    DepthAnythingV3_MultiView,
    DA3_MultiViewPointCloud,
    # Streaming
    DepthAnythingV3_Streaming,
    # 3D
    DA3_ToPointCloud,
    DA3_SavePointCloud,
    DA3_FilterGaussians,
    DA3_ToMesh,
    # Camera
    DA3_CreateCameraParams,
    DA3_ParseCameraPose,
    # Preview
    DA3_PreviewPointCloud,
]
