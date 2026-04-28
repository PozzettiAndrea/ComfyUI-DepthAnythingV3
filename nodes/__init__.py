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

NODE_CLASS_MAPPINGS = {
    cls.__name__: cls for cls in [
        DownloadAndLoadDepthAnythingV3Model,
        DA3_EnableTiledProcessing,
        DA3_DownloadModel,
        LoadSALADModel,
        DepthAnything_V3,
        DepthAnythingV3_MultiView,
        DA3_MultiViewPointCloud,
        DepthAnythingV3_Streaming,
        DA3_ToPointCloud,
        DA3_SavePointCloud,
        DA3_FilterGaussians,
        DA3_ToMesh,
        DA3_CreateCameraParams,
        DA3_ParseCameraPose,
        DA3_PreviewPointCloud,
    ]
}
NODE_DISPLAY_NAME_MAPPINGS = {k: k for k in NODE_CLASS_MAPPINGS}
