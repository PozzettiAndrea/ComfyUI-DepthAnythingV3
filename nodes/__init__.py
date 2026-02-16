"""
ComfyUI-DepthAnythingV3: Depth Anything V3 nodes for ComfyUI
"""
import logging

log = logging.getLogger("depthanythingv3")

try:
    from .load_model import (
        DownloadAndLoadDepthAnythingV3Model,
        NODE_CLASS_MAPPINGS as LOADER_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as LOADER_NODE_DISPLAY_NAME_MAPPINGS,
    )

    from .nodes_inference import (
        DepthAnything_V3,
        NODE_CLASS_MAPPINGS as INFERENCE_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as INFERENCE_NODE_DISPLAY_NAME_MAPPINGS,
    )

    from .nodes_3d import (
        NODE_CLASS_MAPPINGS as THREED_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as THREED_NODE_DISPLAY_NAME_MAPPINGS,
    )

    from .nodes_camera import (
        NODE_CLASS_MAPPINGS as CAMERA_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as CAMERA_NODE_DISPLAY_NAME_MAPPINGS,
    )

    from .nodes_multiview import (
        NODE_CLASS_MAPPINGS as MULTIVIEW_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as MULTIVIEW_NODE_DISPLAY_NAME_MAPPINGS,
    )

    from .streaming import DepthAnythingV3_Streaming
    from .load_salad import LoadSALADModel

    from .preview_nodes import (
        NODE_CLASS_MAPPINGS as PREVIEW_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as PREVIEW_NODE_DISPLAY_NAME_MAPPINGS,
    )

    STREAMING_NODE_CLASS_MAPPINGS = {
        "DepthAnythingV3_Streaming": DepthAnythingV3_Streaming,
        "LoadSALADModel": LoadSALADModel,
    }
    STREAMING_NODE_DISPLAY_NAME_MAPPINGS = {
        "DepthAnythingV3_Streaming": "Depth Anything V3 (Streaming)",
        "LoadSALADModel": "Load SALAD Model",
    }

    # Merge all node mappings
    NODE_CLASS_MAPPINGS = {
        **LOADER_NODE_CLASS_MAPPINGS,
        **INFERENCE_NODE_CLASS_MAPPINGS,
        **THREED_NODE_CLASS_MAPPINGS,
        **CAMERA_NODE_CLASS_MAPPINGS,
        **MULTIVIEW_NODE_CLASS_MAPPINGS,
        **STREAMING_NODE_CLASS_MAPPINGS,
        **PREVIEW_NODE_CLASS_MAPPINGS,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        **LOADER_NODE_DISPLAY_NAME_MAPPINGS,
        **INFERENCE_NODE_DISPLAY_NAME_MAPPINGS,
        **THREED_NODE_DISPLAY_NAME_MAPPINGS,
        **CAMERA_NODE_DISPLAY_NAME_MAPPINGS,
        **MULTIVIEW_NODE_DISPLAY_NAME_MAPPINGS,
        **STREAMING_NODE_DISPLAY_NAME_MAPPINGS,
        **PREVIEW_NODE_DISPLAY_NAME_MAPPINGS,
    }
except Exception as e:
    log.error(f"Could not load nodes: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = [
    'NODE_CLASS_MAPPINGS',
    'NODE_DISPLAY_NAME_MAPPINGS',
]
