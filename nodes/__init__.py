"""
ComfyUI-DepthAnythingV3: Depth Anything V3 nodes for ComfyUI
"""

import sys

# Only do imports when NOT running under pytest
# This prevents relative import errors during test collection
if 'pytest' not in sys.modules:
    # Import from split modules
    from .nodes_loader import (
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

    from .preview_nodes import (
        NODE_CLASS_MAPPINGS as PREVIEW_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as PREVIEW_NODE_DISPLAY_NAME_MAPPINGS,
    )

    # Merge all node mappings
    NODE_CLASS_MAPPINGS = {
        **LOADER_NODE_CLASS_MAPPINGS,
        **INFERENCE_NODE_CLASS_MAPPINGS,
        **THREED_NODE_CLASS_MAPPINGS,
        **CAMERA_NODE_CLASS_MAPPINGS,
        **MULTIVIEW_NODE_CLASS_MAPPINGS,
        **PREVIEW_NODE_CLASS_MAPPINGS,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        **LOADER_NODE_DISPLAY_NAME_MAPPINGS,
        **INFERENCE_NODE_DISPLAY_NAME_MAPPINGS,
        **THREED_NODE_DISPLAY_NAME_MAPPINGS,
        **CAMERA_NODE_DISPLAY_NAME_MAPPINGS,
        **MULTIVIEW_NODE_DISPLAY_NAME_MAPPINGS,
        **PREVIEW_NODE_DISPLAY_NAME_MAPPINGS,
    }
else:
    # Dummy values during pytest to prevent import errors
    DownloadAndLoadDepthAnythingV3Model = None
    DepthAnything_V3 = None
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = [
    'DownloadAndLoadDepthAnythingV3Model',
    'DepthAnything_V3',
    'NODE_CLASS_MAPPINGS',
    'NODE_DISPLAY_NAME_MAPPINGS',
]
