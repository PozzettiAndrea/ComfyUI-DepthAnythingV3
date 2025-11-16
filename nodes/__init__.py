"""
ComfyUI-DepthAnythingV3: Depth Anything V3 nodes for ComfyUI
"""

from .nodes_impl import (
    DownloadAndLoadDepthAnythingV3Model,
    DepthAnything_V3,
    NODE_CLASS_MAPPINGS as IMPL_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as IMPL_NODE_DISPLAY_NAME_MAPPINGS,
)

from .preview_nodes import (
    NODE_CLASS_MAPPINGS as PREVIEW_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as PREVIEW_NODE_DISPLAY_NAME_MAPPINGS,
)

# Merge all node mappings
NODE_CLASS_MAPPINGS = {
    **IMPL_NODE_CLASS_MAPPINGS,
    **PREVIEW_NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **IMPL_NODE_DISPLAY_NAME_MAPPINGS,
    **PREVIEW_NODE_DISPLAY_NAME_MAPPINGS,
}

__all__ = [
    'DownloadAndLoadDepthAnythingV3Model',
    'DepthAnything_V3',
    'NODE_CLASS_MAPPINGS',
    'NODE_DISPLAY_NAME_MAPPINGS',
]
