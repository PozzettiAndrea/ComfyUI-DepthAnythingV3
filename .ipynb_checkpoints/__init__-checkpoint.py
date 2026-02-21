"""
ComfyUI-DepthAnythingV3: Depth Anything V3 nodes for ComfyUI
"""

import sys
import os
import traceback

# Track initialization status
INIT_SUCCESS = False
INIT_ERRORS = []

# Web directory for JavaScript extensions
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
