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

# Only run initialization and imports when loaded by ComfyUI, not during pytest
if 'pytest' not in sys.modules:
    print("[ComfyUI-DepthAnythingV3] Initializing custom node...")

    # Import node classes
    try:
        from .nodes import (
            DownloadAndLoadDepthAnythingV3Model,
            DepthAnything_V3,
            NODE_CLASS_MAPPINGS,
            NODE_DISPLAY_NAME_MAPPINGS,
        )
        print("[ComfyUI-DepthAnythingV3] [OK] Node classes imported successfully")
        INIT_SUCCESS = True
    except Exception as e:
        error_msg = f"Failed to import node classes: {str(e)}"
        INIT_ERRORS.append(error_msg)
        print(f"[ComfyUI-DepthAnythingV3] [WARNING] {error_msg}")
        print(f"[ComfyUI-DepthAnythingV3] Traceback:\n{traceback.format_exc()}")

        # Set all to None if import failed
        DownloadAndLoadDepthAnythingV3Model = None
        DepthAnything_V3 = None
        NODE_CLASS_MAPPINGS = {}
        NODE_DISPLAY_NAME_MAPPINGS = {}

    # Report final status
    if INIT_SUCCESS:
        print("[ComfyUI-DepthAnythingV3] [OK] Loaded successfully!")
    else:
        print(f"[ComfyUI-DepthAnythingV3] [ERROR] Failed to load ({len(INIT_ERRORS)} error(s)):")
        for error in INIT_ERRORS:
            print(f"  - {error}")
        print("[ComfyUI-DepthAnythingV3] Please check the errors above and your installation.")

else:
    # During testing, set dummy values
    print("[ComfyUI-DepthAnythingV3] Running in pytest mode - skipping initialization")
    DownloadAndLoadDepthAnythingV3Model = None
    DepthAnything_V3 = None
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
