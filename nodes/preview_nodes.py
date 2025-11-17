"""
Preview nodes for Point Clouds and Gaussian Splats
"""

class DA3_PreviewPointCloud:
    """
    Preview point cloud PLY files in the browser using VTK.js
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "DepthAnythingV3"
    DESCRIPTION = """
Preview point clouds in 3D using VTK.js (scientific visualization).

Features:
- VTK.js rendering engine
- Trackball camera controls
- Axis orientation widget
- Adjustable point size
- Scientific color mapping support
- Max 2M points

Controls:
- Left Mouse: Rotate view
- Right Mouse: Pan camera
- Mouse Wheel: Zoom in/out
- Slider: Adjust point size
"""

    def preview(self, file_path):
        """
        Preview the point cloud file using VTK.js.

        Args:
            file_path: Path to the PLY file
        """
        # Return the file path to the UI
        return {
            "ui": {
                "file_path": [file_path]
            }
        }


NODE_CLASS_MAPPINGS = {
    "DA3_PreviewPointCloud": DA3_PreviewPointCloud,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DA3_PreviewPointCloud": "DA3 Preview Point Cloud / Gaussians",
}
