"""
Preview nodes for Point Clouds and Gaussian Splats
"""

class DA3_PreviewPointCloud:
    """
    Preview point cloud or Gaussian splat PLY files in the browser using Three.js
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
Preview point clouds and Gaussian splats in 3D using Three.js.

Features:
- Interactive 3D viewing with mouse controls
- Automatic point cloud centering and scaling
- RGB color support
- Works with both point clouds and Gaussian splats

Controls:
- Left Mouse: Rotate view
- Right Mouse: Pan camera
- Mouse Wheel: Zoom in/out
"""

    def preview(self, file_path):
        """
        Preview the point cloud or Gaussian splat file.

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
