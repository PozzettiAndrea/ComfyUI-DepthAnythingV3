"""
Preview nodes for Point Clouds and Gaussian Splats
"""
import logging
from comfy_api.latest import io

log = logging.getLogger("depthanythingv3")


class DA3_PreviewPointCloud(io.ComfyNode):
    """Preview point cloud PLY files in the browser using VTK.js"""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DA3_PreviewPointCloud",
            display_name="DA3 Preview Point Cloud",
            category="DepthAnythingV3",
            is_output_node=True,
            description="""Preview point cloud PLY files in 3D using VTK.js (scientific visualization).

Inputs:
- file_path: Path to PLY file (typically from DA3 Save Point Cloud node)
- color_mode:
  - RGB: Show original texture colors from PLY file
  - View ID: Color points by source view (requires view_id in PLY)

Features:
- VTK.js rendering engine
- Trackball camera controls
- Axis orientation widget
- Adjustable point size
- Toggle between RGB and view-based coloring
- Max 2M points

Controls:
- Left Mouse: Rotate view
- Right Mouse: Pan camera
- Mouse Wheel: Zoom in/out
- Slider: Adjust point size""",
            inputs=[
                io.String.Input("file_path", default="", optional=True),
                io.Combo.Input("color_mode", options=["RGB", "View ID"], default="RGB", optional=True),
            ],
            outputs=[],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        """Force re-execution when color_mode or file_path changes."""
        file_path = kwargs.get('file_path', '')
        color_mode = kwargs.get('color_mode', 'RGB')
        return f"{file_path}_{color_mode}"

    @classmethod
    def execute(cls, file_path="", color_mode="RGB"):
        """Preview the point cloud using VTK.js."""
        import os
        from pathlib import Path

        log.info(f"preview() called with color_mode='{color_mode}', file_path='{file_path}'")

        if not file_path or file_path.strip() == "":
            return io.NodeOutput(ui={"file_path": [""]})

        if color_mode == "RGB":
            log.info(f"Using RGB mode, returning original file: {file_path}")
            return io.NodeOutput(ui={"file_path": [file_path]})

        # For View ID mode, we need to read PLY, recolor, and write temp file
        log.info(f"Attempting View ID mode for: {file_path}")
        try:
            points, colors, confidence, view_id = _read_ply(file_path)

            if view_id is None:
                return io.NodeOutput(ui={"file_path": [file_path]})

            log.info(f"Recoloring {len(view_id)} points by view_id")
            colors = _color_by_view_id(view_id)

            import folder_paths
            output_dir = folder_paths.get_output_directory()
            temp_path = Path(output_dir) / "comfyui_preview_pointcloud.ply"
            _write_ply(temp_path, points, colors, confidence, view_id)
            log.info(f"View ID mode: wrote temp file to {temp_path}")

            return io.NodeOutput(ui={"file_path": [str(temp_path)]})
        except Exception as e:
            log.error(f"Error processing PLY file: {e}")
            return io.NodeOutput(ui={"file_path": [file_path]})


def _read_ply(filepath):
    """Read PLY file and extract points, colors, confidence, and view_id."""
    import numpy as np

    points = []
    colors = []
    confidence_vals = []
    view_ids = []

    has_color = False
    has_confidence = False
    has_view_id = False
    num_vertices = 0

    with open(filepath, 'r') as f:
        line = f.readline().strip()
        if line != "ply":
            raise ValueError("Not a valid PLY file")

        while True:
            line = f.readline().strip()
            if line.startswith("element vertex"):
                num_vertices = int(line.split()[-1])
            elif line.startswith("property") and "red" in line:
                has_color = True
            elif line.startswith("property") and "confidence" in line:
                has_confidence = True
            elif line.startswith("property") and "view_id" in line:
                has_view_id = True
            elif line == "end_header":
                break

        for _ in range(num_vertices):
            line = f.readline().strip()
            parts = line.split()

            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            points.append([x, y, z])

            idx = 3

            if has_color:
                r, g, b = int(parts[idx]), int(parts[idx+1]), int(parts[idx+2])
                colors.append([r / 255.0, g / 255.0, b / 255.0])
                idx += 3

            if has_confidence:
                conf = float(parts[idx])
                confidence_vals.append(conf)
                idx += 1

            if has_view_id:
                vid = int(parts[idx])
                view_ids.append(vid)
                idx += 1

    points = np.array(points, dtype=np.float32)
    colors = np.array(colors, dtype=np.float32) if has_color else None
    confidence = np.array(confidence_vals, dtype=np.float32) if has_confidence else None
    view_id = np.array(view_ids, dtype=np.int32) if has_view_id else None

    return points, colors, confidence, view_id


def _color_by_view_id(view_id):
    """Generate colors based on view ID using a color palette."""
    import numpy as np

    color_palette = np.array([
        [1.0, 0.0, 0.0],  # Red
        [0.0, 0.0, 1.0],  # Blue
        [0.0, 1.0, 0.0],  # Green
        [1.0, 1.0, 0.0],  # Yellow
        [1.0, 0.0, 1.0],  # Magenta
        [0.0, 1.0, 1.0],  # Cyan
        [1.0, 0.5, 0.0],  # Orange
        [0.5, 0.0, 1.0],  # Purple
    ])

    num_views = len(color_palette)
    colors = np.zeros((len(view_id), 3), dtype=np.float32)

    for i in range(len(view_id)):
        view_idx = int(view_id[i]) % num_views
        colors[i] = color_palette[view_idx]

    return colors


def _write_ply(filepath, points, colors=None, confidence=None, view_id=None):
    """Write point cloud to PLY file."""
    import numpy as np

    N = len(points)

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
    ]

    if colors is not None:
        header.extend([
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ])

    if confidence is not None:
        header.append("property float confidence")

    if view_id is not None:
        header.append("property int view_id")

    header.append("end_header")

    with open(filepath, 'w') as f:
        f.write('\n'.join(header) + '\n')

        for i in range(N):
            x, y, z = points[i]
            line = f"{x} {y} {z}"

            if colors is not None:
                r, g, b = (colors[i] * 255).astype(np.uint8)
                line += f" {r} {g} {b}"

            if confidence is not None:
                line += f" {confidence[i]}"

            if view_id is not None:
                line += f" {int(view_id[i])}"

            f.write(line + '\n')
