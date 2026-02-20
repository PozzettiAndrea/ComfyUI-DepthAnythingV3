"""Camera utility nodes for DepthAnythingV3."""
import torch
from comfy_api.latest import io
from .utils import logger


class DA3_CreateCameraParams(io.ComfyNode):
    """Create camera parameters (extrinsics and intrinsics) for conditioning depth estimation."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DA3_CreateCameraParams",
            display_name="DA3 Create Camera Parameters",
            category="DepthAnythingV3",
            description="""Create camera parameters for conditioning DA3 depth estimation.

Provides known camera pose to improve depth estimation accuracy.

Parameters:
- cam_x/y/z: Camera position in world space
- rot_x/y/z: Camera rotation (Euler angles in degrees)
- focal_length: If > 0, uses this value. Otherwise uses fov_degrees.
- fov_degrees: Field of view in degrees (used if focal_length is 0)

Output:
- CAMERA_PARAMS: Dictionary with extrinsics (4x4) and intrinsics (3x3) matrices""",
            inputs=[
                io.Int.Input("image_width", default=512, min=1, max=8192),
                io.Int.Input("image_height", default=512, min=1, max=8192),
                io.Float.Input("cam_x", default=0.0, min=-100.0, max=100.0, step=0.01, optional=True),
                io.Float.Input("cam_y", default=0.0, min=-100.0, max=100.0, step=0.01, optional=True),
                io.Float.Input("cam_z", default=0.0, min=-100.0, max=100.0, step=0.01, optional=True),
                io.Float.Input("rot_x", default=0.0, min=-180.0, max=180.0, step=0.1, optional=True),
                io.Float.Input("rot_y", default=0.0, min=-180.0, max=180.0, step=0.1, optional=True),
                io.Float.Input("rot_z", default=0.0, min=-180.0, max=180.0, step=0.1, optional=True),
                io.Float.Input("focal_length", default=0.0, min=0.0, max=10000.0, step=1.0, optional=True),
                io.Float.Input("fov_degrees", default=60.0, min=1.0, max=180.0, step=1.0, optional=True),
            ],
            outputs=[
                io.Custom("CAMERA_PARAMS").Output(display_name="camera_params"),
            ],
        )

    @classmethod
    def execute(cls, image_width, image_height, cam_x=0.0, cam_y=0.0, cam_z=0.0,
                rot_x=0.0, rot_y=0.0, rot_z=0.0, focal_length=0.0, fov_degrees=60.0):
        import numpy as np

        # Create rotation matrix from Euler angles (XYZ order)
        rx = np.radians(rot_x)
        ry = np.radians(rot_y)
        rz = np.radians(rot_z)

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])

        R = Rz @ Ry @ Rx
        t = np.array([cam_x, cam_y, cam_z])

        # Create extrinsics (world-to-camera, 4x4)
        extrinsics = np.eye(4, dtype=np.float32)
        extrinsics[:3, :3] = R.T
        extrinsics[:3, 3] = -R.T @ t

        # Create intrinsics (3x3)
        if focal_length > 0:
            fx = fy = focal_length
        else:
            fov_rad = np.radians(fov_degrees)
            fx = fy = (image_width / 2.0) / np.tan(fov_rad / 2.0)

        cx = image_width / 2.0
        cy = image_height / 2.0

        intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # Convert to tensors
        extrinsics_tensor = torch.from_numpy(extrinsics).unsqueeze(0).unsqueeze(0)  # [1, 1, 4, 4]
        intrinsics_tensor = torch.from_numpy(intrinsics).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]

        camera_params = {
            "extrinsics": extrinsics_tensor,
            "intrinsics": intrinsics_tensor,
            "image_size": (image_height, image_width),
        }

        return io.NodeOutput(camera_params)


class DA3_ParseCameraPose(io.ComfyNode):
    """Parse camera pose from DA3 output strings into usable format."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DA3_ParseCameraPose",
            display_name="DA3 Parse Camera Pose",
            category="DepthAnythingV3",
            description="""Parse camera pose from DA3 JSON output.

Extracts camera position and rotation from extrinsics matrix,
and focal lengths from intrinsics matrix.

Inputs:
- extrinsics_json: JSON string from DA3 output
- intrinsics_json: JSON string from DA3 output
- batch_index: Which image's parameters to extract (default 0)

Outputs:
- cam_x/y/z: Camera position in world space
- rot_x/y/z: Camera rotation (Euler angles in degrees)
- fx/fy: Focal lengths""",
            inputs=[
                io.String.Input("extrinsics_json", multiline=True),
                io.String.Input("intrinsics_json", multiline=True),
                io.Int.Input("batch_index", default=0, min=0, max=100, optional=True),
            ],
            outputs=[
                io.Float.Output(display_name="cam_x"),
                io.Float.Output(display_name="cam_y"),
                io.Float.Output(display_name="cam_z"),
                io.Float.Output(display_name="rot_x"),
                io.Float.Output(display_name="rot_y"),
                io.Float.Output(display_name="rot_z"),
                io.Float.Output(display_name="fx"),
                io.Float.Output(display_name="fy"),
            ],
        )

    @classmethod
    def execute(cls, extrinsics_json, intrinsics_json, batch_index=0):
        import json
        import numpy as np

        # Default values
        cam_x, cam_y, cam_z = 0.0, 0.0, 0.0
        rot_x, rot_y, rot_z = 0.0, 0.0, 0.0
        fx, fy = 512.0, 512.0

        try:
            # Parse extrinsics
            ext_data = json.loads(extrinsics_json)
            if "extrinsics" in ext_data and isinstance(ext_data["extrinsics"], list):
                if batch_index < len(ext_data["extrinsics"]):
                    img_key = f"image_{batch_index}"
                    ext_matrix = ext_data["extrinsics"][batch_index].get(img_key)

                    if ext_matrix is not None:
                        ext = np.array(ext_matrix)
                        if ext.ndim == 3:
                            ext = ext[0]

                        R = ext[:3, :3]
                        t = ext[:3, 3]

                        cam_pos = -R.T @ t
                        cam_x, cam_y, cam_z = float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])

                        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
                        singular = sy < 1e-6

                        if not singular:
                            rot_x = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
                            rot_y = np.degrees(np.arctan2(-R[2, 0], sy))
                            rot_z = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
                        else:
                            rot_x = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
                            rot_y = np.degrees(np.arctan2(-R[2, 0], sy))
                            rot_z = 0.0

            # Parse intrinsics
            int_data = json.loads(intrinsics_json)
            if "intrinsics" in int_data and isinstance(int_data["intrinsics"], list):
                if batch_index < len(int_data["intrinsics"]):
                    img_key = f"image_{batch_index}"
                    int_matrix = int_data["intrinsics"][batch_index].get(img_key)

                    if int_matrix is not None:
                        intr = np.array(int_matrix)
                        if intr.ndim == 3:
                            intr = intr[0]

                        fx = float(intr[0, 0])
                        fy = float(intr[1, 1])

        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            logger.error(f"Error parsing camera params: {e}")

        return io.NodeOutput(cam_x, cam_y, cam_z, rot_x, rot_y, rot_z, fx, fy)
