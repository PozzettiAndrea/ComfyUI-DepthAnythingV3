"""Camera utility nodes for DepthAnythingV3."""
import torch
from .utils import logger


class DA3_CreateCameraParams:
    """Create camera parameters (extrinsics and intrinsics) for conditioning depth estimation."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_width": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "image_height": ("INT", {"default": 512, "min": 1, "max": 8192}),
            },
            "optional": {
                # Camera position (translation)
                "cam_x": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "cam_y": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "cam_z": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                # Camera rotation (Euler angles in degrees)
                "rot_x": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1}),
                "rot_y": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1}),
                "rot_z": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1}),
                # Intrinsics
                "focal_length": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 1.0}),
                "fov_degrees": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 180.0, "step": 1.0}),
            }
        }

    RETURN_TYPES = ("CAMERA_PARAMS",)
    RETURN_NAMES = ("camera_params",)
    FUNCTION = "create_params"
    CATEGORY = "DepthAnythingV3"
    DESCRIPTION = """
Create camera parameters for conditioning DA3 depth estimation.

Provides known camera pose to improve depth estimation accuracy.

Parameters:
- cam_x/y/z: Camera position in world space
- rot_x/y/z: Camera rotation (Euler angles in degrees)
- focal_length: If > 0, uses this value. Otherwise uses fov_degrees.
- fov_degrees: Field of view in degrees (used if focal_length is 0)

Output:
- CAMERA_PARAMS: Dictionary with extrinsics (4x4) and intrinsics (3x3) matrices
"""

    def create_params(self, image_width, image_height, cam_x=0.0, cam_y=0.0, cam_z=0.0,
                     rot_x=0.0, rot_y=0.0, rot_z=0.0, focal_length=0.0, fov_degrees=60.0):
        import numpy as np

        # Create rotation matrix from Euler angles (XYZ order)
        rx = np.radians(rot_x)
        ry = np.radians(rot_y)
        rz = np.radians(rot_z)

        # Rotation matrices
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

        R = Rz @ Ry @ Rx  # Rotation matrix
        t = np.array([cam_x, cam_y, cam_z])  # Translation

        # Create extrinsics (world-to-camera, 4x4)
        extrinsics = np.eye(4, dtype=np.float32)
        extrinsics[:3, :3] = R.T  # Inverse rotation
        extrinsics[:3, 3] = -R.T @ t  # Inverse translation

        # Create intrinsics (3x3)
        if focal_length > 0:
            fx = fy = focal_length
        else:
            # Calculate from FOV
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

        return (camera_params,)


class DA3_ParseCameraPose:
    """Parse camera pose from DA3 output strings into usable format."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "extrinsics_json": ("STRING", {"multiline": True}),
                "intrinsics_json": ("STRING", {"multiline": True}),
            },
            "optional": {
                "batch_index": ("INT", {"default": 0, "min": 0, "max": 100}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("cam_x", "cam_y", "cam_z", "rot_x", "rot_y", "rot_z", "fx", "fy")
    FUNCTION = "parse"
    CATEGORY = "DepthAnythingV3"
    DESCRIPTION = """
Parse camera pose from DA3 JSON output.

Extracts camera position and rotation from extrinsics matrix,
and focal lengths from intrinsics matrix.

Inputs:
- extrinsics_json: JSON string from DA3 output
- intrinsics_json: JSON string from DA3 output
- batch_index: Which image's parameters to extract (default 0)

Outputs:
- cam_x/y/z: Camera position in world space
- rot_x/y/z: Camera rotation (Euler angles in degrees)
- fx/fy: Focal lengths
"""

    def parse(self, extrinsics_json, intrinsics_json, batch_index=0):
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
                        if ext.ndim == 3:  # [N, 4, 4] or [N, 3, 4]
                            ext = ext[0]  # Take first view

                        # Extract rotation and translation
                        R = ext[:3, :3]
                        t = ext[:3, 3]

                        # Convert world-to-camera to camera position in world
                        # cam_pos = -R^T @ t
                        cam_pos = -R.T @ t
                        cam_x, cam_y, cam_z = float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])

                        # Extract Euler angles from rotation matrix (XYZ order)
                        # R = Rz @ Ry @ Rx, so we extract in reverse
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
                        if intr.ndim == 3:  # [N, 3, 3]
                            intr = intr[0]  # Take first view

                        fx = float(intr[0, 0])
                        fy = float(intr[1, 1])

        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            logger.error(f"Error parsing camera params: {e}")

        return (cam_x, cam_y, cam_z, rot_x, rot_y, rot_z, fx, fy)


NODE_CLASS_MAPPINGS = {
    "DA3_CreateCameraParams": DA3_CreateCameraParams,
    "DA3_ParseCameraPose": DA3_ParseCameraPose,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DA3_CreateCameraParams": "DA3 Create Camera Parameters",
    "DA3_ParseCameraPose": "DA3 Parse Camera Pose",
}
