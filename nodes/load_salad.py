"""Load SALAD Model node — downloads and loads the SALAD VPR model for loop closure."""

import logging
import os

import torch
import folder_paths
import comfy.model_management as mm

logger = logging.getLogger("DepthAnythingV3")

SALAD_CKPT_URL = "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"
SALAD_CKPT_NAME = "dino_salad.ckpt"


class LoadSALADModel:
    """Download and load the SALAD model for visual place recognition / loop closure."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }

    RETURN_TYPES = ("SALAD_MODEL",)
    RETURN_NAMES = ("salad_model",)
    FUNCTION = "load"
    CATEGORY = "DepthAnythingV3"
    DESCRIPTION = """
Load the SALAD (Sinkhorn Algorithm for Locally-Aggregated Descriptors) model for loop closure detection.

Uses DINOv2 ViT-B14 backbone + SALAD aggregator (~340MB).
Auto-downloads from GitHub on first use.

Connect to the Streaming node's salad_model input to enable loop closure for long videos.
"""

    def load(self):
        from .streaming.loop_utils.salad_model import VPRModel

        device = mm.get_torch_device()

        # Download checkpoint if needed
        download_dir = os.path.join(folder_paths.models_dir, "salad")
        os.makedirs(download_dir, exist_ok=True)
        ckpt_path = os.path.join(download_dir, SALAD_CKPT_NAME)

        if not os.path.exists(ckpt_path):
            logger.info(f"Downloading SALAD model to: {ckpt_path}")
            torch.hub.download_url_to_file(SALAD_CKPT_URL, ckpt_path)

        # Load model
        logger.info(f"Loading SALAD model from: {ckpt_path}")
        model = VPRModel()
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model = model.eval().to(device)
        logger.info("SALAD model ready")

        return ({"model": model, "device": device},)
