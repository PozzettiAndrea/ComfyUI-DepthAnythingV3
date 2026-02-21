"""Basic inference nodes for DepthAnythingV3."""
import torch
import torch.nn.functional as F
import comfy.model_management as mm
from comfy.utils import ProgressBar
from comfy_api.latest import io

from .utils import (
    IMAGENET_MEAN, IMAGENET_STD, DEFAULT_PATCH_SIZE,
    format_camera_params, process_tensor_to_image, process_tensor_to_mask,
    resize_to_patch_multiple, logger, check_model_capabilities,
    imagenet_normalize, save_gaussians_to_ply,
)
from .normalization import (
    apply_edge_antialiasing,
    apply_standard_normalization,
    apply_v2_style_normalization,
    apply_raw_normalization,
)


class DepthAnything_V3(io.ComfyNode):
    """
    Unified Depth Anything V3 node with multiple normalization modes.

    This consolidates all depth processing approaches into a single node:
    - Standard: Original V3 min-max normalization
    - V2-Style: Disparity-based normalization with content-aware contrast (by Ltamann/TBG)
    - Raw: No normalization, outputs metric depth for 3D reconstruction

    Always outputs all available data - connect what you need, ignore the rest.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DepthAnything_V3",
            display_name="Depth Anything V3",
            category="image/depth",
            description="""Unified Depth Anything V3 node - all outputs, multiple normalization modes.

**Normalization Modes:**
- Standard: Original V3 min-max normalization (0-1 range, includes sky)
- V2-Style: Disparity-based with content-aware contrast (default, best for ControlNet)
- Raw: No normalization, outputs metric depth (for 3D reconstruction/point clouds)

Connect only the outputs you need - unused outputs are simply ignored.""",
            inputs=[
                io.Custom("DA3MODEL").Input("da3_model"),
                io.Image.Input("images"),
                io.Combo.Input("normalization_mode", options=["Standard", "V2-Style", "Raw"], default="V2-Style"),
                io.Custom("CAMERA_PARAMS").Input("camera_params", optional=True),
                io.Combo.Input("resize_method", options=["resize", "crop", "pad"], default="resize", optional=True,
                               tooltip="Model requires dimensions to be multiples of 14. resize: scale image (default), crop: center crop to multiple, pad: add black borders to multiple"),
                io.Boolean.Input("invert_depth", default=False, optional=True,
                                 tooltip="OFF (default): close=bright, far=dark. ON: far=bright, close=dark."),
                io.Boolean.Input("keep_model_size", default=False, optional=True,
                                 tooltip="Keep model's native patch-aligned output size instead of resizing back to original dimensions"),
            ],
            outputs=[
                io.Image.Output(display_name="depth"),
                io.Image.Output(display_name="confidence"),
                io.Image.Output(display_name="resized_rgb_image"),
                io.Image.Output(display_name="ray_origin"),
                io.Image.Output(display_name="ray_direction"),
                io.String.Output(display_name="extrinsics_json"),
                io.String.Output(display_name="intrinsics_json"),
                io.Mask.Output(display_name="sky_mask"),
                io.Custom("EXTRINSICS").Output(display_name="extrinsics"),
                io.Custom("INTRINSICS").Output(display_name="intrinsics"),
                io.String.Output(display_name="gaussian_ply_path"),
            ],
        )

    @classmethod
    def execute(cls, da3_model, images, normalization_mode="V2-Style", camera_params=None,
                resize_method="resize", invert_depth=False, keep_model_size=False):
        device = mm.get_torch_device()

        # da3_model is now a ModelPatcher — load to GPU via ComfyUI memory management
        mm.load_models_gpu([da3_model])
        model = da3_model.model

        # Get metadata stored by loader
        capabilities = da3_model.model_options.get("da3_capabilities", check_model_capabilities(model))
        dtype = da3_model.model_options.get("da3_dtype", torch.float16)

        if not capabilities["has_sky_segmentation"] and normalization_mode == "V2-Style":
            logger.warning(
                "WARNING: This model does not support sky segmentation. "
                "V2-Style normalization will work but without sky masking. "
                "Use Mono/Metric/Nested models for best V2-Style results."
            )

        B, H, W, C = images.shape
        logger.info(f"Input image size: {H}x{W}")

        # Memory estimation
        memory_required = mm.module_size(model)
        memory_required += images.nelement() * images.element_size() * 4
        mm.free_memory(memory_required, device)

        # Convert from ComfyUI format [B, H, W, C] to PyTorch [B, C, H, W]
        images_pt = images.permute(0, 3, 1, 2)

        # Resize to patch size multiple
        images_pt, orig_H, orig_W = resize_to_patch_multiple(images_pt, DEFAULT_PATCH_SIZE, resize_method)
        model_H, model_W = images_pt.shape[2], images_pt.shape[3]
        logger.info(f"Model input size (after resize): {model_H}x{model_W}")

        # Normalize with ImageNet stats (manual, no torchvision dependency)
        normalized_images = imagenet_normalize(images_pt)

        # Prepare for model: add view dimension [B, N, 3, H, W] where N=1
        normalized_images = normalized_images.unsqueeze(1)

        # Prepare camera parameters if provided
        extrinsics_input = None
        intrinsics_input = None
        if camera_params is not None:
            if capabilities["has_camera_conditioning"]:
                extrinsics_input = camera_params["extrinsics"].to(device).to(dtype)
                intrinsics_input = camera_params["intrinsics"].to(device).to(dtype)
                if extrinsics_input.shape[0] == 1 and B > 1:
                    extrinsics_input = extrinsics_input.expand(B, -1, -1, -1)
                    intrinsics_input = intrinsics_input.expand(B, -1, -1, -1)
                logger.info("Using camera-conditioned depth estimation")
            else:
                logger.warning("Model does not support camera conditioning. Camera params ignored.")

        pbar = ProgressBar(B)
        depth_out = []
        conf_out = []
        sky_out = []
        ray_origin_out = []
        ray_dir_out = []
        extrinsics_list = []
        intrinsics_list = []
        gaussians_list = []

        # Check if model supports 3D Gaussians
        infer_gs = capabilities["has_3d_gaussians"]
        if infer_gs:
            logger.info("Model supports 3D Gaussians - will output raw Gaussians")

        for i in range(B):
            img = normalized_images[i:i+1].to(device, dtype=dtype)

            # Get camera params for this batch item
            ext_i = extrinsics_input[i:i+1] if extrinsics_input is not None else None
            int_i = intrinsics_input[i:i+1] if intrinsics_input is not None else None

            # Run model forward with optional camera conditioning and Gaussians
            output = model(img, extrinsics=ext_i, intrinsics=int_i, infer_gs=infer_gs)

            # Extract depth
            depth = None
            if hasattr(output, 'depth'):
                depth = output.depth
            elif isinstance(output, dict) and 'depth' in output:
                depth = output['depth']

            if depth is None or not torch.is_tensor(depth):
                raise ValueError("Model output does not contain valid depth tensor")

            # Extract confidence
            conf = None
            if hasattr(output, 'depth_conf'):
                conf = output.depth_conf
            elif isinstance(output, dict) and 'depth_conf' in output:
                conf = output['depth_conf']

            if conf is None or not torch.is_tensor(conf):
                conf = torch.ones_like(depth)

            # Extract sky mask
            sky = None
            if hasattr(output, 'sky'):
                sky = output.sky
            elif isinstance(output, dict) and 'sky' in output:
                sky = output['sky']

            if sky is None or not torch.is_tensor(sky):
                sky = torch.zeros_like(depth)
            else:
                # Normalize sky mask to 0-1 range
                sky_min, sky_max = sky.min(), sky.max()
                if sky_max > sky_min:
                    sky = (sky - sky_min) / (sky_max - sky_min)

            # ===== NORMALIZATION DISPATCH =====
            if normalization_mode == "Raw":
                depth_processed = apply_raw_normalization(depth, invert_depth)
            elif normalization_mode == "V2-Style":
                depth_processed = apply_v2_style_normalization(depth, sky, device, invert_depth)
            else:  # "Standard"
                depth_processed = apply_standard_normalization(depth, invert_depth)

            # Normalize confidence
            conf_range = conf.max() - conf.min()
            if conf_range > 1e-8:
                conf = (conf - conf.min()) / conf_range
            else:
                conf = torch.ones_like(conf)

            depth_out.append(depth_processed.cpu())
            conf_out.append(conf.cpu())
            sky_out.append(sky.cpu())

            # Extract ray maps (if available)
            ray = None
            if hasattr(output, 'ray'):
                ray = output.ray
            elif isinstance(output, dict) and 'ray' in output:
                ray = output['ray']

            if ray is not None and torch.is_tensor(ray):
                ray = ray.squeeze(0).squeeze(0)  # [6, H, W]
                ray_origin = ray[:3]
                ray_dir = ray[3:6]
                ray_origin_out.append(ray_origin.cpu())
                ray_dir_out.append(ray_dir.cpu())
            else:
                ray_origin_out.append(torch.zeros(3, depth.shape[-2], depth.shape[-1]))
                ray_dir_out.append(torch.zeros(3, depth.shape[-2], depth.shape[-1]))

            # Extract camera parameters (if available)
            extr = None
            if hasattr(output, 'extrinsics'):
                extr = output.extrinsics
            elif isinstance(output, dict) and 'extrinsics' in output:
                extr = output['extrinsics']

            if extr is not None and torch.is_tensor(extr):
                extrinsics_list.append(extr.cpu())
            else:
                extrinsics_list.append(None)

            intr = None
            if hasattr(output, 'intrinsics'):
                intr = output.intrinsics
            elif isinstance(output, dict) and 'intrinsics' in output:
                intr = output['intrinsics']

            if intr is not None and torch.is_tensor(intr):
                intrinsics_list.append(intr.cpu())
            else:
                intrinsics_list.append(None)

            # Extract 3D Gaussians (only if model supports them and we requested them)
            if infer_gs:
                gs = None
                if hasattr(output, 'gaussians'):
                    gs = output.gaussians
                elif isinstance(output, dict) and 'gaussians' in output:
                    gs = output['gaussians']

                if gs is not None and hasattr(gs, 'means') and torch.is_tensor(gs.means):
                    # Store raw depth alongside Gaussians for pruning
                    gaussians_list.append((gs, depth))

            pbar.update(1)

        # Process outputs based on normalization mode
        normalize_depth_output = (normalization_mode != "Raw")

        depth_final = process_tensor_to_image(depth_out, orig_H, orig_W,
                                               normalize_output=normalize_depth_output,
                                               skip_resize=keep_model_size)
        conf_final = process_tensor_to_image(conf_out, orig_H, orig_W,
                                              normalize_output=True,
                                              skip_resize=keep_model_size)
        sky_final = process_tensor_to_mask(sky_out, orig_H, orig_W, skip_resize=keep_model_size)
        ray_origin_final = cls._process_ray_to_image(ray_origin_out, orig_H, orig_W,
                                                       normalize=True, skip_resize=keep_model_size)
        ray_dir_final = cls._process_ray_to_image(ray_dir_out, orig_H, orig_W,
                                                    normalize=True, skip_resize=keep_model_size)

        # Process resized RGB image to match depth output dimensions
        rgb_resized = images_pt.permute(0, 2, 3, 1).float().cpu()
        if not keep_model_size:
            final_H = (orig_H // 2) * 2
            final_W = (orig_W // 2) * 2
            if rgb_resized.shape[1] != final_H or rgb_resized.shape[2] != final_W:
                rgb_resized = F.interpolate(
                    rgb_resized.permute(0, 3, 1, 2),
                    size=(final_H, final_W),
                    mode="bilinear"
                ).permute(0, 2, 3, 1)
        rgb_resized = torch.clamp(rgb_resized, 0, 1)

        # Scale intrinsics if we resized back to original dimensions
        if not keep_model_size:
            final_H = (orig_H // 2) * 2
            final_W = (orig_W // 2) * 2
            model_H, model_W = images_pt.shape[2], images_pt.shape[3]

            if final_H != model_H or final_W != model_W:
                scale_h = final_H / model_H
                scale_w = final_W / model_W

                for i, intr in enumerate(intrinsics_list):
                    if intr is not None and torch.is_tensor(intr):
                        intr_scaled = intr.squeeze().clone()
                        intr_scaled[0, 0] *= scale_w  # fx
                        intr_scaled[1, 1] *= scale_h  # fy
                        intr_scaled[0, 2] *= scale_w  # cx
                        intr_scaled[1, 2] *= scale_h  # cy
                        intrinsics_list[i] = intr_scaled

        # Format camera parameters as strings (for backward compatibility)
        extrinsics_str = format_camera_params(extrinsics_list, "extrinsics")
        intrinsics_str = format_camera_params(intrinsics_list, "intrinsics")

        # Prepare tensor outputs for direct connection to other nodes
        if extrinsics_list and extrinsics_list[0] is not None:
            extrinsics_tensor = torch.stack([e.squeeze() for e in extrinsics_list if e is not None], dim=0)
        else:
            extrinsics_tensor = torch.eye(4).unsqueeze(0).expand(len(depth_out), -1, -1)

        if intrinsics_list and intrinsics_list[0] is not None:
            # Convert 3x3 intrinsics to 4x4 homogeneous (compatible with Sharp)
            intr_tensors = []
            for i_mat in intrinsics_list:
                if i_mat is not None:
                    k = i_mat.squeeze()
                    if k.shape == (3, 3):
                        k4 = torch.eye(4, dtype=k.dtype)
                        k4[:3, :3] = k
                        intr_tensors.append(k4)
                    else:
                        intr_tensors.append(k)
            intrinsics_tensor = torch.stack(intr_tensors, dim=0)
        else:
            intrinsics_tensor = torch.eye(4).unsqueeze(0).expand(len(depth_out), -1, -1)

        # Save Gaussians to PLY file if available (Giant model only)
        gaussian_ply_path = ""
        if gaussians_list:
            import folder_paths
            from pathlib import Path
            output_dir = Path(folder_paths.get_output_directory())
            # Use the first batch item's Gaussians and depth for pruning
            gs, raw_depth = gaussians_list[0]
            # Raw depth shape: (1, 1, H, W) -> squeeze to (1, H, W) for pruning
            depth_for_pruning = raw_depth.squeeze(0) if raw_depth.dim() == 4 else raw_depth
            # Get extrinsics for world-to-camera transform (preserves scale/position relationship)
            gs_extrinsics = extrinsics_list[0] if extrinsics_list and extrinsics_list[0] is not None else None
            filepath = output_dir / "gaussians_worldspace_0000.ply"
            gaussian_ply_path = save_gaussians_to_ply(
                gs, filepath, depth=depth_for_pruning,
                extrinsics=gs_extrinsics,
                shift_and_scale=False, save_sh_dc_only=False,
                prune_border=True, prune_depth_percent=0.9,
            )

        return io.NodeOutput(depth_final, conf_final, rgb_resized, ray_origin_final, ray_dir_final,
                extrinsics_str, intrinsics_str, sky_final, extrinsics_tensor, intrinsics_tensor, gaussian_ply_path)

    @staticmethod
    def _process_ray_to_image(ray_list, orig_H, orig_W, normalize=True, skip_resize=False):
        """Convert list of ray tensors to ComfyUI IMAGE format."""
        out = torch.cat([r.unsqueeze(0) for r in ray_list], dim=0)

        if normalize:
            for i in range(out.shape[0]):
                ray_batch = out[i]
                ray_min = ray_batch.min()
                ray_max = ray_batch.max()
                if ray_max > ray_min:
                    out[i] = (ray_batch - ray_min) / (ray_max - ray_min)
                else:
                    out[i] = torch.zeros_like(ray_batch)

        out = out.permute(0, 2, 3, 1).float()

        if not skip_resize:
            final_H = (orig_H // 2) * 2
            final_W = (orig_W // 2) * 2

            if out.shape[1] != final_H or out.shape[2] != final_W:
                out = F.interpolate(
                    out.permute(0, 3, 1, 2),
                    size=(final_H, final_W),
                    mode="bilinear"
                ).permute(0, 2, 3, 1)

        if normalize:
            return torch.clamp(out, 0, 1)
        else:
            return out



