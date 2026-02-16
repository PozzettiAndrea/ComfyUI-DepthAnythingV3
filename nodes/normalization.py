"""Shared depth normalization methods for DepthAnythingV3 nodes."""
import torch
import torch.nn.functional as F


def apply_edge_antialiasing(mask):
    """Apply minimal anti-aliasing ONLY to border pixels (1-2px transition)."""
    # Ensure mask is in correct format [B, 1, H, W]
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0)

    # Very small 3x3 averaging kernel for minimal smoothing
    kernel = torch.ones((1, 1, 3, 3), device=mask.device, dtype=mask.dtype) / 9.0

    # Apply minimal blur
    mask_blurred = F.conv2d(mask, kernel, padding=1)

    # Detect edges: where original mask has transitions
    mask_dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    mask_eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=3, stride=1, padding=1)

    # Edge zone is where dilated and eroded differ
    edge_zone = (mask_dilated - mask_eroded).abs()
    edge_zone = (edge_zone > 0.01).float()

    # Apply anti-aliasing ONLY in edge zone
    mask_aa = mask * (1.0 - edge_zone) + mask_blurred * edge_zone

    return mask_aa


def apply_standard_normalization(depth, invert_depth):
    """
    Standard min-max normalization (original V3 approach).

    By default, inverts to match V2-Style convention (close=bright).
    Standard normalization naturally outputs far=bright, so we invert by default.
    """
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    # Invert by default to match V2-Style (close = bright)
    if not invert_depth:
        depth_norm = 1.0 - depth_norm

    return depth_norm


def apply_v2_style_normalization(depth, sky, device, invert_depth=False):
    """
    V2-Style disparity normalization (Ltamann/TBG approach).

    - Converts depth to disparity (1/depth) so sky becomes black
    - Uses content-only normalization (excludes sky)
    - Applies percentile-based contrast enhancement (1st-99th percentile)
    - Adds contrast boost via power transform (hardcoded to 2.0)
    - Applies edge anti-aliasing for natural transitions (hardcoded to True)
    - Optionally inverts the final output if invert_depth is True
    """
    epsilon = 1e-6
    contrast_boost = 2.0  # Hardcoded default
    edge_soften = True    # Hardcoded default

    # 1. Create HARD binary content mask
    if sky.max() > 0.1:
        # Threshold of 0.3 for aggressive sky detection
        content_mask_binary = (sky < 0.3).float()

        # Apply edge anti-aliasing (affects 1-2px border only)
        if edge_soften:
            content_mask_smooth = apply_edge_antialiasing(content_mask_binary)
        else:
            content_mask_smooth = content_mask_binary
    else:
        content_mask_binary = torch.ones_like(depth)
        content_mask_smooth = content_mask_binary

    # Ensure same shape as depth
    while content_mask_binary.dim() < depth.dim():
        content_mask_binary = content_mask_binary.unsqueeze(0)
    while content_mask_smooth.dim() < depth.dim():
        content_mask_smooth = content_mask_smooth.unsqueeze(0)

    # 2. Convert depth to disparity (inverse depth) like V2
    disparity = 1.0 / (depth + epsilon)

    # 3. Use HARD mask for normalization calculations
    disparity_masked = disparity * content_mask_binary

    # 4. Extract ONLY content pixels for normalization
    content_pixels = disparity_masked[content_mask_binary > 0.5]

    if content_pixels.numel() > 100:
        # Get min/max from CONTENT ONLY
        disp_min = content_pixels.min()
        disp_max = content_pixels.max()

        # Use percentile-based normalization for better contrast
        if content_pixels.numel() > 1000:
            sorted_pixels = torch.sort(content_pixels.flatten())[0]
            p1_idx = int(sorted_pixels.numel() * 0.01)
            p99_idx = int(sorted_pixels.numel() * 0.99)
            disp_min = sorted_pixels[p1_idx]
            disp_max = sorted_pixels[p99_idx]

        # Normalize using content-only range
        disparity_norm = (disparity - disp_min) / (disp_max - disp_min + epsilon)
        disparity_norm = torch.clamp(disparity_norm, 0.0, 1.0)
    else:
        # Fallback for very small content regions
        disp_min = disparity.min()
        disp_max = disparity.max()
        disparity_norm = (disparity - disp_min) / (disp_max - disp_min + epsilon)

    # 5. Apply contrast boost
    disparity_contrast = torch.pow(disparity_norm, 1.0 / contrast_boost)

    # 6. Apply SMOOTH mask for final output (with anti-aliased edges)
    disparity_final = disparity_contrast * content_mask_smooth

    # 7. Apply inversion if requested (do this AFTER all processing)
    if invert_depth:
        disparity_final = 1.0 - disparity_final

    return disparity_final


def apply_raw_normalization(depth, invert_depth):
    """Raw/metric depth - no normalization (for 3D reconstruction)."""
    if invert_depth:
        # For raw metric depth, invert as max - depth
        depth = depth.max() - depth

    return depth
