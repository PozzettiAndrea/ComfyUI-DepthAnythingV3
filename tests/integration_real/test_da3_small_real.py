"""
Real model integration tests for Depth Anything V3
These tests download and use actual models on the bridge image
"""
import pytest
import torch
from pathlib import Path
import numpy as np
from PIL import Image


def load_bridge_image():
    """Load the bridge.jpeg test image"""
    # Load the bridge image from assets
    img_path = Path(__file__).parent.parent.parent / "assets" / "bridge.jpeg"
    img = Image.open(img_path).convert("RGB")

    # Resize to reasonable size for testing (keeping aspect ratio)
    img.thumbnail((768, 768), Image.Resampling.LANCZOS)

    # Convert to torch tensor in format (1, H, W, C) with values in [0, 1]
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).unsqueeze(0)

    return img_tensor


@pytest.mark.real_model
def test_da3_small_load(mock_comfy_environment):
    """Test loading real DA3 Small model"""
    from nodes import DownloadAndLoadDepthAnythingV3Model

    loader = DownloadAndLoadDepthAnythingV3Model()

    print("\n=== Loading DA3 Small Model ===")
    # Load smallest DA3 model
    model_dict = loader.loadmodel(
        model="da3_small.safetensors",
        precision="auto"
    )[0]

    assert model_dict is not None
    assert "model" in model_dict
    assert "dtype" in model_dict
    assert "config" in model_dict
    assert model_dict["config"]["encoder"] == "vits"
    print(f"✅ Model loaded successfully: encoder={model_dict['config']['encoder']}, dtype={model_dict['dtype']}")


@pytest.mark.real_model
def test_da3_small_inference_bridge(mock_comfy_environment):
    """Test real depth inference with DA3 Small model on bridge image"""
    from nodes import DownloadAndLoadDepthAnythingV3Model, DepthAnything_V3

    # Load the bridge image
    bridge_image = load_bridge_image()
    print(f"\n=== DA3 Small Inference on Bridge Image ===")
    print(f"Bridge image shape: {bridge_image.shape}")

    # Load model
    loader = DownloadAndLoadDepthAnythingV3Model()
    model_dict = loader.loadmodel(
        model="da3_small.safetensors",
        precision="auto"
    )[0]
    print(f"Model loaded: {model_dict['config']['encoder']}")

    # Run depth estimation
    processor = DepthAnything_V3()
    depth_output = processor.process(
        da3_model=model_dict,
        images=bridge_image
    )[0]

    # Print results
    print(f"Depth output shape: {depth_output.shape}")
    print(f"Depth range: [{depth_output.min():.4f}, {depth_output.max():.4f}]")

    # Save output
    output_dir = Path(__file__).parent.parent / "test_outputs"
    output_dir.mkdir(exist_ok=True)

    # Save input image
    bridge_np = (bridge_image[0].cpu().numpy() * 255).astype(np.uint8)
    bridge_pil = Image.fromarray(bridge_np)
    bridge_pil.save(output_dir / "bridge_input.png")
    print(f"Saved input image to {output_dir / 'bridge_input.png'}")

    # Convert depth tensor to PIL Image and save
    depth_np = (depth_output[0].cpu().numpy() * 255).astype(np.uint8)
    depth_pil = Image.fromarray(depth_np[:, :, 0])
    depth_pil.save(output_dir / "bridge_da3_small_depth.png")
    print(f"Saved depth map to {output_dir / 'bridge_da3_small_depth.png'}")

    # Verify outputs structure
    assert depth_output is not None
    assert depth_output.shape[0] == bridge_image.shape[0]  # Same batch size
    assert depth_output.shape[3] == 3  # 3-channel depth map
    assert depth_output.min() >= 0.0
    assert depth_output.max() <= 1.0
    print("✅ All assertions passed!")


@pytest.mark.real_model
def test_da3_large_load(mock_comfy_environment):
    """Test loading real DA3 Large model"""
    from nodes import DownloadAndLoadDepthAnythingV3Model

    loader = DownloadAndLoadDepthAnythingV3Model()

    # Load DA3 Large model
    model_dict = loader.loadmodel(
        model="da3_large.safetensors",
        precision="auto"
    )[0]

    assert model_dict is not None
    assert "model" in model_dict
    assert "dtype" in model_dict
    assert "config" in model_dict
    assert model_dict["config"]["encoder"] == "vitl"


@pytest.mark.real_model
def test_da3_precision_options(mock_comfy_environment):
    """Test that different precision options work"""
    from nodes import DownloadAndLoadDepthAnythingV3Model

    loader = DownloadAndLoadDepthAnythingV3Model()

    # Test fp16 precision
    model_dict_fp16 = loader.loadmodel(
        model="da3_small.safetensors",
        precision="fp16"
    )[0]

    assert model_dict_fp16["dtype"] == torch.float16

    # Test fp32 precision
    model_dict_fp32 = loader.loadmodel(
        model="da3_small.safetensors",
        precision="fp32"
    )[0]

    assert model_dict_fp32["dtype"] == torch.float32
