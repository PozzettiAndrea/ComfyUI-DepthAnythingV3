"""
Smoke test for ComfyUI-DepthAnythingV3
Tests basic module import functionality without requiring ComfyUI to be installed
"""

import sys
import traceback
from pathlib import Path

# Add nodes directory to path so depth_anything_v3 can be imported
# (depth_anything_v3 is now inside the nodes package)
_nodes_dir = Path(__file__).parent.parent / "nodes"
if str(_nodes_dir) not in sys.path:
    sys.path.insert(0, str(_nodes_dir))

# Mock ComfyUI dependencies before importing nodes
# This allows smoke test to run without ComfyUI installed
mock_folder_paths = type("folder_paths", (), {})()
mock_folder_paths.models_dir = "/tmp/test_models"
mock_folder_paths.get_folder_paths = lambda x: ["/tmp/test_models"]
sys.modules["folder_paths"] = mock_folder_paths

mock_comfy_mm = type("model_management", (), {})()
mock_comfy_mm.get_torch_device = lambda: "cpu"
mock_comfy_mm.soft_empty_cache = lambda: None
mock_comfy_mm.load_models_gpu = lambda x: None
mock_comfy_mm.unet_offload_device = lambda: "cpu"
mock_comfy_mm.is_device_mps = lambda x: False
mock_comfy_mm.get_autocast_device = lambda x: "cpu"

mock_comfy_utils = type("utils", (), {})()
mock_comfy_utils.load_torch_file = lambda x: {}
mock_comfy_utils.ProgressBar = type("ProgressBar", (), {})

mock_comfy = type("comfy", (), {})()
mock_comfy.model_management = mock_comfy_mm
mock_comfy.utils = mock_comfy_utils

sys.modules["comfy"] = mock_comfy
sys.modules["comfy.model_management"] = mock_comfy_mm
sys.modules["comfy.utils"] = mock_comfy_utils

def test_basic_imports():
    """Test that basic Python dependencies can be imported"""
    print("Testing basic dependencies...")
    try:
        import torch
        print(f"  [PASS] torch {torch.__version__}")
    except ImportError as e:
        print(f"  [FAIL] torch: {e}")
        return False

    try:
        import numpy as np
        print(f"  [PASS] numpy {np.__version__}")
    except ImportError as e:
        print(f"  [FAIL] numpy: {e}")
        return False

    try:
        from torchvision import transforms
        print(f"  [PASS] torchvision")
    except ImportError as e:
        print(f"  [FAIL] torchvision: {e}")
        return False

    try:
        import einops
        print(f"  [PASS] einops")
    except ImportError as e:
        print(f"  [FAIL] einops: {e}")
        return False

    return True

def test_config_imports():
    """Test that config modules can be imported"""
    print("\nTesting config imports...")
    try:
        from depth_anything_v3.configs import MODEL_CONFIGS, MODEL_REPOS
        print(f"  [PASS] Config imported ({len(MODEL_CONFIGS)} models)")
        return True
    except Exception as e:
        print(f"  [FAIL] Config import failed: {e}")
        traceback.print_exc()
        return False

def test_model_module_imports():
    """Test that model modules can be imported"""
    print("\nTesting model module imports...")
    modules = [
        ('depth_anything_v3.model.da3', 'DepthAnything3Net'),
        ('depth_anything_v3.model.dpt', 'DPT'),
        ('depth_anything_v3.model.dualdpt', 'DualDPT'),
        ('depth_anything_v3.model.dinov2.dinov2', 'DinoV2'),
    ]

    failed = []
    for module_name, class_name in modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"  [PASS] {module_name}.{class_name}")
        except Exception as e:
            print(f"  [FAIL] {module_name}.{class_name}: {e}")
            failed.append(module_name)

    return len(failed) == 0

def test_node_class_definitions():
    """Test that node classes are defined"""
    print("\nTesting node class definitions...")
    try:
        from nodes import (
            DownloadAndLoadDepthAnythingV3Model,
            DepthAnything_V3,
            NODE_CLASS_MAPPINGS,
            NODE_DISPLAY_NAME_MAPPINGS,
        )

        # Check they're not None
        classes = {
            'DownloadAndLoadDepthAnythingV3Model': DownloadAndLoadDepthAnythingV3Model,
            'DepthAnything_V3': DepthAnything_V3,
        }

        failed = []
        for name, cls in classes.items():
            if cls is None:
                print(f"  [FAIL] {name} is None!")
                failed.append(name)
            else:
                print(f"  [PASS] {name}")

        # Check mappings
        if len(NODE_CLASS_MAPPINGS) == 2:
            print(f"  [PASS] NODE_CLASS_MAPPINGS (2 nodes)")
        else:
            print(f"  [FAIL] NODE_CLASS_MAPPINGS has {len(NODE_CLASS_MAPPINGS)} nodes, expected 2")
            failed.append("NODE_CLASS_MAPPINGS")

        if len(NODE_DISPLAY_NAME_MAPPINGS) == 2:
            print(f"  [PASS] NODE_DISPLAY_NAME_MAPPINGS (2 nodes)")
        else:
            print(f"  [FAIL] NODE_DISPLAY_NAME_MAPPINGS has {len(NODE_DISPLAY_NAME_MAPPINGS)} nodes, expected 2")
            failed.append("NODE_DISPLAY_NAME_MAPPINGS")

        return len(failed) == 0

    except Exception as e:
        print(f"  [FAIL] Node import failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all smoke tests"""
    print("="*60)
    print("ComfyUI-DepthAnythingV3 Smoke Test")
    print("="*60)

    results = []

    # Run tests
    results.append(("Basic imports", test_basic_imports()))
    results.append(("Config imports", test_config_imports()))
    results.append(("Model modules", test_model_module_imports()))
    results.append(("Node classes", test_node_class_definitions()))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("[PASS] All smoke tests passed!")
        sys.exit(0)
    else:
        print("[FAIL] Some smoke tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
