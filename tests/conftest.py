"""
Pytest configuration and fixtures for ComfyUI-DepthAnythingV3 tests
"""
import sys
import os
from pathlib import Path
import pytest
import torch
from unittest.mock import MagicMock


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--use-gpu",
        action="store_true",
        default=False,
        help="Run tests on GPU instead of CPU (much faster for real model tests)"
    )


# Add the custom node directory to Python path so we can import nodes package
custom_nodes_dir = Path(__file__).parent.parent
sys.path.insert(0, str(custom_nodes_dir))

# Mock ComfyUI modules at module level BEFORE pytest starts
# This prevents import errors when pytest tries to load __init__.py files
mock_folder_paths = type("folder_paths", (), {})()
mock_folder_paths.models_dir = "/tmp/test_models"
mock_folder_paths.get_folder_paths = lambda x: ["/tmp/test_models"]
sys.modules["folder_paths"] = mock_folder_paths

mock_comfy = type("comfy", (), {})()
mock_comfy_utils = type("utils", (), {})()
mock_comfy_utils.load_torch_file = lambda x: {}
mock_comfy_utils.ProgressBar = MagicMock()
mock_comfy.utils = mock_comfy_utils

mock_comfy_mm = type("model_management", (), {})()

# Device selection: Check environment variable set by session fixture
def _get_test_device():
    """Get device for testing - GPU if --use-gpu flag is set, else CPU"""
    use_gpu = os.environ.get("PYTEST_USE_GPU", "0") == "1"
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

mock_comfy_mm.get_torch_device = _get_test_device
mock_comfy_mm.soft_empty_cache = lambda: None
mock_comfy_mm.load_models_gpu = lambda x: None
mock_comfy_mm.unet_offload_device = lambda: torch.device("cpu")
mock_comfy_mm.is_device_mps = lambda x: False
mock_comfy_mm.get_autocast_device = lambda x: "cpu"
mock_comfy.model_management = mock_comfy_mm

sys.modules["comfy"] = mock_comfy
sys.modules["comfy.utils"] = mock_comfy_utils
sys.modules["comfy.model_management"] = mock_comfy_mm


def pytest_ignore_collect(collection_path, path, config):
    """Ignore __init__.py files during collection"""
    if collection_path.name == "__init__.py":
        return True
    return False


@pytest.fixture(scope="session", autouse=True)
def setup_test_device(request):
    """Configure test device based on --use-gpu flag"""
    use_gpu = request.config.getoption("--use-gpu")
    if use_gpu:
        os.environ["PYTEST_USE_GPU"] = "1"
        if torch.cuda.is_available():
            print(f"\n🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("\n⚠️  --use-gpu specified but CUDA not available, using CPU")
    else:
        os.environ["PYTEST_USE_GPU"] = "0"
        print("\n💻 Using CPU (use --use-gpu for GPU acceleration)")

    yield

    # Cleanup
    os.environ.pop("PYTEST_USE_GPU", None)


@pytest.fixture(scope="session", autouse=True)
def setup_mock_comfy():
    """Set up mock ComfyUI modules for testing - runs once per session"""
    # Ensure mocks persist throughout test session
    return True


@pytest.fixture
def mock_comfy_environment():
    """Provide access to mocked ComfyUI environment (already set up at module level)"""
    return sys.modules["folder_paths"]


@pytest.fixture
def sample_image():
    """Create a small test image as torch tensor"""
    import numpy as np

    # Create a 512x512 RGB image
    img_np = np.random.rand(512, 512, 3).astype(np.float32)
    img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # (1, H, W, C)

    return img_tensor


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, no model loading)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests with mocked models"
    )
    config.addinivalue_line(
        "markers", "real_model: Tests that download and use real models (slow)"
    )
