# ComfyUI-DepthAnythingV3 Tests

This directory contains the test suite for ComfyUI-DepthAnythingV3.

## Test Structure

```
tests/
├── conftest.py                 # Pytest configuration and fixtures
├── pytest.ini                  # Pytest settings (in parent directory)
├── smoke_test.py              # Basic import checks (no model loading)
└── integration_real/          # Real model integration tests
    └── test_da3_small_real.py # Tests with actual DA3 models on bridge image
```

## Test Types

### 1. Smoke Tests
**File**: `smoke_test.py`
**Purpose**: Verify basic imports and module structure
**Runtime**: < 5 seconds
**Requirements**: No model downloads needed

Run with:
```bash
python tests/smoke_test.py
```

### 2. Integration Tests (Real Models)
**Directory**: `integration_real/`
**Purpose**: Test actual model loading and inference on the bridge image
**Runtime**: ~2-5 minutes (first run downloads ~300MB model)
**Requirements**: Downloads DA3 Small model from HuggingFace

Run with:
```bash
# Run all integration tests
pytest tests/integration_real/ -v -m real_model

# Run specific test
pytest tests/integration_real/test_da3_small_real.py::test_da3_small_inference_bridge -v -s
```

## GitHub Actions Workflows

### Smoke Tests (Fast)
- **Linux**: `.github/workflows/test-unit.yml`
- **Windows**: `.github/workflows/test-windows.yml`
- **macOS**: `.github/workflows/test-macos.yml`

Runs on: Python 3.10, 3.11, 3.12
Duration: ~1 minute per platform

### Integration Tests (Slow)
- **Linux**: `.github/workflows/test-linux-integration.yml`
- **Windows**: `.github/workflows/test-windows-integration.yml`
- **macOS**: `.github/workflows/test-macos-integration.yml`

Runs on: Python 3.10
Duration: ~3-5 minutes per platform (cached models make reruns faster)

**Artifacts**: Each integration test uploads depth map outputs as artifacts for visual inspection.

## Test Image

All integration tests use the bridge image from `assets/bridge.jpeg`:
- Black and white architectural bridge photo
- Resized to 768x768 for testing
- Good depth variation (close bridge structure, distant background)

## Output Artifacts

Integration tests save outputs to `tests/test_outputs/`:
- `bridge_input.png` - Resized input image
- `bridge_da3_small_depth.png` - Generated depth map

These are uploaded as GitHub Actions artifacts for inspection.

## Running Tests Locally

### Install test dependencies:
```bash
pip install -r .github/requirements-ci.txt
pip install -r requirements-dev.txt
pip install pytest pillow
```

### Run smoke tests:
```bash
python tests/smoke_test.py
```

### Run integration tests:
```bash
# Run all real model tests
pytest tests/integration_real/ -v -m real_model -s

# Run with GPU (if available)
pytest tests/integration_real/ -v -m real_model --use-gpu
```

## Markers

- `@pytest.mark.unit` - Fast unit tests (not implemented yet)
- `@pytest.mark.real_model` - Tests that download and use real models

## Model Caching

Models are cached in `~/.cache/huggingface/` to speed up subsequent test runs.
