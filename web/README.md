# Point Cloud & Gaussian Splat Preview

This extension provides interactive 3D visualization for point clouds and Gaussian splats using Three.js.

## Features

- **Interactive 3D Viewing**: Rotate, pan, and zoom with mouse controls
- **Automatic Centering**: Point clouds are automatically centered and scaled to fit the view
- **RGB Color Support**: Displays vertex colors from PLY files
- **Real-time Updates**: Preview updates automatically when workflow executes

## Mouse Controls

- **Left Click + Drag**: Rotate the view around the point cloud
- **Right Click + Drag**: Pan the camera
- **Mouse Wheel**: Zoom in/out

## Usage

### Basic Workflow

```
[Load Image]
    ↓
[DA3 Model Loader]
    ↓
[DA3 Advanced] → depth, confidence, ray_origin, ray_direction
    ↓            ↓            ↓              ↓
[DA3 To Point Cloud] ← source_image (optional)
    ↓
[DA3 Save Point Cloud] → file_path
    ↓
[DA3 Preview Point Cloud / Gaussians]
```

### With Gaussian Splats

```
[Load Image]
    ↓
[DA3 Model Loader] (fine-tuned GS model)
    ↓
[DA3 To 3D Gaussians]
    ↓
[DA3 Save 3D Gaussians] → file_path
    ↓
[DA3 Preview Point Cloud / Gaussians]
```

## Technical Details

- Uses Three.js for WebGL rendering
- Automatically loads Three.js from CDN if not available
- Parses PLY ASCII format
- Supports both point clouds and Gaussian splat PLY files
- Point size: 0.005 units (auto-scaled to view)

## Troubleshooting

### Preview not showing
1. Make sure ComfyUI is restarted after installing the extension
2. Check browser console for JavaScript errors
3. Verify the PLY file was saved correctly

### Colors appear black
- Connect the source image to `DA3_ToPointCloud` → `source_image` input
- Without source image, points will be white by default

### Point cloud too small/large
- The preview automatically scales point clouds to fit the view
- If scaling looks wrong, check your depth/ray values

### Browser compatibility
- Requires WebGL support
- Tested on Chrome, Firefox, Edge
- May not work on older browsers
