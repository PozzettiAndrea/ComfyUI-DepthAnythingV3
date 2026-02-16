from pathlib import Path
from comfy_env import setup_env, copy_files
from comfy_3d_viewers import copy_viewer

setup_env()

SCRIPT_DIR = Path(__file__).resolve().parent
COMFYUI_DIR = SCRIPT_DIR.parent.parent

# Copy pointcloud VTK viewer from comfy-3d-viewers
copy_viewer("pointcloud_vtk", SCRIPT_DIR / "web")

# Copy dynamic widgets JS
try:
    from comfy_dynamic_widgets import get_js_path
    import shutil
    src = Path(get_js_path())
    if src.exists():
        dst = SCRIPT_DIR / "web" / "js" / "dynamic_widgets.js"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(src, dst)
except ImportError:
    pass

# Copy assets
src_dir = SCRIPT_DIR / "assets"
dst_dir = COMFYUI_DIR / "input"
copy_files(src_dir, dst_dir, "**/*")
copied_files = [f.name for f in src_dir.glob("**/*") if f.is_file()]
print(f"[DepthAnythingV3] Copied {len(copied_files)} asset(s) to {dst_dir}: {copied_files}")
