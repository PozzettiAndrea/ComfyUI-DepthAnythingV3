import os
import shutil

def copy_assets_to_input():
    """Copy all files and folders from assets/ to ComfyUI/input/"""
    script_dir = os.path.dirname(__file__)
    comfyui_root = os.path.dirname(os.path.dirname(script_dir))

    assets_dir = os.path.join(script_dir, "assets")
    input_dir = os.path.join(comfyui_root, "input")

    if not os.path.exists(assets_dir):
        return

    # Create input directory if it doesn't exist
    os.makedirs(input_dir, exist_ok=True)

    # Copy all files and directories from assets to input
    for item in os.listdir(assets_dir):
        src_path = os.path.join(assets_dir, item)
        dst_path = os.path.join(input_dir, item)

        # Skip if destination already exists
        if os.path.exists(dst_path):
            continue

        # Copy file or directory
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"[DA3] Copied asset file: {item}")
        elif os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path)
            print(f"[DA3] Copied asset directory: {item}")

copy_assets_to_input()