import os
import shutil

def copy_assets_to_input():
    script_dir = os.path.dirname(__file__)
    comfyui_root = os.path.dirname(os.path.dirname(script_dir))

    src = os.path.join(script_dir, "assets", "bridge.jpeg")
    dst = os.path.join(comfyui_root, "input", "da3_example_bridge.jpeg")

    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)
        print(f"[DA3] Copied example image to input")

def copy_workflows_to_user():
    script_dir = os.path.dirname(__file__)
    comfyui_root = os.path.dirname(os.path.dirname(script_dir))

    workflows_src = os.path.join(script_dir, "workflows")
    workflows_dst = os.path.join(comfyui_root, "user", "default", "workflows")

    if os.path.exists(workflows_src):
        os.makedirs(workflows_dst, exist_ok=True)
        for workflow in os.listdir(workflows_src):
            if workflow.endswith('.json'):
                src = os.path.join(workflows_src, workflow)
                dst = os.path.join(workflows_dst, f"da3_{workflow}")
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    print(f"[DA3] Copied workflow: {workflow}")

copy_assets_to_input()
copy_workflows_to_user()
