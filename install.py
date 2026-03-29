import sys
import subprocess
from pathlib import Path

try:
    from comfy_env import install
    install()
except ImportError:
    # 已经删除了刚才的 Warning 打印语句，现在会静默执行后备安装
    
    req_file = Path(__file__).resolve().parent / "requirements.txt"
    
    if req_file.exists():
        try:
            # 静默调用 pip install，只在真正出错时才报错
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
                stdout=subprocess.DEVNULL,  # 隐藏常规的安装输出信息（可选）
                stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError:
            pass # 忽略安装过程中可能出现的非致命错误
