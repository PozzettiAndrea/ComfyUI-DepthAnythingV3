from .nodes import NODE_CLASSES

# 将作者提供的 List 格式转换为 ComfyUI 官方要求的 Dict 格式
NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in NODE_CLASSES}

# 自动美化节点在 ComfyUI UI 界面上的显示名称（例如把 DA3_ToPointCloud 变成 DA3 ToPointCloud）
NODE_DISPLAY_NAME_MAPPINGS = {
    cls.__name__: cls.__name__.replace("_", " ") for cls in NODE_CLASSES
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
