from typing_extensions import override
from comfy_api.latest import ComfyExtension

WEB_DIRECTORY = "./web"


class DA3Extension(ComfyExtension):
    @override
    async def get_node_list(self):
        from .nodes import NODE_CLASSES
        return NODE_CLASSES


async def comfy_entrypoint():
    return DA3Extension()
