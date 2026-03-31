"""
ComfyUI-Kimodo: Text-driven 3D human motion generation with kinematic constraints.

Wraps the Kimodo project (NVIDIA) as ComfyUI custom nodes.
"""
import traceback

print("[ComfyUI-Kimodo] __init__.py loading...", flush=True)

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    print(f"[ComfyUI-Kimodo] Loaded {len(NODE_CLASS_MAPPINGS)} nodes: {list(NODE_CLASS_MAPPINGS.keys())}", flush=True)
except Exception as e:
    print(f"[ComfyUI-Kimodo] Failed to import nodes: {e}", flush=True)
    traceback.print_exc()
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
__version__ = "0.1.0"
