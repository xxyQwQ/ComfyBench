---
tags:
- Cache
---

# Clean GPU Used
## Documentation
- Class name: `easy cleanGpuUsed`
- Category: `EasyUse/Logic`
- Output node: `True`

The `cleanGpuUsed` node is designed to facilitate the release of GPU resources by clearing the GPU memory cache and unloading all models from memory. This operation is crucial for managing GPU memory efficiently, especially in environments where multiple models are loaded and unloaded dynamically.
## Input types
### Required
- **`anything`**
    - This parameter acts as a placeholder to allow the node to be called without specific input requirements, emphasizing its utility function rather than processing specific data.
    - Comfy dtype: `*`
    - Python dtype: `Any`
### Optional
## Output types
The node doesn't have output types
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class cleanGPUUsed:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"anything": (AlwaysEqualProxy("*"), {})}, "optional": {},
                "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO",
                           }}

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True
    FUNCTION = "empty_cache"
    CATEGORY = "EasyUse/Logic"

    def empty_cache(self, anything, unique_id=None, extra_pnginfo=None):
        cleanGPUUsedForce()
        remove_cache('*')
        return ()

```
