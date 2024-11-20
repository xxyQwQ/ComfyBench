# LayerUtility: Purge VRAM
## Documentation
- Class name: `LayerUtility: PurgeVRAM`
- Category: `😺dzNodes/LayerUtility/SystemIO`
- Output node: `True`

The PurgeVRAM node is designed to free up GPU memory resources by optionally clearing PyTorch's cache and unloading all models from memory. It's a system utility node aimed at optimizing memory usage during runtime, especially useful in environments with limited GPU memory or when running multiple model inference tasks sequentially.
## Input types
### Required
- **`anything`**
    - A placeholder input that does not affect the operation of the node, allowing for flexible integration into workflows.
    - Comfy dtype: `*`
    - Python dtype: `AnyType`
- **`purge_cache`**
    - Determines whether the PyTorch cache should be cleared, helping to free up GPU memory.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`purge_models`**
    - Controls whether all models should be unloaded from memory, further aiding in the optimization of GPU memory usage.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
## Output types
The node doesn't have output types
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class PurgeVRAM:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anything": (any, {}),
                "purge_cache": ("BOOLEAN", {"default": True}),
                "purge_models": ("BOOLEAN", {"default": True}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "purge_vram"
    CATEGORY = '😺dzNodes/LayerUtility/SystemIO'
    OUTPUT_NODE = True

    def purge_vram(self, anything, purge_cache, purge_models):
        import torch.cuda
        import gc
        import comfy.model_management
        gc.collect()
        if purge_cache:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        if purge_models:
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()
        return (None,)

```
