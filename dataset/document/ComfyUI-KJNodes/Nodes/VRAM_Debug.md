---
tags:
- Cache
---

# VRAM Debug
## Documentation
- Class name: `VRAM_Debug`
- Category: `KJNodes/misc`
- Output node: `False`

The VRAM_Debug node is designed to monitor and manage the VRAM usage of models within a deep learning environment. It provides functionalities to clear the GPU cache, unload all models from memory, and perform garbage collection to free up VRAM. Additionally, it can return the amount of VRAM freed during its operation, aiding in debugging and optimizing memory usage.
## Input types
### Required
- **`empty_cache`**
    - Controls whether the GPU cache is cleared, which can help in freeing up unused memory.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`gc_collect`**
    - Determines whether garbage collection is performed, potentially freeing up memory by removing unused objects.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`unload_all_models`**
    - Specifies whether all models should be unloaded from memory, which can significantly reduce VRAM usage.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`any_input`**
    - An optional input that can be passed through the node without modification, allowing for flexible usage.
    - Comfy dtype: `*`
    - Python dtype: `Optional[Any]`
- **`image_pass`**
    - An optional image input that can be passed through the node without modification.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Optional[Any]`
- **`model_pass`**
    - An optional model input that can be passed through the node without modification.
    - Comfy dtype: `MODEL`
    - Python dtype: `Optional[Any]`
## Output types
- **`any_output`**
    - Comfy dtype: `*`
    - Returns the inputs passed to the node along with the VRAM usage information before and after the operation.
    - Python dtype: `Tuple[Any, Optional[Any], Optional[Any], float, float]`
- **`image_pass`**
    - Comfy dtype: `IMAGE`
    - Returns the optional image input passed through the node without modification.
    - Python dtype: `Optional[Any]`
- **`model_pass`**
    - Comfy dtype: `MODEL`
    - Returns the optional model input passed through the node without modification.
    - Python dtype: `Optional[Any]`
- **`freemem_before`**
    - Comfy dtype: `INT`
    - Returns the amount of free VRAM before the operation.
    - Python dtype: `float`
- **`freemem_after`**
    - Comfy dtype: `INT`
    - Returns the amount of free VRAM after the operation.
    - Python dtype: `float`
- **`ui`**
    - Provides a visual representation of the VRAM usage before and after the operation.
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class VRAM_Debug:
    
    @classmethod
    
    def INPUT_TYPES(s):
      return {
        "required": {
            
            "empty_cache": ("BOOLEAN", {"default": True}),
            "gc_collect": ("BOOLEAN", {"default": True}),
            "unload_all_models": ("BOOLEAN", {"default": False}),
        },
        "optional": {
            "any_input": (any, {}),
            "image_pass": ("IMAGE",),
            "model_pass": ("MODEL",),
        }
	}
        
    RETURN_TYPES = (any, "IMAGE","MODEL","INT", "INT",)
    RETURN_NAMES = ("any_output", "image_pass", "model_pass", "freemem_before", "freemem_after")
    FUNCTION = "VRAMdebug"
    CATEGORY = "KJNodes/misc"
    DESCRIPTION = """
Returns the inputs unchanged, they are only used as triggers,  
and performs comfy model management functions and garbage collection,  
reports free VRAM before and after the operations.
"""

    def VRAMdebug(self, gc_collect, empty_cache, unload_all_models, image_pass=None, model_pass=None, any_input=None):
        freemem_before = model_management.get_free_memory()
        print("VRAMdebug: free memory before: ", f"{freemem_before:,.0f}")
        if empty_cache:
            model_management.soft_empty_cache()
        if unload_all_models:
            model_management.unload_all_models()
        if gc_collect:
            import gc
            gc.collect()
        freemem_after = model_management.get_free_memory()
        print("VRAMdebug: free memory after: ", f"{freemem_after:,.0f}")
        print("VRAMdebug: freed memory: ", f"{freemem_after - freemem_before:,.0f}")
        return {"ui": {
            "text": [f"{freemem_before:,.0f}x{freemem_after:,.0f}"]}, 
            "result": (any_input, image_pass, model_pass, freemem_before, freemem_after) 
        }

```
