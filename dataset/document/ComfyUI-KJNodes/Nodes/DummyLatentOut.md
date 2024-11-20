---
tags:
- Latent
---

# Dummy Latent Out
## Documentation
- Class name: `DummyLatentOut`
- Category: `KJNodes/misc`
- Output node: `True`

Provides a simple pass-through for latent data, facilitating the visualization of workflow outputs in the UI without necessitating data persistence on disk.
## Input types
### Required
- **`latent`**
    - The latent data to be passed through. This input is essential for the node's operation as it directly influences the output by being returned unchanged.
    - Comfy dtype: `LATENT`
    - Python dtype: `Tuple[torch.Tensor]`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The unchanged latent data received as input, enabling direct visualization in the UI.
    - Python dtype: `Tuple[torch.Tensor]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class DummyLatentOut:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
            "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "dummy"
    CATEGORY = "KJNodes/misc"
    OUTPUT_NODE = True
    DESCRIPTION = """
Does nothing, used to trigger generic workflow output.    
A way to get previews in the UI without saving anything to disk.
"""

    def dummy(self, latent):
        return (latent,)

```
