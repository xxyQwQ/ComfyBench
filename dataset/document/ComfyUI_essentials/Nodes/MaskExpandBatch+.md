---
tags:
- Mask
- MaskBatch
---

# 🔧 Mask Expand Batch
## Documentation
- Class name: `MaskExpandBatch+`
- Category: `essentials/mask batch`
- Output node: `False`

This node is designed to expand a given mask to a specified size using a chosen method, such as repeating or expanding the mask in various ways. It abstracts the complexity of manipulating mask dimensions and provides a straightforward way to adjust mask sizes for batch processing.
## Input types
### Required
- **`mask`**
    - The mask input represents the initial mask that will be expanded. It is crucial for determining the base pattern or shape that will be manipulated according to the specified size and method.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`size`**
    - The size parameter specifies the target size to which the mask will be expanded. It plays a key role in determining the final dimensions of the output mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`method`**
    - The method parameter dictates the technique used to expand the mask, offering options like 'expand', 'repeat all', 'repeat first', and 'repeat last'. This choice significantly influences the pattern and distribution of the expanded mask.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The output is an expanded mask that has been adjusted to the specified size using the chosen method, suitable for further processing or application in batch operations.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class MaskExpandBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "size": ("INT", { "default": 16, "min": 1, "step": 1, }),
                "method": (["expand", "repeat all", "repeat first", "repeat last"],)
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials/mask batch"

    def execute(self, mask, size, method):
        return (ImageExpandBatch().execute(mask.unsqueeze(1).expand(-1,3,-1,-1), size, method)[0][:,0,:,:],)

```
