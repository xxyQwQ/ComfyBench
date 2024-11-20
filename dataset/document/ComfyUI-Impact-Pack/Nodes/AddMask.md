---
tags:
- Mask
---

# Pixelwise(MASK + MASK)
## Documentation
- Class name: `AddMask`
- Category: `ImpactPack/Operation`
- Output node: `False`

The AddMask node is designed to combine two mask inputs into a single mask output, effectively merging the areas covered by each input mask. This operation is useful in scenarios where the visual or logical integration of mask regions is required, such as in image processing tasks that involve layering or combining features from different sources.
## Input types
### Required
- **`mask1`**
    - The first mask input for the addition operation. It plays a crucial role in determining the initial areas to be combined with the second mask.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`mask2`**
    - The second mask input for the addition operation. It specifies additional areas to be combined with the first mask, contributing to the final merged output.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The result of combining the two input masks. This output represents the merged area covered by both input masks, providing a unified mask that encompasses the features of both.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [InvertMask](../../Comfy/Nodes/InvertMask.md)



## Source code
```python
class AddMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "mask1": ("MASK",),
            "mask2": ("MASK",),
        }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Operation"

    def doit(self, mask1, mask2):
        mask = add_masks(mask1, mask2)
        return (mask,)

```
