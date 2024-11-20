---
tags:
- Flip
- Image
- ImageTransformation
---

# 🔧 Mask Flip
## Documentation
- Class name: `MaskFlip+`
- Category: `essentials/mask`
- Output node: `False`

The MaskFlip+ node is designed for flipping masks along specified axes. It provides functionality to manipulate the orientation of mask data, allowing for horizontal, vertical, or both flips, enhancing flexibility in mask manipulation for various image processing tasks.
## Input types
### Required
- **`mask`**
    - The 'mask' parameter represents the input mask to be flipped. It is crucial for determining the spatial data that will undergo orientation changes.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`axis`**
    - The 'axis' parameter specifies the axes along which the mask will be flipped. It plays a key role in defining the direction of the flip operation, affecting the final output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `Tuple[str]`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The output is a flipped version of the input mask, modified according to the specified axis or axes of flipping.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MaskFlip:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "axis": (["x", "y", "xy"],),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials/mask"

    def execute(self, mask, axis):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        dim = ()
        if "y" in axis:
            dim += (1,)
        if "x" in axis:
            dim += (2,)
        mask = torch.flip(mask, dims=dim)

        return(mask,)

```
