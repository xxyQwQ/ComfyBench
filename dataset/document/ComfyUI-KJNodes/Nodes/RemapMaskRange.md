---
tags:
- ImageTransformation
---

# Remap Mask Range
## Documentation
- Class name: `RemapMaskRange`
- Category: `KJNodes/masking`
- Output node: `False`

The `RemapMaskRange` node is designed to adjust the range of values within a mask to a specified new minimum and maximum. This operation is useful for normalizing mask values or adjusting their scale for further processing.
## Input types
### Required
- **`mask`**
    - The input mask whose values are to be remapped. This mask serves as the basis for scaling to the new specified range.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`min`**
    - The new minimum value to which the lowest value in the mask will be scaled. This parameter sets the lower bound of the new value range for the mask.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`max`**
    - The new maximum value to which the highest value in the mask will be scaled. This parameter sets the upper bound of the new value range for the mask.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The output mask with its values remapped to the new specified range. This mask is scaled such that its values lie between the new minimum and maximum values.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class RemapMaskRange:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "min": ("FLOAT", {"default": 0.0,"min": -10.0, "max": 1.0, "step": 0.01}),
                "max": ("FLOAT", {"default": 1.0,"min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "remap"
    CATEGORY = "KJNodes/masking"
    DESCRIPTION = """
Sets new min and max values for the mask.
"""

    def remap(self, mask, min, max):

         # Find the maximum value in the mask
        mask_max = torch.max(mask)
        
        # If the maximum mask value is zero, avoid division by zero by setting it to 1
        mask_max = mask_max if mask_max > 0 else 1
        
        # Scale the mask values to the new range defined by min and max
        # The highest pixel value in the mask will be scaled to max
        scaled_mask = (mask / mask_max) * (max - min) + min
        
        # Clamp the values to ensure they are within [0.0, 1.0]
        scaled_mask = torch.clamp(scaled_mask, min=0.0, max=1.0)
        
        return (scaled_mask, )

```
