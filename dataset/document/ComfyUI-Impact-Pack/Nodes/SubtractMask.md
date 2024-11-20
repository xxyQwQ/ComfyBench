---
tags:
- MaskMath
---

# Pixelwise(MASK - MASK)
## Documentation
- Class name: `SubtractMask`
- Category: `ImpactPack/Operation`
- Output node: `False`

The SubtractMask node is designed for performing pixel-wise subtraction between two mask images, resulting in a new mask that highlights the differences. This operation is fundamental in image processing tasks where the goal is to isolate changes or differences between two images.
## Input types
### Required
- **`mask1`**
    - The first mask image to be subtracted from. It plays a crucial role in determining the base or reference for the subtraction operation.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`mask2`**
    - The second mask image to be subtracted. This mask is subtracted from the first one to highlight differences.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The resulting mask after subtracting mask2 from mask1, highlighting the differences between the two.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SubtractMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "mask1": ("MASK", ),
                        "mask2": ("MASK", ),
                      }
                }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Operation"

    def doit(self, mask1, mask2):
        mask = subtract_masks(mask1, mask2)
        return (mask,)

```
