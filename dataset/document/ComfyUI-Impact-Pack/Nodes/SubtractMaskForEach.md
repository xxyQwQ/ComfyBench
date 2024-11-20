---
tags:
- MaskMath
---

# Pixelwise(SEGS - SEGS)
## Documentation
- Class name: `SubtractMaskForEach`
- Category: `ImpactPack/Operation`
- Output node: `False`

The SubtractMaskForEach node is designed to perform a bitwise subtraction operation between segmentation masks (SEGS) and a base mask, producing a modified set of segmentation masks. This operation is useful in scenarios where it's necessary to remove specific features or areas from a set of segmentation masks based on another mask.
## Input types
### Required
- **`base_segs`**
    - The base segmentation masks (SEGS) from which features or areas will be subtracted.
    - Comfy dtype: `SEGS`
    - Python dtype: `List[SEG]`
- **`mask_segs`**
    - The segmentation masks (SEGS) that will be used as the subtraction mask, indicating the features or areas to be removed from the base segmentation masks.
    - Comfy dtype: `SEGS`
    - Python dtype: `List[SEG]`
## Output types
- **`segs`**
    - Comfy dtype: `SEGS`
    - The resulting set of segmentation masks (segs) after the subtraction operation, with specific features or areas removed.
    - Python dtype: `List[SEG]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SubtractMaskForEach:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "base_segs": ("SEGS",),
                        "mask_segs": ("SEGS",),
                    }
                }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Operation"

    def doit(self, base_segs, mask_segs):
        mask = core.segs_to_combined_mask(mask_segs)
        mask = make_3d_mask(mask)
        return (core.segs_bitwise_subtract_mask(base_segs, mask), )

```
