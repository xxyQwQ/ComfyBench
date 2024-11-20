---
tags:
- Segmentation
---

# Pixelwise(SEGS & SEGS)
## Documentation
- Class name: `BitwiseAndMaskForEach`
- Category: `ImpactPack/Operation`
- Output node: `False`

The BitwiseAndMaskForEach node applies a bitwise AND operation between a base segmentation (or segments) and a mask for each segment, effectively combining the mask information with each segment. This operation is useful for refining segmentation results by incorporating additional mask constraints.
## Input types
### Required
- **`base_segs`**
    - The base segments to which the mask will be applied. This input is crucial for defining the scope of the operation and determining the segments that will be refined by the mask.
    - Comfy dtype: `SEGS`
    - Python dtype: `torch.Tensor`
- **`mask_segs`**
    - The mask segments to be applied to the base segments. These masks dictate the areas of interest within each segment and are essential for the bitwise AND operation.
    - Comfy dtype: `SEGS`
    - Python dtype: `torch.Tensor`
## Output types
- **`segs`**
    - Comfy dtype: `SEGS`
    - The resulting segments after applying the bitwise AND operation with the mask. This output reflects the refined segmentation, incorporating the constraints imposed by the mask.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class BitwiseAndMaskForEach:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
            {
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

        return SegsBitwiseAndMask().doit(base_segs, mask)

```
