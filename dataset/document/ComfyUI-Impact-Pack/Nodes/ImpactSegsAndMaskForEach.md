---
tags:
- Segmentation
---

# Pixelwise(SEGS & MASKS ForEach)
## Documentation
- Class name: `ImpactSegsAndMaskForEach`
- Category: `ImpactPack/Operation`
- Output node: `False`

This node applies a bitwise AND operation between each segmentation (seg) and its corresponding mask, effectively filtering the segments based on the mask. It's designed for batch processing, where each segment in a collection is paired with a mask to produce a modified set of segments that only include the areas covered by the masks.
## Input types
### Required
- **`segs`**
    - The collection of segments to be processed. Each segment represents a portion of an image that has been identified for further operations, such as filtering or modification.
    - Comfy dtype: `SEGS`
    - Python dtype: `Tuple[torch.Tensor, List[SEG]]`
- **`masks`**
    - A collection of masks to be applied to the corresponding segments. Each mask defines the areas within a segment to retain, effectively filtering out the rest.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`segs`**
    - Comfy dtype: `SEGS`
    - The modified collection of segments after applying the bitwise AND operation with the corresponding masks. This output retains only the areas of each segment that overlap with the mask.
    - Python dtype: `Tuple[torch.Tensor, List[SEG]]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SegsBitwiseAndMaskForEach:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "segs": ("SEGS",),
                        "masks": ("MASK",),
                    }
                }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Operation"

    def doit(self, segs, masks):
        return (core.apply_mask_to_each_seg(segs, masks), )

```
