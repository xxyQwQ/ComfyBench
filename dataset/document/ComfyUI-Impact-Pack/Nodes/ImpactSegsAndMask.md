---
tags:
- Segmentation
---

# Pixelwise(SEGS & MASK)
## Documentation
- Class name: `ImpactSegsAndMask`
- Category: `ImpactPack/Operation`
- Output node: `False`

The node applies a bitwise AND operation between segmentation masks and a given mask, modifying the segmentation data based on the overlap with the mask. This operation is useful for refining segmentation results by incorporating additional mask information, enhancing the accuracy and relevance of the segmented output.
## Input types
### Required
- **`segs`**
    - The segmentation data to be refined. It plays a crucial role in determining the final output by providing the initial set of segments to be modified.
    - Comfy dtype: `SEGS`
    - Python dtype: `Tuple[torch.Tensor, List[SEG]]`
- **`mask`**
    - The mask to apply to the segmentation data. It affects the operation by defining which parts of the segments will be retained or discarded based on the bitwise AND operation.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`segs`**
    - Comfy dtype: `SEGS`
    - The modified segmentation data after applying the bitwise AND operation with the given mask. It represents the refined set of segments that have been altered based on the mask's criteria.
    - Python dtype: `Tuple[torch.Tensor, List[SEG]]`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [ImpactSEGSOrderedFilter](../../ComfyUI-Impact-Pack/Nodes/ImpactSEGSOrderedFilter.md)



## Source code
```python
class SegsBitwiseAndMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "segs": ("SEGS",),
                        "mask": ("MASK",),
                    }
                }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Operation"

    def doit(self, segs, mask):
        return (core.segs_bitwise_and_mask(segs, mask), )

```
