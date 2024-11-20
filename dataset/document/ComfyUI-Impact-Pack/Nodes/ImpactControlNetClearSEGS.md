---
tags:
- ImpactPack
- Segmentation
---

# ImpactControlNetClearSEGS
## Documentation
- Class name: `ImpactControlNetClearSEGS`
- Category: `ImpactPack/Util`
- Output node: `False`

This node is designed to clear control networks from segmentation data, effectively resetting any modifications or enhancements previously applied. It operates by iterating over each segment in the provided segmentation data, creating a new segment with the original properties but without the control network information.
## Input types
### Required
- **`segs`**
    - The segmentation data to be processed. This input is essential for the node's operation as it defines the segments whose control networks will be cleared.
    - Comfy dtype: `SEGS`
    - Python dtype: `Tuple[Tuple[int, int], List[SEG]]`
## Output types
- **`segs`**
    - Comfy dtype: `SEGS`
    - The modified segmentation data with control networks cleared from each segment. This output is crucial for subsequent processing steps that require segmentation data without control network modifications.
    - Python dtype: `Tuple[Tuple[int, int], List[SEG]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ControlNetClearSEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"segs": ("SEGS",), }, }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    @staticmethod
    def doit(segs):
        new_segs = []

        for seg in segs[1]:
            new_seg = SEG(seg.cropped_image, seg.cropped_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, None)
            new_segs.append(new_seg)

        return ((segs[0], new_segs), )

```
