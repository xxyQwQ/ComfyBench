---
tags:
- ImpactPack
- Segmentation
---

# Remove Image from SEGS
## Documentation
- Class name: `RemoveImageFromSEGS`
- Category: `ImpactPack/Util`
- Output node: `False`

The `RemoveImageFromSEGS` node is designed to process a collection of segmented images (SEGS) and remove the image data from each segment, effectively leaving the segments with their metadata and masks intact. This operation is useful for scenarios where the image data is no longer needed or desired, and only the segmentation information is of interest.
## Input types
### Required
- **`segs`**
    - The `segs` parameter represents the collection of segmented images (SEGS) to be processed. It is crucial for specifying the input segments from which the image data will be removed, maintaining the segmentation's other attributes.
    - Comfy dtype: `SEGS`
    - Python dtype: `Tuple[Tuple[int, int], List[SEG]]`
## Output types
- **`segs`**
    - Comfy dtype: `SEGS`
    - The output is a modified version of the input SEGS, where the image data has been removed from each segment, leaving only the segmentation metadata and masks.
    - Python dtype: `Tuple[Tuple[int, int], List[SEG]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RemoveImageFromSEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"segs": ("SEGS", ), }}

    RETURN_TYPES = ("SEGS", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    @staticmethod
    def doit(segs):
        results = []

        if len(segs[1]) > 0:
            for seg in segs[1]:
                new_seg = SEG(None, seg.cropped_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, seg.control_net_wrapper)
                results.append(new_seg)

            return ((segs[0], results), )
        else:
            return (segs, )

```
