---
tags:
- ImpactPack
- Segmentation
---

# SEGS Filter (ordered)
## Documentation
- Class name: `ImpactSEGSOrderedFilter`
- Category: `ImpactPack/Util`
- Output node: `False`

The ImpactSEGSOrderedFilter node is designed to filter and order segmentation elements (SEGS) based on specified criteria such as area, width, height, or position. It allows for the selection of a subset of SEGS by ordering them according to the specified criterion and then taking a specified range from the ordered list. This functionality is crucial for scenarios where prioritization or specific ordering of segmentation elements is needed based on their geometric properties or confidence levels.
## Input types
### Required
- **`segs`**
    - The segmentation elements (SEGS) to be filtered and ordered. This is the primary input on which the ordering and filtering operations are performed.
    - Comfy dtype: `SEGS`
    - Python dtype: `list`
- **`target`**
    - Specifies the criterion used for ordering the SEGS, such as area, width, height, or specific coordinates. This criterion determines how the SEGS will be sorted before selecting a subset.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`order`**
    - Determines the order in which SEGS are sorted based on the target criterion. A boolean value where True indicates descending order and False indicates ascending order.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`take_start`**
    - Specifies the starting index from which to begin selecting SEGS after they have been ordered. This allows for skipping a certain number of top elements.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`take_count`**
    - Defines the number of SEGS to select starting from the take_start position. This determines the size of the subset of ordered SEGS to be returned.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`filtered_SEGS`**
    - Comfy dtype: `SEGS`
    - The subset of SEGS that have been filtered and ordered according to the specified criteria.
    - Python dtype: `list`
- **`remained_SEGS`**
    - Comfy dtype: `SEGS`
    - The SEGS that were not included in the filtered subset, remaining after the ordering and selection process.
    - Python dtype: `list`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [ImpactDilateMaskInSEGS](../../ComfyUI-Impact-Pack/Nodes/ImpactDilateMaskInSEGS.md)
    - [DetailerForEachDebug](../../ComfyUI-Impact-Pack/Nodes/DetailerForEachDebug.md)



## Source code
```python
class SEGSOrderedFilter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "segs": ("SEGS", ),
                        "target": (["area(=w*h)", "width", "height", "x1", "y1", "x2", "y2", "confidence"],),
                        "order": ("BOOLEAN", {"default": True, "label_on": "descending", "label_off": "ascending"}),
                        "take_start": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                        "take_count": ("INT", {"default": 1, "min": 0, "max": sys.maxsize, "step": 1}),
                     },
                }

    RETURN_TYPES = ("SEGS", "SEGS",)
    RETURN_NAMES = ("filtered_SEGS", "remained_SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, segs, target, order, take_start, take_count):
        segs_with_order = []

        for seg in segs[1]:
            x1 = seg.crop_region[0]
            y1 = seg.crop_region[1]
            x2 = seg.crop_region[2]
            y2 = seg.crop_region[3]

            if target == "area(=w*h)":
                value = (y2 - y1) * (x2 - x1)
            elif target == "width":
                value = x2 - x1
            elif target == "height":
                value = y2 - y1
            elif target == "x1":
                value = x1
            elif target == "x2":
                value = x2
            elif target == "y1":
                value = y1
            elif target == "y2":
                value = y2
            elif target == "confidence":
                value = seg.confidence
            else:
                raise Exception(f"[Impact Pack] SEGSOrderedFilter - Unexpected target '{target}'")

            segs_with_order.append((value, seg))

        if order:
            sorted_list = sorted(segs_with_order, key=lambda x: x[0], reverse=True)
        else:
            sorted_list = sorted(segs_with_order, key=lambda x: x[0], reverse=False)

        result_list = []
        remained_list = []

        for i, item in enumerate(sorted_list):
            if take_start <= i < take_start + take_count:
                result_list.append(item[1])
            else:
                remained_list.append(item[1])

        return (segs[0], result_list), (segs[0], remained_list),

```
