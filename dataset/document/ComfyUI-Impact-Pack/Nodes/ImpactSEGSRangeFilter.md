---
tags:
- ImpactPack
- Segmentation
---

# SEGS Filter (range)
## Documentation
- Class name: `ImpactSEGSRangeFilter`
- Category: `ImpactPack/Util`
- Output node: `False`

The ImpactSEGSRangeFilter node is designed to filter segmentation elements (SEGS) based on specified criteria such as area, width, height, coordinates, or confidence level. It allows for the inclusion or exclusion of segments within a given range, enhancing the ability to focus on segments of interest within an image.
## Input types
### Required
- **`segs`**
    - The segmentation elements to be filtered. This is the primary data upon which the filtering operations are performed.
    - Comfy dtype: `SEGS`
    - Python dtype: `tuple`
- **`target`**
    - Specifies the attribute of the segmentation elements to be filtered, such as area, dimensions, or confidence level. This determines the basis on which segments are evaluated and filtered.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`mode`**
    - Determines the filtering mode, either including segments within the specified range ('inside') or excluding them ('outside'). This affects how segments are selected based on the target attribute's value.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `str`
- **`min_value`**
    - The minimum value of the target attribute for a segment to be considered for inclusion or exclusion, depending on the mode.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`max_value`**
    - The maximum value of the target attribute for a segment to be considered for inclusion or exclusion, depending on the mode.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`filtered_SEGS`**
    - Comfy dtype: `SEGS`
    - The segments that meet the filtering criteria based on the specified target and range.
    - Python dtype: `tuple`
- **`remained_SEGS`**
    - Comfy dtype: `SEGS`
    - The segments that do not meet the filtering criteria and are thus excluded based on the specified target and range.
    - Python dtype: `tuple`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SEGSRangeFilter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "segs": ("SEGS", ),
                        "target": (["area(=w*h)", "width", "height", "x1", "y1", "x2", "y2", "length_percent", "confidence(0-100)"],),
                        "mode": ("BOOLEAN", {"default": True, "label_on": "inside", "label_off": "outside"}),
                        "min_value": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                        "max_value": ("INT", {"default": 67108864, "min": 0, "max": sys.maxsize, "step": 1}),
                     },
                }

    RETURN_TYPES = ("SEGS", "SEGS",)
    RETURN_NAMES = ("filtered_SEGS", "remained_SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, segs, target, mode, min_value, max_value):
        new_segs = []
        remained_segs = []

        for seg in segs[1]:
            x1 = seg.crop_region[0]
            y1 = seg.crop_region[1]
            x2 = seg.crop_region[2]
            y2 = seg.crop_region[3]

            if target == "area(=w*h)":
                value = (y2 - y1) * (x2 - x1)
            elif target == "length_percent":
                h = y2 - y1
                w = x2 - x1
                value = max(h/w, w/h)*100
                print(f"value={value}")
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
            elif target == "confidence(0-100)":
                value = seg.confidence*100
            else:
                raise Exception(f"[Impact Pack] SEGSRangeFilter - Unexpected target '{target}'")

            if mode and min_value <= value <= max_value:
                print(f"[in] value={value} / {mode}, {min_value}, {max_value}")
                new_segs.append(seg)
            elif not mode and (value < min_value or value > max_value):
                print(f"[out] value={value} / {mode}, {min_value}, {max_value}")
                new_segs.append(seg)
            else:
                remained_segs.append(seg)
                print(f"[filter] value={value} / {mode}, {min_value}, {max_value}")

        return (segs[0], new_segs), (segs[0], remained_segs),

```
