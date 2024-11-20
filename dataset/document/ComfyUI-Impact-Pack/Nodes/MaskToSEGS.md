---
tags:
- SEGSPrep
- Segmentation
---

# MASK to SEGS
## Documentation
- Class name: `MaskToSEGS`
- Category: `ImpactPack/Operation`
- Output node: `False`

The MaskToSEGS node is designed to transform a given mask into a structured edge graphic signal (SEGS) format, optionally incorporating various preprocessing steps such as combining, cropping, bounding box filling, size filtering, and contour filling. This node is part of the ImpactPack's operation category, facilitating the manipulation and analysis of mask data for further processing or visualization.
## Input types
### Required
- **`mask`**
    - The mask input is the primary data that the node transforms into the SEGS format. It represents the area of interest within an image for further processing or analysis.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`combined`**
    - Indicates whether to combine the input mask with other processing steps or keep it separate, affecting the final SEGS output.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`crop_factor`**
    - Determines the factor by which the mask is cropped, influencing the size and focus area of the resulting SEGS.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`bbox_fill`**
    - Controls whether bounding boxes are filled, impacting the visual representation of the SEGS.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`drop_size`**
    - Specifies the minimum size of areas to be included in the SEGS, filtering out smaller regions.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`contour_fill`**
    - Determines whether contours within the mask are filled, affecting the detail level of the SEGS.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`segs`**
    - Comfy dtype: `SEGS`
    - unknown
    - Python dtype: `unknown`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [DetailerForEachDebugPipe](../../ComfyUI-Impact-Pack/Nodes/DetailerForEachDebugPipe.md)
    - [DetailerForEachDebug](../../ComfyUI-Impact-Pack/Nodes/DetailerForEachDebug.md)



## Source code
```python
class MaskToSEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "mask": ("MASK",),
                                "combined": ("BOOLEAN", {"default": False, "label_on": "True", "label_off": "False"}),
                                "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 100, "step": 0.1}),
                                "bbox_fill": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                                "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
                                "contour_fill": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                             }
                }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Operation"

    @staticmethod
    def doit(mask, combined, crop_factor, bbox_fill, drop_size, contour_fill=False):
        mask = make_2d_mask(mask)
        result = core.mask_to_segs(mask, combined, crop_factor, bbox_fill, drop_size, is_contour=contour_fill)

        return (result, )

```
