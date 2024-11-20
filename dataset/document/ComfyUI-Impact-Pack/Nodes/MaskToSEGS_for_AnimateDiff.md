---
tags:
- SEGSPrep
- Segmentation
---

# MASK to SEGS for AnimateDiff
## Documentation
- Class name: `MaskToSEGS_for_AnimateDiff`
- Category: `ImpactPack/Operation`
- Output node: `False`

This node is designed to convert masks into SEGS format specifically tailored for animations with differential impact, accommodating both 2D and 3D masks. It leverages contour filling options and advanced mask processing techniques to ensure the generated SEGS are optimized for animation sequences, handling batch masks with specialized transformations when necessary.
## Input types
### Required
- **`mask`**
    - The input mask to be converted into SEGS. This mask can be either 2D or 3D, and it is the primary input for generating SEGS tailored for animations.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`combined`**
    - A boolean flag indicating whether the mask and the resulting SEGS should be combined, affecting the processing logic of the node.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`crop_factor`**
    - A factor determining the extent of cropping applied to the mask before conversion, influencing the size and detail of the resulting SEGS.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`bbox_fill`**
    - A boolean flag that determines whether bounding boxes should be filled, impacting the visual representation of the resulting SEGS.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`drop_size`**
    - Specifies the minimum size of features to retain in the SEGS, filtering out smaller elements based on this threshold.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`contour_fill`**
    - A boolean flag indicating whether contours within the mask should be filled, enhancing the detail and completeness of the resulting SEGS.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`segs`**
    - Comfy dtype: `SEGS`
    - The output SEGS generated from the input mask, optimized for animations with differential impact. These SEGS are suitable for further processing or visualization in animation workflows.
    - Python dtype: `Tuple[torch.Tensor]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class MaskToSEGS_for_AnimateDiff:
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
        if (len(mask.shape) == 4 and mask.shape[1] > 1) or (len(mask.shape) == 3 and mask.shape[0] > 1):
            mask = make_3d_mask(mask)
            if contour_fill:
                print(f"[Impact Pack] MaskToSEGS_for_AnimateDiff: 'contour_fill' is ignored because batch mask 'contour_fill' is not supported.")
            result = core.batch_mask_to_segs(mask, combined, crop_factor, bbox_fill, drop_size)
            return (result, )

        mask = make_2d_mask(mask)
        segs = core.mask_to_segs(mask, combined, crop_factor, bbox_fill, drop_size, is_contour=contour_fill)
        all_masks = SEGSToMaskList().doit(segs)[0]

        result_mask = (all_masks[0] * 255).to(torch.uint8)
        for mask in all_masks[1:]:
            result_mask |= (mask * 255).to(torch.uint8)

        result_mask = (result_mask/255.0).to(torch.float32)
        result_mask = utils.to_binary_mask(result_mask, 0.1)[0]

        return MaskToSEGS.doit(result_mask, False, crop_factor, False, drop_size, contour_fill)

```
