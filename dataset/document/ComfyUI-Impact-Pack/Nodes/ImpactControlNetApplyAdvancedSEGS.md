---
tags:
- ControlNet
---

# ControlNetApplyAdvanced (SEGS)
## Documentation
- Class name: `ImpactControlNetApplyAdvancedSEGS`
- Category: `ImpactPack/Util`
- Output node: `False`

This node is designed to apply advanced segmentation (SEGS) adjustments using a control network. It enhances or modifies the segmentation data based on the control network's parameters, offering a way to fine-tune or dynamically adjust the segmentation output for more precise or varied results.
## Input types
### Required
- **`segs`**
    - The segmentation data to be processed. It serves as the primary input for the node, determining the base segmentation that will be adjusted.
    - Comfy dtype: `SEGS`
    - Python dtype: `Tuple[Size, List[SEG]]`
- **`control_net`**
    - The control network used to adjust the segmentation. It defines the transformation or adjustment logic applied to the segmentation data.
    - Comfy dtype: `CONTROL_NET`
    - Python dtype: `ControlNet`
- **`strength`**
    - A scalar value that determines the intensity of the control network's effect on the segmentation. Higher values result in more pronounced adjustments.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`start_percent`**
    - The starting percentage of the effect's application range, allowing for gradual application of the control network's adjustments.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`end_percent`**
    - The ending percentage of the effect's application range, enabling fine-tuning of where the adjustments taper off.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`segs_preprocessor`**
    - An optional preprocessor for the segmentation data before applying the control network adjustments. It can be used for initial modifications or preparations of the SEGS.
    - Comfy dtype: `SEGS_PREPROCESSOR`
    - Python dtype: `SEGS_PREPROCESSOR`
- **`control_image`**
    - An optional image that can influence the control network's adjustments, providing additional context or guidance for the transformations.
    - Comfy dtype: `IMAGE`
    - Python dtype: `IMAGE`
## Output types
- **`segs`**
    - Comfy dtype: `SEGS`
    - The adjusted segmentation data after applying the control network's enhancements or modifications.
    - Python dtype: `Tuple[Size, List[SEG]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ControlNetApplyAdvancedSEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "segs": ("SEGS",),
                    "control_net": ("CONTROL_NET",),
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                    "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                    "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
                    },
                "optional": {
                    "segs_preprocessor": ("SEGS_PREPROCESSOR",),
                    "control_image": ("IMAGE",)
                    }
                }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    @staticmethod
    def doit(segs, control_net, strength, start_percent, end_percent, segs_preprocessor=None, control_image=None):
        new_segs = []

        for seg in segs[1]:
            control_net_wrapper = core.ControlNetAdvancedWrapper(control_net, strength, start_percent, end_percent, segs_preprocessor,
                                                                 seg.control_net_wrapper, original_size=segs[0], crop_region=seg.crop_region,
                                                                 control_image=control_image)
            new_seg = SEG(seg.cropped_image, seg.cropped_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, control_net_wrapper)
            new_segs.append(new_seg)

        return ((segs[0], new_segs), )

```
