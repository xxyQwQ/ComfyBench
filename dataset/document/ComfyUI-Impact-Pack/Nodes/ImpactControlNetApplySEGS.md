---
tags:
- ControlNet
---

# ControlNetApply (SEGS)
## Documentation
- Class name: `ImpactControlNetApplySEGS`
- Category: `ImpactPack/Util`
- Output node: `False`

This node applies a control network to segmentation elements (SEGS), modifying them based on the control network's parameters and potentially an additional control image. It's designed to adjust or enhance specific aspects of the segmentation elements, such as their appearance or structure, by leveraging the control network's capabilities.
## Input types
### Required
- **`segs`**
    - The segmentation elements to be modified. This is the primary input that the control network will be applied to.
    - Comfy dtype: `SEGS`
    - Python dtype: `Tuple[Size, List[SEG]]`
- **`control_net`**
    - The control network to be applied to each segmentation element. It defines the modifications or adjustments to be made.
    - Comfy dtype: `CONTROL_NET`
    - Python dtype: `CONTROL_NET`
- **`strength`**
    - A scalar value that determines the intensity of the control network's effect on the segmentation elements.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`segs_preprocessor`**
    - An optional preprocessor for the segmentation elements before applying the control network.
    - Comfy dtype: `SEGS_PREPROCESSOR`
    - Python dtype: `SEGS_PREPROCESSOR`
- **`control_image`**
    - An optional image that can provide additional context or guidance for the control network's application.
    - Comfy dtype: `IMAGE`
    - Python dtype: `IMAGE`
## Output types
- **`segs`**
    - Comfy dtype: `SEGS`
    - The modified segmentation elements after the control network has been applied.
    - Python dtype: `Tuple[Size, List[SEG]]`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [DetailerForEachDebugPipe](../../ComfyUI-Impact-Pack/Nodes/DetailerForEachDebugPipe.md)
    - [ImpactControlNetApplySEGS](../../ComfyUI-Impact-Pack/Nodes/ImpactControlNetApplySEGS.md)



## Source code
```python
class ControlNetApplySEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "segs": ("SEGS",),
                    "control_net": ("CONTROL_NET",),
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
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
    def doit(segs, control_net, strength, segs_preprocessor=None, control_image=None):
        new_segs = []

        for seg in segs[1]:
            control_net_wrapper = core.ControlNetWrapper(control_net, strength, segs_preprocessor, seg.control_net_wrapper,
                                                         original_size=segs[0], crop_region=seg.crop_region, control_image=control_image)
            new_seg = SEG(seg.cropped_image, seg.cropped_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, control_net_wrapper)
            new_segs.append(new_seg)

        return ((segs[0], new_segs), )

```
