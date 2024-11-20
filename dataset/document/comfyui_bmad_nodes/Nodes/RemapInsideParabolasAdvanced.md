---
tags:
- ImageTransformation
---

# RemapInsideParabolasAdvanced
## Documentation
- Class name: `RemapInsideParabolasAdvanced`
- Category: `Bmad/CV/Transform`
- Output node: `False`

This node specializes in adjusting the mapping of images based on the geometry of two parabolas within a given mask. It allows for fine-tuning the remapping process through curve and orthogonal adjustments, as well as the option to flip the orthogonal direction, enhancing flexibility in image processing tasks that require precise spatial transformations.
## Input types
### Required
- **`dst_mask_with_i_parabolas`**
    - unknown
    - Comfy dtype: `MASK`
    - Python dtype: `unknown`
- **`curve_wise_adjust`**
    - Adjusts the curvature of the remapping process, allowing for finer control over the spatial transformation along the curve defined by the parabolas.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`ortho_wise_adjust`**
    - Adjusts the orthogonal transformation relative to the parabolas' curve, enabling precise control over the image's spatial deformation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`flip_ortho`**
    - Determines whether the orthogonal direction of the remapping should be flipped, offering an additional degree of transformation customization.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`remap`**
    - Comfy dtype: `REMAP`
    - The output is a transformed image that has been remapped according to the specified parabolas and adjustments. This includes the spatially transformed image data, which is the result of applying the curve and orthogonal adjustments, as well as the optional flipping of the orthogonal direction.
    - Python dtype: `tuple`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RemapInsideParabolasAdvanced(RemapBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "dst_mask_with_2_parabolas": ("MASK",),
            "curve_wise_adjust": ("FLOAT", {"default": 1, "min": .3, "max": 2, "step": .01}),
            "ortho_wise_adjust": ("FLOAT", {"default": 1, "min": 1, "max": 3, "step": .01}),
            "flip_ortho": ("BOOLEAN", {"default": False})
        }
        }

    def send_remap(self, dst_mask_with_2_parabolas, curve_wise_adjust, ortho_wise_adjust, flip_ortho):
        from .utils.remaps import remap_inside_parabolas_advanced
        return ({
                    "func": remap_inside_parabolas_advanced,
                    "xargs": [tensor2opencv(dst_mask_with_2_parabolas, 1),
                              curve_wise_adjust, ortho_wise_adjust, flip_ortho],
                    "dims": RemapBase.get_dims(dst_mask_with_2_parabolas)
                },)

```
