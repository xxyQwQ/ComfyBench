---
tags:
- ImageTransformation
---

# RemapFromInsideParabolas
## Documentation
- Class name: `RemapFromInsideParabolas`
- Category: `Bmad/CV/Transform`
- Output node: `False`

This node is designed to perform a remapping operation from the perspective of inside two parabolas, transforming an image based on specified width and height parameters. It utilizes a source mask defined by two parabolas to apply a unique geometric transformation, aiming to adjust the image's representation for specific visualization or processing needs.
## Input types
### Required
- **`src_mask_with_i_parabolas`**
    - Defines the source mask that contains two parabolas, which is crucial for determining the geometric transformation applied to the image.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`width`**
    - Specifies the desired width of the output image after remapping, affecting the scale and aspect ratio of the transformation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Determines the height of the output image, influencing the vertical scale and aspect ratio post-transformation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`remap`**
    - Comfy dtype: `REMAP`
    - The result of the remapping operation, providing a transformed image based on the input parameters and the geometric characteristics of the source mask.
    - Python dtype: `Dict[str, Any]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RemapFromInsideParabolas(RemapBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "src_mask_with_2_parabolas": ("MASK",),
            "width": ("INT", {"default": 512, "min": 16, "max": 4096}),
            "height": ("INT", {"default": 512, "min": 16, "max": 4096}),
        }
        }

    def send_remap(self, src_mask_with_2_parabolas, width, height):
        from .utils.remaps import remap_from_inside_parabolas
        return ({
                    "func": remap_from_inside_parabolas,
                    "xargs": [tensor2opencv(src_mask_with_2_parabolas, 1), width, height],
                    "dims": (width, height)
                },)

```
