---
tags:
- ImageTransformation
---

# RemapInsideParabolas
## Documentation
- Class name: `RemapInsideParabolas`
- Category: `Bmad/CV/Transform`
- Output node: `False`

This node is designed for remapping images based on the geometry of two parabolas within a given mask. It adjusts the image's pixels to align or distort according to these parabolas, enabling advanced image transformation techniques that can simulate or correct specific optical effects.
## Input types
### Required
- **`dst_mask_with_i_parabolas`**
    - A mask image containing two parabolas. This mask is used to determine the transformation geometry for remapping the image.
    - Comfy dtype: `MASK`
    - Python dtype: `numpy.ndarray`
## Output types
- **`remap`**
    - Comfy dtype: `REMAP`
    - The output is a transformed version of the input image, where the pixel positions have been adjusted based on the specified parabolas within the mask.
    - Python dtype: `Dict[str, Any]`
- **`ui`**
    - The output is a transformed version of the input image, where the pixel positions have been adjusted based on the specified parabolas within the mask.
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RemapInsideParabolas(RemapBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "dst_mask_with_2_parabolas": ("MASK",),
        }
        }

    def send_remap(self, dst_mask_with_2_parabolas):
        from .utils.remaps import remap_inside_parabolas_simple
        return ({
                    "func": remap_inside_parabolas_simple,
                    "xargs": [tensor2opencv(dst_mask_with_2_parabolas, 1)],
                    "dims": RemapBase.get_dims(dst_mask_with_2_parabolas)
                },)

```
