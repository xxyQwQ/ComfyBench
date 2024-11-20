---
tags:
- ImageTransformation
---

# RemapFromQuadrilateral (homography)
## Documentation
- Class name: `RemapFromQuadrilateral (homography)`
- Category: `Bmad/CV/Transform`
- Output node: `False`

The RemapFromQuadrilateral node is designed for transforming images based on a source mask that defines a quadrilateral region. It primarily utilizes a homography transformation to map the specified quadrilateral region from the source image to a target rectangular shape, adjusting the image accordingly. This node is particularly useful for correcting or altering the perspective of an image region defined by four points.
## Input types
### Required
- **`src_mask_with_i_points`**
    - Specifies the source mask with four points defining the quadrilateral region to be transformed. This mask is crucial for determining the area of the image that will undergo the homography transformation.
    - Comfy dtype: `MASK`
    - Python dtype: `ndarray`
- **`width`**
    - Defines the width of the target image after the transformation. This parameter directly influences the dimensions of the output image, ensuring that the transformed quadrilateral fits within these specified bounds.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Specifies the height of the target image post-transformation. Similar to the width parameter, it determines the dimensions of the output image, ensuring the transformed area is appropriately scaled.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`remap`**
    - Comfy dtype: `REMAP`
    - The primary output of this node is the transformed image or 'remap', which results from applying the homography transformation to the specified quadrilateral region. This output is crucial for understanding the effect of the transformation on the original image.
    - Python dtype: `ndarray`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class RemapFromQuadrilateral(RemapBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "src_mask_with_4_points": ("MASK",),
            # "mode": (s.modes_list, {"default": s.modes_list[0]}),
            "width": ("INT", {"default": 512, "min": 16, "max": 4096}),
            "height": ("INT", {"default": 512, "min": 16, "max": 4096}),
        }
        }

    @staticmethod
    def homography(*args):
        ret, mask, bb = RemapQuadrilateral.homography(*args)
        return ret, mask, None

    def send_remap(self, src_mask_with_4_points, width, height):
        from .utils.remaps import remap_from_quadrilateral
        remap_data = {
            "func": remap_from_quadrilateral,
            "xargs": [tensor2opencv(src_mask_with_4_points, 1), width, height],
            "dims": (width, height),  # seems kinda redundant, not sure if should refactor
            "custom": RemapFromQuadrilateral.homography
        }
        return (remap_data,)

```
