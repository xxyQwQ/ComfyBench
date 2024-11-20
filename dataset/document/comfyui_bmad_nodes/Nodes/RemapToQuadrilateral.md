---
tags:
- ImageTransformation
---

# RemapToQuadrilateral
## Documentation
- Class name: `RemapToQuadrilateral`
- Category: `Bmad/CV/Transform`
- Output node: `False`

This node is designed to transform images by remapping them to fit within quadrilateral shapes, utilizing various methods including homography and edge pair interpolation. It abstracts complex geometric transformations, making it easier to apply perspective adjustments or corrections based on specified quadrilateral regions within images.
## Input types
### Required
- **`dst_mask_with_i_points`**
    - Specifies the destination mask with i points defining the quadrilateral to which the image will be remapped. This mask is crucial for determining the target shape and area for the remapping process.
    - Comfy dtype: `MASK`
    - Python dtype: `ndarray`
- **`mode`**
    - Defines the method to be used for the remapping process, such as homography or length-wise interpolation, allowing for flexibility in how the image is transformed to fit the quadrilateral.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`remap`**
    - Comfy dtype: `REMAP`
    - The result of the remapping process, which includes the transformed image fitted within the specified quadrilateral shape.
    - Python dtype: `tuple`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class RemapQuadrilateral(RemapBase):
    from .utils.remaps import quad_remap_methods_map

    modes_list = list(quad_remap_methods_map.keys())

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "dst_mask_with_4_points": ("MASK",),
            "mode": (cls.modes_list, {"default": cls.modes_list[0]}),
        }
        }

    @staticmethod
    def homography(custom_data, src, interpolation, mask=None):
        h_matrix, bb = custom_data
        bb_width, bb_height = bb[2] - bb[0], bb[3] - bb[1]
        ret = cv.warpPerspective(src, h_matrix, (bb_width, bb_height), flags=interpolation,
                                 borderMode=cv.BORDER_CONSTANT)
        if mask is not None:
            mask = cv.warpPerspective(mask, h_matrix, (bb_width, bb_height), flags=interpolation,
                                      borderMode=cv.BORDER_CONSTANT)
        return ret, mask, bb

    def send_remap(self, dst_mask_with_4_points, mode):
        from .utils.remaps import remap_quadrilateral
        remap_data = {
            "func": remap_quadrilateral,
            "xargs": [tensor2opencv(dst_mask_with_4_points, 1), mode],
            "dims": RemapBase.get_dims(dst_mask_with_4_points)
        }
        if mode == "HOMOGRAPHY":
            remap_data["custom"] = RemapQuadrilateral.homography
        return (remap_data,)

```
