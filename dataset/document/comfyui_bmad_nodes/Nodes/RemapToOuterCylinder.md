---
tags:
- ImageTransformation
---

# RemapToOuterCylinder
## Documentation
- Class name: `RemapToOuterCylinder`
- Category: `Bmad/CV/Transform`
- Output node: `False`

The RemapToOuterCylinder node is designed to transform an image by mapping it onto the surface of an imaginary outer cylinder. This transformation is based on the field of view and an optional swap of the x and y axes, aiming to simulate a perspective where the image wraps around the viewer in a cylindrical manner.
## Input types
### Required
- **`fov`**
    - The field of view in degrees, determining the extent of the cylindrical projection. It influences the curvature and the scale of the remapped image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`swap_xy`**
    - A boolean flag that, when true, swaps the x and y axes in the output mapping. This can be used to adjust the orientation of the remapped image based on specific requirements.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`remap`**
    - Comfy dtype: `REMAP`
    - The output of the RemapToOuterCylinder node is a transformed image that has been remapped to simulate being wrapped around an outer cylinder. This output is crucial for applications requiring cylindrical perspective transformations.
    - Python dtype: `ndarray`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class OuterCylinderRemap(RemapBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "fov": ("INT", {"default": 90, "min": 1, "max": 179}),
            "swap_xy": ("BOOLEAN", {"default": False}),
        }
        }

    def send_remap(self, fov, swap_xy):
        from .utils.remaps import remap_outer_cylinder
        return ({
                    "func": remap_outer_cylinder,
                    "xargs": [fov, swap_xy]
                },)

```
