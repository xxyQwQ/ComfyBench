---
tags:
- ImageTransformation
---

# RemapToInnerCylinder
## Documentation
- Class name: `RemapToInnerCylinder`
- Category: `Bmad/CV/Transform`
- Output node: `False`

The RemapToInnerCylinder node is designed to transform an image by mapping it onto an inner cylinder, adjusting the field of view and optionally swapping the x and y coordinates based on the provided parameters. This node is part of a suite of remapping functionalities that manipulate images to simulate various perspectives and distortions.
## Input types
### Required
- **`fov`**
    - The 'fov' parameter specifies the field of view in degrees, controlling the extent of the cylindrical projection. It determines how wide the view is on the resulting cylinder, affecting the distortion level of the remapped image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`swap_xy`**
    - The 'swap_xy' parameter indicates whether the x and y coordinates should be swapped in the remapping process. This can be used to change the orientation of the image on the cylindrical surface.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`remap`**
    - Comfy dtype: `REMAP`
    - The output 'remap' provides the transformed image data after applying the inner cylinder mapping. It encapsulates the result of the remapping process, reflecting the adjustments made to the field of view and the orientation of the image.
    - Python dtype: `Dict[str, Any]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class InnerCylinderRemap(RemapBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "fov": ("INT", {"default": 90, "min": 1, "max": 179}),
            "swap_xy": ("BOOLEAN", {"default": False}),
        }
        }

    def send_remap(self, fov, swap_xy):
        from .utils.remaps import remap_inner_cylinder
        return ({
                    "func": remap_inner_cylinder,
                    "xargs": [fov, swap_xy]
                },)

```
