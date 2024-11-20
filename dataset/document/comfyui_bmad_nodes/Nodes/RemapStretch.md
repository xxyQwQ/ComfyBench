---
tags:
- ImageTransformation
---

# RemapStretch
## Documentation
- Class name: `RemapStretch`
- Category: `Bmad/CV/Transform`
- Output node: `False`

The RemapStretch node is designed to apply a stretching transformation to images based on specified power factors along the x and y axes, and a center point around which the stretch is applied. This node allows for the dynamic adjustment of image geometry to simulate effects such as stretching or elongation, providing a versatile tool for image manipulation and enhancement.
## Input types
### Required
- **`power_x`**
    - Specifies the power factor for stretching along the x-axis. A higher value results in more pronounced stretching effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`power_y`**
    - Specifies the power factor for stretching along the y-axis. Similar to `power_x`, a higher value intensifies the stretching effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`center_x`**
    - Determines the x-coordinate of the center point around which the stretching effect is applied. This allows for targeted manipulation of the image geometry.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`center_y`**
    - Determines the y-coordinate of the center point around which the stretching effect is applied, enabling precise control over the area of effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`remap`**
    - Comfy dtype: `REMAP`
    - Produces a transformed image where the specified stretching effects have been applied, altering the original geometry according to the input parameters.
    - Python dtype: `Tuple[np.ndarray, np.ndarray, NoneType]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RemapStretch(RemapBase):
    @classmethod
    def INPUT_TYPES(cls):
        return RemapPinch.INPUT_TYPES_DICT

    def send_remap(self, power_x, power_y, center_x, center_y):
        from .utils.remaps import remap_pinch_or_stretch
        return ({
                    "func": remap_pinch_or_stretch,
                    "xargs": [(1 / power_x, 1 / power_y), (center_x, center_y)]
                },)

```
