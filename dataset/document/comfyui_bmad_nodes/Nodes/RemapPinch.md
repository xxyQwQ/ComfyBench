---
tags:
- ImageTransformation
---

# RemapPinch
## Documentation
- Class name: `RemapPinch`
- Category: `Bmad/CV/Transform`
- Output node: `False`

The RemapPinch node is designed to apply a pinch distortion effect to images, allowing for the adjustment of the distortion's power and center through specified parameters. It leverages a mathematical model to modify the image geometry, creating visually compelling effects that can enhance or alter the appearance of images in creative ways.
## Input types
### Required
- **`power_x`**
    - Specifies the horizontal distortion power. Higher values increase the pinch effect horizontally, affecting the image's geometry. This parameter is crucial for controlling the intensity of the distortion effect, directly influencing the degree to which the image is pinched horizontally.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`power_y`**
    - Determines the vertical distortion power. Similar to power_x, but applies vertically, altering the image's vertical geometry. This parameter is essential for adjusting the vertical intensity of the distortion, impacting how much the image is pinched vertically.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`center_x`**
    - Sets the horizontal center point of the pinch effect. Values range from 0 to 1, representing the relative position across the image width. Adjusting this parameter shifts the focal point of the distortion horizontally, affecting where the pinch effect is most pronounced.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`center_y`**
    - Defines the vertical center point of the pinch effect. Values range from 0 to 1, indicating the position from top to bottom of the image. Modifying this parameter moves the distortion's focal point vertically, altering the central area of the pinch effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`remap`**
    - Comfy dtype: `REMAP`
    - The output is a remapped version of the input image, where the geometry has been altered based on the specified distortion parameters. This remapping results in a visually distorted image that exhibits the pinch effect, with the degree and center of distortion controlled by the input parameters.
    - Python dtype: `Tuple[np.ndarray, np.ndarray, NoneType]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RemapPinch(RemapBase):
    INPUT_TYPES_DICT = {
        "required": {
            "power_x": ("FLOAT", {"default": 1, "min": 1, "max": 3, "step": .05}),
            "power_y": ("FLOAT", {"default": 1, "min": 1, "max": 3, "step": .05}),
            "center_x": ("FLOAT", {"default": .5, "min": 0, "max": 1, "step": .05}),
            "center_y": ("FLOAT", {"default": .5, "min": 0, "max": 1, "step": .05}),
        }
    }

    @classmethod
    def INPUT_TYPES(cls):
        return RemapPinch.INPUT_TYPES_DICT

    def send_remap(self, power_x, power_y, center_x, center_y):
        from .utils.remaps import remap_pinch_or_stretch
        return ({
                    "func": remap_pinch_or_stretch,
                    "xargs": [(power_x, power_y), (center_x, center_y)]
                },)

```
