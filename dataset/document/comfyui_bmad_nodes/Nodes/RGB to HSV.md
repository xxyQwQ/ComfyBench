---
tags:
- Color
- HSVColorSpace
---

# RGB to HSV
## Documentation
- Class name: `RGB to HSV`
- Category: `Bmad/CV/Color A.`
- Output node: `False`

The node converts RGB color values to HSV color space, facilitating color transformations and analysis by separating color hue, saturation, and value. This conversion is essential for tasks that require color space manipulation or analysis, such as image processing and computer vision applications.
## Input types
### Required
- **`rgb_color`**
    - The RGB color input that needs to be converted to HSV. This parameter is crucial for the conversion process, as it represents the color in RGB format that will be transformed into its HSV equivalent.
    - Comfy dtype: `COLOR`
    - Python dtype: `Tuple[int, int, int]`
## Output types
- **`hsv_color`**
    - Comfy dtype: `HSV_COLOR`
    - The output HSV color, represented as a tuple of hue, saturation, and value. This conversion is useful for applications that rely on the HSV color space for more intuitive handling of colors.
    - Python dtype: `Tuple[int, int, int]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ColorToHSVColor:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "rgb_color": ("COLOR",)
        }}

    RETURN_TYPES = ("HSV_COLOR",)
    FUNCTION = "convert"
    CATEGORY = f"{cv_category_path}/Color A."

    def convert(self, rgb_color):
        from colorsys import rgb_to_hsv
        rgb_color = setup_color_to_correct_type(rgb_color)
        (r, g, b) = tuple(rgb_color)
        rgb_color = (r / 255, g / 255, b / 255)
        (h, s, v) = rgb_to_hsv(*rgb_color)
        hsv = (int(h * 179), int(s * 255), int(v * 255))
        return (hsv,)

```
