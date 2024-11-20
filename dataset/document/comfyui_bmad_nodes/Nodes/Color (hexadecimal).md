---
tags:
- Color
---

# Color (hexadecimal)
## Documentation
- Class name: `Color (hexadecimal)`
- Category: `Bmad/image`
- Output node: `False`

This node is designed to convert hexadecimal color codes into a format that can be utilized within the system, ensuring compatibility and proper representation of colors as defined by their hex codes. It validates the hex code format and converts it to a recognized color format for further processing or display.
## Input types
### Required
- **`hex`**
    - The hexadecimal representation of a color, starting with a '#' followed by either 3 or 6 hexadecimal characters. This input is crucial for defining the specific color to be converted and validated.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`color`**
    - Comfy dtype: `COLOR`
    - The validated and possibly transformed color information, maintaining the integrity of the original hexadecimal input for use in color-related operations.
    - Python dtype: `tuple`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ColorRGBFromHex:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"hex": ("STRING", {"default": "#000000"})}}

    RETURN_TYPES = ("COLOR",)
    FUNCTION = "ret"
    CATEGORY = images_category_path

    def ret(self, hex):
        import re
        hex_color_pattern = r'^#([0-9A-Fa-f]{3}|[0-9A-Fa-f]{6})$'
        if re.match(hex_color_pattern, hex) is None:
            print_yellow(f"ColorRGBFromHex node > The following is not a valid hex code:{hex}")
        return (hex,)

```
