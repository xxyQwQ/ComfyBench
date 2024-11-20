---
tags:
- Color
---

# Color (RGB)
## Documentation
- Class name: `Color (RGB)`
- Category: `Bmad/image`
- Output node: `False`

This node is designed for creating a color representation from individual red, green, and blue components. It allows for the precise specification of colors in RGB format, facilitating their use in various image processing and visualization tasks.
## Input types
### Required
- **`r`**
    - Specifies the red component of the color, contributing to the overall hue when combined with green and blue components.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`g`**
    - Specifies the green component of the color, contributing to the overall hue when combined with red and blue components.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`b`**
    - Specifies the blue component of the color, contributing to the overall hue when combined with red and green components.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`color`**
    - Comfy dtype: `COLOR`
    - The RGB representation of the input components, allowing for its use in various image processing and visualization contexts.
    - Python dtype: `list`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ColorRGB:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"r": color255_INPUT, "g": color255_INPUT, "b": color255_INPUT}}

    RETURN_TYPES = ("COLOR",)
    FUNCTION = "ret"
    CATEGORY = images_category_path

    def ret(self, r, g, b):
        return ([r, g, b],)

```
