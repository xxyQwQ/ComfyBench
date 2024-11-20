---
tags:
- Color
---

# ColorDictionary
## Documentation
- Class name: `ColorDictionary`
- Category: `Bmad/CV/Color A.`
- Output node: `False`

The ColorDictionary node provides a predefined dictionary of color names to RGB values, allowing users to retrieve a subset of these colors based on a specified number. It simplifies the process of accessing common colors for various applications by offering a default set of colors and the ability to specify how many colors are needed.
## Input types
### Required
- **`number_of_colors`**
    - Specifies the number of colors to retrieve from the predefined color dictionary. It determines the size of the returned dictionary, allowing for flexible customization of the color selection.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`color_dict`**
    - Comfy dtype: `COLOR_DICT`
    - Returns a dictionary mapping color names to their corresponding RGB values, limited to the specified number of colors.
    - Python dtype: `Dict[str, Tuple[int, int, int]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ColorDefaultDictionary:
    default_color_dict = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "yellow": (255, 255, 0),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "purple": (128, 0, 128),
        "teal": (0, 128, 128),
        "orange": (255, 165, 0),
        "pink": (255, 192, 203),
        #    "brown": (165, 42, 42),
        #    "gray": (128, 128, 128),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"number_of_colors": ("INT", {"default": 8, "min": 2, "max": 12})}}

    RETURN_TYPES = ("COLOR_DICT",)
    FUNCTION = "ret"
    CATEGORY = f"{cv_category_path}/Color A."

    def ret(self, number_of_colors):
        dic = dict(list(self.default_color_dict.items())[0: number_of_colors])
        return (dic,)

```
