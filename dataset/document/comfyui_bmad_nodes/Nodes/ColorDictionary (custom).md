---
tags:
- Color
---

# ColorDictionary (custom)
## Documentation
- Class name: `ColorDictionary (custom)`
- Category: `Bmad/CV/Color A.`
- Output node: `False`

This node allows for the creation of a custom color dictionary by mapping user-defined color names to their corresponding color values. It ensures that the number of color names matches the number of color values, adjusting the lists to match in length if necessary, before returning a dictionary that associates each color name with its color value.
## Input types
### Required
- **`color_names`**
    - A list of user-defined names for colors. These names are used as keys in the resulting color dictionary, allowing for easy identification and retrieval of color values.
    - Comfy dtype: `STRING`
    - Python dtype: `List[str]`
- **`colors`**
    - A list of color values (typically RGB tuples) that correspond to the user-defined color names. These values are used as the dictionary values in the resulting color dictionary, mapping each name to its respective color.
    - Comfy dtype: `COLOR`
    - Python dtype: `List[Tuple[int, int, int]]`
## Output types
- **`color_dict`**
    - Comfy dtype: `COLOR_DICT`
    - The resulting custom color dictionary, mapping each user-defined color name to its corresponding color value.
    - Python dtype: `Dict[str, Tuple[int, int, int]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ColorCustomDictionary:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "color_names": ("STRING", {"default": ""}),
            "colors": ("COLOR", {"default": ""})
        }
        }

    RETURN_TYPES = ("COLOR_DICT",)
    FUNCTION = "ret"
    CATEGORY = f"{cv_category_path}/Color A."
    INPUT_IS_LIST = True

    def ret(self, color_names, colors):
        if len(color_names) != len(colors):
            print_yellow("color_names size is different than colors size!")
            min_len = min(len(color_names), len(colors))
            color_names = color_names[0:min_len]
            colors = colors[0:min_len]

        return (dict(zip(color_names, colors)),)

```
