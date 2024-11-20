# LayerUtility: RGB Value
## Documentation
- Class name: `LayerUtility: RGB Value`
- Category: `😺dzNodes/LayerUtility/Data`
- Output node: `False`

This node converts a given color value, either in hexadecimal string format or as an RGB tuple, into its constituent red, green, and blue (RGB) integer components. It's designed to facilitate color manipulation and analysis within image processing tasks by providing a direct way to extract and work with individual color channels.
## Input types
### Required
- **`color_value`**
    - The color value input can be a hexadecimal string or an RGB tuple, representing the color to be converted into its RGB components. This input is crucial for determining the specific RGB values to be extracted, directly affecting the node's output.
    - Comfy dtype: `*`
    - Python dtype: `Union[str, Tuple[int, int, int]]`
### Optional
## Output types
- **`R`**
    - Comfy dtype: `INT`
    - The red component of the color, extracted as an integer value.
    - Python dtype: `int`
- **`G`**
    - Comfy dtype: `INT`
    - The green component of the color, extracted as an integer value.
    - Python dtype: `int`
- **`B`**
    - Comfy dtype: `INT`
    - The blue component of the color, extracted as an integer value.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ColorValuetoRGBValue:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "color_value": (any, {}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("R", "G", "B")
    FUNCTION = 'color_value_to_rgb_value'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def color_value_to_rgb_value(self, color_value,):
        R, G, B = 0, 0, 0
        if isinstance(color_value, str):
            color = Hex_to_RGB(color_value)
            R, G, B = color[0], color[1], color[2]
        elif isinstance(color_value, tuple):
            R, G, B = color_value[0], color_value[1], color_value[2]
        else:
            log(f"{NODE_NAME}: color_value input type must be tuple or string.", message_type="error")

        return (R, G, B,)

```
