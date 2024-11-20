# LayerUtility: HSV Value
## Documentation
- Class name: `LayerUtility: HSV Value`
- Category: `😺dzNodes/LayerUtility/Data`
- Output node: `False`

The 'LayerUtility: HSV Value' node is designed to convert color values into their corresponding HSV (Hue, Saturation, Value) components. It supports input in both hexadecimal string and RGB tuple formats, facilitating the transformation of color representations into a format that's widely used for color manipulation and analysis.
## Input types
### Required
- **`color_value`**
    - The 'color_value' parameter accepts a color in either hexadecimal string format or RGB tuple format. It is crucial for determining the output HSV values, as the node converts this input into its HSV components.
    - Comfy dtype: `*`
    - Python dtype: `Union[str, Tuple[int, int, int]]`
### Optional
## Output types
- **`H`**
    - Comfy dtype: `INT`
    - Represents the Hue component of the color in the HSV color space, as an integer.
    - Python dtype: `int`
- **`S`**
    - Comfy dtype: `INT`
    - Represents the Saturation component of the color in the HSV color space, as an integer.
    - Python dtype: `int`
- **`V`**
    - Comfy dtype: `INT`
    - Represents the Value component of the color in the HSV color space, as an integer.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ColorValuetoHSVValue:

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
    RETURN_NAMES = ("H", "S", "V")
    FUNCTION = 'color_value_to_hsv_value'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def color_value_to_hsv_value(self, color_value,):
        H, S, V = 0, 0, 0
        if isinstance(color_value, str):
            H, S, V = Hex_to_HSV_255level(color_value)
        elif isinstance(color_value, tuple):
            H, S, V = Hex_to_HSV_255level(RGB_to_Hex(color_value))
        else:
            log(f"{NODE_NAME}: color_value input type must be tuple or string.", message_type="error")

        return (H, S, V,)

```
