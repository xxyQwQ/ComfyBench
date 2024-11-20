# HEX to HUE Node
## Documentation
- Class name: `HexToHueNode`
- Category: `AlekPet Nodes/extras`
- Output node: `False`

The HexToHueNode is designed to convert hexadecimal color values to their corresponding hue values, providing both the hue in degrees and normalized hue as outputs. It serves as a utility for color manipulation, enabling the transformation of color specifications from a web-friendly format to a format that can be used for more detailed color analysis and adjustments.
## Input types
### Required
- **`color_hex`**
    - The hexadecimal representation of a color, which the node converts into hue values. This input is essential for determining the specific color's hue characteristics.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`string_hex`**
    - Comfy dtype: `STRING`
    - The original hexadecimal color value inputted to the node.
    - Python dtype: `str`
- **`float_hue_degrees`**
    - Comfy dtype: `FLOAT`
    - The hue of the color in degrees, providing a measure of the color's position on the color wheel.
    - Python dtype: `float`
- **`float_hue_norm`**
    - Comfy dtype: `FLOAT`
    - The normalized hue value, scaled between -0.5 and 0.5, offering a standardized measure of hue.
    - Python dtype: `float`
- **`string_hue_degrees`**
    - Comfy dtype: `STRING`
    - The hue value in degrees, represented as a string.
    - Python dtype: `str`
- **`string_hue_norm`**
    - Comfy dtype: `STRING`
    - The normalized hue value, represented as a string, for ease of use in contexts where a string format is preferred.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class HexToHueNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "color_hex": (
                    "STRING",
                    {"default": "#00ff33"},
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = (
        "string_hex",
        "float_hue_degrees",
        "float_hue_norm",
        "string_hue_degrees",
        "string_hue_norm",
    )
    FUNCTION = "to_hue"
    CATEGORY = "AlekPet Nodes/extras"

    def to_hue(self, color_hex, unique_id):
        hue_degrees = ColorsCorrectNode.hex_to_hue(color_hex)
        hue_norm = ColorsCorrectNode.degrees_to_hue(hue_degrees)
        PromptServer.instance.send_sync(
            "alekpet_get_color_hex", {"color_hex": color_hex, "unique_id": unique_id}
        )
        return (color_hex, hue_degrees, hue_norm, str(hue_degrees), str(hue_norm))

```
