# RGB Hex To HSV (Inspire)
## Documentation
- Class name: `RGB_HexToHSV __Inspire`
- Category: `InspirePack/Util`
- Output node: `False`

The RGB_HexToHSV node is designed to convert RGB color values in hexadecimal format to their corresponding HSV (Hue, Saturation, Value) representation. This conversion facilitates tasks that require understanding or manipulation of colors in terms of their hue, saturation, and value components, rather than the RGB color model.
## Input types
### Required
- **`rgb_hex`**
    - The 'rgb_hex' parameter represents the RGB color value in hexadecimal format that is to be converted to HSV. This input is crucial for the conversion process, as it directly influences the output HSV values.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`hue`**
    - Comfy dtype: `FLOAT`
    - The 'hue' output represents the color's position on the color wheel, expressed in degrees from 0 to 360.
    - Python dtype: `float`
- **`saturation`**
    - Comfy dtype: `FLOAT`
    - The 'saturation' output indicates the intensity of the color, ranging from 0 (grey) to 1 (full color).
    - Python dtype: `float`
- **`value`**
    - Comfy dtype: `FLOAT`
    - The 'value' output measures the brightness of the color, with 0 being completely dark (black) and 1 being fully bright (the color itself).
    - Python dtype: `float`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RGB_HexToHSV:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "rgb_hex": ("STRING", {"defaultInput": True}),
                    },
                }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("hue", "saturation", "value")
    FUNCTION = "doit"
    CATEGORY = "InspirePack/Util"

    def doit(self, rgb_hex):
        return hex_to_hsv(rgb_hex)

```
