---
tags:
- DataConversion
- DataTypeConversion
- NumericConversion
---

# float
## Documentation
- Class name: `ttN float`
- Category: `🌏 tinyterra/util`
- Output node: `False`

The `ttN float` node is designed to perform conversions involving floating-point numbers. It takes a floating-point number as input and outputs the same number in three different formats: as a float, as an integer by truncating the decimal part, and as a string representation of the float. This node is useful in scenarios where different representations of a floating-point number are needed for further processing or display purposes.
## Input types
### Required
- **`float`**
    - The `float` parameter is the primary input for the `ttN float` node, representing the floating-point number to be converted. It is crucial for determining the output in three different formats: float, integer, and string. The precision and value of this input directly affect the conversion results.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`float`**
    - Comfy dtype: `FLOAT`
    - The original floating-point number inputted to the node, unchanged.
    - Python dtype: `float`
- **`int`**
    - Comfy dtype: `INT`
    - The integer part of the input floating-point number, obtained by truncating the decimal part.
    - Python dtype: `int`
- **`text`**
    - Comfy dtype: `STRING`
    - The string representation of the input floating-point number, preserving its decimal part.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ttN_FLOAT:
    version = '1.0.0'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "float": ("FLOAT", {"default": 0.00, "min": 0.00, "max": 0xffffffffffffffff, 'step': 0.01}),
                },
                "hidden": {"ttNnodeVersion": ttN_FLOAT.version},
        }

    RETURN_TYPES = ("FLOAT", "INT", "STRING",)
    RETURN_NAMES = ("float", "int", "text",)
    FUNCTION = "convert"

    CATEGORY = "🌏 tinyterra/util"

    @staticmethod
    def convert(float):
        return float, int(float), str(float)

```
