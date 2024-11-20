---
tags:
- DataConversion
- DataTypeConversion
- NumericConversion
---

# int
## Documentation
- Class name: `ttN int`
- Category: `🌏 tinyterra/util`
- Output node: `False`

The `ttN int` node is designed for integer manipulation within the tinyterra utility category, offering functionality to convert an integer input into multiple formats.
## Input types
### Required
- **`int`**
    - Specifies the integer value to be converted. This parameter is central to the node's operation, as it determines the base value that will be transformed into different formats.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`int`**
    - Comfy dtype: `INT`
    - Returns the original integer input without modification.
    - Python dtype: `int`
- **`float`**
    - Comfy dtype: `FLOAT`
    - Converts and returns the input integer as a float.
    - Python dtype: `float`
- **`text`**
    - Comfy dtype: `STRING`
    - Converts and returns the input integer as a string.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [VHS_VideoCombine](../../ComfyUI-VideoHelperSuite/Nodes/VHS_VideoCombine.md)



## Source code
```python
class ttN_INT:
    version = '1.0.0'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "int": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                },
                "hidden": {"ttNnodeVersion": ttN_INT.version},
        }

    RETURN_TYPES = ("INT", "FLOAT", "STRING",)
    RETURN_NAMES = ("int", "float", "text",)
    FUNCTION = "convert"

    CATEGORY = "🌏 tinyterra/util"

    @staticmethod
    def convert(int):
        return int, float(int), str(int)

```
