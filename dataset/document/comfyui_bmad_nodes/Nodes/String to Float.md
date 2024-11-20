---
tags:
- DataConversion
- DataTypeConversion
- NumericConversion
---

# String to Float
## Documentation
- Class name: `String to Float`
- Category: `Bmad/api/parseInput`
- Output node: `False`

This node is designed to convert string representations of floating-point numbers into their float equivalents. It ensures that inputs, even if provided as strings, are accurately transformed to float values for further numerical processing.
## Input types
### Required
- **`inStr`**
    - Takes a string input that represents a floating-point number and converts it into a float. This allows for string-based numerical inputs to be accurately processed and utilized in calculations.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`float`**
    - Comfy dtype: `FLOAT`
    - Outputs the converted floating-point number from the string input, facilitating its use in numerical operations and calculations.
    - Python dtype: `float`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class String2Float:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"inStr": ("STRING", {"default": ""})}, }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "convert"
    CATEGORY = f"{api_category_path}/parseInput"

    def convert(self, inStr):
        return (float(inStr),)

```
