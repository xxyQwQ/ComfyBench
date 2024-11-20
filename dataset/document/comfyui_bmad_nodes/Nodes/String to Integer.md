---
tags:
- DataConversion
- DataTypeConversion
- NumericConversion
---

# String to Integer
## Documentation
- Class name: `String to Integer`
- Category: `Bmad/api/parseInput`
- Output node: `False`

This node is designed to convert string representations of numbers into their integer equivalents. It ensures compatibility with both string and integer inputs, facilitating seamless data processing and integration within a pipeline that may receive inputs from various sources.
## Input types
### Required
- **`inStr`**
    - The input string to be converted into an integer. This parameter allows the node to accept both numerical strings and integer values, ensuring flexibility in handling different types of input.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`int`**
    - Comfy dtype: `INT`
    - The integer representation of the input string or integer value.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class String2Int:
    """
    Under the supposition that this node will receive values from the RequestInputs node,
     will still work with integer values in the json, as int() cast will work both int and str types.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"inStr": ("STRING", {"default": ""})}, }

    RETURN_TYPES = ("INT",)
    FUNCTION = "convert"
    CATEGORY = f"{api_category_path}/parseInput"

    def convert(self, inStr):
        return (int(inStr),)

```
