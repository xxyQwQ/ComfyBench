---
tags:
- DataConversion
- DataTypeConversion
- NumericConversion
---

# Input/String to Int Array
## Documentation
- Class name: `Input_String to Int Array`
- Category: `Bmad/api/parseInput`
- Output node: `False`

This node is designed to convert input strings into arrays of integers, accommodating both direct array inputs and string representations of integer arrays separated by commas. It ensures flexibility in handling inputs by accepting polymorphic input types.
## Input types
### Required
- **`inStr`**
    - Accepts a string of integers separated by commas or an array of integers. This input flexibility allows for easy integration with various input sources.
    - Comfy dtype: `STRING`
    - Python dtype: `Union[str, List[int]]`
## Output types
- **`int_array`**
    - Comfy dtype: `INT_ARRAY`
    - Outputs an array of integers, either directly from the input array or by converting a comma-separated string into an integer array.
    - Python dtype: `List[int]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class InputString2IntArray:
    """
    Under the supposition this will be used with RequestInputs, the integers may already come as an array.
    The input is, therefore, polymorphic and both array and string types are accepted as inputs to both allow a valid
    request json and a mock array given via the web UI.

    When using a string: the integers should be separated with commas
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"inStr": ("STRING", {"default": ""})}, }

    RETURN_TYPES = ("INT_ARRAY",)
    FUNCTION = "convert"
    CATEGORY = f"{api_category_path}/parseInput"

    def convert(self, inStr):
        # not really a str, suppose is a list read from the input json
        if isinstance(inStr, list):
            return (inStr,)

        # otherwise suppose it is a valid string
        return ([int(x) for x in inStr.split(',')],)

```
