---
tags:
- String
- Text
---

# Something To String
## Documentation
- Class name: `SomethingToString`
- Category: `KJNodes/text`
- Output node: `False`

This node is designed to convert various data types into a string format, optionally allowing for the addition of a prefix or suffix to the resulting string. It abstracts the complexity of data type conversion and string manipulation, providing a straightforward way to generate string representations of diverse inputs.
## Input types
### Required
- **`input`**
    - The primary data that will be converted to a string. This input can be of any type, and its conversion is central to the node's functionality.
    - Comfy dtype: `*`
    - Python dtype: `Union[int, float, bool, list, None]`
### Optional
- **`prefix`**
    - An optional string to prepend to the converted input, allowing for customization of the resulting string's format.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`suffix`**
    - An optional string to append to the converted input, enhancing the flexibility in formatting the resulting string.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - The output is the string representation of the input, potentially modified by the specified prefix and/or suffix.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SomethingToString:
    @classmethod
    
    def INPUT_TYPES(s):
     return {
        "required": {
        "input": (any, {}),
    },
    "optional": {
        "prefix": ("STRING", {"default": ""}),
        "suffix": ("STRING", {"default": ""}),
    }
    }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "stringify"
    CATEGORY = "KJNodes/text"
    DESCRIPTION = """
Converts any type to a string.
"""

    def stringify(self, input, prefix="", suffix=""):
        if isinstance(input, (int, float, bool)):
            stringified = str(input)
        elif isinstance(input, list):
            stringified = ', '.join(str(item) for item in input)
        else:
            return
        if prefix: # Check if prefix is not empty
            stringified = prefix + stringified # Add the prefix
        if suffix: # Check if suffix is not empty
            stringified = stringified + suffix # Add the suffix

        return (stringified,)

```
