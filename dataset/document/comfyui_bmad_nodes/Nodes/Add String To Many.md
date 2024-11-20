---
tags:
- Concatenate
---

# Add String To Many
## Documentation
- Class name: `Add String To Many`
- Category: `Bmad/conditioning`
- Output node: `False`

This node is designed to either append or prepend a given string to a list of other strings, allowing for batch processing of string manipulation tasks. It supports a dynamic number of input strings and offers flexibility in how the string is added, catering to various use cases where string concatenation or modification is required.
## Input types
### Required
- **`to_add`**
    - Specifies the string to be appended or prepended to each input string, acting as the primary modifier in the concatenation process.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`inputs_len`**
    - Determines the number of input strings to which the operation will be applied, allowing for dynamic adjustment based on user needs.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`operation`**
    - Controls whether the specified string is appended or prepended to the input strings, offering versatility in string manipulation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - The modified strings after the specified string has been either appended or prepended, reflecting the result of the batch string manipulation.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class AddString2Many:
    """
    Append or prepend a string to other, many, strings.
    """

    OPERATION = ["append", "prepend"]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "to_add": ("STRING", {"default": '', "multiline": False}),
            "inputs_len": ("INT", {"default": 3, "min": 2, "max": 32, "step": 1}),
            "operation": (cls.OPERATION, {"default": 'append'}),
        }}

    RETURN_TYPES = tuple(["STRING" for x in range(32)])
    FUNCTION = "add_str"
    CATEGORY = conditioning_category_path

    def add_str(self, to_add, inputs_len, operation, **kwargs):
        new_strs = []
        for r in range(inputs_len):
            str_input_name = f"i{r + 1}"
            new_str = kwargs[str_input_name]
            if operation == "append":
                new_str = new_str + to_add
            else:
                new_str = to_add + new_str
            new_strs.append(new_str)

        return tuple(new_strs)

```
