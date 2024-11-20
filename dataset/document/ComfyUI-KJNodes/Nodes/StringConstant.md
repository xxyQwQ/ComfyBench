---
tags:
- Constant
---

# String Constant
## Documentation
- Class name: `StringConstant`
- Category: `KJNodes/constants`
- Output node: `False`

The StringConstant node is designed to pass through a string input without modification, serving as a constant string value within a flow.
## Input types
### Required
- **`string`**
    - Represents the string input to be passed through. This parameter allows for the input of a single-line string, with a default value of an empty string if none is provided.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - Outputs the same string that was input, effectively acting as a pass-through for the string value.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class StringConstant:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"default": '', "multiline": False}),
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "passtring"
    CATEGORY = "KJNodes/constants"

    def passtring(self, string):
        return (string, )

```
