---
tags:
- String
- Text
---

# String
## Documentation
- Class name: `String-🔬`
- Category: `Logic`
- Output node: `False`

The String node is designed to process and return string values. It encapsulates the functionality to accept a string input, potentially perform operations on it, and then output the processed string.
## Input types
### Required
- **`value`**
    - Represents the string value to be processed by the node. It is the primary input upon which the node operates, allowing for potential manipulation or evaluation of string data.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`STRING`**
    - Comfy dtype: `STRING`
    - Outputs the processed or original string value received as input, depending on the node's implementation.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class String:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"value": ("STRING", {"default": "", "multiline": True})},
        }

    RETURN_TYPES = ("STRING",)

    RETURN_NAMES = ("STRING",)

    FUNCTION = "execute"

    CATEGORY = "Logic"

    def execute(self, value):
        return (value,)

```
