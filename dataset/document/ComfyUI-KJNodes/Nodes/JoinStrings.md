---
tags:
- Concatenate
---

# Join Strings
## Documentation
- Class name: `JoinStrings`
- Category: `KJNodes/constants`
- Output node: `False`

The JoinStrings node is designed to concatenate two input strings using a specified delimiter, producing a single output string. This functionality is essential for string manipulation tasks where combining pieces of text in a structured manner is required.
## Input types
### Required
- **`string1`**
    - The first string to be joined. It serves as the starting point of the concatenated result.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`string2`**
    - The second string to be joined. It is appended to the first string, separated by the specified delimiter.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`delimiter`**
    - A string used to separate the two input strings in the concatenated result. It defines the boundary between the joined strings.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - The result of concatenating string1 and string2 with the delimiter in between.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class JoinStrings:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string1": ("STRING", {"default": '', "forceInput": True}),
                "string2": ("STRING", {"default": '', "forceInput": True}),
                "delimiter": ("STRING", {"default": ' ', "multiline": False}),
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "joinstring"
    CATEGORY = "KJNodes/constants"

    def joinstring(self, string1, string2, delimiter):
        joined_string = string1 + delimiter + string2
        return (joined_string, )

```
