---
tags:
- Text
- TextReplacement
---

# String Function 🐍
## Documentation
- Class name: `StringFunction_pysssss`
- Category: `utils`
- Output node: `True`

The String Function node provides utilities for manipulating strings, including appending multiple strings together, replacing parts of a string based on a pattern, and tidying up strings by removing extra spaces or adjusting tags. It's designed to offer a range of common string operations that can be dynamically applied based on the input parameters.
## Input types
### Required
- **`action`**
    - Specifies the string operation to perform, such as appending strings together or replacing parts of a string with another. This choice determines how the other inputs are used and the overall behavior of the node.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`tidy_tags`**
    - Determines whether the output string should be tidied by removing extra spaces and adjusting tags, enhancing readability and consistency.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`text_a`**
    - The primary text input for operations. It serves as the base string for append and replace actions.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`text_b`**
    - Secondary text input that can be appended to 'text_a' or used as the search pattern for replacements.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`text_c`**
    - In replace actions, this text replaces occurrences of 'text_b' in 'text_a'. In append actions, it's concatenated with 'text_a' and 'text_b'.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - The result of the string manipulation operation, returned as a string.
    - Python dtype: `Tuple[str]`
- **`ui`**
    - Provides a user interface element displaying the result of the string manipulation.
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class StringFunction:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "action": (["append", "replace"], {}),
                "tidy_tags": (["yes", "no"], {}),
            },
            "optional": {
                "text_a": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "text_b": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "text_c": ("STRING", {"multiline": True, "dynamicPrompts": False})
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "exec"
    CATEGORY = "utils"
    OUTPUT_NODE = True

    def exec(self, action, tidy_tags, text_a="", text_b="", text_c=""):     
        tidy_tags = tidy_tags == "yes"
        out = ""
        if action == "append":
            out = (", " if tidy_tags else "").join(filter(None, [text_a, text_b, text_c]))
        else:
           if text_c is None:
               text_c = ""
           if text_b.startswith("/") and text_b.endswith("/"):
               regex = text_b[1:-1]
               out = re.sub(regex, text_c, text_a)
           else:
               out = text_a.replace(text_b, text_c)
        if tidy_tags:
            out = re.sub(r"\s{2,}", " ", out)
            out = out.replace(" ,", ",")
            out = re.sub(r",{2,}", ",", out)
            out = out.strip()
        return {"ui": {"text": (out,)}, "result": (out,)}

```
