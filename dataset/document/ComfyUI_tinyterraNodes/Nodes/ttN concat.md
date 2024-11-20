---
tags:
- Concatenate
- Text
---

# textConcat
## Documentation
- Class name: `ttN concat`
- Category: `🌏 tinyterra/text`
- Output node: `False`

The ttN concat node is designed for concatenating text strings, providing a simple yet flexible way to merge multiple text inputs into a single string. This functionality is essential in text processing workflows where combining pieces of text in various configurations is required.
## Input types
### Required
- **`text1`**
    - The first text string to be concatenated. It serves as the initial part of the final concatenated text.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`text2`**
    - The second text string to be concatenated. It follows the first text in the final concatenated output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`text3`**
    - The third text string to be concatenated. It is added after the first and second texts in the final output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`delimiter`**
    - A string used to separate the concatenated texts. It defines how the texts are joined together in the final output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`concat`**
    - Comfy dtype: `STRING`
    - The result of concatenating the input texts separated by the specified delimiter.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ttN_concat:
    version = '1.0.0'
    def __init__(self):
        pass
    """
    Concatenate 2 strings
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text1": ("STRING", {"multiline": True, "default": '', "dynamicPrompts": True}),
                    "text2": ("STRING", {"multiline": True, "default": '', "dynamicPrompts": True}),
                    "text3": ("STRING", {"multiline": True, "default": '', "dynamicPrompts": True}),
                    "delimiter": ("STRING", {"default":",","multiline": False}),
                    },
                "hidden": {"ttNnodeVersion": ttN_concat.version},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("concat",)
    FUNCTION = "conmeow"

    CATEGORY = "🌏 tinyterra/text"

    def conmeow(self, text1='', text2='', text3='', delimiter=''):
        text1 = '' if text1 == 'undefined' else text1
        text2 = '' if text2 == 'undefined' else text2
        text3 = '' if text3 == 'undefined' else text3

        if delimiter == '\\n':
            delimiter = '\n'

        concat = delimiter.join([text1, text2, text3])
       
        return (concat,)

```
