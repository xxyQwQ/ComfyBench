---
tags:
- Concatenate
- Text
---

# 3x TXT Loader MultiConcat
## Documentation
- Class name: `ttN text3BOX_3WAYconcat`
- Category: `🌏 tinyterra/text`
- Output node: `False`

This node is designed to concatenate three input strings in various combinations, allowing for flexible string manipulation and concatenation strategies. It supports custom delimiters and can generate multiple output strings, including each input string individually, pairs of input strings concatenated together, and all three input strings concatenated.
## Input types
### Required
- **`text1`**
    - The first text string to be concatenated. It plays a crucial role in the concatenation process, affecting the output by being part of multiple concatenation combinations.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`text2`**
    - The second text string to be concatenated. It is essential for creating concatenated pairs with the first and third strings, as well as contributing to the overall three-string concatenation.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`text3`**
    - The third text string to be concatenated. This string is combined with the first and second strings in various ways to produce the node's output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`delimiter`**
    - A string used as the delimiter in the concatenation process. It defines the separator between the concatenated strings, impacting the format and readability of the output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`text1`**
    - Comfy dtype: `STRING`
    - Returns the first input text string as is, without any concatenation.
    - Python dtype: `str`
- **`text2`**
    - Comfy dtype: `STRING`
    - Returns the second input text string as is, without any concatenation.
    - Python dtype: `str`
- **`text3`**
    - Comfy dtype: `STRING`
    - Returns the third input text string as is, without any concatenation.
    - Python dtype: `str`
- **`1 & 2`**
    - Comfy dtype: `STRING`
    - The concatenation of the first and second input text strings, separated by the specified delimiter.
    - Python dtype: `str`
- **`1 & 3`**
    - Comfy dtype: `STRING`
    - The concatenation of the first and third input text strings, separated by the specified delimiter.
    - Python dtype: `str`
- **`2 & 3`**
    - Comfy dtype: `STRING`
    - The concatenation of the second and third input text strings, separated by the specified delimiter.
    - Python dtype: `str`
- **`concat`**
    - Comfy dtype: `STRING`
    - The concatenation of all three input text strings, separated by the specified delimiter.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ttN_text3BOX_3WAYconcat:
    version = '1.0.0'
    def __init__(self):
        pass
    """
    Concatenate 3 strings, in various ways.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text1": ("STRING", {"multiline": True, "default": '', "dynamicPrompts": True}),
                    "text2": ("STRING", {"multiline": True, "default": '', "dynamicPrompts": True}),
                    "text3": ("STRING", {"multiline": True, "default": '', "dynamicPrompts": True}),
                    "delimiter": ("STRING", {"default":",","multiline": False}),
                    },
                "hidden": {"ttNnodeVersion": ttN_text3BOX_3WAYconcat.version},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("text1", "text2", "text3", "1 & 2", "1 & 3", "2 & 3", "concat",)
    FUNCTION = "conmeow"

    CATEGORY = "🌏 tinyterra/text"

    def conmeow(self, text1='', text2='', text3='', delimiter=''):
        text1 = '' if text1 == 'undefined' else text1
        text2 = '' if text2 == 'undefined' else text2
        text3 = '' if text3 == 'undefined' else text3

        if delimiter == '\\n':
            delimiter = '\n'

        t_1n2 = delimiter.join([text1, text2])
        t_1n3 = delimiter.join([text1, text3])
        t_2n3 = delimiter.join([text2, text3])
        concat = delimiter.join([text1, text2, text3])
       
        return text1, text2, text3, t_1n2, t_1n3, t_2n3, concat

```
