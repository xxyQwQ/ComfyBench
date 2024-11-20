---
tags:
- Concatenate
- Text
---

# 7x TXT Loader Concat
## Documentation
- Class name: `ttN text7BOX_concat`
- Category: `🌏 tinyterra/text`
- Output node: `False`

This node is designed to concatenate up to seven text strings, allowing for flexible text manipulation and combination. It supports custom delimiters, including new lines, to format the concatenated result according to specific requirements.
## Input types
### Required
- **`text1`**
    - The first text string to be concatenated. It plays a foundational role in the concatenation process, starting the combined text output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`text2`**
    - The second text string to be concatenated, adding to the sequence initiated by text1.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`text3`**
    - The third text string to be concatenated, further extending the combined text output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`text4`**
    - The fourth text string to be concatenated, contributing to the growing list of combined texts.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`text5`**
    - The fifth text string to be concatenated, adding more content to the concatenation.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`text6`**
    - The sixth text string to be concatenated, continuing to build upon the concatenated text.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`text7`**
    - The seventh and final text string to be concatenated, completing the concatenation process.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`delimiter`**
    - The delimiter used to separate each text string in the concatenation. It can significantly affect the formatting of the output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`text1`**
    - Comfy dtype: `STRING`
    - Returns the first input text string, unchanged.
    - Python dtype: `str`
- **`text2`**
    - Comfy dtype: `STRING`
    - Returns the second input text string, unchanged.
    - Python dtype: `str`
- **`text3`**
    - Comfy dtype: `STRING`
    - Returns the third input text string, unchanged.
    - Python dtype: `str`
- **`text4`**
    - Comfy dtype: `STRING`
    - Returns the fourth input text string, unchanged.
    - Python dtype: `str`
- **`text5`**
    - Comfy dtype: `STRING`
    - Returns the fifth input text string, unchanged.
    - Python dtype: `str`
- **`text6`**
    - Comfy dtype: `STRING`
    - Returns the sixth input text string, unchanged.
    - Python dtype: `str`
- **`text7`**
    - Comfy dtype: `STRING`
    - Returns the seventh input text string, unchanged.
    - Python dtype: `str`
- **`concat`**
    - Comfy dtype: `STRING`
    - Returns the concatenated result of all seven input text strings, separated by the specified delimiter.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ttN_text7BOX_concat:
    version = '1.0.0'
    def __init__(self):
        pass
    """
    Concatenate many strings
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text1": ("STRING", {"multiline": True, "default": '', "dynamicPrompts": True}),
                    "text2": ("STRING", {"multiline": True, "default": '', "dynamicPrompts": True}),
                    "text3": ("STRING", {"multiline": True, "default": '', "dynamicPrompts": True}),
                    "text4": ("STRING", {"multiline": True, "default": '', "dynamicPrompts": True}),
                    "text5": ("STRING", {"multiline": True, "default": '', "dynamicPrompts": True}),
                    "text6": ("STRING", {"multiline": True, "default": '', "dynamicPrompts": True}),
                    "text7": ("STRING", {"multiline": True, "default": '', "dynamicPrompts": True}),
                    "delimiter": ("STRING", {"default":",","multiline": False}),
                    },
                "hidden": {"ttNnodeVersion": ttN_text7BOX_concat.version},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("text1", "text2", "text3", "text4", "text5", "text6", "text7", "concat",)
    FUNCTION = "conmeow"

    CATEGORY = "🌏 tinyterra/text"

    def conmeow(self, text1, text2, text3, text4, text5, text6, text7, delimiter):
        text1 = '' if text1 == 'undefined' else text1
        text2 = '' if text2 == 'undefined' else text2
        text3 = '' if text3 == 'undefined' else text3
        text4 = '' if text4 == 'undefined' else text4
        text5 = '' if text5 == 'undefined' else text5
        text6 = '' if text6 == 'undefined' else text6
        text7 = '' if text7 == 'undefined' else text7

        if delimiter == '\\n':
            delimiter = '\n'
            
        texts = [text1, text2, text3, text4, text5, text6, text7]        
        concat = delimiter.join(text for text in texts if text)
        return text1, text2, text3, text4, text5, text6, text7, concat

```
