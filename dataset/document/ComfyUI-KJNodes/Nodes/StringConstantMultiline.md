---
tags:
- List
- MultilineText
- Text
---

# String Constant Multiline
## Documentation
- Class name: `StringConstantMultiline`
- Category: `KJNodes/constants`
- Output node: `False`

The StringConstantMultiline node is designed to process multiline string inputs, optionally stripping newline characters based on a boolean flag. It enables the manipulation of text data by allowing the preservation or removal of formatting, making it versatile for various text processing tasks.
## Input types
### Required
- **`string`**
    - The primary text input that the node processes. It accepts multiline strings, enabling the handling of text data that spans multiple lines.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`strip_newlines`**
    - A boolean flag that determines whether newline characters in the input string should be removed. When set to true, it strips newline characters, flattening the text to a single line.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - Outputs the processed string, which may have newline characters removed based on the input flag. This allows for flexible text manipulation and formatting.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class StringConstantMultiline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"default": "", "multiline": True}),
                "strip_newlines": ("BOOLEAN", {"default": True}),
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "stringify"
    CATEGORY = "KJNodes/constants"

    def stringify(self, string, strip_newlines):
        new_string = []
        for line in io.StringIO(string):
            if not line.strip().startswith("\n") and strip_newlines:
                line = line.replace("\n", '')
            new_string.append(line)
        new_string = "\n".join(new_string)

        return (new_string, )

```
