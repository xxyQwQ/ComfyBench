---
tags:
- Text
---

# text
## Documentation
- Class name: `ttN text`
- Category: `🌏 tinyterra/text`
- Output node: `False`

The `ttN text` node is designed for text manipulation within the tinyterra ecosystem, offering functionalities to process and transform text data based on dynamic prompts. It encapsulates the ability to handle multiline text inputs, allowing for versatile text processing operations.
## Input types
### Required
- **`text`**
    - The `text` parameter is the primary input for text manipulation, accepting multiline strings. It supports dynamic prompts, enabling flexible and context-aware text processing.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`text`**
    - Comfy dtype: `STRING`
    - The output is a transformed or processed version of the input text, maintaining the string data type.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [SDXLPromptStyler](../../comfyui-copilot/Nodes/SDXLPromptStyler.md)
    - comfy-qr-by-module-split
    - comfy-qr-validate



## Source code
```python
class ttN_text:
    version = '1.0.0'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True}),
                },
                "hidden": {"ttNnodeVersion": ttN_text.version},
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "conmeow"

    CATEGORY = "🌏 tinyterra/text"

    @staticmethod
    def conmeow(text):
        return text,

```
