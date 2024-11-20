# Preview Text Node
## Documentation
- Class name: `PreviewTextNode`
- Category: `AlekPet Nodes/extras`
- Output node: `True`

The PreviewTextNode is designed to generate a preview of text inputs, allowing users to visualize how their text will appear. It primarily serves as a user interface component within a larger system, enhancing the user experience by providing immediate feedback on text input.
## Input types
### Required
- **`text`**
    - The primary text input for which a preview is generated. It is essential for the node's operation, determining the content of the preview.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - unknown
    - Python dtype: `unknown`
- **`ui`**
    - The user interface representation of the text preview, showcasing how the input text will be displayed.
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class PreviewTextNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = True
    FUNCTION = "preview_text"

    CATEGORY = "AlekPet Nodes/extras"

    def preview_text(self, text, prompt=None, extra_pnginfo=None):
        return {
            "ui": {
                "string": [
                    text,
                ]
            },
            "result": (text,),
        }

```
