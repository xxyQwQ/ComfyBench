# LayerUtility: TextBox
## Documentation
- Class name: `LayerUtility: TextBox`
- Category: `😺dzNodes/LayerUtility/Data`
- Output node: `False`

The TextBox node is designed to encapsulate and return the input text as is, supporting multiline text inputs. It serves as a basic utility for handling and passing through text data within a node-based processing pipeline.
## Input types
### Required
- **`text`**
    - The 'text' parameter accepts a string input, potentially spanning multiple lines, and passes it through without modification. It's essential for operations where the exact input text needs to be preserved or displayed.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`text`**
    - Comfy dtype: `STRING`
    - The output is the same as the input text, allowing for the direct passage of text data through the node.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class TextBoxNode:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        return {"required": {
                "text": ("STRING", {"multiline": True}),
            },}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = 'text_box_node'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def text_box_node(self, text):
        return (text,)

```
