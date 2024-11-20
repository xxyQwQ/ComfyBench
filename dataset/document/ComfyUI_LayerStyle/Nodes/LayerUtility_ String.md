---
tags:
- Text
---

# LayerUtility: String
## Documentation
- Class name: `LayerUtility: String`
- Category: `😺dzNodes/LayerUtility/Data`
- Output node: `False`

This node is designed to handle string operations within a layer, providing functionality to manipulate and process text data as part of a larger workflow. It abstracts the complexities of text handling, offering a streamlined interface for operations such as formatting, concatenation, or conversion, within the context of layer-based utilities.
## Input types
### Required
- **`string`**
    - Represents the text input that the node will process, emphasizing its role in the manipulation or analysis of string data. This input is crucial for operations that require textual content to work with.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - Outputs the processed text, reflecting the successful application of the specified string operations.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class StringNode:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        return {"required": {
                "string": ("STRING", {"multiline": False}),
            },}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = 'string_node'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def string_node(self, string):
        return (string,)

```
