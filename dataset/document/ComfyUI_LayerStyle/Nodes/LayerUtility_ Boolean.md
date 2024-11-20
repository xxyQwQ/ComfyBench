# LayerUtility: Boolean
## Documentation
- Class name: `LayerUtility: Boolean`
- Category: `😺dzNodes/LayerUtility/Data`
- Output node: `False`

The Boolean node is designed to encapsulate and convert a boolean value into a boolean and string representation. It serves as a fundamental utility within data processing pipelines, allowing for the straightforward manipulation and interpretation of boolean logic.
## Input types
### Required
- **`bool_value`**
    - Determines the boolean value to be processed by the node. It is crucial for defining the logical state that will be converted and represented in both boolean and string formats.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`boolean`**
    - Comfy dtype: `BOOLEAN`
    - The boolean representation of the input value.
    - Python dtype: `bool`
- **`string`**
    - Comfy dtype: `STRING`
    - The string representation of the boolean input value.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class BooleanNode:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        return {"required": {
                "bool_value": ("BOOLEAN", {"default": False}),
            },}

    RETURN_TYPES = ("BOOLEAN", "STRING",)
    RETURN_NAMES = ("boolean", "string",)
    FUNCTION = 'boolean_node'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def boolean_node(self, bool_value):
        return (bool_value, str(bool_value))

```
