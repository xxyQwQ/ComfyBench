---
tags:
- DataConversion
- DataTypeConversion
- NumericConversion
---

# LayerUtility: Integer
## Documentation
- Class name: `LayerUtility: Integer`
- Category: `😺dzNodes/LayerUtility/Data`
- Output node: `False`

The Integer node is designed to process and manipulate integer values. It allows for the conversion of an integer input into both its integer and string representations, facilitating operations that require numerical manipulation or the integration of integer values into text-based formats.
## Input types
### Required
- **`int_value`**
    - Specifies the integer value to be processed. This parameter enables the node to perform operations on the provided integer, converting it to both its integer and string representations.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`int`**
    - Comfy dtype: `INT`
    - The integer representation of the input value.
    - Python dtype: `int`
- **`string`**
    - Comfy dtype: `STRING`
    - The string representation of the input integer value.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class IntegerNode:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        return {"required": {
                "int_value":("INT", {"default": 0, "min": -99999999999999999999, "max": 99999999999999999999, "step": 1}),
            },}

    RETURN_TYPES = ("INT", "STRING",)
    RETURN_NAMES = ("int", "string",)
    FUNCTION = 'integer_node'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def integer_node(self, int_value):
        return (int_value, str(int_value))

```
