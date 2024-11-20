---
tags:
- DataConversion
- DataTypeConversion
- FloatData
- NumericConversion
---

# LayerUtility: Float
## Documentation
- Class name: `LayerUtility: Float`
- Category: `😺dzNodes/LayerUtility/Data`
- Output node: `False`

The Float node is designed to process floating-point numbers, allowing for the manipulation and conversion of float values within a given range. It provides functionality to input a float value, perform any necessary processing, and output the processed value along with its string representation.
## Input types
### Required
- **`float_value`**
    - Specifies the floating-point number to be processed. This parameter is crucial for determining the node's operation and output, as it directly influences the manipulation and conversion of the float value.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`float`**
    - Comfy dtype: `FLOAT`
    - The processed floating-point number after the node's operation.
    - Python dtype: `float`
- **`string`**
    - Comfy dtype: `STRING`
    - The string representation of the processed floating-point number.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class FloatNode:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        return {"required": {
                "float_value":  ("FLOAT", {"default": 0, "min": -99999999999999999999, "max": 99999999999999999999, "step": 0.00001}),
            },}

    RETURN_TYPES = ("FLOAT", "STRING",)
    RETURN_NAMES = ("float", "string",)
    FUNCTION = 'float_node'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def float_node(self, float_value):
        return (float_value, str(float_value))

```
