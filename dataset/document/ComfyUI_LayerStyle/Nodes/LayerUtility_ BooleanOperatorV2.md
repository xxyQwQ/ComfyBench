# LayerUtility: Boolean Operator V2
## Documentation
- Class name: `LayerUtility: BooleanOperatorV2`
- Category: `😺dzNodes/LayerUtility/Data`
- Output node: `False`

This node performs a variety of boolean operations between two values, supporting both direct value inputs and string representations of numbers. It can execute standard comparisons, logical operations, and even min/max calculations, making it versatile for conditional logic and decision-making processes within data flows.
## Input types
### Required
- **`a_value`**
    - A string representation of the first value, which will be converted to a number if possible. It serves as the primary operand for the boolean operation.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`b_value`**
    - A string representation of the second value, which will be converted to a number if possible. It acts as the secondary operand for the boolean operation.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`operator`**
    - Specifies the boolean operation to perform, such as comparison, logical operation, or min/max calculation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`a`**
    - An optional direct numeric input for the first value, providing an alternative to 'a_value' if a direct number is preferred.
    - Comfy dtype: `*`
    - Python dtype: `float or int`
- **`b`**
    - An optional direct numeric input for the second value, providing an alternative to 'b_value' if a direct number is preferred.
    - Comfy dtype: `*`
    - Python dtype: `float or int`
## Output types
- **`output`**
    - Comfy dtype: `BOOLEAN`
    - The boolean result of the specified operation.
    - Python dtype: `bool`
- **`string`**
    - Comfy dtype: `STRING`
    - The string representation of the boolean result, facilitating easy integration and readability in various contexts.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class BooleanOperatorV2:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        operator_list = ["==", "!=", ">", "<", ">=", "<=", "and", "or", "xor", "not(a)", "min", "max"]
        return {
                    "required":
                    {
                        "a_value": ("STRING", {"default": "", "multiline": False}),
                        "b_value": ("STRING", {"default": "", "multiline": False}),
                        "operator": (operator_list,),
                    },
                    "optional": {
                        "a": (any,),
                        "b": (any,),
                    }
                }

    RETURN_TYPES = ("BOOLEAN", "STRING",)
    RETURN_NAMES = ("output", "string",)
    FUNCTION = 'bool_operator_node_v2'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def bool_operator_node_v2(self, a_value, b_value, operator, a = None, b = None):
        if a is None:
            if a_value != "":
                _numbers = extract_all_numbers_from_str(a_value, checkint=True)
                if len(_numbers) > 0:
                    a = _numbers[0]
                else:
                    a = 0
            else:
                a = 0

        if b is None:
            if b_value != "":
                _numbers = extract_all_numbers_from_str(b_value, checkint=True)
                if len(_numbers) > 0:
                    b = _numbers[0]
                else:
                    b = 0
            else:
                b = 0

        ret_value = False
        if operator == "==":
            ret_value = a == b
        if operator == "!=":
            ret_value = a != b
        if operator == ">":
            ret_value = a > b
        if operator == "<":
            ret_value = a < b
        if operator == ">=":
            ret_value = a >= b
        if operator == "<=":
            ret_value = a <= b
        if operator == "and":
            ret_value = a and b
        if operator == "or":
            ret_value = a or b
        if operator == "xor":
            ret_value = not(a == b)
        if operator == "not(a)":
            ret_value = not a
        if operator == "min":
            ret_value = min(a, b)
        if operator == "max":
            ret_value = max(a, b)

        return (ret_value, str(ret_value))

```
