# LayerUtility: Boolean Operator
## Documentation
- Class name: `LayerUtility: BooleanOperator`
- Category: `😺dzNodes/LayerUtility/Data`
- Output node: `False`

This node performs a variety of boolean operations between two inputs based on a specified operator. It abstracts the complexity of conditional logic into a simple interface, allowing users to easily implement logical conditions and comparisons within their data flows.
## Input types
### Required
- **`a`**
    - The first input for the boolean operation. Its role varies based on the chosen operator, acting either as a standalone value for unary operations or one of the operands for binary operations.
    - Comfy dtype: `*`
    - Python dtype: `Union[int, float, bool, str]`
- **`b`**
    - The second input for the boolean operation, serving as the other operand in binary operations. It is not used in unary operations like 'not(a)'.
    - Comfy dtype: `*`
    - Python dtype: `Union[int, float, bool, str]`
- **`operator`**
    - Specifies the boolean operation to be performed. It supports a range of operators including comparison, logical, and bitwise operations, directly influencing the outcome of the node.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`output`**
    - Comfy dtype: `BOOLEAN`
    - The result of the boolean operation, indicating the truth value of the applied condition.
    - Python dtype: `bool`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class BooleanOperator:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        operator_list = ["==", "!=", ">", "<", ">=", "<=", "and", "or", "xor", "not(a)", "min", "max"]
        return {"required": {
                "a": (any, ),
                "b": (any, ),
                "operator": (operator_list,),
            },}

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("output",)
    FUNCTION = 'bool_operator_node'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def bool_operator_node(self, a, b, operator):
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

        return (ret_value,)

```
