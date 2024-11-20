# LayerUtility: Number Calculator
## Documentation
- Class name: `LayerUtility: NumberCalculator`
- Category: `😺dzNodes/LayerUtility/Data`
- Output node: `False`

The NumberCalculator node performs basic arithmetic operations between two input values based on a specified operator. It supports a variety of operations including addition, subtraction, multiplication, division, exponentiation, integer division, and modulo. This node is designed to facilitate mathematical calculations within data processing pipelines, allowing for dynamic computation based on input parameters.
## Input types
### Required
- **`a`**
    - The first operand in the arithmetic operation. It plays a crucial role in determining the outcome of the computation.
    - Comfy dtype: `*`
    - Python dtype: `Union[int, float]`
- **`b`**
    - The second operand in the arithmetic operation. It is used in conjunction with the first operand to perform the specified mathematical function.
    - Comfy dtype: `*`
    - Python dtype: `Union[int, float]`
- **`operator`**
    - Specifies the arithmetic operation to be performed on the operands. The operator can be one of the following: '+', '-', '*', '/', '**', '//', '%'. This choice dictates the nature of the calculation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`int`**
    - Comfy dtype: `INT`
    - The integer part of the result from the arithmetic operation.
    - Python dtype: `int`
- **`float`**
    - Comfy dtype: `FLOAT`
    - The floating-point representation of the result from the arithmetic operation.
    - Python dtype: `float`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class NumberCalculator:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        operator_list = ["+", "-", "*", "/", "**", "//", "%" ]
        return {"required": {
                "a": (any, {}),
                "b": (any, {}),
                "operator": (operator_list,),
            },}

    RETURN_TYPES = ("INT", "FLOAT",)
    RETURN_NAMES = ("int", "float",)
    FUNCTION = 'number_calculator_node'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def number_calculator_node(self, a, b, operator):
        ret_value = 0
        if operator == "+":
            ret_value = a + b
        if operator == "-":
            ret_value = a - b
        if operator == "*":
            ret_value = a * b
        if operator == "**":
            ret_value = a ** b
        if operator == "%":
            ret_value = a % b
        if operator == "/":
            if b != 0:
                ret_value = a / b
            else:
                ret_value = 0
        if operator == "//":
            if b != 0:
                ret_value = a // b
            else:
                ret_value = 0

        return (int(ret_value), float(ret_value),)

```
