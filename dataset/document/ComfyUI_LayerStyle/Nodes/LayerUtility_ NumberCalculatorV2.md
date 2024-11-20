# LayerUtility: Number Calculator V2
## Documentation
- Class name: `LayerUtility: NumberCalculatorV2`
- Category: `😺dzNodes/LayerUtility/Data`
- Output node: `False`

The NumberCalculatorV2 node is designed to perform a variety of arithmetic operations between two numbers, including basic arithmetic, exponentiation, and nth root calculations. It extends the functionality of basic arithmetic by allowing inputs as strings and parsing them for numerical values, providing a flexible interface for dynamic calculation scenarios.
## Input types
### Required
- **`a_value`**
    - A string representing the first number or a numeric expression. If 'a' is not provided, 'a_value' is parsed to extract the first number for the operation, enhancing flexibility in handling numeric inputs.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`b_value`**
    - A string representing the second number or a numeric expression. Similar to 'a_value', if 'b' is not provided, 'b_value' is parsed to extract the first number, allowing for dynamic numeric operations based on string inputs.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`operator`**
    - Specifies the arithmetic operation to perform on the numbers. It supports a range of operators including basic arithmetic and nth root, enabling complex mathematical calculations.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`a`**
    - An optional numeric input that directly specifies the first operand. If provided, it overrides the number extracted from 'a_value', offering direct control over the calculation.
    - Comfy dtype: `*`
    - Python dtype: `Union[int, float, None]`
- **`b`**
    - An optional numeric input that directly specifies the second operand. If provided, it overrides the number extracted from 'b_value', giving direct input capability for the operation.
    - Comfy dtype: `*`
    - Python dtype: `Union[int, float, None]`
## Output types
- **`int`**
    - Comfy dtype: `INT`
    - The integer part of the calculation result, providing a discrete numerical outcome.
    - Python dtype: `int`
- **`float`**
    - Comfy dtype: `FLOAT`
    - The floating-point part of the calculation result, offering a precise numerical outcome.
    - Python dtype: `float`
- **`string`**
    - Comfy dtype: `STRING`
    - The string representation of the calculation result, enabling easy integration and display in various contexts.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class NumberCalculatorV2:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        operator_list = ["+", "-", "*", "/", "**", "//", "%" , "nth_root"]

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

    RETURN_TYPES = ("INT", "FLOAT", "STRING",)
    RETURN_NAMES = ("int", "float", "string",)
    FUNCTION = 'number_calculator_node_v2'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def number_calculator_node_v2(self, a_value, b_value, operator, a = None, b = None):
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
        if operator == "nth_root":
            ret_value = a ** (1/b)
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

        return (int(ret_value), float(ret_value), str(ret_value))

```
