---
tags:
- DataTypeConversion
- Math
- MathematicalExpressions
---

# 🔧 Simple Math
## Documentation
- Class name: `SimpleMath+`
- Category: `essentials/utilities`
- Output node: `False`

SimpleMath is a node designed to evaluate mathematical expressions dynamically. It interprets and computes expressions from abstract syntax trees (AST), supporting basic arithmetic operations, variable resolution, custom function calls, and array indexing.
## Input types
### Required
- **`value`**
    - The mathematical expression to be evaluated, represented as a string. It is parsed into an abstract syntax tree (AST) for evaluation, supporting operations like addition, subtraction, multiplication, division, and more complex expressions involving variables and functions.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`a`**
    - An optional variable 'a' that can be used within the mathematical expression. Its value affects the computation if 'a' is referenced in the expression.
    - Comfy dtype: `INT,FLOAT`
    - Python dtype: `Union[int, float]`
- **`b`**
    - An optional variable 'b' that can be used within the mathematical expression. Its value affects the computation if 'b' is referenced in the expression.
    - Comfy dtype: `INT,FLOAT`
    - Python dtype: `Union[int, float]`
## Output types
- **`int`**
    - Comfy dtype: `INT`
    - The rounded integer result of the evaluated mathematical expression.
    - Python dtype: `int`
- **`float`**
    - Comfy dtype: `FLOAT`
    - The exact floating-point result of the evaluated mathematical expression.
    - Python dtype: `float`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SimpleMath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "a": ("INT,FLOAT", { "default": 0.0, "step": 0.1 }),
                "b": ("INT,FLOAT", { "default": 0.0, "step": 0.1 }),
            },
            "required": {
                "value": ("STRING", { "multiline": False, "default": "" }),
            },
        }

    RETURN_TYPES = ("INT", "FLOAT", )
    FUNCTION = "execute"
    CATEGORY = "essentials/utilities"

    def execute(self, value, a = 0.0, b = 0.0):
        import ast
        import operator as op

        operators = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.FloorDiv: op.floordiv,
            ast.Pow: op.pow,
            ast.BitXor: op.xor,
            ast.USub: op.neg,
            ast.Mod: op.mod,
        }

        op_functions = {
            'min': min,
            'max': max,
            'round': round,
            'sum': sum,
            'len': len,
        }

        def eval_(node):
            if isinstance(node, ast.Num): # number
                return node.n
            elif isinstance(node, ast.Name): # variable
                if node.id == "a":
                    return a
                if node.id == "b":
                    return b
            elif isinstance(node, ast.BinOp): # <left> <operator> <right>
                return operators[type(node.op)](eval_(node.left), eval_(node.right))
            elif isinstance(node, ast.UnaryOp): # <operator> <operand> e.g., -1
                return operators[type(node.op)](eval_(node.operand))
            elif isinstance(node, ast.Call): # custom function
                if node.func.id in op_functions:
                    args =[eval_(arg) for arg in node.args]
                    return op_functions[node.func.id](*args)
            elif isinstance(node, ast.Subscript): # indexing or slicing
                value = eval_(node.value)
                if isinstance(node.slice, ast.Constant):
                    return value[node.slice.value]
                else:
                    return 0
            else:
                return 0

        result = eval_(ast.parse(value, mode='eval').body)

        if math.isnan(result):
            result = 0.0
        
        return (round(result), result, )

```
