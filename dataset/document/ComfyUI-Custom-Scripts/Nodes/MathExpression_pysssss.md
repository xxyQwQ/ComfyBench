---
tags:
- DataTypeConversion
- Math
- MathematicalExpressions
---

# Math Expression 🐍
## Documentation
- Class name: `MathExpression_pysssss`
- Category: `utils`
- Output node: `True`

The MathExpression node evaluates mathematical expressions dynamically, supporting basic arithmetic, comparisons, and logical operations. It can handle variables and function calls within the expression, offering a flexible way to compute results based on input parameters.
## Input types
### Required
- **`expression`**
    - The mathematical expression to be evaluated. Supports arithmetic, comparisons, logical operations, and function calls, with the ability to include variables 'a', 'b', and 'c'.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`a`**
    - An optional variable that can be used within the expression. Supports integers and floats.
    - Comfy dtype: `INT,FLOAT,IMAGE,LATENT`
    - Python dtype: `Union[int, float]`
- **`b`**
    - An optional variable that can be used within the expression. Supports integers and floats.
    - Comfy dtype: `INT,FLOAT,IMAGE,LATENT`
    - Python dtype: `Union[int, float]`
- **`c`**
    - An optional variable that can be used within the expression. Supports integers and floats.
    - Comfy dtype: `INT,FLOAT,IMAGE,LATENT`
    - Python dtype: `Union[int, float]`
## Output types
- **`int`**
    - Comfy dtype: `INT`
    - The integer part of the evaluated result, representing the outcome of the mathematical expression as an integer.
    - Python dtype: `int`
- **`float`**
    - Comfy dtype: `FLOAT`
    - The floating-point part of the evaluated result, representing the outcome of the mathematical expression as a float.
    - Python dtype: `float`
- **`ui`**
    - A user interface element that displays the evaluated result of the mathematical expression.
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class MathExpression:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expression": ("STRING", {"multiline": True, "dynamicPrompts": False, "pysssss.autocomplete": {
                    "words": autocompleteWords,
                    "separator": ""
                }}),
            },
            "optional": {
                "a": ("INT,FLOAT,IMAGE,LATENT", ),
                "b": ("INT,FLOAT,IMAGE,LATENT",),
                "c": ("INT,FLOAT,IMAGE,LATENT", ),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO",
                       "prompt": "PROMPT"},
        }

    RETURN_TYPES = ("INT", "FLOAT", )
    FUNCTION = "evaluate"
    CATEGORY = "utils"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(s, expression, **kwargs):
        if "random" in expression:
            return float("nan")
        return expression

    def get_widget_value(self, extra_pnginfo, prompt, node_name, widget_name):
        workflow = extra_pnginfo["workflow"] if "workflow" in extra_pnginfo else { "nodes": [] }
        node_id = None
        for node in workflow["nodes"]:
            name = node["type"]
            if "properties" in node:
                if "Node name for S&R" in node["properties"]:
                    name = node["properties"]["Node name for S&R"]
            if name == node_name:
                node_id = node["id"]
                break
            if "title" in node:
                name = node["title"]
            if name == node_name:
                node_id = node["id"]
                break
        if node_id is not None:
            values = prompt[str(node_id)]
            if "inputs" in values:
                if widget_name in values["inputs"]:
                    return values["inputs"][widget_name]
            raise NameError(f"Widget not found: {node_name}.{widget_name}")
        raise NameError(f"Node not found: {node_name}.{widget_name}")

    def get_size(self, target, property):
        if isinstance(target, dict) and "samples" in target:
            # Latent
            if property == "width":
                return target["samples"].shape[3] * 8
            return target["samples"].shape[2] * 8
        else:
            # Image
            if property == "width":
                return target.shape[2]
            return target.shape[1]

    def evaluate(self, expression, prompt, extra_pnginfo={}, a=None, b=None, c=None):
        expression = expression.replace('\n', ' ').replace('\r', '')
        node = ast.parse(expression, mode='eval').body

        lookup = {"a": a, "b": b, "c": c}

        def eval_op(node, l, r):
            l = eval_expr(l)
            r = eval_expr(r)
            l = l if isinstance(l, int) else float(l)
            r = r if isinstance(r, int) else float(r)
            return operators[type(node.op)](l, r)

        def eval_expr(node):
            if isinstance(node, ast.Constant) or isinstance(node, ast.Num):
                return node.n
            elif isinstance(node, ast.BinOp):
                return eval_op(node, node.left, node.right)
            elif isinstance(node, ast.BoolOp):
                return eval_op(node, node.values[0], node.values[1])
            elif isinstance(node, ast.UnaryOp):
                return operators[type(node.op)](eval_expr(node.operand))
            elif isinstance(node, ast.Attribute):
                if node.value.id in lookup:
                    if node.attr == "width" or node.attr == "height":
                        return self.get_size(lookup[node.value.id], node.attr)

                return self.get_widget_value(extra_pnginfo, prompt, node.value.id, node.attr)
            elif isinstance(node, ast.Name):
                if node.id in lookup:
                    val = lookup[node.id]
                    if isinstance(val, (int, float, complex)):
                        return val
                    else:
                        raise TypeError(
                            f"Compex types (LATENT/IMAGE) need to reference their width/height, e.g. {node.id}.width")
                raise NameError(f"Name not found: {node.id}")
            elif isinstance(node, ast.Call):
                if node.func.id in functions:
                    fn = functions[node.func.id]
                    l = len(node.args)
                    if l < fn["args"][0] or (fn["args"][1] is not None and l > fn["args"][1]):
                        if fn["args"][1] is None:
                            toErr = " or more"
                        else:
                            toErr = f" to {fn['args'][1]}"
                        raise SyntaxError(
                            f"Invalid function call: {node.func.id} requires {fn['args'][0]}{toErr} arguments")
                    args = []
                    for arg in node.args:
                        args.append(eval_expr(arg))
                    return fn["call"](*args)
                raise NameError(f"Invalid function call: {node.func.id}")
            elif isinstance(node, ast.Compare):
                l = eval_expr(node.left)
                r = eval_expr(node.comparators[0])
                if isinstance(node.ops[0], ast.Eq):
                    return 1 if l == r else 0
                if isinstance(node.ops[0], ast.NotEq):
                    return 1 if l != r else 0
                if isinstance(node.ops[0], ast.Gt):
                    return 1 if l > r else 0
                if isinstance(node.ops[0], ast.GtE):
                    return 1 if l >= r else 0
                if isinstance(node.ops[0], ast.Lt):
                    return 1 if l < r else 0
                if isinstance(node.ops[0], ast.LtE):
                    return 1 if l <= r else 0
                raise NotImplementedError(
                    "Operator " + node.ops[0].__class__.__name__ + " not supported.")
            else:
                raise TypeError(node)

        r = eval_expr(node)
        return {"ui": {"value": [r]}, "result": (int(r), float(r),)}

```
