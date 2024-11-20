---
tags:
- DataTypeAgnostic
- Debugging
---

# AnyToAny
## Documentation
- Class name: `AnyToAny`
- Category: `Bmad/⚠️⚠️⚠️`
- Output node: `False`

The AnyToAny node is designed to dynamically evaluate a user-defined function on its input, allowing for a wide range of operations from simple transformations to complex computations. This flexibility enables users to apply arbitrary logic to their data, making it a versatile tool for custom data manipulation and analysis.
## Input types
### Required
- **`v`**
    - Represents the input value on which the user-defined function will be applied. Its flexibility allows for any type of data to be processed, making the node adaptable to various use cases.
    - Comfy dtype: `*`
    - Python dtype: `Any`
- **`function`**
    - A string representing a user-defined function to be evaluated on the input. This function is defined in a lambda expression format, enabling dynamic and complex operations on the input data.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`*`**
    - Comfy dtype: `*`
    - The output type is dynamic, determined by the result of the user-defined function applied to the input. This allows for a wide range of possible output types, reflecting the node's versatility.
    - Python dtype: `Any`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class AnyToAny:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "v": ("*",),
            "function": ("STRING", {"multiline": True, "default": ""}),
        }}

    FUNCTION = "eval_it"
    CATEGORY = f"{base_category_path}/⚠️⚠️⚠️"
    RETURN_TYPES = tuple(["*" for x in range(16)])

    def eval_it(self, v, function):
        function = prepare_text_for_eval(function)
        expression = eval(f"lambda v: {function}", {
            "__builtins__": {},
            "tuple": tuple, "list": list},
                          {})
        result = expression(v)
        return result

```
