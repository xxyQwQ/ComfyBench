# Batch Float Math (mtb)
## Documentation
- Class name: `Batch Float Math (mtb)`
- Category: `mtb/utils`
- Output node: `False`

Provides mathematical operations on batches of floating-point numbers, enabling complex computations and manipulations of data sets in a batch processing manner.
## Input types
### Required
- **`reverse`**
    - Determines the order in which the batches of floats are processed. When set to true, the batches are processed in reverse order, affecting the outcome of the operation.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`operation`**
    - Specifies the mathematical operation to be applied to the batches of floats, such as addition, subtraction, multiplication, division, power, or absolute value. This choice directly influences the result of the computation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`floats`**
    - Comfy dtype: `FLOATS`
    - The result of applying the specified mathematical operation on the input batch of floats.
    - Python dtype: `list[float]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class MTB_BatchFloatMath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reverse": ("BOOLEAN", {"default": False}),
                "operation": (
                    ["add", "sub", "mul", "div", "pow", "abs"],
                    {"default": "add"},
                ),
            }
        }

    RETURN_TYPES = ("FLOATS",)
    CATEGORY = "mtb/utils"
    FUNCTION = "execute"

    def execute(self, reverse: bool, operation: str, **kwargs: list[float]):
        res: list[float] = []
        vals = list(kwargs.values())

        if reverse:
            vals = vals[::-1]

        ref_count = len(vals[0])
        for v in vals:
            if len(v) != ref_count:
                raise ValueError(
                    f"All values must have the same length (current: {len(v)}, ref: {ref_count}"
                )

        match operation:
            case "add":
                for i in range(ref_count):
                    result = sum(v[i] for v in vals)
                    res.append(result)
            case "sub":
                for i in range(ref_count):
                    result = vals[0][i] - sum(v[i] for v in vals[1:])
                    res.append(result)
            case "mul":
                for i in range(ref_count):
                    result = vals[0][i] * vals[1][i]
                    res.append(result)
            case "div":
                for i in range(ref_count):
                    result = vals[0][i] / vals[1][i]
                    res.append(result)
            case "pow":
                for i in range(ref_count):
                    result: float = vals[0][i] ** vals[1][i]
                    res.append(result)
            case "abs":
                for i in range(ref_count):
                    result = abs(vals[0][i])
                    res.append(result)
            case _:
                log.info(f"For now this mode ({operation}) is not implemented")

        return (res,)

```
