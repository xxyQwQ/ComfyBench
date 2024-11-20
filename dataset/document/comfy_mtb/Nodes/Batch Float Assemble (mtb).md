---
tags:
- Batch
- FloatData
---

# Batch Float Assemble (mtb)
## Documentation
- Class name: `Batch Float Assemble (mtb)`
- Category: `mtb/batch`
- Output node: `False`

The MTB_BatchFloatAssemble node is designed to aggregate multiple batches of floating-point numbers into a single, unified batch. This process allows for the efficient combination of data streams, optionally reversing the order of the batches before assembly, to suit various data processing and analysis needs.
## Input types
### Required
- **`reverse`**
    - Determines whether the batches of floats should be assembled in reverse order. Setting this to true reverses the order, which can be useful for certain data processing scenarios where the sequence of data matters.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`floats`**
    - Comfy dtype: `FLOATS`
    - The output is a single, consolidated list of floating-point numbers, assembled from the input batches. This unified batch can be used for further processing or analysis.
    - Python dtype: `list[float]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class MTB_BatchFloatAssemble:
    """Assembles mutiple batches of floats into a single stream (batch)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"reverse": ("BOOLEAN", {"default": False})}}

    RETURN_TYPES = ("FLOATS",)
    CATEGORY = "mtb/batch"
    FUNCTION = "assemble_floats"

    def assemble_floats(self, reverse: bool, **kwargs: list[float]):
        res: list[float] = []

        if reverse:
            for x in reversed(kwargs.values()):
                if x:
                    res += x
        else:
            for x in kwargs.values():
                if x:
                    res += x

        return (res,)

```
