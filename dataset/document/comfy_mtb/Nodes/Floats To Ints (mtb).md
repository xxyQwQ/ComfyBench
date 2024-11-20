---
tags:
- DataConversion
- DataTypeConversion
- NumericConversion
---

# Floats To Ints (mtb)
## Documentation
- Class name: `Floats To Ints (mtb)`
- Category: `mtb/utils`
- Output node: `False`

Provides a conversion utility specifically designed for frame interpolation by converting a list of float values into integers. This node is essential for scenarios requiring integer representations of floating-point numbers, such as pixel values in image processing or discrete steps in animations.
## Input types
### Required
- **`floats`**
    - Accepts a list of floating-point numbers to be converted into integers. This input is crucial for the conversion process, enabling the node to process and transform each float value into its integer counterpart.
    - Comfy dtype: `FLOATS`
    - Python dtype: `list[float]`
## Output types
- **`ints`**
    - Comfy dtype: `INTS`
    - Outputs a list of integers, each derived from the corresponding float input. This is useful for applications that require integer values for further processing or analysis.
    - Python dtype: `list[int]`
- **`int`**
    - Comfy dtype: `INT`
    - Additionally provides a single integer value, typically representing a specific or aggregated result derived from the input floats.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class MTB_FloatsToInts:
    """Conversion utility for compatibility with frame interpolation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "floats": ("FLOATS", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("INTS", "INT")
    CATEGORY = "mtb/utils"
    FUNCTION = "convert"

    def convert(self, floats: list[float]):
        vals = [int(x) for x in floats]
        return (vals, vals)

```
