---
tags:
- DataConversion
- DataTypeConversion
- FloatData
- FloatList
- NumericConversion
---

# Float To Floats (mtb)
## Documentation
- Class name: `Float To Floats (mtb)`
- Category: `mtb/utils`
- Output node: `False`

This node serves as a conversion utility designed to transform a single floating-point number into a tuple of floating-point numbers, facilitating compatibility with other extensions that use a list format to represent multiple float values.
## Input types
### Required
- **`float`**
    - The single floating-point number to be converted into a tuple. This input is essential for the node's operation, enabling the conversion from a singular float value to a list format.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`floats`**
    - Comfy dtype: `FLOATS`
    - The tuple of floating-point numbers resulting from the conversion of the input single float. This output facilitates compatibility with other extensions requiring a list format.
    - Python dtype: `Tuple[float]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class MTB_FloatToFloats:
    """Conversion utility for compatibility with other extensions (AD, IPA, Fitz are using FLOAT to represent list of floats.)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "float": ("FLOAT", {"default": 0.0, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("FLOATS",)
    RETURN_NAMES = ("floats",)
    CATEGORY = "mtb/utils"
    FUNCTION = "convert"

    def convert(self, float: float):
        return (float,)

```
