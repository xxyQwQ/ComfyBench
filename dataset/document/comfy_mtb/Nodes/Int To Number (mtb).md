# Int To Number (mtb)
## Documentation
- Class name: `Int To Number (mtb)`
- Category: `mtb/number`
- Output node: `False`

Converts an integer to a number within a specified range, ensuring compatibility with the WAS Suite's numerical data types.
## Input types
### Required
- **`int`**
    - The integer input to be converted to a number. This parameter is crucial for defining the exact value that will undergo conversion.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`number`**
    - Comfy dtype: `NUMBER`
    - The output number, directly corresponding to the input integer.
    - Python dtype: `float`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [CM_NumberToFloat](../../ComfyMath/Nodes/CM_NumberToFloat.md)



## Source code
```python
class MTB_IntToNumber:
    """Node addon for the WAS Suite. Converts a "comfy" INT to a NUMBER."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "int": (
                    "INT",
                    {
                        "default": 0,
                        "min": -1e9,
                        "max": 1e9,
                        "step": 1,
                        "forceInput": True,
                    },
                ),
            }
        }

    RETURN_TYPES = ("NUMBER",)
    FUNCTION = "int_to_number"
    CATEGORY = "mtb/number"

    def int_to_number(self, int):
        return (int,)

```
