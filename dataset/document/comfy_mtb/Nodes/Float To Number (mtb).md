# Float To Number (mtb)
## Documentation
- Class name: `Float To Number (mtb)`
- Category: `mtb/number`
- Output node: `False`

This node is designed to convert floating-point numbers into a more general number format, facilitating compatibility and data exchange within the WAS Suite ecosystem.
## Input types
### Required
- **`float`**
    - The floating-point number to be converted into a general number format. This input is crucial for the node's operation as it directly influences the output number.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`number`**
    - Comfy dtype: `NUMBER`
    - The converted general number format from the input floating-point number.
    - Python dtype: `float`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class MTB_FloatToNumber:
    """Node addon for the WAS Suite. Converts a "comfy" FLOAT to a NUMBER."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "float": (
                    "FLOAT",
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
    FUNCTION = "float_to_number"
    CATEGORY = "mtb/number"

    def float_to_number(self, float):
        return (float,)

```
