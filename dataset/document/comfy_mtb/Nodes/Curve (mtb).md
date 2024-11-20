---
tags:
- AnimationScheduling
- Curve
---

# Curve (mtb)
## Documentation
- Class name: `Curve (mtb)`
- Category: `mtb/curve`
- Output node: `False`

The MTB_Curve node is designed to process FLOAT_CURVE inputs, maintaining the integrity of the curve's structure. It primarily serves to ensure that curve data, whether received as a string or in a structured format, is correctly serialized or deserialized for further processing within the system.
## Input types
### Required
- **`curve`**
    - The 'curve' parameter represents the FLOAT_CURVE input that the node operates on. It is essential for the node's functionality, as it dictates the curve data that will be serialized or deserialized for further processing.
    - Comfy dtype: `FLOAT_CURVE`
    - Python dtype: `Union[str, Dict]`
## Output types
- **`float_curve`**
    - Comfy dtype: `FLOAT_CURVE`
    - The output is a float_curve, which is either the original curve data or its serialized string representation, depending on the input's format.
    - Python dtype: `Union[str, Dict]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class MTB_Curve:
    """A basic FLOAT_CURVE input node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "curve": ("FLOAT_CURVE",),
            },
        }

    RETURN_TYPES = ("FLOAT_CURVE",)
    FUNCTION = "do_curve"

    CATEGORY = "mtb/curve"

    def do_curve(self, curve):
        log.debug(f"Curve: {curve}")
        return (curve,)

```
