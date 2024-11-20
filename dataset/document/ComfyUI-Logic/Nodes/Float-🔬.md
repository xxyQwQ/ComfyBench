---
tags:
- DataConversion
- DataTypeConversion
- Float
- FloatData
- NumericConversion
---

# Float
## Documentation
- Class name: `Float-🔬`
- Category: `Logic`
- Output node: `False`

The Float node is designed to process numerical floating-point values, allowing for precision adjustments and mathematical operations within a logic-based framework.
## Input types
### Required
- **`value`**
    - Represents the floating-point number to be processed or manipulated. Its importance lies in being the primary data that the node operates on, affecting the outcome of the node's execution.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`FLOAT`**
    - Comfy dtype: `FLOAT`
    - Outputs the processed or manipulated floating-point number, reflecting the result of the node's operations.
    - Python dtype: `float`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [SVD_img2vid_Conditioning](../../Comfy/Nodes/SVD_img2vid_Conditioning.md)
    - ezMath
    - [KSampler Adv. (Efficient)](../../efficiency-nodes-comfyui/Nodes/KSampler Adv. (Efficient).md)
    - Reroute
    - workflow/IP Adapter full bundle



## Source code
```python
class Float:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"value": ("FLOAT", {"default": 0, "step": 0.01})},
        }

    RETURN_TYPES = ("FLOAT",)

    RETURN_NAMES = ("FLOAT",)

    FUNCTION = "execute"

    CATEGORY = "Logic"

    def execute(self, value):
        return (value,)

```
