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
- Class name: `easy float`
- Category: `EasyUse/Logic/Type`
- Output node: `False`

The `easy float` node provides a simplified interface for working with floating-point values within a user-defined range. It allows for the adjustment of parameters such as weight, strength, and other numerical values with precision, facilitating fine-tuned control over these settings in various applications.
## Input types
### Required
- **`value`**
    - Specifies the floating-point value to be processed. This parameter allows users to input a numerical value for manipulation or evaluation within the node's functionality.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`float`**
    - Comfy dtype: `FLOAT`
    - Returns the processed floating-point value, potentially adjusted or evaluated based on the node's logic.
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
            "required": {"value": ("FLOAT", {"default": 0, "step": 0.01, "min": -999999, "max": 999999,})},
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    FUNCTION = "execute"
    CATEGORY = "EasyUse/Logic/Type"

    def execute(self, value):
        return (value,)

```
