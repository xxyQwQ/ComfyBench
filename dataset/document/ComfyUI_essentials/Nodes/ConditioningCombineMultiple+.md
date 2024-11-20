---
tags:
- Conditioning
---

# 🔧 Cond Combine Multiple
## Documentation
- Class name: `ConditioningCombineMultiple+`
- Category: `essentials/conditioning`
- Output node: `False`

This node is designed to combine multiple conditioning inputs into a single conditioning output. It allows for the flexible integration of up to five separate conditioning inputs, making it ideal for scenarios where multiple conditioning factors need to be merged to guide the generation process effectively.
## Input types
### Required
- **`conditioning_i`**
    - unknown
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `unknown`
### Optional
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - The resulting combined conditioning output, integrating up to five separate conditioning inputs into a single, cohesive conditioning factor.
    - Python dtype: `tuple`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ConditioningCombineMultiple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning_1": ("CONDITIONING",),
                "conditioning_2": ("CONDITIONING",),
            }, "optional": {
                "conditioning_3": ("CONDITIONING",),
                "conditioning_4": ("CONDITIONING",),
                "conditioning_5": ("CONDITIONING",),
            },
        }
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "execute"
    CATEGORY = "essentials/conditioning"

    def execute(self, conditioning_1, conditioning_2, conditioning_3=None, conditioning_4=None, conditioning_5=None):
        c = conditioning_1 + conditioning_2

        if conditioning_3 is not None:
            c += conditioning_3
        if conditioning_4 is not None:
            c += conditioning_4
        if conditioning_5 is not None:
            c += conditioning_5

        return (c,)

```
