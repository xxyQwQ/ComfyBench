---
tags:
- Conditioning
---

# Combine Conditionings
## Documentation
- Class name: `ImpactCombineConditionings`
- Category: `ImpactPack/Util`
- Output node: `False`

The ImpactCombineConditionings node is designed to merge multiple conditioning inputs into a single, unified conditioning output. This process facilitates the combination of various conditioning elements, enhancing the flexibility and capability of conditioning-based operations within the ImpactPack suite.
## Input types
### Required
- **`conditioning1`**
    - The primary conditioning input to be combined. It serves as the foundational element for the combination process, allowing for the integration of additional conditioning inputs to enrich the overall context.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `tuple`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - The result of combining the input conditionings. This unified conditioning encapsulates the merged context and attributes of the inputs, ready for further processing.
    - Python dtype: `tuple`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - SetNode



## Source code
```python
class CombineConditionings:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "conditioning1": ("CONDITIONING", ),
                     },
                }

    TOOLTIPS = {
        "input": {
            "conditioning1": "input conditionings. (Connecting to the input slot increases the number of additional slots.)",
        },
        "output": ("Combined conditioning", )
    }

    RETURN_TYPES = ("CONDITIONING", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    @staticmethod
    def doit(**kwargs):
        res = []
        for k, v in kwargs.items():
            res += v

        return (res, )

```
