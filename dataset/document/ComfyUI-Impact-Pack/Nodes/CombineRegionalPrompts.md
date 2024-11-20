---
tags:
- Concatenate
- PromptComposer
---

# CombineRegionalPrompts
## Documentation
- Class name: `CombineRegionalPrompts`
- Category: `ImpactPack/Regional`
- Output node: `False`

The CombineRegionalPrompts node is designed to aggregate multiple regional prompts into a single, unified set. This functionality is crucial for scenarios where distinct regional inputs need to be combined to form a comprehensive prompt set for further processing or generation tasks.
## Input types
### Required
- **`regional_prompts1`**
    - Represents the initial set of regional prompts to be combined. This input is essential for the node's operation as it forms the basis for the aggregation process.
    - Comfy dtype: `REGIONAL_PROMPTS`
    - Python dtype: `List[dict]`
## Output types
- **`regional_prompts`**
    - Comfy dtype: `REGIONAL_PROMPTS`
    - The combined set of regional prompts, ready for use in subsequent processing or generation tasks.
    - Python dtype: `List[dict]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class CombineRegionalPrompts:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "regional_prompts1": ("REGIONAL_PROMPTS", ),
                     },
                }

    TOOLTIPS = {
        "input": {
            "regional_prompts1": "input regional_prompts. (Connecting to the input slot increases the number of additional slots.)",
        },
        "output": ("Combined REGIONAL_PROMPTS", )
    }

    RETURN_TYPES = ("REGIONAL_PROMPTS", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Regional"

    @staticmethod
    def doit(**kwargs):
        res = []
        for k, v in kwargs.items():
            res += v

        return (res, )

```
