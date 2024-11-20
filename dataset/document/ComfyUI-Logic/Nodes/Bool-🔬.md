---
tags:
- BooleanLogic
- ConditionalSelection
---

# Bool
## Documentation
- Class name: `Bool-🔬`
- Category: `Logic`
- Output node: `False`

The Bool node is designed to process boolean inputs, performing logical operations based on the provided boolean value.
## Input types
### Required
- **`value`**
    - Represents the boolean input to the node, determining the logical operation to be performed.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`BOOLEAN`**
    - Comfy dtype: `BOOLEAN`
    - Outputs the processed boolean value, reflecting the result of the logical operation.
    - Python dtype: `bool`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class Bool:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"value": ("BOOLEAN", {"default": False})},
        }

    RETURN_TYPES = ("BOOLEAN",)

    RETURN_NAMES = ("BOOLEAN",)

    FUNCTION = "execute"

    CATEGORY = "Logic"

    def execute(self, value):
        return (value,)

```
