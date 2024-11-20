---
tags:
- DataConversion
- DataTypeConversion
- Integer
- NumericConversion
---

# Int
## Documentation
- Class name: `easy int`
- Category: `EasyUse/Logic/Type`
- Output node: `False`

The `easy int` node is designed to simplify the process of generating and manipulating integer values within a user's workflow. It abstracts away the complexities associated with integer operations, offering a straightforward interface for creating, configuring, and utilizing integers in various computational tasks.
## Input types
### Required
- **`value`**
    - Represents an integer value that can be configured by the user. It serves as the primary input for the node, allowing for the specification of integer-based parameters or values that influence the node's operation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`int`**
    - Comfy dtype: `INT`
    - Outputs an integer value as a result of the node's processing, reflecting the manipulation or generation of integer values based on the input provided.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class Int:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"value": ("INT", {"default": 0, "min": -999999, "max": 999999,})},
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "execute"
    CATEGORY = "EasyUse/Logic/Type"

    def execute(self, value):
        return (value,)

```
