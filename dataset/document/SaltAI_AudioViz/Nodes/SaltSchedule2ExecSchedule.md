---
tags:
- Scheduling
---

# Convert Schedule to Iterative Execution List
## Documentation
- Class name: `SaltSchedule2ExecSchedule`
- Category: `SALT/AudioViz/Scheduling`
- Output node: `False`

This node is designed to transform a schedule list into an iterative execution list, facilitating the conversion of a sequence of elements for iterative processing.
## Input types
### Required
- **`list_input`**
    - The input list to be converted for iterative execution. It plays a crucial role in determining the structure and content of the output list.
    - Comfy dtype: `LIST`
    - Python dtype: `List[Any]`
## Output types
- **`float`**
    - Comfy dtype: `FLOAT`
    - The transformed list, now suitable for iterative execution, maintaining the original list's structure but enabling a different mode of processing.
    - Python dtype: `List[float]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltSchedule2ExecSchedule:
    """
        Converts a list to a list output (iterative execution list)
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list_input": ("LIST", {}), 
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "convert"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling"

    def convert(self, list_input):
        return (list_input, )

```
