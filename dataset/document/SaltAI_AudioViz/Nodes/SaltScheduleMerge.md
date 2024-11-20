---
tags:
- Scheduling
---

# Schedule Merge
## Documentation
- Class name: `SaltScheduleMerge`
- Category: `SALT/AudioViz/Scheduling/Filter`
- Output node: `False`

The SaltScheduleMerge node is designed to combine two schedule lists into a single list, effectively appending the second list to the end of the first. This functionality is useful in scenarios where sequential scheduling tasks need to be merged for streamlined processing or analysis.
## Input types
### Required
- **`schedule_list_a`**
    - Represents the first list of scheduled items to be merged. It serves as the initial part of the combined schedule list.
    - Comfy dtype: `LIST`
    - Python dtype: `List[Any]`
- **`schedule_list_b`**
    - Represents the second list of scheduled items to be appended to the end of the first list, completing the merged schedule list.
    - Comfy dtype: `LIST`
    - Python dtype: `List[Any]`
## Output types
- **`schedule_list`**
    - Comfy dtype: `LIST`
    - The combined list resulting from appending the second schedule list to the first, providing a unified sequence of scheduled items.
    - Python dtype: `List[Any]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltScheduleMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule_list_a": ("LIST", ),
                "schedule_list_b": ("LIST", ),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("schedule_list", )
    FUNCTION = "append"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling/Filter"

    def append(self, schedule_list_a, schedule_list_b):
        appended_list = schedule_list_a + schedule_list_b
        return (appended_list, )

```
