---
tags:
- Scheduling
---

# Schedule Split
## Documentation
- Class name: `SaltScheduleSplit`
- Category: `SALT/AudioViz/Scheduling`
- Output node: `False`

This node is designed to divide a given schedule list into two parts based on a specified index, facilitating the manipulation of schedule sequences by separating them into distinct segments.
## Input types
### Required
- **`schedule_list`**
    - The list of scheduled items to be split. This parameter is crucial for determining the point of division within the schedule.
    - Comfy dtype: `LIST`
    - Python dtype: `List[Any]`
- **`split_index`**
    - The index at which the schedule list is to be split. This index determines the boundary between the two resulting lists, playing a key role in the segmentation process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`schedule_list_a`**
    - Comfy dtype: `LIST`
    - The first part of the schedule list, containing all items before the split index.
    - Python dtype: `List[Any]`
- **`schedule_list_b`**
    - Comfy dtype: `LIST`
    - The second part of the schedule list, containing all items from the split index onwards.
    - Python dtype: `List[Any]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltScheduleSplit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule_list": ("LIST", ),
                "split_index": ("INT", {"min": 0}),
            },
        }

    RETURN_TYPES = ("LIST", "LIST")
    RETURN_NAMES = ("schedule_list_a", "schedule_list_b")
    FUNCTION = "split"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling"

    def split(self, schedule_list, split_index):
        if split_index >= len(schedule_list) or split_index < 0:
            raise ValueError("Schedule split_index must be within the range of the schedule_list.")
        first_part = schedule_list[:split_index]
        second_part = schedule_list[split_index:]
        return (first_part, second_part)

```
