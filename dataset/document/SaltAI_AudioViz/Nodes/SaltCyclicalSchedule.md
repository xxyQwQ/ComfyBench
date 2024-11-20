---
tags:
- Scheduling
---

# Schedule Cyclical Loop
## Documentation
- Class name: `SaltCyclicalSchedule`
- Category: `SALT/AudioViz/Scheduling/Filter`
- Output node: `False`

This node is designed to generate a cyclical schedule from a given list of schedule items. It allows for the repetition of a specified segment within the schedule list, optionally incorporating a ping-pong effect for a mirrored repetition pattern. The node aims to facilitate the creation of repetitive, cyclical patterns in scheduling tasks, enhancing flexibility and creativity in schedule design.
## Input types
### Required
- **`schedule_list`**
    - The list of schedule items to be processed. This list serves as the basis for generating the cyclical schedule, with specific segments being repeated according to the other parameters.
    - Comfy dtype: `LIST`
    - Python dtype: `List[Any]`
- **`start_index`**
    - The starting index of the segment within the schedule list to be repeated. This determines the beginning of the pattern that will be cyclically replicated.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`end_index`**
    - The ending index of the segment within the schedule list to be repeated. This marks the end of the pattern that will undergo cyclical replication.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`repetitions`**
    - The number of times the specified segment is to be repeated in the cyclical schedule. This controls the length and repetition rate of the cyclical pattern.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`ping_pong`**
    - A boolean flag that, when set to True, causes the repeated segment to be mirrored in a ping-pong fashion, enhancing the cyclical pattern with a reverse repetition.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`schedule_list`**
    - Comfy dtype: `LIST`
    - The resulting cyclical schedule list, composed of the original schedule items arranged according to the specified cyclical pattern.
    - Python dtype: `List[Any]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltCyclicalSchedule:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule_list": ("LIST", ),
                "start_index": ("INT", {"min": 0}),
                "end_index": ("INT", {"min": 0}),
                "repetitions": ("INT", {"min": 1}),
            },
            "optional": {
                "ping_pong": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("schedule_list",)
    FUNCTION = "generate_cyclical"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling/Filter"

    def generate_cyclical(self, schedule_list, start_index, end_index, repetitions, ping_pong=False):
        if end_index < start_index:
            raise ValueError("Schedule end_index must be greater than or equal to start_index.")
        
        if end_index >= len(schedule_list):
            raise ValueError("Schedule end_index must be within the range of the schedule_list.")
        
        loop_segment = schedule_list[start_index:end_index + 1]
        
        cyclical_schedule = []
        for _ in range(repetitions):
            cyclical_schedule.extend(loop_segment)
            if ping_pong:
                cyclical_schedule.extend(loop_segment[-2:0:-1])
        
        return (cyclical_schedule,)

```
