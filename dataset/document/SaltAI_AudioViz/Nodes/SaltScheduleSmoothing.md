---
tags:
- Scheduling
---

# Schedule Smoothing
## Documentation
- Class name: `SaltScheduleSmoothing`
- Category: `SALT/AudioViz/Scheduling/Filter`
- Output node: `False`

Provides a method to smooth a given schedule list by applying a smoothing factor to each element, effectively filtering out rapid changes and creating a more gradual transition between values.
## Input types
### Required
- **`schedule_list`**
    - The list of schedule values to be smoothed. This list represents a sequence of numerical values that will undergo smoothing to reduce abrupt changes.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
- **`smoothing_factor`**
    - A factor determining the degree of smoothing applied to the schedule list. A higher value results in a smoother transition between schedule values.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`schedule_list`**
    - Comfy dtype: `LIST`
    - The smoothed list of schedule values, where each value has been adjusted according to the smoothing factor to create a gradual transition.
    - Python dtype: `List[float]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltScheduleSmoothing:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule_list": ("LIST", ),
                "smoothing_factor": ("FLOAT", {"min": 0.0, "max": 1.0, "default": 0.5}),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("schedule_list", )
    FUNCTION = "smooth"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling/Filter"

    def smooth(self, schedule_list, smoothing_factor):
        smoothed_schedule = schedule_list[:]
        for i in range(1, len(schedule_list)):
            smoothed_schedule[i] = smoothed_schedule[i-1] * (1 - smoothing_factor) + schedule_list[i] * smoothing_factor
        return (smoothed_schedule, )

```
