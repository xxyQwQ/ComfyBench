---
tags:
- Scheduling
---

# Schedule Numeric Clamp
## Documentation
- Class name: `SaltListClamp`
- Category: `SALT/AudioViz/Scheduling/Filter`
- Output node: `False`

The SaltListClamp node is designed to adjust the values in a schedule list so that they fall within a specified minimum and maximum range. This process, known as clamping, ensures that all values in the list adhere to defined boundaries, enhancing the control over the data's range for further processing or visualization.
## Input types
### Required
- **`schedule_list`**
    - The list of numerical values to be clamped. It serves as the primary data input for the node, determining the values that will be adjusted to fall within the specified range.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
- **`min_value`**
    - The minimum allowable value in the clamped list. This parameter sets the lower boundary for the values, ensuring none fall below it.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`max_value`**
    - The maximum allowable value in the clamped list. This parameter sets the upper boundary for the values, ensuring none exceed it.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`schedule_list`**
    - Comfy dtype: `LIST`
    - The list of clamped values, adjusted to ensure that each element falls within the specified minimum and maximum range.
    - Python dtype: `List[float]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltListClamp:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule_list": ("LIST", ),
                "min_value": ("FLOAT", {"step": 0.01}),
                "max_value": ("FLOAT", {"step": 0.01})
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("schedule_list",)
    FUNCTION = "clamp_values"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling/Filter"

    def clamp_values(self, schedule_list, min_value, max_value):
        if min_value > max_value:
            raise ValueError("Schedules min_value cannot be greater than max_value.")

        clamped_list = [max(min(x, max_value), min_value) for x in schedule_list]

        return (clamped_list, )

```
