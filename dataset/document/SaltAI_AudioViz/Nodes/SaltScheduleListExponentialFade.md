---
tags:
- Scheduling
---

# Schedule Exponential Fade
## Documentation
- Class name: `SaltScheduleListExponentialFade`
- Category: `SALT/AudioViz/Scheduling/Filter`
- Output node: `False`

This node applies an exponential fade effect to a schedule list based on the specified fade type and strength, optionally starting from a given index. It's designed to modify the intensity or presence of elements in a schedule over time, creating smooth transitions either into, out of, or both into and out of a sequence of values.
## Input types
### Required
- **`schedule_list`**
    - The list of values to apply the exponential fade effect to. It represents the sequence of elements whose intensity will be modified over time.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
- **`fade_type`**
    - Specifies the type of fade to apply: 'in' for a gradual increase, 'out' for a gradual decrease, or 'in-and-out' for an increase followed by a decrease.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`strength`**
    - Determines the curve's steepness of the fade effect. Higher values result in a steeper curve.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`start_index`**
    - The index in the schedule list from which to start applying the fade effect. Allows for partial application of the fade.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`schedule_list`**
    - Comfy dtype: `LIST`
    - The modified schedule list after applying the exponential fade effect, with values adjusted according to the specified fade type and strength.
    - Python dtype: `List[float]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltScheduleListExponentialFade:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule_list": ("LIST", ),
                "fade_type": (["in", "out", "in-and-out"],),
                "strength": ("FLOAT", {"min": 0.01, "max": 10.0, "default": 1.0}),
            },
            "optional": {
                "start_index": ("INT", {"min": 0, "default": 0}),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("schedule_list", )
    FUNCTION = "exponential_fade"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling/Filter"

    def exponential_fade(self, schedule_list, fade_type, strength, start_index=0):
        length = len(schedule_list)
        faded_schedule = []
        
        for i in range(length):
            if i < start_index:
                faded_schedule.append(schedule_list[i])
                continue

            if fade_type in ["in", "out"]:
                t = (i - start_index) / max(1, (length - 1 - start_index))
                if fade_type == "in":
                    value = t ** strength
                else:
                    value = ((1 - t) ** strength)
            elif fade_type == "in-and-out":
                midpoint = (length - start_index) // 2 + start_index
                if i <= midpoint:
                    t = (i - start_index) / max(1, (midpoint - start_index))
                    value = t ** strength
                else:
                    t = (i - midpoint) / max(1, (length - 1 - midpoint))
                    value = ((1 - t) ** strength)

            faded_schedule.append(value)
        
        faded_schedule = [original * fade for original, fade in zip(schedule_list, faded_schedule)]

        return (faded_schedule, )

```
