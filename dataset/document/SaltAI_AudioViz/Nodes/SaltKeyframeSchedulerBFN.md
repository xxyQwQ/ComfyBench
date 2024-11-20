---
tags:
- AnimationScheduling
- Scheduling
---

# Keyframe Scheduler (BIG)
## Documentation
- Class name: `SaltKeyframeSchedulerBFN`
- Category: `SALT/AudioViz/Scheduling`
- Output node: `False`

This node is designed for scheduling keyframes in a customizable manner, allowing for the application of easing modes and the inclusion of optional variables to fine-tune the animation process. It provides a flexible framework for generating animation schedules based on keyframe data, easing specifications, and additional parameters, facilitating the creation of dynamic and nuanced animations.
## Input types
### Required
- **`keyframe_schedule`**
    - Specifies the schedule of keyframes to be animated. It is crucial for defining the sequence and timing of the animation, impacting the overall flow and rhythm of the generated animation.
    - Comfy dtype: `STRING`
    - Python dtype: `list`
- **`easing_mode`**
    - Determines the easing function to be applied to the keyframe animation, influencing the acceleration and deceleration of the animation between keyframes. This choice significantly affects the visual smoothness and dynamics of the animation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`end_frame`**
    - Optional parameter that specifies the last frame of the animation, allowing for the animation duration to be limited or extended as needed.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`ndigits`**
    - Optional parameter that defines the precision of the computed keyframe values, enabling control over the granularity of the animation's timing.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`a`**
    - An optional variable that can be used to introduce custom behavior or adjustments to the animation schedule, enhancing flexibility.
    - Comfy dtype: `*`
    - Python dtype: `Union[int, float, bool, list]`
- **`b`**
    - Another optional variable similar to 'a', providing additional capacity for custom modifications to the animation schedule.
    - Comfy dtype: `*`
    - Python dtype: `Union[int, float, bool, list]`
- **`c`**
    - An optional variable that can be introduced for further customization of the animation schedule.
    - Comfy dtype: `*`
    - Python dtype: `list`
- **`d`**
    - Similar to 'c', this optional variable allows for additional adjustments to the animation schedule.
    - Comfy dtype: `*`
    - Python dtype: `list`
- **`e`**
    - Provides further flexibility in customizing the animation schedule through this optional variable.
    - Comfy dtype: `*`
    - Python dtype: `list`
- **`f`**
    - An optional variable for additional customization options within the animation schedule.
    - Comfy dtype: `*`
    - Python dtype: `list`
- **`g`**
    - Allows for further adjustments and customization of the animation schedule.
    - Comfy dtype: `*`
    - Python dtype: `list`
- **`h`**
    - An optional variable for extending the customization capabilities of the animation schedule.
    - Comfy dtype: `*`
    - Python dtype: `list`
## Output types
- **`schedule_list`**
    - Comfy dtype: `LIST`
    - The generated animation schedule, comprising keyframe timings and values adjusted according to the specified easing mode and optional parameters. This output is essential for implementing the planned animation.
    - Python dtype: `list`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltKeyframeSchedulerBFN:
    @classmethod
    def INPUT_TYPES(cls):
        easing_funcs = list(easing_functions.keys())
        easing_funcs.insert(0, "None")
        return {
            "required": {
                "keyframe_schedule": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "easing_mode": (easing_funcs, )
            },
            "optional": {
                "end_frame": ("INT", {"min": 0}),
                "ndigits": ("INT", {"min": 1, "max": 12, "default": 5}),
                "a": (WILDCARD, {}),
                "b": (WILDCARD, {}),
                "c": (WILDCARD, {}),
                "d": (WILDCARD, {}),
                "e": (WILDCARD, {}),
                "f": (WILDCARD, {}),
                "g": (WILDCARD, {}),
                "h": (WILDCARD, {}),
            }
        }

    RETURN_TYPES = ("LIST", )
    RETURN_NAMES = ("schedule_list", )

    FUNCTION = "keyframe_schedule"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling"

    def keyframe_schedule(self, keyframe_schedule, easing_mode, end_frame=0, ndigits=2, a=[0], b=[0], c=[0], d=[0], e=[0], f=[0], g=[0], h=[0]):
        logger.info("Received keyframe_schedule:", keyframe_schedule)
        custom_vars = {"a": a, "b": b, "c": c, "d": d, "e": e, "f": f, "g": g, "h": h}
        scheduler = KeyframeScheduler(end_frame=end_frame, custom_vars=custom_vars)
        schedule = scheduler.generate_schedule(keyframe_schedule, easing_mode=easing_mode, ndigits=ndigits)
        return (schedule, )

```
