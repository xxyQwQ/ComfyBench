---
tags:
- AnimationScheduling
- Scheduling
---

# Keyframe Scheduler
## Documentation
- Class name: `SaltKeyframeScheduler`
- Category: `SALT/AudioViz/Scheduling`
- Output node: `False`

The SaltKeyframeScheduler node is designed for generating a schedule based on keyframes, applying easing functions to interpolate between these keyframes over time. It allows for the customization of the schedule through various parameters, including the selection of easing modes and the adjustment of frame rates, to create dynamic and smooth transitions in animations or visualizations.
## Input types
### Required
- **`keyframe_schedule`**
    - Defines the sequence of keyframes to be scheduled. Each keyframe represents a significant point in the animation or visualization that the schedule aims to reach, serving as the foundation for the generated schedule.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`easing_mode`**
    - Specifies the easing function to be applied between keyframes, determining the interpolation method and thus affecting the smoothness and dynamics of the transition.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`end_frame`**
    - The last frame to be considered in the schedule, allowing for the specification of a custom range within the keyframe sequence.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`ndigits`**
    - Determines the precision of the computed values in the schedule, affecting the granularity of the interpolation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`a`**
    - Custom variable 'a' that can be used for additional control over the scheduling, providing flexibility in the scheduling logic.
    - Comfy dtype: `*`
    - Python dtype: `Union[int, float, bool, list]`
- **`b`**
    - Custom variable 'b', similar to 'a', offers further customization options for the scheduling process, enhancing the node's versatility.
    - Comfy dtype: `*`
    - Python dtype: `Union[int, float, bool, list]`
## Output types
- **`schedule_list`**
    - Comfy dtype: `LIST`
    - The resulting list of scheduled values, interpolated according to the specified easing mode and other parameters, ready for use in animations or visualizations.
    - Python dtype: `List[float]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltKeyframeScheduler:
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
                "b": (WILDCARD, {})
            }
        }

    RETURN_TYPES = ("LIST", )
    RETURN_NAMES = ("schedule_list", )

    FUNCTION = "keyframe_schedule"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling"

    def keyframe_schedule(self, keyframe_schedule, easing_mode, end_frame=0, ndigits=2, a=None, b=None):
        if a and not isinstance(a, (int, float, bool, list)):
            raise ValueError("`a` is not a valid int, float, boolean, or schedule_list")
        if b and not isinstance(b, (int, float, bool, list)):
            raise ValueError("`b` is not a valid int, float, or boolean, or schedule_list")
        
        custom_vars = {}
        if a:
            custom_vars['a'] = a
        if b:
            custom_vars['b'] = b
        
        scheduler = KeyframeScheduler(end_frame=end_frame, custom_vars=custom_vars)
        schedule = scheduler.generate_schedule(keyframe_schedule, easing_mode=easing_mode, ndigits=ndigits)
        return (schedule, )

```
