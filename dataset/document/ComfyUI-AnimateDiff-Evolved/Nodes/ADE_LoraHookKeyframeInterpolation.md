---
tags:
- LoRA
---

# LoRA Hook Keyframes Interpolation 🎭🅐🅓
## Documentation
- Class name: `ADE_LoraHookKeyframeInterpolation`
- Category: `Animate Diff 🎭🅐🅓/conditioning/schedule lora hooks`
- Output node: `False`

This node is designed to create interpolated LoRA hook keyframes for animation conditioning, allowing for the dynamic adjustment of model behavior over a sequence of frames. It leverages interpolation techniques to generate a series of keyframes based on specified start and end percentages, strengths, and intervals, facilitating smooth transitions and precise control over the animation process.
## Input types
### Required
- **`start_percent`**
    - Defines the starting percentage of the animation sequence where the first keyframe is to be applied, influencing the initial point of model behavior adjustment.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`end_percent`**
    - Specifies the ending percentage of the animation sequence, marking where the last keyframe's effect concludes, thus delineating the span of the interpolation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`strength_start`**
    - Sets the initial strength value for the first keyframe, determining the starting intensity of the model's behavior modification.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`strength_end`**
    - Determines the final strength value for the last keyframe, defining the ending intensity of the model's behavior adjustment.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`interpolation`**
    - Chooses the method of interpolation for generating intermediate strength values between keyframes, affecting the smoothness and progression of the animation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `InterpolationMethod`
- **`intervals`**
    - Specifies the number of intervals (or keyframes) to generate between the start and end points, impacting the granularity of the animation conditioning.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`print_keyframes`**
    - A flag to enable or disable printing of the generated keyframes for debugging or verification purposes.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`prev_hook_kf`**
    - An optional parameter allowing for the inclusion of previously defined LoRA hook keyframes, enabling the extension or modification of an existing keyframe sequence.
    - Comfy dtype: `LORA_HOOK_KEYFRAMES`
    - Python dtype: `LoraHookKeyframeGroup`
## Output types
- **`HOOK_KF`**
    - Comfy dtype: `LORA_HOOK_KEYFRAMES`
    - The resulting sequence of interpolated LoRA hook keyframes, ready for use in animation conditioning.
    - Python dtype: `LoraHookKeyframeGroup`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class CreateLoraHookKeyframeInterpolation:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "strength_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "strength_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "interpolation": (InterpolationMethod._LIST, ),
                "intervals": ("INT", {"default": 5, "min": 2, "max": 100, "step": 1}),
                "print_keyframes": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "prev_hook_kf": ("LORA_HOOK_KEYFRAMES",),
            }
        }
    
    RETURN_TYPES = ("LORA_HOOK_KEYFRAMES",)
    RETURN_NAMES = ("HOOK_KF",)
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning/schedule lora hooks"
    FUNCTION = "create_hook_keyframes"

    def create_hook_keyframes(self,
                              start_percent: float, end_percent: float,
                              strength_start: float, strength_end: float, interpolation: str, intervals: int,
                              prev_hook_kf: LoraHookKeyframeGroup=None, print_keyframes=False):
        if prev_hook_kf:
            prev_hook_kf = prev_hook_kf.clone()
        else:
            prev_hook_kf = LoraHookKeyframeGroup()
        percents = InterpolationMethod.get_weights(num_from=start_percent, num_to=end_percent, length=intervals, method=InterpolationMethod.LINEAR)
        strengths = InterpolationMethod.get_weights(num_from=strength_start, num_to=strength_end, length=intervals, method=interpolation)
        
        is_first = True
        for percent, strength in zip(percents, strengths):
            guarantee_steps = 0
            if is_first:
                guarantee_steps = 1
                is_first = False
            prev_hook_kf.add(LoraHookKeyframe(strength=strength, start_percent=percent, guarantee_steps=guarantee_steps))
            if print_keyframes:
                logger.info(f"LoraHookKeyframe - start_percent:{percent} = {strength}")
        return (prev_hook_kf,)

```
