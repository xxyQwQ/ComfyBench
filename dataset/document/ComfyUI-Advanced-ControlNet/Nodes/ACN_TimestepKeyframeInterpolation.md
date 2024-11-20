---
tags:
- AnimationScheduling
- Frame
- Keyframe
---

# Timestep Keyframe Interpolation 🛂🅐🅒🅝
## Documentation
- Class name: `ACN_TimestepKeyframeInterpolation`
- Category: `Adv-ControlNet 🛂🅐🅒🅝/keyframes`
- Output node: `False`

This node specializes in interpolating keyframes over timesteps within an advanced control network, enabling the creation of smooth transitions and animations by adjusting parameters such as strength and percentage over specified intervals. It leverages linear or custom interpolation methods to generate a sequence of keyframes that guide the animation or transformation process, ensuring a seamless progression from one state to another.
## Input types
### Required
- **`start_percent`**
    - Specifies the starting percentage for the interpolation, marking the initial point of the transition within the animation or transformation process. It plays a crucial role in determining the beginning of the keyframe sequence and sets the initial context for the interpolation, directly influencing the starting point of the generated keyframe sequence.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`end_percent`**
    - Defines the ending percentage for the interpolation, indicating the final point of the transition within the animation or transformation process. It is essential for establishing the conclusion of the keyframe sequence and determines the endpoint for the interpolation, ensuring that the transition smoothly concludes at the desired state.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`strength_start`**
    - Sets the starting strength for the interpolation, influencing the initial intensity or impact of the transition on the animation or transformation. It affects the beginning of the keyframe sequence's effectiveness and establishes the initial intensity level for the transition, providing a baseline for the interpolation's impact.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`strength_end`**
    - Determines the ending strength for the interpolation, affecting the final intensity or impact of the transition on the animation or transformation. It is vital for defining the end of the keyframe sequence's effectiveness and sets the final intensity level, ensuring a targeted impact by the conclusion of the transition.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`interpolation`**
    - Defines the method of interpolation used for transitioning between keyframes, such as linear or custom easing functions. It significantly impacts the smoothness and dynamics of the animation or transformation, determining the character and flow of the transition between keyframes.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`intervals`**
    - Specifies the number of intervals or steps in the interpolation process, dictating the granularity and smoothness of the transition. It directly influences the number of keyframes generated and their distribution over the transition, impacting the overall smoothness and detail of the interpolated sequence.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`prev_timestep_kf`**
    - Specifies the previous timestep keyframe group to be considered in the interpolation process, ensuring continuity and leveraging existing keyframe data for the new sequence. This input provides a foundation for the new interpolation, utilizing prior keyframe data to inform and enhance the current transition.
    - Comfy dtype: `TIMESTEP_KEYFRAME`
    - Python dtype: `object`
- **`cn_weights`**
    - Specifies the control network weights for the interpolation, influencing how different aspects are weighted during the transition. It enables precise control over the dynamics and balance of the transition, affecting how various elements are emphasized throughout the interpolation process.
    - Comfy dtype: `CONTROL_NET_WEIGHTS`
    - Python dtype: `object`
- **`latent_keyframe`**
    - Determines the latent keyframes to be included in the interpolation, allowing for the integration of specific transformation states into the sequence. It enriches the transition with predefined keyframe data, adding depth and complexity to the interpolation by incorporating existing keyframe information.
    - Comfy dtype: `LATENT_KEYFRAME`
    - Python dtype: `object`
- **`null_latent_kf_strength`**
    - Sets the strength for null latent keyframes, providing a means to adjust the influence of keyframes that do not directly contribute to the visible transformation. It allows for fine-tuning the overall effect of the interpolation, offering control over the presence and impact of latent elements in the transition.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`inherit_missing`**
    - Indicates whether to inherit missing parameters from previous keyframes, ensuring continuity and consistency across the interpolation sequence. It helps maintain the integrity of the transition, ensuring that any gaps or missing data are seamlessly integrated for a cohesive interpolation experience.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`mask_optional`**
    - Provides an optional mask hint for the interpolation, enabling targeted adjustments to specific regions or aspects of the animation or transformation. It offers additional control over the transition effects, allowing for localized modifications that can enhance or focus the interpolation's impact on certain areas.
    - Comfy dtype: `MASK`
    - Python dtype: `object`
- **`print_keyframes`**
    - Enables logging of keyframe information for debugging or analysis purposes, offering insights into the interpolation process and the generated keyframe sequence. This feature aids in the evaluation and refinement of the interpolation, providing valuable feedback on the generated keyframes and their alignment with the intended transition.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`autosize`**
    - Specifies the autosize configuration, adjusting the size of the control network's output to fit certain dimensions, which can be crucial for ensuring the proper scaling of animations or transformations. This parameter impacts the final output's dimensions, ensuring that the interpolated keyframes are appropriately scaled to match the desired output size.
    - Comfy dtype: `ACNAUTOSIZE`
    - Python dtype: `object`
## Output types
- **`TIMESTEP_KF`**
    - Comfy dtype: `TIMESTEP_KEYFRAME`
    - Returns the interpolated timestep keyframe group, encapsulating the sequence of generated keyframes over the specified intervals. This output is crucial for integrating the smooth transitions into the control network's animation or transformation process.
    - Python dtype: `TimestepKeyframeGroup`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class TimestepKeyframeInterpolationNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "strength_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001},),
                "strength_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001},),
                "interpolation": (SI._LIST, ),
                "intervals": ("INT", {"default": 50, "min": 2, "max": 100, "step": 1}),
            },
            "optional": {
                "prev_timestep_kf": ("TIMESTEP_KEYFRAME", ),
                "cn_weights": ("CONTROL_NET_WEIGHTS", ),
                "latent_keyframe": ("LATENT_KEYFRAME", ),
                "null_latent_kf_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001},),
                "inherit_missing": ("BOOLEAN", {"default": True},),
                "mask_optional": ("MASK", ),
                "print_keyframes": ("BOOLEAN", {"default": False}),
                "autosize": ("ACNAUTOSIZE", {"padding": 70}),
            }
        }
    
    RETURN_NAMES = ("TIMESTEP_KF", )
    RETURN_TYPES = ("TIMESTEP_KEYFRAME", )
    FUNCTION = "load_keyframe"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/keyframes"

    def load_keyframe(self,
                      start_percent: float, end_percent: float,
                      strength_start: float, strength_end: float, interpolation: str, intervals: int,
                      cn_weights: ControlWeights=None,
                      latent_keyframe: LatentKeyframeGroup=None,
                      prev_timestep_kf: TimestepKeyframeGroup=None,
                      null_latent_kf_strength: float=0.0,
                      inherit_missing=True,
                      guarantee_steps=1,
                      mask_optional=None, print_keyframes=False):
        if not prev_timestep_kf:
            prev_timestep_kf = TimestepKeyframeGroup()
        else:
            prev_timestep_kf = prev_timestep_kf.clone()

        percents = SI.get_weights(num_from=start_percent, num_to=end_percent, length=intervals, method=SI.LINEAR)
        strengths = SI.get_weights(num_from=strength_start, num_to=strength_end, length=intervals, method=interpolation)

        is_first = True
        for percent, strength in zip(percents, strengths):
            guarantee_steps = 0
            if is_first:
                guarantee_steps = 1
                is_first = False
            prev_timestep_kf.add(TimestepKeyframe(start_percent=percent, strength=strength, null_latent_kf_strength=null_latent_kf_strength,
                                    control_weights=cn_weights, latent_keyframes=latent_keyframe, inherit_missing=inherit_missing,
                                    guarantee_steps=guarantee_steps, mask_hint_orig=mask_optional))
            if print_keyframes:
                logger.info(f"TimestepKeyframe - start_percent:{percent} = {strength}")
        return (prev_timestep_kf,)

```
