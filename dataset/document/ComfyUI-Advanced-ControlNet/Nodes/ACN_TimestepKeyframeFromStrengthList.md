---
tags:
- AnimationScheduling
- Frame
- Keyframe
---

# Timestep Keyframe From List 🛂🅐🅒🅝
## Documentation
- Class name: `ACN_TimestepKeyframeFromStrengthList`
- Category: `Adv-ControlNet 🛂🅐🅒🅝/keyframes`
- Output node: `False`

This node is designed to generate a sequence of timestep keyframes based on a list of strength values. It facilitates the creation of dynamic animations or effects within a control network by interpolating between specified strength levels over a series of timesteps, allowing for precise control over the animation's progression.
## Input types
### Required
- **`float_strengths`**
    - Specifies the strength values for each keyframe. This input allows for the dynamic generation of keyframes based on varying strength levels, enabling the creation of nuanced animations or effects.
    - Comfy dtype: `FLOAT`
    - Python dtype: `Union[float, List[float]]`
- **`start_percent`**
    - Defines the starting percentage for the keyframe sequence, allowing for the customization of when the animation effects begin within the control network.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`end_percent`**
    - Sets the ending percentage for the keyframe sequence, determining when the animation effects conclude within the control network.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`prev_timestep_kf`**
    - An optional input representing the previous state of timestep keyframes, enabling the continuation or modification of an existing sequence.
    - Comfy dtype: `TIMESTEP_KEYFRAME`
    - Python dtype: `TimestepKeyframeGroup`
- **`cn_weights`**
    - Optional control network weights that can be applied to the keyframes, offering additional customization for the animation's behavior.
    - Comfy dtype: `CONTROL_NET_WEIGHTS`
    - Python dtype: `ControlWeights`
- **`latent_keyframe`**
    - Optional latent keyframes that can be integrated into the timestep keyframes for enhanced control over the animation's characteristics.
    - Comfy dtype: `LATENT_KEYFRAME`
    - Python dtype: `LatentKeyframeGroup`
- **`null_latent_kf_strength`**
    - Defines the strength of a null latent keyframe, allowing for the adjustment of the animation's intensity.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`inherit_missing`**
    - A boolean flag indicating whether missing keyframe attributes should be inherited from previous keyframes, ensuring continuity in the animation.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`mask_optional`**
    - An optional mask that can be applied to the keyframes, providing further customization of the animation's appearance.
    - Comfy dtype: `MASK`
    - Python dtype: `Tensor`
- **`print_keyframes`**
    - A boolean flag that, when set to true, enables logging of the generated keyframes for debugging or informational purposes.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`autosize`**
    - Specifies the autosizing options for the control network, facilitating the automatic adjustment of the network's dimensions based on the provided parameters.
    - Comfy dtype: `ACNAUTOSIZE`
    - Python dtype: `Dict[str, int]`
## Output types
- **`TIMESTEP_KF`**
    - Comfy dtype: `TIMESTEP_KEYFRAME`
    - The output is a sequence of timestep keyframes, which are generated based on the provided strength values and other parameters. This sequence can be used to control animations or effects within a control network.
    - Python dtype: `TimestepKeyframeGroup`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class TimestepKeyframeFromStrengthListNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "float_strengths": ("FLOAT", {"default": -1, "min": -1, "step": 0.001, "forceInput": True}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                "prev_timestep_kf": ("TIMESTEP_KEYFRAME", ),
                "cn_weights": ("CONTROL_NET_WEIGHTS", ),
                "latent_keyframe": ("LATENT_KEYFRAME", ),
                "null_latent_kf_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001},),
                "inherit_missing": ("BOOLEAN", {"default": True},),
                "mask_optional": ("MASK", ),
                "print_keyframes": ("BOOLEAN", {"default": False}),
                "autosize": ("ACNAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_NAMES = ("TIMESTEP_KF", )
    RETURN_TYPES = ("TIMESTEP_KEYFRAME", )
    FUNCTION = "load_keyframe"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/keyframes"

    def load_keyframe(self,
                      start_percent: float, end_percent: float,
                      float_strengths: float,
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

        if type(float_strengths) in (float, int):
            float_strengths = [float(float_strengths)]
        elif isinstance(float_strengths, Iterable):
            pass
        else:
            raise Exception(f"strengths_float must be either an iterable input or a float, but was {type(float_strengths).__repr__}.")
        percents = SI.get_weights(num_from=start_percent, num_to=end_percent, length=len(float_strengths), method=SI.LINEAR)

        is_first = True
        for percent, strength in zip(percents, float_strengths):
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
