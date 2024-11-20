---
tags:
- AnimationScheduling
- Frame
- Keyframe
---

# Timestep Keyframe 🛂🅐🅒🅝
## Documentation
- Class name: `TimestepKeyframe`
- Category: `Adv-ControlNet 🛂🅐🅒🅝/keyframes`
- Output node: `False`

The TimestepKeyframe node is designed to encapsulate the concept of a keyframe within a timeline, specifying the moment and intensity of an effect, along with associated control weights, latent keyframes, and other properties. It serves as a foundational element for defining and manipulating animation or effect progressions over time.
## Input types
### Required
- **`start_percent`**
    - Specifies the starting point of the keyframe as a percentage of the total animation or effect duration, influencing when the keyframe's effect begins.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`prev_timestep_kf`**
    - References a previous timestep keyframe group, allowing for the continuation or modification of existing keyframe sequences.
    - Comfy dtype: `TIMESTEP_KEYFRAME`
    - Python dtype: `TimestepKeyframeGroup`
- **`strength`**
    - Determines the intensity or magnitude of the keyframe's effect, affecting how strongly it influences the animation or effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`cn_weights`**
    - Associates specific control net weights with the keyframe, dictating the influence of various control parameters on the keyframe's effect.
    - Comfy dtype: `CONTROL_NET_WEIGHTS`
    - Python dtype: `ControlWeights`
- **`latent_keyframe`**
    - unknown
    - Comfy dtype: `LATENT_KEYFRAME`
    - Python dtype: `unknown`
- **`null_latent_kf_strength`**
    - Defines the strength of a null or default latent keyframe, providing a baseline effect intensity.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`inherit_missing`**
    - Indicates whether missing properties from the parent or previous keyframes should be inherited by this keyframe.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`guarantee_steps`**
    - Specifies the minimum number of steps for which the keyframe's effect is guaranteed to be applied, ensuring a minimum duration.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`mask_optional`**
    - Optionally includes a mask hint, which can guide the application of the keyframe's effect based on spatial criteria.
    - Comfy dtype: `MASK`
    - Python dtype: `Tensor`
- **`autosize`**
    - Configures automatic sizing for the keyframe, adjusting its dimensions based on specified parameters.
    - Comfy dtype: `ACNAUTOSIZE`
    - Python dtype: `ACNAUTOSIZE`
## Output types
- **`TIMESTEP_KF`**
    - Comfy dtype: `TIMESTEP_KEYFRAME`
    - Outputs a modified or newly created timestep keyframe group, incorporating the defined keyframe settings.
    - Python dtype: `TimestepKeyframeGroup`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class TimestepKeyframeNode:
    OUTDATED_DUMMY = -39

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}, ),
            },
            "optional": {
                "prev_timestep_kf": ("TIMESTEP_KEYFRAME", ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "cn_weights": ("CONTROL_NET_WEIGHTS", ),
                "latent_keyframe": ("LATENT_KEYFRAME", ),
                "null_latent_kf_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "inherit_missing": ("BOOLEAN", {"default": True}, ),
                "guarantee_steps": ("INT", {"default": 1, "min": 0, "max": BIGMAX}),
                "mask_optional": ("MASK", ),
                "autosize": ("ACNAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_NAMES = ("TIMESTEP_KF", )
    RETURN_TYPES = ("TIMESTEP_KEYFRAME", )
    FUNCTION = "load_keyframe"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/keyframes"

    def load_keyframe(self,
                      start_percent: float,
                      strength: float=1.0,
                      cn_weights: ControlWeights=None, control_net_weights: ControlWeights=None, # old name
                      latent_keyframe: LatentKeyframeGroup=None,
                      prev_timestep_kf: TimestepKeyframeGroup=None, prev_timestep_keyframe: TimestepKeyframeGroup=None, # old name
                      null_latent_kf_strength: float=0.0,
                      inherit_missing=True,
                      guarantee_steps=OUTDATED_DUMMY,
                      guarantee_usage=True, # old input
                      mask_optional=None,):
        # if using outdated dummy value, means node on workflow is outdated and should appropriately convert behavior
        if guarantee_steps == self.OUTDATED_DUMMY:
            guarantee_steps = int(guarantee_usage)
        control_net_weights = control_net_weights if control_net_weights else cn_weights
        prev_timestep_keyframe = prev_timestep_keyframe if prev_timestep_keyframe else prev_timestep_kf
        if not prev_timestep_keyframe:
            prev_timestep_keyframe = TimestepKeyframeGroup()
        else:
            prev_timestep_keyframe = prev_timestep_keyframe.clone()
        keyframe = TimestepKeyframe(start_percent=start_percent, strength=strength, null_latent_kf_strength=null_latent_kf_strength,
                                    control_weights=control_net_weights, latent_keyframes=latent_keyframe, inherit_missing=inherit_missing,
                                    guarantee_steps=guarantee_steps, mask_hint_orig=mask_optional)
        prev_timestep_keyframe.add(keyframe)
        return (prev_timestep_keyframe,)

```
