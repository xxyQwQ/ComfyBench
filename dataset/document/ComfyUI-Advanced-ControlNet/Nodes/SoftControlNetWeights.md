---
tags:
- ControlNet
- Weight
---

# ControlNet Soft Weights 🛂🅐🅒🅝
## Documentation
- Class name: `SoftControlNetWeights`
- Category: `Adv-ControlNet 🛂🅐🅒🅝/weights/ControlNet`
- Output node: `False`

This node is designed to dynamically adjust the weights for control networks in a soft, nuanced manner, allowing for more refined and flexible control over the generation process. It focuses on providing a customizable approach to weight manipulation, enabling users to fine-tune the influence of various control parameters based on specific needs or desired outcomes.
## Input types
### Required
- **`weight_i`**
    - Specifies a weight in the control network, serving as an adjustable factor for nuanced control. The index 'i' ranges from 0 to 3, allowing for multiple levels of refinement in control adjustments.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`flip_weights`**
    - A boolean flag that determines whether the control weights should be inverted, offering an alternative perspective on the control influence.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`uncond_multiplier`**
    - Defines an unconditional multiplier that applies universally across all control weights, allowing for a global adjustment to their influence.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`cn_extras`**
    - Allows for the inclusion of additional, custom-defined control weight parameters, providing a mechanism for extending and customizing the control influence beyond the standard set.
    - Comfy dtype: `CN_WEIGHTS_EXTRAS`
    - Python dtype: `dict[str]`
- **`autosize`**
    - Configures automatic sizing parameters for the control network, facilitating adaptive adjustments to the network's dimensions based on specific requirements.
    - Comfy dtype: `ACNAUTOSIZE`
    - Python dtype: `dict[str, int]`
## Output types
- **`CN_WEIGHTS`**
    - Comfy dtype: `CONTROL_NET_WEIGHTS`
    - The computed control network weights after applying the specified adjustments.
    - Python dtype: `Tensor`
- **`TK_SHORTCUT`**
    - Comfy dtype: `TIMESTEP_KEYFRAME`
    - A shortcut to the timestep keyframe, encapsulating the adjusted control weights.
    - Python dtype: `TimestepKeyframeGroup`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SoftControlNetWeights:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "weight_00": ("FLOAT", {"default": 0.09941396206337118, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_01": ("FLOAT", {"default": 0.12050177219802567, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_02": ("FLOAT", {"default": 0.14606275417942507, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_03": ("FLOAT", {"default": 0.17704576264172736, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_04": ("FLOAT", {"default": 0.214600924414215, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_05": ("FLOAT", {"default": 0.26012233262329093, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_06": ("FLOAT", {"default": 0.3152997971191405, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_07": ("FLOAT", {"default": 0.3821815722656249, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_08": ("FLOAT", {"default": 0.4632503906249999, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_09": ("FLOAT", {"default": 0.561515625, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_10": ("FLOAT", {"default": 0.6806249999999999, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_11": ("FLOAT", {"default": 0.825, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_12": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "flip_weights": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "uncond_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}, ),
                "cn_extras": ("CN_WEIGHTS_EXTRAS",),
                "autosize": ("ACNAUTOSIZE", {"padding": 50}),
            }
        }
    
    RETURN_TYPES = ("CONTROL_NET_WEIGHTS", "TIMESTEP_KEYFRAME",)
    RETURN_NAMES = WEIGHTS_RETURN_NAMES
    FUNCTION = "load_weights"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/weights/ControlNet"

    def load_weights(self, weight_00, weight_01, weight_02, weight_03, weight_04, weight_05, weight_06, 
                     weight_07, weight_08, weight_09, weight_10, weight_11, weight_12, flip_weights,
                     uncond_multiplier: float=1.0, cn_extras: dict[str]={}):
        weights = [weight_00, weight_01, weight_02, weight_03, weight_04, weight_05, weight_06, 
                   weight_07, weight_08, weight_09, weight_10, weight_11, weight_12]
        weights = ControlWeights.controlnet(weights, flip_weights=flip_weights, uncond_multiplier=uncond_multiplier, extras=cn_extras)
        return (weights, TimestepKeyframeGroup.default(TimestepKeyframe(control_weights=weights)))

```
