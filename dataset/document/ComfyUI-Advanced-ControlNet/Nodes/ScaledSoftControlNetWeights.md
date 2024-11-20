---
tags:
- ControlNet
- Weight
---

# Scaled Soft Weights 🛂🅐🅒🅝
## Documentation
- Class name: `ScaledSoftControlNetWeights`
- Category: `Adv-ControlNet 🛂🅐🅒🅝/weights`
- Output node: `False`

The ScaledSoftControlNetWeights node is designed to dynamically adjust the influence of control weights within a neural network based on a set of parameters. It allows for the customization of weight scaling, enabling fine-tuned control over the network's behavior in response to specific conditions or inputs.
## Input types
### Required
- **`base_multiplier`**
    - Defines the base scaling factor for the control weights, serving as the foundational adjustment level before any conditional modifications are applied.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`flip_weights`**
    - A boolean flag that, when true, reverses the effect of the control weights, allowing for inverse scaling effects.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`uncond_multiplier`**
    - Specifies an unconditional scaling multiplier that is applied regardless of other conditions, offering an additional layer of weight adjustment.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`cn_extras`**
    - A dictionary of extra parameters for further customization of the control weights, providing extended flexibility in their application.
    - Comfy dtype: `CN_WEIGHTS_EXTRAS`
    - Python dtype: `dict[str]`
- **`autosize`**
    - Configures automatic sizing for the control weights, including padding adjustments to accommodate various network configurations.
    - Comfy dtype: `ACNAUTOSIZE`
    - Python dtype: `dict`
## Output types
- **`CN_WEIGHTS`**
    - Comfy dtype: `CONTROL_NET_WEIGHTS`
    - Represents the adjusted control net weights after applying the scaling factors.
    - Python dtype: `ControlNetWeights`
- **`TK_SHORTCUT`**
    - Comfy dtype: `TIMESTEP_KEYFRAME`
    - A shortcut to timestep keyframe data, encapsulating the effects of the adjusted weights on the network's temporal dynamics.
    - Python dtype: `TimestepKeyframe`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [ControlNetLoaderAdvanced](../../ComfyUI-Advanced-ControlNet/Nodes/ControlNetLoaderAdvanced.md)



## Source code
```python
class ScaledSoftUniversalWeights:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_multiplier": ("FLOAT", {"default": 0.825, "min": 0.0, "max": 1.0, "step": 0.001}, ),
                "flip_weights": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "uncond_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}, ),
                "cn_extras": ("CN_WEIGHTS_EXTRAS",),
                "autosize": ("ACNAUTOSIZE", {"padding": 20}),
            }
        }
    
    RETURN_TYPES = ("CONTROL_NET_WEIGHTS", "TIMESTEP_KEYFRAME",)
    RETURN_NAMES = WEIGHTS_RETURN_NAMES
    FUNCTION = "load_weights"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/weights"

    def load_weights(self, base_multiplier, flip_weights, uncond_multiplier: float=1.0, cn_extras: dict[str]={}):
        weights = ControlWeights.universal(base_multiplier=base_multiplier, flip_weights=flip_weights, uncond_multiplier=uncond_multiplier, extras=cn_extras)
        return (weights, TimestepKeyframeGroup.default(TimestepKeyframe(control_weights=weights))) 

```
