---
tags:
- ControlNet
- Weight
---

# Scaled Soft Masked Weights 🛂🅐🅒🅝
## Documentation
- Class name: `ScaledSoftMaskedUniversalWeights`
- Category: `Adv-ControlNet 🛂🅐🅒🅝/weights`
- Output node: `False`

The ScaledSoftMaskedUniversalWeights node is designed to dynamically adjust the influence of control weights within a neural network based on a set of input parameters. It allows for the customization of weight scaling and flipping, enabling fine-tuned control over the network's behavior in response to conditional inputs or specific task requirements.
## Input types
### Required
- **`mask`**
    - Specifies the tensor mask that determines the spatial application of base multipliers across the control weights, enabling differentiated scaling.
    - Comfy dtype: `MASK`
    - Python dtype: `Tensor`
- **`min_base_multiplier`**
    - Sets the minimum scaling factor for the control weights, defining the lower bound of weight influence.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`max_base_multiplier`**
    - Sets the maximum scaling factor for the control weights, defining the upper bound of weight influence.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`uncond_multiplier`**
    - Specifies an unconditional scaling factor applied to the control weights, offering an additional layer of adjustment independent of other conditions.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`cn_extras`**
    - A dictionary of extra parameters for further customization of the control weights, providing a mechanism for incorporating additional, specific adjustments.
    - Comfy dtype: `CN_WEIGHTS_EXTRAS`
    - Python dtype: `dict[str]`
- **`autosize`**
    - Configures automatic sizing for the control weights, including padding adjustments, to ensure optimal fit within the network's architecture.
    - Comfy dtype: `ACNAUTOSIZE`
    - Python dtype: `str`
## Output types
- **`CN_WEIGHTS`**
    - Comfy dtype: `CONTROL_NET_WEIGHTS`
    - The control net weights adjusted according to the input parameters, ready for use in the network.
    - Python dtype: `ControlWeights`
- **`TK_SHORTCUT`**
    - Comfy dtype: `TIMESTEP_KEYFRAME`
    - A timestep keyframe shortcut that encapsulates the adjusted control weights for temporal application within the network.
    - Python dtype: `TimestepKeyframeGroup`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ScaledSoftMaskedUniversalWeights:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK", ),
                "min_base_multiplier": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}, ),
                "max_base_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}, ),
                #"lock_min": ("BOOLEAN", {"default": False}, ),
                #"lock_max": ("BOOLEAN", {"default": False}, ),
            },
            "optional": {
                "uncond_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}, ),
                "cn_extras": ("CN_WEIGHTS_EXTRAS",),
                "autosize": ("ACNAUTOSIZE", {"padding": 75}),
            }
        }
    
    RETURN_TYPES = ("CONTROL_NET_WEIGHTS", "TIMESTEP_KEYFRAME",)
    RETURN_NAMES = WEIGHTS_RETURN_NAMES
    FUNCTION = "load_weights"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/weights"

    def load_weights(self, mask: Tensor, min_base_multiplier: float, max_base_multiplier: float, lock_min=False, lock_max=False,
                     uncond_multiplier: float=1.0, cn_extras: dict[str]={}):
        # normalize mask
        mask = mask.clone()
        x_min = 0.0 if lock_min else mask.min()
        x_max = 1.0 if lock_max else mask.max()
        if x_min == x_max:
            mask = torch.ones_like(mask) * max_base_multiplier
        else:
            mask = linear_conversion(mask, x_min, x_max, min_base_multiplier, max_base_multiplier)
        weights = ControlWeights.universal_mask(weight_mask=mask, uncond_multiplier=uncond_multiplier, extras=cn_extras)
        return (weights, TimestepKeyframeGroup.default(TimestepKeyframe(control_weights=weights)))

```
