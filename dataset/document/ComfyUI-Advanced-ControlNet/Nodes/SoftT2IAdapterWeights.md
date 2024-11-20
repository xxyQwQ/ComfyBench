---
tags:
- ControlNet
- Weight
---

# T2IAdapter Soft Weights 🛂🅐🅒🅝
## Documentation
- Class name: `SoftT2IAdapterWeights`
- Category: `Adv-ControlNet 🛂🅐🅒🅝/weights/T2IAdapter`
- Output node: `False`

This node is designed to adapt text-to-image (T2I) control weights for a softer blending or transition effect in generated images. It manipulates a set of weights to achieve a desired balance between different control aspects, enhancing the flexibility and precision of image generation processes.
## Input types
### Required
- **`weight_i`**
    - Specifies the weight for a control aspect, influencing the blending or transition effect in the generated images. The index 'i' ranges from 0 to 3, each representing a different aspect of control.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`flip_weights`**
    - When enabled, reverses the order of weights, offering an alternative approach to control the blending or transition effects.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`uncond_multiplier`**
    - Adjusts the unconditional aspect of the weights, providing an additional layer of control over the image generation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`cn_extras`**
    - Includes extra control net weights for extended customization and fine-tuning capabilities in the image generation process.
    - Comfy dtype: `CN_WEIGHTS_EXTRAS`
    - Python dtype: `dict[str]`
- **`autosize`**
    - Configures automatic sizing parameters, affecting how control net weights are applied and adjusted.
    - Comfy dtype: `ACNAUTOSIZE`
    - Python dtype: `dict`
## Output types
- **`CN_WEIGHTS`**
    - Comfy dtype: `CONTROL_NET_WEIGHTS`
    - The computed control net weights after processing input parameters.
    - Python dtype: `Tensor`
- **`TK_SHORTCUT`**
    - Comfy dtype: `TIMESTEP_KEYFRAME`
    - A shortcut representation for timestep keyframes, facilitating easier manipulation and application in the control net process.
    - Python dtype: `Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SoftT2IAdapterWeights:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "weight_00": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_01": ("FLOAT", {"default": 0.62, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_02": ("FLOAT", {"default": 0.825, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_03": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "flip_weights": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "uncond_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}, ),
                "cn_extras": ("CN_WEIGHTS_EXTRAS",),
                "autosize": ("ACNAUTOSIZE", {"padding": 55}),
            }
        }
    
    RETURN_TYPES = ("CONTROL_NET_WEIGHTS", "TIMESTEP_KEYFRAME",)
    RETURN_NAMES = WEIGHTS_RETURN_NAMES
    FUNCTION = "load_weights"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/weights/T2IAdapter"

    def load_weights(self, weight_00, weight_01, weight_02, weight_03, flip_weights,
                     uncond_multiplier: float=1.0, cn_extras: dict[str]={}):
        weights = [weight_00, weight_01, weight_02, weight_03]
        weights = get_properly_arranged_t2i_weights(weights)
        weights.reverse()  # to account for recent ComfyUI changes
        weights = ControlWeights.t2iadapter(weights, flip_weights=flip_weights, uncond_multiplier=uncond_multiplier, extras=cn_extras)
        return (weights, TimestepKeyframeGroup.default(TimestepKeyframe(control_weights=weights)))

```
