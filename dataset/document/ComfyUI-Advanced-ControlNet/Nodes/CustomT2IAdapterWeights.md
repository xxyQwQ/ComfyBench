---
tags:
- ControlNet
- Weight
---

# T2IAdapter Custom Weights 🛂🅐🅒🅝
## Documentation
- Class name: `CustomT2IAdapterWeights`
- Category: `Adv-ControlNet 🛂🅐🅒🅝/weights/T2IAdapter`
- Output node: `False`

This node is designed to adapt and load custom weights for text-to-image (T2I) generation within the Advanced ControlNet framework. It allows for precise control over the influence of various weights on the image generation process, accommodating adjustments and enhancements to the control mechanism.
## Input types
### Required
- **`weight_i`**
    - Specifies a weight in the sequence, setting the foundational influence for the T2I generation process. The index 'i' ranges from 00 to 03, each adjusting the influence on the generation process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`flip_weights`**
    - A boolean flag to reverse the order of weights, accommodating recent changes in ComfyUI.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`uncond_multiplier`**
    - A multiplier for unconditional weights, offering additional control over the generation process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`cn_extras`**
    - A dictionary for extra control net weights, allowing for extended customization.
    - Comfy dtype: `CN_WEIGHTS_EXTRAS`
    - Python dtype: `dict[str]`
- **`autosize`**
    - Specifies the padding size for automatic sizing, enhancing layout adaptability.
    - Comfy dtype: `ACNAUTOSIZE`
    - Python dtype: `dict[str]`
## Output types
- **`CN_WEIGHTS`**
    - Comfy dtype: `CONTROL_NET_WEIGHTS`
    - The customized control net weights adapted for T2I generation.
    - Python dtype: `torch.Tensor`
- **`TK_SHORTCUT`**
    - Comfy dtype: `TIMESTEP_KEYFRAME`
    - A timestep keyframe shortcut, encapsulating the control weights for efficient processing.
    - Python dtype: `TimestepKeyframeGroup`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class CustomT2IAdapterWeights:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "weight_00": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_01": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_02": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_03": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "flip_weights": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "uncond_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}, ),
                "cn_extras": ("CN_WEIGHTS_EXTRAS",),
                "autosize": ("ACNAUTOSIZE", {"padding": 65}),
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
