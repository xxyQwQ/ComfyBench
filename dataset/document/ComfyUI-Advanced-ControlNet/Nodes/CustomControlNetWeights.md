---
tags:
- ControlNet
- Weight
---

# ControlNet Custom Weights 🛂🅐🅒🅝
## Documentation
- Class name: `CustomControlNetWeights`
- Category: `Adv-ControlNet 🛂🅐🅒🅝/weights/ControlNet`
- Output node: `False`

This node is designed to dynamically load and apply custom control net weights for advanced manipulation of neural network behavior. It allows for fine-tuning the influence of various control parameters on the network's output, enabling more precise and tailored generation of content.
## Input types
### Required
- **`weight_i`**
    - Specifies a weight in the control net, contributing to the nuanced adjustment of network behavior. The index 'i' ranges from 00 to 03, allowing for detailed control over the network's processing and output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`flip_weights`**
    - A boolean flag that, when set, reverses the order of weight application, potentially altering the network's behavior in significant ways.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`uncond_multiplier`**
    - Adjusts the unconditional multiplier for the control weights, offering an additional layer of fine-tuning over the network's output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`cn_extras`**
    - Provides a mechanism to include extra control weight parameters, further expanding the customization capabilities for network behavior.
    - Comfy dtype: `CN_WEIGHTS_EXTRAS`
    - Python dtype: `dict[str]`
- **`autosize`**
    - Configures the autosizing feature, which automatically adjusts the size of control weights based on specified parameters, enhancing the adaptability of network behavior.
    - Comfy dtype: `ACNAUTOSIZE`
    - Python dtype: `int`
## Output types
- **`CN_WEIGHTS`**
    - Comfy dtype: `CONTROL_NET_WEIGHTS`
    - The customized control net weights, tailored according to the input parameters.
    - Python dtype: `ControlWeights`
- **`TK_SHORTCUT`**
    - Comfy dtype: `TIMESTEP_KEYFRAME`
    - A shortcut to timestep keyframe data, facilitating easier manipulation and application of control net weights over time.
    - Python dtype: `TimestepKeyframeGroup`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class CustomControlNetWeights:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "weight_00": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_01": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_02": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_03": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_04": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_05": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_06": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_07": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_08": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_09": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_10": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_11": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "weight_12": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
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

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/weights/ControlNet"

    def load_weights(self, weight_00, weight_01, weight_02, weight_03, weight_04, weight_05, weight_06, 
                     weight_07, weight_08, weight_09, weight_10, weight_11, weight_12, flip_weights,
                     uncond_multiplier: float=1.0, cn_extras: dict[str]={}):
        weights = [weight_00, weight_01, weight_02, weight_03, weight_04, weight_05, weight_06, 
                   weight_07, weight_08, weight_09, weight_10, weight_11, weight_12]
        weights = ControlWeights.controlnet(weights, flip_weights=flip_weights, uncond_multiplier=uncond_multiplier, extras=cn_extras)
        return (weights, TimestepKeyframeGroup.default(TimestepKeyframe(control_weights=weights)))

```
