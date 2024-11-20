---
tags:
- ControlNet
- Weight
---

# Default Weights 🛂🅐🅒🅝
## Documentation
- Class name: `ACN_DefaultUniversalWeights`
- Category: `Adv-ControlNet 🛂🅐🅒🅝/weights`
- Output node: `False`

This node specializes in loading and configuring default universal weights for the Advanced ControlNet system, facilitating the customization and optimization of weight parameters for enhanced control in text-to-image adaptation processes.
## Input types
### Optional
- **`cn_extras`**
    - Allows for the inclusion of additional, custom weight parameters, offering extended customization capabilities for the control network.
    - Comfy dtype: `CN_WEIGHTS_EXTRAS`
    - Python dtype: `dict[str]`
- **`autosize`**
    - Configures automatic sizing for the control network, optimizing its structure and parameters for improved performance.
    - Comfy dtype: `ACNAUTOSIZE`
    - Python dtype: `{'padding': int}`
## Output types
- **`CN_WEIGHTS`**
    - Comfy dtype: `CONTROL_NET_WEIGHTS`
    - Outputs the configured control net weights, ready for use in the Advanced ControlNet system.
    - Python dtype: `ControlWeights`
- **`TK_SHORTCUT`**
    - Comfy dtype: `TIMESTEP_KEYFRAME`
    - Provides a shortcut to timestep keyframe configurations, facilitating efficient integration and application within the system.
    - Python dtype: `TimestepKeyframeGroup`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class DefaultWeights:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "cn_extras": ("CN_WEIGHTS_EXTRAS",),
                "autosize": ("ACNAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("CONTROL_NET_WEIGHTS", "TIMESTEP_KEYFRAME",)
    RETURN_NAMES = WEIGHTS_RETURN_NAMES
    FUNCTION = "load_weights"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/weights"

    def load_weights(self, cn_extras: dict[str]={}):
        weights = ControlWeights.default(extras=cn_extras)
        return (weights, TimestepKeyframeGroup.default(TimestepKeyframe(control_weights=weights))) 

```
