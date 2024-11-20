---
tags:
- ControlNet
- ControlNetLoader
---

# Load Advanced ControlNet Model 🛂🅐🅒🅝
## Documentation
- Class name: `ControlNetLoaderAdvanced`
- Category: `Adv-ControlNet 🛂🅐🅒🅝`
- Output node: `False`

This node is designed to load advanced control networks, enabling the customization and enhancement of generative models through specified control net configurations. It supports loading control nets by name and optionally applying timestep keyframe adjustments for fine-tuned control.
## Input types
### Required
- **`control_net_name`**
    - Specifies the name of the control net to load, serving as a key identifier for retrieving the appropriate control net configuration.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
### Optional
- **`tk_optional`**
    - An optional parameter for specifying timestep keyframe adjustments, enhancing the control net's application with temporal fine-tuning.
    - Comfy dtype: `TIMESTEP_KEYFRAME`
    - Python dtype: `TimestepKeyframeGroup`
## Output types
- **`control_net`**
    - Comfy dtype: `CONTROL_NET`
    - Returns the loaded control net, ready for application in generative model customization.
    - Python dtype: `ControlNet`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [ControlNetApplyAdvanced](../../Comfy/Nodes/ControlNetApplyAdvanced.md)
    - [ACN_AdvancedControlNetApply](../../ComfyUI-Advanced-ControlNet/Nodes/ACN_AdvancedControlNetApply.md)
    - [ControlNetApply](../../Comfy/Nodes/ControlNetApply.md)
    - [Control Net Stacker](../../efficiency-nodes-comfyui/Nodes/Control Net Stacker.md)
    - [ImpactControlNetApplySEGS](../../ComfyUI-Impact-Pack/Nodes/ImpactControlNetApplySEGS.md)



## Source code
```python
class ControlNetLoaderAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_net_name": (folder_paths.get_filename_list("controlnet"), ),
            },
            "optional": {
                "tk_optional": ("TIMESTEP_KEYFRAME", ),
            }
        }

    RETURN_TYPES = ("CONTROL_NET", )
    FUNCTION = "load_controlnet"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝"

    def load_controlnet(self, control_net_name,
                        tk_optional: TimestepKeyframeGroup=None,
                        timestep_keyframe: TimestepKeyframeGroup=None,
                        ):
        if timestep_keyframe is not None: # backwards compatibility
            tk_optional = timestep_keyframe
        controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
        controlnet = load_controlnet(controlnet_path, tk_optional)
        return (controlnet,)

```
