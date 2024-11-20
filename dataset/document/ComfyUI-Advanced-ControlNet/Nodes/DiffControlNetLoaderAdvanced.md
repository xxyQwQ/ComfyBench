---
tags:
- ControlNet
- ControlNetLoader
---

# Load Advanced ControlNet Model (diff) 🛂🅐🅒🅝
## Documentation
- Class name: `DiffControlNetLoaderAdvanced`
- Category: `Adv-ControlNet 🛂🅐🅒🅝`
- Output node: `False`

The DiffControlNetLoaderAdvanced node is designed for loading advanced control net models with the capability to handle different models and optional timestep keyframe groups. It extends the functionality of control net loading by incorporating model-specific adjustments and optional parameters for enhanced control net application.
## Input types
### Required
- **`model`**
    - Specifies the model for which the control net is being loaded, enabling model-specific control net adjustments.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
- **`control_net_name`**
    - The name of the control net to be loaded, used to locate and load the specific control net file.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`tk_optional`**
    - An optional timestep keyframe group to be applied to the control net, providing additional control over the animation or transformation process.
    - Comfy dtype: `TIMESTEP_KEYFRAME`
    - Python dtype: `TimestepKeyframeGroup`
- **`autosize`**
    - An optional parameter that allows for automatic resizing of the control net with a specified padding, enhancing flexibility in control net application.
    - Comfy dtype: `ACNAUTOSIZE`
    - Python dtype: `Dict[str, int]`
## Output types
- **`control_net`**
    - Comfy dtype: `CONTROL_NET`
    - The loaded advanced control net, ready for application within the system.
    - Python dtype: `ControlNet`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class DiffControlNetLoaderAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "control_net_name": (folder_paths.get_filename_list("controlnet"), )
            },
            "optional": {
                "tk_optional": ("TIMESTEP_KEYFRAME", ),
                "autosize": ("ACNAUTOSIZE", {"padding": 160}),
            }
        }
    
    RETURN_TYPES = ("CONTROL_NET", )
    FUNCTION = "load_controlnet"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝"

    def load_controlnet(self, control_net_name, model,
                        tk_optional: TimestepKeyframeGroup=None,
                        timestep_keyframe: TimestepKeyframeGroup=None
                        ):
        if timestep_keyframe is not None: # backwards compatibility
            tk_optional = timestep_keyframe
        controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
        controlnet = load_controlnet(controlnet_path, tk_optional, model)
        if is_advanced_controlnet(controlnet):
            controlnet.verify_all_weights()
        return (controlnet,)

```
