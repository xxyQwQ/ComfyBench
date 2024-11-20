# Load ControlNet++ Model (Multi) 🛂🅐🅒🅝
## Documentation
- Class name: `ACN_ControlNet++LoaderAdvanced`
- Category: `Adv-ControlNet 🛂🅐🅒🅝/ControlNet++`
- Output node: `False`

This node is designed to load multiple ControlNet++ models, facilitating advanced model management and usage within the ComfyUI framework. It supports loading various configurations of ControlNet++ models, enabling flexible and dynamic application of these models in different scenarios.
## Input types
### Required
- **`plus_input`**
    - Specifies the PlusPlusInputGroup to be used with the ControlNet++ model, enabling advanced control and customization of the model's behavior.
    - Comfy dtype: `PLUS_INPUT`
    - Python dtype: `PlusPlusInputGroup`
- **`name`**
    - Specifies the name of the ControlNet++ model to be loaded, allowing for dynamic selection from available models.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`control_net`**
    - Comfy dtype: `CONTROL_NET`
    - Returns the loaded ControlNet++ model, ready for application in the specified context.
    - Python dtype: `ControlNetPlusPlus`
- **`image`**
    - Comfy dtype: `IMAGE`
    - Provides an image wrapper around the PlusPlusInputGroup, facilitating its use with the loaded ControlNet++ model.
    - Python dtype: `PlusPlusImageWrapper`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class PlusPlusLoaderAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "plus_input": ("PLUS_INPUT", ),
                "name": (folder_paths.get_filename_list("controlnet"), ),
            }
        }
    
    RETURN_TYPES = ("CONTROL_NET", "IMAGE",)
    FUNCTION = "load_controlnet_plusplus"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/ControlNet++"

    def load_controlnet_plusplus(self, plus_input: PlusPlusInputGroup, name: str):
        controlnet_path = folder_paths.get_full_path("controlnet", name)
        controlnet = load_controlnetplusplus(controlnet_path)
        controlnet.verify_control_type(name, plus_input)
        return (controlnet, PlusPlusImageWrapper(plus_input),)

```
