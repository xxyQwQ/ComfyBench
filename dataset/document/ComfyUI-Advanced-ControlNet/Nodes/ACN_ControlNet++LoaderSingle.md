# Load ControlNet++ Model (Single) 🛂🅐🅒🅝
## Documentation
- Class name: `ACN_ControlNet++LoaderSingle`
- Category: `Adv-ControlNet 🛂🅐🅒🅝/ControlNet++`
- Output node: `False`

This node is designed to load a single instance of the ControlNet++ model, facilitating the integration of advanced control mechanisms into the generation process. It abstracts the complexities of model loading, ensuring seamless access to ControlNet++ functionalities for enhanced content creation.
## Input types
### Required
- **`name`**
    - Specifies the name of the ControlNet++ model to be loaded, serving as a key identifier for selecting the appropriate model instance.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`control_type`**
    - Determines the type of control mechanism to be applied using the ControlNet++ model, influencing how the model will be utilized in the generation process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`control_net`**
    - Comfy dtype: `CONTROL_NET`
    - The loaded ControlNet++ model instance, ready for integration and use in content generation workflows.
    - Python dtype: `ControlNetPlusPlus`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class PlusPlusLoaderSingle:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "name": (folder_paths.get_filename_list("controlnet"), ),
                "control_type": (PlusPlusType._LIST_WITH_NONE, {"default": PlusPlusType.NONE}, ),
            }
        }
    
    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet_plusplus"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/ControlNet++"

    def load_controlnet_plusplus(self, name: str, control_type: str):
        controlnet_path = folder_paths.get_full_path("controlnet", name)
        controlnet = load_controlnetplusplus(controlnet_path)
        controlnet.single_control_type = control_type
        controlnet.verify_control_type(name)
        return (controlnet,)

```
