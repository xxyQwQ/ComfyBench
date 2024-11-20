# Load And Apply IC-Light
## Documentation
- Class name: `LoadAndApplyICLightUnet`
- Category: `IC-Light`
- Output node: `False`

This node is designed to load and apply IC-Light Unet weights to a model, ensuring compatibility with specific model architectures by patching and converting state dictionaries as necessary. It focuses on enhancing models with IC-Light Unet's capabilities, facilitating the integration of advanced lighting effects into generated images.
## Input types
### Required
- **`model`**
    - The model to which IC-Light Unet weights will be applied. It plays a crucial role in the node's operation by being the recipient of the enhanced lighting capabilities provided by IC-Light Unet.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`model_path`**
    - The path to the IC-Light Unet weights file. This parameter is essential for locating and loading the specific weights to be applied to the model.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The model enhanced with IC-Light Unet weights. This output is crucial for utilizing the model with advanced lighting effects in generated images.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class LoadAndApplyICLightUnet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "model_path": (folder_paths.get_filename_list("unet"), )
            } 
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "IC-Light"
    DESCRIPTION = """
  
Loads and applies the diffusers SD1.5 IC-Light models available here:  
https://huggingface.co/lllyasviel/ic-light/tree/main  
  
Used with ICLightConditioning -node  
"""

    def load(self, model, model_path):
        type_str = str(type(model.model.model_config).__name__)
        if "SD15" not in type_str:
            raise Exception(f"Attempted to load {type_str} model, IC-Light is only compatible with SD 1.5 models.")

        print("LoadAndApplyICLightUnet: Checking IC-Light Unet path")
        model_full_path = folder_paths.get_full_path("unet", model_path)
        if not os.path.exists(model_full_path):
            raise Exception("Invalid model path")
        else:
            print("LoadAndApplyICLightUnet: Loading IC-Light Unet weights")
            model_clone = model.clone()

            iclight_state_dict = load_torch_file(model_full_path)
            
            print("LoadAndApplyICLightUnet: Attempting to add patches with IC-Light Unet weights")
            try:          
                if 'conv_in.weight' in iclight_state_dict:
                    iclight_state_dict = convert_iclight_unet(iclight_state_dict)
                    in_channels = iclight_state_dict["diffusion_model.input_blocks.0.0.weight"].shape[1]
                    for key in iclight_state_dict:
                        model_clone.add_patches({key: (iclight_state_dict[key],)}, 1.0, 1.0)
                else:
                    for key in iclight_state_dict:
                        model_clone.add_patches({"diffusion_model." + key: (iclight_state_dict[key],)}, 1.0, 1.0)

                    in_channels = iclight_state_dict["input_blocks.0.0.weight"].shape[1]

            except:
                raise Exception("Could not patch model")
            print("LoadAndApplyICLightUnet: Added LoadICLightUnet patches")

            #Patch ComfyUI's LoRA weight application to accept multi-channel inputs. Thanks @huchenlei
            try:
                ModelPatcher.calculate_weight = calculate_weight_adjust_channel(ModelPatcher.calculate_weight)
            except:
                raise Exception("IC-Light: Could not patch calculate_weight")
            # Mimic the existing IP2P class to enable extra_conds
            def bound_extra_conds(self, **kwargs):
                 return ICLight.extra_conds(self, **kwargs)
            new_extra_conds = types.MethodType(bound_extra_conds, model_clone.model)
            model_clone.add_object_patch("extra_conds", new_extra_conds)
            

            model_clone.model.model_config.unet_config["in_channels"] = in_channels        

            return (model_clone, )

```
