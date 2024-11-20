---
tags:
- Latent
- Normalization
---

# LoadResAdapterNormalization
## Documentation
- Class name: `LoadResAdapterNormalization`
- Category: `KJNodes/experimental`
- Output node: `False`

This node is designed to enhance the adaptability and performance of a given model by loading and applying normalization weights from a ResAdapter. It aims to improve model accuracy and efficiency by integrating external normalization parameters, thereby optimizing the model's response to various inputs.
## Input types
### Required
- **`model`**
    - The model to which the ResAdapter normalization weights will be applied. It is crucial for enhancing the model's adaptability and performance by integrating external normalization parameters.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`resadapter_path`**
    - The file path to the ResAdapter normalization weights. This path is used to locate and load the normalization parameters that are essential for optimizing the model's performance.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - A clone of the original model with the ResAdapter normalization weights applied. This enhanced model is expected to exhibit improved accuracy and efficiency.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class LoadResAdapterNormalization:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "resadapter_path": (folder_paths.get_filename_list("checkpoints"), )
            } 
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_res_adapter"
    CATEGORY = "KJNodes/experimental"

    def load_res_adapter(self, model, resadapter_path):
        print("ResAdapter: Checking ResAdapter path")
        resadapter_full_path = folder_paths.get_full_path("checkpoints", resadapter_path)
        if not os.path.exists(resadapter_full_path):
            raise Exception("Invalid model path")
        else:
            print("ResAdapter: Loading ResAdapter normalization weights")
            from comfy.utils import load_torch_file
            prefix_to_remove = 'diffusion_model.'
            model_clone = model.clone()
            norm_state_dict = load_torch_file(resadapter_full_path)
            new_values = {key[len(prefix_to_remove):]: value for key, value in norm_state_dict.items() if key.startswith(prefix_to_remove)}
            print("ResAdapter: Attempting to add patches with ResAdapter weights")
            try:
                for key in model.model.diffusion_model.state_dict().keys():
                    if key in new_values:
                        original_tensor = model.model.diffusion_model.state_dict()[key]
                        new_tensor = new_values[key].to(model.model.diffusion_model.dtype)
                        if original_tensor.shape == new_tensor.shape:
                            model_clone.add_object_patch(f"diffusion_model.{key}.data", new_tensor)
                        else:
                            print("ResAdapter: No match for key: ",key)
            except:
                raise Exception("Could not patch model, this way of patching was added to ComfyUI on March 3rd 2024, is your ComfyUI up to date?")
            print("ResAdapter: Added resnet normalization patches")
            return (model_clone, )

```
