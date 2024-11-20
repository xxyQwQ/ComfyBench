---
tags:
- Checkpoint
- CheckpointLoader
- Loader
- ModelIO
- ModelLoader
---

# Load Checkpoint With Config (DEPRECATED)
## Documentation
- Class name: `CheckpointLoader`
- Category: `advanced/loaders`
- Output node: `False`

The CheckpointLoader node is designed for advanced loading of model checkpoints, specifically enabling the selection of configurations and checkpoints from predefined folders. It facilitates the dynamic loading of various components such as models, CLIP, and VAE based on the specified configuration and checkpoint names, supporting complex setups and custom model loading strategies.
## Input types
### Required
- **`config_name`**
    - Specifies the name of the configuration to use, allowing the node to locate and load the corresponding model settings from a predefined list of configurations.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`ckpt_name`**
    - Determines the specific checkpoint to load, enabling the node to fetch and apply the model weights from a predefined list of checkpoints.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - Outputs the loaded model, ready for further use or integration within a pipeline.
    - Python dtype: `torch.nn.Module`
- **`clip`**
    - Comfy dtype: `CLIP`
    - Provides the CLIP component loaded alongside the model, if applicable.
    - Python dtype: `torch.nn.Module`
- **`vae`**
    - Comfy dtype: `VAE`
    - Returns the VAE component loaded with the model, facilitating various generative tasks.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class CheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "config_name": (folder_paths.get_filename_list("configs"), ),
                              "ckpt_name": (folder_paths.get_filename_list("checkpoints"), )}}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "advanced/loaders"

    def load_checkpoint(self, config_name, ckpt_name):
        config_path = folder_paths.get_full_path("configs", config_name)
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        return comfy.sd.load_checkpoint(config_path, ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))

```
