---
tags:
- Checkpoint
- CheckpointLoader
- Loader
- ModelIO
- ModelLoader
---

# CheckpointLoader (dirty)
## Documentation
- Class name: `CheckpointLoader (dirty)`
- Category: `Bmad/api/dirty loaders`
- Output node: `False`

This node is designed to load model checkpoints in a flexible manner, accommodating potentially mismatched or loosely specified checkpoint and configuration names. It utilizes utility functions to find the best match for given checkpoint and configuration names within a specified directory, thereby streamlining the process of loading models for further use or analysis.
## Input types
### Required
- **`config_name`**
    - Specifies the configuration name to be used in conjunction with the checkpoint. The node attempts to find a matching configuration file name, enhancing flexibility in specifying configurations.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`ckpt_name`**
    - Indicates the checkpoint name to be loaded. The node employs utility functions to resolve the exact checkpoint file, allowing for a more lenient specification of checkpoint names.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - Returns the loaded model object, ready for use in downstream tasks or analyses.
    - Python dtype: `torch.nn.Module`
- **`clip`**
    - Comfy dtype: `CLIP`
    - Returns the CLIP model associated with the loaded checkpoint, if applicable and requested.
    - Python dtype: `torch.nn.Module`
- **`vae`**
    - Comfy dtype: `VAE`
    - Returns the VAE model associated with the loaded checkpoint, if applicable and requested.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class DirtyCheckpointLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "config_name": ("STRING", {"default": ""}),
            "ckpt_name": ("STRING", {"default": ""})
        }}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = f"{api_category_path}/dirty loaders"

    def load_checkpoint(self, config_name, ckpt_name, output_vae=True, output_clip=True):
        ckpt_name = DirtyLoaderUtils.find_matching_filename(
            ckpt_name, folder_paths.get_filename_list("checkpoints"))

        config_name = DirtyLoaderUtils.find_matching_filename(
            config_name, folder_paths.get_filename_list("checkpoints"))

        loader = nodes.CheckpointLoader()
        return loader.load_checkpoint(config_name, ckpt_name, output_vae, output_clip)

```
