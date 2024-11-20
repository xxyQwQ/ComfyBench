---
tags:
- Checkpoint
- CheckpointLoader
- Loader
- ModelIO
- ModelLoader
---

# CheckpointLoaderSimple (dirty)
## Documentation
- Class name: `CheckpointLoaderSimple (dirty)`
- Category: `Bmad/api/dirty loaders`
- Output node: `False`

This node simplifies the process of loading model checkpoints by automatically finding and loading the appropriate checkpoint file based on a given checkpoint name. It abstracts away the complexities of locating and verifying checkpoint files, making it easier to load models for further use or analysis.
## Input types
### Required
- **`ckpt_name`**
    - Specifies the name of the checkpoint file to load. This parameter is crucial for identifying and retrieving the correct checkpoint file from a predefined list of available checkpoints.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - Returns the loaded model from the specified checkpoint.
    - Python dtype: `torch.nn.Module`
- **`clip`**
    - Comfy dtype: `CLIP`
    - Returns the CLIP model associated with the loaded checkpoint, if available.
    - Python dtype: `torch.nn.Module`
- **`vae`**
    - Comfy dtype: `VAE`
    - Returns the VAE model associated with the loaded checkpoint, if available.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class DirtyCheckpointLoaderSimple:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"ckpt_name": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = f"{api_category_path}/dirty loaders"

    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True):
        ckpt_name = DirtyLoaderUtils.find_matching_filename(
            ckpt_name, folder_paths.get_filename_list("checkpoints"))

        loader = nodes.CheckpointLoaderSimple()
        return loader.load_checkpoint(ckpt_name, output_vae, output_clip)

```
