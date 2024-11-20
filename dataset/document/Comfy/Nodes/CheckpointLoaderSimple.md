---
tags:
- Checkpoint
- CheckpointLoader
- Loader
- ModelIO
- ModelLoader
---

# Load Checkpoint
## Documentation
- Class name: `CheckpointLoaderSimple`
- Category: `loaders`
- Output node: `False`

The CheckpointLoaderSimple node is designed for loading model checkpoints without requiring a configuration name. It simplifies the checkpoint loading process by automatically guessing the appropriate configuration based on the checkpoint name, making it easier to use for quick model setups or testing.
## Input types
### Required
- **`ckpt_name`**
    - Specifies the name of the checkpoint to load. This parameter is crucial for identifying which checkpoint file to retrieve and use for model initialization.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - Returns the main model loaded from the specified checkpoint.
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
- Infra type: `GPU`
- Common nodes:
    - [CLIPTextEncode](../../Comfy/Nodes/CLIPTextEncode.md)
    - [VAEDecode](../../Comfy/Nodes/VAEDecode.md)
    - [LoraLoader](../../Comfy/Nodes/LoraLoader.md)
    - [CLIPSetLastLayer](../../Comfy/Nodes/CLIPSetLastLayer.md)
    - Reroute
    - [CR Apply LoRA Stack](../../ComfyUI_Comfyroll_CustomNodes/Nodes/CR Apply LoRA Stack.md)
    - [VAEEncode](../../Comfy/Nodes/VAEEncode.md)
    - [KSampler](../../Comfy/Nodes/KSampler.md)
    - [UltimateSDUpscale](../../ComfyUI_UltimateSDUpscale/Nodes/UltimateSDUpscale.md)



## Source code
```python
class CheckpointLoaderSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                             }}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "loaders"

    def load_checkpoint(self, ckpt_name):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return out[:3]

```
