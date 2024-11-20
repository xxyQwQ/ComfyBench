---
tags:
- CLIP
- Loader
- ModelIO
---

# Load CLIP
## Documentation
- Class name: `CLIPLoader`
- Category: `advanced/loaders`
- Output node: `False`

The CLIPLoader node is designed for loading CLIP models, supporting various types including stable diffusion, stable cascade, SD3, and stable audio. It abstracts the complexities of loading and configuring CLIP models based on the specified type, facilitating their use in downstream tasks.
## Input types
### Required
- **`clip_name`**
    - Specifies the name of the CLIP model to be loaded. This name is used to locate the model file within a predefined directory structure.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`type`**
    - Determines the specific type of CLIP model to load, such as stable diffusion, stable cascade, SD3, or stable audio. This affects how the model is initialized and configured.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`clip`**
    - Comfy dtype: `CLIP`
    - The loaded CLIP model, ready for use in various applications such as image generation or text-to-image tasks.
    - Python dtype: `comfy.sd.CLIPModel`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class CLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name": (folder_paths.get_filename_list("clip"), ),
                              "type": (["stable_diffusion", "stable_cascade", "sd3", "stable_audio"], ),
                             }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"

    CATEGORY = "advanced/loaders"

    def load_clip(self, clip_name, type="stable_diffusion"):
        if type == "stable_cascade":
            clip_type = comfy.sd.CLIPType.STABLE_CASCADE
        elif type == "sd3":
            clip_type = comfy.sd.CLIPType.SD3
        elif type == "stable_audio":
            clip_type = comfy.sd.CLIPType.STABLE_AUDIO
        else:
            clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION

        clip_path = folder_paths.get_full_path("clip", clip_name)
        clip = comfy.sd.load_clip(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type)
        return (clip,)

```
