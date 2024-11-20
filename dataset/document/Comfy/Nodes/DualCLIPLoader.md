---
tags:
- CLIP
- Loader
- ModelIO
---

# DualCLIPLoader
## Documentation
- Class name: `DualCLIPLoader`
- Category: `advanced/loaders`
- Output node: `False`

The DualCLIPLoader node is designed for loading two CLIP models simultaneously, based on their names and a specified type. It supports loading models for different versions of the Stable Diffusion architecture, enhancing flexibility in model usage for various applications.
## Input types
### Required
- **`clip_name1`**
    - Specifies the name of the first CLIP model to load. The choice of model affects the outcome and capabilities of the loaded model.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`clip_name2`**
    - Specifies the name of the second CLIP model to load, allowing for dual model loading to accommodate complex or comparative tasks.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`type`**
    - Determines the type of CLIP models to load, supporting either 'sdxl' or 'sd3' versions, which influences the model's functionality and compatibility.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`clip`**
    - Comfy dtype: `CLIP`
    - Returns a combined CLIP model object, integrating the functionalities of the two specified CLIP models.
    - Python dtype: `comfy.sd.CLIP`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class DualCLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name1": (folder_paths.get_filename_list("clip"), ),
                              "clip_name2": (folder_paths.get_filename_list("clip"), ),
                              "type": (["sdxl", "sd3"], ),
                             }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"

    CATEGORY = "advanced/loaders"

    def load_clip(self, clip_name1, clip_name2, type):
        clip_path1 = folder_paths.get_full_path("clip", clip_name1)
        clip_path2 = folder_paths.get_full_path("clip", clip_name2)
        if type == "sdxl":
            clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION
        elif type == "sd3":
            clip_type = comfy.sd.CLIPType.SD3

        clip = comfy.sd.load_clip(ckpt_paths=[clip_path1, clip_path2], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type)
        return (clip,)

```
