# TripleCLIPLoader
## Documentation
- Class name: `TripleCLIPLoader`
- Category: `advanced/loaders`
- Output node: `False`

The TripleCLIPLoader node is designed for loading three distinct CLIP models simultaneously, facilitating advanced applications that require the integration of multiple CLIP embeddings or features for enhanced content understanding or generation.
## Input types
### Required
- **`clip_name1`**
    - Specifies the first CLIP model to load, playing a crucial role in the composite CLIP model setup by contributing its unique embeddings or features.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`clip_name2`**
    - Defines the second CLIP model to load, adding another layer of complexity and capability to the overall CLIP model integration.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`clip_name3`**
    - Indicates the third CLIP model to load, further enriching the integrated CLIP model's ability to understand or generate content with diverse embeddings or features.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`clip`**
    - Comfy dtype: `CLIP`
    - Outputs a composite CLIP model that integrates the embeddings or features from the three loaded CLIP models, enabling advanced content understanding or generation tasks.
    - Python dtype: `comfy.sd.CLIP`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class TripleCLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name1": (folder_paths.get_filename_list("clip"), ), "clip_name2": (folder_paths.get_filename_list("clip"), ), "clip_name3": (folder_paths.get_filename_list("clip"), )
                             }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"

    CATEGORY = "advanced/loaders"

    def load_clip(self, clip_name1, clip_name2, clip_name3):
        clip_path1 = folder_paths.get_full_path("clip", clip_name1)
        clip_path2 = folder_paths.get_full_path("clip", clip_name2)
        clip_path3 = folder_paths.get_full_path("clip", clip_name3)
        clip = comfy.sd.load_clip(ckpt_paths=[clip_path1, clip_path2, clip_path3], embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return (clip,)

```
