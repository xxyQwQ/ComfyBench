---
tags:
- SAM
---

# SAM Loader
## Documentation
- Class name: `AV_SAMLoader`
- Category: `Art Venture/Segmentation`
- Output node: `False`

The AV_SAMLoader node is designed for loading SAM models specific to the Art Venture project. It facilitates the retrieval and initialization of SAM models from a predefined directory, ensuring that the models are ready for use in segmentation tasks.
## Input types
### Required
- **`model_name`**
    - Specifies the name of the SAM model to load. This parameter is crucial for identifying and retrieving the correct model file from the available SAM models directory.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`sam_model`**
    - Comfy dtype: `AV_SAM_MODEL`
    - Returns the loaded SAM model, ready for use in segmentation tasks.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SAMLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("sams"),),
            }
        }

    RETURN_TYPES = ("AV_SAM_MODEL",)
    RETURN_NAMES = ("sam_model",)
    FUNCTION = "load_model"
    CATEGORY = "Art Venture/Segmentation"

    def load_model(self, model_name):
        modelname = folder_paths.get_full_path("sams", model_name)

        state_dict = comfy.utils.load_torch_file(modelname)
        encoder_size = state_dict["image_encoder.patch_embed.proj.bias"].shape[0]

        if encoder_size == 1280:
            model_kind = "vit_h"
        elif encoder_size == 1024:
            model_kind = "vit_l"
        else:
            model_kind = "vit_b"

        ensure_package("segment_anything")
        from segment_anything import sam_model_registry

        sam = sam_model_registry[model_kind]()
        sam.load_state_dict(state_dict)

        return (sam,)

```
