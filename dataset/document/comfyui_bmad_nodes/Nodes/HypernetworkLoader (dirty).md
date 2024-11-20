---
tags:
- Loader
- ModelIO
- ModelLoader
---

# HypernetworkLoader (dirty)
## Documentation
- Class name: `HypernetworkLoader (dirty)`
- Category: `Bmad/api/dirty loaders`
- Output node: `False`

The HypernetworkLoader node is designed to enhance or modify the capabilities of a given model by applying a hypernetwork patch. This process involves loading a specified hypernetwork and applying it to the model with a certain strength, thereby potentially altering the model's behavior or performance based on the characteristics of the hypernetwork.
## Input types
### Required
- **`model`**
    - The model to which the hypernetwork patch will be applied. This parameter is crucial as it determines the base model that will be enhanced or modified by the hypernetwork.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`hypernetwork_name`**
    - The name of the hypernetwork to be loaded and applied. This parameter specifies which hypernetwork patch will be used to modify the model.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`strength`**
    - The strength with which the hypernetwork patch will be applied to the model. This affects the intensity of the modification or enhancement.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The modified model after the hypernetwork patch has been applied. This output reflects the changes made to the original model by the hypernetwork.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class DirtyHypernetworkLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",),
                             "hypernetwork_name": ("STRING", {"default": ""}),
                             "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_hypernetwork"

    CATEGORY = f"{api_category_path}/dirty loaders"

    def load_hypernetwork(self, model, hypernetwork_name, strength):
        hypernetwork_name = DirtyLoaderUtils.find_matching_filename(
            hypernetwork_name, folder_paths.get_filename_list("hypernetworks"))

        loader = hyper.HypernetworkLoader()
        return loader.load_hypernetwork(model, hypernetwork_name, strength)

```
