# Automatic CFG - Unpatch function(Deprecated)
## Documentation
- Class name: `Automatic CFG - Unpatch function`
- Category: `model_patches/Automatic_CFG/utils`
- Output node: `False`

This node is designed to revert any previously applied configuration patches to a model, effectively restoring it to its original state without the specific adjustments made for automatic control flow graph (CFG) manipulation.
## Input types
### Required
- **`model`**
    - The model to be unpatched. This input is crucial as it specifies the model from which the automatic CFG-related patches will be removed, ensuring the model is returned to its initial configuration.
    - Comfy dtype: `MODEL`
    - Python dtype: `CustomModel`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The unpatched model, returned after removing specific automatic CFG-related patches, restoring it to its original, unmodified state.
    - Python dtype: `CustomModel`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class simpleDynamicCFGunpatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "unpatch"

    CATEGORY = "model_patches/Automatic_CFG/utils"

    def unpatch(self, model):
        m = model.clone()
        m.model_options.pop("sampler_pre_cfg_function", None)
        return (m, )

```
