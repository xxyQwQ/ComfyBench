---
tags:
- ModelGuidance
---

# Automatic CFG
## Documentation
- Class name: `Automatic CFG`
- Category: `model_patches/Automatic_CFG/presets`
- Output node: `False`

The Automatic CFG node is designed to dynamically modify the configuration of models to optimize their performance for specific tasks. It leverages advanced techniques to adjust model parameters and settings based on the input conditions, aiming to enhance model accuracy, speed, or other performance metrics.
## Input types
### Required
- **`model`**
    - The 'model' parameter represents the model to be modified. It is crucial for the node's operation as it serves as the base upon which modifications are applied to achieve the desired configuration.
    - Comfy dtype: `MODEL`
    - Python dtype: `object`
- **`hard_mode`**
    - The 'hard_mode' parameter dictates the strictness of the configuration adjustments. When set to True, it applies more rigorous modifications to the model, potentially leading to higher performance at the cost of flexibility.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`boost`**
    - The 'boost' parameter enables or disables performance enhancements. When True, it activates additional modifications that can further improve the model's performance.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The modified model with optimized configuration settings. This output reflects the adjustments made based on the input parameters and the desired performance enhancements.
    - Python dtype: `object`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class simpleDynamicCFG:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "hard_mode" : ("BOOLEAN", {"default": True}),
                                "boost" : ("BOOLEAN", {"default": True}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Automatic_CFG/presets"

    def patch(self, model, hard_mode, boost):
        advcfg = advancedDynamicCFG()
        m = advcfg.patch(model,
                         skip_uncond = boost,
                         uncond_sigma_start = 1000,  uncond_sigma_end = 1,
                         automatic_cfg = "hard" if hard_mode else "soft"
                         )[0]
        return (m, )

```
