---
tags:
- ModelGuidance
---

# Automatic CFG - Post rescale only
## Documentation
- Class name: `Automatic CFG - Post rescale only`
- Category: `model_patches/Automatic_CFG/utils`
- Output node: `False`

This node applies a post-conditional generative feedback (CFG) rescaling operation to adjust the balance between the original and degraded image features based on a scaling factor derived from the difference between the original and a secondary image (sag). It dynamically adjusts the scaling of the added CFG effect, aiming to enhance image restoration or modification tasks by recalibrating the intensity of changes applied to the original image, ensuring a more controlled and precise enhancement.
## Input types
### Required
- **`model`**
    - The model to which the post-CFG rescaling operation will be applied, serving as the base for further adjustments.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`subtract_latent_mean`**
    - A boolean flag indicating whether to subtract the mean of the latent space representation.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`subtract_latent_mean_sigma_start`**
    - The starting value of sigma for which subtracting the latent mean is applicable.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`subtract_latent_mean_sigma_end`**
    - The ending value of sigma for which subtracting the latent mean is applicable.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`latent_intensity_rescale`**
    - A boolean flag indicating whether to apply intensity rescaling to the latent space representation.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`latent_intensity_rescale_method`**
    - The method used for rescaling the intensity of the latent space representation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`latent_intensity_rescale_cfg`**
    - The configuration parameters for the latent intensity rescale operation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`latent_intensity_rescale_sigma_start`**
    - The starting value of sigma for which the latent intensity rescale is applicable.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`latent_intensity_rescale_sigma_end`**
    - The ending value of sigma for which the latent intensity rescale is applicable.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The model after applying the dynamic rescaling of the CFG effect, reflecting a balanced enhancement.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class postCFGrescaleOnly:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "subtract_latent_mean" : ("BOOLEAN", {"default": True}),
                                "subtract_latent_mean_sigma_start": ("FLOAT", {"default": 1000,  "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.1}),
                                "subtract_latent_mean_sigma_end":   ("FLOAT", {"default": 7.5, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.1}),
                                "latent_intensity_rescale"     : ("BOOLEAN", {"default": True}),
                                "latent_intensity_rescale_method" : (["soft","hard","range"], {"default": "hard"},),
                                "latent_intensity_rescale_cfg" : ("FLOAT", {"default": 8,  "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.1}),
                                "latent_intensity_rescale_sigma_start": ("FLOAT", {"default": 1000,  "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.1}),
                                "latent_intensity_rescale_sigma_end":   ("FLOAT", {"default": 5, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.1}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Automatic_CFG/utils"

    def patch(self, model,
              subtract_latent_mean, subtract_latent_mean_sigma_start, subtract_latent_mean_sigma_end,
              latent_intensity_rescale, latent_intensity_rescale_method, latent_intensity_rescale_cfg, latent_intensity_rescale_sigma_start, latent_intensity_rescale_sigma_end
              ):
        advcfg = advancedDynamicCFG()
        m = advcfg.patch(model=model, 
                         subtract_latent_mean = subtract_latent_mean,
                         subtract_latent_mean_sigma_start = subtract_latent_mean_sigma_start, subtract_latent_mean_sigma_end = subtract_latent_mean_sigma_end,
                         latent_intensity_rescale = latent_intensity_rescale, latent_intensity_rescale_cfg = latent_intensity_rescale_cfg, latent_intensity_rescale_method = latent_intensity_rescale_method,
                         latent_intensity_rescale_sigma_start = latent_intensity_rescale_sigma_start, latent_intensity_rescale_sigma_end = latent_intensity_rescale_sigma_end,
                         ignore_pre_cfg_func = True
                         )[0]
        return (m, )

```
