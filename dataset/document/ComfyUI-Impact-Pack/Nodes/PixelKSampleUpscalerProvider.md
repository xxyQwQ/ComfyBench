---
tags:
- ImageScaling
- ImageUpscaling
- Upscale
---

# PixelKSampleUpscalerProvider
## Documentation
- Class name: `PixelKSampleUpscalerProvider`
- Category: `ImpactPack/Upscale`
- Output node: `False`

This node specializes in upscaling images through a K-sample strategy, leveraging advanced sampling techniques to enhance pixel quality. It simplifies the application of sophisticated upscaling algorithms, aiming to produce higher resolution images with greater detail and clarity.
## Input types
### Required
- **`scale_method`**
    - Specifies the method used for upscaling, allowing selection from predefined algorithms to tailor the upscaling process to specific needs.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`model`**
    - Identifies the model used for upscaling, enabling the application of specific models tailored to enhance image quality.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
- **`vae`**
    - Specifies the variational autoencoder used in the upscaling process, contributing to the generation of high-quality images.
    - Comfy dtype: `VAE`
    - Python dtype: `str`
- **`seed`**
    - Sets the seed for random number generation, ensuring reproducibility of the upscaling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - Defines the number of steps in the upscaling process, allowing control over the refinement level of the upscaled image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Configures the conditioning factor, influencing the upscaling process by adjusting the balance between the input image and the generated details.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - Specifies the sampler used in the upscaling process, affecting the sampling strategy and ultimately the quality of the upscaled image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Determines the scheduler for the upscaling process, managing the progression of steps to optimize image quality.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`positive`**
    - Represents positive conditioning data that guides the upscaling towards desired outcomes, emphasizing certain visual features.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`negative`**
    - Represents negative conditioning data that the upscaling process should minimize, steering the algorithm away from undesired features.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`denoise`**
    - Controls the level of denoising applied during upscaling, affecting the clarity and detail of the upscaled image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`use_tiled_vae`**
    - Indicates whether a tiled VAE is used, affecting the handling of large images by breaking them into manageable tiles for processing.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`tile_size`**
    - Specifies the size of tiles when using a tiled VAE, impacting the processing of large images by dividing them into smaller, manageable sections.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`upscale_model_opt`**
    - Optionally specifies an upscale model to be used, allowing for customization of the upscaling process with specific models.
    - Comfy dtype: `UPSCALE_MODEL`
    - Python dtype: `str`
- **`pk_hook_opt`**
    - Optionally specifies a post-processing hook, enabling custom post-upscaling adjustments.
    - Comfy dtype: `PK_HOOK`
    - Python dtype: `str`
- **`scheduler_func_opt`**
    - Optionally specifies a scheduler function, allowing for advanced control over the scheduling of upscaling steps.
    - Comfy dtype: `SCHEDULER_FUNC`
    - Python dtype: `str`
## Output types
- **`upscaler`**
    - Comfy dtype: `UPSCALER`
    - Produces the upscaled image, showcasing the enhanced resolution and detail achieved through the upscaling process.
    - Python dtype: `str`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [IterativeLatentUpscale](../../ComfyUI-Impact-Pack/Nodes/IterativeLatentUpscale.md)



## Source code
```python
class PixelKSampleUpscalerProvider:
    upscale_methods = ["nearest-exact", "bilinear", "lanczos", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "scale_method": (s.upscale_methods,),
                    "model": ("MODEL",),
                    "vae": ("VAE",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (core.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "use_tiled_vae": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "tile_size": ("INT", {"default": 512, "min": 320, "max": 4096, "step": 64}),
                    },
                "optional": {
                        "upscale_model_opt": ("UPSCALE_MODEL", ),
                        "pk_hook_opt": ("PK_HOOK", ),
                        "scheduler_func_opt": ("SCHEDULER_FUNC",),
                    }
                }

    RETURN_TYPES = ("UPSCALER",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Upscale"

    def doit(self, scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise,
             use_tiled_vae, upscale_model_opt=None, pk_hook_opt=None, tile_size=512, scheduler_func_opt=None):
        upscaler = core.PixelKSampleUpscaler(scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler,
                                             positive, negative, denoise, use_tiled_vae, upscale_model_opt, pk_hook_opt,
                                             tile_size=tile_size, scheduler_func=scheduler_func_opt)
        return (upscaler, )

```
