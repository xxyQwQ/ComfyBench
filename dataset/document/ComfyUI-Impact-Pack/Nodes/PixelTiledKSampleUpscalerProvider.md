---
tags:
- ImageScaling
- ImageUpscaling
- Upscale
---

# PixelTiledKSampleUpscalerProvider
## Documentation
- Class name: `PixelTiledKSampleUpscalerProvider`
- Category: `ImpactPack/Upscale`
- Output node: `False`

This node provides a mechanism for upscaling images using a tiled K-sample approach, leveraging multiple samples per tile to enhance image quality and detail. It integrates with external sampling extensions to facilitate the tiling process, optimizing performance and output quality for high-resolution image generation.
## Input types
### Required
- **`scale_method`**
    - Specifies the method used for scaling the image, affecting the overall upscaling process and the quality of the output image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`model`**
    - Defines the model used for generating the samples, directly influencing the style and characteristics of the upscaled image.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
- **`vae`**
    - The variational autoencoder used in conjunction with the model to improve image quality through latent space manipulation.
    - Comfy dtype: `VAE`
    - Python dtype: `str`
- **`seed`**
    - Determines the random seed for sample generation, ensuring reproducibility of the results.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - The number of steps to perform in the sampling process, impacting the detail and quality of the final image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Controls the conditioning factor, adjusting the influence of the guidance on the sampling process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - Identifies the specific sampler to be used, affecting the sampling strategy and final image characteristics.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Specifies the scheduler for controlling the sampling process, optimizing the balance between quality and performance.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`positive`**
    - A positive text prompt to guide the image generation towards desired attributes.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`negative`**
    - A negative text prompt to steer the image generation away from undesired attributes.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`denoise`**
    - The level of denoising applied to the samples, refining the clarity and quality of the upscaled image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`tile_width`**
    - The width of each tile in the tiling strategy, determining the granularity of the upscaling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`tile_height`**
    - The height of each tile, which, along with tile width, defines the tiling pattern and affects the upscaling precision.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`tiling_strategy`**
    - Describes the strategy for laying out tiles during the upscaling process, influencing the overall efficiency and quality of the output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`upscale_model_opt`**
    - Optional upscale model to enhance the upscaling process with a specific algorithm or technique.
    - Comfy dtype: `UPSCALE_MODEL`
    - Python dtype: `str`
- **`pk_hook_opt`**
    - Optional post-processing hook to apply custom operations or adjustments after the upscaling process.
    - Comfy dtype: `PK_HOOK`
    - Python dtype: `str`
- **`tile_cnet_opt`**
    - Optional control network to influence the tiling process, potentially improving the coherence and quality of the upscaled image.
    - Comfy dtype: `CONTROL_NET`
    - Python dtype: `str`
- **`tile_cnet_strength`**
    - Determines the strength of the control network's influence on the tiling and upscaling process, allowing for fine-tuning of the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`upscaler`**
    - Comfy dtype: `UPSCALER`
    - The result of the upscaling process, encapsulating the enhanced image data ready for further processing or display.
    - Python dtype: `object`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class PixelTiledKSampleUpscalerProvider:
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
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "tile_width": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                    "tile_height": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                    "tiling_strategy": (["random", "padded", 'simple'], ),
                    },
                "optional": {
                        "upscale_model_opt": ("UPSCALE_MODEL", ),
                        "pk_hook_opt": ("PK_HOOK", ),
                        "tile_cnet_opt": ("CONTROL_NET", ),
                        "tile_cnet_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    }
                }

    RETURN_TYPES = ("UPSCALER",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Upscale"

    def doit(self, scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, tile_width, tile_height, tiling_strategy, upscale_model_opt=None,
             pk_hook_opt=None, tile_cnet_opt=None, tile_cnet_strength=1.0):
        if "BNK_TiledKSampler" in nodes.NODE_CLASS_MAPPINGS:
            upscaler = core.PixelTiledKSampleUpscaler(scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise,
                                                      tile_width, tile_height, tiling_strategy, upscale_model_opt, pk_hook_opt, tile_cnet_opt,
                                                      tile_size=max(tile_width, tile_height), tile_cnet_strength=tile_cnet_strength)
            return (upscaler, )
        else:
            print("[ERROR] PixelTiledKSampleUpscalerProvider: ComfyUI_TiledKSampler custom node isn't installed. You must install BlenderNeko/ComfyUI_TiledKSampler extension to use this node.")

```
