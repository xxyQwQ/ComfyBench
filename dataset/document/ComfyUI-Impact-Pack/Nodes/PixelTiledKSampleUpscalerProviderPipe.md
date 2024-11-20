---
tags:
- ImageScaling
- ImageUpscaling
- Upscale
---

# PixelTiledKSampleUpscalerProviderPipe
## Documentation
- Class name: `PixelTiledKSampleUpscalerProviderPipe`
- Category: `ImpactPack/Upscale`
- Output node: `False`

This node provides a pipeline for upscaling images using a tiled K-sample upscaling method. It integrates various components such as model selection, VAE, seed, steps, configuration, sampler name, scheduler, denoising, and tiling strategy to perform image upscaling. The node is designed to work with specific upscale models and supports customization through options like upscale model optimization, post-kernel hooks, and tile convolution network options.
## Input types
### Required
- **`scale_method`**
    - Specifies the method used for scaling the image, affecting the overall upscaling process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`seed`**
    - Determines the random seed used for generating samples, influencing the upscaling's randomness and reproducibility.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - Defines the number of steps to perform in the upscaling process, impacting the detail and quality of the output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Configuration parameter that influences the sampling behavior and the quality of the generated image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - The name of the sampler used in the upscaling process, affecting the sampling strategy and output quality.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Scheduler used to adjust the sampling process over the steps, influencing the progression of detail enhancement.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`denoise`**
    - Indicates whether denoising is applied during the upscaling process, affecting the clarity and quality of the output image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `bool`
- **`tile_width`**
    - The width of the tiles used in the tiling strategy, affecting the granularity of processing and potentially the output quality.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`tile_height`**
    - The height of the tiles used in the tiling strategy, similar to tile width, it affects the granularity of processing.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`tiling_strategy`**
    - The strategy used for tiling the image during upscaling, influencing how the image is divided and processed.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`basic_pipe`**
    - A pipeline of basic components required for the upscaling process, including model, VAE, and other essential elements.
    - Comfy dtype: `BASIC_PIPE`
    - Python dtype: `tuple`
### Optional
- **`upscale_model_opt`**
    - Optional parameter for optimizing the upscale model, allowing for customization of the upscaling process.
    - Comfy dtype: `UPSCALE_MODEL`
    - Python dtype: `dict`
- **`pk_hook_opt`**
    - Optional parameter for post-kernel hooks, enabling further customization of the upscaling process.
    - Comfy dtype: `PK_HOOK`
    - Python dtype: `dict`
- **`tile_cnet_opt`**
    - Optional parameter for tile convolution network options, allowing for adjustments in how tiles are processed.
    - Comfy dtype: `CONTROL_NET`
    - Python dtype: `dict`
- **`tile_cnet_strength`**
    - Strength of the tile convolution network, affecting the intensity of processing on each tile.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`upscaler`**
    - Comfy dtype: `UPSCALER`
    - The upscaler object configured with the provided parameters, ready to perform image upscaling.
    - Python dtype: `PixelTiledKSampleUpscaler`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class PixelTiledKSampleUpscalerProviderPipe:
    upscale_methods = ["nearest-exact", "bilinear", "lanczos", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "scale_method": (s.upscale_methods,),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "tile_width": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                    "tile_height": ("INT", {"default": 512, "min": 320, "max": MAX_RESOLUTION, "step": 64}),
                    "tiling_strategy": (["random", "padded", 'simple'], ),
                    "basic_pipe": ("BASIC_PIPE",)
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

    def doit(self, scale_method, seed, steps, cfg, sampler_name, scheduler, denoise, tile_width, tile_height, tiling_strategy, basic_pipe, upscale_model_opt=None, pk_hook_opt=None,
             tile_cnet_opt=None, tile_cnet_strength=1.0):
        if "BNK_TiledKSampler" in nodes.NODE_CLASS_MAPPINGS:
            model, _, vae, positive, negative = basic_pipe
            upscaler = core.PixelTiledKSampleUpscaler(scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise,
                                                      tile_width, tile_height, tiling_strategy, upscale_model_opt, pk_hook_opt, tile_cnet_opt,
                                                      tile_size=max(tile_width, tile_height), tile_cnet_strength=tile_cnet_strength)
            return (upscaler, )
        else:
            print("[ERROR] PixelTiledKSampleUpscalerProviderPipe: ComfyUI_TiledKSampler custom node isn't installed. You must install BlenderNeko/ComfyUI_TiledKSampler extension to use this node.")

```
