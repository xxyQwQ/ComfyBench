---
tags:
- ImageScaling
- ImageUpscaling
- Upscale
---

# PixelKSampleUpscalerProviderPipe
## Documentation
- Class name: `PixelKSampleUpscalerProviderPipe`
- Category: `ImpactPack/Upscale`
- Output node: `False`

This node is designed to provide an upscaling service by leveraging the PixelKSample upscaling technique within a pipeline structure. It extends the capabilities of the PixelKSampleUpscalerProvider by adapting its functionality for use in a more complex, pipelined environment, allowing for seamless integration and enhanced scalability in image processing workflows.
## Input types
### Required
- **`scale_method`**
    - Specifies the method used for upscaling, allowing selection from predefined methods such as 'nearest-exact', 'bilinear', 'lanczos', and 'area'. This choice directly influences the quality and characteristics of the upscaled image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`seed`**
    - Determines the random seed for generating consistent results across upscaling sessions, ensuring reproducibility.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - Defines the number of steps to be used in the upscaling process, affecting the detail and quality of the output image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Specifies the configuration for the upscaling process, influencing the balance between fidelity to the original image and the introduction of new details.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - Selects the sampler to be used in the upscaling process, influencing the initial phase of image generation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Chooses the scheduling strategy for sampling during the upscaling process, affecting the distribution and application of samples.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`denoise`**
    - Sets the denoising level to be applied during the upscaling process, potentially improving image clarity at the expense of some detail.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`use_tiled_vae`**
    - A boolean parameter that enables or disables the use of tiled VAEs, impacting the upscaling process by potentially enhancing detail at the cost of increased computational requirements.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`basic_pipe`**
    - Integrates the node within a basic pipeline structure, facilitating its combination with other processing steps.
    - Comfy dtype: `BASIC_PIPE`
    - Python dtype: `str`
- **`tile_size`**
    - Defines the size of the tiles used in tiled VAE upscaling, affecting the granularity of the upscaling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`upscale_model_opt`**
    - Optional parameter for selecting a specific upscale model, providing flexibility in choosing the algorithm for upscaling.
    - Comfy dtype: `UPSCALE_MODEL`
    - Python dtype: `str`
- **`pk_hook_opt`**
    - Optional parameter for specifying post-processing hooks, which can modify the upscaling results or apply additional effects.
    - Comfy dtype: `PK_HOOK`
    - Python dtype: `str`
- **`scheduler_func_opt`**
    - Optional parameter for specifying a custom scheduling function, offering additional control over the sampling process.
    - Comfy dtype: `SCHEDULER_FUNC`
    - Python dtype: `str`
- **`tile_cnet_opt`**
    - Optional parameter for specifying a control network option, which can influence the upscaling process by adjusting the content in a targeted manner.
    - Comfy dtype: `CONTROL_NET`
    - Python dtype: `str`
- **`tile_cnet_strength`**
    - Determines the strength of the control network's effect on the upscaling process, allowing for fine-tuning of the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`upscaler`**
    - Comfy dtype: `UPSCALER`
    - Produces an upscaled version of the input image, utilizing the specified upscaling method and other parameters to enhance image resolution and detail.
    - Python dtype: `object`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [IterativeImageUpscale](../../ComfyUI-Impact-Pack/Nodes/IterativeImageUpscale.md)



## Source code
```python
class PixelKSampleUpscalerProviderPipe(PixelKSampleUpscalerProvider):
    upscale_methods = ["nearest-exact", "bilinear", "lanczos", "area"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "scale_method": (s.upscale_methods,),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (core.SCHEDULERS, ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "use_tiled_vae": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "basic_pipe": ("BASIC_PIPE",),
                    "tile_size": ("INT", {"default": 512, "min": 320, "max": 4096, "step": 64}),
                    },
                "optional": {
                        "upscale_model_opt": ("UPSCALE_MODEL", ),
                        "pk_hook_opt": ("PK_HOOK", ),
                        "scheduler_func_opt": ("SCHEDULER_FUNC",),
                        "tile_cnet_opt": ("CONTROL_NET", ),
                        "tile_cnet_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    }
                }

    RETURN_TYPES = ("UPSCALER",)
    FUNCTION = "doit_pipe"

    CATEGORY = "ImpactPack/Upscale"

    def doit_pipe(self, scale_method, seed, steps, cfg, sampler_name, scheduler, denoise,
                  use_tiled_vae, basic_pipe, upscale_model_opt=None, pk_hook_opt=None,
                  tile_size=512, scheduler_func_opt=None, tile_cnet_opt=None, tile_cnet_strength=1.0):
        model, _, vae, positive, negative = basic_pipe
        upscaler = core.PixelKSampleUpscaler(scale_method, model, vae, seed, steps, cfg, sampler_name, scheduler,
                                             positive, negative, denoise, use_tiled_vae, upscale_model_opt, pk_hook_opt,
                                             tile_size=tile_size, scheduler_func=scheduler_func_opt,
                                             tile_cnet_opt=tile_cnet_opt, tile_cnet_strength=tile_cnet_strength)
        return (upscaler, )

```
