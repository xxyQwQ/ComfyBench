---
tags:
- ImageScaling
- ImageUpscaling
- Upscale
---

# Upscaler (SEGS/pipe)
## Documentation
- Class name: `SEGSUpscalerPipe`
- Category: `ImpactPack/Upscale`
- Output node: `False`

The SEGSUpscalerPipe node is designed for upscaling images using segmentation maps as guidance. It integrates with a basic pipeline to enhance image resolution while considering the segmentation details, ensuring that the upscale process respects the original image's segmented regions.
## Input types
### Required
- **`image`**
    - The original image to be upscaled. It serves as the primary input for the upscaling process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`segs`**
    - Segmentation maps corresponding to the original image. These maps guide the upscaling process to maintain fidelity to the segmented regions.
    - Comfy dtype: `SEGS`
    - Python dtype: `torch.Tensor`
- **`basic_pipe`**
    - A collection of models and configurations used in the upscaling process, including model, clip, vae, and settings for sampling and scheduling.
    - Comfy dtype: `BASIC_PIPE`
    - Python dtype: `Tuple[torch.nn.Module, Any, torch.nn.Module, Dict]`
- **`rescale_factor`**
    - The factor by which the image will be upscaled. It determines the increase in resolution.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`resampling_method`**
    - The method used for resampling during the upscaling process, affecting the quality of the upscaled image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`supersample`**
    - A boolean indicating whether to apply supersampling, which can enhance the upscaling quality by reducing aliasing.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `bool`
- **`rounding_modulus`**
    - A value used to round the upscaled image dimensions, ensuring they are multiples of this modulus.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`seed`**
    - The seed for random number generation, ensuring reproducibility of the upscaling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - The number of steps to perform in the upscaling process, affecting the detail and quality of the result.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Configuration settings for the upscaling process, including parameters for the model and sampling.
    - Comfy dtype: `FLOAT`
    - Python dtype: `Dict`
- **`sampler_name`**
    - The name of the sampler to use during upscaling, influencing the texture and details of the upscaled image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - The scheduler used to adjust the learning rate during the upscaling process, affecting the convergence and quality.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `Any`
- **`denoise`**
    - A boolean indicating whether to apply denoising, which can improve the visual quality of the upscaled image by reducing noise.
    - Comfy dtype: `FLOAT`
    - Python dtype: `bool`
- **`feather`**
    - The feathering amount applied to the edges of the segmentation maps, smoothing transitions between segments.
    - Comfy dtype: `INT`
    - Python dtype: `float`
- **`inpaint_model`**
    - The model used for inpainting missing or uncertain areas in the upscaled image, enhancing overall quality.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `torch.nn.Module`
- **`noise_mask_feather`**
    - The amount of feathering applied to the noise mask, affecting the blending of noise-reduced areas.
    - Comfy dtype: `INT`
    - Python dtype: `float`
### Optional
- **`upscale_model_opt`**
    - Optional configurations for the upscale model, allowing customization of the upscaling process.
    - Comfy dtype: `UPSCALE_MODEL`
    - Python dtype: `Dict`
- **`upscaler_hook_opt`**
    - Optional hooks to modify the behavior of the upscaler, enabling advanced customization.
    - Comfy dtype: `UPSCALER_HOOK`
    - Python dtype: `Dict`
- **`scheduler_func_opt`**
    - Optional scheduler functions to further customize the learning rate adjustment during upscaling.
    - Comfy dtype: `SCHEDULER_FUNC`
    - Python dtype: `Dict`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The upscaled image, enhanced in resolution while preserving the details and regions defined by the segmentation maps.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SEGSUpscalerPipe:
    @classmethod
    def INPUT_TYPES(s):
        resampling_methods = ["lanczos", "nearest", "bilinear", "bicubic"]

        return {"required": {
                    "image": ("IMAGE",),
                    "segs": ("SEGS",),
                    "basic_pipe": ("BASIC_PIPE",),
                    "rescale_factor": ("FLOAT", {"default": 2, "min": 0.01, "max": 100.0, "step": 0.01}),
                    "resampling_method": (resampling_methods,),
                    "supersample": (["true", "false"],),
                    "rounding_modulus": ("INT", {"default": 8, "min": 8, "max": 1024, "step": 8}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                    "scheduler": (core.SCHEDULERS,),
                    "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
                    "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                    "inpaint_model": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "noise_mask_feather": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
                    },
                "optional": {
                    "upscale_model_opt": ("UPSCALE_MODEL",),
                    "upscaler_hook_opt": ("UPSCALER_HOOK",),
                    "scheduler_func_opt": ("SCHEDULER_FUNC",),
                    }
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Upscale"

    @staticmethod
    def doit(image, segs, basic_pipe, rescale_factor, resampling_method, supersample, rounding_modulus,
             seed, steps, cfg, sampler_name, scheduler, denoise, feather, inpaint_model, noise_mask_feather,
             upscale_model_opt=None, upscaler_hook_opt=None, scheduler_func_opt=None):

        model, clip, vae, positive, negative = basic_pipe

        return SEGSUpscaler.doit(image, segs, model, clip, vae, rescale_factor, resampling_method, supersample, rounding_modulus,
                                 seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, feather, inpaint_model, noise_mask_feather,
                                 upscale_model_opt=upscale_model_opt, upscaler_hook_opt=upscaler_hook_opt, scheduler_func_opt=scheduler_func_opt)

```
