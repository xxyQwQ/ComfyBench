---
tags:
- Sampling
---

# KSampler (pipe)
## Documentation
- Class name: `ImpactKSamplerBasicPipe`
- Category: `ImpactPack/sampling`
- Output node: `False`

The ImpactKSamplerBasicPipe node is designed for advanced sampling techniques within the ComfyUI Impact Pack. It leverages a basic sampling pipeline to generate or modify latent images based on specified parameters, such as noise addition and classifier guidance, to achieve desired visual effects or enhancements.
## Input types
### Required
- **`basic_pipe`**
    - Represents the core components required for sampling, including models and configurations necessary for the operation.
    - Comfy dtype: `BASIC_PIPE`
    - Python dtype: `tuple`
- **`seed`**
    - Used to initialize the random number generator for reproducible noise generation during sampling.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - Defines the total number of steps to perform during the sampling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Specifies the classifier free guidance value, influencing the direction and strength of the generated modifications.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - Identifies the specific sampler to use, allowing for different sampling strategies.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Determines the noise schedule to be applied throughout the sampling steps.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`latent_image`**
    - The input latent image to be processed or modified through sampling.
    - Comfy dtype: `LATENT`
    - Python dtype: `torch.Tensor`
- **`denoise`**
    - Controls the amount of noise reduction, affecting the final output's clarity and detail.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`scheduler_func_opt`**
    - Optional parameter for providing a custom noise schedule function, offering more control over the sampling process.
    - Comfy dtype: `SCHEDULER_FUNC`
    - Python dtype: `function`
## Output types
- **`basic_pipe`**
    - Comfy dtype: `BASIC_PIPE`
    - Passes through the input basic_pipe, including all its components, unchanged.
    - Python dtype: `tuple`
- **`latent`**
    - Comfy dtype: `LATENT`
    - The resulting latent image after applying the sampling process, reflecting the specified modifications.
    - Python dtype: `torch.Tensor`
- **`vae`**
    - Comfy dtype: `VAE`
    - The VAE component from the input basic_pipe, included in the output for potential further processing.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [VAEDecode](../../Comfy/Nodes/VAEDecode.md)
    - Reroute
    - [ImpactKSamplerBasicPipe](../../ComfyUI-Impact-Pack/Nodes/ImpactKSamplerBasicPipe.md)
    - [VAEEncode](../../Comfy/Nodes/VAEEncode.md)



## Source code
```python
class KSamplerBasicPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"basic_pipe": ("BASIC_PIPE",),
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                     "scheduler": (core.SCHEDULERS, ),
                     "latent_image": ("LATENT", ),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     },
                "optional":
                    {
                        "scheduler_func_opt": ("SCHEDULER_FUNC", ),
                    }
                }

    TOOLTIPS = {
        "input": {
            "basic_pipe": "basic_pipe input for sampling",
            "seed": "Random seed to use for generating CPU noise for sampling.",
            "steps": "total sampling steps",
            "cfg": "classifier free guidance value",
            "sampler_name": "sampler",
            "scheduler": "noise schedule",
            "latent_image": "input latent image",
            "denoise": "The amount of noise to remove. This amount is the noise added at the start, and the higher it is, the more the input latent will be modified before being returned.",
            "scheduler_func_opt": "[OPTIONAL] Noise schedule generation function. If this is set, the scheduler widget will be ignored.",
        },
        "output": ("passthrough input basic_pipe", "result latent", "VAE in basic_pipe")
    }

    RETURN_TYPES = ("BASIC_PIPE", "LATENT", "VAE")
    FUNCTION = "sample"

    CATEGORY = "ImpactPack/sampling"

    @staticmethod
    def sample(basic_pipe, seed, steps, cfg, sampler_name, scheduler, latent_image, denoise=1.0, scheduler_func_opt=None):
        model, clip, vae, positive, negative = basic_pipe
        latent = impact_sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, scheduler_func=scheduler_func_opt)
        return basic_pipe, latent, vae

```
