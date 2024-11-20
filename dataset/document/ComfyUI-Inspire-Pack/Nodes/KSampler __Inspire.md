---
tags:
- Sampling
---

# KSampler (inspire)
## Documentation
- Class name: `KSampler __Inspire`
- Category: `InspirePack/a1111_compat`
- Output node: `False`

The KSampler __Inspire node is designed to perform sampling operations within the Inspire framework, leveraging advanced sampling techniques to generate or modify data based on specified conditions. It abstracts complex sampling algorithms, providing a user-friendly interface for diverse sampling tasks.
## Input types
### Required
- **`model`**
    - Specifies the model to be used for sampling, central to determining the output's characteristics.
    - Comfy dtype: `MODEL`
    - Python dtype: `object`
- **`seed`**
    - Determines the randomness seed, ensuring reproducibility of results.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - Defines the number of steps to perform in the sampling process, affecting the detail and quality of the output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Controls the conditioning factor, influencing the direction and strength of the sampling process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - Selects the specific sampler algorithm to use, impacting the sampling behavior and results.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Chooses the scheduler for controlling the sampling process, further refining the output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`positive`**
    - Sets positive conditioning to guide the sampling towards desired attributes.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`negative`**
    - Applies negative conditioning to steer the sampling away from certain attributes.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`latent_image`**
    - Provides an initial latent image to be modified or enhanced through sampling.
    - Comfy dtype: `LATENT`
    - Python dtype: `object`
- **`denoise`**
    - Adjusts the level of denoising applied during sampling, affecting the clarity and sharpness of the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`noise_mode`**
    - Specifies the mode of noise application, influencing the texture and details of the sampled output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`batch_seed_mode`**
    - Determines the mode for generating seeds in batch operations, affecting the diversity of the output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`variation_seed`**
    - Sets the seed for introducing variations, allowing for controlled randomness in the output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`variation_strength`**
    - Controls the strength of variations applied, impacting the degree of change from the original.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`variation_method`**
    - Defines the method used for applying variations, influencing the nature of the modifications.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler_func_opt`**
    - Optionally selects a specific scheduling function, offering further customization of the sampling process.
    - Comfy dtype: `SCHEDULER_FUNC`
    - Python dtype: `str`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The result of the sampling process, typically a modified or newly generated latent image.
    - Python dtype: `object`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class KSampler_inspire:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                     "scheduler": (common.SCHEDULERS, ),
                     "positive": ("CONDITIONING", ),
                     "negative": ("CONDITIONING", ),
                     "latent_image": ("LATENT", ),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "noise_mode": (["GPU(=A1111)", "CPU"],),
                     "batch_seed_mode": (["incremental", "comfy", "variation str inc:0.01", "variation str inc:0.05"],),
                     "variation_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "variation_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     },
                "optional":
                    {
                        "variation_method": (["linear", "slerp"],),
                        "scheduler_func_opt": ("SCHEDULER_FUNC",),
                    }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/a1111_compat"

    @staticmethod
    def doit(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, noise_mode,
             batch_seed_mode="comfy", variation_seed=None, variation_strength=None, variation_method="linear", scheduler_func_opt=None):
        return (inspire_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, noise_mode,
                                 incremental_seed_mode=batch_seed_mode, variation_seed=variation_seed, variation_strength=variation_strength, variation_method=variation_method,
                                 scheduler_func=scheduler_func_opt)[0], )

```
