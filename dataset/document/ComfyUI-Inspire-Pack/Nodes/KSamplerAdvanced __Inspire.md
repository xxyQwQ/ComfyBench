---
tags:
- Sampling
---

# KSamplerAdvanced (inspire)
## Documentation
- Class name: `KSamplerAdvanced __Inspire`
- Category: `InspirePack/a1111_compat`
- Output node: `False`

This node is an advanced version of the KSampler tailored for the Inspire pack, designed to enhance sampling strategies with more sophisticated algorithms and options. It aims to provide users with enhanced control and flexibility over the sampling process, enabling the creation of more refined and complex generative art.
## Input types
### Required
- **`model`**
    - Specifies the model used for the sampling process, serving as the core component that drives the generation of outputs.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
- **`add_noise`**
    - Determines whether noise should be added to the sampling process, influencing the diversity and uniqueness of the generated results.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`noise_seed`**
    - Sets the seed for noise generation, ensuring reproducibility and consistency in the sampling outcomes.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - Defines the number of steps to be taken in the sampling process, affecting the detail and quality of the generated outputs.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Controls the configuration for the sampling, adjusting parameters such as creativity and variance in the generated results.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - Selects the specific sampler algorithm to be used, allowing for customization of the sampling strategy.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Chooses the scheduler for controlling the sampling process, further customizing the generation approach.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`positive`**
    - Specifies the positive conditioning to guide the sampling towards desired themes or concepts.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`negative`**
    - Indicates the negative conditioning to avoid certain themes or elements in the generated results.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`latent_image`**
    - Provides the initial latent image to be used in the sampling process, serving as a starting point for generation.
    - Comfy dtype: `LATENT`
    - Python dtype: `str`
- **`start_at_step`**
    - Determines the starting step of the sampling process, allowing for mid-process interventions or adjustments.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`end_at_step`**
    - Specifies the ending step of the sampling process, defining the scope of the generation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`noise_mode`**
    - Selects the mode of noise application, influencing the texture and quality of the generated images.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`return_with_leftover_noise`**
    - Decides whether to include leftover noise in the final output, affecting the final visual appearance.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`batch_seed_mode`**
    - Determines the mode for seed progression throughout the batch, affecting the variation and consistency of generated samples.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`variation_seed`**
    - Sets the seed for variation, enabling controlled diversity in the sampling outcomes.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`variation_strength`**
    - Controls the strength of variation applied to the sampling, adjusting the degree of diversity in the results.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`variation_method`**
    - Specifies the method of applying variation, influencing how diversity is introduced into the sampling process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`noise_opt`**
    - Specifies the optional noise parameters to be used in the sampling process, allowing for further customization of noise characteristics.
    - Comfy dtype: `NOISE`
    - Python dtype: `str`
- **`scheduler_func_opt`**
    - Optionally provides a custom scheduling function, offering advanced control over the sampling dynamics.
    - Comfy dtype: `SCHEDULER_FUNC`
    - Python dtype: `str`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - Outputs the final latent representation after the sampling process, reflecting the culmination of the applied conditions and configurations.
    - Python dtype: `str`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class KSamplerAdvanced_inspire:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                     "add_noise": ("BOOLEAN", {"default": True, "label_on": "enable", "label_off": "disable"}),
                     "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.5, "round": 0.01}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                     "scheduler": (common.SCHEDULERS, ),
                     "positive": ("CONDITIONING", ),
                     "negative": ("CONDITIONING", ),
                     "latent_image": ("LATENT", ),
                     "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                     "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                     "noise_mode": (["GPU(=A1111)", "CPU"],),
                     "return_with_leftover_noise": ("BOOLEAN", {"default": False, "label_on": "enable", "label_off": "disable"}),
                     "batch_seed_mode": (["incremental", "comfy", "variation str inc:0.01", "variation str inc:0.05"],),
                     "variation_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "variation_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     },
                "optional":
                    {
                        "variation_method": (["linear", "slerp"],),
                        "noise_opt": ("NOISE",),
                        "scheduler_func_opt": ("SCHEDULER_FUNC",),
                    }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/a1111_compat"

    @staticmethod
    def sample(model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, noise_mode, return_with_leftover_noise,
               denoise=1.0, batch_seed_mode="comfy", variation_seed=None, variation_strength=None, noise_opt=None, callback=None, variation_method="linear", scheduler_func_opt=None):
        force_full_denoise = True

        if return_with_leftover_noise:
            force_full_denoise = False

        disable_noise = False

        if not add_noise:
            disable_noise = True

        return inspire_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step,
                                force_full_denoise=force_full_denoise, noise_mode=noise_mode, incremental_seed_mode=batch_seed_mode,
                                variation_seed=variation_seed, variation_strength=variation_strength, noise=noise_opt, callback=callback, variation_method=variation_method,
                                scheduler_func=scheduler_func_opt)

    def doit(self, *args, **kwargs):
        return (self.sample(*args, **kwargs)[0],)

```
