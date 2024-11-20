---
tags:
- Sampling
---

# KSamplerAdvanced [pipe] (inspire)
## Documentation
- Class name: `KSamplerAdvancedPipe __Inspire`
- Category: `InspirePack/a1111_compat`
- Output node: `False`

This node is designed for advanced sampling in generative models, focusing on creating or modifying latent images through a comprehensive set of parameters. It integrates various components such as noise addition, seed manipulation, and conditioning adjustments to refine the generation process, aiming to produce high-quality, customizable results.
## Input types
### Required
- **`basic_pipe`**
    - A tuple containing the model, clip, VAE, and positive and negative conditioning components, serving as the foundational setup for the sampling process.
    - Comfy dtype: `BASIC_PIPE`
    - Python dtype: `tuple`
- **`add_noise`**
    - A boolean flag indicating whether noise should be added to the generation process, affecting the diversity and quality of the output.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`noise_seed`**
    - An integer seed specifically for noise generation, contributing to the variability and uniqueness of the generated images.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - The number of steps to perform in the sampling process, affecting the detail and quality of the generated images.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - A configuration parameter that influences the sampling behavior, potentially affecting the style and characteristics of the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - The name of the sampler to use, determining the specific sampling algorithm applied.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Specifies the scheduler for controlling the sampling process, affecting the progression of image generation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`latent_image`**
    - The initial latent image to be modified or enhanced through the sampling process.
    - Comfy dtype: `LATENT`
    - Python dtype: `torch.Tensor`
- **`start_at_step`**
    - The step at which to start the sampling process, allowing for control over the generation's initiation point.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`end_at_step`**
    - The final step of the sampling process, defining the endpoint of image generation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`noise_mode`**
    - Determines the mode of noise application, influencing the texture and quality of the output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`return_with_leftover_noise`**
    - A flag indicating whether to return the image with any residual noise, affecting the final image's appearance.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`batch_seed_mode`**
    - Specifies the mode for seed generation in batch processing, impacting the diversity of generated images.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`variation_seed`**
    - An optional seed for introducing variations, allowing for controlled randomness in the output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`variation_strength`**
    - Determines the strength of the applied variations, affecting the degree of change from the original image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`noise_opt`**
    - Optional noise parameters for further customization of the noise application process.
    - Comfy dtype: `NOISE`
    - Python dtype: `dict`
- **`scheduler_func_opt`**
    - An optional scheduler function for advanced control over the sampling schedule.
    - Comfy dtype: `SCHEDULER_FUNC`
    - Python dtype: `function`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - Represents the generated or modified latent image, showcasing the effectiveness of the sampling process.
    - Python dtype: `torch.Tensor`
- **`vae`**
    - Comfy dtype: `VAE`
    - The VAE component used in the sampling process, potentially modified or utilized as part of the generation.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class KSamplerAdvanced_inspire_pipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"basic_pipe": ("BASIC_PIPE",),
                     "add_noise": ("BOOLEAN", {"default": True, "label_on": "enable", "label_off": "disable"}),
                     "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.5, "round": 0.01}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                     "scheduler": (common.SCHEDULERS, ),
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
                        "noise_opt": ("NOISE",),
                        "scheduler_func_opt": ("SCHEDULER_FUNC",),
                    }
                }

    RETURN_TYPES = ("LATENT", "VAE", )
    FUNCTION = "sample"

    CATEGORY = "InspirePack/a1111_compat"

    def sample(self, basic_pipe, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, latent_image, start_at_step, end_at_step, noise_mode, return_with_leftover_noise,
               denoise=1.0, batch_seed_mode="comfy", variation_seed=None, variation_strength=None, noise_opt=None, scheduler_func_opt=None):
        model, clip, vae, positive, negative = basic_pipe
        latent = KSamplerAdvanced_inspire().sample(model=model, add_noise=add_noise, noise_seed=noise_seed,
                                                   steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
                                                   positive=positive, negative=negative, latent_image=latent_image,
                                                   start_at_step=start_at_step, end_at_step=end_at_step,
                                                   noise_mode=noise_mode, return_with_leftover_noise=return_with_leftover_noise,
                                                   denoise=denoise, batch_seed_mode=batch_seed_mode, variation_seed=variation_seed,
                                                   variation_strength=variation_strength, noise_opt=noise_opt, scheduler_func_opt=scheduler_func_opt)[0]
        return latent, vae

```
