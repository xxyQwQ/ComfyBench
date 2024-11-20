---
tags:
- Sampling
---

# KSampler [pipe] (inspire)
## Documentation
- Class name: `KSamplerPipe __Inspire`
- Category: `InspirePack/a1111_compat`
- Output node: `False`

The KSamplerPipe //Inspire node is designed to facilitate the sampling process within the Inspire Pack framework, leveraging advanced sampling techniques to generate or manipulate latent representations of data. This node acts as a conduit for integrating various sampling strategies, enhancing the flexibility and efficiency of generating new, inspired content or variations thereof.
## Input types
### Required
- **`basic_pipe`**
    - Represents the core components required for the sampling process, including models and configurations, crucial for generating or manipulating latent representations.
    - Comfy dtype: `BASIC_PIPE`
    - Python dtype: `tuple`
- **`seed`**
    - A seed value to ensure reproducibility of the sampling process, allowing for consistent generation of content across different runs.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - Defines the number of steps to be taken in the sampling process, affecting the detail and quality of the generated content.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Configuration settings for the sampling process, dictating how the sampler operates and influences the generation or manipulation of content.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - Specifies the particular sampling strategy to be used, enabling customization of the sampling process based on specific requirements or preferences.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - A scheduler to control the sampling process, providing a mechanism to adjust sampling parameters dynamically over time.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `object`
- **`latent_image`**
    - The initial latent representation to be used as a starting point for the sampling process, which can be manipulated or enhanced through sampling.
    - Comfy dtype: `LATENT`
    - Python dtype: `object`
- **`denoise`**
    - A parameter to control the level of denoising applied during the sampling process, affecting the clarity and quality of the generated content.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`noise_mode`**
    - Determines how noise is applied during the sampling process, influencing the variation and uniqueness of the generated content.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`batch_seed_mode`**
    - Controls how seeds are managed across batches in the sampling process, affecting the diversity and reproducibility of generated content.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`variation_seed`**
    - An optional seed for introducing variations in the sampling process, allowing for the creation of diverse content from a single initial representation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`variation_strength`**
    - Determines the strength of the variations introduced by the variation_seed, affecting the degree of difference from the original content.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`scheduler_func_opt`**
    - An optional scheduler function to further customize the sampling process, providing additional control over how sampling parameters are adjusted.
    - Comfy dtype: `SCHEDULER_FUNC`
    - Python dtype: `object`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The resulting latent representation after the sampling process, which can be used for further generation or manipulation of content.
    - Python dtype: `object`
- **`vae`**
    - Comfy dtype: `VAE`
    - The variational autoencoder used in the sampling process, which can be utilized for additional content generation or manipulation tasks.
    - Python dtype: `object`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class KSampler_inspire_pipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"basic_pipe": ("BASIC_PIPE",),
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                     "scheduler": (common.SCHEDULERS, ),
                     "latent_image": ("LATENT", ),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "noise_mode": (["GPU(=A1111)", "CPU"],),
                     "batch_seed_mode": (["incremental", "comfy", "variation str inc:0.01", "variation str inc:0.05"],),
                     "variation_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "variation_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     },
                "optional":
                    {
                        "scheduler_func_opt": ("SCHEDULER_FUNC",),
                    }
                }

    RETURN_TYPES = ("LATENT", "VAE")
    FUNCTION = "sample"

    CATEGORY = "InspirePack/a1111_compat"

    def sample(self, basic_pipe, seed, steps, cfg, sampler_name, scheduler, latent_image, denoise, noise_mode, batch_seed_mode="comfy",
               variation_seed=None, variation_strength=None, scheduler_func_opt=None):
        model, clip, vae, positive, negative = basic_pipe
        latent = inspire_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, noise_mode, incremental_seed_mode=batch_seed_mode,
                                  variation_seed=variation_seed, variation_strength=variation_strength, scheduler_func=scheduler_func_opt)[0]
        return latent, vae

```
